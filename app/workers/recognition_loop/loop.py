"""Continuous recognition worker.

A single daemon thread owns the camera reader and drives the pipeline. It:
- reconnects to RTSP with exponential backoff,
- applies the frame-skip policy,
- decides report / suppress / unknown per recognized face,
- publishes attendance events and persists recognition/unknown/audit rows.

This is deliberately NOT inside FastAPI request handlers; the API only starts/stops
this thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
from sqlalchemy.orm import Session

from app.ai.components import AIComponents
from app.ai.faiss.index import FaceIndex
from app.ai.tracker.base import Tracker
from app.config.settings import Settings
from app.core.logging import get_logger
from app.core.metrics import (
    CAMERA_CONNECTED,
    CAMERA_RECONNECTS,
    FRAMES_PROCESSED,
    FRAMES_SKIPPED,
    LIVENESS_FAILED,
    PROCESSING_TIME,
    RECOGNITIONS,
    REPORTS_FAILED,
    REPORTS_PUBLISHED,
    UNKNOWN_FACES,
)
from app.database.session import sync_session
from app.repositories.camera_repo import CameraRepository
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.repositories.unknown_face_repo import UnknownFaceRepository
from app.services.attendance_reporter.base import AttendanceEvent, AttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor
from app.services.settings_service import SettingsService
from app.storage.snapshot import SnapshotStorage
from app.workers.camera.reader import CameraReader
from app.workers.recognition_loop.pipeline import FaceEvent, RecognitionPipeline

MIN_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
TUNING_REFRESH_SECONDS = 30


class RecognitionLoop:
    def __init__(
        self,
        *,
        reader: CameraReader,
        pipeline: RecognitionPipeline,
        ai: AIComponents,
        face_index: FaceIndex,
        duplicate_suppressor: DuplicateSuppressor,
        attendance_reporter: AttendanceReporter,
        settings_service: SettingsService,
        env_settings: Settings,
        recognition_log_repo: RecognitionLogRepository,
        unknown_face_repo: UnknownFaceRepository,
        camera_repo: CameraRepository,
        snapshot_storage: SnapshotStorage,
        session_factory: Callable[[], Session] | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        self._reader = reader
        self._pipeline = pipeline
        self._ai = ai
        self._index = face_index
        self._suppressor = duplicate_suppressor
        self._reporter = attendance_reporter
        self._settings_service = settings_service
        self._env = env_settings
        self._log_repo = recognition_log_repo
        self._unknown_repo = unknown_face_repo
        self._camera_repo = camera_repo
        self._snapshots = snapshot_storage
        self._session_factory = session_factory or sync_session
        self._tracker = tracker
        self._log = get_logger(__name__)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_index = 0
        self._tuning_last_refresh = 0.0
        self._frame_skip = self._env.frame_skip

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="recognition-loop", daemon=True)
        self._thread.start()
        self._log.info("recognition loop started", extra={"camera_id": self._env.camera_id})

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._reader.close()
        self._log.info("recognition loop stopped", extra={"camera_id": self._env.camera_id})

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- main loop --------------------------------------------------------------
    def _run(self) -> None:
        backoff = MIN_BACKOFF_SECONDS
        while not self._stop.is_set():
            ok = self._reader.open()
            if not ok:
                CAMERA_RECONNECTS.inc()
                CAMERA_CONNECTED.set(0)
                self._set_camera_status("error", error="failed to open RTSP stream")
                self._log.warning(
                    "camera connect failed; retrying",
                    extra={"backoff_seconds": backoff},
                )
                self._sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                continue
            backoff = MIN_BACKOFF_SECONDS
            CAMERA_CONNECTED.set(1)
            self._set_camera_status("running", connected=True)
            self._log.info("camera connected", extra={"camera_id": self._env.camera_id})
            self._consume_frames()
        self._set_camera_status("stopped")

    def _consume_frames(self) -> None:
        self._frame_index = 0
        while not self._stop.is_set():
            self._refresh_tuning()
            frame = self._reader.read()
            if frame is None:
                CAMERA_CONNECTED.set(0)
                CAMERA_RECONNECTS.inc()
                self._set_camera_status("error", error="RTSP stream ended")
                self._log.warning("camera stream lost; reopening")
                self._reader.close()
                return  # outer loop re-opens with backoff

            if self._frame_index % self._frame_skip != 0:
                FRAMES_SKIPPED.inc()
                self._frame_index += 1
                continue

            self._frame_index += 1
            FRAMES_PROCESSED.inc()
            try:
                with PROCESSING_TIME.time():
                    if self._tracker is None:
                        events = self._pipeline.process_frame(frame)
                    else:
                        detections = self._pipeline.detect(frame)
                        tracked = self._tracker.update(detections)
                        tracking = [(t.track_id, t.face) for t in tracked]
                        events = self._pipeline.process_frame(frame, tracking=tracking)
                    self._handle_events(frame, events)
            except Exception:  # noqa: BLE001 - pipeline must never kill the loop
                self._log.exception("frame processing failed")

    # -- per-frame event handling ------------------------------------------------
    def _handle_events(self, frame: np.ndarray, events: list[FaceEvent]) -> None:
        for ev in events:
            if not ev.quality_passed:
                continue
            if not ev.live:
                LIVENESS_FAILED.inc()
                if self._snapshots.enabled:
                    self._snapshots.save(frame, prefix="spoof", track_id=ev.track_id)
                continue
            if ev.employee_code is None:
                self._handle_unknown(frame, ev)
                continue
            self._handle_match(frame, ev)

    def _handle_match(self, frame: np.ndarray, ev: FaceEvent) -> None:
        RECOGNITIONS.labels(ev.employee_code or "unknown").inc()
        if not self._suppressor.check_and_record(ev.employee_code or ""):
            self._write_log(ev, reported=False, response="duplicate_suppressed")
            return
        snapshot_path = self._snapshots.save(
            frame,
            prefix="attendance",
            employee_code=ev.employee_code,
            track_id=ev.track_id,
        )
        event = AttendanceEvent(
            employee_code=ev.employee_code or "",
            camera_id=self._env.camera_id,
            timestamp=datetime.now(UTC),
            confidence=ev.confidence,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            track_id=ev.track_id,
        )
        result = self._reporter.report(event)
        if result.success:
            REPORTS_PUBLISHED.inc()
        else:
            REPORTS_FAILED.inc()
            self._log.error(
                "attendance report failed",
                extra={"employee_code": ev.employee_code, "detail": result.detail},
            )
        self._write_log(
            ev,
            reported=result.success,
            response=result.detail,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
        )

    def _handle_unknown(self, frame: np.ndarray, ev: FaceEvent) -> None:
        UNKNOWN_FACES.inc()
        path = self._snapshots.save(frame, prefix="unknown", track_id=ev.track_id)
        with self._session_factory() as session:
            self._unknown_repo.add_entry(
                session,
                camera_id=self._env.camera_id,
                timestamp=datetime.now(UTC),
                snapshot_path=str(path) if path else "not_saved",
                confidence_of_best_nonmatch=ev.best_score,
                track_id=ev.track_id,
            )
        self._log.info(
            "unknown face",
            extra={"event": "unknown_face", "best_score": ev.best_score, "track_id": ev.track_id},
        )

    def _write_log(
        self,
        ev: FaceEvent,
        *,
        reported: bool,
        response: str,
        snapshot_path: str | None = None,
    ) -> None:
        with self._session_factory() as session:
            self._log_repo.add_entry(
                session,
                employee_code=ev.employee_code,
                camera_id=self._env.camera_id,
                timestamp=datetime.now(UTC),
                confidence=ev.confidence,
                reported=reported,
                attendance_response=response,
                snapshot_path=snapshot_path,
                track_id=ev.track_id,
            )

    # -- helpers -----------------------------------------------------------------
    def _refresh_tuning(self) -> None:
        now = time.monotonic()
        if now - self._tuning_last_refresh < TUNING_REFRESH_SECONDS:
            return
        self._tuning_last_refresh = now
        try:
            with self._session_factory() as session:
                rt = self._settings_service.get_runtime(session, self._env)
            self._frame_skip = max(0, rt.frame_skip)
            self._pipeline.recognition_threshold = rt.recognition_threshold
            self._suppressor.timeout_seconds = rt.duplicate_timeout_seconds
        except Exception:  # noqa: BLE001 - tuning refresh must not kill the loop
            self._log.exception("failed to refresh runtime tuning")

    def _set_camera_status(
        self,
        status: str,
        *,
        connected: bool | None = None,
        error: str | None = None,
    ) -> None:
        try:
            with self._session_factory() as session:
                self._camera_repo.set_status(
                    session,
                    self._env.camera_id,
                    status,
                    last_connected_at=datetime.now(UTC) if connected else None,
                    last_error=error,
                )
        except Exception:  # noqa: BLE001 - status write is best-effort
            self._log.exception("failed to persist camera status")

    def _sleep(self, seconds: float) -> None:
        """Sleep in small slices so stop() is honoured promptly."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop.is_set():
            time.sleep(min(0.25, end - time.monotonic()))
