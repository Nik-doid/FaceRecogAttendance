"""The always-on camera loop.

Previously the RTSP reader, the pipeline and the frame loop all lived inside the
``/camera/ws`` handler, which meant the camera *was* the WebSocket connection: closing
the browser tab closed the camera and stopped attendance. Everything here exists to
break that coupling. The runner owns the reader and the pipeline for the life of the
process; ``websocket.py`` is reduced to a viewer that reads whatever the runner last
published into the :class:`FrameHub`.

It runs on a daemon thread with its own event loop rather than as a task on the
FastAPI loop. ``CameraReader.read`` is a blocking OpenCV call, and on a two-core box an
unhealthy RTSP socket blocking the shared loop would stall ``/health`` and every
viewer alongside it. Every heavy call inside ``process_frames`` is already dispatched
through ``asyncio.to_thread``, so the pipeline works unchanged on a private loop.

Detection is time-driven, not frame-counted. Counting frames at 30fps implies "scan
every 330ms" while a frame that reaches recognition costs over a second, so the honest
rule is "start a scan at most every ``CAMERA_SCAN_INTERVAL_MS``, and never while one is
already running".
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from app.camera.hub import FrameHub
from app.config.settings import Settings
from app.core.face_processing.gallery import GalleryHandle
from app.core.logging import get_logger
from app.runtime import Models
from app.schemas.face_processing import (
    AttendanceSinkType,
    FaceProcessConfig,
    FrameContext,
    FrameResult,
)
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor
from app.services.face_recognition.process import FaceRecognitionProcess
from app.workers.camera.reader import CameraReader

log = get_logger(__name__)

JPEG_QUALITY = 70
FRAME_INTERVAL = 0.033  # ~30 fps
MIN_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0


@dataclass
class RunnerState:
    """What the runner is doing, for /camera/status and for tests."""

    running: bool = False
    connected: bool = False
    frames: int = 0
    scans: int = 0
    reconnects: int = 0
    last_error: str | None = None
    started_at: datetime | None = None
    last_frame_at: datetime | None = None
    last_detection_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "connected": self.connected,
            "frames": self.frames,
            "scans": self.scans,
            "reconnects": self.reconnects,
            "last_error": self.last_error,
            "started_at": _iso(self.started_at),
            "last_frame_at": _iso(self.last_frame_at),
            "last_detection_at": _iso(self.last_detection_at),
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class CameraRunnerAlreadyRunningError(RuntimeError):
    """Raised when start() is called on a runner that is already going."""


@dataclass
class _Deps:
    settings: Settings
    models: Models
    gallery: GalleryHandle
    hub: FrameHub
    reporter: AttendanceReporter | None = None
    suppressor: DuplicateSuppressor | None = None
    reader_factory: Any = None
    process_factory: Any = None
    state: RunnerState = field(default_factory=RunnerState)


class CameraRunner:
    """Owns the camera and the pipeline; publishes into a :class:`FrameHub`."""

    def __init__(
        self,
        settings: Settings,
        models: Models,
        gallery: GalleryHandle,
        hub: FrameHub,
        *,
        reporter: AttendanceReporter | None = None,
        suppressor: DuplicateSuppressor | None = None,
        reader_factory: Any = None,
        process_factory: Any = None,
    ) -> None:
        self._deps = _Deps(
            settings=settings,
            models=models,
            gallery=gallery,
            hub=hub,
            reporter=reporter,
            suppressor=suppressor,
            reader_factory=reader_factory or (lambda: _default_reader(settings)),
            process_factory=process_factory,
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- control surface ------------------------------------------------------
    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> RunnerState:
        return self._deps.state

    def start(self) -> RunnerState:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise CameraRunnerAlreadyRunningError("camera runner is already running")
            self._stop.clear()
            self._deps.state = RunnerState(running=True, started_at=datetime.now(UTC))
            self._thread = threading.Thread(
                target=self._run, name="camera-runner", daemon=True
            )
            self._thread.start()
            return self._deps.state

    def stop(self, timeout: float = 10.0) -> RunnerState:
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Daemon thread, so it dies with the process; say so rather than
                # reporting a clean stop that did not happen.
                log.warning("camera runner did not stop within the timeout")
        self._deps.state.running = False
        self._deps.state.connected = False
        self._deps.hub.clear()
        return self._deps.state

    # -- the loop -------------------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._loop())
        except Exception as exc:  # noqa: BLE001 - the thread must never die silently
            log.exception("camera runner crashed")
            self._deps.state.last_error = str(exc)
        finally:
            self._deps.state.running = False
            self._deps.state.connected = False
            asyncio.set_event_loop(None)
            loop.close()
            log.info("camera runner stopped")

    async def _loop(self) -> None:
        state = self._deps.state
        process = self._build_process()
        reader = None
        backoff = MIN_BACKOFF_SECONDS
        detecting: asyncio.Task[FrameResult] | None = None
        detected_at: datetime | None = None
        detected_shape = (0, 0)
        last_scan = 0.0
        scan_interval = max(0.05, self._deps.settings.camera_scan_interval_ms / 1000.0)

        log.info("camera runner started", extra={"camera_id": self._deps.settings.camera_id})
        try:
            while not self._stop.is_set():
                if reader is None:
                    reader = self._deps.reader_factory()
                    if not reader.open():
                        state.connected = False
                        state.last_error = "failed to open camera stream"
                        reader = None
                        # Unattended, so a camera that is not up yet must be waited
                        # for, not treated as fatal the way a browser session could.
                        await asyncio.sleep(backoff)
                        backoff = min(MAX_BACKOFF_SECONDS, backoff * 2)
                        continue
                    state.connected = True
                    state.last_error = None
                    backoff = MIN_BACKOFF_SECONDS

                try:
                    frame = reader.read()
                except Exception as exc:  # noqa: BLE001 - a camera may raise, not just
                    # return None. OpenCV can throw on a half-open RTSP socket, and
                    # letting that escape kills the runner thread for good -- exactly
                    # the unattended failure this loop exists to survive.
                    log.warning("camera read raised", extra={"error": str(exc)})
                    state.last_error = str(exc)
                    frame = None
                if frame is None:
                    state.connected = False
                    state.reconnects += 1
                    log.warning("camera read failed; reconnecting")
                    _close(reader)
                    reader = None
                    continue

                state.frames += 1
                state.last_frame_at = datetime.now(UTC)
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    self._deps.hub.publish_frame(jpeg.tobytes())

                if detecting is not None and detecting.done():
                    payload = self._harvest(detecting, detected_shape)
                    detecting = None
                    if payload is not None:
                        self._deps.hub.publish_detection(payload)
                        state.last_detection_at = detected_at

                now = time.monotonic()
                if detecting is None and now - last_scan >= scan_interval:
                    last_scan = now
                    detected_shape = frame.shape[:2]
                    detected_at = datetime.now(UTC)
                    state.scans += 1
                    detecting = asyncio.create_task(
                        process.process_frames(
                            frame,
                            FrameContext(
                                camera_id=self._deps.settings.camera_id,
                                captured_at=detected_at,
                            ),
                        )
                    )

                await asyncio.sleep(FRAME_INTERVAL)
        finally:
            if detecting is not None:
                detecting.cancel()
            if reader is not None:
                _close(reader)

    def _build_process(self) -> FaceRecognitionProcess:
        if self._deps.process_factory is not None:
            result: FaceRecognitionProcess = self._deps.process_factory()
            return result
        settings = self._deps.settings
        # This is the one pipeline allowed to record attendance: it is the only one
        # that knows which camera it is and when the frame was captured.
        sink = (
            AttendanceSinkType.RABBITMQ
            if self._deps.reporter is not None
            else AttendanceSinkType.NULL
        )
        return FaceRecognitionProcess(
            FaceProcessConfig(
                attendance_sink=sink,
                palm_score_threshold=settings.palm_score_threshold,
                # Only reached for a face with no landmarks to anchor the palm search
                # to, which the looking gate makes rare.
                palm_scan_grid=settings.palm_scan_grid,
                palm_scan_overlap=settings.palm_scan_overlap,
                looking_max_yaw_ratio=settings.looking_max_yaw_ratio,
                looking_max_roll_degrees=settings.looking_max_roll_degrees,
                palm_search_margin=settings.palm_search_margin,
                recognition_threshold=settings.recognition_threshold,
            ),
            self._deps.models,
            self._deps.gallery.current,
            settings.models_dir,
            reporter=self._deps.reporter,
            suppressor=self._deps.suppressor,
        )

    def _harvest(
        self, task: asyncio.Task[FrameResult], shape: tuple[int, int]
    ) -> dict[str, Any] | None:
        """One bad frame must not take the camera down with it."""
        try:
            result = task.result()
        except asyncio.CancelledError:
            return None
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            log.exception("frame detection failed")
            self._deps.state.last_error = str(exc)
            return None
        return detection_payload(result, shape)


def _default_reader(settings: Settings) -> CameraReader:
    return CameraReader(
        rtsp_url=settings.rtsp_url,
        source=settings.camera_source,
        device_index=settings.camera_device_index,
        max_width=settings.max_frame_width,
    )


def _close(reader: Any) -> None:
    try:
        reader.close()
    except Exception:  # noqa: BLE001 - closing must never mask the real failure
        log.debug("closing the camera reader raised", exc_info=True)


def detection_payload(result: FrameResult, shape: tuple[int, int]) -> dict[str, Any]:
    """The message shape the page renders. Shared with the browser-webcam route."""
    height, width = shape
    return {
        "palm": result.palm.detected,
        "score": round(result.palm.score, 3),
        # Boxes are in frame pixels; the page scales by these dimensions.
        "width": width,
        "height": height,
        # kps are deliberately omitted: the page draws boxes, not landmarks.
        "faces": [
            {
                "bbox": [round(v, 1) for v in face.bbox],
                "score": round(face.score, 3),
                "looking": face.looking,
                "yaw": face.yaw_ratio,
                "palm": face.palm,
                "employee_code": face.employee_code,
                "confidence": round(face.confidence, 3),
            }
            for face in result.faces
        ],
    }


def encode_jpeg(frame: np.ndarray) -> bytes | None:
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg.tobytes() if ok else None
