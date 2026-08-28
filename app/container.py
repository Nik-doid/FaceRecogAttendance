"""Dependency-injection container.

Constructs every long-lived dependency once and hands them to the API (FastAPI
dependency overrides) and the background worker. Tests can construct a ``Container``
with fakes injected via constructor kwargs, or override individual services.
"""

from __future__ import annotations

import threading

from app.ai.components import AIComponents, load_ai_components
from app.ai.faiss.index import FaceIndex
from app.ai.tracker.base import Tracker
from app.ai.tracker.bytetrack import ByteTrackTracker
from app.camera.hub import FrameHub
from app.camera.runner import CameraRunner
from app.config.settings import Settings
from app.core.face_processing.gallery import (
    Gallery,
    GalleryHandle,
    build_cache,
    build_gallery,
)
from app.core.face_processing.photos import build_sources
from app.core.logging import get_logger
from app.database.session import sync_session
from app.repositories.audit_repo import AuditLogRepository
from app.repositories.camera_repo import CameraRepository
from app.repositories.face_embedding_repo import FaceEmbeddingRepository
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.repositories.setting_repo import SettingRepository
from app.repositories.unknown_face_repo import UnknownFaceRepository
from app.runtime import Models, load_models
from app.services.attendance_consumer import ConsumerRunner, build_attendance_consumer
from app.services.attendance_reporter import build_attendance_reporter
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.camera_service import CameraService
from app.services.duplicate_suppressor import DuplicateSuppressor
from app.services.erp_sync import ErpSyncScheduler, build_erp_sync
from app.services.index_service import IndexService
from app.services.settings_service import SettingsService
from app.storage.snapshot import SnapshotStorage
from app.workers.camera.reader import CameraReader
from app.workers.frame_buffer import FrameBuffer
from app.workers.recognition_loop.loop import RecognitionLoop
from app.workers.recognition_loop.pipeline import RecognitionPipeline

EMBEDDING_DIM = 512


class Container:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        load_models: bool = True,
        ai: AIComponents | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._log = get_logger(__name__)
        # Reentrant: the ``gallery`` property builds under the lock and reaches
        # through ``models``, which takes it again.
        self._lock = threading.RLock()
        self._models: Models | None = None
        self.gallery_handle = GalleryHandle()
        self.frame_hub = FrameHub()
        self._camera_runner: CameraRunner | None = None
        self.attendance_consumer: ConsumerRunner | None = None

        # Shared singletons.
        self.face_index = FaceIndex(dim=EMBEDDING_DIM)
        self.settings_service = SettingsService(SettingRepository())
        self.duplicate_suppressor = DuplicateSuppressor(self.settings.duplicate_timeout_seconds)
        self.snapshot_storage = SnapshotStorage(
            self.settings.snapshots_dir, enabled=self.settings.snapshot_enabled
        )
        self.frame_buffer = FrameBuffer()
        self.attendance_reporter: AttendanceReporter = build_attendance_reporter(self.settings)

        # Repositories.
        self.face_embedding_repo = FaceEmbeddingRepository()
        self.recognition_log_repo = RecognitionLogRepository()
        self.unknown_face_repo = UnknownFaceRepository()
        self.camera_repo = CameraRepository()
        self.audit_repo = AuditLogRepository()

        # ERP attendance-log sync (optional).
        self.erp_sync_service = build_erp_sync(self.settings, self.snapshot_storage)
        if self.settings.erp_sync_enabled:
            self.erp_sync_scheduler: ErpSyncScheduler | None = ErpSyncScheduler(
                lambda: self.erp_sync_service.sync_once(sync_session),
                interval_seconds=self.settings.erp_sync_interval_seconds,
            )
        else:
            self.erp_sync_scheduler = None

        # AI components (loaded lazily if requested).
        self.ai: AIComponents | None = ai
        if self.ai is None and load_models:
            self.ai = load_ai_components(self.settings)

        # Index service (owns the rebuild thread).
        self.index_service = self._build_index_service()
        self.camera_service = CameraService(self._build_loop)

    # -- construction helpers -----------------------------------------------------
    def _build_index_service(self) -> IndexService:
        assert self.ai is not None
        svc = IndexService(
            settings=self.settings,
            face_index=self.face_index,
            detector=self.ai.detector,
            recognizer=self.ai.recognizer,
            quality=self.ai.quality,
            repo=self.face_embedding_repo,
        )
        # Startup log: FAISS index status
        self._log.info(
            "FAISS index loaded: size=%d employees=%s",
            self.face_index.size,
            sorted(self.face_index.employee_codes),
        )
        return svc

    def _build_loop(self) -> RecognitionLoop:
        assert self.ai is not None
        from app.database.session import sync_session

        tracker: Tracker | None = None
        if self.settings.tracking_enabled:
            tracker = ByteTrackTracker()

        pipeline = RecognitionPipeline(
            self.ai,
            self.face_index,
            recognition_threshold=self.settings.recognition_threshold,
        )
        return RecognitionLoop(
            reader=CameraReader(
                self.settings.rtsp_url,
                source=self.settings.camera_source,
                device_index=self.settings.camera_device_index,
                max_width=self.settings.max_frame_width,
            ),
            pipeline=pipeline,
            ai=self.ai,
            face_index=self.face_index,
            duplicate_suppressor=self.duplicate_suppressor,
            attendance_reporter=self.attendance_reporter,
            settings_service=self.settings_service,
            env_settings=self.settings,
            recognition_log_repo=self.recognition_log_repo,
            unknown_face_repo=self.unknown_face_repo,
            camera_repo=self.camera_repo,
            snapshot_storage=self.snapshot_storage,
            session_factory=sync_session,
            tracker=tracker,
            frame_buffer=self.frame_buffer,
        )

    # -- lifecycle -----------------------------------------------------------------
    def start_rebuild(self) -> bool:
        from app.database.session import sync_session

        return self.index_service.start_rebuild(sync_session)

    def start_erp_sync(self) -> bool:
        """Start the ERP sync scheduler thread if enabled. Returns whether it started."""
        if self.erp_sync_scheduler is None:
            return False
        self.erp_sync_scheduler.start()
        return True

    @property
    def models(self) -> Models:
        """The process-wide SCRFD + ArcFace sessions, built on first use.

        Lazy rather than eager so a control-plane-only test can construct a Container
        without paying a 174 MiB model load it will never use.
        """
        with self._lock:
            if self._models is None:
                self._models = load_models(self.settings)
            return self._models

    @property
    def gallery(self) -> Gallery:
        """The current enrolled employees.

        Returns the empty gallery until the background build finishes, which is
        deliberately not an error: recognition returns faces unmatched and the rest of
        the pipeline (detect, gaze, palm) runs normally in the meantime.
        """
        return self.gallery_handle.current

    def build_gallery_now(self) -> Gallery:
        """Enumerate the photo sources, embed, and swap in the result. Blocking."""
        sources = build_sources(
            self.settings.employee_photos_source,
            timeout=self.settings.employee_photos_timeout_seconds,
            auth_header=self.settings.employee_photos_auth_header,
            manifest_name=self.settings.employee_photos_manifest,
        )
        try:
            gallery = build_gallery(
                self.models,
                sources,
                cache=build_cache(
                    self.settings.storage_path,
                    self.settings.models_dir / self.settings.recognize_model,
                ),
            )
        finally:
            for source in sources:
                source.close()
        self.gallery_handle.swap(gallery)
        return gallery

    def start_gallery_build(self) -> None:
        """Enrol on a daemon thread so the API is up in seconds, not minutes."""

        def run() -> None:
            try:
                self.build_gallery_now()
            except Exception:  # noqa: BLE001 - the thread must never die silently
                self._log.exception("gallery build failed")

        threading.Thread(target=run, name="gallery-build", daemon=True).start()

    @property
    def camera_runner(self) -> CameraRunner:
        """The always-on capture loop. Built on first use, since it needs the models."""
        with self._lock:
            if self._camera_runner is None:
                self._camera_runner = CameraRunner(
                    self.settings,
                    self.models,
                    self.gallery_handle,
                    self.frame_hub,
                    reporter=self.attendance_reporter,
                    suppressor=self.duplicate_suppressor,
                )
            return self._camera_runner

    def start_attendance_consumer(self) -> None:
        """Drain the attendance queue into the attendance table, in-process.

        Off by default: nothing should start writing to an ERP database because a
        service happened to boot. Set ATTENDANCE_CONSUMER_INPROC=false and run
        `python -m app.services.attendance_consumer` to give it its own process.
        """
        if not self.settings.attendance_consumer_enabled:
            return
        if not self.settings.attendance_consumer_inproc:
            self._log.info("attendance consumer is enabled but configured out-of-process")
            return
        self.attendance_consumer = ConsumerRunner(build_attendance_consumer(self.settings))
        self.attendance_consumer.start()

    def shutdown(self) -> None:
        with self._lock:
            if self.attendance_consumer is not None:
                self.attendance_consumer.stop()
            if self._camera_runner is not None:
                self._camera_runner.stop()
            if self.erp_sync_scheduler is not None:
                self.erp_sync_scheduler.stop()
            try:
                self.camera_service.stop()
            except Exception:  # noqa: BLE001
                self._log.exception("error stopping camera worker")
            try:
                self.attendance_reporter.close()
            except Exception:  # noqa: BLE001
                self._log.exception("error closing attendance reporter")
