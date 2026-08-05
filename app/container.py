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
from app.config.settings import Settings
from app.core.logging import get_logger
from app.repositories.audit_repo import AuditLogRepository
from app.repositories.camera_repo import CameraRepository
from app.repositories.face_embedding_repo import FaceEmbeddingRepository
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.repositories.setting_repo import SettingRepository
from app.repositories.unknown_face_repo import UnknownFaceRepository
from app.services.attendance_reporter import build_attendance_reporter
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.camera_service import CameraService
from app.services.duplicate_suppressor import DuplicateSuppressor
from app.services.index_service import IndexService
from app.services.settings_service import SettingsService
from app.storage.snapshot import SnapshotStorage
from app.workers.camera.reader import CameraReader
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
        self._lock = threading.Lock()

        # Shared singletons.
        self.face_index = FaceIndex(dim=EMBEDDING_DIM)
        self.settings_service = SettingsService(SettingRepository())
        self.duplicate_suppressor = DuplicateSuppressor(self.settings.duplicate_timeout_seconds)
        self.snapshot_storage = SnapshotStorage(
            self.settings.snapshots_dir, enabled=self.settings.snapshot_enabled
        )
        self.attendance_reporter: AttendanceReporter = build_attendance_reporter(self.settings)

        # Repositories.
        self.face_embedding_repo = FaceEmbeddingRepository()
        self.recognition_log_repo = RecognitionLogRepository()
        self.unknown_face_repo = UnknownFaceRepository()
        self.camera_repo = CameraRepository()
        self.audit_repo = AuditLogRepository()

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
        return IndexService(
            settings=self.settings,
            face_index=self.face_index,
            detector=self.ai.detector,
            recognizer=self.ai.recognizer,
            quality=self.ai.quality,
            repo=self.face_embedding_repo,
        )

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
        )

    # -- lifecycle -----------------------------------------------------------------
    def start_rebuild(self) -> bool:
        from app.database.session import sync_session

        return self.index_service.start_rebuild(sync_session)

    def shutdown(self) -> None:
        with self._lock:
            try:
                self.camera_service.stop()
            except Exception:  # noqa: BLE001
                self._log.exception("error stopping camera worker")
            try:
                self.attendance_reporter.close()
            except Exception:  # noqa: BLE001
                self._log.exception("error closing attendance reporter")
