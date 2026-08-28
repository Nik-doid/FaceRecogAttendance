"""Dependency-injection container.

Constructs every long-lived dependency once and hands them to the API and the camera
runner. Models and the gallery are lazy: a control-plane-only test can build a
Container without paying a 174 MiB model load it will never use.
"""

from __future__ import annotations

import threading

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
from app.runtime import Models, load_models
from app.services.attendance_consumer import ConsumerRunner, build_attendance_consumer
from app.services.attendance_reporter import build_attendance_reporter
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.duplicate_suppressor import DuplicateSuppressor


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._log = get_logger(__name__)
        # Reentrant: the ``gallery`` property builds under the lock and reaches
        # through ``models``, which takes it again.
        self._lock = threading.RLock()

        self.attendance_reporter: AttendanceReporter = build_attendance_reporter(self.settings)
        # A publish-rate limiter, not a business rule: one hand-raise spans several
        # scans. How many punches a day count is decided by the consumer.
        self.duplicate_suppressor = DuplicateSuppressor(
            self.settings.duplicate_timeout_seconds
        )
        self.frame_hub = FrameHub()
        self.gallery_handle = GalleryHandle()
        self._models: Models | None = None
        self._camera_runner: CameraRunner | None = None
        self.attendance_consumer: ConsumerRunner | None = None

    # -- lazily built singletons ----------------------------------------------
    @property
    def models(self) -> Models:
        """The process-wide SCRFD + ArcFace sessions, built on first use."""
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

    @property
    def camera_runner(self) -> CameraRunner:
        """The always-on capture loop. Built on first use, since it needs the models."""
        with self._lock:
            if self._camera_runner is None:
                self._camera_runner = CameraRunner(
                    self.settings,
                    # A factory, so building the runner does not load the models --
                    # /camera/status must be able to say "stopped" for free.
                    lambda: self.models,
                    self.gallery_handle,
                    self.frame_hub,
                    reporter=self.attendance_reporter,
                    suppressor=self.duplicate_suppressor,
                )
            return self._camera_runner

    # -- lifecycle -------------------------------------------------------------
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
            try:
                self.attendance_reporter.close()
            except Exception:  # noqa: BLE001
                self._log.exception("error closing the attendance reporter")
