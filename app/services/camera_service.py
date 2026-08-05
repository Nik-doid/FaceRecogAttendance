"""Camera worker lifecycle manager.

Separates "which camera + how it processes" (the RecognitionLoop, built by the DI
container) from "start/stop/status" (this service, used by the API). A single
camera is the documented deployment model; the service makes it trivial to later
extend to multiple workers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging import get_logger
from app.workers.recognition_loop.loop import RecognitionLoop


class CameraAlreadyRunningError(RuntimeError):
    pass


@dataclass
class CameraWorkerState:
    running: bool
    started_at: datetime | None = field(default=None)
    last_error: str | None = field(default=None)


class CameraService:
    def __init__(self, loop_factory: Callable[[], RecognitionLoop]) -> None:
        self._loop_factory = loop_factory
        self._loop: RecognitionLoop | None = None
        self._log = get_logger(__name__)

    def start(self) -> CameraWorkerState:
        if self._loop is not None and self._loop.running:
            raise CameraAlreadyRunningError("camera worker is already running")
        loop = self._loop_factory()
        loop.start()
        self._loop = loop
        self._log.info("camera worker started")
        return CameraWorkerState(running=True, started_at=datetime.now())

    def stop(self) -> CameraWorkerState:
        if self._loop is None:
            return CameraWorkerState(running=False)
        self._loop.stop()
        self._loop = None
        self._log.info("camera worker stopped")
        return CameraWorkerState(running=False)

    @property
    def state(self) -> CameraWorkerState:
        if self._loop is not None and self._loop.running:
            return CameraWorkerState(running=True, started_at=datetime.now())
        return CameraWorkerState(running=False)
