"""Runs the consumer on a daemon thread, mirroring the camera runner's shape."""

from __future__ import annotations

import threading

from app.core.logging import get_logger
from app.services.attendance_consumer.consumer import AttendanceConsumer

log = get_logger(__name__)


class ConsumerRunner:
    def __init__(self, consumer: AttendanceConsumer) -> None:
        self._consumer = consumer
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._thread = threading.Thread(
            target=self._consumer.run_forever, name="attendance-consumer", daemon=True
        )
        self._thread.start()
        log.info("attendance consumer thread started")

    def stop(self, timeout: float = 10.0) -> None:
        self._consumer.stop()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
