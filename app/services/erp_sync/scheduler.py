"""Lightweight interval scheduler for the ERP attendance-log sync.

Runs the sync on a configurable interval in a daemon thread, mirroring the
RecognitionLoop worker pattern (never runs inside the FastAPI event loop). A cron
expression can be approximated by setting the interval; callers may also trigger runs
on demand via the API without starting this scheduler.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from app.core.logging import get_logger

log = get_logger(__name__)


class ErpSyncScheduler:
    def __init__(
        self,
        run_once: Callable[[], object],
        *,
        interval_seconds: int,
    ) -> None:
        self._run_once = run_once
        self._interval = max(1, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interval_seconds(self) -> int:
        return self._interval

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="erp-sync-scheduler", daemon=True
        )
        self._thread.start()
        log.info("ERP sync scheduler started", extra={"interval_seconds": self._interval})

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("ERP sync scheduler stopped")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sleep(self._interval)
            if self._stop.is_set():
                break
            try:
                result = self._run_once()
                log.info("ERP sync scheduler pass done", extra={"result": str(result)})
            except Exception:  # noqa: BLE001 - scheduler must never die
                log.exception("ERP sync scheduler pass failed")

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop.is_set():
            time.sleep(min(0.25, end - time.monotonic()))
