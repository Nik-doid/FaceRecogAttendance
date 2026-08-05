"""Recognition-side duplicate suppression.

The existing attendance system owns the business rule "one check-in per day". This
class only stops us from hammering the broker with the SAME person while they stand
in front of the camera: if an employee was reported within the timeout window, we
keep recognizing them but skip the report. It is a performance/politeness concern,
not a business rule.
"""

from __future__ import annotations

import threading
import time


class DuplicateSuppressor:
    def __init__(self, timeout_seconds: int) -> None:
        self._timeout = timeout_seconds
        self._last_reported: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def timeout_seconds(self) -> int:
        with self._lock:
            return self._timeout

    @timeout_seconds.setter
    def timeout_seconds(self, value: int) -> None:
        with self._lock:
            self._timeout = value

    def check_and_record(self, employee_code: str, now: float | None = None) -> bool:
        """Return True if a report is allowed for this employee, and record it.

        Uses ``time.monotonic`` so the window is immune to system clock jumps.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            last = self._last_reported.get(employee_code)
            if last is not None and (now - last) < self._timeout:
                return False
            self._last_reported[employee_code] = now
            return True

    def clear(self) -> None:
        with self._lock:
            self._last_reported.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._last_reported)
