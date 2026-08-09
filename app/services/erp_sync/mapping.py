"""Mapping helpers: camera -> ERP device/branch, and check-in/out resolution.

The ERP ``ct_hr_employee_attendance_log`` table is keyed by the physical attendance
device the punch came from. Our camera therefore needs to map to that device plus a
branch, mirroring how the C# software passes ``device_id`` (machine number) and
``branch_id`` into its insert.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime


class CameraNotMappedError(KeyError):
    """Raised when a recognition event's camera has no ERP device/branch mapping."""


class InOutResolver:
    """Decides the ``in_out_mode`` column value.

    Supported policies (from ``erp_in_out_mode``):
    - a literal int string (e.g. ``"1"``): every event uses that value (all check-ins);
    - ``"toggle"``: alternate check-in(1)/check-out(2) per employee per calendar day,
      so both a first appearance (check-in) and later appearance (check-out) exist.
    """

    CHECK_IN = 1
    CHECK_OUT = 2

    def __init__(self, policy: str) -> None:
        self._policy = policy
        # (employee_code, log_date) -> last emitted in_out_mode, to drive toggling.
        self._last: dict[tuple[str, str], int] = defaultdict(int)

    @property
    def policy(self) -> str:
        return self._policy

    def resolve(self, employee_code: str, log_date: date) -> int:
        if self._policy == "toggle":
            key = (employee_code, log_date.isoformat())
            previous = self._last[key]
            following = (
                self.CHECK_OUT if previous == self.CHECK_IN else self.CHECK_IN
            )
            self._last[key] = following
            return following
        try:
            return int(self._policy)
        except ValueError:
            return self.CHECK_IN

    def reset(self) -> None:
        """Clear toggle state (e.g. after the service restarts)."""
        self._last.clear()


class CameraMapping:
    """Resolve a camera_id to an ERP device_id + branch_id from settings."""

    def __init__(self, mapping: dict[str, dict[str, int]]) -> None:
        self._mapping = mapping or {}

    def resolve(self, camera_id: str) -> tuple[int, int]:
        entry = self._mapping.get(camera_id)
        if entry is None:
            raise CameraNotMappedError(camera_id)
        device_id = int(entry["device_id"])
        branch_id = int(entry["branch_id"])
        return device_id, branch_id

    def contains(self, camera_id: str) -> bool:
        return camera_id in self._mapping


def format_datetime(dt: datetime) -> str:
    """Format a datetime to ``yyyy-MM-dd HH:mm:ss`` (matches the C# insert)."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")
