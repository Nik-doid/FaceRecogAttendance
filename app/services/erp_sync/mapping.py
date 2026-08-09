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
    - ``"toggle"``: proper per-employee-per-day rule — the first punch of the day is a
      check-in(1), the second is a check-out(2), and any further punches that day are
      skipped entirely (no duplicate check-ins/outs). The rule is seeded from rows
      already present in ``ct_hr_employee_attendance_log`` for that employee/day, so a
      punch already saved by the C# software is never written twice.
    """

    CHECK_IN = 1
    CHECK_OUT = 2

    def __init__(self, policy: str) -> None:
        self._policy = policy
        # (attendance_id_no, log_date) -> in/out modes already seen (ERP + this run).
        self._seen: dict[tuple[str, str], set[int]] = defaultdict(set)

    @property
    def policy(self) -> str:
        return self._policy

    def seed(self, attendance_id_no: str, log_date: date, modes: set[int]) -> None:
        """Pre-load the modes already present in the ERP log for this employee/day."""
        self._seen[(attendance_id_no, log_date.isoformat())].update(modes)

    def resolve(self, attendance_id_no: str, log_date: date) -> int | None:
        """Return the in_out_mode for this punch, or None to skip (duplicate).

        With a literal policy every event uses that value. With ``"toggle"`` the
        first event of the day becomes a check-in, the second a check-out, and any
        event after both exist for the day is skipped (None).
        """
        if self._policy != "toggle":
            try:
                return int(self._policy)
            except ValueError:
                return self.CHECK_IN
        key = (attendance_id_no, log_date.isoformat())
        seen = self._seen[key]
        if self.CHECK_IN not in seen:
            seen.add(self.CHECK_IN)
            return self.CHECK_IN
        if self.CHECK_OUT not in seen:
            seen.add(self.CHECK_OUT)
            return self.CHECK_OUT
        return None

    def reset(self) -> None:
        """Clear per-day state (e.g. after a sync run)."""
        self._seen.clear()


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
