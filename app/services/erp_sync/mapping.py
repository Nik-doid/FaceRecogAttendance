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

    The ERP expects ``in_out_mode = 255`` for all punches. The first punch
    of the day is inserted; subsequent punches on the same day for the same
    employee update the existing row (check-out). No more than 2 rows per
    employee per day (check-in + check-out).
    """

    MODE = 255

    def __init__(self, policy: str) -> None:
        self._policy = policy
        # (attendance_id_no, log_date) -> punch count this run
        self._count: dict[tuple[str, str], int] = defaultdict(int)
        # (attendance_id_no, log_date) -> modes already seeded from ERP
        self._seeded: dict[tuple[str, str], set[int]] = defaultdict(set)

    @property
    def policy(self) -> str:
        return self._policy

    def seed(self, attendance_id_no: str, log_date: date, modes: set[int]) -> None:
        """Pre-load the modes already present in the ERP log for this employee/day."""
        self._seeded[(attendance_id_no, log_date.isoformat())].update(modes)
        # If ERP already has a punch, count it
        self._count[(attendance_id_no, log_date.isoformat())] = len(modes)

    def resolve(self, attendance_id_no: str, log_date: date) -> tuple[int | None, bool]:
        """Return (in_out_mode, is_update) for this punch, or (None, False) to skip.

        - First punch: returns (MODE, False) -> INSERT
        - Second punch: returns (MODE, True) -> UPDATE
        - Third+ punch: returns (None, False) -> skip
        """
        key = (attendance_id_no, log_date.isoformat())
        count = self._count[key]

        if count >= 2:
            return None, False

        self._count[key] = count + 1
        is_update = count >= 1
        return self.MODE, is_update

    def reset(self) -> None:
        """Clear per-day state (e.g. after a sync run)."""
        self._count.clear()
        self._seeded.clear()


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
