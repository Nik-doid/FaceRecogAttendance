"""ERP attendance-log sync: data model and result types.

The sync mirrors exactly what the C# attendance software inserts into
``ct_hr_employee_attendance_log`` (see ``AttendanceLog.cs`` tab InOutMode/VerifyMode
handling and its INSERT statement), so the face-recognition service becomes a drop-in
data source for the same table.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ErpAttendanceRow:
    """One row to write to ct_hr_employee_attendance_log, mirroring the C# insert."""

    attendance_id_no: str
    in_out_mode: int
    verify_mode: str
    log_date_time: str  # yyyy-MM-dd HH:mm:ss
    device_id: int
    branch_id: int
    created_by: str
    created_date: str  # yyyy-MM-dd HH:mm:ss
    log_date_only: str  # yyyy-MM-dd


@dataclass
class ErpSyncStats:
    """Counters from one sync run."""

    scanned: int = 0
    inserted: int = 0
    failed: int = 0
    skipped_unmapped_camera: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ErpSyncResult:
    """Outcome of a sync run, exposed via the API."""

    ok: bool
    stats: ErpSyncStats
    detail: str = ""
