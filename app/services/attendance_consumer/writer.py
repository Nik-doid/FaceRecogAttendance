"""Turn one attendance message into one attendance-table write.

All the decisions live here and in :mod:`policy`; :mod:`consumer` only moves messages.
That split is deliberate -- this class needs no broker to test, so the punch matrix
(first of day, second, third, replay, unknown employee, unmapped camera) is exercised
without any AMQP in sight.

Failures are sorted into two kinds, because the broker must treat them differently:
:class:`PermanentFailure` can never succeed however often it is retried, and
:class:`~app.services.attendance_consumer.mysql.TransientDatabaseError` may succeed in
a moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.core.metrics import ATTENDANCE_WRITTEN
from app.services.attendance_consumer.mysql import AttendanceMysql, AttendanceRow
from app.services.attendance_consumer.policy import (
    IN_OUT_MODE,
    Insert,
    Skip,
    Update,
    decide,
    format_date,
    format_datetime,
    to_local,
)

log = get_logger(__name__)

REASON_MALFORMED = "malformed_message"
REASON_UNMAPPED_CAMERA = "unmapped_camera"
REASON_UNKNOWN_EMPLOYEE = "unknown_employee"


class PermanentFailure(Exception):
    """No amount of retrying will help. ``reason`` keys the dead-letter metrics."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


@dataclass(frozen=True)
class WriteConfig:
    camera_mapping: dict[str, dict[str, int]]
    verify_mode: str = "FACE"
    created_by: str = "system"
    timezone: str = "UTC"
    min_punch_gap_seconds: int = 60


@dataclass(frozen=True)
class Outcome:
    action: str  # "inserted" | "updated" | "skipped"
    employee_code: str
    detail: str = ""


class AttendanceLogWriter:
    def __init__(self, client: AttendanceMysql, config: WriteConfig) -> None:
        self._client = client
        self._config = config

    def handle(self, message: dict[str, Any]) -> Outcome:
        """Apply one event. Raises PermanentFailure or TransientDatabaseError."""
        employee_code = _require(message, "employee_code")
        camera_id = _require(message, "camera_id")
        captured_at = _timestamp(message)

        device_id, branch_id = self._resolve_camera(camera_id)
        attendance_id_no = self._client.lookup_employee_id(employee_code)
        if not attendance_id_no:
            raise PermanentFailure(
                REASON_UNKNOWN_EMPLOYEE,
                f"{employee_code} is not in the employee table",
            )

        punch_at = to_local(captured_at, self._config.timezone)
        log_date_only = format_date(punch_at)
        rows = self._client.day_rows(attendance_id_no, log_date_only)
        decision = decide(rows, punch_at, self._config.min_punch_gap_seconds)

        if isinstance(decision, Skip):
            log.info(
                "attendance punch skipped",
                extra={
                    "event": "attendance_skipped",
                    "employee_code": employee_code,
                    "reason": decision.reason,
                },
            )
            ATTENDANCE_WRITTEN.labels(action="skipped").inc()
            return Outcome("skipped", employee_code, decision.reason)

        row = AttendanceRow(
            attendance_id_no=attendance_id_no,
            in_out_mode=IN_OUT_MODE,
            verify_mode=self._config.verify_mode,
            log_date_time=format_datetime(punch_at),
            device_id=device_id,
            branch_id=branch_id,
            created_by=self._config.created_by,
            created_date=format_datetime(to_local(datetime.now(UTC), self._config.timezone)),
            log_date_only=log_date_only,
        )

        if isinstance(decision, Insert):
            self._client.insert(row)
            action = "inserted"
        else:
            assert isinstance(decision, Update)
            if not self._client.update(decision.row_id, row):
                # The row was there a moment ago and is not now. Inserting instead
                # would be guessing at the day's shape; let it be redelivered and
                # re-decided against whatever the table looks like then.
                raise PermanentFailure(
                    REASON_MALFORMED, f"attendance row {decision.row_id} disappeared"
                )
            action = "updated"

        ATTENDANCE_WRITTEN.labels(action=action).inc()
        log.info(
            "attendance recorded",
            extra={
                "event": "attendance_recorded",
                "employee_code": employee_code,
                "camera_id": camera_id,
                "action": action,
                "punches_today": len(rows) + (1 if action == "inserted" else 0),
            },
        )
        return Outcome(action, employee_code)

    def _resolve_camera(self, camera_id: str) -> tuple[int, int]:
        entry = self._config.camera_mapping.get(camera_id)
        if not entry:
            raise PermanentFailure(
                REASON_UNMAPPED_CAMERA, f"{camera_id} is not in ERP_CAMERA_MAPPING"
            )
        try:
            return int(entry["device_id"]), int(entry["branch_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PermanentFailure(
                REASON_UNMAPPED_CAMERA, f"{camera_id} mapping is malformed: {exc}"
            ) from exc


def _require(message: dict[str, Any], key: str) -> str:
    value = message.get(key)
    if not value or not isinstance(value, str):
        raise PermanentFailure(REASON_MALFORMED, f"missing {key}")
    return str(value)


def _timestamp(message: dict[str, Any]) -> datetime:
    raw = message.get("timestamp")
    if not isinstance(raw, str):
        raise PermanentFailure(REASON_MALFORMED, "missing timestamp")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise PermanentFailure(REASON_MALFORMED, f"bad timestamp {raw!r}") from exc
    # Events are published as aware UTC; a naive one would silently be treated as
    # local time by astimezone, shifting the punch.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
