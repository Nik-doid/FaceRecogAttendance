"""No-op reporter used in tests and when ATTENDANCE_BROKER=null."""

from __future__ import annotations

from app.services.attendance_reporter.base import (
    AttendanceEvent,
    AttendanceReporter,
    ReportResult,
)


class NullAttendanceReporter(AttendanceReporter):
    def __init__(self) -> None:
        self.events: list[AttendanceEvent] = []

    def report(self, event: AttendanceEvent) -> ReportResult:
        self.events.append(event)
        return ReportResult(success=True, detail="recorded (null reporter)")

    def close(self) -> None:
        pass
