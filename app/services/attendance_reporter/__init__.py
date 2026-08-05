"""Attendance Reporter — the integration seam to the EXISTING attendance system.

ARCHITECTURE DECISION (read before changing anything here):
This is the ONLY module that knows how attendance events reach the existing
attendance system. Everything upstream (recognition pipeline) depends on the
``AttendanceReporter`` interface and nothing else. If the integration contract
changes (e.g. from RabbitMQ to REST, Kafka, or a direct DB write), you swap the
concrete reporter in ``build_attendance_reporter`` and the pipeline is untouched.

Confirmed contract with the existing system: **message queue** (RabbitMQ topic
exchange). The attendance system consumes ``{exchange}/{routing_key}`` messages.
Deduplication, shift windows and persistence are the attendance system's business;
this service only avoids spamming the broker via the DuplicateSuppressor.
"""

from app.services.attendance_reporter.base import (
    AttendanceEvent,
    AttendanceReporter,
    ReportResult,
)
from app.services.attendance_reporter.factory import build_attendance_reporter
from app.services.attendance_reporter.null import NullAttendanceReporter
from app.services.attendance_reporter.rabbitmq import RabbitMQAttendanceReporter

__all__ = [
    "AttendanceEvent",
    "AttendanceReporter",
    "NullAttendanceReporter",
    "RabbitMQAttendanceReporter",
    "ReportResult",
    "build_attendance_reporter",
]
