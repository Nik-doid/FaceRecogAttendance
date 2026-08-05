"""Reporter factory — the single place that decides HOW events leave the service.

To integrate with a different mechanism (REST / Kafka / direct DB write / internal
call), add a new reporter class in this package and a branch here. No other code
changes.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.services.attendance_reporter.base import AttendanceReporter
from app.services.attendance_reporter.null import NullAttendanceReporter
from app.services.attendance_reporter.rabbitmq import RabbitMQAttendanceReporter


def build_attendance_reporter(settings: Settings) -> AttendanceReporter:
    if settings.attendance_broker == "null":
        return NullAttendanceReporter()
    if settings.attendance_broker == "rabbitmq":
        return RabbitMQAttendanceReporter(
            url=settings.attendance_mq_url,
            exchange=settings.attendance_exchange,
            routing_key=settings.attendance_routing_key,
            queue=settings.attendance_queue,
            retries=settings.attendance_publish_retries,
        )
    raise ValueError(f"Unknown attendance broker: {settings.attendance_broker}")
