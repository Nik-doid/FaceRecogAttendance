"""Builds the consumer from settings. The single place that reads config for it."""

from __future__ import annotations

from app.config.settings import Settings
from app.core.logging import get_logger
from app.services.attendance_consumer.consumer import AttendanceConsumer
from app.services.attendance_consumer.mysql import AttendanceMysql, MysqlConfig
from app.services.attendance_consumer.writer import AttendanceLogWriter, WriteConfig

log = get_logger(__name__)


def build_attendance_consumer(settings: Settings) -> AttendanceConsumer:
    if settings.attendance_timezone.upper() == "UTC":
        # Almost certainly not what an attendance table wants: in Nepal (UTC+05:45)
        # a 09:15 arrival lands as 03:30, and a 22:00 punch on the previous day.
        log.warning(
            "ATTENDANCE_TIMEZONE is UTC, so punches are logged in UTC. Set it to the "
            "zone the attendance table is read in (e.g. Asia/Kathmandu) if that is wrong."
        )

    client = AttendanceMysql(
        MysqlConfig(
            host=settings.erp_db_host,
            port=settings.erp_db_port,
            database=settings.erp_db_name,
            user=settings.erp_db_user,
            password=settings.erp_db_password,
            employee_table=settings.erp_employee_table,
            employee_code_column=settings.erp_employee_code_column,
            employee_id_column=settings.erp_employee_id_column,
            employee_active_filter=settings.erp_employee_active_filter,
        )
    )
    writer = AttendanceLogWriter(
        client,
        WriteConfig(
            camera_mapping=settings.erp_camera_mapping,
            verify_mode=settings.erp_verify_mode,
            created_by=settings.erp_created_by,
            timezone=settings.attendance_timezone,
            min_punch_gap_seconds=settings.attendance_min_punch_gap_seconds,
        ),
    )
    return AttendanceConsumer(
        writer,
        url=settings.attendance_mq_url,
        queue=settings.attendance_queue,
        dead_letter_exchange=settings.attendance_dead_letter_exchange,
        prefetch=settings.attendance_consumer_prefetch,
    )
