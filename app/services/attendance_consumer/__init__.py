"""RabbitMQ to ct_hr_employee_attendance_log.

The other half of the flow: the camera pipeline publishes attendance events, and this
package consumes them and writes the attendance table. Split into a pure decision
layer (:mod:`policy`), a database layer (:mod:`mysql`), the two joined
(:mod:`writer`), and the broker plumbing (:mod:`consumer`) -- so the punch rules can
be tested exhaustively without a broker or a database in sight.
"""

from app.services.attendance_consumer.consumer import AttendanceConsumer
from app.services.attendance_consumer.factory import build_attendance_consumer
from app.services.attendance_consumer.mysql import AttendanceMysql, MysqlConfig
from app.services.attendance_consumer.runner import ConsumerRunner
from app.services.attendance_consumer.writer import AttendanceLogWriter, WriteConfig

__all__ = [
    "AttendanceConsumer",
    "AttendanceLogWriter",
    "AttendanceMysql",
    "ConsumerRunner",
    "MysqlConfig",
    "WriteConfig",
    "build_attendance_consumer",
]
