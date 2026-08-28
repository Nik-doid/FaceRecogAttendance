"""The only code that writes to a database this service does not own.

Rewritten from ``app/services/erp_sync/client.py``, which opened a fresh connection
per statement (three TCP handshakes and three auth round-trips per event) and read
``cursor.rowcount`` after the ``with`` block had already closed the cursor. Here one
connection is kept and revalidated with ``ping(reconnect=True)``.

The day-state query replaces ``SELECT DISTINCT in_out_mode``, which could never
report more than one punch because every row is written with the same mode. Counting
actual rows is what lets :mod:`policy` tell a first punch from a third.

Correct for exactly one consumer at ``prefetch_count=1``. The read-then-write is not
in a transaction, so two consumers could both see one row and both insert. Scaling out
means wrapping the pair in ``START TRANSACTION; SELECT ... FOR UPDATE; COMMIT`` on a
non-autocommit connection -- deliberately not built now, since nothing needs it.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.logging import get_logger
from app.services.attendance_consumer.policy import DayRow

log = get_logger(__name__)

TABLE = "ct_hr_employee_attendance_log"

_INSERT = f"""
INSERT INTO {TABLE}
    (attendance_id_no, in_out_mode, verify_mode, log_date_time,
     device_id, branch_id, created_by, created_date, log_date_only)
VALUES
    (%(attendance_id_no)s, %(in_out_mode)s, %(verify_mode)s, %(log_date_time)s,
     %(device_id)s, %(branch_id)s, %(created_by)s, %(created_date)s, %(log_date_only)s)
"""

_UPDATE = f"""
UPDATE {TABLE}
   SET log_date_time = %(log_date_time)s,
       verify_mode   = %(verify_mode)s,
       device_id     = %(device_id)s,
       branch_id     = %(branch_id)s,
       created_by    = %(created_by)s,
       created_date  = %(created_date)s
 WHERE id = %(row_id)s
"""

_DAY_ROWS = f"""
SELECT id, log_date_time
  FROM {TABLE}
 WHERE attendance_id_no = %s AND log_date_only = %s
 ORDER BY log_date_time, id
"""


class TransientDatabaseError(RuntimeError):
    """The write may succeed later: connection lost, deadlock, timeout."""


@dataclass(frozen=True)
class MysqlConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    employee_table: str = "ct_hr_employee_master"
    employee_code_column: str = "emp_code"
    employee_id_column: str = "emp_id"
    employee_active_filter: str = ""
    connect_timeout: int = 10
    lookup_ttl_seconds: int = 3600


@dataclass(frozen=True)
class AttendanceRow:
    attendance_id_no: str
    in_out_mode: int
    verify_mode: str
    log_date_time: str
    device_id: int
    branch_id: int
    created_by: str
    created_date: str
    log_date_only: str

    def as_params(self) -> dict[str, Any]:
        return {
            "attendance_id_no": self.attendance_id_no,
            "in_out_mode": self.in_out_mode,
            "verify_mode": self.verify_mode,
            "log_date_time": self.log_date_time,
            "device_id": self.device_id,
            "branch_id": self.branch_id,
            "created_by": self.created_by,
            "created_date": self.created_date,
            "log_date_only": self.log_date_only,
        }


class AttendanceMysql:
    """A thin, long-lived PyMySQL client for the attendance and employee tables."""

    def __init__(self, config: MysqlConfig) -> None:
        self._config = config
        self._connection: Any = None
        self._lock = threading.Lock()
        self._employee_cache: dict[str, tuple[str | None, float]] = {}

    # -- connection -----------------------------------------------------------
    def _driver(self) -> Any:
        try:
            import pymysql  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyMySQL is not installed; it is required to write attendance rows"
            ) from exc
        return pymysql

    def _connect(self) -> Any:
        """Return a live connection, reconnecting if the server dropped it."""
        if self._connection is not None:
            try:
                self._connection.ping(reconnect=True)
                return self._connection
            except Exception:  # noqa: BLE001 - fall through to a fresh connection
                self._connection = None

        driver = self._driver()
        try:
            self._connection = driver.connect(
                host=self._config.host,
                port=self._config.port,
                database=self._config.database,
                user=self._config.user,
                password=self._config.password,
                connect_timeout=self._config.connect_timeout,
                autocommit=True,
            )
        except Exception as exc:  # noqa: BLE001 - every driver error is retryable here
            raise TransientDatabaseError(str(exc)) from exc
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                with contextlib.suppress(Exception):  # best-effort close
                    self._connection.close()
                self._connection = None

    # -- reads ----------------------------------------------------------------
    def day_rows(self, attendance_id_no: str, log_date_only: str) -> list[DayRow]:
        """Every punch already recorded for this employee on this local date."""
        with self._lock:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_DAY_ROWS, (attendance_id_no, log_date_only))
                    fetched = cursor.fetchall()
            except Exception as exc:  # noqa: BLE001
                raise TransientDatabaseError(str(exc)) from exc
        return [DayRow(row_id=int(row[0]), log_date_time=_as_datetime(row[1])) for row in fetched]

    def lookup_employee_id(self, employee_code: str) -> str | None:
        """Resolve employee_code to the attendance table's id, or None if unknown.

        Memoised: the mapping changes when someone is hired, not per punch, and this
        otherwise runs a query for every message.
        """
        now = time.monotonic()
        cached = self._employee_cache.get(employee_code)
        if cached is not None and now - cached[1] < self._config.lookup_ttl_seconds:
            return cached[0]

        config = self._config
        # Identifiers come from configuration, not from the message, so they are
        # interpolated; only the code itself is ever bound as a parameter.
        sql = (
            f"SELECT {config.employee_id_column} FROM {config.employee_table} "
            f"WHERE {config.employee_code_column}=%s"
        )
        if config.employee_active_filter:
            sql += f" AND {config.employee_active_filter}"
        sql += " LIMIT 1"

        with self._lock:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, (employee_code,))
                    row = cursor.fetchone()
            except Exception as exc:  # noqa: BLE001
                raise TransientDatabaseError(str(exc)) from exc

        resolved = str(row[0]) if row else None
        self._employee_cache[employee_code] = (resolved, now)
        return resolved

    # -- writes ---------------------------------------------------------------
    def insert(self, row: AttendanceRow) -> None:
        with self._lock:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_INSERT, row.as_params())
            except Exception as exc:  # noqa: BLE001
                raise TransientDatabaseError(str(exc)) from exc

    def update(self, row_id: int, row: AttendanceRow) -> bool:
        """Move an existing row's timestamp forward. False if the row vanished."""
        params = row.as_params()
        params["row_id"] = row_id
        with self._lock:
            connection = self._connect()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(_UPDATE, params)
                    # Read inside the block: the previous implementation read this
                    # after the `with` had closed the cursor.
                    affected = int(cursor.rowcount)
            except Exception as exc:  # noqa: BLE001
                raise TransientDatabaseError(str(exc)) from exc
        return affected > 0


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
