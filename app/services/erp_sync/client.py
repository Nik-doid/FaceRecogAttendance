"""Thin MySQL client for writing attendance rows to the ERP database.

Uses ``PyMySQL`` (pure-Python, lazy-imported like pika in the RabbitMQ reporter) so
the control plane and tests can run without the driver installed; the client is only
constructed when ERP sync is actually enabled.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

_INSERT = (
    "INSERT INTO ct_hr_employee_attendance_log"
    "(attendance_id_no, in_out_mode, verify_mode, log_date_time, device_id, "
    "branch_id, created_by, created_date, log_date_only)"
    "VALUES(%(attendance_id_no)s,%(in_out_mode)s,%(verify_mode)s,%(log_date_time)s,"
    "%(device_id)s,%(branch_id)s,%(created_by)s,%(created_date)s,%(log_date_only)s)"
    "ON DUPLICATE KEY UPDATE attendance_id_no=attendance_id_no"
)

_UPDATE = (
    "UPDATE ct_hr_employee_attendance_log "
    "SET in_out_mode=%(in_out_mode)s, "
    "verify_mode=%(verify_mode)s, "
    "log_date_time=%(log_date_time)s, "
    "device_id=%(device_id)s, "
    "branch_id=%(branch_id)s, "
    "created_by=%(created_by)s, "
    "created_date=%(created_date)s, "
    "log_date_only=%(log_date_only)s "
    "WHERE attendance_id_no=%(attendance_id_no)s AND log_date_only=%(log_date_only)s "
    "ORDER BY id DESC LIMIT 1"
)

@dataclass(frozen=True)
class ErpDbConfig:
    host: str = "localhost"
    port: int = 3306
    database: str = "attendance"
    user: str = "root"
    password: str = ""
    # Employee code -> attendance id lookup (mirrors ct_hr_employee_master table).
    employee_table: str = "ct_hr_employee_master"
    employee_code_column: str = "emp_code"
    employee_id_column: str = "emp_id"
    employee_active_filter: str = "_status"


class ErpMysqlClient:
    """Executes the ERP attendance-log INSERT. Opens a fresh connection per call."""

    def __init__(self, config: ErpDbConfig) -> None:
        self._config = config

    def _get_driver(self) -> Any:
        try:
            import pymysql  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyMySQL is not installed; add it to dependencies to enable ERPDB sync"
            ) from exc
        return pymysql

    def _connect(self) -> Any:
        driver = self._get_driver()
        return driver.connect(
            host=self._config.host,
            port=self._config.port,
            database=self._config.database,
            user=self._config.user,
            password=self._config.password,
            autocommit=True,
        )

    def insert_row(self, row: dict[str, object]) -> bool:
        """Insert one attendance row. Returns True on success, False on failure."""
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(_INSERT, row)
            return True
        except Exception as exc:  # noqa: BLE001 - DB failures are surfaced to the caller
            log.warning("ERP insert failed: %s", exc)
            return False
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def insert_many(self, rows: list[dict[str, object]]) -> list[bool]:
        """Insert several rows. Returns a per-row success list (transactional per row)."""
        return [self.insert_row(r) for r in rows]

    def update_row(self, row: dict[str, object]) -> bool:
        """Update the latest attendance row for the employee+day. Returns True on success."""
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(_UPDATE, row)
            return int(cursor.rowcount) > 0
        except Exception as exc:  # noqa: BLE001
            log.warning("ERP update failed: %s", exc)
            return False
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def upsert_many(
        self,
        rows: list[dict[str, object]],
        is_update: list[bool],
    ) -> list[bool]:
        """Upsert several rows. Returns a per-row success list."""
        results: list[bool] = []
        for row, do_update in zip(rows, is_update, strict=True):
            if do_update:
                results.append(self.update_row(row))
            else:
                results.append(self.insert_row(row))
        return results

    def lookup_employee_id(self, employee_code: str) -> str | None:
        """Resolve an employee code to its ERP attendance id (numeric enroll number).

        Lookup chain: ``ct_hr_employee_master.emp_code`` -> ``emp_id``,
        written as the ``attendance_id_no`` in the log.
        """
        conn = None
        try:
            conn = self._connect()
            table = self._config.employee_table
            code_col = self._config.employee_code_column
            id_col = self._config.employee_id_column
            sql = (
                f"SELECT {id_col} FROM {table} "
                f"WHERE {code_col}=%s"
            )
            if self._config.employee_active_filter:
                sql += f" AND {self._config.employee_active_filter}"
            sql += " LIMIT 1"
            with conn.cursor() as cursor:
                cursor.execute(sql, (employee_code,))
                row = cursor.fetchone()
            if row is None:
                return None
            return str(row[0])
        except Exception as exc:  # noqa: BLE001 - failures surface to the caller
            log.warning("ERP employee lookup failed: %s", exc)
            return None
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def existing_in_out_modes(self, attendance_id_no: str, log_date_only: str) -> set[int]:
        """In/out modes already present in the ERP log for an employee on a day.

        Used to seed the in/out dedup rule so a punch already recorded is never
        written twice.
        """
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT DISTINCT in_out_mode FROM ct_hr_employee_attendance_log "
                    "WHERE attendance_id_no=%s AND log_date_only=%s",
                    (attendance_id_no, log_date_only),
                )
                return {int(row[0]) for row in cursor.fetchall()}
        except Exception as exc:  # noqa: BLE001 - failures surface to the caller
            log.warning("ERP existing in/out lookup failed: %s", exc)
            return set()
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
