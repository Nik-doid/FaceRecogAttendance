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
)

@dataclass(frozen=True)
class ErpDbConfig:
    host: str = "localhost"
    port: int = 3306
    database: str = "attendance"
    user: str = "root"
    password: str = ""


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
