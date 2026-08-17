"""ERP attendance-log sync service.

Pulls recognised + reported recognition events that have not yet been written to the
existing attendance system's ``ct_hr_employee_attendance_log`` table and inserts them,
mirroring the C# software's INSERT. On success each row is stamped ``erp_synced_at`` so
the cron never re-inserts (idempotent).

The employee code recorded by recognition is resolved to the ERP attendance id via the
``ct_hr_employee_master``-style table, exactly like the C# software. In/out mode follows
the ERP state for that employee+day, so a punch the C# software already recorded is never
written twice, and attendance snapshots are only kept for events that were actually saved
to the ERP DB.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.metrics import ERP_SYNC_FAILED, ERP_SYNC_INSERTED, ERP_SYNC_PENDING
from app.models.recognition_log import RecognitionLog
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.services.erp_sync.base import ErpSyncResult, ErpSyncStats
from app.services.erp_sync.client import ErpMysqlClient
from app.services.erp_sync.mapping import (
    CameraMapping,
    CameraNotMappedError,
    InOutResolver,
    format_datetime,
)
from app.storage.snapshot import SnapshotStorage

log = get_logger(__name__)

SKIP_UNMAPPED_CAMERA = "unmapped_camera"
SKIP_NO_EMPLOYEE = "no_employee_id"
SKIP_DUPLICATE_IN_OUT = "duplicate_in_out"


class ErpSyncService:
    def __init__(
        self,
        repo: RecognitionLogRepository,
        client: ErpMysqlClient,
        camera_mapping: CameraMapping,
        in_out_resolver: InOutResolver,
        *,
        verify_mode: str,
        created_by: str,
        batch_size: int = 500,
        enabled: bool = True,
        snapshot_storage: SnapshotStorage | None = None,
    ) -> None:
        self._repo = repo
        self._client = client
        self._camera_mapping = camera_mapping
        self._in_out = in_out_resolver
        self._verify_mode = verify_mode
        self._created_by = created_by
        self._batch_size = batch_size
        self._enabled = enabled
        self._snapshots = snapshot_storage

    @property
    def enabled(self) -> bool:
        return self._enabled

    def count_pending(self, session: Session) -> int:
        """Number of recognition events still awaiting an ERP attendance-log write."""
        return self._repo.count_pending_erp_sync(session)

    def sync_once(self, session_factory: Callable[[], Session]) -> ErpSyncResult:
        """Run one sync pass. Safely no-ops when disabled or no pending rows."""
        stats = ErpSyncStats()
        if not self._enabled:
            return ErpSyncResult(ok=True, stats=stats, detail="erp sync disabled")
        try:
            with session_factory() as session:
                pending = self._repo.list_pending_erp_sync(
                    session, limit=self._batch_size
                )
                stats.scanned = len(pending)
                ERP_SYNC_PENDING.set(self._repo.count_pending_erp_sync(session))
                if not pending:
                    return ErpSyncResult(ok=True, stats=stats, detail="no pending events")

                rows: list[dict[str, object]] = []
                is_update: list[bool] = []
                synced_ids: list[int] = []
                skipped: dict[int, str] = {}
                now = datetime.now(UTC)
                seeded: set[tuple[str, date]] = set()

                for record in pending:
                    row, record_id, skip_reason, is_upd = self._to_row(record, now, stats, seeded)
                    if skip_reason:
                        skipped[record_id] = skip_reason
                        self._discard_snapshot(record)
                        continue
                    if row is None:
                        continue
                    rows.append(row)
                    is_update.append(is_upd)
                    synced_ids.append(record_id)

                if skipped:
                    by_reason: dict[str, list[int]] = {}
                    for rec_id, reason in skipped.items():
                        by_reason.setdefault(reason, []).append(rec_id)
                    for reason, rec_ids in by_reason.items():
                        self._repo.mark_erp_skipped(session, rec_ids, reason)

                if rows:
                    results = self._client.upsert_many(rows, is_update)
                    inserted_ids: list[int] = []
                    updated_ids: list[int] = []
                    for ok, rec_id, upd in zip(results, synced_ids, is_update, strict=True):
                        if ok:
                            stats.inserted += 1
                            ERP_SYNC_INSERTED.inc()
                            if upd:
                                updated_ids.append(rec_id)
                            else:
                                inserted_ids.append(rec_id)
                        else:
                            stats.failed += 1
                            ERP_SYNC_FAILED.inc()
                            stats.errors.append(f"ERP upsert failed for log id {rec_id}")

                    if inserted_ids or updated_ids:
                        self._repo.mark_erp_synced(session, inserted_ids + updated_ids, now)
        except Exception as exc:  # noqa: BLE001 - never crash the scheduler thread
            log.exception("ERP sync pass failed")
            stats.errors.append(str(exc))
            return ErpSyncResult(ok=False, stats=stats, detail=str(exc))

        return ErpSyncResult(ok=True, stats=stats)

    def _discard_snapshot(self, record: RecognitionLog) -> None:
        """Delete the snapshot for an event the ERP never accepted (not kept offline)."""
        if self._snapshots is not None and record.snapshot_path:
            self._snapshots.remove(record.snapshot_path)

    def _to_row(
        self,
        record: RecognitionLog,
        now: datetime,
        stats: ErpSyncStats,
        seeded: set[tuple[str, date]],
    ) -> tuple[dict[str, object] | None, int, str | None, bool]:
        """Build the ERP row for a record, or a skip reason.

        Returns ``(row, record_id, None, is_update)`` to upsert,
        ``(None, record_id, reason, False)`` to skip, or ``(None, record_id, None, False)``
        for an internal no-op (never occurs).
        """
        assert record.employee_code is not None
        try:
            device_id, branch_id = self._camera_mapping.resolve(record.camera_id)
        except CameraNotMappedError:
            stats.skipped_unmapped_camera += 1
            log.warning(
                "camera not mapped to ERP device; skipping",
                extra={"camera_id": record.camera_id, "employee_code": record.employee_code},
            )
            return None, record.id, SKIP_UNMAPPED_CAMERA, False

        employee_id = self._client.lookup_employee_id(record.employee_code)
        if not employee_id:
            stats.skipped_no_employee += 1
            log.warning(
                "no ERP employee id for code; skipping",
                extra={"employee_code": record.employee_code, "camera_id": record.camera_id},
            )
            return None, record.id, SKIP_NO_EMPLOYEE, False

        log_dt = record.timestamp
        log_date = log_dt.date()
        key = (employee_id, log_date)
        if key not in seeded:
            existing = self._client.existing_in_out_modes(employee_id, log_date.isoformat())
            self._in_out.seed(employee_id, log_date, existing)
            seeded.add(key)

        in_out, is_update = self._in_out.resolve(employee_id, log_date)
        if in_out is None:
            stats.skipped_duplicate_in_out += 1
            log.info(
                "check-in and check-out already recorded for the day; skipping",
                extra={"employee_id": employee_id, "log_date": log_date.isoformat()},
            )
            return None, record.id, SKIP_DUPLICATE_IN_OUT, False

        row: dict[str, object] = {
            "attendance_id_no": employee_id,
            "in_out_mode": in_out,
            "verify_mode": self._verify_mode,
            "log_date_time": format_datetime(log_dt),
            "device_id": device_id,
            "branch_id": branch_id,
            "created_by": self._created_by,
            "created_date": format_datetime(now),
            "log_date_only": log_date.isoformat(),
        }
        return row, record.id, None, is_update
