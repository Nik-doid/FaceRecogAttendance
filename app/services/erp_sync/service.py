"""ERP attendance-log sync service.

Pulls recognised + reported recognition events that have not yet been written to the
existing attendance system's ``ct_hr_employee_attendance_log`` table and inserts them,
mirroring the C# software's INSERT. On success each row is stamped ``erp_synced_at`` so
the cron never re-inserts (idempotent).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

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

log = get_logger(__name__)


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
    ) -> None:
        self._repo = repo
        self._client = client
        self._camera_mapping = camera_mapping
        self._in_out = in_out_resolver
        self._verify_mode = verify_mode
        self._created_by = created_by
        self._batch_size = batch_size
        self._enabled = enabled

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
                synced_ids: list[int] = []
                now = datetime.now(UTC)

                for record in pending:
                    mapped = self._to_row(record, now, stats)
                    if mapped is None:
                        continue
                    rows.append(mapped[0])
                    synced_ids.append(mapped[1])

                if rows:
                    results = self._client.insert_many(rows)
                    for ok, rec_id in zip(results, synced_ids, strict=True):
                        if ok:
                            stats.inserted += 1
                            ERP_SYNC_INSERTED.inc()
                        else:
                            stats.failed += 1
                            ERP_SYNC_FAILED.inc()
                            stats.errors.append(f"ERP insert failed for log id {rec_id}")

                    if stats.inserted:
                        self._repo.mark_erp_synced(session, synced_ids, now)
        except Exception as exc:  # noqa: BLE001 - never crash the scheduler thread
            log.exception("ERP sync pass failed")
            stats.errors.append(str(exc))
            return ErpSyncResult(ok=False, stats=stats, detail=str(exc))

        return ErpSyncResult(ok=True, stats=stats)

    def _to_row(
        self, record: RecognitionLog, now: datetime, stats: ErpSyncStats
    ) -> tuple[dict[str, object], int] | None:
        assert record.employee_code is not None
        try:
            device_id, branch_id = self._camera_mapping.resolve(record.camera_id)
        except CameraNotMappedError:
            stats.skipped_unmapped_camera += 1
            log.warning(
                "camera not mapped to ERP device; skipping",
                extra={"camera_id": record.camera_id, "employee_code": record.employee_code},
            )
            return None

        log_dt = record.timestamp
        log_date = log_dt.date()
        in_out = self._in_out.resolve(record.employee_code, log_date)

        row: dict[str, object] = {
            "attendance_id_no": record.employee_code,
            "in_out_mode": in_out,
            "verify_mode": self._verify_mode,
            "log_date_time": format_datetime(log_dt),
            "device_id": device_id,
            "branch_id": branch_id,
            "created_by": self._created_by,
            "created_date": format_datetime(now),
            "log_date_only": log_date.isoformat(),
        }
        return row, record.id
