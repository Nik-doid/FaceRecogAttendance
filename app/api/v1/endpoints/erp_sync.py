"""ERP attendance-log sync endpoints.

Triggers a run of the service that writes recognised attendance events into the
existing C# attendance system's ``ct_hr_employee_attendance_log`` table, and reports
sync status/pending backlog.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import AdminDep, ContainerDep, db_session, run_db
from app.core.security import rate_limit
from app.database.session import sync_session
from app.schemas.erp_sync import ErpSyncResponse, ErpSyncStatsOut, ErpSyncStatusResponse

router = APIRouter(tags=["sync"])


@router.get("/sync/attendance-log/status", response_model=ErpSyncStatusResponse)
async def erp_sync_status(container: ContainerDep) -> ErpSyncStatusResponse:
    service = container.erp_sync_service
    with db_session() as session:
        pending = await run_db(service.count_pending, session)
    scheduler = container.erp_sync_scheduler
    return ErpSyncStatusResponse(
        message="erp attendance-log sync status",
        enabled=service.enabled,
        pending=pending,
        interval_seconds=scheduler.interval_seconds if scheduler else None,
    )


@router.post(
    "/sync/attendance-log",
    response_model=ErpSyncResponse,
    dependencies=[Depends(rate_limit())],
)
async def erp_sync_run(
    actor: AdminDep,
    container: ContainerDep,
) -> ErpSyncResponse:
    """Run one ERP sync pass and record the outcome in the audit log."""
    result = await run_db(container.erp_sync_service.sync_once, sync_session)

    with db_session() as session:
        await run_db(
            container.audit_repo.add_entry,
            session,
            actor=actor,
            action="sync.attendance_log",
            resource="erp",
            detail=(
                f"scanned={result.stats.scanned} inserted={result.stats.inserted} "
                f"failed={result.stats.failed} ok={result.ok}"
            ),
        )

    return ErpSyncResponse(
        ok=result.ok,
        stats=ErpSyncStatsOut(
            scanned=result.stats.scanned,
            inserted=result.stats.inserted,
            failed=result.stats.failed,
            skipped_unmapped_camera=result.stats.skipped_unmapped_camera,
        ),
        detail=result.detail,
    )
