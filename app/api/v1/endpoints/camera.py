"""Camera control endpoints (privileged operations require JWT)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AdminDep, ContainerDep, db_session, run_db
from app.core.security import rate_limit
from app.schemas.camera import CameraActionResponse, CameraStatusResponse
from app.services.camera_service import CameraAlreadyRunningError

router = APIRouter(tags=["camera"])


@router.get("/camera/status", response_model=CameraStatusResponse)
async def camera_status(container: ContainerDep) -> CameraStatusResponse:
    camera_id = container.settings.camera_id
    worker = container.camera_service.state

    with db_session() as session:
        row = await run_db(container.camera_repo.get_by_id, session, camera_id)

    status_str = "running" if worker.running else (row.status if row else "stopped")
    return CameraStatusResponse(
        camera_id=camera_id,
        status=status_str,
        last_connected_at=row.last_connected_at if row else None,
        last_error=row.last_error if row else None,
    )


@router.post(
    "/camera/start",
    response_model=CameraActionResponse,
    dependencies=[Depends(rate_limit())],
)
async def camera_start(
    actor: AdminDep,
    container: ContainerDep,
) -> CameraActionResponse:
    try:
        state = container.camera_service.start()
    except CameraAlreadyRunningError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    with db_session() as session:
        await run_db(
            container.audit_repo.add_entry,
            session,
            actor=actor,
            action="camera.start",
            resource=container.settings.camera_id,
        )

    return CameraActionResponse(
        camera_id=container.settings.camera_id,
        action="start",
        status="running" if state.running else "error",
    )


@router.post(
    "/camera/stop",
    response_model=CameraActionResponse,
    dependencies=[Depends(rate_limit())],
)
async def camera_stop(
    actor: AdminDep,
    container: ContainerDep,
) -> CameraActionResponse:
    state = container.camera_service.stop()

    with db_session() as session:
        await run_db(
            container.audit_repo.add_entry,
            session,
            actor=actor,
            action="camera.stop",
            resource=container.settings.camera_id,
        )

    return CameraActionResponse(
        camera_id=container.settings.camera_id,
        action="stop",
        status="stopped" if not state.running else "error",
    )
