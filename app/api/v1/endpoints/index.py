"""Face index control endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import AdminDep, ContainerDep, db_session, run_db
from app.core.security import rate_limit
from app.schemas.common import MessageResponse
from app.schemas.index import IndexStatusResponse, RebuildResponse

router = APIRouter(tags=["index"])


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status(container: ContainerDep) -> IndexStatusResponse:
    return IndexStatusResponse(
        status="building" if container.index_service.building else "idle",
        size=container.face_index.size,
        employees=container.index_service.employee_count,
        last_built_at=container.index_service.last_built_at,
        last_error=container.index_service.last_error,
    )


@router.post(
    "/index/rebuild",
    response_model=RebuildResponse,
    dependencies=[Depends(rate_limit())],
)
async def index_rebuild(
    actor: AdminDep,
    container: ContainerDep,
) -> RebuildResponse:
    """Regenerate embeddings from the current employee photo store.

    Runs in the background; poll /index/status to see progress. The FAISS index is
    swapped atomically when the rebuild finishes, so live recognition is never served
    from a half-built index.
    """
    if not container.start_rebuild():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an index rebuild is already in progress",
        )

    with db_session() as session:
        await run_db(
            container.audit_repo.add_entry,
            session,
            actor=actor,
            action="index.rebuild",
            resource="face_index",
        )

    return RebuildResponse(message="index rebuild started in background")


@router.get("/index/build-info", response_model=MessageResponse)
async def build_info(container: ContainerDep) -> MessageResponse:
    """Documentation convenience: where employee photos are read from."""
    return MessageResponse(
        message="employee photo sources",
        detail=", ".join(str(p) for p in container.settings.employee_photos_source),
    )
