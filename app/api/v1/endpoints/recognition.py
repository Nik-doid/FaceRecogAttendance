"""Read endpoints for recognition logs and unknown faces (debug/audit)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep, db_session, run_db
from app.schemas.common import Page
from app.schemas.recognition import RecognitionLogOut, UnknownFaceOut

router = APIRouter(tags=["recognition"])


@router.get("/recognition/logs", response_model=Page[RecognitionLogOut])
async def recognition_logs(
    container: ContainerDep,
    limit: int = Query(default=50, ge=1, le=500),
    employee_code: str | None = Query(default=None),
) -> Page[RecognitionLogOut]:
    with db_session() as session:
        items = await run_db(
            container.recognition_log_repo.list_recent,
            session,
            limit,
            employee_code,
        )
        total = await run_db(
            container.recognition_log_repo.count_filtered, session, employee_code
        )
    return Page(
        items=[RecognitionLogOut.model_validate(i) for i in items],
        total=total,
    )


@router.get("/unknown-faces", response_model=Page[UnknownFaceOut])
async def unknown_faces(
    container: ContainerDep,
    limit: int = Query(default=50, ge=1, le=500),
) -> Page[UnknownFaceOut]:
    with db_session() as session:
        items = await run_db(container.unknown_face_repo.list_recent, session, limit)
        total = await run_db(container.unknown_face_repo.count, session)
    return Page(
        items=[UnknownFaceOut.model_validate(i) for i in items],
        total=total,
    )
