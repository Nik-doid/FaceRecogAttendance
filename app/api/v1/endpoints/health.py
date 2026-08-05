"""Health and liveness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app import __version__
from app.api.deps import ContainerDep, db_session, run_db
from app.config.settings import settings
from app.core.logging import get_logger
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])
log = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    database = "up"
    try:
        with db_session() as session:
            await run_db(session.execute, text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        log.error("health check DB failure", extra={"error": str(exc)})
        database = "down"

    camera_state = container.camera_service.state
    camera = "running" if camera_state.running else "stopped"

    return HealthResponse(
        status="ok" if database == "up" else "degraded",
        service=settings.app_name,
        version=__version__,
        database=database,
        camera=camera,
        index_size=container.face_index.size,
    )
