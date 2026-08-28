"""Health and liveness."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import ContainerDep
from app.config.settings import settings
from app.core.logging import get_logger
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])
log = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    """Report what the service can currently do.

    There is no local database to ping any more. What matters instead is whether the
    camera is capturing and whether anyone is enrolled: a running camera with an empty
    gallery detects faces and recognises nobody, which is degraded, not healthy.
    """
    runner = container.camera_runner
    gallery = container.gallery
    ready = container.gallery_handle.ready.is_set()

    return HealthResponse(
        status="ok" if ready else "degraded",
        service=settings.app_name,
        version=__version__,
        database="n/a",
        camera="running" if runner.running else "stopped",
        index_size=gallery.index.size,
    )
