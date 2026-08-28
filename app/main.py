"""FastAPI application factory and lifecycle.

Startup order matters:
1. Configure structured logging.
2. Build the DI container (loads ONNX models once — fail fast on bad config).
3. Seed this service's own DB rows (settings + camera row).
4. Build the face index in the background (never blocks the API from accepting
   /health etc.).
5. Start the camera worker only if CAMERA_AUTOSTART=true (controlled via the API).

Shutdown is the reverse: stop the worker, close the attendance reporter's MQ
connection, dispose DB pools.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import __version__
from app.api.deps import set_container
from app.api.router import api_router
from app.config.settings import settings
from app.container import Container
from app.core.logging import get_logger, setup_logging
from app.database.session import close_engines, sync_session

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def _seed_own_database(container: Container) -> None:
    """Ensure settings defaults and the camera row exist before the worker starts."""
    with sync_session() as session:
        container.settings_service.seed_defaults(session, settings)
        container.camera_repo.get_or_create(
            session,
            settings.camera_id,
            name=settings.camera_id,
            rtsp_url=settings.rtsp_url,
        )
        log.info("own database seeded")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(settings.log_level, quiet_native=settings.quiet_native_logs)
    container = Container(load_models=True)
    set_container(container)
    _seed_own_database(container)

    # Enrol employee photos in the background so the API is up in seconds rather
    # than minutes. Recognition switches on when it finishes; until then the
    # pipeline still detects, gates on gaze and scans for palms.
    container.start_gallery_build()

    if settings.camera_autostart:
        # The legacy worker keeps its own DB-backed index, so it only pays for a
        # rebuild when it is actually going to run.
        container.start_rebuild()
        container.camera_service.start()

    container.start_erp_sync()

    log.info(
        "service started",
        extra={"camera_id": settings.camera_id, "env": settings.app_env},
    )
    yield
    container.shutdown()
    await close_engines()
    log.info("service stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Face recognition service for a single RTSP camera. Reports attendance "
            "events to the existing attendance system via a message queue."
        ),
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """The debug webcam page (talks to /api/v1/webcam/ws)."""
        return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
