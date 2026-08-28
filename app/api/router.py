"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import (
    camera,
    erp_sync,
    health,
    index,
    metrics,
    recognition,
    websocket,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(metrics.router)
api_router.include_router(index.router)
api_router.include_router(camera.router)
api_router.include_router(recognition.router)
api_router.include_router(erp_sync.router)
api_router.include_router(websocket.router)
