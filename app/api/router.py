"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import camera, gallery, health, metrics, websocket

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(metrics.router)
api_router.include_router(gallery.router)
api_router.include_router(camera.router)
api_router.include_router(websocket.router)
