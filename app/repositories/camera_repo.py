"""Persistence for camera runtime state."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.repositories.base import BaseRepository


class CameraRepository(BaseRepository[Camera]):
    def __init__(self) -> None:
        super().__init__(Camera)

    def get_by_id(self, session: Session, camera_id: str) -> Camera | None:
        return session.scalar(select(Camera).where(Camera.camera_id == camera_id))

    def get_or_create(
        self,
        session: Session,
        camera_id: str,
        name: str | None = None,
        rtsp_url: str | None = None,
    ) -> Camera:
        cam = self.get_by_id(session, camera_id)
        if cam is None:
            cam = Camera(camera_id=camera_id, name=name, rtsp_url=rtsp_url)
            session.add(cam)
            session.commit()
        return cam

    def set_status(
        self,
        session: Session,
        camera_id: str,
        status: str,
        *,
        last_connected_at: datetime | None = None,
        last_error: str | None = None,
    ) -> Camera:
        cam = self.get_or_create(session, camera_id)
        cam.status = status
        if last_connected_at is not None:
            cam.last_connected_at = last_connected_at
        if last_error is not None:
            cam.last_error = last_error
        session.commit()
        return cam
