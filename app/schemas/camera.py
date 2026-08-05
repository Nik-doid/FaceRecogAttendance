"""Camera control schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CameraStatusResponse(BaseModel):
    camera_id: str
    status: str
    last_connected_at: datetime | None
    last_error: str | None


class CameraActionResponse(BaseModel):
    camera_id: str
    action: str
    status: str
