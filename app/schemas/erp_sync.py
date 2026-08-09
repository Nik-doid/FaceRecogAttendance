"""ERP attendance-log sync schemas."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import MessageResponse


class ErpSyncStatsOut(BaseModel):
    scanned: int = 0
    inserted: int = 0
    failed: int = 0
    skipped_unmapped_camera: int = 0


class ErpSyncResponse(BaseModel):
    ok: bool
    stats: ErpSyncStatsOut
    detail: str = ""


class ErpSyncStatusResponse(MessageResponse):
    enabled: bool = False
    pending: int = 0
    interval_seconds: int | None = None
