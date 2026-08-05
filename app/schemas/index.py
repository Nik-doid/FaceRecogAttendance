"""Face index control schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class IndexStatusResponse(BaseModel):
    status: Literal["idle", "building"]
    size: int
    employees: int
    last_built_at: datetime | None
    last_error: str | None


class RebuildResponse(BaseModel):
    message: str
    detail: str | None = None
