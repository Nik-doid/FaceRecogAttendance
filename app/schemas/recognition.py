"""Recognition / unknown-face read schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecognitionLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    employee_code: str | None
    camera_id: str
    timestamp: datetime
    confidence: float
    reported: bool
    attendance_response: str | None
    snapshot_path: str | None
    track_id: int | None


class UnknownFaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    timestamp: datetime
    snapshot_path: str
    confidence_of_best_nonmatch: float
    track_id: int | None
