"""Unknown (unmatched) faces: snapshot saved, never reported to attendance."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class UnknownFace(Base):
    __tablename__ = "unknown_faces"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    snapshot_path: Mapped[str] = mapped_column(Text)
    confidence_of_best_nonmatch: Mapped[float] = mapped_column(Float, default=0.0)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
