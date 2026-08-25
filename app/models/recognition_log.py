"""Every recognition event observed by the pipeline, for debugging and audit."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    employee_code: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # Whether the attendance event was actually sent to the attendance system.
    reported: Mapped[bool] = mapped_column(Boolean, default=False)
    # Free-form response/ack from the attendance system (e.g. MQ delivery confirm).
    attendance_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Set once this row has been successfully written to the ERP attendance DB,
    # so the cron sync is idempotent and never duplicates a check-in/out.
    erp_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    # Set when the sync deliberately skipped this row (e.g. a duplicate in/out for the
    # day, an unmapped camera, or no employee mapping in the ERP). Rows with a reason
    # are excluded from pending sync, so they are not retried forever.
    erp_skip_reason: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
