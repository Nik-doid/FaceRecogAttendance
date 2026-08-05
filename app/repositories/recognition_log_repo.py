"""Persistence for recognition events (audit/debug)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.recognition_log import RecognitionLog
from app.repositories.base import BaseRepository


class RecognitionLogRepository(BaseRepository[RecognitionLog]):
    def __init__(self) -> None:
        super().__init__(RecognitionLog)

    def add_entry(
        self,
        session: Session,
        *,
        employee_code: str | None,
        camera_id: str,
        timestamp: datetime,
        confidence: float,
        reported: bool,
        attendance_response: str | None = None,
        snapshot_path: str | None = None,
        track_id: int | None = None,
    ) -> RecognitionLog:
        log = RecognitionLog(
            employee_code=employee_code,
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=confidence,
            reported=reported,
            attendance_response=attendance_response,
            snapshot_path=snapshot_path,
            track_id=track_id,
        )
        session.add(log)
        session.commit()
        return log

    def list_recent(
        self, session: Session, limit: int = 50, employee_code: str | None = None
    ) -> list[RecognitionLog]:
        stmt = select(RecognitionLog).order_by(RecognitionLog.timestamp.desc()).limit(limit)
        if employee_code:
            stmt = stmt.where(RecognitionLog.employee_code == employee_code)
        return list(session.scalars(stmt).all())

    def count_filtered(self, session: Session, employee_code: str | None = None) -> int:
        stmt = select(func.count()).select_from(RecognitionLog)
        if employee_code:
            stmt = stmt.where(RecognitionLog.employee_code == employee_code)
        return int(session.scalar(stmt) or 0)
