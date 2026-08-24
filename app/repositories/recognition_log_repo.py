"""Persistence for recognition events (audit/debug)."""

from __future__ import annotations

from datetime import date, datetime, time

from sqlalchemy import func, select, update
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

    def list_pending_erp_sync(
        self, session: Session, *, employee_code: str | None = None, limit: int = 500
    ) -> list[RecognitionLog]:
        """Recognition events that were reported but not yet written to the ERP DB.

        Only recognised (employee_code is not null) and successfully reported rows are
        candidates; ``limit`` bounds the batch a single sync run will process so the
        scheduler stays responsive even with a large backlog.
        """
        stmt = (
            select(RecognitionLog)
            .where(
                RecognitionLog.employee_code.is_not(None),
                RecognitionLog.reported.is_(True),
                RecognitionLog.erp_synced_at.is_(None),
                RecognitionLog.erp_skip_reason.is_(None),
            )
            .order_by(RecognitionLog.timestamp.asc())
            .limit(limit)
        )
        if employee_code:
            stmt = stmt.where(RecognitionLog.employee_code == employee_code)
        return list(session.scalars(stmt).all())

    def mark_erp_synced(self, session: Session, log_ids: list[int], synced_at: datetime) -> None:
        """Stamp rows as synced after a successful ERP write (idempotency marker)."""
        if not log_ids:
            return
        session.execute(
            update(RecognitionLog)
            .where(RecognitionLog.id.in_(log_ids))
            .values(erp_synced_at=synced_at)
        )
        session.commit()

    def mark_erp_skipped(self, session: Session, log_ids: list[int], reason: str) -> None:
        """Stamp rows the sync deliberately skipped so they are not retried forever."""
        if not log_ids:
            return
        session.execute(
            update(RecognitionLog)
            .where(RecognitionLog.id.in_(log_ids))
            .values(erp_skip_reason=reason)
        )
        session.commit()

    def count_pending_erp_sync(self, session: Session) -> int:
        stmt = select(func.count()).select_from(RecognitionLog).where(
            RecognitionLog.employee_code.is_not(None),
            RecognitionLog.reported.is_(True),
            RecognitionLog.erp_synced_at.is_(None),
            RecognitionLog.erp_skip_reason.is_(None),
        )
        return int(session.scalar(stmt) or 0)

    def latest_attendance_snapshot(
        self, session: Session, employee_code: str, day: date
    ) -> str | None:
        """The most recent attendance snapshot saved for an employee on a day.

        The recognition loop reuses this instead of writing a second snapshot, so an
        employee has at most one attendance snapshot per day — mirroring how the
        external system never captures more than one enrolment snapshot per punch.
        """
        stmt = (
            select(RecognitionLog)
            .where(
                RecognitionLog.employee_code == employee_code,
                RecognitionLog.snapshot_path.is_not(None),
                RecognitionLog.timestamp >= datetime.combine(day, time.min),
                RecognitionLog.timestamp < datetime.combine(day, time.max),
            )
            .order_by(RecognitionLog.timestamp.desc())
            .limit(1)
        )
        log = session.scalar(stmt)
        return log.snapshot_path if log is not None else None
