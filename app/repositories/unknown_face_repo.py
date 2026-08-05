"""Persistence for unknown-face snapshots."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.unknown_face import UnknownFace
from app.repositories.base import BaseRepository


class UnknownFaceRepository(BaseRepository[UnknownFace]):
    def __init__(self) -> None:
        super().__init__(UnknownFace)

    def add_entry(
        self,
        session: Session,
        *,
        camera_id: str,
        timestamp: datetime,
        snapshot_path: str,
        confidence_of_best_nonmatch: float,
        track_id: int | None = None,
    ) -> UnknownFace:
        entry = UnknownFace(
            camera_id=camera_id,
            timestamp=timestamp,
            snapshot_path=snapshot_path,
            confidence_of_best_nonmatch=confidence_of_best_nonmatch,
            track_id=track_id,
        )
        session.add(entry)
        session.commit()
        return entry

    def list_recent(self, session: Session, limit: int = 50) -> list[UnknownFace]:
        stmt = select(UnknownFace).order_by(UnknownFace.timestamp.desc()).limit(limit)
        return list(session.scalars(stmt).all())
