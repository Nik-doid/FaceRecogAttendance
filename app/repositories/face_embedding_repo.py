"""Persistence for the face embedding cache."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.face_embedding import FaceEmbedding
from app.repositories.base import BaseRepository


class FaceEmbeddingRepository(BaseRepository[FaceEmbedding]):
    def __init__(self) -> None:
        super().__init__(FaceEmbedding)

    def replace_all(self, session: Session, rows: list[FaceEmbedding]) -> None:
        """Drop every cached embedding and insert a fresh batch (used on rebuild)."""
        session.query(FaceEmbedding).delete()
        if rows:
            session.add_all(rows)
        session.commit()

    def list_all_rows(self, session: Session) -> list[FaceEmbedding]:
        return list(session.scalars(select(FaceEmbedding)).all())
