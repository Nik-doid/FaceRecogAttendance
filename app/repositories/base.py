"""Generic sync repository base.

Design decision: repositories are written once against the synchronous SQLAlchemy
``Session`` and shared by BOTH sides of the service:
- the background recognition worker (sync code) calls them directly;
- the async FastAPI endpoints call them via ``asyncio.to_thread`` so the event loop
  never blocks.

This keeps a single data-access implementation per entity (DRY) while still giving
async request handlers. DB calls here are short; they are not the bottleneck.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.base import Base


class BaseRepository[ModelT: Base]:
    def __init__(self, model: type[ModelT]) -> None:
        self._model = model

    def get(self, session: Session, obj_id: int) -> ModelT | None:
        return session.get(self._model, obj_id)

    def add(self, session: Session, obj: ModelT) -> ModelT:
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj

    def list_all(self, session: Session) -> list[ModelT]:
        return list(session.scalars(select(self._model)).all())

    def count(self, session: Session) -> int:
        return int(session.scalar(select(func.count()).select_from(self._model)) or 0)
