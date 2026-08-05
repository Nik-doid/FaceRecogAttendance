"""Persistence for runtime settings stored in the DB."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.setting import Setting
from app.repositories.base import BaseRepository


class SettingRepository(BaseRepository[Setting]):
    def __init__(self) -> None:
        super().__init__(Setting)

    def get_value(self, session: Session, key: str) -> str | None:
        row = session.get(Setting, key)
        return row.value if row else None

    def get_many(self, session: Session, keys: list[str]) -> dict[str, str]:
        stmt = select(Setting).where(Setting.key.in_(keys))
        return {row.key: row.value for row in session.scalars(stmt).all()}

    def set(self, session: Session, key: str, value: str) -> Setting:
        row = session.get(Setting, key)
        if row is None:
            row = Setting(key=key, value=value)
            session.add(row)
        else:
            row.value = value
        session.commit()
        return row

    def seed(self, session: Session, values: dict[str, str]) -> None:
        """Insert defaults for any key not already present (startup only)."""
        existing = self.get_many(session, list(values.keys()))
        for key, value in values.items():
            if key not in existing:
                session.add(Setting(key=key, value=value))
        session.commit()
