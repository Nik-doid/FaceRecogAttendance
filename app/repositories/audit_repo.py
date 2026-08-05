"""Persistence for the audit trail of privileged actions."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog
from app.repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    def __init__(self) -> None:
        super().__init__(AuditLog)

    def add_entry(
        self,
        session: Session,
        *,
        actor: str,
        action: str,
        resource: str,
        detail: str | None = None,
    ) -> AuditLog:
        entry = AuditLog(actor=actor, action=action, resource=resource, detail=detail)
        session.add(entry)
        session.commit()
        return entry

    def list_recent(self, session: Session, limit: int = 100) -> list[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        return list(session.scalars(stmt).all())
