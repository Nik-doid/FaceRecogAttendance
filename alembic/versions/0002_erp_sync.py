"""add erp_synced_at to recognition_logs

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-07
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "recognition_logs",
        sa.Column("erp_synced_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_recognition_logs_erp_synced_at",
        "recognition_logs",
        ["erp_synced_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_logs_erp_synced_at", table_name="recognition_logs")
    op.drop_column("recognition_logs", "erp_synced_at")
