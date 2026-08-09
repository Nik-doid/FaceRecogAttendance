"""add erp_skip_reason to recognition_logs

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-09
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "recognition_logs",
        sa.Column("erp_skip_reason", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_recognition_logs_erp_skip_reason",
        "recognition_logs",
        ["erp_skip_reason"],
    )


def downgrade() -> None:
    op.drop_index("ix_recognition_logs_erp_skip_reason", table_name="recognition_logs")
    op.drop_column("recognition_logs", "erp_skip_reason")
