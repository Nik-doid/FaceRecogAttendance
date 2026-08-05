"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-04
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "face_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_code", sa.String(64), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("source_image_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_face_embeddings_employee_code", "face_embeddings", ["employee_code"])

    op.create_table(
        "recognition_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_code", sa.String(64), nullable=True),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reported", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("attendance_response", sa.Text(), nullable=True),
        sa.Column("snapshot_path", sa.Text(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_recognition_logs_employee_code", "recognition_logs", ["employee_code"])
    op.create_index("ix_recognition_logs_camera_id", "recognition_logs", ["camera_id"])
    op.create_index("ix_recognition_logs_timestamp", "recognition_logs", ["timestamp"])

    op.create_table(
        "unknown_faces",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("snapshot_path", sa.Text(), nullable=False),
        sa.Column("confidence_of_best_nonmatch", sa.Float(), nullable=False, server_default="0"),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_unknown_faces_camera_id", "unknown_faces", ["camera_id"])
    op.create_index("ix_unknown_faces_timestamp", "unknown_faces", ["timestamp"])

    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=True),
        sa.Column("rtsp_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="stopped"),
        sa.Column("last_connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_cameras_camera_id", "cameras", ["camera_id"], unique=True)

    op.create_table(
        "settings",
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("resource", sa.String(128), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_audit_logs_actor", "audit_logs", ["actor"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade() -> None:
    for table in (
        "audit_logs",
        "settings",
        "cameras",
        "unknown_faces",
        "recognition_logs",
        "face_embeddings",
    ):
        op.drop_table(table)
