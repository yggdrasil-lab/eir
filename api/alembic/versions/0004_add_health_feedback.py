"""Add health_feedback table for subjective and behavioral telemetry.

Revision ID: 0004_add_health_feedback
Revises: 0003_add_oauth_token
Create Date: 2026-11-01
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004_add_health_feedback"
down_revision: str | None = "0003_add_oauth_token"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "health_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("feedback_type", sa.Text, nullable=False),
        sa.Column("rating", sa.Text, nullable=True),
        sa.Column("score", sa.Numeric(4, 2), nullable=True),
        sa.Column("tags", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("payload", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("logged_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index(
        "health_feedback_date_idx",
        "health_feedback",
        ["date"],
        postgresql_where=sa.text("deleted_at is null"),
    )
    op.create_index(
        "health_feedback_type_date_idx",
        "health_feedback",
        ["feedback_type", "date"],
        postgresql_where=sa.text("deleted_at is null"),
    )
    op.create_index(
        "health_feedback_dedup_idx",
        "health_feedback",
        ["date", "feedback_type"],
        unique=True,
        postgresql_where=sa.text("deleted_at is null"),
    )


def downgrade() -> None:
    op.drop_index("health_feedback_dedup_idx", table_name="health_feedback")
    op.drop_index("health_feedback_type_date_idx", table_name="health_feedback")
    op.drop_index("health_feedback_date_idx", table_name="health_feedback")
    op.drop_table("health_feedback")
