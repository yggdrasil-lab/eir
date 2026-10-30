"""Add health_sample table for granular time-series telemetry.

Revision ID: 0002_add_health_sample
Revises: 0001_initial_biometrics
Create Date: 2026-10-30
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_add_health_sample"
down_revision: str | None = "0001_initial_biometrics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "health_sample",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("sample_type", sa.Text, nullable=False),
        sa.Column("measured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Numeric(10, 4), nullable=False),
        sa.Column("unit", sa.Text, nullable=True),
        sa.Column("external_id", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index(
        "health_sample_measured_at_idx",
        "health_sample",
        [sa.text("measured_at DESC")],
        postgresql_where=sa.text("deleted_at is null"),
    )
    op.create_index(
        "health_sample_type_idx",
        "health_sample",
        ["sample_type", sa.text("measured_at DESC")],
        postgresql_where=sa.text("deleted_at is null"),
    )
    op.create_index(
        "health_sample_dedup_idx",
        "health_sample",
        ["source", "sample_type", "measured_at", "ended_at"],
        unique=True,
        postgresql_where=sa.text("deleted_at is null"),
    )


def downgrade() -> None:
    op.drop_index("health_sample_dedup_idx", table_name="health_sample")
    op.drop_index("health_sample_type_idx", table_name="health_sample")
    op.drop_index("health_sample_measured_at_idx", table_name="health_sample")
    op.drop_table("health_sample")
