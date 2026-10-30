"""Initial biometric_measurement table for Eir health microservice.

Revision ID: 0001_initial_biometrics
Revises:
Create Date: 2026-10-30
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_biometrics"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "biometric_measurement",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("measured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("weight_kg", sa.Numeric(5, 2), nullable=True),
        sa.Column("body_fat_pct", sa.Numeric(4, 2), nullable=True),
        sa.Column("muscle_mass_kg", sa.Numeric(5, 2), nullable=True),
        sa.Column("water_pct", sa.Numeric(4, 2), nullable=True),
        sa.Column("bone_mass_kg", sa.Numeric(4, 2), nullable=True),
        sa.Column("resting_hr_bpm", sa.Integer, nullable=True),
        sa.Column("hrv_ms", sa.Numeric(5, 2), nullable=True),
        sa.Column("sleep_duration_s", sa.Integer, nullable=True),
        sa.Column("sleep_deep_s", sa.Integer, nullable=True),
        sa.Column("sleep_rem_s", sa.Integer, nullable=True),
        sa.Column("sleep_score", sa.Integer, nullable=True),
        sa.Column("step_count", sa.Integer, nullable=True),
        sa.Column("active_energy_kcal", sa.Integer, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index(
        "biometric_measured_at_idx",
        "biometric_measurement",
        [sa.text("measured_at DESC")],
        postgresql_where=sa.text("deleted_at is null"),
    )
    op.create_index(
        "biometric_source_idx",
        "biometric_measurement",
        ["source"],
        postgresql_where=sa.text("deleted_at is null"),
    )


def downgrade() -> None:
    op.drop_index("biometric_source_idx", table_name="biometric_measurement")
    op.drop_index("biometric_measured_at_idx", table_name="biometric_measurement")
    op.drop_table("biometric_measurement")
