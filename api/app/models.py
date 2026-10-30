"""SQLAlchemy models for Eir health and biometric records."""

from datetime import datetime
from decimal import Decimal
import uuid

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    Numeric,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """The standard tail shared across tables."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")


class BiometricMeasurement(TimestampMixin, Base):
    """Point-in-time biometrics from scale (Withings) or wearables (Apple Watch)."""

    __tablename__ = "biometric_measurement"

    id: Mapped[uuid.UUID] = _uuid_pk()
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)  # 'withings', 'apple_health', 'manual'
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    body_fat_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    muscle_mass_kg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    water_pct: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    bone_mass_kg: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    resting_hr_bpm: Mapped[int | None] = mapped_column(Integer)
    hrv_ms: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    sleep_duration_s: Mapped[int | None] = mapped_column(Integer)
    sleep_deep_s: Mapped[int | None] = mapped_column(Integer)
    sleep_rem_s: Mapped[int | None] = mapped_column(Integer)
    sleep_score: Mapped[int | None] = mapped_column(Integer)
    step_count: Mapped[int | None] = mapped_column(Integer)
    active_energy_kcal: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("biometric_measured_at_idx", measured_at.desc(), postgresql_where=text("deleted_at is null")),
        Index("biometric_source_idx", source, postgresql_where=text("deleted_at is null")),
    )
