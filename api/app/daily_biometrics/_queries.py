"""Soft-delete-safe SQLAlchemy queries for daily biometrics reads and rollups."""

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import BiometricMeasurement, HealthFeedback, HealthSample


def fetch_day_measurements(
    db: Session,
    start_dt: datetime,
    end_dt: datetime,
) -> list[BiometricMeasurement]:
    """Returns active BiometricMeasurement rows within [start_dt, end_dt] ordered ASC."""
    stmt = (
        select(BiometricMeasurement)
        .where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.measured_at >= start_dt,
            BiometricMeasurement.measured_at <= end_dt,
        )
        .order_by(BiometricMeasurement.measured_at.asc())
    )
    return list(db.scalars(stmt))


def fetch_latest_day_measurement(
    db: Session,
    start_dt: datetime,
    end_dt: datetime,
) -> BiometricMeasurement | None:
    """Returns the most recent active BiometricMeasurement within [start_dt, end_dt]."""
    stmt = (
        select(BiometricMeasurement)
        .where(
            BiometricMeasurement.deleted_at.is_(None),
            BiometricMeasurement.measured_at >= start_dt,
            BiometricMeasurement.measured_at <= end_dt,
        )
        .order_by(BiometricMeasurement.measured_at.desc())
        .limit(1)
    )
    return db.scalar(stmt)


def fetch_latest_measurement(
    db: Session,
    *,
    source: str | None = None,
) -> BiometricMeasurement | None:
    """Returns the most recent active BiometricMeasurement, optionally filtered by source."""
    stmt = select(BiometricMeasurement).where(BiometricMeasurement.deleted_at.is_(None))
    if source:
        stmt = stmt.where(BiometricMeasurement.source == source)
    stmt = stmt.order_by(BiometricMeasurement.measured_at.desc()).limit(1)
    return db.scalar(stmt)


def fetch_granular_samples(
    db: Session,
    sample_types: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
    *,
    limit: int | None = None,
) -> list[HealthSample]:
    """Returns active HealthSample rows matching sample_types in [start_dt, end_dt] ASC."""
    stmt = (
        select(HealthSample)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(list(sample_types)),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.asc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt))


def fetch_latest_sample_value(
    db: Session,
    sample_types: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
) -> Decimal | None:
    """Returns the latest HealthSample value matching sample_types within [start_dt, end_dt]."""
    stmt = (
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(list(sample_types)),
            HealthSample.measured_at >= start_dt,
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )
    return db.scalar(stmt)


def fetch_latest_sample_value_on_or_before(
    db: Session,
    sample_types: Sequence[str],
    end_dt: datetime,
) -> Decimal | None:
    """Returns the latest HealthSample value matching sample_types on or before end_dt."""
    stmt = (
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(list(sample_types)),
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )
    return db.scalar(stmt)


def fetch_avg_sample_value(
    db: Session,
    sample_types: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
) -> float | None:
    """Returns the arithmetic mean of HealthSample values in [start_dt, end_dt]."""
    stmt = select(func.avg(HealthSample.value)).where(
        HealthSample.deleted_at.is_(None),
        HealthSample.sample_type.in_(list(sample_types)),
        HealthSample.measured_at >= start_dt,
        HealthSample.measured_at <= end_dt,
    )
    return db.scalar(stmt)


def fetch_sum_sample_value(
    db: Session,
    sample_types: Sequence[str],
    start_dt: datetime,
    end_dt: datetime,
) -> Decimal | None:
    """Returns the sum of HealthSample values in [start_dt, end_dt]."""
    stmt = select(func.sum(HealthSample.value)).where(
        HealthSample.deleted_at.is_(None),
        HealthSample.sample_type.in_(list(sample_types)),
        HealthSample.measured_at >= start_dt,
        HealthSample.measured_at <= end_dt,
    )
    return db.scalar(stmt)


def fetch_day_feedback(db: Session, target_date: date) -> list[HealthFeedback]:
    """Returns active HealthFeedback rows for target_date."""
    stmt = select(HealthFeedback).where(
        HealthFeedback.deleted_at.is_(None),
        HealthFeedback.date == target_date,
    )
    return list(db.scalars(stmt))
