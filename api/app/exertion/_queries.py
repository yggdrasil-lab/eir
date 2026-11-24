"""Database queries for exertion analysis."""

from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models import HealthSample


def fetch_exertion_samples(
    db: Session,
    start_dt: datetime,
    end_dt: datetime,
) -> list[HealthSample]:
    """Fetches active energy, heart rate, and step intervals overlapping [start_dt, end_dt]."""
    return list(
        db.scalars(
            select(HealthSample)
            .where(
                HealthSample.deleted_at.is_(None),
                HealthSample.sample_type.in_(["active_energy", "heart_rate", "step_count"]),
                HealthSample.measured_at <= end_dt,
                or_(
                    HealthSample.measured_at >= start_dt,
                    and_(
                        HealthSample.sample_type == "step_count",
                        HealthSample.ended_at.is_not(None),
                        HealthSample.ended_at > start_dt,
                    ),
                ),
            )
            .order_by(HealthSample.measured_at.asc())
        )
    )


def fetch_latest_weight_kg(db: Session, end_dt: datetime) -> float:
    """Returns the latest recorded body weight in kg on or before end_dt, defaulting to 85.0."""
    latest = db.scalar(
        select(HealthSample.value)
        .where(
            HealthSample.deleted_at.is_(None),
            HealthSample.sample_type.in_(["weight", "weight_kg"]),
            HealthSample.measured_at <= end_dt,
        )
        .order_by(HealthSample.measured_at.desc())
        .limit(1)
    )
    return float(latest) if latest else 85.0
