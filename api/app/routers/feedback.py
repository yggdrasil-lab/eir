"""Feedback router for active subjective and behavioral telemetry."""

from datetime import date
from typing import Any
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import HealthFeedback
from app.schemas import (
    FastingFeedbackCreate,
    HealthFeedbackCreate,
    HealthFeedbackRead,
    WellnessFeedbackCreate,
    WorkoutFeedbackCreate,
)

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _upsert_feedback(
    db: Session,
    target_date: date,
    feedback_type: str,
    rating: str | None,
    score: Any | None,
    tags: list[str],
    payload: dict[str, Any],
    notes: str | None,
    logged_at: Any,
) -> HealthFeedback:
    """Helper to upsert a health feedback record for (date, feedback_type)."""
    stmt = (
        select(HealthFeedback)
        .where(
            HealthFeedback.deleted_at.is_(None),
            HealthFeedback.date == target_date,
            HealthFeedback.feedback_type == feedback_type,
        )
        .limit(1)
    )
    existing = db.scalar(stmt)

    if existing:
        existing.rating = rating
        existing.score = score
        existing.tags = tags
        existing.payload = payload
        existing.notes = notes
        existing.logged_at = logged_at
        db.commit()
        db.refresh(existing)
        return existing

    record = HealthFeedback(
        id=uuid.uuid4(),
        date=target_date,
        feedback_type=feedback_type,
        rating=rating,
        score=score,
        tags=tags,
        payload=payload,
        notes=notes,
        logged_at=logged_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("", response_model=HealthFeedbackRead, status_code=status.HTTP_200_OK)
def save_feedback(
    body: HealthFeedbackCreate,
    db: Session = Depends(get_db),
) -> HealthFeedback:
    """Generic endpoint to save or update any health feedback record."""
    return _upsert_feedback(
        db=db,
        target_date=body.date,
        feedback_type=body.feedback_type,
        rating=body.rating,
        score=body.score,
        tags=body.tags,
        payload=body.payload,
        notes=body.notes,
        logged_at=body.logged_at,
    )


@router.post("/fasting", response_model=HealthFeedbackRead, status_code=status.HTTP_200_OK)
def save_fasting_feedback(
    body: FastingFeedbackCreate,
    db: Session = Depends(get_db),
) -> HealthFeedback:
    """Convenience endpoint for daily fasting protocol check-ins."""
    energy_score_map = {"high": 5.0, "normal": 3.0, "low": 1.0}
    score = energy_score_map.get(body.energy.lower())

    payload = {
        "target_window": body.target_window,
        "energy": body.energy,
        "headache": body.headache,
    }
    tags = [f"energy:{body.energy}"]
    if body.headache and body.headache != "none":
        tags.append(f"symptom:{body.headache}")

    return _upsert_feedback(
        db=db,
        target_date=body.date,
        feedback_type="fasting",
        rating=body.adherence,
        score=score,
        tags=tags,
        payload=payload,
        notes=body.notes,
        logged_at=body.logged_at,
    )


@router.post("/workout", response_model=HealthFeedbackRead, status_code=status.HTTP_200_OK)
def save_workout_feedback(
    body: WorkoutFeedbackCreate,
    db: Session = Depends(get_db),
) -> HealthFeedback:
    """Convenience endpoint for post-workout Session RPE effort debriefs."""
    effort_score_map = {
        "too_easy": 5.0,
        "optimal": 8.0,
        "hard": 9.5,
        "too_hard": 10.0,
    }
    score = effort_score_map.get(body.effort_rating.lower())

    payload = {
        "routine_name": body.routine_name,
        "duration_minutes": body.duration_minutes,
        "total_volume_kg": float(body.total_volume_kg) if body.total_volume_kg is not None else None,
        "completed_sets": body.completed_sets,
    }

    return _upsert_feedback(
        db=db,
        target_date=body.date,
        feedback_type="workout",
        rating=body.effort_rating,
        score=score,
        tags=body.tags,
        payload=payload,
        notes=body.notes,
        logged_at=body.logged_at,
    )


@router.post("/wellness", response_model=HealthFeedbackRead, status_code=status.HTTP_200_OK)
def save_wellness_feedback(
    body: WellnessFeedbackCreate,
    db: Session = Depends(get_db),
) -> HealthFeedback:
    """Convenience endpoint for subjective morning readiness and muscle soreness."""
    payload = {
        "stress_level": body.stress_level,
    }

    return _upsert_feedback(
        db=db,
        target_date=body.date,
        feedback_type="wellness",
        rating=body.readiness_rating,
        score=body.soreness_score,
        tags=body.sore_areas,
        payload=payload,
        notes=body.notes,
        logged_at=body.logged_at,
    )


@router.get("/daily/{target_date}", response_model=list[HealthFeedbackRead])
def get_daily_feedbacks(
    target_date: date,
    db: Session = Depends(get_db),
) -> list[HealthFeedback]:
    """Returns all active health feedback logged for a specific date."""
    stmt = (
        select(HealthFeedback)
        .where(
            HealthFeedback.deleted_at.is_(None),
            HealthFeedback.date == target_date,
        )
        .order_by(HealthFeedback.logged_at.asc())
    )
    return list(db.scalars(stmt))


@router.get("/history", response_model=list[HealthFeedbackRead])
def get_feedback_history(
    feedback_type: str | None = Query(default=None, description="Optional type filter (fasting, workout, wellness)"),
    limit: int = Query(default=30, ge=1, le=365),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[HealthFeedback]:
    """Returns historical feedback entries ordered newest first."""
    stmt = select(HealthFeedback).where(HealthFeedback.deleted_at.is_(None))
    if feedback_type:
        stmt = stmt.where(HealthFeedback.feedback_type == feedback_type)
    stmt = stmt.order_by(HealthFeedback.date.desc(), HealthFeedback.logged_at.desc()).offset(offset).limit(limit)
    return list(db.scalars(stmt))
