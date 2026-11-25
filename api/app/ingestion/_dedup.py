"""Idempotent time-series upsert and source-priority scale deduplication."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingestion._types import NormalizedSample
from app.models import HealthSample

SCALE_DEDUP_METRICS: frozenset[str] = frozenset(
    {"weight", "body_fat_pct", "muscle_mass_kg", "water_pct", "bone_mass_kg"}
)
SCALE_SOURCE_PRIORITY: dict[str, int] = {
    "withings": 2,
    "apple_health": 1,
}


def normalize_utc_second(dt: datetime) -> datetime:
    """Normalizes naive (SQLite) and tz-aware (Postgres) datetimes to UTC second precision."""
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).replace(microsecond=0)


def _exact_key(source: str, sample_type: str, measured_at: datetime, ended_at: datetime) -> tuple[str, str, datetime, datetime]:
    return (
        source,
        sample_type,
        normalize_utc_second(measured_at),
        normalize_utc_second(ended_at),
    )


def _fetch_existing_samples(
    db: Session,
    samples: Sequence[NormalizedSample],
    dedup_window_s: int,
) -> list[HealthSample]:
    min_utc = min(normalize_utc_second(s.measured_at) for s in samples)
    max_utc = max(normalize_utc_second(s.ended_at or s.measured_at) for s in samples)
    dialect_name = db.bind.dialect.name if db.bind is not None else ""
    pad = timedelta(hours=14) if dialect_name == "sqlite" else timedelta(seconds=dedup_window_s)
    stmt = select(HealthSample).where(
        HealthSample.deleted_at.is_(None),
        HealthSample.measured_at >= min_utc - pad,
        HealthSample.measured_at <= max_utc + pad,
    )
    return list(db.scalars(stmt))


def _find_scale_match(
    existing_samples: Sequence[HealthSample],
    sample_type: str,
    measured_utc: datetime,
    dedup_window_s: int,
) -> HealthSample | None:
    best_match: HealthSample | None = None
    best_rank: tuple[int, float] | None = None
    for candidate in existing_samples:
        if candidate.sample_type != sample_type:
            continue
        diff_s = abs((normalize_utc_second(candidate.measured_at) - measured_utc).total_seconds())
        if diff_s > dedup_window_s:
            continue
        rank = (SCALE_SOURCE_PRIORITY.get(candidate.source, 0), -diff_s)
        if best_rank is None or rank > best_rank:
            best_match = candidate
            best_rank = rank
    return best_match


def _update_existing_sample(
    target: HealthSample,
    sample: NormalizedSample,
    existing_map: dict[tuple[str, str, datetime, datetime], HealthSample],
    *,
    update_timestamps: bool,
    promote_source: bool,
) -> None:
    old_key = _exact_key(target.source, target.sample_type, target.measured_at, target.ended_at)
    if existing_map.get(old_key) is target:
        existing_map.pop(old_key, None)

    if promote_source:
        target.source = sample.source
    target.value = sample.value
    if sample.unit is not None:
        target.unit = sample.unit
    if update_timestamps:
        target.measured_at = sample.measured_at
        target.ended_at = sample.ended_at or sample.measured_at
    if sample.external_id is not None:
        target.external_id = sample.external_id
    if sample.extra:
        target.extra = {**(target.extra or {}), **dict(sample.extra)}

    new_key = _exact_key(target.source, target.sample_type, target.measured_at, target.ended_at)
    existing_map[new_key] = target


def _handle_scale_dedup(
    matched: HealthSample,
    sample: NormalizedSample,
    existing_map: dict[tuple[str, str, datetime, datetime], HealthSample],
) -> None:
    incoming_prio = SCALE_SOURCE_PRIORITY.get(sample.source, 0)
    existing_prio = SCALE_SOURCE_PRIORITY.get(matched.source, 0)
    if incoming_prio < existing_prio:
        return
    _update_existing_sample(
        matched,
        sample,
        existing_map,
        update_timestamps=True,
        promote_source=incoming_prio > existing_prio,
    )


def _upsert_single_sample(
    db: Session,
    sample: NormalizedSample,
    existing_samples: list[HealthSample],
    existing_map: dict[tuple[str, str, datetime, datetime], HealthSample],
    dedup_window_s: int,
) -> int:
    ended_at = sample.ended_at or sample.measured_at
    key = _exact_key(sample.source, sample.sample_type, sample.measured_at, ended_at)

    exact = existing_map.get(key)
    if exact is not None:
        _update_existing_sample(exact, sample, existing_map, update_timestamps=False, promote_source=False)
        return 0

    if sample.sample_type in SCALE_DEDUP_METRICS:
        matched = _find_scale_match(existing_samples, sample.sample_type, key[2], dedup_window_s)
        if matched is not None:
            _handle_scale_dedup(matched, sample, existing_map)
            return 0

    row = HealthSample(
        id=uuid.uuid4(),
        source=sample.source,
        sample_type=sample.sample_type,
        measured_at=sample.measured_at,
        ended_at=ended_at,
        value=sample.value,
        unit=sample.unit,
        external_id=sample.external_id,
        extra=dict(sample.extra),
    )
    db.add(row)
    existing_map[key] = row
    existing_samples.append(row)
    return 1


def upsert_health_samples(
    db: Session,
    samples: Sequence[NormalizedSample],
    *,
    dedup_window_s: int = 60,
) -> tuple[dict[str, int], int]:
    """Batch-upserts NormalizedSample rows with +-dedup_window_s Withings-wins scale dedup."""
    if not samples:
        return {}, 0

    existing_samples = _fetch_existing_samples(db, samples, dedup_window_s)
    existing_map = {
        _exact_key(row.source, row.sample_type, row.measured_at, row.ended_at): row
        for row in existing_samples
    }
    metrics_counts: dict[str, int] = {}
    created_count = 0

    for sample in samples:
        metrics_counts[sample.sample_type] = metrics_counts.get(sample.sample_type, 0) + 1
        created_count += _upsert_single_sample(
            db, sample, existing_samples, existing_map, dedup_window_s
        )

    return metrics_counts, created_count
