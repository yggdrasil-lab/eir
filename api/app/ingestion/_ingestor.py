"""TelemetryIngestor facade unifying Apple Health sync, Withings samples, and rollups."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app.config import settings
from app.daily_biometrics import DailyBiometrics, resolve_sleep_affected_days
from app.ingestion._dedup import upsert_health_samples
from app.ingestion._shortcuts import extract_unexpected_keys, normalize_apple_health_payload
from app.ingestion._types import IngestResult, NormalizedSample, RollupPort, WorkoutSyncPort
from app.kratos_sync import KratosReconciler
from app.models import BiometricMeasurement

if TYPE_CHECKING:
    from app.schemas import WithingsMeasurementPayload

logger = logging.getLogger(__name__)

_DIRECT_SCALE_OPTIONAL_METRICS: tuple[tuple[str, str, str], ...] = (
    ("body_fat_pct", "body_fat_pct", "%"),
    ("muscle_mass_kg", "muscle_mass_kg", "kg"),
    ("water_pct", "water_pct", "%"),
    ("bone_mass_kg", "bone_mass_kg", "kg"),
    ("resting_hr_bpm", "resting_hr", "bpm"),
)


def _log_unexpected_keys(payload: Any, keys: Sequence[str]) -> None:
    if not keys:
        return
    extra = getattr(payload, "__pydantic_extra__", None) or (
        payload if isinstance(payload, dict) else {}
    )
    logger.info("Apple Health sync received. Extra keys: %s", list(keys))
    for key in keys:
        val = extra.get(key)
        val_str = str(val)
        preview = f"{val_str[:120]}..." if len(val_str) > 120 else val_str
        logger.info("  Key [%s] (type %s): %s", key, type(val).__name__, preview)


def _decompose_direct_scale(payload: WithingsMeasurementPayload) -> tuple[NormalizedSample, ...]:
    samples: list[NormalizedSample] = [
        NormalizedSample(
            sample_type="weight",
            measured_at=payload.measured_at,
            ended_at=payload.measured_at,
            value=Decimal(payload.weight_kg),
            unit="kg",
            source="withings",
        )
    ]
    for attr_name, sample_type, unit in _DIRECT_SCALE_OPTIONAL_METRICS:
        raw_val = getattr(payload, attr_name, None)
        if raw_val is None:
            continue
        samples.append(
            NormalizedSample(
                sample_type=sample_type,
                measured_at=payload.measured_at,
                ended_at=payload.measured_at,
                value=Decimal(raw_val),
                unit=unit,
                source="withings",
            )
        )
    return tuple(samples)


class TelemetryIngestor:
    """Deep module orchestrating normalized telemetry persistence, rollups, and workout sync."""

    def __init__(
        self,
        db: Session,
        *,
        rollups: RollupPort | None = None,
        workout_sync: WorkoutSyncPort | None = None,
    ) -> None:
        self._db = db
        self._rollups: RollupPort = rollups if rollups is not None else DailyBiometrics(db)
        self._workout_sync: WorkoutSyncPort | None = workout_sync

    def ingest_samples(
        self,
        samples: Sequence[NormalizedSample],
        *,
        sync_workouts: bool = False,
    ) -> IngestResult:
        """Persists samples idempotently, materializes daily rollups, and optionally syncs workouts."""
        if not samples:
            return IngestResult(
                status="empty",
                total_samples_ingested=0,
                samples_created=0,
                metrics_summary={},
                days_processed=(),
                detected_workouts=(),
            )

        metrics_counts, created_count = upsert_health_samples(self._db, samples)
        self._db.commit()

        affected_days = resolve_sleep_affected_days(samples, settings.tz)
        batch = self._rollups.materialize_rollups(affected_days)
        self._db.commit()

        detected_workouts = tuple(batch.detected_workouts)
        if sync_workouts and detected_workouts:
            syncer = self._workout_sync if self._workout_sync is not None else KratosReconciler(self._db)
            syncer.sync(list(detected_workouts))

        return IngestResult(
            status="ok",
            total_samples_ingested=len(samples),
            samples_created=created_count,
            metrics_summary=metrics_counts,
            days_processed=tuple(batch.days),
            detected_workouts=detected_workouts,
        )

    def ingest_apple_health(self, payload: Any) -> IngestResult:
        """Normalizes an Apple Health / Shortcuts payload and ingests it with workout sync."""
        unexpected_keys = extract_unexpected_keys(payload)
        _log_unexpected_keys(payload, unexpected_keys)
        samples = normalize_apple_health_payload(payload)
        return self.ingest_samples(samples, sync_workouts=True)

    def ingest_direct_scale(self, payload: WithingsMeasurementPayload) -> BiometricMeasurement:
        """Ingests a direct scale reading through HealthSample dedup and unified daily rollup."""
        samples = _decompose_direct_scale(payload)
        upsert_health_samples(self._db, samples)
        self._db.commit()

        batch = self._rollups.materialize_rollups([payload.measured_at])
        measurement = batch.results[0].measurement
        measurement.source = "withings"
        if payload.notes:
            measurement.notes = payload.notes
        if payload.extra:
            measurement.extra = {**(measurement.extra or {}), **dict(payload.extra)}
        self._db.commit()
        self._db.refresh(measurement)
        return measurement
