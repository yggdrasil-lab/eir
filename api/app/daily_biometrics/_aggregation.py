"""Sleep stage filtering, daily rollup upsert, and DailyBiometricSummary assembly."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.daily_biometrics._queries import (
    fetch_avg_sample_value,
    fetch_day_feedback,
    fetch_day_measurements,
    fetch_granular_samples,
    fetch_latest_day_measurement,
    fetch_latest_sample_value,
    fetch_latest_sample_value_on_or_before,
    fetch_sum_sample_value,
)
from app.daily_biometrics._registry import METRIC_REGISTRY
from app.daily_biometrics._sleep import (
    EPISODE_GAP,
    HRV_PAD,
    SegmentedSleepResult,
    compute_overnight_hrv,
    segment_sleep_day,
)
from app.daily_biometrics._types import DailyRollupResult, ExertionPort
from app.daily_biometrics._window import day_bounds, sleep_fetch_bounds
from app.exertion import deduplicate_step_samples
from app.models import BiometricMeasurement, HealthSample

if TYPE_CHECKING:
    from app.schemas import DailyBiometricSummary, DetectedWorkoutRead

_EXCLUDED_SLEEP_STAGES = frozenset({"awake", "inbed", "in_bed", "in bed"})
_EXTRA_SUMMARY_FIELDS = (
    "wrist_temperature_c",
    "respiratory_rate_bpm",
    "oxygen_saturation_pct",
    "walking_asymmetry_pct",
    "vo2_max",
)
_ROLLUP_LATEST_FIELDS = (
    ("weight_kg", "weight"),
    ("body_fat_pct", "body_fat_pct"),
    ("muscle_mass_kg", "muscle_mass_kg"),
    ("water_pct", "water_pct"),
    ("bone_mass_kg", "bone_mass_kg"),
    ("hrv_ms", "hrv"),
)
_AVG_METRIC_SPECS = (
    ("wrist_temperature_c", "wrist_temperature", 2),
    ("respiratory_rate_bpm", "respiratory_rate", 1),
    ("oxygen_saturation_pct", "oxygen_saturation", 1),
    ("walking_asymmetry_pct", "walking_asymmetry", 2),
)


def compute_sleep_totals(samples: Sequence[HealthSample]) -> tuple[bool, int, int, int]:
    """Sums non-awake sleep sample durations (seconds) into (has_sleep, total, deep, rem)."""
    sleep_total_s = 0
    sleep_deep_s = 0
    sleep_rem_s = 0
    for sample in samples:
        stage = str((sample.extra or {}).get("stage") or "").strip().lower()
        if stage in _EXCLUDED_SLEEP_STAGES:
            continue
        val = int(round(sample.value))
        sleep_total_s += val
        if stage == "deep":
            sleep_deep_s += val
        elif stage == "rem":
            sleep_rem_s += val
    return bool(samples), sleep_total_s, sleep_deep_s, sleep_rem_s


def _reconcile_energy_with_workouts(
    energy_kcal: Decimal | None,
    detected_workouts: Sequence[DetectedWorkoutRead],
) -> Decimal | None:
    detected_total = sum(w.active_energy_kcal for w in detected_workouts)
    if energy_kcal is not None and detected_total > float(energy_kcal):
        return Decimal(detected_total)
    if energy_kcal is None and detected_total > 0:
        return Decimal(detected_total)
    return energy_kcal


def _find_or_init_daily_measurement(
    db: Session,
    target_date: date,
    start_dt: datetime,
    end_dt: datetime,
    tz: ZoneInfo,
) -> BiometricMeasurement:
    existing = fetch_latest_day_measurement(db, start_dt, end_dt)
    if existing is not None:
        return existing
    measurement = BiometricMeasurement(
        id=uuid.uuid4(),
        measured_at=datetime.combine(target_date, time(8, 0)).replace(tzinfo=tz),
        source="daily_rollup",
        notes=f"Daily health rollup for {target_date.isoformat()}",
    )
    db.add(measurement)
    return measurement


def _apply_rollup_sleep_columns(
    measurement: BiometricMeasurement,
    segmented: SegmentedSleepResult,
    has_sleep_samples: bool,
) -> None:
    if segmented.overnight is not None:
        measurement.sleep_duration_s = segmented.overnight.sleep_s
        measurement.sleep_deep_s = segmented.overnight.deep_s if segmented.overnight.deep_s > 0 else None
        measurement.sleep_rem_s = segmented.overnight.rem_s if segmented.overnight.rem_s > 0 else None
        return
    if measurement.source != "manual" or has_sleep_samples:
        measurement.sleep_duration_s = None
        measurement.sleep_deep_s = None
        measurement.sleep_rem_s = None


def _populate_rollup_vitals(
    db: Session,
    measurement: BiometricMeasurement,
    target_date: date,
    tz: ZoneInfo,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[SegmentedSleepResult, Decimal | None]:
    for attr, metric_key in _ROLLUP_LATEST_FIELDS:
        val = fetch_latest_sample_value(db, METRIC_REGISTRY.aliases_for(metric_key), start_dt, end_dt)
        if val is not None:
            setattr(measurement, attr, val)

    rhr = fetch_latest_sample_value(db, METRIC_REGISTRY.aliases_for("resting_hr"), start_dt, end_dt)
    if rhr is not None:
        measurement.resting_hr_bpm = int(round(rhr))

    sleep_start, sleep_end = sleep_fetch_bounds(target_date, tz)
    sleep_samples = fetch_granular_samples(
        db,
        METRIC_REGISTRY.aliases_for("sleep"),
        sleep_start,
        sleep_end + EPISODE_GAP,
    )
    segmented = segment_sleep_day(sleep_samples, target_date, tz)
    _apply_rollup_sleep_columns(measurement, segmented, bool(sleep_samples))

    hrv_samples = fetch_granular_samples(
        db,
        METRIC_REGISTRY.aliases_for("hrv"),
        sleep_start - HRV_PAD,
        sleep_end,
    )
    overnight_hrv = compute_overnight_hrv(hrv_samples, segmented.overnight, target_date, tz)
    return segmented, overnight_hrv


def _write_sleep_rollup_extras(
    cur_extra: dict,
    segmented: SegmentedSleepResult,
    overnight_hrv: Decimal | None,
) -> None:
    cur_extra["sleep_segmented_v1"] = True
    if segmented.naps:
        cur_extra["nap_duration_s"] = segmented.nap_duration_s
        cur_extra["nap_count"] = segmented.nap_count
        cur_extra["naps"] = [
            {
                "start_at": ep.start_at.isoformat(),
                "end_at": ep.end_at.isoformat(),
                "duration_s": ep.sleep_s,
            }
            for ep in segmented.naps
        ]
    else:
        for key in ("nap_duration_s", "nap_count", "naps"):
            cur_extra.pop(key, None)

    if overnight_hrv is not None:
        cur_extra["overnight_hrv_ms"] = float(overnight_hrv)
    else:
        cur_extra.pop("overnight_hrv_ms", None)


def _populate_rollup_extras(
    db: Session,
    measurement: BiometricMeasurement,
    start_dt: datetime,
    end_dt: datetime,
    segmented: SegmentedSleepResult,
    overnight_hrv: Decimal | None,
) -> None:
    cur_extra = dict(measurement.extra or {})
    for key, metric_key, digits in _AVG_METRIC_SPECS:
        avg_val = fetch_avg_sample_value(db, METRIC_REGISTRY.aliases_for(metric_key), start_dt, end_dt)
        if avg_val is not None:
            cur_extra[key] = round(float(avg_val), digits)

    vo2max = fetch_latest_sample_value_on_or_before(db, METRIC_REGISTRY.aliases_for("vo2_max"), end_dt)
    if vo2max is not None:
        cur_extra["vo2_max"] = round(float(vo2max), 1)

    _write_sleep_rollup_extras(cur_extra, segmented, overnight_hrv)
    measurement.extra = dict(cur_extra)


def upsert_daily_rollup(
    db: Session,
    target_date: date,
    tz: ZoneInfo,
    exertion: ExertionPort,
) -> DailyRollupResult:
    """Computes and flushes a single-day BiometricMeasurement rollup with single-pass exertion."""
    start_dt, end_dt = day_bounds(target_date, tz)
    measurement = _find_or_init_daily_measurement(db, target_date, start_dt, end_dt, tz)
    segmented, overnight_hrv = _populate_rollup_vitals(db, measurement, target_date, tz, start_dt, end_dt)

    detected_workouts = tuple(exertion.detect_day(target_date))
    energy_kcal = fetch_sum_sample_value(db, METRIC_REGISTRY.aliases_for("active_energy"), start_dt, end_dt)
    reconciled_energy = _reconcile_energy_with_workouts(energy_kcal, detected_workouts)
    if reconciled_energy is not None:
        measurement.active_energy_kcal = int(round(reconciled_energy))

    step_samples = fetch_granular_samples(db, METRIC_REGISTRY.aliases_for("step_count"), start_dt, end_dt)
    if step_samples:
        measurement.step_count = int(round(deduplicate_step_samples(step_samples)))

    _populate_rollup_extras(db, measurement, start_dt, end_dt, segmented, overnight_hrv)
    db.flush()
    return DailyRollupResult(day=target_date, measurement=measurement, detected_workouts=detected_workouts)


def _apply_measurement_rollups(
    summary: DailyBiometricSummary,
    records: Sequence[BiometricMeasurement],
) -> list[str]:
    readiness_notes: list[str] = []
    for rec in records:
        _apply_single_measurement(summary, rec)
        if rec.notes:
            readiness_notes.append(rec.notes)
    return readiness_notes


def _apply_nap_extras(summary: DailyBiometricSummary, extra: dict) -> None:
    from app.schemas import SleepNapRead

    raw_naps = extra.get("naps")
    if not isinstance(raw_naps, list) or not raw_naps:
        return
    summary.naps = [
        SleepNapRead(
            start_at=datetime.fromisoformat(item["start_at"]),
            end_at=datetime.fromisoformat(item["end_at"]),
            duration_hours=round(int(item["duration_s"]) / 3600.0, 2),
        )
        for item in raw_naps
    ]
    summary.nap_count = int(extra.get("nap_count", len(summary.naps)))
    nap_s = extra.get("nap_duration_s")
    total_nap_s = int(nap_s) if nap_s is not None else sum(int(item["duration_s"]) for item in raw_naps)
    summary.nap_duration_hours = round(total_nap_s / 3600.0, 2)


def _apply_single_measurement(summary: DailyBiometricSummary, rec: BiometricMeasurement) -> None:
    if rec.weight_kg is not None:
        summary.latest_weight_kg = rec.weight_kg
    if rec.body_fat_pct is not None:
        summary.latest_body_fat_pct = rec.body_fat_pct
    if rec.resting_hr_bpm is not None:
        summary.resting_hr_bpm = rec.resting_hr_bpm
    if rec.hrv_ms is not None:
        summary.hrv_ms = rec.hrv_ms
    if rec.sleep_duration_s is not None:
        summary.sleep_duration_hours = round(rec.sleep_duration_s / 3600.0, 2)
    if rec.sleep_deep_s is not None:
        summary.sleep_deep_hours = round(rec.sleep_deep_s / 3600.0, 2)
    if rec.sleep_rem_s is not None:
        summary.sleep_rem_hours = round(rec.sleep_rem_s / 3600.0, 2)
    if rec.sleep_score is not None:
        summary.sleep_score = rec.sleep_score
    if rec.step_count is not None:
        summary.step_count = (summary.step_count or 0) + rec.step_count
    if rec.active_energy_kcal is not None:
        summary.active_energy_kcal = (summary.active_energy_kcal or 0) + rec.active_energy_kcal
    extra = rec.extra or {}
    for key in _EXTRA_SUMMARY_FIELDS:
        if key in extra:
            setattr(summary, key, extra[key])
    if extra.get("overnight_hrv_ms") is not None:
        summary.overnight_hrv_ms = Decimal(f"{float(extra['overnight_hrv_ms']):.2f}")
    _apply_nap_extras(summary, extra)


def _fill_summary_fallbacks(
    db: Session,
    summary: DailyBiometricSummary,
    target_date: date,
    tz: ZoneInfo,
    start_dt: datetime,
    end_dt: datetime,
    records: Sequence[BiometricMeasurement],
) -> None:
    granular_energy = fetch_sum_sample_value(db, METRIC_REGISTRY.aliases_for("active_energy"), start_dt, end_dt)
    if granular_energy is not None:
        summary.active_energy_kcal = int(round(granular_energy))

    if summary.step_count is None:
        raw_steps = fetch_granular_samples(db, METRIC_REGISTRY.aliases_for("step_count"), start_dt, end_dt)
        if raw_steps:
            summary.step_count = int(round(deduplicate_step_samples(raw_steps)))

    for attr, metric_key, digits in _AVG_METRIC_SPECS:
        if getattr(summary, attr) is None:
            val = fetch_avg_sample_value(db, METRIC_REGISTRY.aliases_for(metric_key), start_dt, end_dt)
            if val is not None:
                setattr(summary, attr, round(float(val), digits))

    if summary.vo2_max is None:
        vm = fetch_latest_sample_value_on_or_before(db, METRIC_REGISTRY.aliases_for("vo2_max"), end_dt)
        if vm is not None:
            summary.vo2_max = round(float(vm), 1)

    _fill_sleep_fallbacks(db, summary, target_date, tz, records)


def _apply_segmented_sleep_summary(
    summary: DailyBiometricSummary,
    segmented: SegmentedSleepResult,
    *,
    allow_clear: bool,
) -> None:
    from app.schemas import SleepNapRead

    if segmented.overnight is not None:
        summary.sleep_duration_hours = round(segmented.overnight.sleep_s / 3600.0, 2)
        summary.sleep_deep_hours = (
            round(segmented.overnight.deep_s / 3600.0, 2) if segmented.overnight.deep_s > 0 else None
        )
        summary.sleep_rem_hours = (
            round(segmented.overnight.rem_s / 3600.0, 2) if segmented.overnight.rem_s > 0 else None
        )
    elif allow_clear:
        summary.sleep_duration_hours = None
        summary.sleep_deep_hours = None
        summary.sleep_rem_hours = None

    summary.nap_count = segmented.nap_count
    summary.nap_duration_hours = (
        round(segmented.nap_duration_s / 3600.0, 2) if segmented.nap_duration_s is not None else None
    )
    summary.naps = [
        SleepNapRead(
            start_at=ep.start_at,
            end_at=ep.end_at,
            duration_hours=round(ep.sleep_s / 3600.0, 2),
        )
        for ep in segmented.naps
    ]


def _fill_sleep_fallbacks(
    db: Session,
    summary: DailyBiometricSummary,
    target_date: date,
    tz: ZoneInfo,
    records: Sequence[BiometricMeasurement],
) -> None:
    if any(bool((rec.extra or {}).get("sleep_segmented_v1")) for rec in records):
        return

    sleep_start, sleep_end = sleep_fetch_bounds(target_date, tz)
    sleep_samples = fetch_granular_samples(
        db,
        METRIC_REGISTRY.aliases_for("sleep"),
        sleep_start,
        sleep_end + EPISODE_GAP,
    )
    segmented_overnight = None
    if sleep_samples:
        segmented = segment_sleep_day(sleep_samples, target_date, tz)
        segmented_overnight = segmented.overnight
        allow_clear = not any(rec.source == "manual" for rec in records)
        _apply_segmented_sleep_summary(summary, segmented, allow_clear=allow_clear)

    hrv_samples = fetch_granular_samples(
        db,
        METRIC_REGISTRY.aliases_for("hrv"),
        sleep_start - HRV_PAD,
        sleep_end,
    )
    if hrv_samples:
        overnight_hrv = compute_overnight_hrv(hrv_samples, segmented_overnight, target_date, tz)
        if overnight_hrv is not None:
            summary.overnight_hrv_ms = overnight_hrv


def _compute_total_sleep_hours(
    sleep_hours: float | None,
    nap_hours: float | None,
) -> float | None:
    if sleep_hours is None and nap_hours is None:
        return None
    return round((sleep_hours or 0.0) + (nap_hours or 0.0), 2)


def build_daily_summary(
    db: Session,
    target_date: date,
    tz: ZoneInfo,
    exertion: ExertionPort,
) -> DailyBiometricSummary:
    """Assembles the canonical DailyBiometricSummary for a single local calendar day."""
    from app.schemas import DailyBiometricSummary

    start_dt, end_dt = day_bounds(target_date, tz)
    records = fetch_day_measurements(db, start_dt, end_dt)
    summary = DailyBiometricSummary(date=target_date)
    summary.readiness_notes = _apply_measurement_rollups(summary, records)
    _fill_summary_fallbacks(db, summary, target_date, tz, start_dt, end_dt, records)
    summary.total_sleep_hours = _compute_total_sleep_hours(
        summary.sleep_duration_hours,
        summary.nap_duration_hours,
    )

    for fb in fetch_day_feedback(db, target_date):
        if fb.feedback_type in ("fasting", "workout", "wellness"):
            setattr(summary, fb.feedback_type, fb)

    summary.detected_workouts = exertion.detect_day(target_date)
    if summary.detected_workouts:
        detected_total = sum(w.active_energy_kcal for w in summary.detected_workouts)
        if summary.active_energy_kcal is None or detected_total > summary.active_energy_kcal:
            summary.active_energy_kcal = detected_total
    return summary
