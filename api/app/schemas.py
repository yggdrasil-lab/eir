"""Pydantic schemas for Eir API request and response validation."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_numeric(v: Any) -> Decimal | None:
    """Robustly parse numeric values from iOS Shortcuts (handles units, arrays, and multiline)."""
    if v is None:
        return None
    if isinstance(v, (int, float, Decimal)):
        return Decimal(str(v))
    if isinstance(v, dict):
        for k in ("value", "qty", "amount", "total"):
            if k in v:
                return _clean_numeric(v[k])
        return None
    if isinstance(v, (list, tuple)):
        # Sum list of numbers if iOS passes multiple samples
        cleaned = [_clean_numeric(x) for x in v]
        nums = [x for x in cleaned if x is not None]
        return sum(nums, Decimal(0)) if nums else None
    if isinstance(v, str):
        # Extract digits and decimals, ignoring suffixes like 'kcal', 'min', 'bpm'
        # If string contains multiple numbers/lines, sum all matches
        matches = re.findall(r"[-+]?\d*\.?\d+", v.replace(",", ""))
        if not matches:
            return None
        nums = []
        for m in matches:
            try:
                nums.append(Decimal(m))
            except InvalidOperation:
                pass
        return sum(nums, Decimal(0)) if nums else None
    return None


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthRead(BaseModel):
    status: str
    version: str
    database: str


# ------------------------------------------------------------- Integrations
class AppleWorkoutPayload(BaseModel):
    """Workout telemetry pushed from Apple Watch via iOS Shortcuts."""

    workout_type: str = "Workout"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None
    duration_minutes: Decimal | None = None
    active_energy_kcal: Decimal | None = None
    avg_heart_rate_bpm: int | None = None
    max_heart_rate_bpm: int | None = None
    distance_meters: Decimal | None = None
    pool_length_meters: Decimal | None = None
    strokes: int | None = None
    notes: str | None = None
    forward_to_kratos: bool = False
    extra: dict = Field(default_factory=dict)

    @field_validator("duration_minutes", "active_energy_kcal", "distance_meters", mode="before")
    @classmethod
    def parse_decimals(cls, v: Any) -> Decimal | None:
        return _clean_numeric(v)

    @field_validator("duration_seconds", "avg_heart_rate_bpm", "max_heart_rate_bpm", mode="before")
    @classmethod
    def parse_ints(cls, v: Any) -> int | None:
        val = _clean_numeric(v)
        return int(round(val)) if val is not None else None


class AppleBiometricsPayload(BaseModel):
    """Daily or morning biometrics from Apple Health via iOS Shortcuts."""

    measured_at: datetime = Field(default_factory=datetime.now)
    weight_kg: Decimal | None = None
    body_fat_pct: Decimal | None = None
    resting_hr_bpm: int | None = None
    hrv_ms: Decimal | None = None
    sleep_duration_s: int | None = None
    sleep_deep_s: int | None = None
    sleep_rem_s: int | None = None
    sleep_score: int | None = None
    step_count: int | None = None
    active_energy_kcal: int | None = None
    notes: str | None = None
    extra: dict = Field(default_factory=dict)


class WithingsMeasurementPayload(BaseModel):
    """Direct body scale measurement from Withings Body Smart scale."""

    measured_at: datetime
    weight_kg: Decimal
    body_fat_pct: Decimal | None = None
    muscle_mass_kg: Decimal | None = None
    water_pct: Decimal | None = None
    bone_mass_kg: Decimal | None = None
    resting_hr_bpm: int | None = None
    notes: str | None = None
    extra: dict = Field(default_factory=dict)


class WithingsNotificationPayload(BaseModel):
    """Webhook callback from Withings notification subscription."""

    userid: str | None = None
    appli: int | None = None  # 1 = weight, 44 = sleep, 16 = activity
    startdate: int | None = None
    enddate: int | None = None


# ----------------------------------------------------------------- Biometrics
class BiometricCreate(BaseModel):
    """Manual biometric entry."""

    measured_at: datetime = Field(default_factory=datetime.now)
    source: str = "manual"
    weight_kg: Decimal | None = None
    body_fat_pct: Decimal | None = None
    muscle_mass_kg: Decimal | None = None
    water_pct: Decimal | None = None
    bone_mass_kg: Decimal | None = None
    resting_hr_bpm: int | None = None
    hrv_ms: Decimal | None = None
    sleep_duration_s: int | None = None
    sleep_deep_s: int | None = None
    sleep_rem_s: int | None = None
    sleep_score: int | None = None
    step_count: int | None = None
    active_energy_kcal: int | None = None
    notes: str | None = None
    extra: dict = Field(default_factory=dict)


class BiometricRead(ORMModel):
    id: uuid.UUID
    measured_at: datetime
    source: str
    weight_kg: Decimal | None
    body_fat_pct: Decimal | None
    muscle_mass_kg: Decimal | None
    water_pct: Decimal | None
    bone_mass_kg: Decimal | None
    resting_hr_bpm: int | None
    hrv_ms: Decimal | None
    sleep_duration_s: int | None
    sleep_deep_s: int | None
    sleep_rem_s: int | None
    sleep_score: int | None
    step_count: int | None
    active_energy_kcal: int | None
    notes: str | None
    extra: dict
    created_at: datetime


class DailyBiometricSummary(BaseModel):
    """Daily aggregated vitals rollup for morning briefings and retrospectives."""

    date: date
    latest_weight_kg: Decimal | None = None
    latest_body_fat_pct: Decimal | None = None
    resting_hr_bpm: int | None = None
    hrv_ms: Decimal | None = None
    sleep_duration_hours: float | None = None
    sleep_deep_hours: float | None = None
    sleep_score: int | None = None
    step_count: int | None = None
    active_energy_kcal: int | None = None
    readiness_notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------- Granular Samples
class HealthSampleItem(BaseModel):
    """A single granular time-series sample (e.g. active energy slice)."""

    measured_at: datetime
    ended_at: datetime | None = None
    value: Decimal
    sample_type: str | None = None
    unit: str | None = None
    source: str | None = None
    external_id: str | None = None
    extra: dict = Field(default_factory=dict)

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: Any) -> Decimal:
        parsed = _clean_numeric(v)
        if parsed is None:
            raise ValueError(f"Cannot parse numeric value from {v}")
        return parsed


class HealthSampleBulkPayload(BaseModel):
    """Bulk ingestion payload accepting either object arrays or parallel arrays from iOS Shortcuts."""

    source: str = "apple_health"
    sample_type: str = "active_energy"
    unit: str | None = "kcal"
    samples: list[HealthSampleItem] = Field(default_factory=list)
    start_dates: list[datetime] = Field(default_factory=list)
    end_dates: list[datetime] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)

    def to_sample_items(self) -> list[HealthSampleItem]:
        """Unifies input format (samples array or parallel arrays) into items."""
        items: list[HealthSampleItem] = []
        for s in self.samples:
            items.append(
                HealthSampleItem(
                    measured_at=s.measured_at,
                    ended_at=s.ended_at or s.measured_at,
                    value=s.value,
                    sample_type=s.sample_type or self.sample_type,
                    unit=s.unit or self.unit,
                    source=s.source or self.source,
                    external_id=s.external_id,
                    extra=s.extra,
                )
            )
        if self.start_dates and self.values:
            for i, (st, val) in enumerate(zip(self.start_dates, self.values)):
                end = self.end_dates[i] if i < len(self.end_dates) else st
                cleaned_val = _clean_numeric(val)
                if cleaned_val is not None:
                    items.append(
                        HealthSampleItem(
                            measured_at=st,
                            ended_at=end,
                            value=cleaned_val,
                            sample_type=self.sample_type,
                            unit=self.unit,
                            source=self.source,
                        )
                    )
        return items


class HealthSampleRead(ORMModel):
    id: uuid.UUID
    source: str
    sample_type: str
    measured_at: datetime
    ended_at: datetime
    value: Decimal
    unit: str | None = None
    created_at: datetime


class HealthSampleBulkResponse(BaseModel):
    status: str = "ok"
    ingested: int
    sample_type: str
    daily_total: Decimal | None = None


# ------------------------------------------------------------- Comprehensive Sync
DEFAULT_METRIC_UNITS: dict[str, str] = {
    "active_energy": "kcal",
    "basal_energy": "kcal",
    "heart_rate": "bpm",
    "resting_hr": "bpm",
    "hrv": "ms",
    "sleep": "s",
    "sleep_duration": "s",
    "step_count": "count",
    "weight": "kg",
    "body_fat": "%",
    "body_fat_pct": "%",
    "distance": "m",
    "flights": "count",
    "vo2_max": "ml/kg/min",
    "oxygen_saturation": "%",
    "body_temperature": "C",
    "wrist_temperature": "C",
}

METRIC_ALIASES: dict[str, str] = {
    "weight_kg": "weight",
    "body_fat_percentage": "body_fat_pct",
    "body_fat": "body_fat_pct",
    "resting_heart_rate": "resting_hr",
    "resting_hr_bpm": "resting_hr",
    "heart_rate_variability": "hrv",
    "hrv_sdnn": "hrv",
    "hrv_ms": "hrv",
    "sleep_analysis": "sleep",
    "sleep_duration_s": "sleep",
    "sleep_duration": "sleep",
    "steps": "step_count",
    "step": "step_count",
    "active_energy_kcal": "active_energy",
    "energy": "active_energy",
    "distance_meters": "distance",
}


class AppleHealthComprehensiveSyncPayload(BaseModel):
    """Universal payload for high-resolution multi-metric Apple Health synchronization."""

    model_config = ConfigDict(extra="allow")

    source: str = "apple_health"
    samples: list[HealthSampleItem] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def extract_all_samples(self) -> list[HealthSampleItem]:
        """Extracts and normalizes samples from explicit items, metric groups, or flat parallel arrays."""
        items: list[HealthSampleItem] = list(self.samples)

        # 1. Process explicit metrics dictionary: {"heart_rate": {"start_dates": [...], "values": [...]}}
        for raw_metric_name, data in self.metrics.items():
            canonical_type = METRIC_ALIASES.get(raw_metric_name, raw_metric_name)
            default_unit = DEFAULT_METRIC_UNITS.get(canonical_type)
            if isinstance(data, dict):
                s_dates = data.get("start_dates") or data.get("dates") or []
                e_dates = data.get("end_dates") or []
                vals = data.get("values") or data.get("durations") or []
                unit = data.get("unit") or default_unit
                for i, (st, val) in enumerate(zip(s_dates, vals)):
                    end = e_dates[i] if i < len(e_dates) else st
                    cleaned_val = _clean_numeric(val)
                    if cleaned_val is not None:
                        items.append(
                            HealthSampleItem(
                                measured_at=st,
                                ended_at=end,
                                value=cleaned_val,
                                sample_type=canonical_type,
                                unit=unit,
                                source=self.source,
                            )
                        )

        # 2. Process flat parallel array patterns from extra fields: e.g. "heart_rate_dates" & "heart_rate_values"
        extra_fields = getattr(self, "__pydantic_extra__", {}) or {}
        prefixes: set[str] = set()
        for k in extra_fields.keys():
            if k.endswith("_dates"):
                prefix = k[:-6]
                if prefix.endswith("_start"):
                    prefix = prefix[:-6]
                prefixes.add(prefix)

        for prefix in prefixes:
            s_dates = extra_fields.get(f"{prefix}_start_dates") or extra_fields.get(f"{prefix}_dates") or []
            e_dates = extra_fields.get(f"{prefix}_end_dates") or []
            vals = (
                extra_fields.get(f"{prefix}_values")
                or extra_fields.get(f"{prefix}_durations")
                or extra_fields.get(prefix)
                or []
            )
            canonical_type = METRIC_ALIASES.get(prefix, prefix)
            default_unit = DEFAULT_METRIC_UNITS.get(canonical_type)
            unit = extra_fields.get(f"{prefix}_unit") or default_unit

            if isinstance(s_dates, list) and isinstance(vals, list) and s_dates and vals:
                for i, (st, val) in enumerate(zip(s_dates, vals)):
                    end = e_dates[i] if (isinstance(e_dates, list) and i < len(e_dates)) else st
                    cleaned_val = _clean_numeric(val)
                    if cleaned_val is not None:
                        items.append(
                            HealthSampleItem(
                                measured_at=st,
                                ended_at=end,
                                value=cleaned_val,
                                sample_type=canonical_type,
                                unit=unit,
                                source=self.source,
                            )
                        )
        return items


class AppleHealthSyncResponse(BaseModel):
    status: str = "ok"
    total_samples_ingested: int
    metrics_summary: dict[str, int] = Field(default_factory=dict)
    days_processed: list[date] = Field(default_factory=list)
