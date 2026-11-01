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


class WithingsAuthStatusResponse(BaseModel):
    """Status of Withings OAuth2 connection."""

    status: str
    linked: bool
    provider: str = "withings"
    user_id: str | None = None
    expires_at: datetime | None = None
    scope: str | None = None


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


# ------------------------------------------------------------- Active Health Feedback
class HealthFeedbackBase(BaseModel):
    date: date
    feedback_type: str  # 'fasting', 'workout', 'wellness', 'sport'
    rating: str | None = None  # 'completed', 'optimal', 'too_hard', etc.
    score: Decimal | None = None  # 1.0 - 10.0 scale
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class HealthFeedbackCreate(HealthFeedbackBase):
    pass


class HealthFeedbackRead(ORMModel, HealthFeedbackBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class FastingFeedbackCreate(BaseModel):
    date: date
    target_window: str = "16:8"
    adherence: str  # 'completed', 'early', 'off_plan'
    energy: str = "normal"  # 'high', 'normal', 'low'
    headache: str = "none"  # 'none', 'mild', 'sugar_spike'
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class WorkoutFeedbackCreate(BaseModel):
    date: date
    routine_name: str | None = None
    duration_minutes: int | None = None
    total_volume_kg: Decimal | None = None
    completed_sets: int | None = None
    effort_rating: str  # 'too_easy', 'optimal', 'hard', 'too_hard'
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


class WellnessFeedbackCreate(BaseModel):
    date: date
    readiness_rating: str | None = None  # 'high', 'moderate', 'low'
    soreness_score: Decimal | None = None  # 1.0 - 5.0
    sore_areas: list[str] = Field(default_factory=list)  # e.g. ['Legs', 'Back']
    stress_level: str | None = None  # 'low', 'normal', 'high'
    notes: str | None = None
    logged_at: datetime = Field(default_factory=datetime.now)


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
    fasting: HealthFeedbackRead | None = None
    workout: HealthFeedbackRead | None = None
    wellness: HealthFeedbackRead | None = None


# ------------------------------------------------------------- Granular Samples
def _clean_datetime(v: Any) -> Any:
    """Parses ISO strings, timestamps, and iOS human-formatted date strings."""
    if v is None or isinstance(v, datetime):
        return v
    if isinstance(v, str):
        cleaned = v.replace("\u202f", " ").replace(" at ", " ").strip()
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            pass
        for fmt in (
            "%d %b %Y %I:%M %p",
            "%d %b %Y %I:%M:%S %p",
            "%d %B %Y %I:%M %p",
            "%b %d, %Y %I:%M %p",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %I:%M %p",
        ):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                pass
    return v


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

    @field_validator("measured_at", "ended_at", mode="before")
    @classmethod
    def parse_datetimes(cls, v: Any) -> Any:
        return _clean_datetime(v)

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: Any) -> Decimal:
        parsed = _clean_numeric(v)
        if parsed is None:
            raise ValueError(f"Cannot parse numeric value from {v}")
        return parsed


class HealthSampleRead(ORMModel):
    id: uuid.UUID
    source: str
    sample_type: str
    measured_at: datetime
    ended_at: datetime
    value: Decimal
    unit: str | None = None
    created_at: datetime


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
    "muscle_mass_kg": "kg",
    "bone_mass_kg": "kg",
    "water_pct": "%",
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
    "muscle_mass": "muscle_mass_kg",
    "lean_body_mass": "muscle_mass_kg",
    "bone_mass": "bone_mass_kg",
    "water_percentage": "water_pct",
    "hydration_pct": "water_pct",
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


def _to_list(val: Any) -> list[Any]:
    """Coerces list, tuple, or newline-delimited string from iOS Shortcuts into a list."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        lines = [line.strip() for line in val.splitlines() if line.strip()]
        return lines
    return [val]


def _parse_duration_seconds(v: Any) -> Decimal | None:
    """Parses duration from seconds, 'MM:SS', or 'HH:MM:SS' string into seconds."""
    if v is None:
        return None
    if isinstance(v, (int, float, Decimal)):
        return Decimal(str(v))
    if isinstance(v, str):
        v = v.strip()
        if ":" in v:
            parts = v.split(":")
            try:
                if len(parts) == 2:
                    return Decimal(int(parts[0]) * 60 + int(parts[1]))
                elif len(parts) == 3:
                    return Decimal(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
            except (ValueError, TypeError):
                pass
        return _clean_numeric(v)
    return None


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
                s_dates = _to_list(data.get("start_dates") or data.get("dates"))
                e_dates = _to_list(data.get("end_dates"))
                vals = _to_list(data.get("values") or data.get("durations"))
                sources = _to_list(data.get("sources") or data.get("devices"))
                unit = data.get("unit") or default_unit
                for i, (st, val) in enumerate(zip(s_dates, vals)):
                    end = e_dates[i] if i < len(e_dates) else st
                    cleaned_val = _clean_numeric(val)
                    if cleaned_val is not None:
                        extra_meta = {}
                        if i < len(sources) and sources[i]:
                            extra_meta["device"] = str(sources[i]).strip()
                        items.append(
                            HealthSampleItem(
                                measured_at=st,
                                ended_at=end,
                                value=cleaned_val,
                                sample_type=canonical_type,
                                unit=unit,
                                source=self.source,
                                extra=extra_meta,
                            )
                        )

        # 2. Process flat parallel array patterns from extra fields: e.g. "heart_rate_dates" & "heart_rate_values"
        extra_fields = dict(getattr(self, "__pydantic_extra__", {}) or {})
        # Unwrap nested dictionaries (e.g. keyless {"": {...}} or {"payload": {...}} from iOS Shortcuts)
        for ek, ev in list(extra_fields.items()):
            if isinstance(ev, dict) and any(subk.endswith("_dates") or subk in ("dates", "start_dates") for subk in ev.keys()):
                extra_fields.update(ev)

        prefixes: set[str] = set()
        for k in extra_fields.keys():
            if k.endswith("_dates"):
                prefix = k[:-6]
                if prefix.endswith("_start"):
                    prefix = prefix[:-6]
                prefixes.add(prefix)

        for prefix in prefixes:
            s_dates = _to_list(extra_fields.get(f"{prefix}_start_dates") or extra_fields.get(f"{prefix}_dates"))
            e_dates = _to_list(extra_fields.get(f"{prefix}_end_dates"))
            durations = _to_list(extra_fields.get(f"{prefix}_durations"))
            sources = _to_list(
                extra_fields.get(f"{prefix}_sources")
                or extra_fields.get(f"{prefix}_devices")
                or extra_fields.get(f"{prefix}_source_names")
            )
            raw_vals = _to_list(
                extra_fields.get(f"{prefix}_values")
                or extra_fields.get(prefix)
            )
            canonical_type = METRIC_ALIASES.get(prefix, prefix)
            default_unit = DEFAULT_METRIC_UNITS.get(canonical_type)
            unit = extra_fields.get(f"{prefix}_unit") or default_unit

            if canonical_type == "sleep":
                stages = extra_fields.get(f"{prefix}_stages") or (
                    raw_vals if raw_vals and any(isinstance(x, str) and not re.match(r"^[\d\.:]+$", str(x).strip()) for x in raw_vals) else []
                )
                stage_list = _to_list(stages)
                max_len = max(len(s_dates), len(e_dates), len(durations), len(raw_vals), len(stage_list), len(sources))

                for i in range(max_len):
                    st = s_dates[i] if i < len(s_dates) else None
                    if not st:
                        continue
                    end = e_dates[i] if i < len(e_dates) else st
                    dur = durations[i] if i < len(durations) else None
                    val = raw_vals[i] if i < len(raw_vals) else None
                    stage = stage_list[i] if i < len(stage_list) else None

                    duration_s: Decimal | None = None
                    st_dt = _clean_datetime(st)
                    end_dt = _clean_datetime(end)
                    if isinstance(st_dt, datetime) and isinstance(end_dt, datetime) and end_dt > st_dt:
                        duration_s = Decimal(str(round((end_dt - st_dt).total_seconds(), 2)))
                    elif dur:
                        duration_s = _parse_duration_seconds(dur)
                    elif val and (isinstance(val, (int, float, Decimal)) or (isinstance(val, str) and re.match(r"^[\d\.:]+$", val.strip()))):
                        duration_s = _parse_duration_seconds(val)

                    if duration_s is not None and duration_s > 0:
                        extra_meta = {}
                        if stage and isinstance(stage, str) and stage.strip():
                            extra_meta["stage"] = stage.strip()
                        if i < len(sources) and sources[i]:
                            extra_meta["device"] = str(sources[i]).strip()
                        items.append(
                            HealthSampleItem(
                                measured_at=st,
                                ended_at=end,
                                value=duration_s,
                                sample_type="sleep",
                                unit="s",
                                source=self.source,
                                extra=extra_meta,
                            )
                        )
            else:
                vals = raw_vals or durations
                if s_dates and vals:
                    for i, (st, val) in enumerate(zip(s_dates, vals)):
                        end = e_dates[i] if i < len(e_dates) else st
                        cleaned_val = _clean_numeric(val)
                        if cleaned_val is not None:
                            extra_meta = {}
                            if i < len(sources) and sources[i]:
                                extra_meta["device"] = str(sources[i]).strip()
                            items.append(
                                HealthSampleItem(
                                    measured_at=st,
                                    ended_at=end,
                                    value=cleaned_val,
                                    sample_type=canonical_type,
                                    unit=unit,
                                    source=self.source,
                                    extra=extra_meta,
                                )
                            )

        # 3. Process root-level metric fields: e.g. {"sample_type": "active_energy", "dates": [...], "values": [...]}
        root_st = extra_fields.get("sample_type")
        root_dates = _to_list(extra_fields.get("start_dates") or extra_fields.get("dates"))
        root_vals = _to_list(extra_fields.get("values") or extra_fields.get("durations"))
        root_sources = _to_list(extra_fields.get("sources") or extra_fields.get("devices"))
        if root_st and root_dates and root_vals:
            canonical_type = METRIC_ALIASES.get(root_st, root_st)
            default_unit = DEFAULT_METRIC_UNITS.get(canonical_type)
            unit = extra_fields.get("unit") or default_unit
            root_e_dates = _to_list(extra_fields.get("end_dates"))
            for i, (st, val) in enumerate(zip(root_dates, root_vals)):
                end = root_e_dates[i] if i < len(root_e_dates) else st
                cleaned_val = _clean_numeric(val)
                if cleaned_val is not None:
                    extra_meta = {}
                    if i < len(root_sources) and root_sources[i]:
                        extra_meta["device"] = str(root_sources[i]).strip()
                    items.append(
                        HealthSampleItem(
                            measured_at=st,
                            ended_at=end,
                            value=cleaned_val,
                            sample_type=canonical_type,
                            unit=unit,
                            source=self.source,
                            extra=extra_meta,
                        )
                    )

        return items


class AppleHealthSyncResponse(BaseModel):
    status: str = "ok"
    total_samples_ingested: int
    metrics_summary: dict[str, int] = Field(default_factory=dict)
    days_processed: list[date] = Field(default_factory=list)
