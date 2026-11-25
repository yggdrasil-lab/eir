"""Pure Apple Health / iOS Shortcuts payload normalization and parsing helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from app.daily_biometrics._registry import METRIC_REGISTRY
from app.ingestion._types import IngestionError, NormalizedSample

_DATETIME_FORMATS: tuple[str, ...] = (
    "%d %b %Y %I:%M %p",
    "%d %b %Y %I:%M:%S %p",
    "%d %B %Y %I:%M %p",
    "%b %d, %Y %I:%M %p",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %I:%M %p",
)
_PAYLOAD_STRUCTURAL_KEYS = frozenset({"source", "samples", "metrics"})
_NUMERIC_TOKEN_RE = re.compile(r"^[\d\.:]+$")


def clean_numeric(v: Any) -> Decimal | None:
    """Robustly parse numeric values from iOS Shortcuts (handles units, arrays, and multiline)."""
    if v is None:
        return None
    if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool):
        return Decimal(str(v))
    if isinstance(v, Mapping):
        for key in ("value", "qty", "amount", "total"):
            if key in v:
                return clean_numeric(v[key])
        return None
    if isinstance(v, (list, tuple)):
        nums = [parsed for item in v if (parsed := clean_numeric(item)) is not None]
        return sum(nums, Decimal(0)) if nums else None
    if isinstance(v, str):
        return _parse_numeric_string(v)
    return None


def _parse_numeric_string(raw: str) -> Decimal | None:
    matches = re.findall(r"[-+]?\d*\.?\d+", raw.replace(",", ""))
    if not matches:
        return None
    nums: list[Decimal] = []
    for token in matches:
        try:
            nums.append(Decimal(token))
        except InvalidOperation:
            pass
    return sum(nums, Decimal(0)) if nums else None


def clean_datetime(v: Any) -> Any:
    """Parses ISO strings and iOS human-formatted date strings into datetime when possible."""
    if v is None or isinstance(v, datetime):
        return v
    if not isinstance(v, str):
        return v
    cleaned = v.replace("\u202f", " ").replace(" at ", " ").strip()
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        pass
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return v


def to_list(val: Any) -> list[Any]:
    """Coerces list, tuple, or newline-delimited string from iOS Shortcuts into a list."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        return [line.strip() for line in val.splitlines() if line.strip()]
    return [val]


def parse_duration_seconds(v: Any) -> Decimal | None:
    """Parses duration from seconds, 'MM:SS', or 'HH:MM:SS' string into seconds."""
    if v is None:
        return None
    if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool):
        return Decimal(str(v))
    if not isinstance(v, str):
        return None
    text = v.strip()
    if ":" in text:
        parsed_colon = _parse_colon_duration(text)
        if parsed_colon is not None:
            return parsed_colon
    return clean_numeric(text)


def _parse_colon_duration(text: str) -> Decimal | None:
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return Decimal(int(parts[0]) * 60 + int(parts[1]))
        if len(parts) == 3:
            return Decimal(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    except (ValueError, TypeError):
        return None
    return None


def _require_datetime(raw: Any, field_name: str) -> datetime:
    parsed = clean_datetime(raw)
    if isinstance(parsed, datetime):
        return parsed
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    raise IngestionError(f"Cannot parse datetime for {field_name}: {raw!r}")


def _raw_extra_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {k: v for k, v in payload.items() if k not in _PAYLOAD_STRUCTURAL_KEYS}
    return dict(getattr(payload, "__pydantic_extra__", {}) or {})


def extract_unexpected_keys(payload: Any) -> list[str]:
    """Returns top-level extra/unexpected keys carried on an Apple Health sync payload."""
    return list(_raw_extra_mapping(payload).keys())


def _unwrap_extra_fields(extra_fields: Mapping[str, Any]) -> dict[str, Any]:
    unwrapped = dict(extra_fields)
    for value in list(unwrapped.values()):
        if isinstance(value, Mapping) and any(
            sub_key.endswith("_dates") or sub_key in ("dates", "start_dates")
            for sub_key in value.keys()
        ):
            unwrapped.update(value)
    return unwrapped


def _build_extra_meta(
    sources: Sequence[Any],
    index: int,
    *,
    stage: Any = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if isinstance(stage, str) and stage.strip():
        meta["stage"] = stage.strip()
    if index < len(sources) and sources[index]:
        meta["device"] = str(sources[index]).strip()
    return meta


def _resolve_sleep_duration(
    start_raw: Any,
    end_raw: Any,
    dur_raw: Any,
    val_raw: Any,
) -> Decimal | None:
    st_dt = clean_datetime(start_raw)
    end_dt = clean_datetime(end_raw)
    if isinstance(st_dt, datetime) and isinstance(end_dt, datetime) and end_dt > st_dt:
        return Decimal(str(round((end_dt - st_dt).total_seconds(), 2)))
    if dur_raw:
        return parse_duration_seconds(dur_raw)
    if val_raw is None:
        return None
    if isinstance(val_raw, (int, float, Decimal)) and not isinstance(val_raw, bool):
        return parse_duration_seconds(val_raw)
    if isinstance(val_raw, str) and _NUMERIC_TOKEN_RE.match(val_raw.strip()):
        return parse_duration_seconds(val_raw)
    return None


def _build_sleep_sample(
    *,
    source: str,
    st_raw: Any,
    end_raw: Any,
    dur_raw: Any,
    val_raw: Any,
    stage: Any,
    sources: Sequence[Any],
    idx: int,
) -> NormalizedSample | None:
    if not st_raw:
        return None
    duration_s = _resolve_sleep_duration(st_raw, end_raw, dur_raw, val_raw)
    if duration_s is None or duration_s <= 0:
        return None
    return NormalizedSample(
        sample_type="sleep",
        measured_at=_require_datetime(st_raw, "sleep.measured_at"),
        ended_at=_require_datetime(end_raw, "sleep.ended_at"),
        value=duration_s,
        unit="s",
        source=source,
        extra=_build_extra_meta(sources, idx, stage=stage),
    )


def _extract_sleep_series(
    *,
    source: str,
    start_dates: Sequence[Any],
    end_dates: Sequence[Any],
    durations: Sequence[Any],
    raw_vals: Sequence[Any],
    stages_raw: Any,
    sources: Sequence[Any],
) -> list[NormalizedSample]:
    has_text_stages = any(
        isinstance(x, str) and not _NUMERIC_TOKEN_RE.match(x.strip()) for x in raw_vals
    )
    stages = to_list(stages_raw if stages_raw is not None else (raw_vals if has_text_stages else ()))
    max_len = max(len(start_dates), len(end_dates), len(durations), len(raw_vals), len(stages), len(sources), 0)
    results: list[NormalizedSample] = []

    for idx in range(max_len):
        st_raw = start_dates[idx] if idx < len(start_dates) else None
        sample = _build_sleep_sample(
            source=source,
            st_raw=st_raw,
            end_raw=end_dates[idx] if idx < len(end_dates) else st_raw,
            dur_raw=durations[idx] if idx < len(durations) else None,
            val_raw=raw_vals[idx] if idx < len(raw_vals) else None,
            stage=stages[idx] if idx < len(stages) else None,
            sources=sources,
            idx=idx,
        )
        if sample is not None:
            results.append(sample)
    return results


def _extract_numeric_series(
    *,
    sample_type: str,
    unit: str | None,
    source: str,
    start_dates: Sequence[Any],
    end_dates: Sequence[Any],
    values: Sequence[Any],
    sources: Sequence[Any],
) -> list[NormalizedSample]:
    if not start_dates or not values:
        return []
    results: list[NormalizedSample] = []
    for idx, (st_raw, val_raw) in enumerate(zip(start_dates, values)):
        cleaned_val = clean_numeric(val_raw)
        if cleaned_val is None:
            continue
        end_raw = end_dates[idx] if idx < len(end_dates) else st_raw
        measured_at = _require_datetime(st_raw, f"{sample_type}.measured_at")
        ended_at = _require_datetime(end_raw, f"{sample_type}.ended_at")
        results.append(
            NormalizedSample(
                sample_type=sample_type,
                measured_at=measured_at,
                ended_at=ended_at,
                value=cleaned_val,
                unit=unit,
                source=source,
                extra=_build_extra_meta(sources, idx),
            )
        )
    return results


def _extract_series_samples(
    *,
    raw_metric: str,
    source: str,
    start_dates: Sequence[Any],
    end_dates: Sequence[Any],
    raw_vals: Sequence[Any],
    durations: Sequence[Any],
    sources: Sequence[Any],
    explicit_unit: str | None,
    stages_raw: Any = None,
    sleep_mode: bool = False,
) -> list[NormalizedSample]:
    canonical_type = METRIC_REGISTRY.canonicalize(raw_metric)
    unit = explicit_unit or METRIC_REGISTRY.default_unit_for(canonical_type)
    if sleep_mode and canonical_type == "sleep":
        return _extract_sleep_series(
            source=source,
            start_dates=start_dates,
            end_dates=end_dates,
            durations=durations,
            raw_vals=raw_vals,
            stages_raw=stages_raw,
            sources=sources,
        )
    return _extract_numeric_series(
        sample_type=canonical_type,
        unit=unit,
        source=source,
        start_dates=start_dates,
        end_dates=end_dates,
        values=raw_vals or durations,
        sources=sources,
    )


def _extract_explicit_samples(raw_samples: Sequence[Any], default_source: str) -> list[NormalizedSample]:
    results: list[NormalizedSample] = []
    for item in raw_samples:
        if isinstance(item, Mapping):
            raw_st = item.get("sample_type")
            measured_at = _require_datetime(item.get("measured_at"), "samples.measured_at")
            raw_ended = item.get("ended_at")
            ended_at = _require_datetime(raw_ended, "samples.ended_at") if raw_ended is not None else measured_at
            value = clean_numeric(item.get("value"))
            if value is None:
                raise IngestionError(f"Cannot parse numeric value from {item.get('value')!r}")
            unit = item.get("unit")
            source = str(item.get("source") or default_source)
            external_id = item.get("external_id")
            extra = item.get("extra") or {}
        else:
            raw_st = getattr(item, "sample_type", None)
            measured_at = _require_datetime(item.measured_at, "samples.measured_at")
            raw_ended = getattr(item, "ended_at", None)
            ended_at = _require_datetime(raw_ended, "samples.ended_at") if raw_ended is not None else measured_at
            value = Decimal(item.value)
            unit = getattr(item, "unit", None)
            source = getattr(item, "source", None) or default_source
            external_id = getattr(item, "external_id", None)
            extra = getattr(item, "extra", None) or {}
        sample_type = METRIC_REGISTRY.canonicalize(raw_st) if raw_st else "unknown"
        results.append(
            NormalizedSample(
                sample_type=sample_type,
                measured_at=measured_at,
                ended_at=ended_at,
                value=value,
                unit=unit,
                source=source,
                external_id=external_id,
                extra=extra,
            )
        )
    return results


def _extract_metrics_dict_samples(metrics_map: Mapping[str, Any], source: str) -> list[NormalizedSample]:
    results: list[NormalizedSample] = []
    for raw_metric, data in metrics_map.items():
        if not isinstance(data, Mapping):
            continue
        results.extend(
            _extract_series_samples(
                raw_metric=raw_metric,
                source=source,
                start_dates=to_list(data.get("start_dates") or data.get("dates")),
                end_dates=to_list(data.get("end_dates")),
                raw_vals=to_list(data.get("values") or data.get("durations")),
                durations=to_list(data.get("durations")),
                sources=to_list(data.get("sources") or data.get("devices")),
                explicit_unit=data.get("unit"),
            )
        )
    return results


def _collect_prefixed_keys(extra_fields: Mapping[str, Any]) -> list[str]:
    prefixes: dict[str, None] = {}
    for key in extra_fields.keys():
        if key.endswith("_dates"):
            prefix = key[:-6]
            if prefix.endswith("_start"):
                prefix = prefix[:-6]
            prefixes[prefix] = None
    return list(prefixes.keys())


def _extract_prefixed_samples(extra_fields: Mapping[str, Any], source: str) -> list[NormalizedSample]:
    results: list[NormalizedSample] = []
    for prefix in _collect_prefixed_keys(extra_fields):
        raw_vals = to_list(extra_fields.get(f"{prefix}_values") or extra_fields.get(prefix))
        results.extend(
            _extract_series_samples(
                raw_metric=prefix,
                source=source,
                start_dates=to_list(
                    extra_fields.get(f"{prefix}_start_dates") or extra_fields.get(f"{prefix}_dates")
                ),
                end_dates=to_list(extra_fields.get(f"{prefix}_end_dates")),
                raw_vals=raw_vals,
                durations=to_list(extra_fields.get(f"{prefix}_durations")),
                sources=to_list(
                    extra_fields.get(f"{prefix}_sources")
                    or extra_fields.get(f"{prefix}_devices")
                    or extra_fields.get(f"{prefix}_source_names")
                ),
                explicit_unit=extra_fields.get(f"{prefix}_unit"),
                stages_raw=extra_fields.get(f"{prefix}_stages"),
                sleep_mode=True,
            )
        )
    return results


def _extract_root_series_samples(extra_fields: Mapping[str, Any], source: str) -> list[NormalizedSample]:
    root_st = extra_fields.get("sample_type")
    if not isinstance(root_st, str) or not root_st.strip():
        return []
    return _extract_series_samples(
        raw_metric=root_st.strip(),
        source=source,
        start_dates=to_list(extra_fields.get("start_dates") or extra_fields.get("dates")),
        end_dates=to_list(extra_fields.get("end_dates")),
        raw_vals=to_list(extra_fields.get("values") or extra_fields.get("durations")),
        durations=to_list(extra_fields.get("durations")),
        sources=to_list(extra_fields.get("sources") or extra_fields.get("devices")),
        explicit_unit=extra_fields.get("unit"),
    )


def normalize_apple_health_payload(
    payload: Any,
    *,
    default_source: str = "apple_health",
) -> tuple[NormalizedSample, ...]:
    """Extracts and normalizes all Shortcuts/Apple Health sample formats into NormalizedSample."""
    if isinstance(payload, Mapping):
        source = str(payload.get("source") or default_source)
        raw_samples = payload.get("samples") or ()
        metrics_map = payload.get("metrics") or {}
    else:
        source = str(getattr(payload, "source", None) or default_source)
        raw_samples = getattr(payload, "samples", ()) or ()
        metrics_map = getattr(payload, "metrics", {}) or {}

    extra_fields = _unwrap_extra_fields(_raw_extra_mapping(payload))
    samples = _extract_explicit_samples(raw_samples, source)
    if isinstance(metrics_map, Mapping):
        samples.extend(_extract_metrics_dict_samples(metrics_map, source))
    samples.extend(_extract_prefixed_samples(extra_fields, source))
    samples.extend(_extract_root_series_samples(extra_fields, source))
    return tuple(samples)
