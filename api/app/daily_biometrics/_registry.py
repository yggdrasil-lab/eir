"""Stdlib-only canonical metric specification, alias, and unit registry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """Declarative specification for a single canonical health metric."""

    canonical: str
    default_unit: str | None = None
    query_aliases: tuple[str, ...] = ()
    ingest_only_aliases: tuple[str, ...] = ()
    legacy_unit_keys: tuple[str, ...] = ()
    include_self_in_input_aliases: bool = False


class MetricAliasRegistry:
    """Single source of truth for metric canonicalization, DB query aliases, and units."""

    def __init__(self, specs: tuple[MetricSpec, ...]) -> None:
        self._specs: dict[str, MetricSpec] = {s.canonical: s for s in specs}
        self._input_aliases: dict[str, str] = {}
        self._default_units: dict[str, str] = {}

        for spec in specs:
            if spec.default_unit is not None:
                self._default_units[spec.canonical] = spec.default_unit
                for legacy_key in spec.legacy_unit_keys:
                    self._default_units[legacy_key] = spec.default_unit
            if spec.include_self_in_input_aliases:
                self._input_aliases[spec.canonical] = spec.canonical
            for alias in (*spec.query_aliases, *spec.ingest_only_aliases):
                self._input_aliases[alias] = spec.canonical

    def canonicalize(self, raw: str) -> str:
        """Maps an input metric name or alias to its canonical sample_type."""
        return self._input_aliases.get(raw, raw)

    def aliases_for(self, metric_name: str) -> tuple[str, ...]:
        """Returns canonical + clean DB query aliases for a metric (excluding ingest typos)."""
        canonical = self.canonicalize(metric_name)
        spec = self._specs.get(canonical)
        if spec is None:
            return (canonical,)
        ordered = (canonical, *spec.query_aliases)
        return tuple(dict.fromkeys(ordered))

    def default_unit_for(self, metric_name: str) -> str | None:
        """Returns the default unit string for a metric name or canonical type."""
        if metric_name in self._default_units:
            return self._default_units[metric_name]
        return self._default_units.get(self.canonicalize(metric_name))

    def input_aliases(self) -> dict[str, str]:
        """Returns a copy of all ingest alias -> canonical mappings."""
        return dict(self._input_aliases)

    def default_units(self) -> dict[str, str]:
        """Returns a copy of all metric -> default unit mappings."""
        return dict(self._default_units)


_DEFAULT_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("active_energy", "kcal", query_aliases=("active_energy_kcal", "energy")),
    MetricSpec("basal_energy", "kcal"),
    MetricSpec("heart_rate", "bpm"),
    MetricSpec("resting_hr", "bpm", query_aliases=("resting_heart_rate", "resting_hr_bpm")),
    MetricSpec("hrv", "ms", query_aliases=("heart_rate_variability", "hrv_sdnn", "hrv_ms")),
    MetricSpec(
        "sleep",
        "s",
        query_aliases=("sleep_analysis", "sleep_duration", "sleep_duration_s"),
        legacy_unit_keys=("sleep_duration",),
    ),
    MetricSpec("step_count", "count", query_aliases=("steps", "step")),
    MetricSpec("weight", "kg", query_aliases=("weight_kg",)),
    MetricSpec(
        "body_fat_pct",
        "%",
        query_aliases=("body_fat_percentage", "body_fat"),
        legacy_unit_keys=("body_fat",),
    ),
    MetricSpec("muscle_mass_kg", "kg", query_aliases=("muscle_mass", "lean_body_mass")),
    MetricSpec("bone_mass_kg", "kg", query_aliases=("bone_mass",)),
    MetricSpec("water_pct", "%", query_aliases=("water_percentage", "hydration_pct")),
    MetricSpec("distance", "m", query_aliases=("distance_meters",)),
    MetricSpec("flights", "count"),
    MetricSpec("vo2_max", "ml/kg/min", query_aliases=("cardio_fitness", "vo2max")),
    MetricSpec("oxygen_saturation", "%", query_aliases=("spo2", "blood_oxygen")),
    MetricSpec("body_temperature", "C"),
    MetricSpec("wrist_temperature", "C", query_aliases=("wrist_temp", "wrist_temperature_c")),
    MetricSpec(
        "respiratory_rate",
        "breaths/min",
        query_aliases=("sleeping_respiratory_rate", "respiratory_rate_bpm", "respiration_rate"),
    ),
    MetricSpec(
        "walking_asymmetry",
        "%",
        query_aliases=("walking asymmetry", "walking_asymmetry_percentage", "asymmetry"),
        ingest_only_aliases=("walking_ asymmetry",),
        include_self_in_input_aliases=True,
    ),
)

METRIC_REGISTRY = MetricAliasRegistry(_DEFAULT_SPECS)
