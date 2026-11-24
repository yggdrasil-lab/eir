"""Calorie estimation, HR zone/TRIMP/RPE scoring, and post-exercise HR recovery."""

from collections.abc import Sequence
from datetime import datetime, timedelta, tzinfo

from app.exertion._slots import sample_local_dt
from app.exertion._types import ExertionScore
from app.models import HealthSample


def _select_met(classification: str | None, avg_hr: int) -> float:
    c = (classification or "").lower()
    if c in ("badminton", "sport"):
        if avg_hr >= 160:
            return 9.0
        return 8.0 if avg_hr >= 145 else 7.0
    if c in ("swim", "swimming"):
        return 7.5 if avg_hr >= 130 else 6.0
    if c in ("gym", "resistance"):
        return 5.5 if avg_hr >= 125 else 4.5
    if c in ("walk", "walking"):
        return 4.2 if avg_hr >= 105 else 3.5
    if avg_hr >= 150:
        return 8.0
    return 6.5 if avg_hr >= 130 else 5.0


def estimate_workout_active_calories(
    duration_mins: int,
    avg_hr: int | None,
    classification: str | None,
    weight_kg: float = 85.0,
) -> int:
    """Estimates net active calories when Apple Watch passive mode undercounts."""
    if not avg_hr or duration_mins <= 0:
        return 0
    met = _select_met(classification, avg_hr)
    net_active = (met - 1.0) * weight_kg * (duration_mins / 60.0)
    return int(round(net_active))


def _classify_effort_and_rpe(
    pct_z3: float,
    pct_z4: float,
    pct_z5: float,
    avg_hr: float,
    duration_mins: int,
) -> tuple[str, float]:
    high_intensity_pct = pct_z4 + pct_z5
    if (pct_z5 >= 50.0 and avg_hr >= 160.0) or (
        pct_z5 >= 40.0 and duration_mins >= 50 and avg_hr >= 160.0
    ):
        return "too_hard", round(min(10.0, 9.0 + (avg_hr - 160.0) * 0.1), 1)
    if pct_z5 >= 35.0 or high_intensity_pct >= 60.0 or avg_hr >= 150.0:
        return "hard", round(min(8.9, 8.0 + (pct_z5 / 100.0) * 1.5), 1)
    if high_intensity_pct >= 15.0 or pct_z3 >= 30.0 or avg_hr >= 120.0:
        return "optimal", round(min(7.9, 5.0 + (pct_z3 + pct_z4) / 100.0 * 2.5), 1)
    return "too_easy", round(max(1.0, 2.0 + (avg_hr - 60.0) * 0.05), 1)


def compute_workout_exertion_metrics(
    all_hrs: Sequence[float],
    duration_mins: int,
    max_hr: int = 186,
) -> ExertionScore:
    """Computes HR zones from max_hr, Edwards TRIMP, RPE, and effort rating."""
    if not all_hrs:
        return ExertionScore(effort_rating="optimal", rpe=6.0, trimp=0.0, hr_zones=None)

    z1_max = float(round(max_hr * 0.60))
    z2_max = float(round(max_hr * 0.70))
    z3_max = float(round(max_hr * 0.80))
    z4_max = float(round(max_hr * 0.90))

    total_pts = len(all_hrs)
    avg_hr = sum(all_hrs) / total_pts
    counts = (
        sum(1 for h in all_hrs if h < z1_max),
        sum(1 for h in all_hrs if z1_max <= h < z2_max),
        sum(1 for h in all_hrs if z2_max <= h < z3_max),
        sum(1 for h in all_hrs if z3_max <= h < z4_max),
        sum(1 for h in all_hrs if h >= z4_max),
    )
    pcts = tuple(round((c / total_pts) * 100.0, 1) for c in counts)
    mins = tuple(round((c / total_pts) * duration_mins, 1) for c in counts)
    trimp = round(sum(m * weight for m, weight in zip(mins, (1.0, 2.0, 3.0, 4.0, 5.0))), 1)
    effort_rating, rpe = _classify_effort_and_rpe(
        pcts[2], pcts[3], pcts[4], avg_hr, duration_mins
    )
    hr_zones = {
        "z1_recovery_pct": pcts[0],
        "z2_aerobic_pct": pcts[1],
        "z3_tempo_pct": pcts[2],
        "z4_threshold_pct": pcts[3],
        "z5_anaerobic_pct": pcts[4],
        "z1_mins": mins[0],
        "z2_mins": mins[1],
        "z3_mins": mins[2],
        "z4_mins": mins[3],
        "z5_mins": mins[4],
    }
    return ExertionScore(effort_rating=effort_rating, rpe=rpe, trimp=trimp, hr_zones=hr_zones)


def compute_hr_recovery_60s(
    samples: Sequence[HealthSample],
    e_time: datetime,
    all_hrs: Sequence[float],
    tz: tzinfo,
) -> int | None:
    """Computes post-workout heart rate drop within 90s after ended_at."""
    if not all_hrs:
        return None
    post_e_time = e_time + timedelta(seconds=90)
    post_hrs = [
        float(s.value)
        for s in samples
        if s.sample_type in ("heart_rate", "resting_hr")
        and e_time <= sample_local_dt(s, tz) <= post_e_time
    ]
    if not post_hrs:
        return None
    drop = all_hrs[-1] - min(post_hrs)
    return int(round(drop)) if drop > 0 else None
