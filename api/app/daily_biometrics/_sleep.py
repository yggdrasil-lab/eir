"""Pure sleep episode clustering, overnight/nap segmentation, and overnight HRV computation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from app.daily_biometrics._window import morning_hrv_bounds

if TYPE_CHECKING:
    from app.models import HealthSample

EXCLUDED_SLEEP_STAGES = frozenset({"awake", "inbed", "in_bed", "in bed"})
STAGED_SLEEP_LABELS = frozenset({"core", "deep", "rem"})
EPISODE_GAP = timedelta(minutes=60)
MIN_SLEEP_SECONDS = 900
HRV_PAD = timedelta(minutes=15)


@dataclass(frozen=True)
class _NormalizedSleepRow:
    start_at: datetime
    end_at: datetime
    asleep_s: int
    deep_s: int
    rem_s: int
    is_staged: bool


@dataclass(frozen=True)
class SleepEpisode:
    """Contiguous sleep episode aggregated from one or more sleep HealthSample rows."""

    start_at: datetime
    end_at: datetime
    sleep_s: int
    deep_s: int
    rem_s: int
    has_stages: bool


@dataclass(frozen=True)
class SegmentedSleepResult:
    """Day-attributed overnight sleep and distinct daytime nap episodes."""

    overnight: SleepEpisode | None
    naps: tuple[SleepEpisode, ...]
    nap_duration_s: int | None
    nap_count: int


def _sample_local_dt(dt_val: datetime, tz: ZoneInfo) -> datetime:
    """Normalizes DB-loaded datetimes to tz (naive SQLite rows already hold tz wall time)."""
    return dt_val.astimezone(tz) if dt_val.tzinfo is not None else dt_val.replace(tzinfo=tz)


def _normalize_sleep_sample(sample: HealthSample, tz: ZoneInfo) -> _NormalizedSleepRow:
    start = _sample_local_dt(sample.measured_at, tz)
    raw_end = _sample_local_dt(sample.ended_at, tz) if sample.ended_at is not None else start
    if raw_end > start:
        end = raw_end
    elif sample.value > 0:
        end = start + timedelta(seconds=float(sample.value))
    else:
        end = start

    stage = str((sample.extra or {}).get("stage") or "").strip().lower()
    val = max(0, int(round(sample.value)))
    asleep_s = 0 if stage in EXCLUDED_SLEEP_STAGES else val
    return _NormalizedSleepRow(
        start_at=start,
        end_at=end,
        asleep_s=asleep_s,
        deep_s=val if stage == "deep" else 0,
        rem_s=val if stage == "rem" else 0,
        is_staged=stage in STAGED_SLEEP_LABELS,
    )


def cluster_sleep_episodes(
    samples: Sequence[HealthSample],
    tz: ZoneInfo,
) -> tuple[SleepEpisode, ...]:
    """Clusters ordered sleep samples into contiguous episodes separated by >60m gaps."""
    if not samples:
        return ()

    rows = sorted((_normalize_sleep_sample(s, tz) for s in samples), key=lambda r: (r.start_at, r.end_at))
    episodes: list[SleepEpisode] = []
    cur_start = rows[0].start_at
    cur_end = rows[0].end_at
    sleep_s = rows[0].asleep_s
    deep_s = rows[0].deep_s
    rem_s = rows[0].rem_s
    has_stages = rows[0].is_staged

    for row in rows[1:]:
        if row.start_at - cur_end <= EPISODE_GAP:
            cur_end = max(cur_end, row.end_at)
            sleep_s += row.asleep_s
            deep_s += row.deep_s
            rem_s += row.rem_s
            has_stages = has_stages or row.is_staged
            continue
        episodes.append(SleepEpisode(cur_start, cur_end, sleep_s, deep_s, rem_s, has_stages))
        cur_start = row.start_at
        cur_end = row.end_at
        sleep_s = row.asleep_s
        deep_s = row.deep_s
        rem_s = row.rem_s
        has_stages = row.is_staged

    episodes.append(SleepEpisode(cur_start, cur_end, sleep_s, deep_s, rem_s, has_stages))
    return tuple(episodes)


def _is_nocturnal_candidate(episode: SleepEpisode, target_date: date, tz: ZoneInfo) -> bool:
    local_start = episode.start_at.astimezone(tz)
    local_end = episode.end_at.astimezone(tz)
    nocturnal_cutoff = datetime.combine(target_date, time(13, 0)).replace(tzinfo=tz)
    early_start = datetime.combine(target_date, time(0, 0)).replace(tzinfo=tz)
    early_end = datetime.combine(target_date, time(7, 0)).replace(tzinfo=tz)
    return local_start < nocturnal_cutoff or (local_start < early_end and local_end > early_start)


def _select_overnight_episode(
    episodes: Sequence[SleepEpisode],
    target_date: date,
    tz: ZoneInfo,
) -> SleepEpisode | None:
    candidates = [ep for ep in episodes if _is_nocturnal_candidate(ep, target_date, tz)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda ep: (
            ep.has_stages or (ep.deep_s + ep.rem_s) > 0,
            ep.sleep_s,
            -ep.start_at.timestamp(),
        ),
    )


def segment_sleep_day(
    samples: Sequence[HealthSample],
    target_date: date,
    tz: ZoneInfo,
) -> SegmentedSleepResult:
    """Attributes episodes ending on target_date and separates overnight sleep from daytime naps."""
    episodes = cluster_sleep_episodes(samples, tz)
    day_episodes = [
        ep
        for ep in episodes
        if ep.end_at.astimezone(tz).date() == target_date and ep.sleep_s >= MIN_SLEEP_SECONDS
    ]
    if not day_episodes:
        return SegmentedSleepResult(overnight=None, naps=(), nap_duration_s=None, nap_count=0)

    overnight = _select_overnight_episode(day_episodes, target_date, tz)
    nap_floor = (
        overnight.end_at
        if overnight is not None
        else datetime.combine(target_date, time(10, 0)).replace(tzinfo=tz)
    )
    naps = tuple(ep for ep in day_episodes if ep is not overnight and ep.start_at >= nap_floor)
    nap_duration_s = sum(ep.sleep_s for ep in naps) if naps else None
    return SegmentedSleepResult(
        overnight=overnight,
        naps=naps,
        nap_duration_s=nap_duration_s,
        nap_count=len(naps),
    )


def _mean_hrv_in_window(
    hrv_samples: Sequence[HealthSample],
    start_dt: datetime,
    end_dt: datetime,
    tz: ZoneInfo,
) -> Decimal | None:
    matched: list[Decimal] = []
    for sample in hrv_samples:
        sample_dt = _sample_local_dt(sample.measured_at, tz)
        if start_dt <= sample_dt <= end_dt:
            matched.append(Decimal(str(sample.value)))
    if not matched:
        return None
    mean_val = sum(matched, Decimal(0)) / Decimal(len(matched))
    return mean_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def compute_overnight_hrv(
    hrv_samples: Sequence[HealthSample],
    overnight: SleepEpisode | None,
    target_date: date,
    tz: ZoneInfo,
) -> Decimal | None:
    """Computes overnight HRV mean (+-15m sleep bounds), falling back to 00:00-10:00 local."""
    if not hrv_samples:
        return None
    if overnight is not None:
        overnight_mean = _mean_hrv_in_window(
            hrv_samples,
            overnight.start_at - HRV_PAD,
            overnight.end_at + HRV_PAD,
            tz,
        )
        if overnight_mean is not None:
            return overnight_mean
    m_start, m_end = morning_hrv_bounds(target_date, tz)
    return _mean_hrv_in_window(hrv_samples, m_start, m_end, tz)
