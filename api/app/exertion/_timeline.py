"""5-minute workout timeline binning and Watch mode transition split detection."""

from collections.abc import Sequence
from datetime import datetime, timedelta

from app.exertion._slots import is_exertion_slot
from app.exertion._types import SampleSlot
from app.schemas import SuggestedSplitRead, TimelineBinRead, WorkoutTimelineRead

ACTIVE_WATCH_HR_SAMPLES_MIN = 15


def build_timeline_bins(slots: Sequence[SampleSlot]) -> list[TimelineBinRead]:
    """Converts 5-minute biometric slots into timeline bins with Watch mode flags."""
    bins: list[TimelineBinRead] = []
    for slot in slots:
        hr_count = len(slot.hr_values)
        avg_hr = int(round(sum(slot.hr_values) / hr_count)) if hr_count else None
        peak_hr = int(round(max(slot.hr_values))) if hr_count else None
        bins.append(
            TimelineBinRead(
                started_at=slot.started_at,
                ended_at=slot.ended_at,
                avg_hr_bpm=avg_hr,
                peak_hr_bpm=peak_hr,
                hr_sample_count=hr_count,
                step_count=int(round(slot.steps)),
                active_energy_kcal=int(round(slot.energy)),
                is_active_watch_mode=hr_count >= ACTIVE_WATCH_HR_SAMPLES_MIN,
            )
        )
    return bins


def _is_transition_gap(slot: SampleSlot, bin_item: TimelineBinRead) -> bool:
    if bin_item.is_active_watch_mode:
        return False
    if not is_exertion_slot(slot):
        return True
    return (bin_item.avg_hr_bpm is None or bin_item.avg_hr_bpm < 105) and bin_item.active_energy_kcal < 20


def _within_split_bounds(
    split_at: datetime,
    resume_at: datetime,
    started_at: datetime,
    ended_at: datetime,
) -> bool:
    min_bound = started_at + timedelta(minutes=10)
    max_bound = ended_at - timedelta(minutes=10)
    return split_at >= min_bound and resume_at <= max_bound and split_at <= resume_at


def _resolve_passive_to_active(
    bins: Sequence[TimelineBinRead],
    gap_flags: Sequence[bool],
    idx: int,
) -> SuggestedSplitRead | None:
    split_idx = idx
    checked = 0
    while checked < 2 and split_idx - 1 >= 1 and gap_flags[split_idx - 1]:
        split_idx -= 1
        checked += 1
    if not any(not is_gap for is_gap in gap_flags[:split_idx]):
        return None
    reason = (
        "Active Watch workout mode started after passive transition gap"
        if split_idx < idx
        else "Active Watch workout mode started after passive monitoring"
    )
    return SuggestedSplitRead(
        split_at=bins[split_idx].started_at,
        resume_at=bins[idx].started_at,
        reason=reason,
        confidence="high",
    )


def _resolve_active_to_passive(
    bins: Sequence[TimelineBinRead],
    gap_flags: Sequence[bool],
    idx: int,
) -> tuple[SuggestedSplitRead | None, int]:
    gap_end = idx
    while gap_end < len(bins) and (gap_end - idx) < 2 and gap_flags[gap_end]:
        gap_end += 1

    if gap_end >= len(bins):
        return None, len(bins)

    if bins[gap_end].is_active_watch_mode:
        if gap_end == idx:
            return None, idx + 1
        split = SuggestedSplitRead(
            split_at=bins[idx].started_at,
            resume_at=bins[gap_end].started_at,
            reason="Active Watch workout mode resumed after transition gap",
            confidence="high",
        )
        return split, gap_end + 1

    if gap_end == idx:
        split = SuggestedSplitRead(
            split_at=bins[idx].started_at,
            resume_at=bins[idx].started_at,
            reason="Active Watch workout mode ended while passive exertion continued",
            confidence="high",
        )
        return split, idx + 1

    if not any(not is_gap for is_gap in gap_flags[gap_end:]):
        return None, len(bins)
    split = SuggestedSplitRead(
        split_at=bins[idx].started_at,
        resume_at=bins[gap_end].started_at,
        reason="Active Watch workout mode ended before transition gap",
        confidence="high",
    )
    return split, gap_end + 1


def suggest_timeline_splits(
    slots: Sequence[SampleSlot],
    bins: Sequence[TimelineBinRead],
    started_at: datetime,
    ended_at: datetime,
) -> list[SuggestedSplitRead]:
    """Detects passive<->active Watch mode transitions within [started_at + 10m, ended_at - 10m]."""
    if (ended_at - started_at) < timedelta(minutes=20) or len(bins) < 3:
        return []

    gap_flags = [_is_transition_gap(slot, bin_item) for slot, bin_item in zip(slots, bins)]
    suggestions: list[SuggestedSplitRead] = []
    idx = 1
    while idx < len(bins):
        prev_active = bins[idx - 1].is_active_watch_mode
        curr_active = bins[idx].is_active_watch_mode
        candidate: SuggestedSplitRead | None = None
        next_idx = idx + 1

        if not prev_active and curr_active:
            candidate = _resolve_passive_to_active(bins, gap_flags, idx)
        elif prev_active and not curr_active:
            candidate, next_idx = _resolve_active_to_passive(bins, gap_flags, idx)

        if candidate is not None and _within_split_bounds(
            candidate.split_at, candidate.resume_at, started_at, ended_at
        ):
            suggestions.append(candidate)
        idx = next_idx

    return suggestions


def build_workout_timeline(
    slots: Sequence[SampleSlot],
    started_at: datetime,
    ended_at: datetime,
) -> WorkoutTimelineRead:
    """Assembles a WorkoutTimelineRead from 5-minute slots and transition split hints."""
    duration_mins = max(1, int(round((ended_at - started_at).total_seconds() / 60)))
    bins = build_timeline_bins(slots)
    suggested_splits = suggest_timeline_splits(slots, bins, started_at, ended_at)
    return WorkoutTimelineRead(
        started_at=started_at,
        ended_at=ended_at,
        duration_minutes=duration_mins,
        bins=bins,
        suggested_splits=suggested_splits,
    )
