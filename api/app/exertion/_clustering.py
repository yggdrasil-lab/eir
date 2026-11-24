"""Active slot clustering and cluster noise filtering for daily workout detection."""

from decimal import Decimal

from app.exertion._slots import is_exertion_slot
from app.exertion._types import SampleSlot


def _gap_splits_cluster(slots: tuple[SampleSlot, ...], curr: list[int], next_idx: int) -> bool:
    """Returns True when the inter-slot gap contains >= 10m stationary rest or transit walking."""
    curr_total_steps = sum((slots[c].steps for c in curr), Decimal(0))
    curr_cadence = float(curr_total_steps) / (len(curr) * 5) if curr else 0.0
    consecutive_stationary = 0
    for g_idx in range(curr[-1] + 1, next_idx):
        g_steps = slots[g_idx].steps
        if g_steps < Decimal(50):
            consecutive_stationary += 1
            if consecutive_stationary >= 2:
                return True
        else:
            consecutive_stationary = 0
        if (curr_cadence < 25.0 and g_steps >= Decimal(250)) or (g_steps >= Decimal(500)):
            return True
    return False


def _is_swim_prefix(slots: tuple[SampleSlot, ...], curr: list[int]) -> bool:
    if len(curr) < 4:
        return False
    total_steps = sum((slots[idx].steps for idx in curr), Decimal(0))
    cadence = float(total_steps) / (len(curr) * 5)
    if cadence >= 20.0:
        return False
    minimal_count = sum(1 for idx in curr if slots[idx].steps < Decimal(25))
    return minimal_count >= 4 or (minimal_count / len(curr)) >= 0.6


def _is_sustained_walk_suffix(
    slots: tuple[SampleSlot, ...],
    active_indices: list[int],
    start_pos: int,
) -> bool:
    run = [active_indices[start_pos]]
    for pos in range(start_pos + 1, len(active_indices)):
        prev_idx = run[-1]
        next_idx = active_indices[pos]
        if next_idx - prev_idx > 3 or _gap_splits_cluster(slots, [prev_idx], next_idx):
            break
        run.append(next_idx)
    if len(run) < 2:
        return False
    run_steps = sum((slots[idx].steps for idx in run), Decimal(0))
    run_cadence = float(run_steps) / (len(run) * 5)
    return run_steps >= Decimal(500) and run_cadence >= 35.0


def _swim_transitions_to_walk(
    slots: tuple[SampleSlot, ...],
    active_indices: list[int],
    pos: int,
    curr: list[int],
) -> bool:
    if not _is_swim_prefix(slots, curr):
        return False
    prev_slot = slots[curr[-1]]
    next_slot = slots[active_indices[pos]]
    watch_dropped = len(prev_slot.hr_values) >= 15 and len(next_slot.hr_values) < 15
    walk_entry = next_slot.steps >= Decimal(150) or (watch_dropped and next_slot.steps >= Decimal(100))
    if not walk_entry:
        return False
    return _is_sustained_walk_suffix(slots, active_indices, pos)


def _walk_transitions_to_swim(
    slots: tuple[SampleSlot, ...],
    active_indices: list[int],
    pos: int,
    curr: list[int],
) -> bool:
    next_idx = active_indices[pos]
    next_slot = slots[next_idx]
    if next_slot.steps >= Decimal(50) or len(next_slot.hr_values) < 15:
        return False
    gap_steps = sum((slots[g].steps for g in range(curr[-1] + 1, next_idx)), Decimal(0))
    curr_steps = sum((slots[c].steps for c in curr), Decimal(0))
    if curr_steps < Decimal(200) and gap_steps < Decimal(150):
        return False
    if sum(1 for c in curr if slots[c].steps < Decimal(25)) >= 3:
        return False
    lookahead = active_indices[pos : pos + 5]
    swim_slots = sum(
        1 for idx in lookahead
        if slots[idx].steps < Decimal(25) and len(slots[idx].hr_values) >= 15
    )
    return swim_slots >= 4


def cluster_active_slots(slots: tuple[SampleSlot, ...]) -> list[tuple[int, int]]:
    """Groups contiguous or short-gap exertion slots into candidate workout index ranges."""
    active_indices = [i for i, slot in enumerate(slots) if is_exertion_slot(slot)]
    if not active_indices:
        return []
    clusters: list[tuple[int, int]] = []
    curr = [active_indices[0]]
    for pos in range(1, len(active_indices)):
        idx = active_indices[pos]
        can_join = (
            (idx - curr[-1] <= 3)
            and not _gap_splits_cluster(slots, curr, idx)
            and not _swim_transitions_to_walk(slots, active_indices, pos, curr)
            and not _walk_transitions_to_swim(slots, active_indices, pos, curr)
        )
        if can_join:
            curr.append(idx)
            continue
        clusters.append((curr[0], curr[-1]))
        curr = [idx]
    clusters.append((curr[0], curr[-1]))
    return clusters


def passes_cluster_noise_filter(
    duration_mins: int,
    block_energy: Decimal,
    block_steps: Decimal,
    all_hrs: list[float],
    avg_hr: int | None,
    peak_hr: int | None,
) -> bool:
    """Filters out transient activity spikes that do not meet minimum workout thresholds."""
    hr_spread = (max(all_hrs) - min(all_hrs)) if all_hrs else 0.0
    cadence = float(block_steps) / duration_mins if duration_mins > 0 else 0.0
    is_high_intensity = duration_mins >= 30 and (block_energy >= Decimal(250) or (avg_hr is not None and avg_hr >= 120))
    is_sustained_walk = duration_mins >= 15 and block_steps >= Decimal(900) and (avg_hr is None or avg_hr >= 90)
    is_resistance = duration_mins >= 30 and block_energy >= Decimal(120) and (hr_spread >= 20 or (peak_hr is not None and peak_hr >= 115))
    is_swim = duration_mins >= 20 and block_energy >= Decimal(100) and (avg_hr is not None and avg_hr >= 110) and cadence < 25.0
    return is_high_intensity or is_sustained_walk or is_resistance or is_swim
