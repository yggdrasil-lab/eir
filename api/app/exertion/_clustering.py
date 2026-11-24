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


def cluster_active_slots(slots: tuple[SampleSlot, ...]) -> list[tuple[int, int]]:
    """Groups contiguous or short-gap exertion slots into candidate workout index ranges."""
    active_indices = [i for i, slot in enumerate(slots) if is_exertion_slot(slot)]
    if not active_indices:
        return []
    clusters: list[tuple[int, int]] = []
    curr = [active_indices[0]]
    for idx in active_indices[1:]:
        if (idx - curr[-1] <= 3) and not _gap_splits_cluster(slots, curr, idx):
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
    is_sustained_walk = duration_mins >= 25 and block_steps >= Decimal(1500) and (avg_hr is None or avg_hr >= 90)
    is_resistance = duration_mins >= 30 and block_energy >= Decimal(120) and (hr_spread >= 20 or (peak_hr is not None and peak_hr >= 115))
    is_swim = duration_mins >= 20 and block_energy >= Decimal(100) and (avg_hr is not None and avg_hr >= 110) and cadence < 25.0
    return is_high_intensity or is_sustained_walk or is_resistance or is_swim
