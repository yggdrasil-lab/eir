"""Exertion analysis deep module: workout detection, window scoring, and timelines."""

from app.exertion._analyzer import ExertionAnalyzer
from app.exertion._steps import deduplicate_step_samples, filter_deduplicated_step_samples
from app.exertion._types import ExertionProfile, InvalidWindowError

__all__ = [
    "ExertionAnalyzer",
    "ExertionProfile",
    "InvalidWindowError",
    "deduplicate_step_samples",
    "filter_deduplicated_step_samples",
]
