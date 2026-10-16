from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Episode:
    start_sec: float
    end_sec: float
    start_index: int
    end_index: int
    duration_sec: float


def detect_reference_episodes(
    times_sec,
    reference_map,
    threshold: float,
    min_duration_sec: float = 0.0,
    gap_tolerance_sec: float = 0.0,
) -> list[Episode]:
    times = np.asarray(times_sec, dtype=float)
    values = np.asarray(reference_map, dtype=float)
    below = np.isfinite(values) & (values < float(threshold))
    episodes: list[Episode] = []
    start: int | None = None
    last_low: int | None = None
    for idx, flag in enumerate(below):
        if bool(flag):
            if start is None:
                start = idx
            last_low = idx
            continue
        if start is not None and last_low is not None:
            gap = float(times[idx] - times[last_low]) if idx < len(times) else np.inf
            if gap <= float(gap_tolerance_sec):
                continue
            duration = float(times[last_low] - times[start] + _median_step(times))
            if duration >= float(min_duration_sec):
                episodes.append(Episode(float(times[start]), float(times[last_low]), start, last_low + 1, duration))
            start = None
            last_low = None
    if start is not None and last_low is not None:
        duration = float(times[last_low] - times[start] + _median_step(times))
        if duration >= float(min_duration_sec):
            episodes.append(Episode(float(times[start]), float(times[last_low]), start, last_low + 1, duration))
    return episodes


def episode_documented(episode: Episode, times_sec, display_map, threshold: float, grace_after_sec: float = 0.0) -> bool:
    times = np.asarray(times_sec, dtype=float)
    display = np.asarray(display_map, dtype=float)
    in_window = (times >= episode.start_sec) & (times <= episode.end_sec + float(grace_after_sec))
    return bool(np.any(np.isfinite(display[in_window]) & (display[in_window] < float(threshold))))


def _median_step(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 0.0

