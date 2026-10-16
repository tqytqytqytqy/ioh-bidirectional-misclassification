import numpy as np


def held_display(values, step_bins, phase_bins):
    values = np.asarray(values, dtype=float)
    if step_bins < 1 or not 0 <= phase_bins < step_bins:
        raise ValueError('Invalid sampling schedule')
    updates = np.full(len(values), -1, dtype=int)
    scheduled = np.arange(phase_bins, len(values), step_bins)
    valid = scheduled[np.isfinite(values[scheduled])]
    updates[valid] = valid
    held = np.maximum.accumulate(updates)
    display = np.full(len(values), np.nan)
    available = held >= 0
    display[available] = values[held[available]]
    return display, held


def reference_events(values, threshold, gap_bins, min_low_bins):
    values = np.asarray(values, dtype=float)
    low = np.flatnonzero(np.isfinite(values) & (values < threshold))
    if not len(low):
        return []
    missing = np.r_[0, np.cumsum(~np.isfinite(values))]
    gaps = np.diff(low) - 1
    unknown = missing[low[1:]] - missing[low[:-1] + 1]
    splits = np.flatnonzero((gaps > gap_bins) | (unknown > 0)) + 1
    groups = np.split(low, splits)
    return [(int(g[0]), int(g[-1]) + 1, len(g)) for g in groups if len(g) >= min_low_bins]


def classify_events(values, display, held_index, events, threshold):
    values = np.asarray(values, dtype=float)
    visible = np.isfinite(display) & (display < threshold)
    fresh = (held_index == np.arange(len(values))) & visible
    vc = np.r_[0, np.cumsum(visible)]
    fc = np.r_[0, np.cumsum(fresh)]
    categories, carry_in, first_inherited = [], [], []
    for start, end, _ in events:
        any_visible = vc[end] > vc[start]
        any_fresh = fc[end] > fc[start]
        categories.append(0 if any_fresh else 1 if any_visible else 2)
        carry_in.append(bool(visible[start] and held_index[start] < start))
        first = start + int(np.argmax(visible[start:end])) if any_visible else -1
        first_inherited.append(bool(first >= 0 and held_index[first] < start))
    return {'category': np.asarray(categories, dtype=int),
            'carry_in': np.asarray(carry_in, dtype=bool),
            'first_inherited': np.asarray(first_inherited, dtype=bool)}
