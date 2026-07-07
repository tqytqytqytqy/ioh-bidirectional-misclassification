from __future__ import annotations

import numpy as np


def _valid_numeric(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def decompose_deficit(reference_map, display_map, threshold: float, dt_min: float) -> dict[str, float]:
    """Pointwise hidden/overdisplay decomposition for a reference/display MAP pair."""
    reference = _valid_numeric(reference_map)
    display = _valid_numeric(display_map)
    if reference.shape != display.shape:
        raise ValueError("reference_map and display_map must have the same shape")
    valid_reference = np.isfinite(reference)
    visible_display = np.isfinite(display)
    r = np.where(valid_reference, np.maximum(float(threshold) - reference, 0.0), 0.0)
    d = np.where(visible_display, np.maximum(float(threshold) - display, 0.0), 0.0)
    hidden = np.maximum(r - d, 0.0)
    overdisplay = np.maximum(d - r, 0.0)
    concordant = np.minimum(r, d)
    true_auc = float(np.nansum(r) * float(dt_min))
    display_auc = float(np.nansum(d) * float(dt_min))
    hidden_auc = float(np.nansum(hidden) * float(dt_min))
    overdisplay_auc = float(np.nansum(overdisplay) * float(dt_min))
    concordant_auc = float(np.nansum(concordant) * float(dt_min))
    normotensive_display_min = float(np.sum((r > 0) & (d == 0)) * float(dt_min))
    hypotensive_display_discordance_min = float(np.sum((r == 0) & (d > 0)) * float(dt_min))
    true_hypotension_min = float(np.sum(r > 0) * float(dt_min))
    return {
        "threshold": float(threshold),
        "true_auc": true_auc,
        "display_auc": display_auc,
        "hidden_auc": hidden_auc,
        "overdisplay_auc": overdisplay_auc,
        "concordant_auc": concordant_auc,
        "normotensive_display_min": normotensive_display_min,
        "normotensive_display_discordance_min": normotensive_display_min,
        "hypotensive_display_discordance_min": hypotensive_display_discordance_min,
        "true_hypotension_min": true_hypotension_min,
        "hidden_deficit_ratio": hidden_auc / true_auc if true_auc > 0 else np.nan,
        "overdisplay_deficit_ratio": overdisplay_auc / true_auc if true_auc > 0 else np.nan,
        "net_bias_ratio": (display_auc - true_auc) / true_auc if true_auc > 0 else np.nan,
    }


def emulate_last_visible(times_sec, values, interval_sec: float, offset_sec: float, carry_forward_limit_sec: float | None = None) -> np.ndarray:
    """Sample a reference trajectory and forward-fill the last visible value."""
    times = np.asarray(times_sec, dtype=float)
    vals = np.asarray(values, dtype=float)
    out = np.full(len(times), np.nan, dtype=float)
    if len(times) == 0 or interval_sec <= 0:
        return out
    sample_times = np.arange(times[0] + float(offset_sec), times[-1] + 1e-9, float(interval_sec))
    samples: list[tuple[float, float]] = []
    for sample_time in sample_times:
        idx = int(np.searchsorted(times, sample_time, side="left"))
        if idx >= len(times):
            idx = len(times) - 1
        if idx > 0 and abs(times[idx - 1] - sample_time) <= abs(times[idx] - sample_time):
            idx -= 1
        if np.isfinite(vals[idx]):
            samples.append((float(times[idx]), float(vals[idx])))
    sample_i = 0
    current = np.nan
    current_time = np.nan
    for i, time in enumerate(times):
        while sample_i < len(samples) and samples[sample_i][0] <= time + 1e-9:
            current_time, current = samples[sample_i]
            sample_i += 1
        if carry_forward_limit_sec is not None and np.isfinite(current_time) and time - current_time > float(carry_forward_limit_sec):
            out[i] = np.nan
        else:
            out[i] = current
    return out


def display_from_event_times(times_sec, event_times_sec, event_values, carry_forward_limit_sec: float | None = None) -> np.ndarray:
    """Forward-fill display values from actual event times onto a regular time grid."""
    times = np.asarray(times_sec, dtype=float)
    ev_times = np.asarray(event_times_sec, dtype=float)
    ev_values = np.asarray(event_values, dtype=float)
    out = np.full(len(times), np.nan, dtype=float)
    order = np.argsort(ev_times)
    ev_times = ev_times[order]
    ev_values = ev_values[order]
    event_i = 0
    current = np.nan
    current_time = np.nan
    for i, time in enumerate(times):
        while event_i < len(ev_times) and ev_times[event_i] <= time + 1e-9:
            if np.isfinite(ev_values[event_i]):
                current = ev_values[event_i]
                current_time = ev_times[event_i]
            event_i += 1
        if carry_forward_limit_sec is not None and np.isfinite(current_time) and time - current_time > float(carry_forward_limit_sec):
            out[i] = np.nan
        else:
            out[i] = current
    return out
