from __future__ import annotations
import os
import numpy as np
import pandas as pd

def median_grid(times, values, start, count, low=-np.inf, high=np.inf):
    t = np.asarray(times, float)
    v = np.asarray(values, float)
    out = np.full(count, np.nan)
    valid = np.isfinite(t) & np.isfinite(v) & (v >= low) & (v <= high)
    bins = np.floor((t[valid] - start) / 10).astype(int)
    keep = (bins >= 0) & (bins < count)
    if keep.any():
        grouped = pd.DataFrame({'bin': bins[keep], 'value': v[valid][keep]}).groupby('bin').value.median()
        out[grouped.index.to_numpy(int)] = grouped.to_numpy(float)
    return out

def run_length(mask):
    a = np.asarray(mask, bool)
    if not a.any():
        return 0
    edges = np.flatnonzero(np.diff(np.r_[False, a, False].astype(int)))
    return int(np.max(edges[1::2] - edges[::2]))

def low_plateau(values):
    a = np.asarray(values, float)
    for i in range(max(0, len(a) - 5)):
        block = a[i:i + 6]
        if np.isfinite(block).all() and np.median(block) < 40 and (np.ptp(block) <= 2):
            return True
    return False

def numeric_screen(mbp, sbp, dbp, first_valid):
    end = min(len(mbp), first_valid + 60)
    m, s, d = (np.asarray(a, float)[first_valid:end] for a in [mbp, sbp, dbp])
    paired = np.isfinite(m) & np.isfinite(s) & np.isfinite(d)
    return {'low_plateau_flag': low_plateau(m), 'low_40_run_sec': run_length(np.isfinite(m) & (m < 40)) * 10, 'large_jump_flag': bool(np.any(np.isfinite(m[:-1]) & np.isfinite(m[1:]) & (np.abs(np.diff(m)) >= 30))), 'low_pulse_pressure_flag': run_length(paired & (s - d >= 0) & (s - d <= 5)) >= 6, 'inverted_sbp_dbp_flag': run_length(paired & (s < d)) >= 2, 'map_bounds_flag': run_length(paired & ((m < d - 5) | (m > s + 5))) >= 6, 'paired_pressure_bins': int(paired.sum())}

def decode_waveform_time(sample_count, interval, start=0, first_index=0):
    if not np.isfinite(interval) or interval <= 0 or interval > 1:
        raise ValueError('invalid waveform interval')
    return start + np.arange(first_index, first_index + sample_count) * interval
