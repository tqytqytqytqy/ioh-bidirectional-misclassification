"""Frozen audit definitions. No filesystem access or reference filling."""
import re
import numpy as np
import pandas as pd

FAMILIES = ("ABP", "ART", "FAP", "Ao")


def numeric_alias(name):
    m = re.fullmatch(r"((?:ABP|ART|FAP|Ao)\d*)([msd]) \[mmHg\]", name, re.I)
    return (wave_alias(m[1]), m[2].lower()) if m else None


def wave_alias(name):
    m = re.fullmatch(r"(ABP|ART|FAP|Ao)(\d*)", name, re.I)
    if not m:
        return None
    base = {f.upper():f for f in FAMILIES}[m[1].upper()]
    return base + m[2]


def family_rank(family):
    base = re.sub(r"\d+$", "", family)
    return FAMILIES.index(base), family


def map_valid(values):
    a = np.asarray(values, dtype=float)
    return np.isfinite(a) & (a >= 20) & (a <= 180)


def bin_medians(ticks, values, counter_freq, duration):
    if not np.isfinite(counter_freq) or counter_freq <= 0:
        raise ValueError("Missing/invalid counter frequency")
    t = np.asarray(ticks, dtype=float) / counter_freq
    v = np.asarray(values, dtype=float)
    n = int(np.floor(duration / 10))
    keep = np.isfinite(t) & np.isfinite(v) & (t >= 0) & (t < n * 10)
    bins = np.full(n, np.nan)
    if keep.any():
        s = pd.Series(v[keep]).groupby(np.floor(t[keep] / 10).astype(int)).median()
        bins[s.index.to_numpy()] = s.to_numpy()
    return bins, {
        "finite_samples": int(np.isfinite(v).sum()),
        "negative_time_samples": int((np.isfinite(v) & (t < 0)).sum()),
        "outside_header_samples": int((np.isfinite(v) & (t >= duration)).sum()),
        "partial_end_bin_samples_excluded": int((np.isfinite(v) & (t >= n * 10) & (t < duration)).sum()),
        "in_full_bins_samples": int(keep.sum()),
        "raw_map_outside_20_180_samples": int((keep & ~map_valid(v)).sum()),
    }


def mask_runs(mask):
    a = np.asarray(mask, dtype=bool)
    d = np.diff(np.r_[False, a, False].astype(int))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def coverage(values, duration, valid=None):
    a = np.asarray(values, dtype=float)
    ok = map_valid(a) if valid is None else np.asarray(valid, dtype=bool)
    n = len(ok)
    count = int(ok.sum())
    starts, ends = mask_runs(ok)
    r = {
        "duration_seconds": float(duration), "full_bins": n, "valid_bins": count,
        "missing_bins": int((~np.isfinite(a)).sum()),
        "invalid_present_bins": int((np.isfinite(a) & ~ok).sum()),
        "valid_fraction": count / n if n else 0.,
        "valid_bin_hours": count / 360.,
        "whole_eligible": bool(duration >= 3600 and n >= 360 and count / n >= .8),
        "longest_no_missing_valid_seconds": int(max(ends-starts, default=0) * 10),
        "window_eligible": False, "best_hour_start_bin": None,
        "best_hour_valid_bins": 0, "best_hour_valid_fraction": None,
        "earliest_eligible_hour_start_bin": None,
        "eligible_sliding_hour_windows": 0,
        "eligible_nonoverlap_hour_windows": 0,
        "nonoverlap_hour_grid_denominator": n // 360,
    }
    if n >= 360:
        cs = np.r_[0, np.cumsum(ok)]
        wins = cs[360:] - cs[:-360]
        best = int(np.argmax(wins))
        eligible = np.flatnonzero(wins >= 288)
        r.update({
            "window_eligible": bool(len(eligible)), "best_hour_start_bin": best,
            "best_hour_valid_bins": int(wins[best]),
            "best_hour_valid_fraction": float(wins[best] / 360),
            "earliest_eligible_hour_start_bin": int(eligible[0]) if len(eligible) else None,
            "eligible_sliding_hour_windows": int(len(eligible)),
            "eligible_nonoverlap_hour_windows": int((ok[:n//360*360].reshape(-1,360).sum(axis=1) >= 288).sum()),
        })
    return r


def choose_family(bins):
    return min(bins, key=lambda f: (-int(map_valid(bins[f]).sum()), family_rank(f))) if bins else None


def coherent(mean, systolic, diastolic):
    m, s, d = (np.asarray(x, dtype=float) for x in (mean, systolic, diastolic))
    return map_valid(m) & np.isfinite(s) & np.isfinite(d) & (d >= 20) & (d <= m) & (m <= s) & (s <= 250) & (s-d >= 10) & (s-d <= 150)


def low_runs(values):
    a = np.asarray(values, dtype=float)
    low = map_valid(a) & (a < 65)
    starts, ends = mask_runs(low)
    lens = ends-starts
    return {"low_bins": int(low.sum()), "low_bin_hours": float(low.sum()/360),
            "observed_low_runs": int(len(lens)), "runs_ge60s": int((lens >= 6).sum()),
            "longest_run_seconds": int(max(lens, default=0)*10),
            "runs_ge60s_left_or_right_censored": int(sum((l >= 6) and (s == 0 or e == len(a) or not np.isfinite(a[s-1]) or not np.isfinite(a[e])) for s,e,l in zip(starts,ends,lens)))}


def interval_summary(ticks, counter_freq):
    t = np.asarray(ticks, dtype=float)
    t = t[np.isfinite(t)]
    u = np.unique(t)
    diff = np.diff(u) / counter_freq
    vals, counts = np.unique(np.round(diff, 6), return_counts=True)
    return {"timestamps": int(len(t)), "unique_timestamps": int(len(u)),
            "duplicate_timestamps": int(len(t)-len(u)),
            "backward_steps_file_order": int((np.diff(t) < 0).sum()),
            "positive_interval_counts_seconds": {str(float(v)):int(n) for v,n in zip(vals,counts)},
            "intervals_ge10s": int((diff >= 10).sum()),
            "intervals_ge60s": int((diff >= 60).sum()),
            "maximum_interval_seconds": float(max(diff, default=0))}


def age_at_year(anchor_age, anchor_year, year):
    return int(anchor_age) + int(year) - int(anchor_year)


def overlap_seconds(a, b, c, d):
    return max(0., min(b,d)-max(a,c))


def wave_quality(values, sample_rate):
    a = np.asarray(values, dtype=float)
    fin = np.isfinite(a)
    valid = fin & (a >= 20) & (a <= 250)
    stable = fin[1:] & fin[:-1] & (np.abs(np.diff(a)) <= .01)
    starts, ends = mask_runs(stable)
    flat = max(ends-starts+1, default=0) / sample_rate
    nsecs = int(len(a) / sample_rate)
    spread_ok = 0
    for s in range(nsecs):
        z = a[int(s*sample_rate):int((s+1)*sample_rate)]
        z = z[np.isfinite(z)]
        if len(z) >= .8*sample_rate and 10 <= np.ptp(z) <= 150:
            spread_ok += 1
    finite_fraction = float(fin.mean()) if len(a) else 0
    plausible_fraction = float(valid.mean()) if len(a) else 0
    return {"samples": int(len(a)), "finite_samples": int(fin.sum()),
            "finite_fraction": finite_fraction,
            "amplitude_20_250_fraction": plausible_fraction,
            "minimum": float(a[fin].min()) if fin.any() else None,
            "maximum": float(a[fin].max()) if fin.any() else None,
            "median": float(np.median(a[fin])) if fin.any() else None,
            "longest_flatline_seconds": float(flat), "flatline_ge5s": bool(flat >= 5),
            "one_second_spread_10_150_fraction": spread_ok/nsecs if nsecs else 0,
            "automated_plausible_block": bool(finite_fraction >= .8 and plausible_fraction >= .8 and flat < 5 and nsecs and spread_ok/nsecs >= .8)}
