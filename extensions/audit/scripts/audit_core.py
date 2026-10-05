"""Feasibility diagnostics, not clinical treatment or cuff-cycle inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reconcile_records(left, right):
    def normalise(frame):
        return frame[["time_sec", "value"]].dropna().drop_duplicates().sort_values(
            ["time_sec", "value"], kind="stable").to_numpy()
    a, b = normalise(left), normalise(right)
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-8)


def paired_timer_count(classified, pairs):
    c=classified.sort_values("time_sec")
    p=pairs.sort_values("display_time_sec")
    if len(c)!=len(p) or not np.allclose(c.time_sec,p.display_time_sec,rtol=0,atol=1e-8):
        raise ValueError("Classified events and paired event identities do not reconcile")
    return int(np.sum(c.reason.eq("timer_same_map").to_numpy() & p.paired.to_numpy(dtype=bool)))


def clean_track(frame, start, end):
    work = frame.iloc[:, :2].copy()
    work.columns = ["time_sec", "value"]
    work = work.apply(pd.to_numeric, errors="coerce")
    conflicts = int(work.groupby("time_sec")["value"].nunique().gt(1).sum())
    backward = int(np.sum(np.diff(work.time_sec.dropna().to_numpy()) < 0))
    duplicate_rows = int(work.duplicated(["time_sec", "value"]).sum())
    work = work.dropna().sort_values("time_sec", kind="stable")
    # Conflicting timestamps are retained in the source but excluded from diagnostics.
    bad_times = work.groupby("time_sec").value.nunique()
    work = work[~work.time_sec.isin(bad_times[bad_times.gt(1)].index)]
    work = work.drop_duplicates("time_sec")
    work = work[work.time_sec.between(start, end)].copy()
    work["time_sec"] -= start
    return work.reset_index(drop=True), {
        "conflicting_timestamps": conflicts,
        "backward_steps_source": backward,
        "exact_duplicate_source_rows": duplicate_rows,
    }


def classify_legacy(times, values, gap_sec=120, hold_sec=90):
    rows, last_time, last_value = [], None, None
    for time, value in zip(times, values):
        first = last_time is None
        changed = first or abs(value - last_value) >= 1
        gap = np.inf if first else time - last_time
        if first or changed or (gap >= gap_sec and gap >= hold_sec):
            reason = "initial_record" if first else "map_value_change" if changed else "timer_same_map"
            rows.append((time, value, reason))
            last_time, last_value = time, value
    return pd.DataFrame(rows, columns=["time_sec", "value", "reason"])


def nearest_distance(query, reference):
    query, reference = np.asarray(query), np.sort(np.asarray(reference))
    if not len(reference):
        return np.full(len(query), np.inf)
    i = np.searchsorted(reference, query)
    lo, hi = np.clip(i - 1, 0, len(reference) - 1), np.clip(i, 0, len(reference) - 1)
    return np.minimum(np.abs(query-reference[lo]), np.abs(query-reference[hi]))


def triplet_updates(map_track, sbp_track, dbp_track, tolerance=2):
    if map_track.empty or sbp_track.empty or dbp_track.empty:
        return pd.DataFrame(), {"complete_rows": 0, "ordered_rows": 0}
    joint = map_track.rename(columns={"value": "map"})
    for name, sub in [("sbp", sbp_track), ("dbp", dbp_track)]:
        right = sub.rename(columns={"value": name, "time_sec": name+"_time"})
        joint = pd.merge_asof(joint, right, left_on="time_sec", right_on=name+"_time",
                              direction="nearest", tolerance=tolerance)
    complete = joint[["map", "sbp", "dbp"]].notna().all(axis=1)
    ordered = complete & joint.dbp.le(joint["map"]) & joint["map"].le(joint.sbp)
    joint = joint[complete].copy()
    delta = joint[["map", "sbp", "dbp"]].diff().abs()
    change = delta.ge(1).any(axis=1)
    initial = pd.Series(False, index=joint.index)
    if len(joint):
        initial.iloc[0] = True
    joint["any_change"] = change
    joint["map_change"] = delta["map"].ge(1)
    joint["sd_only_change"] = change & ~joint.map_change
    joint["initial"] = initial
    joint["after_record_gap"] = joint.time_sec.diff().gt(10)
    return joint, {"complete_rows": int(complete.sum()), "ordered_rows": int(ordered.sum())}


def pair_support(query_times, art_times, art_values, before=-30, after=30):
    order = np.argsort(art_times)
    t, v = np.asarray(art_times)[order], np.asarray(art_values)[order]
    out = []
    expected = int(np.floor((after-before)/10))+1
    for time in query_times:
        l, r = np.searchsorted(t, [time+before, time+after], side="left")
        r = np.searchsorted(t, time+after, side="right")
        n = int(np.isfinite(v[l:r]).sum())
        out.append(n/expected >= .8 and n > 0)
    return np.asarray(out, dtype=bool)


def supported_duration(times, duration, max_gap=5):
    times = np.asarray(times)
    if not len(times):
        return 0.
    gaps = np.diff(np.r_[times, duration])
    return float(np.clip(gaps, 0, max_gap).sum())


def rate_transitions(frame, continuity_sec=5):
    if frame.empty:
        return pd.DataFrame(columns=["time_sec", "kind"])
    work = frame[frame.value.ge(0)].copy()
    previous = work.value.shift()
    continuous = work.time_sec.diff().between(0, continuity_sec, inclusive="right")
    starts = continuous & previous.eq(0) & work.value.gt(0)
    increases = continuous & previous.gt(0) & work.value.gt(previous+1e-8)
    out = work.loc[starts | increases, ["time_sec"]].copy()
    out["kind"] = np.where(starts.loc[out.index], "observed_zero_to_positive", "observed_rate_increase")
    return out.reset_index(drop=True)
