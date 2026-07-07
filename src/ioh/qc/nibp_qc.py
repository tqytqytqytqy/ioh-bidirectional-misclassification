from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_nibp_display_events(
    raw_nibp: pd.DataFrame,
    same_value_hold_sec: float = 90.0,
    min_gap_new_event_sec: float = 120.0,
    plausible_range: tuple[float, float] = (20.0, 180.0),
) -> pd.DataFrame:
    """Collapse raw/forward-filled NIBP rows into display-level cuff events."""
    if raw_nibp.empty:
        return pd.DataFrame(columns=["case_id", "display_time_sec", "nibp_map"])
    rename = {}
    if "caseid" in raw_nibp.columns and "case_id" not in raw_nibp.columns:
        rename["caseid"] = "case_id"
    if "time" in raw_nibp.columns and "time_sec" not in raw_nibp.columns:
        rename["time"] = "time_sec"
    if "nibp_map" in raw_nibp.columns and "map" not in raw_nibp.columns:
        rename["nibp_map"] = "map"
    work = raw_nibp.rename(columns=rename)[["case_id", "time_sec", "map"]].copy()
    work["time_sec"] = pd.to_numeric(work["time_sec"], errors="coerce")
    work["map"] = pd.to_numeric(work["map"], errors="coerce")
    work = work.dropna(subset=["case_id", "time_sec", "map"])
    work = work[work["map"].between(float(plausible_range[0]), float(plausible_range[1]))]
    work = work.sort_values(["case_id", "time_sec", "map"]).drop_duplicates(["case_id", "time_sec", "map"])
    rows: list[dict] = []
    for case_id, sub in work.groupby("case_id", dropna=False):
        last_time = None
        last_value = None
        for row in sub.sort_values("time_sec").itertuples(index=False):
            time_sec = float(row.time_sec)
            value = float(row.map)
            first = last_time is None
            value_changed = last_value is None or abs(value - float(last_value)) >= 1.0
            gap = np.inf if last_time is None else time_sec - float(last_time)
            long_gap = gap >= float(min_gap_new_event_sec)
            stale_hold = (not value_changed) and gap < float(same_value_hold_sec)
            if first or value_changed or (long_gap and not stale_hold):
                rows.append({"case_id": case_id, "display_time_sec": time_sec, "nibp_map": value})
                last_time = time_sec
                last_value = value
    return pd.DataFrame(rows, columns=["case_id", "display_time_sec", "nibp_map"])


def retention_audit(raw: pd.DataFrame, display_events: pd.DataFrame) -> pd.DataFrame:
    raw_n = len(raw)
    valid_n = len(raw[pd.to_numeric(raw.get("map", raw.get("nibp_map")), errors="coerce").between(20, 180)]) if raw_n else 0
    event_n = len(display_events)
    return pd.DataFrame(
        [
            {
                "raw_records": raw_n,
                "candidate_records": valid_n,
                "display_events": event_n,
                "candidate_retention_rate": valid_n / raw_n if raw_n else np.nan,
                "display_retention_rate": event_n / raw_n if raw_n else np.nan,
            }
        ]
    )

