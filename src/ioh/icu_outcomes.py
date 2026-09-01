from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"case_id", "subjectid", "icu_days"}
ENDPOINT_LIMITATION = (
    "VitalDB icu_days cannot distinguish planned from unplanned ICU admission."
)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"ICU outcome source is missing columns: {missing}")


def build_prolonged_icu_outcome(
    frame: pd.DataFrame,
    *,
    threshold_days: int = 2,
) -> pd.DataFrame:
    """Build a nullable postoperative ICU-stay outcome at a frozen threshold."""

    _require_columns(frame)
    if threshold_days < 1:
        raise ValueError("threshold_days must be at least 1")
    if frame["case_id"].duplicated().any():
        raise ValueError("ICU outcome source must contain one row per case_id")

    output = frame[["case_id", "subjectid", "icu_days"]].copy()
    output["postoperative_icu_days"] = pd.to_numeric(
        output.pop("icu_days"), errors="coerce"
    )
    if output["postoperative_icu_days"].lt(0).any():
        raise ValueError("postoperative ICU duration contains negative values")

    evaluable = output["postoperative_icu_days"].notna()
    outcome = pd.Series(pd.NA, index=output.index, dtype="boolean")
    outcome.loc[evaluable] = output.loc[
        evaluable, "postoperative_icu_days"
    ].ge(threshold_days)
    output["prolonged_icu_ge2d"] = outcome
    output["outcome_evaluable"] = evaluable
    output["threshold_days"] = int(threshold_days)
    output["outcome_definition"] = (
        f"postoperative ICU length of stay >= {threshold_days} days"
    )
    output["endpoint_limitation"] = ENDPOINT_LIMITATION
    return output


def summarize_icu_gate(
    frame: pd.DataFrame,
    *,
    threshold_days: int = 2,
    min_completeness: float = 0.95,
    min_events: int = 100,
) -> pd.DataFrame:
    """Apply analysis-plan-defined structural feasibility rules to the ICU endpoint."""

    if not 0 < min_completeness <= 1:
        raise ValueError("min_completeness must be in (0, 1]")
    if min_events < 1:
        raise ValueError("min_events must be positive")

    outcome = build_prolonged_icu_outcome(
        frame, threshold_days=threshold_days
    )
    n_total = int(len(outcome))
    n_observed = int(outcome["outcome_evaluable"].sum())
    n_events = int(outcome["prolonged_icu_ge2d"].fillna(False).sum())
    completeness = n_observed / n_total if n_total else np.nan
    event_rate = n_events / n_observed if n_observed else np.nan

    completeness_pass = bool(
        n_total > 0 and np.isfinite(completeness) and completeness >= min_completeness
    )
    events_pass = n_events >= min_events
    overall_pass = completeness_pass and events_pass
    rows = [
        {
            "criterion": "outcome_completeness",
            "value": n_observed,
            "denominator": n_total,
            "proportion": completeness,
            "pass_rule": f">= {min_completeness:.0%}",
            "status": "PASS" if completeness_pass else "NOT_PASS",
            "limitation": "",
        },
        {
            "criterion": "event_count",
            "value": n_events,
            "denominator": n_observed,
            "proportion": event_rate,
            "pass_rule": f">= {min_events} events",
            "status": "PASS" if events_pass else "NOT_PASS",
            "limitation": "",
        },
        {
            "criterion": "nonnegative_duration",
            "value": 0,
            "denominator": n_observed,
            "proportion": 0.0 if n_observed else np.nan,
            "pass_rule": "zero negative ICU durations",
            "status": "PASS",
            "limitation": "",
        },
        {
            "criterion": "overall_gate",
            "value": n_events,
            "denominator": n_observed,
            "proportion": event_rate,
            "pass_rule": (
                f"completeness >= {min_completeness:.0%}; events >= {min_events}; "
                "no invalid durations"
            ),
            "status": (
                "PASS_WITH_ENDPOINT_LIMITATION"
                if overall_pass
                else "NOT_PASS"
            ),
            "limitation": ENDPOINT_LIMITATION,
        },
    ]
    return pd.DataFrame(rows)
