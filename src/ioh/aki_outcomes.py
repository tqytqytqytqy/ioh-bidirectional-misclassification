from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact


DAY_SEC = 24 * 60 * 60


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def derive_last_preoperative_creatinine(
    manifest: pd.DataFrame,
    labs: pd.DataFrame,
    *,
    lookback_days: int = 90,
    plausible_creatinine_range: tuple[float, float] = (0.1, 20.0),
) -> pd.DataFrame:
    """Return the latest creatinine at or before anaesthesia start."""

    _require_columns(manifest, {"case_id", "anestart"}, "manifest")
    _require_columns(labs, {"caseid", "dt", "name", "result"}, "labs")
    cases = manifest[["case_id", "anestart"]].copy()
    cases["anestart"] = pd.to_numeric(cases["anestart"], errors="coerce")
    creatinine = labs.loc[
        labs["name"].astype("string").str.lower().eq("cr"),
        ["caseid", "dt", "result"],
    ].copy()
    creatinine["dt"] = pd.to_numeric(creatinine["dt"], errors="coerce")
    creatinine["result"] = pd.to_numeric(creatinine["result"], errors="coerce")
    lower, upper = plausible_creatinine_range
    creatinine = creatinine[
        creatinine["result"].between(float(lower), float(upper))
    ].merge(
        cases,
        left_on="caseid",
        right_on="case_id",
        how="inner",
        validate="many_to_one",
    )
    eligible = creatinine[
        creatinine["dt"].le(creatinine["anestart"])
        & creatinine["dt"].ge(
            creatinine["anestart"] - int(lookback_days) * DAY_SEC
        )
    ]
    latest = (
        eligible.sort_values(["caseid", "dt"])
        .groupby("caseid", sort=False)
        .tail(1)
        .set_index("caseid")["result"]
    )
    output = cases[["case_id"]].copy().set_index("case_id")
    output = output.join(latest.rename("last_preop_cr_90d"))
    return output.reset_index()


def build_creatinine_aki_outcomes(
    manifest: pd.DataFrame,
    labs: pd.DataFrame,
    *,
    plausible_creatinine_range: tuple[float, float] = (0.1, 20.0),
) -> pd.DataFrame:
    """Construct postoperative creatinine-only KDIGO AKI through day 7.

    The primary baseline is the VitalDB clinical-information field ``preop_cr``.
    Postoperative observation begins strictly after ``aneend`` and ends at the
    earlier of discharge or seven days. Urine-output and renal-replacement
    criteria are unavailable and are therefore not imputed.
    """

    _require_columns(
        manifest,
        {"case_id", "subjectid", "aneend", "dis", "preop_cr"},
        "manifest",
    )
    _require_columns(labs, {"caseid", "dt", "name", "result"}, "labs")

    cases = manifest[
        ["case_id", "subjectid", "aneend", "dis", "preop_cr"]
    ].copy()
    if cases["case_id"].duplicated().any():
        raise ValueError("manifest must contain one row per case_id")
    for column in ("aneend", "dis", "preop_cr"):
        cases[column] = pd.to_numeric(cases[column], errors="coerce")

    creatinine = labs.loc[
        labs["name"].astype("string").str.lower().eq("cr"),
        ["caseid", "dt", "result"],
    ].copy()
    creatinine["dt"] = pd.to_numeric(creatinine["dt"], errors="coerce")
    creatinine["result"] = pd.to_numeric(creatinine["result"], errors="coerce")
    lower, upper = plausible_creatinine_range
    creatinine = creatinine[
        creatinine["result"].between(float(lower), float(upper))
    ]
    creatinine = creatinine.merge(
        cases[["case_id", "aneend", "dis"]],
        left_on="caseid",
        right_on="case_id",
        how="inner",
        validate="many_to_one",
    )

    post = creatinine[
        creatinine["dt"].gt(creatinine["aneend"])
        & creatinine["dt"].le(creatinine["aneend"] + 7 * DAY_SEC)
        & (creatinine["dis"].isna() | creatinine["dt"].le(creatinine["dis"]))
    ].copy()
    post["hours_after_aneend"] = (post["dt"] - post["aneend"]) / 3600.0
    post_48h = post[post["dt"].le(post["aneend"] + 2 * DAY_SEC)]

    outcome = cases.copy().set_index("case_id")
    outcome["baseline_creatinine_mg_dl"] = outcome["preop_cr"]
    outcome = outcome.join(
        post_48h.groupby("caseid")["result"].max().rename("postop_cr_max_48h")
    ).join(post.groupby("caseid")["result"].max().rename("postop_cr_max_7d"))
    outcome = outcome.join(
        post_48h.groupby("caseid").size().rename("postop_cr_count_48h")
    ).join(post.groupby("caseid").size().rename("postop_cr_count_7d"))
    outcome[["postop_cr_count_48h", "postop_cr_count_7d"]] = outcome[
        ["postop_cr_count_48h", "postop_cr_count_7d"]
    ].fillna(0).astype(int)

    baseline = outcome["baseline_creatinine_mg_dl"]
    delta_48h = outcome["postop_cr_max_48h"] - baseline
    ratio_7d = outcome["postop_cr_max_7d"] / baseline
    evaluable = baseline.gt(0) & outcome["postop_cr_max_7d"].notna()
    absolute_48h = delta_48h.ge(0.3)
    relative_7d = ratio_7d.ge(1.5)

    aki = pd.Series(pd.NA, index=outcome.index, dtype="boolean")
    aki.loc[evaluable] = (absolute_48h | relative_7d).loc[evaluable]
    outcome["aki_creatinine_7d"] = aki

    stage = pd.Series(np.nan, index=outcome.index, dtype=float)
    stage.loc[evaluable] = 0
    stage.loc[evaluable & (absolute_48h | relative_7d)] = 1
    stage.loc[evaluable & ratio_7d.ge(2.0)] = 2
    stage3 = ratio_7d.ge(3.0) | (
        outcome["postop_cr_max_7d"].ge(4.0) & absolute_48h
    )
    stage.loc[evaluable & stage3] = 3
    outcome["aki_stage_creatinine"] = stage.astype("Int64")
    outcome["creatinine_delta_48h_mg_dl"] = delta_48h
    outcome["creatinine_ratio_7d"] = ratio_7d

    post_with_baseline = post.merge(
        outcome[["baseline_creatinine_mg_dl"]],
        left_on="caseid",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    post_with_baseline["qualifying_aki_value"] = (
        (
            post_with_baseline["hours_after_aneend"].le(48.0)
            & post_with_baseline["result"]
            .sub(post_with_baseline["baseline_creatinine_mg_dl"])
            .ge(0.3)
        )
        | post_with_baseline["result"]
        .div(post_with_baseline["baseline_creatinine_mg_dl"])
        .ge(1.5)
    )
    first_aki = (
        post_with_baseline[post_with_baseline["qualifying_aki_value"]]
        .sort_values(["caseid", "dt"])
        .groupby("caseid", sort=False)
        .head(1)
        .set_index("caseid")["hours_after_aneend"]
    )
    outcome = outcome.join(first_aki.rename("first_aki_criterion_hour"))
    outcome["aki_evaluable"] = evaluable
    outcome["aki_definition"] = (
        "KDIGO serum-creatinine criteria only: >=0.3 mg/dL by 48 h or "
        ">=1.5 times baseline by 7 d; urine output and dialysis unavailable"
    )
    outcome["baseline_definition"] = "VitalDB preop_cr clinical-information field"
    return outcome.reset_index()


def aggregate_invisibility_features(
    frequency: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    threshold: float = 65.0,
    interval_min: float = 5.0,
) -> pd.DataFrame:
    """Aggregate case-level invisibility and decision-opportunity features."""

    _require_columns(
        frequency,
        {
            "case_id",
            "threshold",
            "interval_min",
            "true_auc",
            "hidden_auc",
            "concordant_auc",
            "anesthesia_hours",
            "true_hypotension_min",
        },
        "frequency",
    )
    _require_columns(
        episodes,
        {
            "case_id",
            "threshold",
            "interval_min",
            "reference_episodes",
            "phase_episode_pairs",
            "expected_missed_episodes",
            "detected_with_60s_remaining_pairs",
            "detected_with_120s_remaining_pairs",
            "reference_episode_auc_phase_sum",
            "pre_detection_reference_auc_sum",
        },
        "episodes",
    )

    selected = frequency[
        frequency["threshold"].eq(float(threshold))
        & frequency["interval_min"].eq(float(interval_min))
    ].copy()
    if selected["case_id"].duplicated().any():
        raise ValueError("frequency must contain one selected row per case_id")
    selected["true_twa"] = selected["true_auc"] / selected["anesthesia_hours"]
    selected["hidden_twa"] = selected["hidden_auc"] / selected["anesthesia_hours"]
    selected["concordant_twa"] = (
        selected["concordant_auc"] / selected["anesthesia_hours"]
    )
    selected["hdr"] = selected["hidden_auc"] / selected["true_auc"].replace(0, np.nan)
    selected["true_twa65"] = selected["true_twa"]
    selected["hidden_twa65"] = selected["hidden_twa"]
    selected["concordant_twa65"] = selected["concordant_twa"]
    selected["hdr65"] = selected["hdr"]

    closure_error = (
        selected["true_auc"]
        - selected["hidden_auc"]
        - selected["concordant_auc"]
    ).abs()
    if bool(closure_error.gt(1e-8).any()):
        raise ValueError("true AUC must equal hidden plus concordant AUC")

    episode_selected = episodes[
        episodes["threshold"].eq(float(threshold))
        & episodes["interval_min"].eq(float(interval_min))
    ].copy()
    episode_case = episode_selected.groupby("case_id", sort=False).agg(
        reference_episodes=("reference_episodes", "sum"),
        expected_missed_episodes=("expected_missed_episodes", "sum"),
        phase_episode_pairs=("phase_episode_pairs", "sum"),
        detected_with_60s_remaining_pairs=(
            "detected_with_60s_remaining_pairs",
            "sum",
        ),
        detected_with_120s_remaining_pairs=(
            "detected_with_120s_remaining_pairs",
            "sum",
        ),
        reference_episode_auc_phase_sum=(
            "reference_episode_auc_phase_sum",
            "sum",
        ),
        pre_detection_reference_auc_sum=(
            "pre_detection_reference_auc_sum",
            "sum",
        ),
    )
    episode_case["episode_complete_miss_probability"] = (
        episode_case["expected_missed_episodes"]
        / episode_case["reference_episodes"].replace(0, np.nan)
    )
    episode_case["episode_1min_opportunity_probability"] = (
        episode_case["detected_with_60s_remaining_pairs"]
        / episode_case["phase_episode_pairs"].replace(0, np.nan)
    )
    episode_case["episode_2min_opportunity_probability"] = (
        episode_case["detected_with_120s_remaining_pairs"]
        / episode_case["phase_episode_pairs"].replace(0, np.nan)
    )
    episode_case["predisplay_auc_fraction"] = (
        episode_case["pre_detection_reference_auc_sum"]
        / episode_case["reference_episode_auc_phase_sum"].replace(0, np.nan)
    )

    keep = [
        "case_id",
        "true_auc",
        "hidden_auc",
        "concordant_auc",
        "anesthesia_hours",
        "true_hypotension_min",
        "true_twa",
        "hidden_twa",
        "concordant_twa",
        "hdr",
        "true_twa65",
        "hidden_twa65",
        "concordant_twa65",
        "hdr65",
    ]
    return selected[keep].merge(
        episode_case.reset_index(),
        on="case_id",
        how="left",
        validate="one_to_one",
    )


def summarize_hidden_presence(
    frame: pd.DataFrame,
    *,
    exposure_column: str = "hidden_auc",
    outcome_column: str = "aki",
) -> pd.DataFrame:
    """Summarize any hidden burden and fail closed when overlap is absent.

    No continuity correction is applied. A zero cell is retained because it is
    the scientific reason that an ordinary binary regression is not stable.
    """

    _require_columns(frame, {exposure_column, outcome_column}, "presence frame")
    working = pd.DataFrame(
        {
            "exposure": pd.to_numeric(frame[exposure_column], errors="coerce"),
            "outcome": pd.to_numeric(frame[outcome_column], errors="coerce"),
        }
    ).dropna()
    working = working[working["outcome"].isin([0, 1])].copy()
    working["present"] = working["exposure"].gt(0)

    absent = working.loc[~working["present"], "outcome"]
    present = working.loc[working["present"], "outcome"]
    absent_events = int(absent.sum())
    present_events = int(present.sum())
    absent_non_events = int(len(absent) - absent_events)
    present_non_events = int(len(present) - present_events)

    if len(absent) == 0 or len(present) == 0:
        odds_ratio = np.nan
        fisher_p = np.nan
        status = "NOT_ESTIMABLE_NO_OVERLAP"
        reason = "one exposure group is absent"
    else:
        fisher = fisher_exact(
            [
                [present_events, present_non_events],
                [absent_events, absent_non_events],
            ],
            alternative="two-sided",
        )
        odds_ratio = float(fisher.statistic)
        fisher_p = float(fisher.pvalue)
        cells = [
            absent_events,
            absent_non_events,
            present_events,
            present_non_events,
        ]
        if any(value == 0 for value in cells):
            status = "NOT_ESTIMABLE_COMPLETE_SEPARATION"
            reason = "at least one zero outcome cell produces complete separation"
        else:
            status = "ESTIMABLE"
            reason = "all four exposure-outcome cells observed"

    return pd.DataFrame(
        [
            {
                "definition": f"any {exposure_column} > 0",
                "n_evaluable": int(len(working)),
                "absent_n": int(len(absent)),
                "absent_events": absent_events,
                "absent_non_events": absent_non_events,
                "absent_aki_risk": (
                    float(absent.mean()) if len(absent) else np.nan
                ),
                "present_n": int(len(present)),
                "present_events": present_events,
                "present_non_events": present_non_events,
                "present_aki_risk": (
                    float(present.mean()) if len(present) else np.nan
                ),
                "crude_odds_ratio": odds_ratio,
                "fisher_exact_p_value": fisher_p,
                "estimability_status": status,
                "estimability_reason": reason,
            }
        ]
    )


def fit_modified_poisson(
    frame: pd.DataFrame,
    *,
    formula: str,
    term: str,
    cluster_column: str,
) -> dict[str, float | int | str]:
    """Fit a Poisson log-link model with cluster-robust standard errors."""

    if cluster_column not in frame:
        raise ValueError(f"cluster column is missing: {cluster_column}")
    working = frame.reset_index(drop=True).copy()
    for column in working.columns:
        if isinstance(working[column].dtype, pd.BooleanDtype):
            working[column] = working[column].astype("Float64").astype(float)
    model = smf.glm(formula=formula, data=working, family=sm.families.Poisson())
    row_labels = pd.Index(model.data.row_labels)
    groups = working.loc[row_labels, cluster_column]
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    if term not in fitted.params:
        raise ValueError(f"model term is missing: {term}")
    ci = fitted.conf_int().loc[term]
    outcome = pd.Series(fitted.model.endog)
    return {
        "formula": formula,
        "term": term,
        "n_cases": int(fitted.nobs),
        "n_subjects": int(pd.Series(groups).nunique()),
        "events": int(outcome.sum()),
        "n_parameters": int(len(fitted.params)),
        "coefficient": float(fitted.params[term]),
        "standard_error": float(fitted.bse[term]),
        "relative_risk": float(np.exp(fitted.params[term])),
        "ci_low": float(np.exp(ci.iloc[0])),
        "ci_high": float(np.exp(ci.iloc[1])),
        "p_value": float(fitted.pvalues[term]),
        "aic": float(fitted.aic),
        "log_likelihood": float(fitted.llf),
    }
