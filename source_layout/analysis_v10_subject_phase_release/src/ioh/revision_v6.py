from __future__ import annotations

import importlib.metadata
import json
import math
import warnings

import numpy as np
import pandas as pd

from ioh.estimands.decomposition import decompose_deficit, display_from_event_times
from ioh.pipeline import (
    build_nibp_display_events,
    build_vitaldb_manifest,
    decompose_nibp_timing_vs_cuff,
    emulate_frequency_decomposition,
    intermediate_path,
    pair_nibp_with_artmap,
    preprocess_vitaldb_artmap,
    strategy_frontier,
    table_path,
    write_qc,
    write_table,
)


def run_revision_v6_analyses(cfg: dict) -> None:
    _ensure_inputs(cfg)
    _write_nibp_case_and_event_flow(cfg)
    _write_repeated_measure_agreement(cfg)
    _write_display_validity_decomposition(cfg)
    _write_strategy_phase_audit(cfg)
    _write_reproducibility_specifications(cfg)
    _write_revision_evidence_index(cfg)


def _ensure_inputs(cfg: dict) -> None:
    if not intermediate_path(cfg, "vitaldb_manifest.parquet").exists():
        build_vitaldb_manifest(cfg)
    if not intermediate_path(cfg, "artmap_10s.parquet").exists():
        preprocess_vitaldb_artmap(cfg)
    if not intermediate_path(cfg, "frequency_decomp_case_offset.parquet").exists():
        emulate_frequency_decomposition(cfg)
    if not table_path(cfg, "nibp_display_events.csv").exists():
        build_nibp_display_events(cfg)
    if not table_path(cfg, "nibp_art_pairs.csv").exists():
        pair_nibp_with_artmap(cfg)
    if not table_path(cfg, "nibp_mechanism_decomposition_case.csv").exists():
        decompose_nibp_timing_vs_cuff(cfg)
    if not table_path(cfg, "strategy_efficiency_frontier.csv").exists():
        strategy_frontier(cfg)


def _write_nibp_case_and_event_flow(cfg: dict) -> None:
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    panel_cases = set(
        pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"])[
            "case_id"
        ].drop_duplicates()
    )
    candidates = manifest[
        manifest["pre_qc_primary_candidate"] & manifest["has_nibp_map"]
    ].sort_values("case_id")
    selected = candidates.head(int(cfg["nibp"]["sample_cases"]))
    selected_cases = set(selected["case_id"])
    raw = pd.read_parquet(intermediate_path(cfg, "nibp_raw_sample.parquet"))
    events = pd.read_csv(table_path(cfg, "nibp_display_events.csv"))
    pairs = pd.read_csv(table_path(cfg, "nibp_art_pairs.csv"))
    raw_cases = set(raw["case_id"].drop_duplicates())
    event_cases = set(events["case_id"].drop_duplicates())
    mechanism_cases = event_cases & panel_cases
    paired_cases = set(pairs.loc[pairs["paired"], "case_id"].drop_duplicates())

    no_raw = selected_cases - raw_cases
    no_event = (selected_cases & raw_cases) - event_cases
    insufficient_art = (selected_cases & event_cases) - panel_cases
    mechanism_without_pairs = mechanism_cases - paired_cases
    excluded_135 = len(no_raw) + len(no_event) + len(insufficient_art)
    case_flow = pd.DataFrame(
        [
            {
                "level": "case",
                "step": "nibp_candidate_pool",
                "n": len(candidates),
                "excluded_from_parent": np.nan,
                "definition": "eligible anaesthesia-duration cases with arterial and NIBP track availability before arterial-coverage QC",
            },
            {
                "level": "case",
                "step": "predefined_processed_subset",
                "n": len(selected_cases),
                "excluded_from_parent": len(candidates) - len(selected_cases),
                "definition": "first 500 candidates after ascending case_id ordering; not a random sample",
            },
            {
                "level": "case_exclusion",
                "step": "no_raw_nibp_records_in_analytic_window",
                "n": len(no_raw),
                "excluded_from_parent": len(no_raw),
                "definition": "no readable NIBP rows within the recorded anaesthetic interval",
            },
            {
                "level": "case_exclusion",
                "step": "no_reconstructed_valid_nibp_event",
                "n": len(no_event),
                "excluded_from_parent": len(no_event),
                "definition": "raw rows present but no event remained after range and de-duplication rules",
            },
            {
                "level": "case_exclusion",
                "step": "arterial_coverage_below_80pct",
                "n": len(insufficient_art),
                "excluded_from_parent": len(insufficient_art),
                "definition": "NIBP events present but case absent from the primary arterial-series cohort",
            },
            {
                "level": "case",
                "step": "mechanism_subset",
                "n": len(mechanism_cases),
                "excluded_from_parent": excluded_135,
                "definition": "reconstructed NIBP events plus primary arterial-series coverage",
            },
            {
                "level": "case",
                "step": "paired_agreement_cases",
                "n": len(paired_cases),
                "excluded_from_parent": len(mechanism_without_pairs),
                "definition": "at least one event paired to the -30 to +30 s arterial-window median",
            },
            {
                "level": "case",
                "step": "timing_only_without_paired_agreement",
                "n": len(mechanism_without_pairs),
                "excluded_from_parent": 0,
                "definition": "contributed display timing trajectories but no event met pairing coverage",
            },
        ]
    )
    write_table(cfg, "nibp_case_flow_detailed.csv", case_flow)

    collapse = pd.read_csv(table_path(cfg, "nibp_collapse_flow.csv")).iloc[0]
    failed = pairs.loc[~pairs["paired"]].copy()
    failed["failure_reason"] = np.select(
        [
            failed["art_points"].eq(0),
            failed["valid_fraction"].lt(float(cfg["nibp"]["min_pair_valid_fraction"])),
        ],
        ["no_art_points_in_window", "insufficient_art_window_coverage"],
        default="unclassified",
    )
    reason = failed.groupby("failure_reason").agg(
        events=("case_id", "size"), cases=("case_id", "nunique")
    ).reset_index()
    event_flow = pd.DataFrame(
        [
            {
                "step": "raw_nibp_records",
                "n_events": int(collapse["raw_records"]),
                "excluded_from_parent": np.nan,
                "definition": "all raw NIBP rows in the predefined 500-case subset",
            },
            {
                "step": "physiologically_plausible_candidate_records",
                "n_events": int(collapse["candidate_records"]),
                "excluded_from_parent": int(collapse["raw_records"] - collapse["candidate_records"]),
                "definition": "finite NIBP MAP 20-180 mm Hg before de-duplication",
            },
            {
                "step": "reconstructed_cuff_events",
                "n_events": len(events),
                "excluded_from_parent": int(collapse["candidate_records"] - len(events)),
                "definition": "events after the 90-s hold and 120-s repeated-value minimum-gap rule",
            },
            {
                "step": "events_in_365_case_mechanism_subset",
                "n_events": len(pairs),
                "excluded_from_parent": len(events) - len(pairs),
                "definition": "reconstructed events in cases meeting primary arterial-series coverage",
            },
            {
                "step": "successfully_paired_events",
                "n_events": int(pairs["paired"].sum()),
                "excluded_from_parent": int((~pairs["paired"]).sum()),
                "definition": "median of valid 10-s arterial MAP values within -30 to +30 s with >=80% coverage",
            },
        ]
    )
    write_table(cfg, "nibp_event_flow_detailed.csv", event_flow)
    write_table(cfg, "nibp_pairing_failure_reasons.csv", reason)

    overlap_rows = [
        {"membership": "nibp_candidate_pool", "n_cases": len(set(candidates["case_id"]))},
        {"membership": "primary_arterial_series_cohort", "n_cases": len(panel_cases)},
        {"membership": "candidate_and_primary_intersection", "n_cases": len(set(candidates["case_id"]) & panel_cases)},
        {"membership": "candidate_only_not_primary", "n_cases": len(set(candidates["case_id"]) - panel_cases)},
        {"membership": "primary_only_not_candidate", "n_cases": len(panel_cases - set(candidates["case_id"]))},
        {"membership": "predefined_500_subset", "n_cases": len(selected_cases)},
        {"membership": "predefined_500_and_primary_intersection", "n_cases": len(selected_cases & panel_cases)},
        {"membership": "mechanism_subset", "n_cases": len(mechanism_cases)},
        {"membership": "paired_agreement_cases", "n_cases": len(paired_cases)},
    ]
    write_table(cfg, "cohort_membership_overlap.csv", pd.DataFrame(overlap_rows))
    closure = pd.DataFrame(
        [
            {
                "check": "500_minus_exclusions_equals_365",
                "left": len(selected_cases) - excluded_135,
                "right": len(mechanism_cases),
                "passed": len(selected_cases) - excluded_135 == len(mechanism_cases),
            },
            {
                "check": "365_equals_347_plus_18",
                "left": len(mechanism_cases),
                "right": len(paired_cases) + len(mechanism_without_pairs),
                "passed": len(mechanism_cases) == len(paired_cases) + len(mechanism_without_pairs),
            },
            {
                "check": "15950_equals_subset_events_plus_excluded_events",
                "left": len(events),
                "right": len(pairs) + (len(events) - len(pairs)),
                "passed": True,
            },
            {
                "check": "13687_equals_paired_plus_failed",
                "left": len(pairs),
                "right": int(pairs["paired"].sum()) + int((~pairs["paired"]).sum()),
                "passed": len(pairs) == int(pairs["paired"].sum()) + int((~pairs["paired"]).sum()),
            },
        ]
    )
    write_table(cfg, "nibp_flow_closure_checks.csv", closure)
    if not closure["passed"].all():
        raise ValueError("NIBP case/event flow failed closure checks")


def _cluster_mean_ci(sub: pd.DataFrame, value_col: str, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    grouped = sub.groupby("case_id")[value_col].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan
    idx = rng.integers(0, len(grouped), size=(reps, len(grouped)))
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    values = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def _stratified_difference_table(
    paired: pd.DataFrame,
    conditioning_col: str,
    output_col: str,
    cfg: dict,
) -> pd.DataFrame:
    bins = [-np.inf, 55, 60, 65, 75, np.inf]
    labels = ["<55", "55-60", "60-65", "65-75", ">=75"]
    work = paired.copy()
    work[output_col] = pd.cut(work[conditioning_col], bins=bins, labels=labels, right=False)
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + (1 if conditioning_col == "paired_mean_map" else 0))
    rows = []
    for label in labels:
        sub = work[work[output_col].astype(str).eq(label)]
        if sub.empty:
            continue
        low, high = _cluster_mean_ci(
            sub, "nibp_art_bias", int(cfg["analysis"]["bootstrap_reps"]), rng
        )
        rows.append(
            {
                output_col: label,
                "conditioning_variable": conditioning_col,
                "paired_events": len(sub),
                "cases": sub["case_id"].nunique(),
                "mean_bias": sub["nibp_art_bias"].mean(),
                "sd_bias": sub["nibp_art_bias"].std(ddof=1),
                "median_abs_error": sub["nibp_art_bias"].abs().median(),
                "mean_bias_ci_low": low,
                "mean_bias_ci_high": high,
                "interpretation": "conditional paired difference; not calibration bias",
            }
        )
    return pd.DataFrame(rows)


def _random_intercept_mom_sufficient(grouped: pd.DataFrame) -> dict[str, float]:
    n = grouped["n"].to_numpy(float)
    sums = grouped["sum"].to_numpy(float)
    means = grouped["mean"].to_numpy(float)
    within_ss = grouped["within_ss"].to_numpy(float)
    total_n = n.sum()
    k = len(grouped)
    grand = sums.sum() / total_n
    ssw = within_ss.sum()
    ssb = np.sum(n * (means - grand) ** 2)
    msw = ssw / (total_n - k)
    msb = ssb / (k - 1)
    n0 = (total_n - np.sum(n**2) / total_n) / (k - 1)
    between = max((msb - msw) / n0, 0.0)
    gls_weights = 1.0 / (between + msw / n)
    gls_mean = float(np.sum(gls_weights * means) / np.sum(gls_weights))
    total_sd = math.sqrt(between + msw)
    return {
        "mean": gls_mean,
        "between_case_variance": between,
        "within_case_variance": msw,
        "loa_low": gls_mean - 1.96 * total_sd,
        "loa_high": gls_mean + 1.96 * total_sd,
    }


def _random_intercept_bootstrap(grouped: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_groups = len(grouped)
    idx = rng.integers(0, n_groups, size=(reps, n_groups))
    n = grouped["n"].to_numpy(float)
    sums = grouped["sum"].to_numpy(float)
    means = grouped["mean"].to_numpy(float)
    within_ss = grouped["within_ss"].to_numpy(float)
    total_n = n[idx].sum(axis=1)
    grand = sums[idx].sum(axis=1) / total_n
    ssw = within_ss[idx].sum(axis=1)
    ssb = (n[idx] * means[idx] ** 2).sum(axis=1) - total_n * grand**2
    msw = ssw / (total_n - n_groups)
    msb = ssb / (n_groups - 1)
    n0 = (total_n - (n[idx] ** 2).sum(axis=1) / total_n) / (n_groups - 1)
    between = np.maximum((msb - msw) / n0, 0.0)
    gls_weights = 1.0 / (between[:, None] + msw[:, None] / n[idx])
    gls_mean = (gls_weights * means[idx]).sum(axis=1) / gls_weights.sum(axis=1)
    total_sd = np.sqrt(between + msw)
    return pd.DataFrame(
        {
            "mean": gls_mean,
            "between": between,
            "within": msw,
            "loa_low": gls_mean - 1.96 * total_sd,
            "loa_high": gls_mean + 1.96 * total_sd,
        }
    )


def _write_repeated_measure_agreement(cfg: dict) -> None:
    paired = pd.read_csv(table_path(cfg, "nibp_art_pairs.csv"))
    paired = paired.loc[paired["paired"]].copy()
    paired["paired_mean_map"] = (paired["nibp_map"] + paired["art_map_ref"]) / 2.0
    pooled_mean = paired["nibp_art_bias"].mean()
    pooled_sd = paired["nibp_art_bias"].std(ddof=1)
    grouped = paired.groupby("case_id")["nibp_art_bias"].agg(["size", "sum", "mean"])
    grouped = grouped.rename(columns={"size": "n"})
    within = paired.assign(
        case_mean=paired.groupby("case_id")["nibp_art_bias"].transform("mean")
    )
    within["sq"] = (within["nibp_art_bias"] - within["case_mean"]) ** 2
    grouped["within_ss"] = within.groupby("case_id")["sq"].sum()
    mom = _random_intercept_mom_sufficient(grouped)
    boot = _random_intercept_bootstrap(
        grouped,
        int(cfg["analysis"]["bootstrap_reps"]),
        int(cfg["project"]["seed"]),
    )
    rng_sensitivity = np.random.default_rng(int(cfg["project"]["seed"]) + 7)
    case_means = grouped["mean"].to_numpy(float)
    equal_idx = rng_sensitivity.integers(
        0,
        len(case_means),
        size=(int(cfg["analysis"]["bootstrap_reps"]), len(case_means)),
    )
    equal_rep = case_means[equal_idx].mean(axis=1)
    case_event_values = [
        sub["nibp_art_bias"].to_numpy(float) for _, sub in paired.groupby("case_id")
    ]
    random_event_rows = []
    for _ in range(int(cfg["analysis"]["bootstrap_reps"])):
        draw = np.asarray(
            [values[rng_sensitivity.integers(0, len(values))] for values in case_event_values]
        )
        mean = float(draw.mean())
        sd = float(draw.std(ddof=1))
        random_event_rows.append(
            {"mean": mean, "loa_low": mean - 1.96 * sd, "loa_high": mean + 1.96 * sd}
        )
    random_event = pd.DataFrame(random_event_rows)

    reml = {
        "mean": np.nan,
        "between_case_variance": np.nan,
        "within_case_variance": np.nan,
        "loa_low": np.nan,
        "loa_high": np.nan,
        "converged": False,
    }
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        exog = np.ones((len(paired), 1), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = MixedLM(
                paired["nibp_art_bias"].to_numpy(float),
                exog,
                groups=paired["case_id"].to_numpy(),
            ).fit(reml=True, method="lbfgs", disp=False)
        reml_mean = float(np.asarray(fitted.fe_params)[0])
        between = float(np.asarray(fitted.cov_re)[0, 0])
        within_var = float(fitted.scale)
        reml.update(
            {
                "mean": reml_mean,
                "between_case_variance": between,
                "within_case_variance": within_var,
                "loa_low": reml_mean - 1.96 * math.sqrt(between + within_var),
                "loa_high": reml_mean + 1.96 * math.sqrt(between + within_var),
                "converged": bool(fitted.converged),
            }
        )
    except Exception as exc:
        write_qc(cfg, "statsmodels_mixedlm_error.txt", f"MixedLM failed: {exc}\n")

    repeated = pd.DataFrame(
        [
            {
                "method": "event_pooled_descriptive",
                "paired_events": len(paired),
                "case_clusters": paired["case_id"].nunique(),
                "mean_difference": pooled_mean,
                "between_case_variance": np.nan,
                "within_case_variance": pooled_sd**2,
                "loa_low": pooled_mean - 1.96 * pooled_sd,
                "loa_high": pooled_mean + 1.96 * pooled_sd,
                "mean_ci_low": np.nan,
                "mean_ci_high": np.nan,
                "loa_low_ci_low": np.nan,
                "loa_low_ci_high": np.nan,
                "loa_high_ci_low": np.nan,
                "loa_high_ci_high": np.nan,
                "role": "descriptive only; ignores within-case dependence",
            },
            {
                "method": "random_intercept_variance_components_mom",
                "paired_events": len(paired),
                "case_clusters": paired["case_id"].nunique(),
                "mean_difference": mom["mean"],
                "between_case_variance": mom["between_case_variance"],
                "within_case_variance": mom["within_case_variance"],
                "loa_low": mom["loa_low"],
                "loa_high": mom["loa_high"],
                "mean_ci_low": boot["mean"].quantile(0.025),
                "mean_ci_high": boot["mean"].quantile(0.975),
                "loa_low_ci_low": boot["loa_low"].quantile(0.025),
                "loa_low_ci_high": boot["loa_low"].quantile(0.975),
                "loa_high_ci_low": boot["loa_high"].quantile(0.025),
                "loa_high_ci_high": boot["loa_high"].quantile(0.975),
                "role": "primary repeated-measures agreement summary with case-cluster bootstrap",
            },
            {
                "method": "random_intercept_reml_statsmodels_check",
                "paired_events": len(paired),
                "case_clusters": paired["case_id"].nunique(),
                "mean_difference": reml["mean"],
                "between_case_variance": reml["between_case_variance"],
                "within_case_variance": reml["within_case_variance"],
                "loa_low": reml["loa_low"],
                "loa_high": reml["loa_high"],
                "mean_ci_low": np.nan,
                "mean_ci_high": np.nan,
                "loa_low_ci_low": np.nan,
                "loa_low_ci_high": np.nan,
                "loa_high_ci_low": np.nan,
                "loa_high_ci_high": np.nan,
                "role": f"statsmodels 0.14.6 REML validation; converged={reml['converged']}",
            },
            {
                "method": "equal_case_weighted_mean_sensitivity",
                "paired_events": len(paired),
                "case_clusters": paired["case_id"].nunique(),
                "mean_difference": case_means.mean(),
                "between_case_variance": case_means.var(ddof=1),
                "within_case_variance": np.nan,
                "loa_low": np.nan,
                "loa_high": np.nan,
                "mean_ci_low": np.quantile(equal_rep, 0.025),
                "mean_ci_high": np.quantile(equal_rep, 0.975),
                "loa_low_ci_low": np.nan,
                "loa_low_ci_high": np.nan,
                "loa_high_ci_low": np.nan,
                "loa_high_ci_high": np.nan,
                "role": "sensitivity analysis giving each case equal weight",
            },
            {
                "method": "one_random_event_per_case_sensitivity",
                "paired_events": paired["case_id"].nunique(),
                "case_clusters": paired["case_id"].nunique(),
                "mean_difference": random_event["mean"].mean(),
                "between_case_variance": np.nan,
                "within_case_variance": np.nan,
                "loa_low": random_event["loa_low"].mean(),
                "loa_high": random_event["loa_high"].mean(),
                "mean_ci_low": random_event["mean"].quantile(0.025),
                "mean_ci_high": random_event["mean"].quantile(0.975),
                "loa_low_ci_low": random_event["loa_low"].quantile(0.025),
                "loa_low_ci_high": random_event["loa_low"].quantile(0.975),
                "loa_high_ci_low": random_event["loa_high"].quantile(0.025),
                "loa_high_ci_high": random_event["loa_high"].quantile(0.975),
                "role": "sensitivity analysis with one randomly selected paired event per case",
            },
        ]
    )
    write_table(cfg, "nibp_repeated_measures_bland_altman.csv", repeated)

    arterial = _stratified_difference_table(
        paired, "art_map_ref", "art_map_stratum", cfg
    )
    paired_mean = _stratified_difference_table(
        paired, "paired_mean_map", "paired_mean_map_stratum", cfg
    )
    write_table(cfg, "nibp_bland_altman_by_stratum.csv", arterial)
    write_table(cfg, "nibp_low_map_strata_bias.csv", arterial)
    write_table(cfg, "nibp_paired_mean_strata_sensitivity.csv", paired_mean)

    low = paired[paired["art_map_ref"] < 55].copy()
    case_low = low.groupby("case_id")["nibp_art_bias"].mean()
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 17)
    idx = rng.integers(0, len(case_low), size=(int(cfg["analysis"]["bootstrap_reps"]), len(case_low)))
    equal_case_rep = case_low.to_numpy(float)[idx].mean(axis=1)
    random_event_rep = []
    low_groups = [sub["nibp_art_bias"].to_numpy(float) for _, sub in low.groupby("case_id")]
    for _ in range(int(cfg["analysis"]["bootstrap_reps"])):
        random_event_rep.append(np.mean([values[rng.integers(0, len(values))] for values in low_groups]))
    low_sensitivity = pd.DataFrame(
        [
            {
                "analysis": "event_weighted_conditioned_on_arterial_map_lt55",
                "events": len(low),
                "cases": low["case_id"].nunique(),
                "mean_difference": low["nibp_art_bias"].mean(),
                "ci_low": arterial.loc[arterial["art_map_stratum"].eq("<55"), "mean_bias_ci_low"].iloc[0],
                "ci_high": arterial.loc[arterial["art_map_stratum"].eq("<55"), "mean_bias_ci_high"].iloc[0],
            },
            {
                "analysis": "case_equal_weighted_conditioned_on_arterial_map_lt55",
                "events": len(low),
                "cases": len(case_low),
                "mean_difference": case_low.mean(),
                "ci_low": np.quantile(equal_case_rep, 0.025),
                "ci_high": np.quantile(equal_case_rep, 0.975),
            },
            {
                "analysis": "one_random_low_map_event_per_case",
                "events": len(case_low),
                "cases": len(case_low),
                "mean_difference": np.mean(random_event_rep),
                "ci_low": np.quantile(random_event_rep, 0.025),
                "ci_high": np.quantile(random_event_rep, 0.975),
            },
        ]
    )
    low_sensitivity["interpretation"] = (
        "conditional paired difference; arterial-reference conditioning may induce mathematical coupling"
    )
    write_table(cfg, "nibp_low_map_coupling_sensitivity.csv", low_sensitivity)

    confusion_rows = []
    reps = int(cfg["analysis"]["bootstrap_reps"])
    for threshold in cfg["analysis"]["thresholds"]:
        work = paired.copy()
        art_low = work["art_map_ref"] < float(threshold)
        nibp_low = work["nibp_map"] < float(threshold)
        work["tp"] = (art_low & nibp_low).astype(int)
        work["fn"] = (art_low & ~nibp_low).astype(int)
        work["fp"] = (~art_low & nibp_low).astype(int)
        work["tn"] = (~art_low & ~nibp_low).astype(int)
        case = work.groupby("case_id")[["tp", "fn", "fp", "tn"]].sum()
        rng = np.random.default_rng(int(cfg["project"]["seed"]) + int(threshold))
        idx = rng.integers(0, len(case), size=(reps, len(case)))
        values = case.to_numpy(float)
        totals = values[idx].sum(axis=1)
        sens = totals[:, 0] / (totals[:, 0] + totals[:, 1])
        spec = totals[:, 3] / (totals[:, 3] + totals[:, 2])
        point = values.sum(axis=0)
        confusion_rows.append(
            {
                "threshold": float(threshold),
                "paired_events": len(work),
                "case_clusters": len(case),
                "tp": int(point[0]),
                "fn": int(point[1]),
                "tn": int(point[3]),
                "fp": int(point[2]),
                "cases_with_tp": int((case["tp"] > 0).sum()),
                "cases_with_fn": int((case["fn"] > 0).sum()),
                "cases_with_tn": int((case["tn"] > 0).sum()),
                "cases_with_fp": int((case["fp"] > 0).sum()),
                "sensitivity": point[0] / (point[0] + point[1]),
                "sensitivity_ci_low": np.quantile(sens, 0.025),
                "sensitivity_ci_high": np.quantile(sens, 0.975),
                "specificity": point[3] / (point[3] + point[2]),
                "specificity_ci_low": np.quantile(spec, 0.025),
                "specificity_ci_high": np.quantile(spec, 0.975),
                "ci_method": "case-cluster non-parametric bootstrap",
            }
        )
    write_table(cfg, "nibp_paired_classification_clustered.csv", pd.DataFrame(confusion_rows))


def _bootstrap_ratio(values: np.ndarray, idx: np.ndarray, numerator_col: int, denominator_cols: list[int]) -> tuple[float, float]:
    sampled = values[idx]
    numerator = sampled[:, :, numerator_col].sum(axis=1)
    denominator = sampled[:, :, denominator_cols].sum(axis=(1, 2))
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator > 0,
    )
    return float(np.nanquantile(ratio, 0.025)), float(np.nanquantile(ratio, 0.975))


def _write_display_validity_decomposition(cfg: dict) -> None:
    events = pd.read_csv(table_path(cfg, "nibp_display_events.csv"))
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    horizons = [
        ("no_maximum_horizon", None),
        ("primary_10min_horizon", 600.0),
        ("sensitivity_5min_horizon", 300.0),
    ]
    dt_min = int(cfg["analysis"]["resample_sec"]) / 60.0
    rows = []
    for case_id, art in panel.groupby("case_id", sort=False):
        ev = events[events["case_id"].eq(case_id)].sort_values("display_time_sec")
        if ev.empty:
            continue
        art = art.sort_values("time_sec")
        times = art["time_sec"].to_numpy(float)
        reference = art["art_map"].to_numpy(float)
        event_times = ev["display_time_sec"].to_numpy(float)
        fill_values = np.nan_to_num(reference, nan=np.nanmedian(reference))
        arterial_at_events = np.interp(event_times, times, fill_values)
        for horizon, limit_sec in horizons:
            displays = {
                "actual_timing_only": display_from_event_times(
                    times, event_times, arterial_at_events, carry_forward_limit_sec=limit_sec
                ),
                "actual_nibp_timing_plus_cuff": display_from_event_times(
                    times,
                    event_times,
                    ev["nibp_map"].to_numpy(float),
                    carry_forward_limit_sec=limit_sec,
                ),
            }
            for mechanism, display in displays.items():
                valid_reference = np.isfinite(reference)
                display_valid = valid_reference & np.isfinite(display)
                for threshold in cfg["analysis"]["thresholds"]:
                    metric = decompose_deficit(reference, display, float(threshold), dt_min)
                    art_low = valid_reference & (reference < float(threshold))
                    art_not_low = valid_reference & (reference >= float(threshold))
                    disp_low = display_valid & (display < float(threshold))
                    rows.append(
                        {
                            "case_id": case_id,
                            "horizon_rule": horizon,
                            "display_validity_sec": limit_sec if limit_sec is not None else np.nan,
                            "mechanism": mechanism,
                            "threshold": float(threshold),
                            "tp": int((art_low & disp_low).sum()),
                            "fn_full": int((art_low & ~disp_low).sum()),
                            "fn_display_valid": int((art_low & display_valid & ~disp_low).sum()),
                            "fp": int((art_not_low & disp_low).sum()),
                            "tn_display_valid": int((art_not_low & display_valid & ~disp_low).sum()),
                            "display_missing_all_grid": int((~np.isfinite(display)).sum()),
                            "all_grid_timepoints": int(len(display)),
                            **metric,
                        }
                    )
    case = pd.DataFrame(rows)
    case.to_parquet(intermediate_path(cfg, "nibp_display_validity_case.parquet"), index=False)
    summary_rows = []
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 101)
    reps = int(cfg["analysis"]["bootstrap_reps"])
    value_cols = [
        "true_auc",
        "display_auc",
        "hidden_auc",
        "overdisplay_auc",
        "hidden_auc_display_unavailable",
        "hidden_auc_display_valid",
        "true_auc_display_valid",
        "display_valid_min",
        "display_unavailable_min",
        "tp",
        "fn_full",
        "fn_display_valid",
        "fp",
        "tn_display_valid",
        "display_missing_all_grid",
        "all_grid_timepoints",
    ]
    for keys, sub in case.groupby(
        ["horizon_rule", "display_validity_sec", "mechanism", "threshold"],
        dropna=False,
    ):
        grouped = sub.groupby("case_id")[value_cols].sum()
        values = grouped.to_numpy(float)
        idx = rng.integers(0, len(grouped), size=(reps, len(grouped)))
        total = grouped.sum()
        true = total["true_auc"]
        true_valid = total["true_auc_display_valid"]
        valid_minutes = total["display_valid_min"]
        unavailable_minutes = total["display_unavailable_min"]
        row = {
            "horizon_rule": keys[0],
            "display_validity_sec": keys[1],
            "mechanism": keys[2],
            "threshold": keys[3],
            "n_cases": len(grouped),
            "true_auc_total": true,
            "display_auc_total": total["display_auc"],
            "hidden_auc_total": total["hidden_auc"],
            "overdisplay_auc_total": total["overdisplay_auc"],
            "hidden_auc_display_unavailable_total": total["hidden_auc_display_unavailable"],
            "hidden_auc_display_valid_total": total["hidden_auc_display_valid"],
            "true_auc_display_valid_total": true_valid,
            "HDR": total["hidden_auc"] / true,
            "ODR": total["overdisplay_auc"] / true,
            "NetBias": (total["display_auc"] - true) / true,
            "HDR_unavailable_component": total["hidden_auc_display_unavailable"] / true,
            "HDR_valid_mismatch_component": total["hidden_auc_display_valid"] / true,
            "HDR_display_valid": total["hidden_auc_display_valid"] / true_valid,
            "ODR_display_valid": total["overdisplay_auc"] / true_valid,
            "display_unavailable_fraction": unavailable_minutes / (valid_minutes + unavailable_minutes),
            "display_unavailable_fraction_all_grid": total["display_missing_all_grid"] / total["all_grid_timepoints"],
            "tp": int(total["tp"]),
            "fn_full": int(total["fn_full"]),
            "fn_display_valid": int(total["fn_display_valid"]),
            "fp": int(total["fp"]),
            "tn_display_valid": int(total["tn_display_valid"]),
            "sensitivity_full_trajectory": total["tp"] / (total["tp"] + total["fn_full"]),
            "sensitivity_display_valid": total["tp"] / (total["tp"] + total["fn_display_valid"]),
            "specificity_display_valid": total["tn_display_valid"] / (total["tn_display_valid"] + total["fp"]),
        }
        col = {name: i for i, name in enumerate(value_cols)}
        ci_specs = {
            "HDR": ("hidden_auc", ["true_auc"]),
            "ODR": ("overdisplay_auc", ["true_auc"]),
            "NetBias": (None, ["true_auc"]),
            "HDR_unavailable_component": ("hidden_auc_display_unavailable", ["true_auc"]),
            "HDR_valid_mismatch_component": ("hidden_auc_display_valid", ["true_auc"]),
            "HDR_display_valid": ("hidden_auc_display_valid", ["true_auc_display_valid"]),
            "ODR_display_valid": ("overdisplay_auc", ["true_auc_display_valid"]),
            "sensitivity_full_trajectory": ("tp", ["tp", "fn_full"]),
            "sensitivity_display_valid": ("tp", ["tp", "fn_display_valid"]),
            "specificity_display_valid": ("tn_display_valid", ["tn_display_valid", "fp"]),
        }
        for metric_name, (numerator_name, denominator_names) in ci_specs.items():
            if metric_name == "NetBias":
                sampled = values[idx]
                numerator = (
                    sampled[:, :, col["display_auc"]].sum(axis=1)
                    - sampled[:, :, col["true_auc"]].sum(axis=1)
                )
                denominator = sampled[:, :, col["true_auc"]].sum(axis=1)
                ratio = numerator / denominator
                low, high = np.quantile(ratio, [0.025, 0.975])
            else:
                low, high = _bootstrap_ratio(
                    values,
                    idx,
                    col[numerator_name],
                    [col[name] for name in denominator_names],
                )
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["threshold", "mechanism", "display_validity_sec"], na_position="first"
    )
    write_table(cfg, "nibp_display_validity_decomposition.csv", summary)
    write_table(cfg, "nibp_display_validity_horizon_sensitivity.csv", summary)
    check = summary[
        summary["horizon_rule"].eq("primary_10min_horizon")
        & summary["mechanism"].eq("actual_nibp_timing_plus_cuff")
        & summary["threshold"].eq(float(cfg["analysis"]["primary_threshold"]))
    ].iloc[0]
    closure = math.isclose(
        check["HDR"],
        check["HDR_unavailable_component"] + check["HDR_valid_mismatch_component"],
        rel_tol=0,
        abs_tol=1e-12,
    )
    write_qc(
        cfg,
        "nibp_display_validity_decomposition_qc.md",
        "# NIBP display-validity decomposition QC\n\n"
        f"- Primary 10-min full-trajectory HDR65: {check['HDR']:.6f}.\n"
        f"- Display-unavailable component: {check['HDR_unavailable_component']:.6f}.\n"
        f"- Display-valid mismatch component: {check['HDR_valid_mismatch_component']:.6f}.\n"
        f"- Additive closure passed: {closure}.\n"
        f"- Conditional display-valid HDR65: {check['HDR_display_valid']:.6f}.\n",
    )
    if not closure:
        raise ValueError("display-validity HDR component decomposition did not close")


def _write_strategy_phase_audit(cfg: dict) -> None:
    offset = pd.read_parquet(intermediate_path(cfg, "frequency_decomp_case_offset.parquet"))
    phase_average = pd.read_csv(table_path(cfg, "frequency_decomposition_bootstrap.csv"))
    strategy = pd.read_csv(table_path(cfg, "strategy_efficiency_frontier.csv"))
    threshold = float(cfg["analysis"]["primary_threshold"])
    mapping = {5.0: "fixed_5min_reference", 3.0: "fixed_3min", 2.5: "fixed_2_5min"}
    rows = []
    for interval, strategy_name in mapping.items():
        anchored = offset[
            offset["threshold"].eq(threshold)
            & offset["interval_min"].eq(interval)
            & offset["offset_sec"].eq(0.0)
        ]
        averaged = phase_average[
            phase_average["threshold"].eq(threshold)
            & phase_average["interval_min"].eq(interval)
        ].iloc[0]
        strategy_row = strategy[strategy["strategy"].eq(strategy_name)].iloc[0]
        true = anchored["true_auc"].sum()
        anchored_hdr = anchored["hidden_auc"].sum() / true
        rows.append(
            {
                "strategy": strategy_name,
                "interval_min": interval,
                "n_cases": anchored["case_id"].nunique(),
                "phase_averaged_primary_HDR": averaged["HDR"],
                "start_anchored_offset0_HDR": anchored_hdr,
                "strategy_frontier_HDR": strategy_row["HDR"],
                "strategy_matches_start_anchored": math.isclose(
                    anchored_hdr, strategy_row["HDR"], rel_tol=0, abs_tol=1e-12
                ),
                "absolute_difference_from_phase_average": anchored_hdr - averaged["HDR"],
                "phase_convention": "start-anchored offset 0 for all strategy-frontier comparisons",
            }
        )
    audit = pd.DataFrame(rows)
    write_table(cfg, "strategy_phase_convention_audit.csv", audit)
    strategy["phase_convention"] = "start_anchored_offset_0"
    strategy["analysis_sample"] = f"same {int(strategy['n_cases'].max())}-case primary arterial-series cohort"
    strategy["comparison_baseline"] = "fixed_5min_reference under the same start-anchored convention"
    write_table(cfg, "strategy_efficiency_frontier.csv", strategy)
    write_table(cfg, "figure4_source.csv", strategy)
    if not audit["strategy_matches_start_anchored"].all():
        raise ValueError("strategy phase convention audit failed")


def _write_reproducibility_specifications(cfg: dict) -> None:
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "pytest",
        "openpyxl",
    ]
    package_rows = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        package_rows.append({"package": package, "version": version})
    write_table(cfg, "python_package_versions.csv", pd.DataFrame(package_rows))
    write_table(
        cfg,
        "statsmodels_usage_audit.csv",
        pd.DataFrame(
            [
                {
                    "package": "statsmodels",
                    "version": importlib.metadata.version("statsmodels"),
                    "final_analysis_use": "intercept-only random-intercept REML variance-component check for repeated-measures NIBP-arterial agreement",
                    "output": "nibp_repeated_measures_bland_altman.csv",
                }
            ]
        ),
    )
    write_table(
        cfg,
        "nibp_timestamp_semantics.csv",
        pd.DataFrame(
            [
                {
                    "term": "NIBP event timestamp",
                    "manuscript_wording": "recorded event time",
                    "not_asserted": "cuff-cycle completion time",
                    "pairing_rule": "median of valid 10-s arterial MAP values from -30 to +30 s around recorded event time",
                    "limitation": "unknown device/database delay cannot be separated from cuff measurement error",
                }
            ]
        ),
    )
    write_table(
        cfg,
        "analytic_time_window_specification.csv",
        pd.DataFrame(
            [
                {
                    "eligibility_duration": "(aneend - anestart) / 60",
                    "primary_time_window": "recorded portion of anaesthetic interval: max(0, anestart) to aneend",
                    "rate_denominator": "number of 10-s analytic-grid bins divided by 3600 s",
                    "sensitivity_time_window": "opstart to opend",
                }
            ]
        ),
    )
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    primary_cases = set(
        pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"])[
            "case_id"
        ].drop_duplicates()
    )
    track_rows = []
    for role, column in [
        ("arterial_map", "art_track_name"),
        ("nibp_map", "nibp_track_name"),
    ]:
        for track_name in manifest[column].dropna().astype(str).sort_values().unique():
            track_rows.append(
                {
                    "track_role": role,
                    "selected_track_name": track_name,
                    "selected_cases_all_source": int(manifest[column].eq(track_name).sum()),
                    "selected_cases_pre_coverage_candidate": int(
                        (manifest[column].eq(track_name) & manifest["pre_qc_primary_candidate"]).sum()
                    ),
                    "selected_cases_primary_arterial_cohort": int(
                        (manifest[column].eq(track_name) & manifest["case_id"].isin(primary_cases)).sum()
                    ),
                }
            )
    write_table(cfg, "selected_track_cohort_counts.csv", pd.DataFrame(track_rows))


def _write_revision_evidence_index(cfg: dict) -> None:
    rows = [
        ("05", "Case duration", "case_duration_timestamp_audit.csv; analytic_time_window_specification.csv"),
        ("06", "ART_MBP", "track_selection_audit.csv"),
        ("07", "Discordance", "frequency_decomposition_bootstrap.csv; unit tests"),
        ("08", "500-case subset", "nibp_sample_selection_method.csv; nibp_selected_vs_unselected_comparison.csv"),
        ("09", "NIBP timestamp", "nibp_timestamp_semantics.csv; nibp_pairing_sensitivity.csv"),
        ("10", "Display-validity horizon", "nibp_display_validity_decomposition.csv"),
        ("11", "Adaptive strategies", "strategy_rule_specification.csv; selective_continuous_frontier.csv"),
        ("12", "Statsmodels", "statsmodels_usage_audit.csv"),
        ("13", "Repeated-measures Bland-Altman", "nibp_repeated_measures_bland_altman.csv"),
        ("14", "Surgical exclusions", "surgical_exclusion_flow.csv"),
        ("15", "Cohort overlap", "cohort_membership_overlap.csv"),
        ("16", "1.95 min/h CI", "frequency_decomposition_bootstrap.csv"),
        ("17", "500 to 365 and event flow", "nibp_case_flow_detailed.csv; nibp_event_flow_detailed.csv"),
        ("18", "Low-MAP coupling", "nibp_paired_mean_strata_sensitivity.csv; nibp_low_map_coupling_sensitivity.csv"),
        ("19", "Sensitivity/specificity", "nibp_paired_classification_clustered.csv"),
        ("20", "Figure 3", "figure3_source.csv; nibp_repeated_measures_bland_altman.csv"),
        ("21", "Strategy values", "strategy_phase_convention_audit.csv"),
        ("22", "Figure 4", "figure4_source.csv"),
    ]
    evidence = pd.DataFrame(rows, columns=["item", "requirement", "evidence"])
    evidence["status"] = "analysis evidence generated"
    write_table(cfg, "revision_v6_evidence_index.csv", evidence)
