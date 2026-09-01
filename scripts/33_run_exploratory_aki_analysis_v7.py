from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from _common import load_args
from ioh.aki_outcomes import (
    aggregate_invisibility_features,
    build_creatinine_aki_outcomes,
    clean_asa_for_adjustment,
    derive_last_preoperative_creatinine,
    fit_modified_poisson,
    summarize_hidden_presence,
    summarize_asa_validity,
)
from ioh.pipeline import (
    ensure_dirs,
    intermediate_path,
    out_root,
    table_path,
    write_qc,
    write_table,
)


BASE_COVARIATES = (
    "age10 + male + bmi5 + asa_high + emop + preop_htn + preop_dm + "
    "baseline_cr05 + duration_hr + C(department)"
)
RECORDED_ASA_COVARIATES = BASE_COVARIATES.replace("asa_high", "asa_high_recorded")
REFERENCE_BURDEN = "bs(log_true_twa, df=3, degree=2, include_intercept=False)"
COVARIATE_DESCRIPTION = (
    "age, sex, body mass index, ASA Physical Status >=3, emergency surgery, "
    "hypertension, diabetes, baseline creatinine, anaesthesia duration, and "
    "surgical department"
)


def _load_laboratory_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    explicit_path = os.environ.get("IOH_VITALDB_LABS", "").strip()
    labs_path = (
        Path(explicit_path).expanduser()
        if explicit_path
        else Path(cfg["data"]["vitaldb_root"]) / "labs.csv"
    )
    if not labs_path.is_file():
        raise FileNotFoundError(
            f"VitalDB laboratory file not found: {labs_path}; set IOH_VITALDB_LABS "
            "to an explicit official labs or labs.csv.gz file"
        )
    labs = pd.read_csv(labs_path, usecols=["caseid", "dt", "name", "result"])
    audit = {
        "input_mode": "explicit_official_download" if explicit_path else "configured_data_root",
        "input_filename": labs_path.name,
        "input_sha256": hashlib.sha256(labs_path.read_bytes()).hexdigest(),
        "laboratory_rows": int(len(labs)),
        "laboratory_cases": int(labs["caseid"].nunique()),
    }
    return labs, audit


def _analysis_frame(
    manifest: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    *,
    outcome_column: str = "aki_creatinine_7d",
) -> pd.DataFrame:
    outcome_columns = [
        "case_id",
        "subjectid",
        "baseline_creatinine_mg_dl",
        "postop_cr_max_48h",
        "postop_cr_max_7d",
        "postop_cr_count_48h",
        "postop_cr_count_7d",
        "aki_creatinine_7d",
        "aki_stage_creatinine",
        "aki_evaluable",
    ]
    frame = manifest.merge(
        outcomes[outcome_columns],
        on=["case_id", "subjectid"],
        how="left",
        validate="one_to_one",
    ).merge(features, on="case_id", how="left", validate="one_to_one")
    frame["aki"] = pd.to_numeric(frame[outcome_column], errors="coerce")
    frame["age10"] = pd.to_numeric(frame["age"], errors="coerce") / 10.0
    frame["male"] = (
        frame["sex"].astype("string").str.upper().str.strip().eq("M").astype(float)
    )
    frame["bmi5"] = pd.to_numeric(frame["bmi"], errors="coerce") / 5.0
    asa_recorded = pd.to_numeric(frame["asa"], errors="coerce")
    asa_clean = clean_asa_for_adjustment(frame["asa"])
    frame["asa_high"] = asa_clean.ge(3).astype(float)
    frame.loc[asa_clean.isna(), "asa_high"] = np.nan
    frame["asa_high_recorded"] = asa_recorded.ge(3).astype(float)
    frame.loc[asa_recorded.isna(), "asa_high_recorded"] = np.nan
    for column in ("emop", "preop_htn", "preop_dm"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["baseline_cr05"] = (
        pd.to_numeric(frame["baseline_creatinine_mg_dl"], errors="coerce") / 0.5
    )
    frame["duration_hr"] = (
        pd.to_numeric(frame["duration_min"], errors="coerce") / 60.0
    )
    frame["log_true_twa"] = np.log1p(pd.to_numeric(frame["true_twa"], errors="coerce"))
    frame["hdr10"] = pd.to_numeric(frame["hdr"], errors="coerce") / 0.10
    frame["predisplay10"] = (
        pd.to_numeric(frame["predisplay_auc_fraction"], errors="coerce") / 0.10
    )
    frame["hidden_twa10"] = (
        pd.to_numeric(frame["hidden_twa"], errors="coerce") / 10.0
    )
    return frame


def _recorded_asa_sensitivity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy using the recorded ASA >=3 indicator for sensitivity models."""

    sensitivity = frame.copy()
    sensitivity["asa_high"] = sensitivity["asa_high_recorded"]
    return sensitivity


def _fit_spec(
    frame: pd.DataFrame,
    *,
    label: str,
    analysis_role: str,
    exposure_term: str,
    exposure_label: str,
    exposure_scale: str,
    outcome_definition: str,
    baseline_definition: str,
    threshold: float,
    adjustment: str = "clinical_plus_reference",
    require_reference_hypotension: bool = True,
    additional_filter: pd.Series | None = None,
) -> dict:
    selected = frame.copy()
    if require_reference_hypotension:
        selected = selected[selected["true_auc"].gt(0)].copy()
    if additional_filter is not None:
        selected = selected.loc[additional_filter.reindex(selected.index, fill_value=False)]
    if adjustment == "none":
        formula = f"aki ~ {exposure_term}"
        covariate_adjustment = "none"
        reference_burden_adjustment = "none"
    elif adjustment == "clinical":
        formula = f"aki ~ {BASE_COVARIATES} + {exposure_term}"
        covariate_adjustment = COVARIATE_DESCRIPTION
        reference_burden_adjustment = "none"
    elif adjustment == "clinical_plus_reference":
        formula = (
            f"aki ~ {BASE_COVARIATES} + {REFERENCE_BURDEN} + {exposure_term}"
        )
        covariate_adjustment = COVARIATE_DESCRIPTION
        reference_burden_adjustment = (
            "quadratic B-spline (3 df) of log(1 + reference AUC per anaesthesia-hour)"
        )
    else:
        raise ValueError(f"Unknown adjustment sequence: {adjustment}")
    estimate = fit_modified_poisson(
        selected,
        formula=formula,
        term=exposure_term,
        cluster_column="subjectid",
    )
    estimate.update(
        {
            "analysis": label,
            "analysis_role": analysis_role,
            "outcome_definition": outcome_definition,
            "baseline_definition": baseline_definition,
            "map_threshold_mm_hg": float(threshold),
            "display_interval_min": 5.0,
            "exposure": exposure_label,
            "exposure_scale": exposure_scale,
            "population": (
                "cases with reference AUC >0"
                if require_reference_hypotension
                else "all AKI-evaluable complete cases, including zero reference AUC"
            ),
            "adjustment_sequence": adjustment,
            "reference_burden_adjustment": reference_burden_adjustment,
            "covariate_adjustment": covariate_adjustment,
            "variance_estimator": "subject-cluster robust sandwich",
            "positive_adverse_association_supported": bool(
                estimate["ci_low"] > 1.0
            ),
            "causal_boundary": (
                "Exploratory association under counterfactual display emulation; "
                "not a causal effect of monitoring frequency"
            ),
        }
    )
    return estimate


def _gate_row(
    criterion: str,
    value: int | float,
    denominator: int | float | None,
    rule: str,
    passed: bool,
) -> dict:
    return {
        "criterion": criterion,
        "value": value,
        "denominator": denominator,
        "proportion": (
            float(value) / float(denominator)
            if denominator not in (None, 0) and np.isfinite(float(denominator))
            else np.nan
        ),
        "pass_rule": rule,
        "status": "PASS" if passed else "NOT PASS",
    }


def _missingness_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["outcome_evaluable"] = working["aki"].notna()
    variables = {
        "Age, yr": pd.to_numeric(working["age"], errors="coerce"),
        "Body mass index, kg/m2": pd.to_numeric(working["bmi"], errors="coerce"),
        "ASA Physical Status >=3": working["asa_high"],
        "Emergency surgery": pd.to_numeric(working["emop"], errors="coerce"),
        "Preoperative hypertension": pd.to_numeric(
            working["preop_htn"], errors="coerce"
        ),
        "Preoperative diabetes": pd.to_numeric(
            working["preop_dm"], errors="coerce"
        ),
        "Anaesthesia duration, h": working["duration_hr"],
        "Reference AUC per anaesthesia-hour": working["true_twa"],
    }
    rows: list[dict] = []
    for label, values in variables.items():
        evaluable = values[working["outcome_evaluable"]].dropna()
        nonevaluable = values[~working["outcome_evaluable"]].dropna()
        pooled_sd = np.sqrt((evaluable.var(ddof=1) + nonevaluable.var(ddof=1)) / 2.0)
        smd = (
            (evaluable.mean() - nonevaluable.mean()) / pooled_sd
            if np.isfinite(pooled_sd) and pooled_sd > 0
            else np.nan
        )
        rows.append(
            {
                "variable": label,
                "outcome_evaluable_n": int(len(evaluable)),
                "outcome_evaluable_mean": float(evaluable.mean()),
                "outcome_evaluable_sd": float(evaluable.std(ddof=1)),
                "outcome_nonevaluable_n": int(len(nonevaluable)),
                "outcome_nonevaluable_mean": float(nonevaluable.mean()),
                "outcome_nonevaluable_sd": float(nonevaluable.std(ddof=1)),
                "standardized_mean_difference": float(smd),
            }
        )
    return pd.DataFrame(rows)


def _make_forest(
    cfg: dict,
    sequential: pd.DataFrame,
    incremental: pd.DataFrame,
) -> None:
    plot_rows = sequential.copy()
    hdr = incremental.loc[
        incremental["analysis"].eq("HDR65 incremental model")
    ].copy()
    plot_rows = pd.concat([plot_rows, hdr], ignore_index=True, sort=False)
    labels = {
        "Absolute hidden burden, unadjusted": "Absolute hidden burden\nUnadjusted",
        "Absolute hidden burden, clinical adjusted": (
            "Absolute hidden burden\nClinical covariates"
        ),
        "Absolute hidden burden, clinical plus reference burden": (
            "Absolute hidden burden\nClinical + reference burden"
        ),
        "HDR65 incremental model": "HDR65\nClinical + reference burden",
    }
    plot_rows["plot_label"] = plot_rows["analysis"].map(labels)
    plot_rows["sort_order"] = plot_rows["analysis"].map(
        {label: index for index, label in enumerate(labels)}
    )
    plot_rows = plot_rows.sort_values("sort_order").reset_index(drop=True)
    write_table(
        cfg,
        "figure_s1_exploratory_aki_forest_source.csv",
        plot_rows[
            [
                "analysis",
                "plot_label",
                "adjustment_sequence",
                "population",
                "n_cases",
                "events",
                "relative_risk",
                "ci_low",
                "ci_high",
                "exposure_scale",
            ]
        ],
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    y = np.arange(len(plot_rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    colors = ["#777777", "#C44E52", "#4C72B0", "#2A7F62"]
    for position, (_, row), color in zip(y, plot_rows.iterrows(), colors):
        ax.errorbar(
            row["relative_risk"],
            position,
            xerr=[
                [row["relative_risk"] - row["ci_low"]],
                [row["ci_high"] - row["relative_risk"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
    ax.axvline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.70, 1.40)
    ax.set_xticks([0.75, 0.9, 1.0, 1.1, 1.3])
    ax.set_xticklabels(["0.75", "0.90", "1.00", "1.10", "1.30"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{row.plot_label}\nN={int(row.n_cases):,}; AKI={int(row.events):,}"
            for row in plot_rows.itertuples()
        ]
    )
    ax.set_xlabel("Relative risk (95% CI)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_title("Exploratory associations with postoperative AKI", loc="left")
    fig.text(
        0.01,
        0.01,
        "Absolute hidden burden is scaled per 10 mm Hg·min/h; HDR65 is scaled per 10 percentage points. Models are exploratory and noncausal.",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.39, bottom=0.19, right=0.97, top=0.88)
    figure_dir = out_root(cfg) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "Supplemental_Figure_S1_exploratory_AKI_forest"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def run(cfg: dict) -> None:
    ensure_dirs(cfg)
    panel_case_ids = pd.Index(
        pd.read_parquet(
            intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"]
        )["case_id"].drop_duplicates()
    )
    manifest_all = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    manifest = manifest_all[manifest_all["case_id"].isin(panel_case_ids)].copy()
    labs, labs_audit = _load_laboratory_data(cfg)
    write_table(cfg, "aki_labs_source_audit.csv", pd.DataFrame([labs_audit]))

    outcomes = build_creatinine_aki_outcomes(manifest, labs)
    outcomes.to_parquet(
        intermediate_path(cfg, "aki_creatinine_case_v7.parquet"), index=False
    )
    frequency = pd.read_parquet(
        intermediate_path(cfg, "frequency_decomp_case_mean_v7.parquet")
    )
    episodes = pd.read_parquet(
        intermediate_path(cfg, "episode_observability_case_5min.parquet")
    )
    feature_sets = {
        threshold: aggregate_invisibility_features(
            frequency, episodes, threshold=threshold, interval_min=5.0
        )
        for threshold in (65.0, 60.0, 55.0)
    }
    primary = _analysis_frame(manifest, outcomes, feature_sets[65.0])
    recorded_asa_primary = _recorded_asa_sensitivity_frame(primary)
    primary.to_parquet(
        intermediate_path(cfg, "aki_invisibility_analysis_case_v7.parquet"),
        index=False,
    )
    r_columns = [
        "case_id",
        "subjectid",
        "aki",
        "age10",
        "male",
        "bmi5",
        "asa_high",
        "asa_high_recorded",
        "emop",
        "preop_htn",
        "preop_dm",
        "baseline_cr05",
        "duration_hr",
        "department",
        "log_true_twa",
        "hidden_twa10",
        "hdr10",
        "true_auc",
    ]
    primary[r_columns].to_csv(
        intermediate_path(cfg, "aki_model_frame_v73.csv"), index=False
    )
    write_table(
        cfg,
        "asa_physical_status_data_quality.csv",
        summarize_asa_validity(manifest["asa"]),
    )

    presence = summarize_hidden_presence(primary)
    write_table(cfg, "aki_hidden_presence.csv", presence)

    sequential_results = [
        _fit_spec(
            primary,
            label="Absolute hidden burden, unadjusted",
            analysis_role="total association, unadjusted",
            exposure_term="hidden_twa10",
            exposure_label="hidden AUC per anaesthesia-hour at MAP <65 mm Hg",
            exposure_scale="per 10 mm Hg*min per anaesthesia-hour",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            adjustment="none",
            require_reference_hypotension=False,
        ),
        _fit_spec(
            primary,
            label="Absolute hidden burden, clinical adjusted",
            analysis_role="primary exploratory AKI association model",
            exposure_term="hidden_twa10",
            exposure_label="hidden AUC per anaesthesia-hour at MAP <65 mm Hg",
            exposure_scale="per 10 mm Hg*min per anaesthesia-hour",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            adjustment="clinical",
            require_reference_hypotension=False,
        ),
        _fit_spec(
            primary,
            label="Absolute hidden burden, clinical plus reference burden",
            analysis_role="incremental association beyond total reference burden",
            exposure_term="hidden_twa10",
            exposure_label="hidden AUC per anaesthesia-hour at MAP <65 mm Hg",
            exposure_scale="per 10 mm Hg*min per anaesthesia-hour",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            adjustment="clinical_plus_reference",
            require_reference_hypotension=False,
        ),
    ]
    primary_result = sequential_results[1]

    hdr65_result = _fit_spec(
        primary,
        label="HDR65 incremental model",
        analysis_role="secondary compositional incremental model",
        exposure_term="hdr10",
        exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
        exposure_scale="per 10-percentage-point increase",
        outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
        baseline_definition="VitalDB preop_cr clinical-information field",
        threshold=65.0,
    )
    results = [hdr65_result]

    asa_sensitivity_results = [
        _fit_spec(
            recorded_asa_primary,
            label="Absolute hidden burden, clinical adjusted (recorded ASA sensitivity)",
            analysis_role="ASA coding sensitivity for the primary exploratory association",
            exposure_term="hidden_twa10",
            exposure_label="hidden AUC per anaesthesia-hour at MAP <65 mm Hg",
            exposure_scale="per 10 mm Hg*min per anaesthesia-hour",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            adjustment="clinical",
            require_reference_hypotension=False,
        ),
        _fit_spec(
            recorded_asa_primary,
            label="Absolute hidden burden, clinical plus reference burden (recorded ASA sensitivity)",
            analysis_role="ASA coding sensitivity beyond total reference burden",
            exposure_term="hidden_twa10",
            exposure_label="hidden AUC per anaesthesia-hour at MAP <65 mm Hg",
            exposure_scale="per 10 mm Hg*min per anaesthesia-hour",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            adjustment="clinical_plus_reference",
            require_reference_hypotension=False,
        ),
        _fit_spec(
            recorded_asa_primary,
            label="HDR65 incremental model (recorded ASA sensitivity)",
            analysis_role="ASA coding sensitivity for the compositional incremental model",
            exposure_term="hdr10",
            exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
        ),
    ]

    for threshold in (60.0, 55.0):
        threshold_frame = _analysis_frame(manifest, outcomes, feature_sets[threshold])
        results.append(
            _fit_spec(
                threshold_frame,
                label=f"MAP <{int(threshold)} mm Hg threshold",
                analysis_role="threshold sensitivity",
                exposure_term="hdr10",
                exposure_label=f"hidden-deficit ratio at MAP <{int(threshold)} mm Hg",
                exposure_scale="per 10-percentage-point increase",
                outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
                baseline_definition="VitalDB preop_cr clinical-information field",
                threshold=threshold,
            )
        )

    results.append(
        _fit_spec(
            primary,
            label="Pre-display AUC fraction",
            analysis_role="actionability metric sensitivity",
            exposure_term="predisplay10",
            exposure_label="reference episode AUC accrued before first low display",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            additional_filter=primary["predisplay10"].notna(),
        )
    )
    outcome_48h = outcomes.copy()
    outcome_48h["aki_creatinine_48h"] = pd.Series(
        pd.NA, index=outcome_48h.index, dtype="boolean"
    )
    evaluable_48h = (
        outcome_48h["baseline_creatinine_mg_dl"].gt(0)
        & outcome_48h["postop_cr_max_48h"].notna()
    )
    outcome_48h.loc[evaluable_48h, "aki_creatinine_48h"] = (
        outcome_48h.loc[evaluable_48h, "postop_cr_max_48h"]
        .sub(outcome_48h.loc[evaluable_48h, "baseline_creatinine_mg_dl"])
        .ge(0.3)
        | outcome_48h.loc[evaluable_48h, "postop_cr_max_48h"]
        .div(outcome_48h.loc[evaluable_48h, "baseline_creatinine_mg_dl"])
        .ge(1.5)
    )
    outcome_48h["aki_creatinine_7d"] = outcome_48h["aki_creatinine_48h"]
    frame_48h = _analysis_frame(manifest, outcome_48h, feature_sets[65.0])
    results.append(
        _fit_spec(
            frame_48h,
            label="AKI within 48 h",
            analysis_role="outcome-window sensitivity",
            exposure_term="hdr10",
            exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined AKI within 48 h",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
        )
    )

    latest_baseline = derive_last_preoperative_creatinine(manifest, labs)
    alternate_manifest = manifest.merge(
        latest_baseline, on="case_id", how="left", validate="one_to_one"
    )
    alternate_manifest["preop_cr"] = alternate_manifest["last_preop_cr_90d"]
    alternate_outcomes = build_creatinine_aki_outcomes(alternate_manifest, labs)
    alternate_frame = _analysis_frame(
        alternate_manifest, alternate_outcomes, feature_sets[65.0]
    )
    results.append(
        _fit_spec(
            alternate_frame,
            label="Latest laboratory baseline within 90 d",
            analysis_role="baseline-definition sensitivity",
            exposure_term="hdr10",
            exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="latest measured creatinine within 90 d before anaesthesia start",
            threshold=65.0,
        )
    )

    results.append(
        _fit_spec(
            primary,
            label="At least two postoperative creatinine values",
            analysis_role="outcome-ascertainment sensitivity",
            exposure_term="hdr10",
            exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
            additional_filter=primary["postop_cr_count_7d"].ge(2),
        )
    )
    results.append(
        _fit_spec(
            primary,
            label="Exclude baseline creatinine >=4 mg/dL",
            analysis_role="advanced-kidney-dysfunction sensitivity",
            exposure_term="hdr10",
            exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr; baseline creatinine <4 mg/dL",
            threshold=65.0,
            additional_filter=primary["baseline_creatinine_mg_dl"].lt(4.0),
        )
    )
    one_case = primary.sort_values("case_id").drop_duplicates("subjectid", keep="first")
    results.append(
        _fit_spec(
            one_case,
            label="One case per subject",
            analysis_role="independence sensitivity",
            exposure_term="hdr10",
            exposure_label="hidden-deficit ratio at MAP <65 mm Hg",
            exposure_scale="per 10-percentage-point increase",
            outcome_definition="creatinine-defined KDIGO AKI through postoperative day 7",
            baseline_definition="VitalDB preop_cr clinical-information field",
            threshold=65.0,
        )
    )

    ordered_columns = [
        "analysis",
        "analysis_role",
        "outcome_definition",
        "baseline_definition",
        "map_threshold_mm_hg",
        "display_interval_min",
        "exposure",
        "exposure_scale",
        "population",
        "adjustment_sequence",
        "n_cases",
        "n_subjects",
        "events",
        "n_parameters",
        "relative_risk",
        "ci_low",
        "ci_high",
        "p_value",
        "positive_adverse_association_supported",
        "reference_burden_adjustment",
        "covariate_adjustment",
        "variance_estimator",
        "causal_boundary",
        "aic",
        "log_likelihood",
        "formula",
        "term",
        "coefficient",
        "standard_error",
    ]
    sequential_table = pd.DataFrame(sequential_results)[ordered_columns]
    write_table(cfg, "exploratory_aki_sequential_models.csv", sequential_table)

    result_table = pd.DataFrame(results)
    result_table = result_table[ordered_columns]
    write_table(cfg, "exploratory_aki_incremental_models.csv", result_table)
    asa_sensitivity_table = pd.DataFrame(asa_sensitivity_results)[ordered_columns]
    write_table(
        cfg,
        "exploratory_aki_asa_coding_sensitivity.csv",
        asa_sensitivity_table,
    )
    pd.concat([sequential_table, result_table], ignore_index=True).to_csv(
        intermediate_path(cfg, "aki_models_for_r_verification_v73.csv"),
        index=False,
    )
    main_table_rows = [
        {
            "analysis": "Any hidden burden versus none",
            "analysis_role": "binary presence estimability",
            "population": "all AKI-evaluable cases",
            "adjustment": "descriptive 2x2 table",
            "exposure": "any hidden AUC >0 at MAP <65 mm Hg",
            "exposure_scale": "presence versus absence",
            "n_cases": int(presence.loc[0, "n_evaluable"]),
            "events": int(
                presence.loc[0, "absent_events"]
                + presence.loc[0, "present_events"]
            ),
            "reference_group": (
                f"Absent: {int(presence.loc[0, 'absent_n'])} cases / "
                f"{int(presence.loc[0, 'absent_events'])} AKI"
            ),
            "exposed_group": (
                f"Present: {int(presence.loc[0, 'present_n'])} cases / "
                f"{int(presence.loc[0, 'present_events'])} AKI"
            ),
            "effect_measure": "not estimated",
            "relative_risk": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "estimation_status": presence.loc[0, "estimability_status"],
            "interpretation": presence.loc[0, "estimability_reason"],
        }
    ]
    for row in sequential_table.itertuples(index=False):
        main_table_rows.append(
            {
                "analysis": row.analysis,
                "analysis_role": row.analysis_role,
                "population": row.population,
                "adjustment": row.adjustment_sequence,
                "exposure": row.exposure,
                "exposure_scale": row.exposure_scale,
                "n_cases": int(row.n_cases),
                "events": int(row.events),
                "reference_group": "",
                "exposed_group": "",
                "effect_measure": "relative risk",
                "relative_risk": row.relative_risk,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "p_value": row.p_value,
                "estimation_status": "ESTIMATED",
                "interpretation": row.analysis_role,
            }
        )
    main_table_rows.append(
        {
            "analysis": hdr65_result["analysis"],
            "analysis_role": hdr65_result["analysis_role"],
            "population": hdr65_result["population"],
            "adjustment": hdr65_result["adjustment_sequence"],
            "exposure": hdr65_result["exposure"],
            "exposure_scale": hdr65_result["exposure_scale"],
            "n_cases": int(hdr65_result["n_cases"]),
            "events": int(hdr65_result["events"]),
            "reference_group": "",
            "exposed_group": "",
            "effect_measure": "relative risk",
            "relative_risk": hdr65_result["relative_risk"],
            "ci_low": hdr65_result["ci_low"],
            "ci_high": hdr65_result["ci_high"],
            "p_value": hdr65_result["p_value"],
            "estimation_status": "ESTIMATED",
            "interpretation": "secondary compositional incremental association",
        }
    )
    main_table = pd.DataFrame(main_table_rows)
    write_table(cfg, "main_table4_exploratory_aki.csv", main_table)

    n_total = len(primary)
    n_subjects = int(primary["subjectid"].nunique())
    repeated_subjects = int(primary["subjectid"].value_counts().gt(1).sum())
    n_baseline = int(outcomes["baseline_creatinine_mg_dl"].notna().sum())
    n_post48 = int(outcomes["postop_cr_count_48h"].gt(0).sum())
    n_post7 = int(outcomes["postop_cr_count_7d"].gt(0).sum())
    n_evaluable = int(outcomes["aki_evaluable"].sum())
    n_aki = int(outcomes["aki_creatinine_7d"].fillna(False).sum())
    stage_counts = (
        outcomes["aki_stage_creatinine"].value_counts(dropna=False).to_dict()
    )
    events_per_parameter = primary_result["events"] / primary_result["n_parameters"]
    gate = pd.DataFrame(
        [
            _gate_row("Primary arterial cohort", n_total, None, "informational", True),
            _gate_row("Unique subjects", n_subjects, n_total, "informational", True),
            _gate_row("Subjects with repeated cases", repeated_subjects, n_subjects, "subject-cluster variance required", True),
            _gate_row("Baseline preoperative creatinine available", n_baseline, n_total, ">=90%", n_baseline / n_total >= 0.90),
            _gate_row("Postoperative creatinine within 48 h", n_post48, n_total, ">=90%", n_post48 / n_total >= 0.90),
            _gate_row("Postoperative creatinine within 7 d or discharge", n_post7, n_total, ">=90%", n_post7 / n_total >= 0.90),
            _gate_row("AKI-evaluable cases", n_evaluable, n_total, ">=90%", n_evaluable / n_total >= 0.90),
            _gate_row("Creatinine-defined AKI events", n_aki, n_evaluable, ">=100 events for compact exploratory model", n_aki >= 100),
            _gate_row("KDIGO creatinine stage 1", int(stage_counts.get(1, 0)), n_evaluable, "informational", True),
            _gate_row("KDIGO creatinine stage 2", int(stage_counts.get(2, 0)), n_evaluable, "informational", True),
            _gate_row("KDIGO creatinine stage 3", int(stage_counts.get(3, 0)), n_evaluable, "creatinine criteria only", True),
            _gate_row("Primary adjusted-model cases", primary_result["n_cases"], n_evaluable, "informational", True),
            _gate_row("Primary adjusted-model events", primary_result["events"], primary_result["n_cases"], ">=5 events per model parameter", events_per_parameter >= 5.0),
        ]
    )
    gate = pd.concat(
        [
            gate,
            pd.DataFrame(
                [
                    {
                        "criterion": "Binary hidden-presence effect estimability",
                        "value": int(presence.loc[0, "present_n"]),
                        "denominator": int(presence.loc[0, "n_evaluable"]),
                        "proportion": float(
                            presence.loc[0, "present_n"]
                            / presence.loc[0, "n_evaluable"]
                        ),
                        "pass_rule": (
                            "report descriptively and do not fit an ordinary binary "
                            "model when any exposure-outcome cell is zero"
                        ),
                        "status": presence.loc[0, "estimability_status"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    gate["events_per_model_parameter"] = np.nan
    gate.loc[
        gate["criterion"].eq("Primary adjusted-model events"),
        "events_per_model_parameter",
    ] = events_per_parameter
    write_table(cfg, "aki_outcome_gate0.csv", gate)
    if bool(gate["status"].eq("NOT PASS").any()):
        failed = gate.loc[gate["status"].eq("NOT PASS"), "criterion"].tolist()
        raise RuntimeError(f"AKI outcome Gate 0 failed: {failed}")

    stage_table = pd.DataFrame(
        [
            {"outcome": "No creatinine-defined AKI", "n": int(stage_counts.get(0, 0)), "denominator": n_evaluable},
            {"outcome": "KDIGO creatinine stage 1", "n": int(stage_counts.get(1, 0)), "denominator": n_evaluable},
            {"outcome": "KDIGO creatinine stage 2", "n": int(stage_counts.get(2, 0)), "denominator": n_evaluable},
            {"outcome": "KDIGO creatinine stage 3", "n": int(stage_counts.get(3, 0)), "denominator": n_evaluable},
            {"outcome": "Not evaluable", "n": n_total - n_evaluable, "denominator": n_total},
        ]
    )
    stage_table["proportion"] = stage_table["n"] / stage_table["denominator"]
    write_table(cfg, "aki_outcome_summary.csv", stage_table)
    write_table(cfg, "aki_outcome_missingness_comparison.csv", _missingness_comparison(primary))

    quartile = primary[
        primary["true_auc"].gt(0) & primary["aki"].notna() & primary["hdr"].notna()
    ].copy()
    quartile["hdr_quartile"] = pd.qcut(
        quartile["hdr"],
        4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
        duplicates="drop",
    )
    quartile_table = quartile.groupby("hdr_quartile", observed=True).agg(
        n_cases=("case_id", "size"),
        aki_events=("aki", "sum"),
        aki_risk=("aki", "mean"),
        hdr_median=("hdr", "median"),
        hdr_min=("hdr", "min"),
        hdr_max=("hdr", "max"),
        reference_twa_median=("true_twa", "median"),
    ).reset_index()
    write_table(cfg, "aki_unadjusted_by_hdr_quartile.csv", quartile_table)

    write_qc(
        cfg,
        "exploratory_aki_analysis_qc.md",
        "# Exploratory AKI analysis quality control\n\n"
        f"All Gate 0 criteria passed. The arterial cohort contained {n_total:,} cases "
        f"from {n_subjects:,} subjects. Creatinine-defined postoperative AKI was "
        f"evaluable in {n_evaluable:,} cases; {n_aki:,} met serum-creatinine criteria "
        f"({100 * n_aki / n_evaluable:.1f}%). Hidden burden was present in "
        f"{int(presence.loc[0, 'present_n']):,} cases and absent in "
        f"{int(presence.loc[0, 'absent_n']):,}; the absent group contained "
        f"{int(presence.loc[0, 'absent_events'])} AKI events. The binary contrast was "
        f"therefore classified as {presence.loc[0, 'estimability_status']} and was "
        "not entered into an ordinary adjusted regression.\n\n"
        f"The primary exploratory association model included {primary_result['n_cases']:,} "
        f"cases and {primary_result['events']:,} events. Per 10 mm Hg*min/h greater "
        f"absolute hidden burden, the clinical-adjusted RR was "
        f"{primary_result['relative_risk']:.3f} (95% CI "
        f"{primary_result['ci_low']:.3f} to {primary_result['ci_high']:.3f}; "
        f"P={primary_result['p_value']:.3f}). After adding nonlinear total reference "
        f"burden, the RR was {sequential_results[2]['relative_risk']:.3f} (95% CI "
        f"{sequential_results[2]['ci_low']:.3f} to "
        f"{sequential_results[2]['ci_high']:.3f}; "
        f"P={sequential_results[2]['p_value']:.3f}).\n\n"
        "AKI was defined using serum creatinine only: an increase of at least 0.3 mg/dL "
        "within 48 h after anaesthesia end or at least 1.5 times baseline within 7 d, "
        "censored at discharge. The primary baseline was VitalDB preop_cr. Urine-output "
        "and renal-replacement-therapy criteria were unavailable.\n\n"
        "Regression estimates used modified Poisson models with subject-cluster robust "
        "standard errors. Sequential models separated the total association of absolute "
        "hidden burden from its incremental association after a 3-df quadratic B-spline "
        "of log(1 + total reference AUC per anaesthesia-hour). HDR remained a secondary "
        "compositional estimand among cases with positive reference AUC. These post hoc "
        "associations under counterfactual display emulation are noncausal.\n",
    )
    _make_forest(cfg, sequential_table, result_table)


if __name__ == "__main__":
    run(load_args())
