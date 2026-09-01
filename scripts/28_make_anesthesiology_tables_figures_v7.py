from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import load_args
from ioh.pipeline import (
    _frequency_population_table,
    intermediate_path,
    out_root,
    table_path,
    write_table,
)
from ioh.revision_v7 import (
    phase_averaged_episode_observability,
    phase_averaged_initial_display_decomposition,
    summarize_episode_observability,
)


COLORS = {
    55.0: "#0072B2",
    60.0: "#D55E00",
    65.0: "#009E73",
    "hidden": "#C44E52",
    "stale": "#4C72B0",
    "reference": "#222222",
    "display": "#E69F00",
}


def _case_mean_from_offsets(offset: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [
        "true_auc",
        "display_auc",
        "hidden_auc",
        "overdisplay_auc",
        "concordant_auc",
        "hidden_auc_display_unavailable",
        "hidden_auc_display_valid",
        "true_auc_display_valid",
        "display_auc_display_valid",
        "normotensive_display_min",
        "normotensive_display_discordance_min",
        "reference_hypotension_with_display_unavailable_min",
        "hypotensive_display_discordance_min",
        "true_hypotension_min",
        "display_valid_min",
        "display_unavailable_min",
        "episode_count",
        "episode_detected",
        "missed_episodes",
        "mean_detection_delay_min",
        "anesthesia_hours",
    ]
    return (
        offset.groupby(["case_id", "threshold", "interval_min"], dropna=False)[
            mean_columns
        ]
        .mean()
        .reset_index()
    )


def _case_to_subject(cfg: dict, case_ids) -> pd.DataFrame:
    manifest = pd.read_parquet(
        intermediate_path(cfg, "vitaldb_manifest.parquet"),
        columns=["case_id", "subjectid"],
    )
    mapping = manifest[manifest["case_id"].isin(set(case_ids))][
        ["case_id", "subjectid"]
    ].drop_duplicates()
    if mapping["case_id"].nunique() != len(set(case_ids)):
        raise AssertionError("every analyzed case must have one subject identifier")
    return mapping


def _write_initial_display_policy_sensitivity(
    cfg: dict,
    case_to_subject: pd.DataFrame,
    primary_auc: pd.DataFrame,
) -> None:
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    threshold = float(cfg["analysis"]["primary_threshold"])
    interval_min = float(cfg["analysis"]["reference_interval_min"])
    interval_sec = int(round(interval_min * 60.0))
    step_sec = int(cfg["analysis"]["resample_sec"])
    frames: list[pd.DataFrame] = []
    case_groups = panel.groupby("case_id", sort=False)
    for index, (case_id, sub) in enumerate(case_groups, start=1):
        sub = sub.sort_values("time_sec")
        frame = phase_averaged_initial_display_decomposition(
            sub["time_sec"].to_numpy(float),
            sub["art_map"].to_numpy(float),
            threshold=threshold,
            interval_sec=interval_sec,
            step_sec=step_sec,
        )
        frame.insert(0, "case_id", case_id)
        frames.append(frame)
        if index % 250 == 0:
            print(
                f"Initial-display sensitivity: {index:,}/{panel['case_id'].nunique():,} cases",
                flush=True,
            )
    case_table = pd.concat(frames, ignore_index=True)
    case_table.to_parquet(
        intermediate_path(cfg, "initial_display_policy_case_v10.parquet"),
        index=False,
    )

    descriptions = {
        "unavailable_before_first_scheduled_sample": (
            "Primary convention: no displayed value before the first scheduled sample; "
            "concurrent reference deficit is counted as hidden."
        ),
        "baseline_initialized_at_first_reference": (
            "Sensitivity: initialize the display with the first finite reference value, "
            "then follow the same phase-specific schedule."
        ),
        "exclude_before_first_display": (
            "Sensitivity: exclude reference time points before the first finite displayed value "
            "from both numerator and denominator."
        ),
    }
    summary_rows: list[pd.DataFrame] = []
    for policy, sub in case_table.groupby("initial_display_policy", sort=False):
        summary = _frequency_population_table(
            cfg,
            sub,
            case_to_cluster=case_to_subject,
            cluster_column="subjectid",
        )
        summary.insert(0, "initial_display_policy", policy)
        summary.insert(1, "policy_description", descriptions[policy])
        excluded_total = float(sub["excluded_reference_time_min"].sum())
        reference_minutes = float(sub["anesthesia_hours"].sum()) * 60.0
        summary["excluded_reference_time_min_total"] = excluded_total
        summary["excluded_reference_time_fraction"] = (
            excluded_total / reference_minutes if reference_minutes > 0 else np.nan
        )
        summary_rows.append(summary)
    sensitivity = pd.concat(summary_rows, ignore_index=True)
    order = {name: index for index, name in enumerate(descriptions)}
    sensitivity["_order"] = sensitivity["initial_display_policy"].map(order)
    sensitivity = sensitivity.sort_values("_order").drop(columns="_order")

    primary = sensitivity[
        sensitivity["initial_display_policy"].eq(
            "unavailable_before_first_scheduled_sample"
        )
    ].iloc[0]
    frozen = primary_auc[
        primary_auc["threshold"].eq(threshold)
        & primary_auc["interval_min"].eq(interval_min)
    ].iloc[0]
    for metric in ("HDR", "ODR", "NetBias"):
        if not np.isclose(primary[metric], frozen[metric], rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"recomputed initial-display primary {metric} does not match frozen primary"
            )
    write_table(cfg, "initial_display_policy_sensitivity.csv", sensitivity)


def _write_initial_display_episode_sensitivity(
    cfg: dict,
    case_to_subject: pd.DataFrame,
) -> None:
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    threshold = float(cfg["analysis"]["primary_threshold"])
    interval_min = float(cfg["analysis"]["reference_interval_min"])
    interval_sec = int(round(interval_min * 60.0))
    step_sec = int(cfg["analysis"]["resample_sec"])
    min_duration_sec = 60
    case_ids = panel["case_id"].drop_duplicates().tolist()

    primary = pd.read_parquet(
        intermediate_path(cfg, "episode_observability_case_5min.parquet")
    )
    primary = primary[
        primary["threshold"].eq(threshold)
        & primary["interval_min"].eq(interval_min)
    ].copy()
    primary["initial_display_episode_policy"] = (
        "unavailable_before_first_scheduled_sample"
    )

    sensitivity_rows: list[dict] = []
    for index, (case_id, sub) in enumerate(panel.groupby("case_id", sort=False), start=1):
        sub = sub.sort_values("time_sec")
        times = sub["time_sec"].to_numpy(float)
        reference = sub["art_map"].to_numpy(float)
        baseline = phase_averaged_episode_observability(
            times,
            reference,
            threshold=threshold,
            interval_sec=interval_sec,
            min_duration_sec=min_duration_sec,
            step_sec=step_sec,
            initialize_at_start=True,
        )
        sensitivity_rows.append(
            {
                "case_id": case_id,
                "threshold": threshold,
                "interval_min": interval_min,
                "initial_display_episode_policy": "baseline_initialized_at_first_reference",
                **baseline,
            }
        )

        after_first_interval = times >= float(interval_sec)
        excluded = phase_averaged_episode_observability(
            times[after_first_interval] - float(interval_sec),
            reference[after_first_interval],
            threshold=threshold,
            interval_sec=interval_sec,
            min_duration_sec=min_duration_sec,
            step_sec=step_sec,
            initialize_at_start=True,
        )
        sensitivity_rows.append(
            {
                "case_id": case_id,
                "threshold": threshold,
                "interval_min": interval_min,
                "initial_display_episode_policy": "exclude_first_sampling_interval_then_initialize",
                **excluded,
            }
        )
        if index % 250 == 0:
            print(
                f"Initial-display episode sensitivity: {index:,}/{len(case_ids):,} cases",
                flush=True,
            )

    sensitivity_case = pd.DataFrame(sensitivity_rows)
    sensitivity_case.to_parquet(
        intermediate_path(cfg, "initial_display_episode_policy_case_v10.parquet"),
        index=False,
    )
    combined = pd.concat([primary, sensitivity_case], ignore_index=True, sort=False)
    descriptions = {
        "unavailable_before_first_scheduled_sample": (
            "Primary convention: no displayed value before the first scheduled sample."
        ),
        "baseline_initialized_at_first_reference": (
            "Initialize the display with the first finite reference value at analytic start."
        ),
        "exclude_first_sampling_interval_then_initialize": (
            "Exclude the first 5 min of the analytic interval, then initialize the display "
            "with the first finite remaining reference value."
        ),
    }
    summaries: list[pd.DataFrame] = []
    for policy, sub in combined.groupby(
        "initial_display_episode_policy", sort=False
    ):
        summary = summarize_episode_observability(
            sub,
            all_case_ids=case_ids,
            group_columns=["threshold", "interval_min"],
            bootstrap_reps=int(cfg["analysis"]["bootstrap_reps"]),
            seed=int(cfg["project"]["seed"]),
            case_to_cluster=case_to_subject,
            cluster_column="subjectid",
        )
        summary.insert(0, "initial_display_episode_policy", policy)
        summary.insert(1, "policy_description", descriptions[policy])
        summaries.append(summary)
    output = pd.concat(summaries, ignore_index=True)
    order = {name: index for index, name in enumerate(descriptions)}
    output["_order"] = output["initial_display_episode_policy"].map(order)
    output = output.sort_values("_order").drop(columns="_order")

    frozen = pd.read_csv(table_path(cfg, "episode_observability_5min_overall.csv"))
    frozen = frozen[
        frozen["threshold"].eq(threshold)
        & frozen["interval_min"].eq(interval_min)
    ].iloc[0]
    primary_summary = output[
        output["initial_display_episode_policy"].eq(
            "unavailable_before_first_scheduled_sample"
        )
    ].iloc[0]
    for metric in (
        "episode_detection_probability",
        "complete_miss_probability",
        "reference_auc_before_first_low_display_fraction",
    ):
        if not np.isclose(primary_summary[metric], frozen[metric], rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"recomputed initial-display episode primary {metric} does not match frozen primary"
            )
    write_table(cfg, "initial_display_episode_sensitivity.csv", output)


def _write_bootstrap_cluster_sensitivity(
    cfg: dict,
    subject_auc: pd.DataFrame,
    case_auc: pd.DataFrame,
) -> None:
    subject_episode = pd.read_csv(
        table_path(cfg, "episode_observability_5min_overall.csv")
    )
    case_episode = pd.read_csv(
        table_path(
            cfg,
            "episode_observability_5min_overall_case_cluster_sensitivity.csv",
        )
    )
    selectors = lambda frame: frame[
        frame["threshold"].eq(65.0) & frame["interval_min"].eq(5.0)
    ].iloc[0]
    auc_subject_row = selectors(subject_auc)
    auc_case_row = selectors(case_auc)
    episode_subject_row = selectors(subject_episode)
    episode_case_row = selectors(case_episode)
    rows: list[dict] = []

    def add_pair(domain, metric, point_column, low_column, high_column, left, right):
        if not np.isclose(left[point_column], right[point_column], rtol=0.0, atol=1e-12):
            raise AssertionError(f"{metric} point estimate changed with bootstrap cluster")
        for row in (left, right):
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "point_estimate": row[point_column],
                    "ci_low": row[low_column],
                    "ci_high": row[high_column],
                    "bootstrap_cluster": row["bootstrap_cluster"],
                    "n_clusters": row.get("n_clusters", row.get("n_clusters_total")),
                    "bootstrap_reps": row["bootstrap_reps"],
                }
            )

    for metric, point, low, high in (
        ("hidden deficit ratio", "HDR", "HDR_ci_low", "HDR_ci_high"),
        ("overdisplay deficit ratio", "ODR", "ODR_ci_low", "ODR_ci_high"),
        ("net relative AUC difference", "NetBias", "NetBias_ci_low", "NetBias_ci_high"),
    ):
        add_pair("AUC decomposition", metric, point, low, high, auc_subject_row, auc_case_row)
    for metric, point in (
        ("episode detection probability", "episode_detection_probability"),
        ("complete-miss probability", "complete_miss_probability"),
        ("first low display with at least 1 min remaining", "detection_with_1min_remaining_probability"),
        ("first low display with at least 2 min remaining", "detection_with_2min_remaining_probability"),
        ("reference AUC before first low display", "reference_auc_before_first_low_display_fraction"),
    ):
        add_pair(
            "episode observability",
            metric,
            point,
            f"{point}_ci_low" if point.startswith("detection_with") or point.startswith("reference_auc") else point.replace("probability", "ci_low"),
            f"{point}_ci_high" if point.startswith("detection_with") or point.startswith("reference_auc") else point.replace("probability", "ci_high"),
            episode_subject_row,
            episode_case_row,
        )
    write_table(cfg, "bootstrap_cluster_sensitivity.csv", pd.DataFrame(rows))


def _figure1_flow_source(
    cohort_flow: pd.DataFrame, nibp_count: dict[str, int]
) -> pd.DataFrame:
    values = cohort_flow.set_index("step")
    primary_specs = [
        ("VitalDB source cases", "source_cases", None),
        ("Adults", "adult", "age <18 yr"),
        ("General anaesthesia", "general_anaesthesia", "not general anaesthesia"),
        ("Eligible surgery groups", "eligible_surgery", "prespecified surgery groups"),
        ("Anaesthesia duration >=60 min", "duration_ge_60min", "anaesthesia duration <60 min"),
        ("Pre-QC arterial candidates", "has_art_map", "no candidate arterial MAP track"),
        ("Primary arterial waveform cohort", "art_coverage_ge_80pct", "valid arterial MAP coverage <80%"),
    ]
    rows = [
        {
            "branch": "primary",
            "stage": label,
            "n_cases": int(values.loc[step, "remaining"]),
            "excluded_from_previous": int(values.loc[step, "excluded"]),
            "exclusion_reason": reason,
        }
        for label, step, reason in primary_specs
    ]
    prior = rows[-1]["n_cases"]
    for stage, key, reason in (
        ("Primary cohort with NIBP track", "primary_art_coverage_processing_pool", "no NIBP MAP track"),
        ("Reconstructed NIBP display events", "reconstructed_nibp_display_events", "display event reconstruction unavailable"),
        ("Cases with paired NIBP-ART events", "paired_nibp_events_primary_window", "no paired event in the primary window"),
    ):
        current = int(nibp_count[key])
        rows.append(
            {
                "branch": "NIBP agreement",
                "stage": stage,
                "n_cases": current,
                "excluded_from_previous": prior - current,
                "exclusion_reason": reason,
            }
        )
        prior = current
    return pd.DataFrame(rows)


def _write_publication_tables(cfg: dict) -> None:
    offset = pd.read_parquet(
        intermediate_path(cfg, "frequency_decomp_case_offset.parquet")
    )
    case_mean = _case_mean_from_offsets(offset)
    case_mean.to_parquet(
        intermediate_path(cfg, "frequency_decomp_case_mean_v7.parquet"), index=False
    )
    case_to_subject = _case_to_subject(cfg, case_mean["case_id"].unique())
    auc = _frequency_population_table(
        cfg,
        case_mean,
        case_to_cluster=case_to_subject,
        cluster_column="subjectid",
    )
    auc_case_cluster = _frequency_population_table(cfg, case_mean)
    write_table(cfg, "frequency_decomposition_bootstrap.csv", auc)
    write_table(cfg, "primary_auc_by_threshold_interval.csv", auc)
    write_table(
        cfg,
        "frequency_decomposition_case_cluster_sensitivity.csv",
        auc_case_cluster,
    )
    _write_initial_display_policy_sensitivity(cfg, case_to_subject, auc)
    _write_initial_display_episode_sensitivity(cfg, case_to_subject)
    _write_bootstrap_cluster_sensitivity(cfg, auc, auc_case_cluster)

    overall = pd.read_csv(table_path(cfg, "episode_observability_5min_overall.csv"))
    duration = pd.read_csv(
        table_path(cfg, "episode_observability_5min_by_duration.csv")
    )
    overall65 = overall[overall["threshold"].eq(65.0)].copy()
    overall65["stratum"] = "MAP <65 mm Hg, all episodes"
    duration65 = duration[duration["threshold"].eq(65.0)].copy()
    duration65["stratum"] = "MAP <65 mm Hg, " + duration65["duration_bin"]
    overall55 = overall[overall["threshold"].eq(55.0)].copy()
    overall55["stratum"] = "MAP <55 mm Hg, all episodes"
    duration55 = duration[
        duration["threshold"].eq(55.0)
        & duration["duration_bin"].eq("1-<3 min")
    ].copy()
    duration55["stratum"] = "MAP <55 mm Hg, 1-<3 min"
    table2_columns = [
        "stratum",
        "n_cases_total",
        "cases_with_episodes",
        "reference_episodes",
        "expected_detected_episodes",
        "expected_missed_episodes",
        "episode_detection_probability",
        "episode_detection_ci_low",
        "episode_detection_ci_high",
        "complete_miss_probability",
        "complete_miss_ci_low",
        "complete_miss_ci_high",
        "detection_with_1min_remaining_probability",
        "detection_with_1min_remaining_probability_ci_low",
        "detection_with_1min_remaining_probability_ci_high",
        "detection_with_2min_remaining_probability",
        "detection_with_2min_remaining_probability_ci_low",
        "detection_with_2min_remaining_probability_ci_high",
        "reference_auc_before_first_low_display_fraction",
        "reference_auc_before_first_low_display_fraction_ci_low",
        "reference_auc_before_first_low_display_fraction_ci_high",
        "mean_detection_delay_min",
        "mean_detection_delay_min_ci_low",
        "mean_detection_delay_min_ci_high",
        "mean_remaining_reference_time_min",
        "mean_remaining_reference_time_min_ci_low",
        "mean_remaining_reference_time_min_ci_high",
        "mean_stale_display_after_recovery_min",
        "mean_stale_display_after_recovery_min_ci_low",
        "mean_stale_display_after_recovery_min_ci_high",
    ]
    main_table2 = pd.concat(
        [
            overall65[table2_columns],
            duration65[table2_columns],
            overall55[table2_columns],
            duration55[table2_columns],
        ],
        ignore_index=True,
    )
    order = {
        "MAP <65 mm Hg, all episodes": 0,
        "MAP <65 mm Hg, 1-<3 min": 1,
        "MAP <65 mm Hg, 3-<5 min": 2,
        "MAP <65 mm Hg, >=5 min": 3,
        "MAP <55 mm Hg, all episodes": 4,
        "MAP <55 mm Hg, 1-<3 min": 5,
    }
    main_table2["sort_order"] = main_table2["stratum"].map(order)
    main_table2 = main_table2.sort_values("sort_order").drop(columns="sort_order")
    write_table(cfg, "main_table2_map65_5min_event_observability.csv", main_table2)

    episode_interval = pd.read_csv(
        table_path(cfg, "episode_observability_by_threshold_interval.csv")
    )
    episode65 = episode_interval[episode_interval["threshold"].eq(65.0)].copy()
    auc65 = auc[auc["threshold"].eq(65.0)].copy()
    main_table3 = episode65.merge(
        auc65[
            [
                "interval_min",
                "HDR",
                "HDR_ci_low",
                "HDR_ci_high",
                "ODR",
                "ODR_ci_low",
                "ODR_ci_high",
                "NetBias",
                "NetBias_ci_low",
                "NetBias_ci_high",
                "normotensive_display_discordance_min_per_anesthesia_hour",
                "normotensive_display_discordance_ci_low",
                "normotensive_display_discordance_ci_high",
                "hypotensive_display_discordance_min_per_anesthesia_hour",
                "hypotensive_display_discordance_ci_low",
                "hypotensive_display_discordance_ci_high",
            ]
        ],
        on="interval_min",
        how="left",
        validate="one_to_one",
    )
    write_table(cfg, "main_table3_map65_sampling_interval_sensitivity.csv", main_table3)

    nibp_repeated = pd.read_csv(
        table_path(cfg, "nibp_repeated_measures_bland_altman.csv")
    )
    nibp_classification = pd.read_csv(
        table_path(cfg, "nibp_paired_classification_clustered.csv")
    )
    primary_agreement = nibp_repeated[
        nibp_repeated["method"].eq("random_intercept_variance_components_mom")
    ].copy()
    write_table(cfg, "supp_nibp_primary_repeated_measures_agreement.csv", primary_agreement)
    write_table(cfg, "supp_nibp_paired_classification.csv", nibp_classification)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        figure_dir / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _figure1(cfg: dict) -> None:
    cohort_flow = pd.read_csv(table_path(cfg, "cohort_flow.csv"))
    nibp_flow = pd.read_csv(table_path(cfg, "nibp_case_event_flow_v7.csv"))
    count = dict(zip(nibp_flow["step"], nibp_flow["n_cases"]))
    flow = _figure1_flow_source(cohort_flow, count)
    write_table(cfg, "figure1_cohort_flow_source.csv", flow)

    time_min = np.arange(0, 10.01, 1 / 6)
    reference = np.full_like(time_min, 74.0)
    reference[(time_min >= 1.5) & (time_min < 3.5)] = 61.0
    reference[(time_min >= 4.5) & (time_min < 6.5)] = 61.0
    display = np.full_like(time_min, np.nan)
    display[(time_min >= 0.0) & (time_min < 5.0)] = 74.0
    display[(time_min >= 5.0) & (time_min < 10.0)] = 61.0
    display[time_min >= 10.0] = 74.0
    schematic = pd.DataFrame(
        {"time_min": time_min, "arterial_map": reference, "displayed_map": display}
    )
    write_table(cfg, "figure1_schematic_source.csv", schematic)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 4.15),
        gridspec_kw={"width_ratios": [1.25, 1.45]},
    )
    ax = axes[0]
    ax.axis("off")
    flow_index = flow.set_index("stage")
    y_positions = [0.94, 0.78, 0.62, 0.46]
    labels = [
        ("VitalDB source", int(flow_index.loc["VitalDB source cases", "n_cases"])),
        (
            "Eligible clinical cohort",
            int(flow_index.loc["Anaesthesia duration >=60 min", "n_cases"]),
        ),
        (
            "Pre-QC arterial candidates",
            int(flow_index.loc["Pre-QC arterial candidates", "n_cases"]),
        ),
        (
            "Primary waveform cohort",
            int(flow_index.loc["Primary arterial waveform cohort", "n_cases"]),
        ),
    ]
    for y, (label, number) in zip(y_positions, labels):
        ax.text(
            0.34,
            y,
            f"{label}\n{number:,} cases",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7.8,
            bbox={"boxstyle": "round,pad=0.24", "fc": "#F4F4F4", "ec": "#555555", "lw": 0.8},
        )
    for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate(
            "",
            xy=(0.34, y2 + 0.055),
            xytext=(0.34, y1 - 0.055),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#555555"},
        )
    exclusion_labels = [
        (
            0.86,
            "Excluded 939\nAge <18: 57; not general\nanaesthesia: 342; surgery\ngroups: 430; duration <60 min: 110",
        ),
        (0.70, "Excluded 2,134\nNo candidate arterial MAP track"),
        (0.54, "Excluded 880\nValid arterial MAP coverage <80%"),
    ]
    for y, label in exclusion_labels:
        ax.text(
            0.64,
            y,
            label,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=6.3,
            color="#444444",
        )
    ax.text(0.02, 0.37, "NIBP agreement branch", transform=ax.transAxes, weight="bold", fontsize=8)
    branch = [
        ("NIBP processing pool", int(count["primary_art_coverage_processing_pool"])),
        ("Display events reconstructed", int(count["reconstructed_nibp_display_events"])),
        ("Cases with paired events", int(count["paired_nibp_events_primary_window"])),
    ]
    for index, (label, number) in enumerate(branch):
        y = 0.29 - index * 0.12
        ax.text(
            0.42,
            y,
            f"{label}: {number:,}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7.4,
            bbox={"boxstyle": "round,pad=0.22", "fc": "#EAF2F8", "ec": "#4C72B0", "lw": 0.8},
        )
    ax.text(-0.02, 1.01, "A", transform=ax.transAxes, weight="bold", fontsize=11)

    ax = axes[1]
    ax.step(time_min, reference, where="post", color=COLORS["reference"], lw=1.7, label="Arterial reference")
    ax.step(time_min, display, where="post", color=COLORS["display"], lw=1.7, label="5-min displayed MAP")
    ax.axhline(65, color="#777777", lw=0.9, ls="--", label="65 mm Hg threshold")
    ax.fill_between(
        time_min,
        reference,
        65,
        where=(reference < 65) & (display >= 65),
        step="post",
        color=COLORS["hidden"],
        alpha=0.35,
        label="Hidden interval",
    )
    ax.fill_between(
        time_min,
        display,
        65,
        where=(display < 65) & (reference >= 65),
        step="post",
        color=COLORS["stale"],
        alpha=0.30,
        label="Stale low display",
    )
    ax.scatter([0, 5, 10], [74, 61, 74], color=COLORS["display"], s=18, zorder=4)
    ax.set_xlim(0, 10)
    ax.set_ylim(54, 78)
    ax.set_xlabel("Time from analytic start, min")
    ax.set_ylabel("MAP, mm Hg")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    ax.text(-0.13, 1.02, "B", transform=ax.transAxes, weight="bold", fontsize=11)
    fig.subplots_adjust(bottom=0.23, wspace=0.40)
    _save_figure(fig, out_root(cfg) / "figures", "Figure_1_study_flow_and_observability")


def _figure2(cfg: dict) -> None:
    interval = pd.read_csv(
        table_path(cfg, "episode_observability_by_threshold_interval.csv")
    )
    duration = pd.read_csv(
        table_path(cfg, "episode_observability_5min_by_duration.csv")
    )
    source = pd.concat(
        [
            interval.assign(panel="sampling_interval"),
            duration.assign(panel="duration_at_5min"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_table(cfg, "figure2_episode_observability_source.csv", source)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    for threshold in [55.0, 60.0, 65.0]:
        sub = interval[interval["threshold"].eq(threshold)].sort_values("interval_min")
        ax.errorbar(
            sub["interval_min"],
            sub["episode_detection_probability"],
            yerr=np.vstack(
                [
                    sub["episode_detection_probability"] - sub["episode_detection_ci_low"],
                    sub["episode_detection_ci_high"] - sub["episode_detection_probability"],
                ]
            ),
            marker="o",
            ms=4,
            capsize=2,
            lw=1.4,
            color=COLORS[threshold],
            label=f"MAP <{int(threshold)} mm Hg",
        )
    ax.set_xticks([1, 2.5, 3, 5, 10])
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_ylim(0.2, 1.04)
    ax.set_xlabel("Sampling interval, min")
    ax.set_ylabel("Episode detection probability")
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.14, 1.03, "A", transform=ax.transAxes, weight="bold", fontsize=11)

    ax = axes[1]
    sub = duration[duration["threshold"].eq(65.0)].copy()
    order = ["1-<3 min", "3-<5 min", ">=5 min"]
    sub["order"] = sub["duration_bin"].map({value: idx for idx, value in enumerate(order)})
    sub = sub.sort_values("order")
    x = np.arange(len(sub))
    point = sub["complete_miss_probability"].to_numpy(float)
    ax.bar(x, point, width=0.58, color=["#C44E52", "#DD8452", "#55A868"])
    ax.errorbar(
        x,
        point,
        yerr=np.vstack(
            [
                point - sub["complete_miss_ci_low"].to_numpy(float),
                sub["complete_miss_ci_high"].to_numpy(float) - point,
            ]
        ),
        fmt="none",
        ecolor="#222222",
        capsize=3,
        lw=1.0,
    )
    for xpos, value in zip(x, point):
        ax.text(xpos, value + 0.035, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["1 to <3 min", "3 to <5 min", "≥5 min"])
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("Reference episode duration")
    ax.set_ylabel("Complete-miss probability at 5 min")
    ax.text(-0.14, 1.03, "B", transform=ax.transAxes, weight="bold", fontsize=11)
    fig.subplots_adjust(bottom=0.19, wspace=0.33)
    _save_figure(fig, out_root(cfg) / "figures", "Figure_2_episode_detection")


def _figure3(cfg: dict) -> None:
    overall = pd.read_csv(
        table_path(cfg, "episode_observability_5min_overall.csv")
    )
    duration = pd.read_csv(
        table_path(cfg, "episode_observability_5min_by_duration.csv")
    )
    overall65 = overall[overall["threshold"].eq(65.0)].copy()
    overall65["display_group"] = "<65\nAll"
    overall65["order"] = 0
    duration65 = duration[duration["threshold"].eq(65.0)].copy()
    duration65["display_group"] = duration65["duration_bin"].map(
        {
            "1-<3 min": "<65\n1 to <3",
            "3-<5 min": "<65\n3 to <5",
            ">=5 min": "<65\n≥5",
        }
    )
    duration65["order"] = duration65["duration_bin"].map(
        {"1-<3 min": 1, "3-<5 min": 2, ">=5 min": 3}
    )
    severe_short = duration[
        duration["threshold"].eq(55.0)
        & duration["duration_bin"].eq("1-<3 min")
    ].copy()
    severe_short["display_group"] = "<55\n1 to <3"
    severe_short["order"] = 4
    sub = pd.concat([overall65, duration65, severe_short], ignore_index=True)
    sub = sub.sort_values("order")
    sub["detected_with_at_least_2min"] = sub[
        "detection_with_2min_remaining_probability"
    ]
    sub["detected_with_1_to_less_than_2min"] = (
        sub["detection_with_1min_remaining_probability"]
        - sub["detection_with_2min_remaining_probability"]
    )
    sub["detected_with_less_than_1min"] = (
        sub["episode_detection_probability"]
        - sub["detection_with_1min_remaining_probability"]
    )
    sub["completely_missed"] = sub["complete_miss_probability"]
    write_table(cfg, "figure3_timing_consequences_source.csv", sub)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.65), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    x = np.arange(len(sub))
    ax = axes[0]
    stack_specs = [
        ("detected_with_at_least_2min", "First low display ≥2 min before recovery", "#0072B2"),
        ("detected_with_1_to_less_than_2min", "First low display 1 to <2 min before recovery", "#56B4E9"),
        ("detected_with_less_than_1min", "First low display <1 min before recovery", "#E69F00"),
        ("completely_missed", "Completely missed", "#CC79A7"),
    ]
    bottom = np.zeros(len(sub), dtype=float)
    for metric, label, color in stack_specs:
        values = sub[metric].to_numpy(float)
        ax.bar(x, values, bottom=bottom, width=0.68, label=label, color=color)
        bottom += values
    ax.axvline(3.5, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(x, sub["display_group"])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Phase-episode proportion")
    ax.set_xlabel("Threshold / episode duration")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=1)
    ax.text(-0.12, 1.03, "A", transform=ax.transAxes, weight="bold", fontsize=11)

    ax = axes[1]
    metric = "reference_auc_before_first_low_display_fraction"
    values = sub[metric].to_numpy(float)
    low = sub[f"{metric}_ci_low"].to_numpy(float)
    high = sub[f"{metric}_ci_high"].to_numpy(float)
    ax.errorbar(
        x,
        values,
        yerr=np.vstack([values - low, high - values]),
        fmt="o",
        color="#009E73",
        ecolor="#009E73",
        capsize=3,
        ms=5,
        lw=1.2,
    )
    ax.axhline(0.5, color="#777777", lw=0.8, ls="--")
    ax.axvline(3.5, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(x, sub["display_group"])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Reference AUC accrued\nbefore first low display")
    ax.set_xlabel("Threshold / episode duration")
    ax.text(-0.16, 1.03, "B", transform=ax.transAxes, weight="bold", fontsize=11)
    fig.subplots_adjust(bottom=0.36, wspace=0.36)
    _save_figure(fig, out_root(cfg) / "figures", "Figure_3_timing_consequences")


def _write_legends(cfg: dict) -> None:
    legends = pd.DataFrame(
        [
            {
                "figure": "Figure 1",
                "title": "Study flow and temporal observability under an emulated 5-min display",
                "legend": (
                    "Panel A shows sequential eligibility and waveform-coverage exclusions for the primary arterial cohort and the nested NIBP agreement branch. "
                    "Panel B illustrates how last-observation-carried-forward display can hide an arterial hypotensive interval and retain a low value after arterial recovery. "
                    "The schematic is an analytic illustration rather than a patient trace. ART, arterial; MAP, mean arterial pressure; NIBP, noninvasive blood pressure."
                ),
            },
            {
                "figure": "Figure 2",
                "title": "Phase-averaged observability of reference hypotensive episodes",
                "legend": (
                    "Panel A shows the probability that a reference episode lasting at least 60 s was represented by at least one displayed value below the same threshold across all possible 10-s sampling phases. "
                    "Panel B shows complete-miss probability at MAP <65 mm Hg with a 5-min interval, stratified by reference episode duration. Error bars are 95% subject-cluster bootstrap confidence intervals."
                ),
            },
            {
                "figure": "Figure 3",
                "title": "Decision-opportunity windows under an emulated 5-min display",
                "legend": (
                    "Panel A partitions each phase-episode combination into first low display at least 2 min before arterial recovery, 1 to less than 2 min before recovery, less than 1 min before recovery, or complete miss. "
                    "Panel B shows the proportion of reference hypotension area under the deficit curve accrued strictly before the first low display; a completely missed episode contributes its entire reference area. "
                    "The vertical dashed line separates the secondary MAP <55 mm Hg, 1- to less than 3-min severe-episode anchor from the MAP <65 mm Hg groups. "
                    "The horizontal dashed line in Panel B marks 50% of reference AUC accrued before the first low display. Error bars are 95% subject-cluster bootstrap confidence intervals."
                ),
            },
        ]
    )
    write_table(cfg, "figure_legends_v7.csv", legends)


def run(cfg: dict) -> None:
    _set_style()
    _write_publication_tables(cfg)
    _figure1(cfg)
    _figure2(cfg)
    _figure3(cfg)
    _write_legends(cfg)


if __name__ == "__main__":
    run(load_args())
