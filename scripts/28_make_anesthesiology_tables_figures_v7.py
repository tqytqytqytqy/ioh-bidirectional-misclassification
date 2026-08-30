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


def _write_publication_tables(cfg: dict) -> None:
    offset = pd.read_parquet(
        intermediate_path(cfg, "frequency_decomp_case_offset.parquet")
    )
    case_mean = _case_mean_from_offsets(offset)
    case_mean.to_parquet(
        intermediate_path(cfg, "frequency_decomp_case_mean_v7.parquet"), index=False
    )
    auc = _frequency_population_table(cfg, case_mean)
    write_table(cfg, "frequency_decomposition_bootstrap.csv", auc)
    write_table(cfg, "primary_auc_by_threshold_interval.csv", auc)

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
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    panel_cases = pd.read_parquet(
        intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"]
    )["case_id"].nunique()
    nibp_flow = pd.read_csv(table_path(cfg, "nibp_case_event_flow_v7.csv"))
    count = dict(zip(nibp_flow["step"], nibp_flow["n_cases"]))
    flow = pd.DataFrame(
        [
            {"stage": "VitalDB source cases", "n_cases": len(manifest)},
            {
                "stage": "Pre-QC arterial candidates",
                "n_cases": int(manifest["pre_qc_primary_candidate"].sum()),
            },
            {"stage": "Primary arterial waveform cohort", "n_cases": panel_cases},
            {
                "stage": "Primary cohort with NIBP track",
                "n_cases": int(count["primary_art_coverage_processing_pool"]),
            },
            {
                "stage": "Reconstructed NIBP display events",
                "n_cases": int(count["reconstructed_nibp_display_events"]),
            },
            {
                "stage": "Cases with paired NIBP-ART events",
                "n_cases": int(count["paired_nibp_events_primary_window"]),
            },
        ]
    )
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.0, 1.45]})
    ax = axes[0]
    ax.axis("off")
    y_positions = [0.9, 0.73, 0.56]
    labels = [
        ("VitalDB source", len(manifest)),
        ("Pre-QC arterial candidates", int(manifest["pre_qc_primary_candidate"].sum())),
        ("Primary waveform cohort", panel_cases),
    ]
    for y, (label, number) in zip(y_positions, labels):
        ax.text(
            0.5,
            y,
            f"{label}\n{number:,} cases",
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.28", "fc": "#F4F4F4", "ec": "#555555", "lw": 0.8},
        )
    for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate(
            "",
            xy=(0.5, y2 + 0.065),
            xytext=(0.5, y1 - 0.065),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#555555"},
        )
    ax.text(0.05, 0.43, "NIBP agreement branch", transform=ax.transAxes, weight="bold")
    branch = [
        ("NIBP processing pool", int(count["primary_art_coverage_processing_pool"])),
        ("Display events reconstructed", int(count["reconstructed_nibp_display_events"])),
        ("Cases with paired events", int(count["paired_nibp_events_primary_window"])),
    ]
    for index, (label, number) in enumerate(branch):
        y = 0.33 - index * 0.13
        ax.text(
            0.5,
            y,
            f"{label}: {number:,}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.22", "fc": "#EAF2F8", "ec": "#4C72B0", "lw": 0.8},
        )
    ax.text(-0.08, 1.02, "A", transform=ax.transAxes, weight="bold", fontsize=11)

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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)
    ax.text(-0.13, 1.02, "B", transform=ax.transAxes, weight="bold", fontsize=11)
    fig.subplots_adjust(bottom=0.26, wspace=0.34)
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
        1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.45, 1.0]}
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
    ax.tick_params(axis="x", labelsize=7.5)
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
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Reference AUC accrued\nbefore first low display")
    ax.set_xlabel("Threshold / episode duration")
    ax.text(-0.16, 1.03, "B", transform=ax.transAxes, weight="bold", fontsize=11)
    fig.subplots_adjust(bottom=0.37, wspace=0.36)
    _save_figure(fig, out_root(cfg) / "figures", "Figure_3_timing_consequences")


def _write_legends(cfg: dict) -> None:
    legends = pd.DataFrame(
        [
            {
                "figure": "Figure 1",
                "title": "Study flow and temporal observability under an emulated 5-min display",
                "legend": (
                    "Panel A shows derivation of the primary arterial waveform cohort and the nested NIBP agreement branch. "
                    "Panel B illustrates how last-observation-carried-forward display can hide an arterial hypotensive interval and retain a low value after arterial recovery. "
                    "The schematic is an analytic illustration rather than a patient trace. MAP, mean arterial pressure; NIBP, non-invasive blood pressure."
                ),
            },
            {
                "figure": "Figure 2",
                "title": "Phase-averaged observability of reference hypotensive episodes",
                "legend": (
                    "Panel A shows the probability that a reference episode lasting at least 60 s was represented by at least one displayed value below the same threshold across all possible 10-s sampling phases. "
                    "Panel B shows complete-miss probability at MAP <65 mm Hg with a 5-min interval, stratified by reference episode duration. Error bars are 95% case-cluster bootstrap confidence intervals."
                ),
            },
            {
                "figure": "Figure 3",
                "title": "Decision-opportunity windows under an emulated 5-min display",
                "legend": (
                    "Panel A partitions each phase-episode combination into first low display at least 2 min before arterial recovery, 1 to less than 2 min before recovery, less than 1 min before recovery, or complete miss. "
                    "Panel B shows the proportion of reference hypotension area under the deficit curve accrued strictly before the first low display; a completely missed episode contributes its entire reference area. "
                    "The MAP <55 mm Hg, 1- to less than 3-min group is a secondary severe-episode anchor. Error bars are 95% case-cluster bootstrap confidence intervals."
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
