from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pandas as pd

from _common import load_args
from ioh.pipeline import (
    build_nibp_display_events,
    decompose_nibp_timing_vs_cuff,
    ensure_dirs,
    intermediate_path,
    pair_nibp_with_artmap,
    table_path,
    write_qc,
    write_table,
)
from ioh.revision_v6 import (
    _write_display_validity_decomposition,
    _write_repeated_measure_agreement,
)
from ioh.revision_v7 import (
    case_phase_averaged_episode_rows,
    frequency_offset_to_case_episode_rows,
    summarize_episode_observability,
)


def _format_continuous(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "not available"
    median, lower, upper = values.quantile([0.5, 0.25, 0.75])
    return f"{median:.1f} [{lower:.1f}, {upper:.1f}]"


def _cohort_characteristics(
    manifest: pd.DataFrame, case_ids: pd.Index
) -> pd.DataFrame:
    cohort = manifest[manifest["case_id"].isin(set(case_ids))].copy()
    cohort["surgery_duration_min"] = (
        pd.to_numeric(cohort["opend"], errors="coerce")
        - pd.to_numeric(cohort["opstart"], errors="coerce")
    ) / 60.0
    rows: list[dict] = []

    def continuous(label: str, column: str) -> None:
        values = pd.to_numeric(cohort[column], errors="coerce")
        rows.append(
            {
                "characteristic": label,
                "level": "",
                "summary": _format_continuous(values),
                "nonmissing_n": int(values.notna().sum()),
                "missing_n": int(values.isna().sum()),
                "summary_definition": "median [interquartile range]",
            }
        )

    def binary(label: str, column: str) -> None:
        values = pd.to_numeric(cohort[column], errors="coerce")
        nonmissing = values.notna()
        count = int(values.eq(1).sum())
        denominator = int(nonmissing.sum())
        rows.append(
            {
                "characteristic": label,
                "level": "Yes",
                "summary": f"{count} ({100.0 * count / denominator:.1f}%)"
                if denominator
                else "not available",
                "nonmissing_n": denominator,
                "missing_n": int((~nonmissing).sum()),
                "summary_definition": "n (%) among nonmissing cases",
            }
        )

    continuous("Age, yr", "age")
    sex = cohort["sex"].astype("string").str.upper().str.strip()
    sex_nonmissing = sex.notna() & sex.ne("")
    male = int(sex.str.startswith("M", na=False).sum())
    sex_denominator = int(sex_nonmissing.sum())
    rows.append(
        {
            "characteristic": "Sex",
            "level": "Male",
            "summary": f"{male} ({100.0 * male / sex_denominator:.1f}%)",
            "nonmissing_n": sex_denominator,
            "missing_n": int((~sex_nonmissing).sum()),
            "summary_definition": "n (%) among nonmissing cases",
        }
    )
    continuous("Body mass index, kg m-2", "bmi")

    asa = pd.to_numeric(cohort["asa"], errors="coerce")
    asa_denominator = int(asa.notna().sum())
    for level in [1, 2, 3, 4]:
        count = int(asa.eq(level).sum())
        rows.append(
            {
                "characteristic": "ASA Physical Status",
                "level": str(level),
                "summary": f"{count} ({100.0 * count / asa_denominator:.1f}%)"
                if asa_denominator
                else "not available",
                "nonmissing_n": asa_denominator,
                "missing_n": int(asa.isna().sum()),
                "summary_definition": "n (%) among nonmissing cases",
            }
        )
    binary("Emergency surgery", "emop")
    binary("Preoperative hypertension", "preop_htn")
    binary("Preoperative diabetes mellitus", "preop_dm")
    continuous("Anaesthesia duration, min", "duration_min")
    continuous("Surgery duration, min", "surgery_duration_min")
    binary("NIBP track available", "has_nibp_map")
    return pd.DataFrame(rows)


def _run_episode_observability(cfg: dict) -> None:
    start = time.monotonic()
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    all_case_ids = pd.Index(panel["case_id"].drop_duplicates())
    print(f"ART cohort: {len(all_case_ids):,} cases; {len(panel):,} 10-s rows", flush=True)

    canonical_offset = intermediate_path(cfg, "frequency_decomp_case_offset.parquet")
    reference_offset = intermediate_path(
        cfg, "frequency_decomp_case_offset_v6_reference.parquet"
    )
    if not canonical_offset.exists():
        if not reference_offset.exists():
            raise FileNotFoundError(
                "phase-averaged frequency decomposition input is missing"
            )
        shutil.copy2(reference_offset, canonical_offset)
    offset = pd.read_parquet(canonical_offset)
    frequency_case = frequency_offset_to_case_episode_rows(offset)
    frequency_case.to_parquet(
        intermediate_path(cfg, "episode_observability_case_all_intervals.parquet"),
        index=False,
    )
    frequency_summary = summarize_episode_observability(
        frequency_case,
        all_case_ids=all_case_ids,
        group_columns=["threshold", "interval_min"],
        bootstrap_reps=int(cfg["analysis"]["bootstrap_reps"]),
        seed=int(cfg["project"]["seed"]),
    ).sort_values(["threshold", "interval_min"])
    write_table(
        cfg,
        "episode_observability_by_threshold_interval.csv",
        frequency_summary,
    )
    print(
        f"All-interval episode summary complete in {time.monotonic() - start:.1f}s",
        flush=True,
    )

    detailed_frames: list[pd.DataFrame] = []
    for index, (case_id, sub) in enumerate(panel.groupby("case_id", sort=False), start=1):
        sub = sub.sort_values("time_sec")
        rows = case_phase_averaged_episode_rows(
            case_id=case_id,
            times=sub["time_sec"].to_numpy(float),
            reference=sub["art_map"].to_numpy(float),
            thresholds=cfg["analysis"]["thresholds"],
            intervals_min=[5.0],
            min_duration_sec=60.0,
            step_sec=int(cfg["analysis"]["resample_sec"]),
        )
        if not rows.empty:
            detailed_frames.append(rows)
        if index % 100 == 0 or index == len(all_case_ids):
            print(
                f"Detailed 5-min observability: {index:,}/{len(all_case_ids):,} cases",
                flush=True,
            )
    detailed = pd.concat(detailed_frames, ignore_index=True)
    detailed.to_parquet(
        intermediate_path(cfg, "episode_observability_case_5min.parquet"),
        index=False,
    )

    summary_specs = {
        "episode_observability_5min_overall.csv": ["threshold", "interval_min"],
        "episode_observability_5min_by_duration.csv": [
            "threshold",
            "interval_min",
            "duration_bin",
        ],
        "episode_observability_5min_by_nadir.csv": [
            "threshold",
            "interval_min",
            "nadir_bin",
        ],
        "episode_observability_5min_duration_severity.csv": [
            "threshold",
            "interval_min",
            "duration_bin",
            "nadir_bin",
        ],
    }
    summaries: dict[str, pd.DataFrame] = {}
    for filename, group_columns in summary_specs.items():
        summary = summarize_episode_observability(
            detailed,
            all_case_ids=all_case_ids,
            group_columns=group_columns,
            bootstrap_reps=int(cfg["analysis"]["bootstrap_reps"]),
            seed=int(cfg["project"]["seed"]),
        )
        summaries[filename] = summary
        write_table(cfg, filename, summary)

    actionability_qc_frames: list[pd.DataFrame] = []
    for filename, summary in summaries.items():
        qc = summary.copy()
        qc.insert(0, "source_table", filename)
        one_min = pd.to_numeric(
            qc["detection_with_1min_remaining_probability"], errors="coerce"
        )
        two_min = pd.to_numeric(
            qc["detection_with_2min_remaining_probability"], errors="coerce"
        )
        detected = pd.to_numeric(
            qc["episode_detection_probability"], errors="coerce"
        )
        pre_detection = pd.to_numeric(
            qc["reference_auc_before_first_low_display_fraction"], errors="coerce"
        )
        qc["probability_hierarchy_passed"] = (
            two_min.ge(0.0)
            & two_min.le(one_min + 1e-12)
            & one_min.le(detected + 1e-12)
            & detected.le(1.0 + 1e-12)
        )
        qc["pre_detection_auc_range_passed"] = pre_detection.between(
            -1e-12, 1.0 + 1e-12
        )
        ci_checks = []
        for metric in (
            "detection_with_1min_remaining_probability",
            "detection_with_2min_remaining_probability",
            "reference_auc_before_first_low_display_fraction",
        ):
            point = pd.to_numeric(qc[metric], errors="coerce")
            lower = pd.to_numeric(qc[f"{metric}_ci_low"], errors="coerce")
            upper = pd.to_numeric(qc[f"{metric}_ci_high"], errors="coerce")
            ci_checks.append(
                lower.le(point + 1e-12)
                & point.le(upper + 1e-12)
                & lower.ge(-1e-12)
                & upper.le(1.0 + 1e-12)
            )
        qc["confidence_interval_passed"] = np.logical_and.reduce(ci_checks)
        qc["passed"] = qc[
            [
                "probability_hierarchy_passed",
                "pre_detection_auc_range_passed",
                "confidence_interval_passed",
            ]
        ].all(axis=1)
        actionability_qc_frames.append(qc)

    actionability_qc = pd.concat(actionability_qc_frames, ignore_index=True)
    qc_columns = [
        "source_table",
        "threshold",
        "interval_min",
        "duration_bin",
        "nadir_bin",
        "episode_detection_probability",
        "detection_with_1min_remaining_probability",
        "detection_with_2min_remaining_probability",
        "reference_auc_before_first_low_display_fraction",
        "probability_hierarchy_passed",
        "pre_detection_auc_range_passed",
        "confidence_interval_passed",
        "passed",
    ]
    for column in qc_columns:
        if column not in actionability_qc.columns:
            actionability_qc[column] = np.nan
    actionability_qc = actionability_qc[qc_columns]
    write_table(cfg, "episode_actionability_5min_qc.csv", actionability_qc)
    if not bool(actionability_qc["passed"].all()):
        raise AssertionError("episode actionability metrics failed closure checks")
    write_qc(
        cfg,
        "episode_actionability_5min_qc.md",
        "# Episode actionability quality control\n\n"
        "All reported 5-min actionability estimates passed probability hierarchy, "
        "range, and case-cluster bootstrap confidence-interval closure checks. "
        "Specifically, P(>=2 min remaining) <= P(>=1 min remaining) <= "
        "P(detected), and all probabilities and pre-detection AUC fractions were "
        "bounded by 0 and 1.\n\n"
        f"Rows checked: {len(actionability_qc)}.\n",
    )

    detailed_overall = summaries["episode_observability_5min_overall.csv"]
    frequency_5min = frequency_summary[frequency_summary["interval_min"].eq(5.0)]
    check_columns = [
        "reference_episodes",
        "expected_detected_episodes",
        "expected_missed_episodes",
        "episode_detection_probability",
        "complete_miss_probability",
        "mean_detection_delay_min",
    ]
    check = frequency_5min[["threshold", *check_columns]].merge(
        detailed_overall[["threshold", *check_columns]],
        on="threshold",
        how="outer",
        suffixes=("_frequency", "_detailed"),
        validate="one_to_one",
    )
    for column in check_columns:
        check[f"abs_diff_{column}"] = (
            check[f"{column}_frequency"] - check[f"{column}_detailed"]
        ).abs()
    diff_columns = [column for column in check if column.startswith("abs_diff_")]
    check["passed"] = check[diff_columns].max(axis=1).le(1e-12)
    write_table(cfg, "episode_observability_5min_consistency.csv", check)
    if not bool(check["passed"].all()):
        raise AssertionError("detailed 5-min episode metrics do not match frequency analysis")
    write_qc(
        cfg,
        "episode_observability_consistency.md",
        "# Episode observability consistency\n\n"
        "The independently assembled duration/severity analysis reproduced the existing "
        "phase-averaged 5-min episode counts, detection probabilities, complete-miss "
        "probabilities, and detection delays to an absolute tolerance of 1e-12.\n\n"
        "```csv\n"
        + check.to_csv(index=False)
        + "```\n",
    )
    print(
        f"Detailed episode analysis complete in {time.monotonic() - start:.1f}s",
        flush=True,
    )


def _write_nibp_flow(cfg: dict) -> None:
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    panel_cases = set(
        pd.read_parquet(
            intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"]
        )["case_id"].drop_duplicates()
    )
    pre_qc = set(
        manifest.loc[
            manifest["pre_qc_primary_candidate"] & manifest["has_nibp_map"],
            "case_id",
        ]
    )
    processing_pool = pre_qc & panel_cases
    raw = pd.read_parquet(intermediate_path(cfg, "nibp_raw_sample.parquet"))
    events = pd.read_csv(table_path(cfg, "nibp_display_events.csv"))
    pairs = pd.read_csv(table_path(cfg, "nibp_art_pairs.csv"))
    raw_cases = set(raw["case_id"].drop_duplicates())
    event_cases = set(events["case_id"].drop_duplicates())
    paired_cases = set(pairs.loc[pairs["paired"], "case_id"].drop_duplicates())
    flow = pd.DataFrame(
        [
            {
                "step": "pre_qc_cases_with_art_and_nibp_tracks",
                "n_cases": len(pre_qc),
                "n_events": np.nan,
                "definition": "pre-QC anaesthetic-interval candidates with ART and NIBP tracks",
            },
            {
                "step": "primary_art_coverage_processing_pool",
                "n_cases": len(processing_pool),
                "n_events": np.nan,
                "definition": "pre-QC ART/NIBP cases retained in the primary arterial-series cohort",
            },
            {
                "step": "readable_nibp_records_in_window",
                "n_cases": len(raw_cases),
                "n_events": len(raw),
                "definition": "cases and raw monitor rows readable within the anaesthetic interval",
            },
            {
                "step": "reconstructed_nibp_display_events",
                "n_cases": len(event_cases),
                "n_events": len(events),
                "definition": "cases and display-level events after range and de-duplication rules",
            },
            {
                "step": "paired_nibp_events_primary_window",
                "n_cases": len(paired_cases),
                "n_events": int(pairs["paired"].sum()),
                "definition": "events paired to median ART MAP in the -30 to +30 s window",
            },
        ]
    )
    write_table(cfg, "nibp_case_event_flow_v7.csv", flow)
    if len(processing_pool) != int(cfg["nibp"].get("expected_processing_cases", len(processing_pool))):
        print(
            f"NIBP processing pool observed: {len(processing_pool):,} cases",
            flush=True,
        )


def _run_full_nibp(cfg: dict) -> None:
    start = time.monotonic()
    print("Starting all-eligible NIBP reconstruction", flush=True)
    build_nibp_display_events(cfg)
    print(
        f"NIBP display reconstruction complete in {time.monotonic() - start:.1f}s",
        flush=True,
    )
    pair_nibp_with_artmap(cfg)
    print(f"NIBP pairing complete in {time.monotonic() - start:.1f}s", flush=True)
    decompose_nibp_timing_vs_cuff(cfg)
    print(
        f"NIBP timing/cuff decomposition complete in {time.monotonic() - start:.1f}s",
        flush=True,
    )
    _write_repeated_measure_agreement(cfg)
    _write_display_validity_decomposition(cfg)
    _write_nibp_flow(cfg)
    print(f"Full NIBP analyses complete in {time.monotonic() - start:.1f}s", flush=True)


def run(cfg: dict) -> None:
    ensure_dirs(cfg)
    panel_case_ids = pd.Index(
        pd.read_parquet(
            intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"]
        )["case_id"].drop_duplicates()
    )
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    write_table(
        cfg,
        "table1_primary_cohort_characteristics.csv",
        _cohort_characteristics(manifest, panel_case_ids),
    )
    _run_episode_observability(cfg)
    if os.environ.get("IOH_SKIP_FULL_NIBP", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        print("Skipping full NIBP reconstruction by IOH_SKIP_FULL_NIBP", flush=True)
    else:
        _run_full_nibp(cfg)


if __name__ == "__main__":
    run(load_args())
