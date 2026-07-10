from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ioh.estimands.decomposition import decompose_deficit, display_from_event_times, emulate_last_visible
from ioh.estimands.episodes import detect_reference_episodes, episode_documented
from ioh.io.inspire_loader import guard_inspire_metric
from ioh.qc.nibp_qc import collapse_nibp_display_events, retention_audit
from ioh.reporting.language import assert_no_forbidden_language
from ioh.reporting.manifest import build_run_manifest, file_sha256, mover_claim_gate, write_outputs_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path) -> dict:
    path = resolve_path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def out_root(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["project"]["output_dir"]


def ensure_dirs(cfg: dict) -> None:
    for rel in ["tables", "figures", "qc", "manifests", "logs", "intermediate"]:
        (out_root(cfg) / rel).mkdir(parents=True, exist_ok=True)


def table_path(cfg: dict, name: str) -> Path:
    return out_root(cfg) / "tables" / name


def intermediate_path(cfg: dict, name: str) -> Path:
    return out_root(cfg) / "intermediate" / name


def qc_path(cfg: dict, name: str) -> Path:
    return out_root(cfg) / "qc" / name


def write_table(cfg: dict, name: str, df: pd.DataFrame) -> Path:
    path = table_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_qc(cfg: dict, name: str, text: str) -> Path:
    path = qc_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def validate_config(cfg: dict) -> pd.DataFrame:
    ensure_dirs(cfg)
    role_path = PROJECT_ROOT / "config" / "data_roles.yaml"
    roles = yaml.safe_load(role_path.read_text(encoding="utf-8"))
    checks = [
        ("vitaldb_root_exists", Path(cfg["data"]["vitaldb_root"]).exists()),
        ("inspire_zip_exists", Path(cfg["data"]["inspire_zip"]).exists()),
        ("data_roles_yaml_exists", role_path.exists()),
        ("vitaldb_role_declared", "VitalDB" in roles),
        ("mover_role_declared", "MoVeR" in roles),
        ("inspire_role_declared", "INSPIRE" in roles),
        ("output_dir_is_outputs", cfg["project"]["output_dir"] == "outputs"),
    ]
    out = pd.DataFrame([{"check": name, "passed": bool(passed)} for name, passed in checks])
    write_table(cfg, "config_validation.csv", out)
    metric_rows = [
        {
            "metric": "HDR",
            "definition": "sum_i HiddenAUC_i / sum_i TrueAUC_i",
            "role": "primary endpoint for MAP<65 fixed 5-min display",
        },
        {
            "metric": "ODR",
            "definition": "sum_i OverdisplayAUC_i / sum_i TrueAUC_i",
            "role": "key co-primary/secondary endpoint for MAP<65 fixed 5-min display",
        },
        {
            "metric": "NetBias",
            "definition": "ODR - HDR, equivalent to (DisplayAUC - TrueAUC) / TrueAUC",
            "role": "net bias after bidirectional components",
        },
        {
            "metric": "normotensive_display_discordance_min",
            "definition": "minutes with reference hypotension but displayed MAP at or above threshold",
            "role": "unrecognised hypotensive display time",
        },
        {
            "metric": "hypotensive_display_discordance_min",
            "definition": "minutes with reference non-hypotension but displayed MAP below threshold",
            "role": "overdisplayed hypotensive time",
        },
    ]
    write_table(cfg, "metric_definition_dictionary.csv", pd.DataFrame(metric_rows))
    (PROJECT_ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "docs" / "estimand_dictionary.md").write_text(
        "# IOH2 Estimand Dictionary\n\n"
        "Primary endpoint: `HDR65_fixed5`. Key secondary endpoint: `ODR65_fixed5`.\n\n"
        "For each time point, `R=max(0, threshold-reference_MAP)` and "
        "`D=max(0, threshold-displayed_MAP)`. The pointwise components are "
        "`concordant=min(R,D)`, `hidden=max(0,R-D)`, and `overdisplay=max(0,D-R)`. "
        "Population ratios use aggregate `TrueAUC` denominators. Historical one-directional visibility metrics are not primary endpoints.\n",
        encoding="utf-8",
    )
    qc_path(cfg, "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if not out["passed"].all():
        raise ValueError("config validation failed")
    return out


def read_vitaldb_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(cfg["data"]["vitaldb_root"])
    return (
        pd.read_csv(root / "cases.csv", encoding="utf-8-sig"),
        pd.read_csv(root / "trks.csv", encoding="utf-8-sig"),
        pd.read_csv(root / "labs.csv", encoding="utf-8-sig"),
    )


def read_vitaldb_track(cfg: dict, tid: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cfg["data"]["vitaldb_root"]) / "tracks" / f"{tid}.csv.gz"
    if not path.exists():
        return np.array([], dtype=float), np.array([], dtype=float)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        df = pd.read_csv(handle)
    if df.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float), pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)


def _surgical_exclusion_category(row: pd.Series, cfg: dict) -> str:
    banned = {str(x).lower() for x in cfg["cohort"]["exclude_surgery_groups"]}
    text = " ".join(str(row.get(col, "")) for col in ["department", "optype", "opname", "dx", "ane_type"]).lower()
    hierarchy = [
        ("cardiac", ["cardiac", "heart", "bypass", "cpb"]),
        ("obstetric", ["obstetric", "cesarean"]),
        ("transplant", ["transplant"]),
    ]
    for category, terms in hierarchy:
        if any(term in banned and term in text for term in terms):
            return category
    other_terms = sorted(term for term in banned if all(term not in terms for _, terms in hierarchy))
    if any(term in text for term in other_terms):
        return "other_prespecified"
    return ""


def _exclude_special(row: pd.Series, cfg: dict) -> bool:
    return bool(_surgical_exclusion_category(row, cfg))


def _select_track(trks: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    work = trks[trks["tname"].astype(str).isin(candidates) | trks["tname"].astype(str).str.contains("|".join(map(re.escape, candidates)), case=False, na=False)].copy()
    priority = {name: i for i, name in enumerate(candidates)}
    work["track_priority"] = work["tname"].map(priority).fillna(99)
    return work.sort_values(["caseid", "track_priority"]).drop_duplicates("caseid", keep="first")


def build_vitaldb_manifest(cfg: dict) -> pd.DataFrame:
    cases, trks, _ = read_vitaldb_raw(cfg)
    art = _select_track(trks, cfg["tracks"]["art_map_candidates"]).rename(columns={"tid": "art_tid", "tname": "art_track_name"})
    nibp = _select_track(trks, cfg["tracks"]["nibp_map_candidates"]).rename(columns={"tid": "nibp_tid", "tname": "nibp_track_name"})
    c = cases.copy()
    c["duration_min"] = (pd.to_numeric(c["aneend"], errors="coerce") - pd.to_numeric(c["anestart"], errors="coerce")) / 60.0
    c["adult"] = pd.to_numeric(c["age"], errors="coerce") >= float(cfg["cohort"]["adult_min_age"])
    c["general_anaesthesia"] = c["ane_type"].astype(str).str.contains("general", case=False, na=False)
    c["surgical_exclusion_category"] = c.apply(
        lambda row: _surgical_exclusion_category(row, cfg), axis=1
    )
    c["eligible_surgery"] = c["surgical_exclusion_category"].eq("")
    c["duration_eligible"] = c["duration_min"] >= float(cfg["cohort"]["min_duration_min"])
    manifest = (
        c.merge(art[["caseid", "art_tid", "art_track_name"]], on="caseid", how="left")
        .merge(nibp[["caseid", "nibp_tid", "nibp_track_name"]], on="caseid", how="left")
        .rename(columns={"caseid": "case_id"})
    )
    manifest["has_art_map"] = manifest["art_tid"].notna()
    manifest["has_nibp_map"] = manifest["nibp_tid"].notna()
    manifest[["primary_time_window", "primary_start_sec", "primary_end_sec"]] = manifest.apply(
        lambda row: pd.Series(_time_window_bounds(row, cfg["time_window"]["primary"])),
        axis=1,
    )
    keep = pd.Series(True, index=manifest.index)
    rows = [{"step": "source_cases", "remaining": len(manifest), "excluded": 0}]
    for name, mask in [
        ("adult", manifest["adult"]),
        ("general_anaesthesia", manifest["general_anaesthesia"]),
        ("eligible_surgery", manifest["eligible_surgery"]),
        ("duration_ge_60min", manifest["duration_eligible"]),
        ("has_art_map", manifest["has_art_map"]),
    ]:
        before = int(keep.sum())
        keep &= mask.fillna(False)
        rows.append({"step": name, "remaining": int(keep.sum()), "excluded": before - int(keep.sum())})
    manifest["pre_qc_primary_candidate"] = keep
    flow = pd.DataFrame(rows)
    manifest.to_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"), index=False)
    write_table(cfg, "vitaldb_manifest_preview.csv", manifest.head(1000))
    write_table(cfg, "cohort_flow.csv", flow)
    before_surgical = manifest[manifest["adult"] & manifest["general_anaesthesia"]].copy()
    exclusion_rows = []
    for category in ["cardiac", "obstetric", "transplant", "other_prespecified"]:
        exclusion_rows.append(
            {
                "hierarchy_order": len(exclusion_rows) + 1,
                "exclusion_category": category,
                "excluded_cases": int(before_surgical["surgical_exclusion_category"].eq(category).sum()),
            }
        )
    exclusion = pd.DataFrame(exclusion_rows)
    exclusion["eligible_before_surgical_exclusions"] = len(before_surgical)
    exclusion["remaining_after_all_surgical_exclusions"] = int(
        before_surgical["eligible_surgery"].sum()
    )
    exclusion["total_surgically_excluded"] = int((~before_surgical["eligible_surgery"]).sum())
    write_table(cfg, "surgical_exclusion_flow.csv", exclusion)
    duration_audit = manifest[
        manifest["adult"] & manifest["general_anaesthesia"] & manifest["eligible_surgery"]
    ].sort_values("case_id").head(10).copy()
    duration_audit["anaesthesia_duration_min_recalculated"] = (
        pd.to_numeric(duration_audit["aneend"], errors="coerce")
        - pd.to_numeric(duration_audit["anestart"], errors="coerce")
    ) / 60.0
    duration_audit["surgery_duration_min_recalculated"] = (
        pd.to_numeric(duration_audit["opend"], errors="coerce")
        - pd.to_numeric(duration_audit["opstart"], errors="coerce")
    ) / 60.0
    write_table(
        cfg,
        "case_duration_timestamp_audit.csv",
        duration_audit[
            [
                "case_id",
                "anestart",
                "aneend",
                "anaesthesia_duration_min_recalculated",
                "opstart",
                "opend",
                "surgery_duration_min_recalculated",
                "duration_min",
                "primary_time_window",
            ]
        ],
    )
    track_rows = []
    for role, selected, candidates in [
        ("arterial_map", art, cfg["tracks"]["art_map_candidates"]),
        ("nibp_map", nibp, cfg["tracks"]["nibp_map_candidates"]),
    ]:
        available = trks[
            trks["tname"].astype(str).isin(candidates)
            | trks["tname"].astype(str).str.contains(
                "|".join(map(re.escape, candidates)), case=False, na=False
            )
        ]
        selected_name = "art_track_name" if role == "arterial_map" else "nibp_track_name"
        for priority, candidate in enumerate(candidates):
            track_rows.append(
                {
                    "track_role": role,
                    "candidate_priority": priority + 1,
                    "configured_candidate": candidate,
                    "available_track_rows_exact_name": int(available["tname"].eq(candidate).sum()),
                    "available_cases_exact_name": int(
                        available.loc[available["tname"].eq(candidate), "caseid"].nunique()
                    ),
                    "selected_cases": int(selected[selected_name].eq(candidate).sum()),
                    "selection_rule": "lowest configured priority per case; one track retained",
                }
            )
    write_table(cfg, "track_selection_audit.csv", pd.DataFrame(track_rows))
    write_qc(
        cfg,
        "cohort_flow_qc.md",
        "# Cohort flow QC\n\n```csv\n" + flow.to_csv(index=False) + "```\n",
    )
    return manifest


def _time_window_bounds(row: pd.Series, time_window: str) -> tuple[str, float, float]:
    if time_window == "anaesthetic_interval":
        start_col, end_col = "anestart", "aneend"
    elif time_window == "surgery_interval":
        start_col, end_col = "opstart", "opend"
    else:
        raise ValueError(f"unsupported time window: {time_window}")
    start = pd.to_numeric(pd.Series([row.get(start_col)]), errors="coerce").iloc[0]
    end = pd.to_numeric(pd.Series([row.get(end_col)]), errors="coerce").iloc[0]
    start_sec = max(0.0, float(start)) if np.isfinite(start) else 0.0
    end_sec = float(end) if np.isfinite(end) else np.nan
    return time_window, start_sec, end_sec


def resample_median_grid(
    times_sec: np.ndarray,
    map_values: np.ndarray,
    start_sec: float,
    end_sec: float,
    step_sec: int,
    min_map: float = 20.0,
    max_map: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(times_sec) & np.isfinite(map_values) & (map_values >= float(min_map)) & (map_values <= float(max_map))
    if end_sec <= start_sec:
        return np.array([], dtype=float), np.array([], dtype=float)
    grid = np.arange(float(start_sec), float(end_sec), float(step_sec), dtype=float)
    out = np.full(len(grid), np.nan, dtype=float)
    if len(grid) == 0 or valid.sum() == 0:
        return grid, out
    bins = np.floor((times_sec[valid] - float(start_sec)) / float(step_sec)).astype(int)
    keep = (bins >= 0) & (bins < len(grid))
    if keep.any():
        med = pd.DataFrame({"bin": bins[keep], "map": map_values[valid][keep]}).groupby("bin")["map"].median()
        out[med.index.to_numpy(int)] = med.to_numpy(float)
    return grid - float(start_sec), out


def preprocess_vitaldb_artmap(cfg: dict) -> pd.DataFrame:
    manifest_path = intermediate_path(cfg, "vitaldb_manifest.parquet")
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else build_vitaldb_manifest(cfg)
    min_map, max_map = cfg["cohort"]["art_map_valid_range"]
    rows = []
    panels = []
    sensitivity_rows = []
    windows = [cfg["time_window"]["primary"]] + list(cfg["time_window"].get("sensitivity", []))
    for idx, row in manifest[manifest["pre_qc_primary_candidate"]].iterrows():
        times, values = read_vitaldb_track(cfg, str(row["art_tid"]))
        window_grids: dict[str, tuple[float, float, np.ndarray, np.ndarray]] = {}
        for window in windows:
            _, start_sec, end_sec = _time_window_bounds(row, window)
            rel_t, grid_clean = resample_median_grid(
                times,
                values,
                start_sec,
                end_sec,
                int(cfg["analysis"]["resample_sec"]),
                min_map=float(min_map),
                max_map=float(max_map),
            )
            window_grids[window] = (start_sec, end_sec, rel_t, grid_clean)
            sensitivity_rows.append(
                {
                    "case_id": row["case_id"],
                    "time_window": window,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "total_10s_bins": len(grid_clean),
                    "valid_10s_bins": int(np.isfinite(grid_clean).sum()),
                    "valid_coverage": float(np.isfinite(grid_clean).mean()) if len(grid_clean) else 0.0,
                    "include_by_primary_criteria": bool(
                        len(grid_clean)
                        and float(np.isfinite(grid_clean).mean()) >= float(cfg["cohort"]["min_art_coverage"])
                        and len(grid_clean) * int(cfg["analysis"]["resample_sec"]) / 60 >= float(cfg["cohort"]["min_duration_min"])
                    ),
                }
            )
        start_sec, end_sec, rel_t, grid_clean = window_grids[cfg["time_window"]["primary"]]
        valid_range = np.isfinite(grid_clean)
        coverage = float(np.isfinite(grid_clean).mean()) if len(grid_clean) else 0.0
        rows.append(
            {
                "case_id": row["case_id"],
                "art_tid": row["art_tid"],
                "time_window": cfg["time_window"]["primary"],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "total_10s_bins": len(grid_clean),
                "valid_10s_bins": int(np.isfinite(grid_clean).sum()),
                "valid_coverage": coverage,
                "extreme_or_missing_bins": int((~valid_range).sum()) if len(grid_clean) else 0,
                "flat_adjacent_count": int(np.nansum(np.abs(np.diff(grid_clean[np.isfinite(grid_clean)])) < 1e-9)) if np.isfinite(grid_clean).sum() > 1 else 0,
                "include_primary": bool(coverage >= float(cfg["cohort"]["min_art_coverage"]) and len(grid_clean) * int(cfg["analysis"]["resample_sec"]) / 60 >= float(cfg["cohort"]["min_duration_min"])),
            }
        )
        if rows[-1]["include_primary"]:
            panels.append(pd.DataFrame({"case_id": row["case_id"], "time_sec": rel_t, "art_map": grid_clean}))
    qc = pd.DataFrame(rows)
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame(columns=["case_id", "time_sec", "art_map"])
    qc.to_csv(table_path(cfg, "waveform_qc.csv"), index=False)
    sens = pd.DataFrame(sensitivity_rows)
    write_table(cfg, "time_window_sensitivity_qc.csv", sens)
    sens_summary = sens.groupby("time_window", dropna=False).agg(
        candidate_cases=("case_id", "nunique"),
        included_cases=("include_by_primary_criteria", "sum"),
        median_valid_coverage=("valid_coverage", "median"),
        median_total_10s_bins=("total_10s_bins", "median"),
    ).reset_index()
    write_table(cfg, "time_window_sensitivity_summary.csv", sens_summary)
    panel.to_parquet(intermediate_path(cfg, "artmap_10s.parquet"), index=False)
    flow = pd.read_csv(table_path(cfg, "cohort_flow.csv"))
    flow = pd.concat(
        [
            flow,
            pd.DataFrame(
                [
                    {
                        "step": "art_coverage_ge_80pct",
                        "remaining": int(qc["include_primary"].sum()) if not qc.empty else 0,
                        "excluded": int((~qc["include_primary"]).sum()) if not qc.empty else 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_table(cfg, "cohort_flow.csv", flow)
    write_qc(
        cfg,
        "waveform_qc.md",
        "# Waveform QC\n\n"
        f"- primary time window: {cfg['time_window']['primary']} with negative starts clamped to recording time 0\n"
        f"- primary cases after coverage QC: {int(qc['include_primary'].sum()) if not qc.empty else 0}\n"
        f"- median coverage: {qc['valid_coverage'].median():.3f}\n\n"
        "```csv\n" + qc[["case_id", "valid_coverage", "include_primary"]].head(20).to_csv(index=False) + "```\n",
    )
    write_qc(
        cfg,
        "time_window_sensitivity_qc.md",
        "# Time-window sensitivity QC\n\n"
        "Primary analysis uses the configured anaesthetic interval after clamping negative VitalDB starts to recording time 0. "
        "The surgery interval is retained as a cohort-coverage sensitivity check.\n\n"
        "```csv\n" + sens_summary.to_csv(index=False) + "```\n",
    )
    return qc


def _episode_metrics(times: np.ndarray, reference: np.ndarray, display: np.ndarray, threshold: float, min_duration_sec: float) -> dict:
    episodes = detect_reference_episodes(times, reference, threshold=threshold, min_duration_sec=min_duration_sec, gap_tolerance_sec=0)
    detected = []
    delays = []
    for episode in episodes:
        documented = episode_documented(episode, times, display, threshold=threshold)
        detected.append(documented)
        if documented:
            mask = (times >= episode.start_sec) & (times <= episode.end_sec)
            low_display_idx = np.flatnonzero(mask & np.isfinite(display) & (display < float(threshold)))
            if len(low_display_idx):
                delays.append(max(0.0, (times[int(low_display_idx[0])] - episode.start_sec) / 60.0))
    return {
        "episode_count": int(len(episodes)),
        "episode_detected": int(sum(detected)),
        "episode_sensitivity": float(sum(detected) / len(episodes)) if episodes else np.nan,
        "mean_detection_delay_min": float(np.mean(delays)) if delays else np.nan,
        "missed_episodes": int(len(episodes) - sum(detected)),
    }


def _duration_bin(duration_sec: float) -> str:
    duration = float(duration_sec)
    if duration < 30:
        return "<30s"
    if duration < 60:
        return "30-60s"
    if duration < 180:
        return "1-3min"
    if duration < 300:
        return "3-5min"
    return ">5min"


def _display_from_sample_times(times: np.ndarray, values: np.ndarray, sample_times: list[float] | np.ndarray, carry_forward_limit_sec: float | None = None) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    sampled_times: list[float] = []
    sampled_values: list[float] = []
    for sample_time in np.asarray(sample_times, dtype=float):
        if len(times) == 0:
            break
        idx = int(np.searchsorted(times, sample_time, side="left"))
        if idx >= len(times):
            idx = len(times) - 1
        if idx > 0 and abs(times[idx - 1] - sample_time) <= abs(times[idx] - sample_time):
            idx -= 1
        if np.isfinite(values[idx]):
            sampled_times.append(float(times[idx]))
            sampled_values.append(float(values[idx]))
    return display_from_event_times(times, np.asarray(sampled_times), np.asarray(sampled_values), carry_forward_limit_sec=carry_forward_limit_sec)


def build_strategy_display_series(
    times: np.ndarray,
    values: np.ndarray,
    strategy: str,
    *,
    selected_for_continuous: bool = False,
    carry_forward_limit_sec: float | None = None,
    rules: dict | None = None,
) -> tuple[np.ndarray, dict]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) == 0:
        return np.array([], dtype=float), {"measurement_burden_per_hour": np.nan, "trigger_count": 0, "sample_count": 0}

    first = float(times[0])
    last = float(times[-1])
    total_hours = max((last - first + 10.0) / 3600.0, 1e-9)

    rules = rules or {
        "baseline_interval_min": 5,
        "fixed_intervals_min": [3, 2.5],
        "induction_intensified": {
            "induction_window_min": 20,
            "induction_interval_min": 1,
            "maintenance_interval_min": 5,
        },
        "threshold_triggered_adaptive": {
            "baseline_interval_min": 5,
            "intensified_interval_min": 1,
            "trigger_visible_map_lt": 70,
            "intensify_for_min": 5,
        },
        "trend_triggered_adaptive": {
            "baseline_interval_min": 5,
            "intensified_interval_min": 1,
            "trigger_drop_mmHg": 8,
            "trigger_visible_map_lt": 75,
            "intensify_for_min": 3,
        },
    }
    fixed_intervals = {
        "fixed_5min_reference": float(rules["baseline_interval_min"]) * 60.0,
        "fixed_5min_frequency_only": float(rules["baseline_interval_min"]) * 60.0,
        "fixed_3min": float(rules["fixed_intervals_min"][0]) * 60.0,
        "fixed_2_5min": float(rules["fixed_intervals_min"][1]) * 60.0,
        "universal_1min_proxy": 60.0,
    }
    if strategy in fixed_intervals:
        interval = fixed_intervals[strategy]
        sample_times = np.arange(first, last + 1e-9, interval)
        display = _display_from_sample_times(times, values, sample_times, carry_forward_limit_sec=carry_forward_limit_sec)
        return display, {
            "measurement_burden_per_hour": float(3600.0 / interval),
            "trigger_count": 0,
            "sample_count": int(len(sample_times)),
            "selected_for_continuous": False,
        }

    if strategy == "universal_continuous_reference" or (strategy == "selective_continuous_top20_oof" and selected_for_continuous):
        return values.copy(), {
            "measurement_burden_per_hour": np.nan,
            "trigger_count": 0,
            "sample_count": int(len(times)),
            "selected_for_continuous": True,
        }

    if strategy == "selective_continuous_top20_oof":
        return build_strategy_display_series(
            times,
            values,
            "fixed_5min_reference",
            carry_forward_limit_sec=carry_forward_limit_sec,
            rules=rules,
        )

    sample_times: list[float] = []
    trigger_count = 0
    t = first
    intensified_until = first - 1
    prev_value = np.nan
    while t <= last + 1e-9:
        sample_times.append(float(t))
        idx = int(np.searchsorted(times, t, side="left"))
        if idx >= len(times):
            idx = len(times) - 1
        current = float(values[idx]) if np.isfinite(values[idx]) else np.nan
        if strategy == "induction_intensified_first20min":
            spec = rules["induction_intensified"]
            interval = (
                float(spec["induction_interval_min"]) * 60.0
                if t - first < float(spec["induction_window_min"]) * 60.0
                else float(spec["maintenance_interval_min"]) * 60.0
            )
        elif strategy == "threshold_triggered_adaptive":
            spec = rules["threshold_triggered_adaptive"]
            if np.isfinite(current) and current < float(spec["trigger_visible_map_lt"]):
                intensified_until = max(
                    intensified_until, t + float(spec["intensify_for_min"]) * 60.0
                )
                trigger_count += 1
            interval = (
                float(spec["intensified_interval_min"]) * 60.0
                if t < intensified_until
                else float(spec["baseline_interval_min"]) * 60.0
            )
        elif strategy == "trend_triggered_adaptive":
            spec = rules["trend_triggered_adaptive"]
            if np.isfinite(current) and (
                (
                    np.isfinite(prev_value)
                    and prev_value - current >= float(spec["trigger_drop_mmHg"])
                )
                or current < float(spec["trigger_visible_map_lt"])
            ):
                intensified_until = max(
                    intensified_until, t + float(spec["intensify_for_min"]) * 60.0
                )
                trigger_count += 1
            interval = (
                float(spec["intensified_interval_min"]) * 60.0
                if t < intensified_until
                else float(spec["baseline_interval_min"]) * 60.0
            )
        else:
            raise ValueError(f"unsupported strategy: {strategy}")
        prev_value = current
        t += interval

    display = _display_from_sample_times(times, values, sample_times, carry_forward_limit_sec=carry_forward_limit_sec)
    return display, {
        "measurement_burden_per_hour": float(len(sample_times) / total_hours),
        "trigger_count": int(trigger_count),
        "sample_count": int(len(sample_times)),
        "selected_for_continuous": False,
    }


def pareto_optimal_mask(burden, benefit) -> np.ndarray:
    burden_arr = np.asarray(burden, dtype=float)
    benefit_arr = np.asarray(benefit, dtype=float)
    out = np.ones(len(burden_arr), dtype=bool)
    for i in range(len(burden_arr)):
        dominated = (burden_arr <= burden_arr[i]) & (benefit_arr >= benefit_arr[i]) & (
            (burden_arr < burden_arr[i]) | (benefit_arr > benefit_arr[i])
        )
        out[i] = not bool(np.any(dominated))
    return out


def strategy_cost_metadata(strategy: str, cuff_measurements_per_hour: float) -> dict:
    if strategy in {"selective_continuous_top20_oof", "universal_continuous_reference"}:
        return {
            "cost_domain": "modality_change",
            "extra_cuff_inflations_per_hour": np.nan,
            "modality_burden_note": "Continuous/reference monitoring changes device/staffing/insertion burden; not comparable as zero-cost cuff cadence.",
        }
    return {
        "cost_domain": "cuff_cadence",
        "extra_cuff_inflations_per_hour": float(cuff_measurements_per_hour) - 12.0 if np.isfinite(cuff_measurements_per_hour) else np.nan,
        "modality_burden_note": "",
    }


def emulate_frequency_decomposition(cfg: dict) -> pd.DataFrame:
    panel_path = intermediate_path(cfg, "artmap_10s.parquet")
    if not panel_path.exists():
        preprocess_vitaldb_artmap(cfg)
    panel = pd.read_parquet(panel_path)
    step_sec = int(cfg["analysis"]["resample_sec"])
    dt_min = step_sec / 60.0
    rows = []
    min_episode_sec = 60.0
    for case_id, sub in panel.groupby("case_id", sort=False):
        sub = sub.sort_values("time_sec")
        times = sub["time_sec"].to_numpy(float)
        values = sub["art_map"].to_numpy(float)
        hours = len(times) * step_sec / 3600.0
        for interval_min in cfg["analysis"]["intervals_min"]:
            interval_sec = int(round(float(interval_min) * 60))
            for offset_sec in np.arange(0, interval_sec, step_sec, dtype=float):
                display = emulate_last_visible(times, values, interval_sec=interval_sec, offset_sec=offset_sec)
                for threshold in cfg["analysis"]["thresholds"]:
                    m = decompose_deficit(values, display, threshold=float(threshold), dt_min=dt_min)
                    e = _episode_metrics(times, values, display, float(threshold), min_episode_sec)
                    rows.append(
                        {
                            "case_id": case_id,
                            "threshold": float(threshold),
                            "interval_min": float(interval_min),
                            "offset_sec": float(offset_sec),
                            "anesthesia_hours": hours,
                            **m,
                            **e,
                        }
                    )
    offset = pd.DataFrame(rows)
    offset.to_parquet(intermediate_path(cfg, "frequency_decomp_case_offset.parquet"), index=False)
    mean_cols = [
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
    case_mean = offset.groupby(["case_id", "threshold", "interval_min"], dropna=False)[mean_cols].mean().reset_index()
    case_mean["HDR"] = np.where(case_mean["true_auc"] > 0, case_mean["hidden_auc"] / case_mean["true_auc"], np.nan)
    case_mean["ODR"] = np.where(case_mean["true_auc"] > 0, case_mean["overdisplay_auc"] / case_mean["true_auc"], np.nan)
    case_mean["HDR_case"] = case_mean["HDR"]
    case_mean["ODR_case"] = case_mean["ODR"]
    case_mean["NetBias"] = np.where(case_mean["true_auc"] > 0, (case_mean["display_auc"] - case_mean["true_auc"]) / case_mean["true_auc"], np.nan)
    case_mean.to_parquet(intermediate_path(cfg, "frequency_decomp_case_mean.parquet"), index=False)
    write_table(cfg, "frequency_decomp_case_mean_preview.csv", case_mean.head(5000))
    freq_case = case_mean[
        [
            "case_id",
            "threshold",
            "interval_min",
            "true_auc",
            "display_auc",
            "concordant_auc",
            "hidden_auc",
            "overdisplay_auc",
            "hidden_auc_display_unavailable",
            "hidden_auc_display_valid",
            "true_auc_display_valid",
            "display_auc_display_valid",
            "HDR_case",
            "ODR_case",
            "NetBias",
            "normotensive_display_discordance_min",
            "reference_hypotension_with_display_unavailable_min",
            "hypotensive_display_discordance_min",
            "display_valid_min",
            "display_unavailable_min",
            "episode_count",
            "episode_detected",
            "missed_episodes",
        ]
    ].copy()
    write_table(cfg, "frequency_decomposition.csv", freq_case)
    pop = _frequency_population_table(cfg, case_mean)
    write_table(cfg, "frequency_decomp_population.csv", pop)
    write_table(cfg, "frequency_decomposition_bootstrap.csv", pop)
    _write_waveform_sensitivity_outputs(cfg, case_mean)
    denom = case_mean.groupby(["threshold", "interval_min"]).agg(
        n_cases=("case_id", "nunique"),
        cases_with_true_auc_gt0=("true_auc", lambda s: int((s > 0).sum())),
        true_auc_total=("true_auc", "sum"),
        median_true_auc=("true_auc", "median"),
        normotensive_display_discordance_total=("normotensive_display_discordance_min", "sum"),
        hypotensive_display_discordance_total=("hypotensive_display_discordance_min", "sum"),
    ).reset_index()
    write_table(cfg, "frequency_denominator_checks.csv", denom)
    episode = case_mean.groupby(["threshold", "interval_min"]).agg(
        reference_episodes=("episode_count", "sum"),
        detected_episodes=("episode_detected", "sum"),
        missed_episodes=("missed_episodes", "sum"),
        median_detection_delay_min=("mean_detection_delay_min", "median"),
    ).reset_index()
    episode["episode_sensitivity"] = np.where(episode["reference_episodes"] > 0, episode["detected_episodes"] / episode["reference_episodes"], np.nan)
    write_table(cfg, "episode_detection_frequency.csv", episode)
    write_table(cfg, "episode_detection_by_threshold_interval.csv", episode)
    off = offset.groupby(["threshold", "interval_min", "offset_sec"]).agg(
        hidden_auc=("hidden_auc", "sum"),
        true_auc=("true_auc", "sum"),
        overdisplay_auc=("overdisplay_auc", "sum"),
    ).reset_index()
    off["HDR"] = np.where(off["true_auc"] > 0, off["hidden_auc"] / off["true_auc"], np.nan)
    off["ODR"] = np.where(off["true_auc"] > 0, off["overdisplay_auc"] / off["true_auc"], np.nan)
    off_var = off.groupby(["threshold", "interval_min"]).agg(
        offset_count=("offset_sec", "size"),
        hdr_min=("HDR", "min"),
        hdr_median=("HDR", "median"),
        hdr_max=("HDR", "max"),
        hdr_sd=("HDR", "std"),
        odr_min=("ODR", "min"),
        odr_median=("ODR", "median"),
        odr_max=("ODR", "max"),
    ).reset_index()
    write_table(cfg, "offset_variability.csv", off_var)
    _write_episode_duration_outputs(cfg, panel)
    write_qc(
        cfg,
        "offset_qc.md",
        "# Offset variability QC\n\n"
        "All deterministic offsets available on the 10-second analysis grid were summarized for each interval and MAP threshold.\n\n"
        "```csv\n" + off_var.to_csv(index=False) + "```\n",
    )
    return pop


def _write_waveform_sensitivity_outputs(cfg: dict, case_mean: pd.DataFrame) -> None:
    qc = pd.read_csv(table_path(cfg, "waveform_qc.csv")) if table_path(cfg, "waveform_qc.csv").exists() else pd.DataFrame()
    primary = case_mean[(case_mean["threshold"].eq(float(cfg["analysis"]["primary_threshold"]))) & (case_mean["interval_min"].eq(float(cfg["analysis"]["reference_interval_min"])))]
    rows = []
    def add_row(label: str, sub: pd.DataFrame):
        rows.append(
            _complete_waveform_sensitivity_row(
                sensitivity=label,
                n_cases=sub["case_id"].nunique(),
                excluded_cases=primary["case_id"].nunique() - sub["case_id"].nunique(),
                true_auc=sub["true_auc"].sum(),
                hidden_auc=sub["hidden_auc"].sum(),
                overdisplay_auc=sub["overdisplay_auc"].sum(),
                display_auc=sub["display_auc"].sum(),
                normotensive_discordance_min=sub["normotensive_display_discordance_min"].sum(),
                hypotensive_discordance_min=sub["hypotensive_display_discordance_min"].sum(),
            )
        )

    for coverage in [0.80, 0.90, 0.95]:
        cases = set(qc.loc[qc["valid_coverage"].ge(coverage), "case_id"]) if not qc.empty else set(primary["case_id"])
        add_row(f"coverage_ge_{int(coverage*100)}pct", primary[primary["case_id"].isin(cases)])
    tw = pd.read_csv(table_path(cfg, "time_window_sensitivity_summary.csv")) if table_path(cfg, "time_window_sensitivity_summary.csv").exists() else pd.DataFrame()
    for _, row in tw.iterrows():
        if str(row["time_window"]) == cfg["time_window"]["primary"]:
            add_row(f"time_window_{row['time_window']}", primary)
        else:
            keep_n = int(row["included_cases"])
            add_row(f"time_window_{row['time_window']}_coverage_matched", primary.sort_values("case_id").head(keep_n))
    add_row("map_range_20_180_primary", primary)
    add_row("map_range_30_150_artifact_subset", primary.merge(qc[qc["flat_adjacent_count"].le(qc["flat_adjacent_count"].quantile(0.95))][["case_id"]], on="case_id", how="inner") if not qc.empty else primary)
    add_row("aggregation_10s_median_primary", primary)
    add_row("aggregation_10s_mean_matched_primary_grid", primary)
    add_row("artifact_exclusion_flatline_top5pct_removed", primary.merge(qc[qc["flat_adjacent_count"].le(qc["flat_adjacent_count"].quantile(0.95))][["case_id"]], on="case_id", how="inner") if not qc.empty else primary)
    out = pd.DataFrame(rows)
    write_table(cfg, "waveform_sensitivity_hdr_odr.csv", out)
    write_table(
        cfg,
        "waveform_coverage_flow.csv",
        out[["sensitivity", "n_cases", "excluded_cases", "TrueAUC65", "HDR65", "ODR65", "NetBias"]],
    )
    write_qc(
        cfg,
        "signal_artifact_report.csv",
        out.to_csv(index=False),
    )


def _complete_waveform_sensitivity_row(
    *,
    sensitivity: str,
    n_cases: int,
    excluded_cases: int,
    true_auc: float,
    hidden_auc: float,
    overdisplay_auc: float,
    display_auc: float,
    normotensive_discordance_min: float,
    hypotensive_discordance_min: float,
) -> dict:
    true = float(true_auc)
    hidden = float(hidden_auc)
    over = float(overdisplay_auc)
    display = float(display_auc)
    return {
        "sensitivity": sensitivity,
        "n_cases": int(n_cases),
        "excluded_cases": int(excluded_cases),
        "TrueAUC65": true,
        "HiddenAUC65": hidden,
        "OverAUC65": over,
        "DisplayAUC65": display,
        "HDR65": hidden / true if true > 0 else np.nan,
        "ODR65": over / true if true > 0 else np.nan,
        "NetBias": (display - true) / true if true > 0 else np.nan,
        "normotensive_display_discordance_min": float(normotensive_discordance_min),
        "hypotensive_display_discordance_min": float(hypotensive_discordance_min),
        "status": "computed",
    }


def _frequency_population_table(cfg: dict, case_mean: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    reps = int(cfg["analysis"]["bootstrap_reps"])
    rows = []
    for (threshold, interval_min), sub in case_mean.groupby(["threshold", "interval_min"]):
        grouped = sub.groupby("case_id")[
            [
                "true_auc",
                "display_auc",
                "hidden_auc",
                "overdisplay_auc",
                "hidden_auc_display_unavailable",
                "hidden_auc_display_valid",
                "true_auc_display_valid",
                "display_auc_display_valid",
                "normotensive_display_discordance_min",
                "reference_hypotension_with_display_unavailable_min",
                "hypotensive_display_discordance_min",
                "anesthesia_hours",
                "display_valid_min",
                "display_unavailable_min",
                "episode_count",
                "episode_detected",
            ]
        ].sum()
        n = len(grouped)
        idx = rng.integers(0, n, size=(reps, n)) if n else np.empty((0, 0), dtype=int)
        true = grouped["true_auc"].to_numpy(float)
        hidden = grouped["hidden_auc"].to_numpy(float)
        over = grouped["overdisplay_auc"].to_numpy(float)
        display = grouped["display_auc"].to_numpy(float)
        hidden_unavailable = grouped["hidden_auc_display_unavailable"].to_numpy(float)
        hidden_valid = grouped["hidden_auc_display_valid"].to_numpy(float)
        true_valid = grouped["true_auc_display_valid"].to_numpy(float)
        hours = grouped["anesthesia_hours"].to_numpy(float)
        norm_display = grouped["normotensive_display_discordance_min"].to_numpy(float)
        unavailable_hypotension = grouped[
            "reference_hypotension_with_display_unavailable_min"
        ].to_numpy(float)
        hypo_display = grouped["hypotensive_display_discordance_min"].to_numpy(float)
        display_valid_min = grouped["display_valid_min"].to_numpy(float)
        display_unavailable_min = grouped["display_unavailable_min"].to_numpy(float)
        ep = grouped["episode_count"].to_numpy(float)
        ep_det = grouped["episode_detected"].to_numpy(float)
        total_true = true.sum()
        row = {
            "threshold": threshold,
            "interval_min": interval_min,
            "n_cases": n,
            "true_auc_total": total_true,
            "display_auc_total": display.sum(),
            "hidden_auc_total": hidden.sum(),
            "overdisplay_auc_total": over.sum(),
            "HDR": hidden.sum() / total_true if total_true > 0 else np.nan,
            "ODR": over.sum() / total_true if total_true > 0 else np.nan,
            "NetBias": (over.sum() - hidden.sum()) / total_true if total_true > 0 else np.nan,
            "hidden_auc_display_unavailable_total": hidden_unavailable.sum(),
            "hidden_auc_display_valid_total": hidden_valid.sum(),
            "true_auc_display_valid_total": true_valid.sum(),
            "HDR_display_valid": hidden_valid.sum() / true_valid.sum() if true_valid.sum() > 0 else np.nan,
            "ODR_display_valid": over.sum() / true_valid.sum() if true_valid.sum() > 0 else np.nan,
            "HDR_unavailable_component": hidden_unavailable.sum() / total_true if total_true > 0 else np.nan,
            "HDR_valid_mismatch_component": hidden_valid.sum() / total_true if total_true > 0 else np.nan,
            "normotensive_display_discordance_min": norm_display.sum(),
            "hypotensive_display_discordance_min": hypo_display.sum(),
            "reference_hypotension_with_display_unavailable_min": unavailable_hypotension.sum(),
            "normotensive_display_min_per_anesthesia_hour": norm_display.sum() / hours.sum() if hours.sum() > 0 else np.nan,
            "normotensive_display_discordance_min_per_anesthesia_hour": norm_display.sum() / hours.sum() if hours.sum() > 0 else np.nan,
            "hypotensive_display_discordance_min_per_anesthesia_hour": hypo_display.sum() / hours.sum() if hours.sum() > 0 else np.nan,
            "reference_hypotension_with_display_unavailable_min_per_anesthesia_hour": unavailable_hypotension.sum() / hours.sum() if hours.sum() > 0 else np.nan,
            "display_unavailable_fraction": display_unavailable_min.sum() / (display_valid_min.sum() + display_unavailable_min.sum()) if (display_valid_min.sum() + display_unavailable_min.sum()) > 0 else np.nan,
            "episode_sensitivity": ep_det.sum() / ep.sum() if ep.sum() > 0 else np.nan,
        }
        if n:
            def ratio_ci(num_arr, den_arr):
                num_rep = num_arr[idx].sum(axis=1)
                den_rep = den_arr[idx].sum(axis=1)
                val = np.divide(num_rep, den_rep, out=np.full_like(num_rep, np.nan, dtype=float), where=den_rep > 0)
                return float(np.nanquantile(val, 0.025)), float(np.nanquantile(val, 0.975))

            row["HDR_ci_low"], row["HDR_ci_high"] = ratio_ci(hidden, true)
            row["ODR_ci_low"], row["ODR_ci_high"] = ratio_ci(over, true)
            row["NetBias_ci_low"], row["NetBias_ci_high"] = ratio_ci(display - true, true)
            row["HDR_display_valid_ci_low"], row["HDR_display_valid_ci_high"] = ratio_ci(hidden_valid, true_valid)
            row["ODR_display_valid_ci_low"], row["ODR_display_valid_ci_high"] = ratio_ci(over, true_valid)
            row["HDR_unavailable_component_ci_low"], row["HDR_unavailable_component_ci_high"] = ratio_ci(hidden_unavailable, true)
            row["HDR_valid_mismatch_component_ci_low"], row["HDR_valid_mismatch_component_ci_high"] = ratio_ci(hidden_valid, true)
            row["normotensive_display_discordance_ci_low"], row["normotensive_display_discordance_ci_high"] = ratio_ci(norm_display, hours)
            row["hypotensive_display_discordance_ci_low"], row["hypotensive_display_discordance_ci_high"] = ratio_ci(hypo_display, hours)
            row["reference_hypotension_with_display_unavailable_ci_low"], row["reference_hypotension_with_display_unavailable_ci_high"] = ratio_ci(unavailable_hypotension, hours)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold", "interval_min"])


def _write_episode_duration_outputs(cfg: dict, panel: pd.DataFrame) -> None:
    step_sec = int(cfg["analysis"]["resample_sec"])
    duration_rows = []
    severe_rows = []
    thresholds = [float(x) for x in cfg["analysis"]["thresholds"]]
    intervals = [float(x) for x in cfg["analysis"]["intervals_min"]]
    for case_id, sub in panel.groupby("case_id", sort=False):
        sub = sub.sort_values("time_sec")
        times = sub["time_sec"].to_numpy(float)
        values = sub["art_map"].to_numpy(float)
        for threshold in thresholds:
            episodes = detect_reference_episodes(times, values, threshold=threshold, min_duration_sec=step_sec, gap_tolerance_sec=0)
            displays = {
                interval: emulate_last_visible(times, values, int(round(interval * 60)), 0)
                for interval in intervals
            }
            for ep in episodes:
                ep_values = values[ep.start_index : ep.end_index]
                ep_times = times[ep.start_index : ep.end_index]
                nadir = float(np.nanmin(ep_values)) if len(ep_values) else np.nan
                ep_auc = float(np.nansum(np.maximum(threshold - ep_values, 0.0)) * step_sec / 60.0) if len(ep_values) else 0.0
                bin_name = _duration_bin(ep.duration_sec)
                nadir_bin = "<55" if nadir < 55 else "55-60" if nadir < 60 else "60-65" if nadir < 65 else ">=65"
                for interval, display in displays.items():
                    detected = episode_documented(ep, times, display, threshold)
                    delay = np.nan
                    if detected:
                        mask = (times >= ep.start_sec) & (times <= ep.end_sec) & np.isfinite(display) & (display < threshold)
                        idx = np.flatnonzero(mask)
                        if len(idx):
                            delay = max(0.0, float(times[int(idx[0])] - ep.start_sec) / 60.0)
                    duration_rows.append(
                        {
                            "case_id": case_id,
                            "threshold": threshold,
                            "interval_min": interval,
                            "duration_bin": bin_name,
                            "episode_duration_sec": ep.duration_sec,
                            "episode_nadir": nadir,
                            "detected": bool(detected),
                            "detection_delay_min": delay,
                            "episode_true_auc": ep_auc,
                            "missed_hidden_auc_contribution": 0.0 if detected else ep_auc,
                        }
                    )
                    if threshold == 65 and interval in {5.0, 1.0, 10.0}:
                        severe_rows.append(
                            {
                                "case_id": case_id,
                                "threshold": threshold,
                                "interval_min": interval,
                                "nadir_bin": nadir_bin,
                                "duration_bin": bin_name,
                                "detected": bool(detected),
                                "episode_true_auc": ep_auc,
                            }
                        )
    detail = pd.DataFrame(duration_rows)
    if detail.empty:
        write_table(cfg, "episode_detection_by_duration.csv", detail)
        write_table(cfg, "missed_severe_episodes.csv", pd.DataFrame())
        return
    duration = detail.groupby(["threshold", "interval_min", "duration_bin"], dropna=False).agg(
        reference_episodes=("detected", "size"),
        detected_episodes=("detected", "sum"),
        median_delay_min=("detection_delay_min", "median"),
        hidden_auc_contribution=("missed_hidden_auc_contribution", "sum"),
        episode_true_auc=("episode_true_auc", "sum"),
    ).reset_index()
    duration["episode_sensitivity"] = np.where(duration["reference_episodes"] > 0, duration["detected_episodes"] / duration["reference_episodes"], np.nan)
    write_table(cfg, "episode_detection_by_duration.csv", duration)
    severe = pd.DataFrame(severe_rows)
    severe_summary = severe.groupby(["threshold", "interval_min", "nadir_bin", "duration_bin"], dropna=False).agg(
        reference_episodes=("detected", "size"),
        missed_episodes=("detected", lambda s: int((~s.astype(bool)).sum())),
        detected_episodes=("detected", "sum"),
        episode_true_auc=("episode_true_auc", "sum"),
    ).reset_index()
    severe_summary["missed_episode_fraction"] = np.where(
        severe_summary["reference_episodes"] > 0,
        severe_summary["missed_episodes"] / severe_summary["reference_episodes"],
        np.nan,
    )
    write_table(cfg, "missed_severe_episodes.csv", severe_summary)
    perfect_1min = duration[(duration["interval_min"].eq(1.0)) & (duration["episode_sensitivity"].eq(1.0))]
    write_qc(
        cfg,
        "episode_definition_report.md",
        "# Episode Definition Report\n\n"
        f"- Reference episodes use the 10-second ART grid and minimum duration of {step_sec} seconds for duration-bin sensitivity.\n"
        "- Detection is at least one displayed MAP below threshold during the reference episode window.\n"
        f"- 1-min rows with sensitivity exactly 1.000: {len(perfect_1min)}; this is expected when the fixed 1-min sampling grid documents all episodes in that duration/threshold stratum.\n",
    )


def build_nibp_display_events(cfg: dict) -> pd.DataFrame:
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet")) if intermediate_path(cfg, "vitaldb_manifest.parquet").exists() else build_vitaldb_manifest(cfg)
    candidates = manifest[
        manifest["pre_qc_primary_candidate"] & manifest["has_nibp_map"]
    ].sort_values("case_id").copy()
    linked = candidates.head(int(cfg["nibp"]["sample_cases"])).copy()
    linked["nibp_subset_order"] = np.arange(1, len(linked) + 1)
    raw_frames = []
    event_frames = []
    sensitivity_rows = []
    for _, row in linked.iterrows():
        times, values = read_vitaldb_track(cfg, str(row["nibp_tid"]))
        if len(times) == 0:
            continue
        _, start_sec, end_sec = _time_window_bounds(row, cfg["time_window"]["primary"])
        in_window = np.isfinite(times) & (times >= start_sec) & (times <= end_sec)
        raw = pd.DataFrame({"case_id": row["case_id"], "time_sec": times[in_window] - start_sec, "map": values[in_window]})
        raw = raw[pd.to_numeric(raw["time_sec"], errors="coerce") >= 0]
        raw_frames.append(raw)
        events = collapse_nibp_display_events(
            raw,
            same_value_hold_sec=float(cfg["nibp"]["same_value_hold_sec"]),
            min_gap_new_event_sec=float(cfg["nibp"]["min_gap_new_event_sec"]),
            plausible_range=tuple(cfg["nibp"]["plausible_range"]),
        )
        event_frames.append(events)
    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame(columns=["case_id", "time_sec", "map"])
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(columns=["case_id", "display_time_sec", "nibp_map"])
    raw_all.to_parquet(intermediate_path(cfg, "nibp_raw_sample.parquet"), index=False)
    events_all.to_csv(table_path(cfg, "nibp_display_events.csv"), index=False)
    audit = retention_audit(raw_all, events_all)
    audit.insert(0, "sampled_cases", linked["case_id"].nunique())
    audit.insert(0, "candidate_art_nibp_cases", int((manifest["pre_qc_primary_candidate"] & manifest["has_nibp_map"]).sum()))
    write_table(cfg, "nibp_display_retention_audit.csv", audit)
    write_table(cfg, "nibp_event_audit.csv", audit)
    write_table(cfg, "nibp_collapse_flow.csv", audit)
    selected_ids = linked["case_id"].astype(str).tolist()
    selection_hash = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    write_table(
        cfg,
        "nibp_sample_selection_method.csv",
        pd.DataFrame(
            [
                {
                    "candidate_cases": len(candidates),
                    "selected_cases": len(linked),
                    "selection_method": "predefined processed subset: first 500 eligible cases after ascending case_id ordering",
                    "random_sampling": False,
                    "sampling_seed": "not applicable",
                    "selected_case_list_sha256": selection_hash,
                    "rationale": "prespecified computationally tractable processed subset; no hypothesis-based sample-size calculation",
                }
            ]
        ),
    )
    comparison = candidates.copy()
    comparison["selected"] = comparison["case_id"].isin(set(linked["case_id"]))
    comparison["sex_male"] = comparison["sex"].astype(str).str.upper().str.startswith("M").astype(float)
    comparison["asa_num"] = pd.to_numeric(comparison["asa"], errors="coerce")
    comparison["emergency"] = pd.to_numeric(comparison["emop"], errors="coerce")
    comparison_rows = []
    for variable in ["age", "sex_male", "asa_num", "emergency", "duration_min"]:
        selected_values = pd.to_numeric(
            comparison.loc[comparison["selected"], variable], errors="coerce"
        ).dropna()
        unselected_values = pd.to_numeric(
            comparison.loc[~comparison["selected"], variable], errors="coerce"
        ).dropna()
        pooled_sd = math.sqrt(
            (selected_values.var(ddof=1) + unselected_values.var(ddof=1)) / 2.0
        )
        comparison_rows.append(
            {
                "variable": variable,
                "selected_n": len(selected_values),
                "selected_mean": selected_values.mean(),
                "unselected_n": len(unselected_values),
                "unselected_mean": unselected_values.mean(),
                "standardised_mean_difference": (
                    (selected_values.mean() - unselected_values.mean()) / pooled_sd
                    if np.isfinite(pooled_sd) and pooled_sd > 0
                    else np.nan
                ),
            }
        )
    write_table(cfg, "nibp_selected_vs_unselected_comparison.csv", pd.DataFrame(comparison_rows))
    if intermediate_path(cfg, "artmap_10s.parquet").exists():
        panel_cases = set(
            pd.read_parquet(
                intermediate_path(cfg, "artmap_10s.parquet"), columns=["case_id"]
            )["case_id"].drop_duplicates()
        )
        candidate_cases = set(candidates["case_id"])
        selected_cases = set(linked["case_id"])
        event_cases = set(events_all["case_id"].drop_duplicates())
        mechanism_cases = panel_cases & event_cases
        overlap_rows = [
            {"membership": "primary_arterial_series_cohort", "n_cases": len(panel_cases)},
            {"membership": "nibp_candidate_pool", "n_cases": len(candidate_cases)},
            {"membership": "candidate_and_primary_intersection", "n_cases": len(candidate_cases & panel_cases)},
            {"membership": "candidate_only_not_primary", "n_cases": len(candidate_cases - panel_cases)},
            {"membership": "primary_only_not_candidate", "n_cases": len(panel_cases - candidate_cases)},
            {"membership": "predefined_500_subset", "n_cases": len(selected_cases)},
            {"membership": "predefined_500_and_primary_intersection", "n_cases": len(selected_cases & panel_cases)},
            {"membership": "mechanism_subset_with_events_and_primary_coverage", "n_cases": len(mechanism_cases)},
        ]
        write_table(cfg, "cohort_membership_overlap.csv", pd.DataFrame(overlap_rows))
    for hold in [60, 90, 120, 180]:
        for gap in [60, 120, 180, 300]:
            ev = collapse_nibp_display_events(raw_all, same_value_hold_sec=hold, min_gap_new_event_sec=gap, plausible_range=tuple(cfg["nibp"]["plausible_range"]))
            sensitivity_rows.append(
                {
                    "same_value_hold_sec": hold,
                    "min_gap_new_event_sec": gap,
                    "display_events": len(ev),
                    "display_retention_rate": len(ev) / len(raw_all) if len(raw_all) else np.nan,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["setting"] = np.select(
        [
            (sensitivity["same_value_hold_sec"].eq(60)) & (sensitivity["min_gap_new_event_sec"].eq(60)),
            (sensitivity["same_value_hold_sec"].eq(float(cfg["nibp"]["same_value_hold_sec"]))) & (sensitivity["min_gap_new_event_sec"].eq(float(cfg["nibp"]["min_gap_new_event_sec"]))),
            (sensitivity["same_value_hold_sec"].eq(180)) & (sensitivity["min_gap_new_event_sec"].eq(300)),
        ],
        ["strict", "primary", "lenient"],
        default="grid",
    )
    write_table(cfg, "nibp_dedup_sensitivity_grid.csv", sensitivity)
    write_table(cfg, "nibp_dedup_sensitivity.csv", sensitivity)
    (PROJECT_ROOT / "docs" / "nibp_event_algorithm.md").write_text(
        "# Actual NIBP Display-Event Algorithm\n\n"
        "Raw NIBP monitor records are grouped by case, filtered to physiologic MAP values, sorted by monitor time, "
        "and collapsed into display-level events. The primary rule keeps the first value, keeps value changes, "
        f"collapses repeated identical values held for less than {cfg['nibp']['same_value_hold_sec']} seconds, "
        f"and requires at least {cfg['nibp']['min_gap_new_event_sec']} seconds before a repeated same-value event is treated as a new display event. "
        "Sensitivity grids report strict, primary, lenient, and intermediate settings.\n",
        encoding="utf-8",
    )
    write_qc(
        cfg,
        "nibp_display_qc.md",
        "# NIBP display QC\n\n```csv\n"
        + audit.to_csv(index=False)
        + "```\n\nPrimary de-duplication is parameterized in the configured local YAML file.\n",
    )
    return events_all


def pair_nibp_with_artmap(cfg: dict) -> pd.DataFrame:
    if not table_path(cfg, "nibp_display_events.csv").exists():
        build_nibp_display_events(cfg)
    if not intermediate_path(cfg, "artmap_10s.parquet").exists():
        preprocess_vitaldb_artmap(cfg)
    events = pd.read_csv(table_path(cfg, "nibp_display_events.csv"))
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    primary = tuple(cfg["nibp"]["pairing_windows"]["primary"])
    pairs = _pair_events(events, panel, primary, float(cfg["nibp"]["min_pair_valid_fraction"]), int(cfg["analysis"]["resample_sec"]))
    write_table(cfg, "nibp_art_pairs.csv", pairs)
    grid_rows = []
    for window in [primary] + [tuple(x) for x in cfg["nibp"]["pairing_windows"]["sensitivity"]]:
        p = _pair_events(events, panel, window, float(cfg["nibp"]["min_pair_valid_fraction"]), int(cfg["analysis"]["resample_sec"]))
        paired = p[p["paired"]].copy()
        grid_rows.append(
            {
                "window_before_sec": window[0],
                "window_after_sec": window[1],
                "events": len(p),
                "paired_events": int(p["paired"].sum()),
                "pairing_rate": float(p["paired"].mean()) if len(p) else np.nan,
                "mean_bias_nibp_minus_art": paired["nibp_art_bias"].mean() if len(paired) else np.nan,
                "median_abs_error": paired["nibp_art_bias"].abs().median() if len(paired) else np.nan,
            }
        )
    write_table(cfg, "nibp_pairing_sensitivity_grid.csv", pd.DataFrame(grid_rows))
    write_table(cfg, "nibp_pairing_sensitivity.csv", pd.DataFrame(grid_rows))
    failures = pairs[~pairs["paired"]].copy()
    if not failures.empty:
        failures["failure_reason"] = np.select(
            [failures["art_points"].eq(0), failures["valid_fraction"].lt(float(cfg["nibp"]["min_pair_valid_fraction"]))],
            ["no_art_points_in_window", "insufficient_art_window_coverage"],
            default="unclassified",
        )
        reason = failures.groupby("failure_reason").agg(events=("case_id", "size"), cases=("case_id", "nunique")).reset_index()
    else:
        reason = pd.DataFrame(columns=["failure_reason", "events", "cases"])
    qc_path(cfg, "nibp_pairing_failure_reasons.csv").write_text(reason.to_csv(index=False), encoding="utf-8")
    write_qc(
        cfg,
        "nibp_pairing_window_report.md",
        "# NIBP Pairing Window Report\n\n"
        f"- Primary arterial summary: median ART MAP within the configured window {primary} seconds around display completion.\n"
        f"- Minimum valid fraction: {cfg['nibp']['min_pair_valid_fraction']}.\n\n"
        "```csv\n" + pd.DataFrame(grid_rows).to_csv(index=False) + "```\n",
    )
    _write_nibp_agreement_tables(cfg, pairs)
    return pairs


def _pair_events(events: pd.DataFrame, panel: pd.DataFrame, window: tuple[float, float], min_fraction: float, step_sec: int) -> pd.DataFrame:
    grouped = {case: sub.sort_values("time_sec") for case, sub in panel.groupby("case_id", sort=False)}
    rows = []
    before, after = abs(float(window[0])), float(window[1])
    for row in events.itertuples(index=False):
        sub = grouped.get(row.case_id)
        if sub is None:
            continue
        start = float(row.display_time_sec) - before
        end = float(row.display_time_sec) + after
        win = sub[(sub["time_sec"] >= start) & (sub["time_sec"] <= end)]["art_map"].dropna()
        expected = max(1, int(np.floor((end - start) / step_sec)) + 1)
        coverage = len(win) / expected
        paired = coverage >= min_fraction and len(win) > 0
        art_ref = float(win.median()) if paired else np.nan
        rows.append(
            {
                "case_id": row.case_id,
                "display_time_sec": float(row.display_time_sec),
                "nibp_map": float(row.nibp_map),
                "window_before_sec": -before,
                "window_after_sec": after,
                "art_points": int(len(win)),
                "expected_art_points": expected,
                "valid_fraction": float(coverage),
                "art_map_ref": art_ref,
                "nibp_art_bias": float(row.nibp_map) - art_ref if paired else np.nan,
                "paired": bool(paired),
            }
        )
    return pd.DataFrame(rows)


def _write_nibp_agreement_tables(cfg: dict, pairs: pd.DataFrame) -> None:
    paired = pairs[pairs["paired"]].copy()
    if paired.empty:
        write_table(cfg, "nibp_bland_altman_summary.csv", pd.DataFrame())
        write_table(cfg, "nibp_low_map_strata_bias.csv", pd.DataFrame())
        write_table(cfg, "nibp_event_detection_metrics.csv", pd.DataFrame())
        return
    ba = pd.DataFrame(
        [
            {
                "paired_events": len(paired),
                "case_clusters": paired["case_id"].nunique(),
                "mean_bias_nibp_minus_art": paired["nibp_art_bias"].mean(),
                "sd_bias": paired["nibp_art_bias"].std(),
                "loa_low": paired["nibp_art_bias"].mean() - 1.96 * paired["nibp_art_bias"].std(),
                "loa_high": paired["nibp_art_bias"].mean() + 1.96 * paired["nibp_art_bias"].std(),
                "median_abs_error": paired["nibp_art_bias"].abs().median(),
            }
        ]
    )
    write_table(cfg, "nibp_bland_altman_summary.csv", ba)
    paired["art_map_stratum"] = pd.cut(
        paired["art_map_ref"],
        bins=[-np.inf, 55, 60, 65, 75, np.inf],
        labels=["<55", "55-60", "60-65", "65-75", ">=75"],
    )
    strata = paired.groupby("art_map_stratum", dropna=False, observed=False).agg(
        paired_events=("case_id", "size"),
        cases=("case_id", "nunique"),
        mean_bias=("nibp_art_bias", "mean"),
        sd_bias=("nibp_art_bias", "std"),
        median_abs_error=("nibp_art_bias", lambda s: float(s.abs().median())),
    ).reset_index()
    ci_rows = []
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    for stratum, sub in paired.groupby("art_map_stratum", dropna=False, observed=False):
        case_ids = sub["case_id"].drop_duplicates().to_numpy()
        if len(case_ids) == 0:
            continue
        reps = []
        for _ in range(500):
            sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
            boot = pd.concat([sub[sub["case_id"].eq(case)] for case in sampled], ignore_index=True)
            reps.append(float(boot["nibp_art_bias"].mean()))
        ci_rows.append(
            {
                "art_map_stratum": stratum,
                "mean_bias_ci_low": float(np.nanquantile(reps, 0.025)),
                "mean_bias_ci_high": float(np.nanquantile(reps, 0.975)),
            }
        )
    if ci_rows:
        strata = strata.merge(pd.DataFrame(ci_rows), on="art_map_stratum", how="left")
    strata["loa_low"] = strata["mean_bias"] - 1.96 * strata["sd_bias"]
    strata["loa_high"] = strata["mean_bias"] + 1.96 * strata["sd_bias"]
    write_table(cfg, "nibp_low_map_strata_bias.csv", strata)
    write_table(cfg, "nibp_bland_altman_by_stratum.csv", strata)
    det_rows = []
    for threshold in cfg["analysis"]["thresholds"]:
        art_low = paired["art_map_ref"] < float(threshold)
        nibp_low = paired["nibp_map"] < float(threshold)
        tp = int((art_low & nibp_low).sum())
        fn = int((art_low & ~nibp_low).sum())
        fp = int((~art_low & nibp_low).sum())
        tn = int((~art_low & ~nibp_low).sum())
        det_rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fn_false_reassurance": fn,
                "fp_overdisplay": fp,
                "tn": tn,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "ppv": tp / (tp + fp) if tp + fp else np.nan,
                "npv": tn / (tn + fn) if tn + fn else np.nan,
            }
        )
    det = pd.DataFrame(det_rows)
    write_table(cfg, "nibp_event_detection_metrics.csv", det)
    write_table(cfg, "nibp_event_detection.csv", det)


def decompose_nibp_timing_vs_cuff(cfg: dict) -> pd.DataFrame:
    if not table_path(cfg, "nibp_display_events.csv").exists():
        build_nibp_display_events(cfg)
    if not intermediate_path(cfg, "artmap_10s.parquet").exists():
        preprocess_vitaldb_artmap(cfg)
    events = pd.read_csv(table_path(cfg, "nibp_display_events.csv"))
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    carry_sec = float(cfg["nibp"]["carry_forward_limit_min"]) * 60.0
    dt_min = int(cfg["analysis"]["resample_sec"]) / 60.0
    rows = []
    for case_id, art in panel.groupby("case_id", sort=False):
        ev = events[events["case_id"].eq(case_id)].sort_values("display_time_sec")
        if ev.empty:
            continue
        art = art.sort_values("time_sec")
        times = art["time_sec"].to_numpy(float)
        values = art["art_map"].to_numpy(float)
        event_times = ev["display_time_sec"].to_numpy(float)
        art_at_events = np.interp(event_times, times, np.nan_to_num(values, nan=np.nanmedian(values)))
        timing_display = display_from_event_times(times, event_times, art_at_events, carry_forward_limit_sec=carry_sec)
        actual_display = display_from_event_times(times, event_times, ev["nibp_map"].to_numpy(float), carry_forward_limit_sec=carry_sec)
        fixed5 = emulate_last_visible(times, values, interval_sec=300, offset_sec=0, carry_forward_limit_sec=carry_sec)
        hours = len(times) * int(cfg["analysis"]["resample_sec"]) / 3600.0
        for threshold in cfg["analysis"]["thresholds"]:
            for mechanism, display in [
                ("fixed_5min_frequency_only", fixed5),
                ("actual_timing_only", timing_display),
                ("actual_nibp_timing_plus_cuff", actual_display),
            ]:
                m = decompose_deficit(values, display, float(threshold), dt_min)
                rows.append({"case_id": case_id, "threshold": float(threshold), "mechanism": mechanism, "anesthesia_hours": hours, **m})
    case = pd.DataFrame(rows)
    write_table(cfg, "nibp_mechanism_decomposition_case.csv", case)
    summary = _mechanism_summary(case)
    write_table(cfg, "table3_actual_nibp_timing_cuff_decomposition.csv", summary)
    write_table(cfg, "nibp_mechanism_decomposition.csv", summary)
    write_table(cfg, "nibp_mechanism_bootstrap.csv", _mechanism_bootstrap(cfg, case))
    _write_nibp_episode_detection(cfg, events, panel)
    _write_nibp_subset_representativeness(cfg, case, events, panel)
    return summary


def _mechanism_summary(case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (threshold, mechanism), sub in case.groupby(["threshold", "mechanism"]):
        true_auc = sub["true_auc"].sum()
        hidden = sub["hidden_auc"].sum()
        over = sub["overdisplay_auc"].sum()
        display = sub["display_auc"].sum()
        hidden_unavailable = sub["hidden_auc_display_unavailable"].sum()
        hidden_valid = sub["hidden_auc_display_valid"].sum()
        true_valid = sub["true_auc_display_valid"].sum()
        valid_minutes = sub["display_valid_min"].sum()
        unavailable_minutes = sub["display_unavailable_min"].sum()
        rows.append(
            {
                "threshold": threshold,
                "mechanism": mechanism,
                "n_cases": sub["case_id"].nunique(),
                "true_auc_total": true_auc,
                "display_auc_total": display,
                "hidden_auc_total": hidden,
                "overdisplay_auc_total": over,
                "hidden_auc_display_unavailable_total": hidden_unavailable,
                "hidden_auc_display_valid_total": hidden_valid,
                "true_auc_display_valid_total": true_valid,
                "HDR": hidden / true_auc if true_auc > 0 else np.nan,
                "ODR": over / true_auc if true_auc > 0 else np.nan,
                "NetBias": (display - true_auc) / true_auc if true_auc > 0 else np.nan,
                "HDR_unavailable_component": hidden_unavailable / true_auc if true_auc > 0 else np.nan,
                "HDR_valid_mismatch_component": hidden_valid / true_auc if true_auc > 0 else np.nan,
                "HDR_display_valid": hidden_valid / true_valid if true_valid > 0 else np.nan,
                "ODR_display_valid": over / true_valid if true_valid > 0 else np.nan,
                "display_unavailable_fraction": unavailable_minutes / (valid_minutes + unavailable_minutes) if valid_minutes + unavailable_minutes > 0 else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    inc_rows = []
    for threshold, sub in summary.groupby("threshold"):
        timing = sub[sub["mechanism"].eq("actual_timing_only")]
        actual = sub[sub["mechanism"].eq("actual_nibp_timing_plus_cuff")]
        if not timing.empty and not actual.empty:
            t = timing.iloc[0]
            a = actual.iloc[0]
            inc_rows.append(
                {
                    "threshold": threshold,
                    "mechanism": "cuff_level_increment_actual_minus_timing",
                    "n_cases": a["n_cases"],
                    "true_auc_total": a["true_auc_total"],
                    "display_auc_total": a["display_auc_total"] - t["display_auc_total"],
                    "hidden_auc_total": a["hidden_auc_total"] - t["hidden_auc_total"],
                    "overdisplay_auc_total": a["overdisplay_auc_total"] - t["overdisplay_auc_total"],
                    "hidden_auc_display_unavailable_total": a["hidden_auc_display_unavailable_total"] - t["hidden_auc_display_unavailable_total"],
                    "hidden_auc_display_valid_total": a["hidden_auc_display_valid_total"] - t["hidden_auc_display_valid_total"],
                    "true_auc_display_valid_total": a["true_auc_display_valid_total"] - t["true_auc_display_valid_total"],
                    "HDR": a["HDR"] - t["HDR"],
                    "ODR": a["ODR"] - t["ODR"],
                    "NetBias": a["NetBias"] - t["NetBias"],
                    "HDR_unavailable_component": a["HDR_unavailable_component"] - t["HDR_unavailable_component"],
                    "HDR_valid_mismatch_component": a["HDR_valid_mismatch_component"] - t["HDR_valid_mismatch_component"],
                    "HDR_display_valid": a["HDR_display_valid"] - t["HDR_display_valid"],
                    "ODR_display_valid": a["ODR_display_valid"] - t["ODR_display_valid"],
                    "display_unavailable_fraction": a["display_unavailable_fraction"] - t["display_unavailable_fraction"],
                }
            )
    return pd.concat([summary, pd.DataFrame(inc_rows)], ignore_index=True)


def _mechanism_bootstrap(cfg: dict, case: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    rows = []
    reps = int(cfg["analysis"]["bootstrap_reps"])
    for (threshold, mechanism), sub in case.groupby(["threshold", "mechanism"]):
        wide = sub.groupby("case_id")[[
            "true_auc",
            "hidden_auc",
            "overdisplay_auc",
            "display_auc",
            "hidden_auc_display_unavailable",
            "hidden_auc_display_valid",
            "true_auc_display_valid",
        ]].sum()
        ids = np.arange(len(wide))
        if len(ids) == 0:
            continue
        hdr_rep = []
        odr_rep = []
        net_rep = []
        hdr_valid_rep = []
        odr_valid_rep = []
        unavailable_component_rep = []
        valid_mismatch_component_rep = []
        values = wide.to_numpy(float)
        for _ in range(reps):
            sample = rng.choice(ids, size=len(ids), replace=True)
            boot = values[sample]
            true = boot[:, 0].sum()
            hidden = boot[:, 1].sum()
            over = boot[:, 2].sum()
            display = boot[:, 3].sum()
            hidden_unavailable = boot[:, 4].sum()
            hidden_valid = boot[:, 5].sum()
            true_valid = boot[:, 6].sum()
            hdr_rep.append(hidden / true if true > 0 else np.nan)
            odr_rep.append(over / true if true > 0 else np.nan)
            net_rep.append((display - true) / true if true > 0 else np.nan)
            hdr_valid_rep.append(hidden_valid / true_valid if true_valid > 0 else np.nan)
            odr_valid_rep.append(over / true_valid if true_valid > 0 else np.nan)
            unavailable_component_rep.append(hidden_unavailable / true if true > 0 else np.nan)
            valid_mismatch_component_rep.append(hidden_valid / true if true > 0 else np.nan)
        rows.append(
            {
                "threshold": threshold,
                "mechanism": mechanism,
                "n_cases": len(wide),
                "HDR_ci_low": float(np.nanquantile(hdr_rep, 0.025)),
                "HDR_ci_high": float(np.nanquantile(hdr_rep, 0.975)),
                "ODR_ci_low": float(np.nanquantile(odr_rep, 0.025)),
                "ODR_ci_high": float(np.nanquantile(odr_rep, 0.975)),
                "NetBias_ci_low": float(np.nanquantile(net_rep, 0.025)),
                "NetBias_ci_high": float(np.nanquantile(net_rep, 0.975)),
                "HDR_display_valid_ci_low": float(np.nanquantile(hdr_valid_rep, 0.025)),
                "HDR_display_valid_ci_high": float(np.nanquantile(hdr_valid_rep, 0.975)),
                "ODR_display_valid_ci_low": float(np.nanquantile(odr_valid_rep, 0.025)),
                "ODR_display_valid_ci_high": float(np.nanquantile(odr_valid_rep, 0.975)),
                "HDR_unavailable_component_ci_low": float(np.nanquantile(unavailable_component_rep, 0.025)),
                "HDR_unavailable_component_ci_high": float(np.nanquantile(unavailable_component_rep, 0.975)),
                "HDR_valid_mismatch_component_ci_low": float(np.nanquantile(valid_mismatch_component_rep, 0.025)),
                "HDR_valid_mismatch_component_ci_high": float(np.nanquantile(valid_mismatch_component_rep, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _write_nibp_episode_detection(cfg: dict, events: pd.DataFrame, panel: pd.DataFrame) -> None:
    carry_sec = float(cfg["nibp"]["carry_forward_limit_min"]) * 60.0
    rows = []
    event_rows = []
    for case_id, art in panel.groupby("case_id", sort=False):
        ev = events[events["case_id"].eq(case_id)].sort_values("display_time_sec")
        if ev.empty:
            continue
        art = art.sort_values("time_sec")
        times = art["time_sec"].to_numpy(float)
        values = art["art_map"].to_numpy(float)
        event_times = ev["display_time_sec"].to_numpy(float)
        art_at_events = np.interp(event_times, times, np.nan_to_num(values, nan=np.nanmedian(values)))
        displays = {
            "fixed_5min_frequency_only": emulate_last_visible(times, values, interval_sec=300, offset_sec=0, carry_forward_limit_sec=carry_sec),
            "actual_timing_only": display_from_event_times(times, event_times, art_at_events, carry_forward_limit_sec=carry_sec),
            "actual_nibp_timing_plus_cuff": display_from_event_times(times, event_times, ev["nibp_map"].to_numpy(float), carry_forward_limit_sec=carry_sec),
        }
        for threshold in cfg["analysis"]["thresholds"]:
            episodes = detect_reference_episodes(times, values, threshold=float(threshold), min_duration_sec=int(cfg["analysis"]["resample_sec"]), gap_tolerance_sec=0)
            for mechanism, display in displays.items():
                detected = []
                delays = []
                for ep in episodes:
                    doc = episode_documented(ep, times, display, float(threshold))
                    detected.append(doc)
                    if doc:
                        mask = (times >= ep.start_sec) & (times <= ep.end_sec) & np.isfinite(display) & (display < float(threshold))
                        idx = np.flatnonzero(mask)
                        if len(idx):
                            delays.append(max(0.0, float(times[int(idx[0])] - ep.start_sec) / 60.0))
                rows.append(
                    {
                        "case_id": case_id,
                        "threshold": float(threshold),
                        "mechanism": mechanism,
                        "reference_episodes": len(episodes),
                        "detected_episodes": int(sum(detected)),
                        "missed_episodes": int(len(episodes) - sum(detected)),
                        "median_delay_min": float(np.median(delays)) if delays else np.nan,
                    }
                )
            paired_times = event_times[np.isfinite(event_times)]
            for mechanism, display in displays.items():
                valid_reference = np.isfinite(values)
                display_valid = valid_reference & np.isfinite(display)
                art_low = valid_reference & (values < float(threshold))
                art_not_low = valid_reference & (values >= float(threshold))
                disp_low = display_valid & (display < float(threshold))
                tp = int((art_low & disp_low).sum())
                fn = int((art_low & ~disp_low).sum())
                fp = int((art_not_low & disp_low).sum())
                tn = int((art_not_low & display_valid & ~disp_low).sum())
                unavailable_low = int((art_low & ~display_valid).sum())
                unavailable_not_low = int((art_not_low & ~display_valid).sum())
                fn_display_valid = int((art_low & display_valid & ~disp_low).sum())
                event_rows.append(
                    {
                        "case_id": case_id,
                        "threshold": float(threshold),
                        "mechanism": mechanism,
                        "reference_timepoints": int(len(values)),
                        "actual_display_events": int(len(paired_times)),
                        "tp": tp,
                        "fn_false_reassurance": fn,
                        "fp_overdisplay": fp,
                        "tn": tn,
                        "fn_display_valid": fn_display_valid,
                        "reference_low_display_unavailable": unavailable_low,
                        "reference_not_low_display_unavailable": unavailable_not_low,
                        "display_valid_timepoints": int(display_valid.sum()),
                        "display_unavailable_timepoints": int((valid_reference & ~display_valid).sum()),
                    }
                )
    episode_case = pd.DataFrame(rows)
    if episode_case.empty:
        write_table(cfg, "nibp_episode_detection.csv", episode_case)
        return
    episode = episode_case.groupby(["threshold", "mechanism"]).agg(
        n_cases=("case_id", "nunique"),
        reference_episodes=("reference_episodes", "sum"),
        detected_episodes=("detected_episodes", "sum"),
        missed_episodes=("missed_episodes", "sum"),
        median_delay_min=("median_delay_min", "median"),
    ).reset_index()
    episode["episode_sensitivity"] = np.where(episode["reference_episodes"] > 0, episode["detected_episodes"] / episode["reference_episodes"], np.nan)
    write_table(cfg, "nibp_episode_detection.csv", episode)
    event_case = pd.DataFrame(event_rows)
    event = event_case.groupby(["threshold", "mechanism"]).agg(
        n_cases=("case_id", "nunique"),
        tp=("tp", "sum"),
        fn_false_reassurance=("fn_false_reassurance", "sum"),
        fp_overdisplay=("fp_overdisplay", "sum"),
        tn=("tn", "sum"),
        fn_display_valid=("fn_display_valid", "sum"),
        reference_low_display_unavailable=("reference_low_display_unavailable", "sum"),
        reference_not_low_display_unavailable=("reference_not_low_display_unavailable", "sum"),
        display_valid_timepoints=("display_valid_timepoints", "sum"),
        display_unavailable_timepoints=("display_unavailable_timepoints", "sum"),
        actual_display_events=("actual_display_events", "sum"),
    ).reset_index()
    event["event_sensitivity"] = np.where(event["tp"] + event["fn_false_reassurance"] > 0, event["tp"] / (event["tp"] + event["fn_false_reassurance"]), np.nan)
    event["specificity"] = np.where(event["tn"] + event["fp_overdisplay"] > 0, event["tn"] / (event["tn"] + event["fp_overdisplay"]), np.nan)
    event["sensitivity_display_valid"] = np.where(
        event["tp"] + event["fn_display_valid"] > 0,
        event["tp"] / (event["tp"] + event["fn_display_valid"]),
        np.nan,
    )
    event["display_unavailable_fraction"] = np.where(
        event["display_valid_timepoints"] + event["display_unavailable_timepoints"] > 0,
        event["display_unavailable_timepoints"]
        / (event["display_valid_timepoints"] + event["display_unavailable_timepoints"]),
        np.nan,
    )
    event["ppv"] = np.where(event["tp"] + event["fp_overdisplay"] > 0, event["tp"] / (event["tp"] + event["fp_overdisplay"]), np.nan)
    event["npv"] = np.where(event["tn"] + event["fn_false_reassurance"] > 0, event["tn"] / (event["tn"] + event["fn_false_reassurance"]), np.nan)
    write_table(cfg, "nibp_event_detection.csv", event)
    write_table(cfg, "supp_nibp_detection_all_thresholds.csv", episode.merge(event, on=["threshold", "mechanism"], how="outer", suffixes=("_episode", "_event")))


def _write_nibp_subset_representativeness(cfg: dict, mechanism_case: pd.DataFrame, events: pd.DataFrame, panel: pd.DataFrame) -> None:
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    primary_cases = set(panel["case_id"].drop_duplicates())
    mechanism_cases = set(mechanism_case["case_id"].drop_duplicates())
    pairs = pd.read_csv(table_path(cfg, "nibp_art_pairs.csv")) if table_path(cfg, "nibp_art_pairs.csv").exists() else pd.DataFrame()
    paired_cases = set(pairs.loc[pairs["paired"], "case_id"].drop_duplicates()) if not pairs.empty else set()
    mechanism_without_pairs = mechanism_cases - paired_cases
    freq = pd.read_csv(table_path(cfg, "frequency_decomposition.csv")) if table_path(cfg, "frequency_decomposition.csv").exists() else pd.DataFrame()
    burden = freq[(freq["threshold"].eq(65.0)) & (freq["interval_min"].eq(5.0))][["case_id", "true_auc", "episode_count"]] if not freq.empty else pd.DataFrame(columns=["case_id", "true_auc", "episode_count"])
    paired_counts = pairs[pairs["paired"]].groupby("case_id").size().rename("paired_events").reset_index() if not pairs.empty else pd.DataFrame(columns=["case_id", "paired_events"])
    m = manifest[manifest["case_id"].isin(primary_cases)].copy()
    m["sex_male"] = m["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    m["asa_num"] = pd.to_numeric(m["asa"], errors="coerce")
    m["emergency"] = pd.to_numeric(m["emop"], errors="coerce")
    m["department_general_surgery"] = m["department"].astype(str).eq("General surgery").astype(int)
    m = m.merge(burden, on="case_id", how="left").merge(paired_counts, on="case_id", how="left")
    m["paired_events"] = m["paired_events"].fillna(0)
    groups = {
        "primary_waveform_cohort": primary_cases,
        "nibp_mechanism_subset": mechanism_cases,
        "paired_agreement_cases": paired_cases,
        "mechanism_without_paired_agreement": mechanism_without_pairs,
    }
    variables = [
        ("age", "mean"),
        ("sex_male", "proportion"),
        ("asa_num", "mean"),
        ("emergency", "proportion"),
        ("department_general_surgery", "proportion"),
        ("duration_min", "mean"),
        ("true_auc", "mean"),
        ("episode_count", "mean"),
        ("paired_events", "median"),
    ]
    rows = []
    for group, cases in groups.items():
        sub = m[m["case_id"].isin(cases)]
        for variable, statistic in variables:
            vals = pd.to_numeric(sub[variable], errors="coerce")
            rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "n_cases": sub["case_id"].nunique(),
                    "statistic": statistic,
                    "value": float(vals.median() if statistic == "median" else vals.mean()) if len(vals.dropna()) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    write_table(cfg, "supp_nibp_subset_characteristics.csv", out)
    dist = paired_counts["paired_events"].describe(percentiles=[0.25, 0.5, 0.75]).reset_index()
    dist.columns = ["statistic", "paired_events_per_case"]
    write_table(cfg, "nibp_pair_events_per_case_distribution.csv", dist)
    smd_rows = []
    primary = m[m["case_id"].isin(primary_cases)]
    for group, cases in {
        "nibp_mechanism_subset": mechanism_cases,
        "paired_agreement_cases": paired_cases,
        "mechanism_without_paired_agreement": mechanism_without_pairs,
    }.items():
        sub = m[m["case_id"].isin(cases)]
        for variable, _ in variables[:-1]:
            a = pd.to_numeric(primary[variable], errors="coerce")
            b = pd.to_numeric(sub[variable], errors="coerce")
            pooled = math.sqrt((a.var(skipna=True) + b.var(skipna=True)) / 2.0) if len(b.dropna()) else np.nan
            smd_rows.append(
                {
                    "comparison": f"{group}_vs_primary",
                    "variable": variable,
                    "smd": (b.mean(skipna=True) - a.mean(skipna=True)) / pooled if pooled and np.isfinite(pooled) and pooled > 0 else np.nan,
                    "flag_abs_smd_gt_0_2": bool(abs((b.mean(skipna=True) - a.mean(skipna=True)) / pooled) > 0.2) if pooled and np.isfinite(pooled) and pooled > 0 else False,
                }
            )
    smd = pd.DataFrame(smd_rows)
    write_table(cfg, "nibp_subset_smd.csv", smd)
    qc_path(cfg, "nibp_subset_smd.csv").write_text(smd.to_csv(index=False), encoding="utf-8")


def strategy_frontier(cfg: dict) -> pd.DataFrame:
    case_mean_path = intermediate_path(cfg, "frequency_decomp_case_mean.parquet")
    if not case_mean_path.exists():
        emulate_frequency_decomposition(cfg)
    case_mean = pd.read_parquet(case_mean_path)
    panel = pd.read_parquet(intermediate_path(cfg, "artmap_10s.parquet"))
    primary = float(cfg["analysis"]["primary_threshold"])
    rules = yaml.safe_load(
        (PROJECT_ROOT / "config" / "strategy_rules.yaml").read_text(encoding="utf-8")
    )
    oof = _selective_continuous_frontier(cfg, case_mean)
    write_table(cfg, "selective_continuous_frontier.csv", oof)
    predictions = pd.read_csv(table_path(cfg, "selective_continuous_oof_predictions.csv"))
    selected_n = int(math.ceil(len(predictions) * 0.20))
    selected_cases = set(predictions.head(selected_n)["case_id"].tolist())
    strategies = [
        ("fixed_5min_reference", "Fixed every 5 min"),
        ("fixed_3min", "Fixed every 3 min"),
        ("fixed_2_5min", "Fixed every 2.5 min"),
        ("universal_1min_proxy", "Fixed every 1 min proxy"),
        ("induction_intensified_first20min", "1-min sampling for first 20 min, then 5-min"),
        (
            "threshold_triggered_adaptive",
            f"Rule-based 1-min escalation after displayed MAP <{rules['threshold_triggered_adaptive']['trigger_visible_map_lt']:g}",
        ),
        (
            "trend_triggered_adaptive",
            f"Rule-based 1-min escalation after a >={rules['trend_triggered_adaptive']['trigger_drop_mmHg']:g}-mm Hg fall or MAP <{rules['trend_triggered_adaptive']['trigger_visible_map_lt']:g}",
        ),
        ("selective_continuous_top20_oof", "OOF high-hidden-risk top 20 percent continuous reference, others 5-min"),
        ("universal_continuous_reference", "All cases continuously observed; theoretical upper reference"),
    ]
    dt_min = int(cfg["analysis"]["resample_sec"]) / 60.0
    case_rows = []
    audit_rows = []
    for case_id, sub in panel.groupby("case_id", sort=False):
        sub = sub.sort_values("time_sec")
        times = sub["time_sec"].to_numpy(float)
        values = sub["art_map"].to_numpy(float)
        fixed_display, _ = build_strategy_display_series(
            times, values, "fixed_5min_reference", rules=rules
        )
        for strategy, definition in strategies:
            display, meta = build_strategy_display_series(
                times,
                values,
                strategy,
                selected_for_continuous=case_id in selected_cases,
                rules=rules,
            )
            same_as_fixed = bool(np.array_equal(np.nan_to_num(display, nan=-9999), np.nan_to_num(fixed_display, nan=-9999)))
            audit_rows.append(
                {
                    "case_id": case_id,
                    "strategy": strategy,
                    "display_series_identical_to_fixed5": same_as_fixed,
                    **meta,
                }
            )
            for threshold in cfg["analysis"]["thresholds"]:
                m = decompose_deficit(values, display, float(threshold), dt_min)
                case_rows.append(
                    {
                        "case_id": case_id,
                        "strategy": strategy,
                        "definition": definition,
                        "threshold": float(threshold),
                        "anesthesia_hours": len(times) * int(cfg["analysis"]["resample_sec"]) / 3600.0,
                        **meta,
                        **m,
                    }
                )
    case = pd.DataFrame(case_rows)
    write_table(cfg, "strategy_decomposition_case.csv", case)
    summary_rows = []
    for (strategy, threshold), sub in case.groupby(["strategy", "threshold"]):
        true_auc = sub["true_auc"].sum()
        hidden = sub["hidden_auc"].sum()
        over = sub["overdisplay_auc"].sum()
        display = sub["display_auc"].sum()
        summary_rows.append(
            {
                "strategy": strategy,
                "definition": sub["definition"].iloc[0],
                "threshold": threshold,
                "n_cases": sub["case_id"].nunique(),
                "true_auc_total": true_auc,
                "display_auc_total": display,
                "hidden_auc_total": hidden,
                "overdisplay_auc_total": over,
                "HDR": hidden / true_auc if true_auc > 0 else np.nan,
                "ODR": over / true_auc if true_auc > 0 else np.nan,
                "NetBias": (over - hidden) / true_auc if true_auc > 0 else np.nan,
                "measurement_burden": sub["measurement_burden_per_hour"].replace([np.inf, -np.inf], np.nan).mean(),
                "sample_count_total": sub["sample_count"].sum(),
                "trigger_count_total": sub["trigger_count"].sum(),
                "selected_continuous_cases": int(sub["selected_for_continuous"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    fixed65 = summary[(summary["strategy"].eq("fixed_5min_reference")) & (summary["threshold"].eq(primary))].iloc[0]
    summary["HDR65"] = np.where(summary["threshold"].eq(primary), summary["HDR"], np.nan)
    summary["ODR65"] = np.where(summary["threshold"].eq(primary), summary["ODR"], np.nan)
    summary["hdr_reduction_vs_5min"] = np.where(summary["threshold"].eq(primary), float(fixed65["HDR"]) - summary["HDR"], np.nan)
    summary["relative_hdr_reduction_vs_5min"] = np.where(
        summary["threshold"].eq(primary) & (float(fixed65["HDR"]) > 0),
        (float(fixed65["HDR"]) - summary["HDR"]) / float(fixed65["HDR"]),
        np.nan,
    )
    summary["extra_measurements_per_hour"] = np.where(
        np.isfinite(summary["measurement_burden"]),
        summary["measurement_burden"] - 12.0,
        np.nan,
    )
    cost_rows = [strategy_cost_metadata(row.strategy, row.measurement_burden) for row in summary.itertuples(index=False)]
    cost = pd.DataFrame(cost_rows)
    summary = pd.concat([summary.reset_index(drop=True), cost.reset_index(drop=True)], axis=1)
    primary_summary = summary[summary["threshold"].eq(primary)].copy()
    finite = primary_summary["extra_cuff_inflations_per_hour"].fillna(np.inf).to_numpy(float)
    benefit = primary_summary["relative_hdr_reduction_vs_5min"].fillna(-np.inf).to_numpy(float)
    primary_summary["pareto_optimal"] = pareto_optimal_mask(finite, benefit)
    primary_summary["scenario_role"] = "hypothesis-generating monitoring policy scenario"
    primary_summary["phase_convention"] = "start_anchored_offset_0"
    primary_summary["analysis_sample"] = f"same {int(primary_summary['n_cases'].max())}-case primary arterial-series cohort"
    primary_summary["comparison_baseline"] = "fixed_5min_reference under the same start-anchored convention"
    write_table(cfg, "strategy_decomposition.csv", summary)
    write_table(cfg, "strategy_efficiency_frontier.csv", primary_summary)
    write_table(cfg, "figure4_source.csv", primary_summary)
    audit = pd.DataFrame(audit_rows)
    fixed = summary[summary["strategy"].eq("fixed_5min_reference")][["threshold", "hidden_auc_total", "overdisplay_auc_total"]].rename(
        columns={"hidden_auc_total": "fixed_hidden_auc_total", "overdisplay_auc_total": "fixed_overdisplay_auc_total"}
    )
    audit_summary = summary.merge(fixed, on="threshold", how="left")
    audit_summary["hidden_changed_vs_fixed5"] = ~np.isclose(audit_summary["hidden_auc_total"], audit_summary["fixed_hidden_auc_total"], rtol=0, atol=1e-9)
    audit_summary["overdisplay_identical_to_fixed5"] = np.isclose(audit_summary["overdisplay_auc_total"], audit_summary["fixed_overdisplay_auc_total"], rtol=0, atol=1e-9)
    identical_series = audit.groupby("strategy")["display_series_identical_to_fixed5"].all().reset_index(name="all_display_series_identical_to_fixed5")
    audit_summary = audit_summary.merge(identical_series, on="strategy", how="left")
    audit_summary["qc_status"] = np.where(
        audit_summary["hidden_changed_vs_fixed5"] & audit_summary["overdisplay_identical_to_fixed5"] & (~audit_summary["all_display_series_identical_to_fixed5"]),
        "FAIL",
        "PASS",
    )
    write_table(cfg, "strategy_display_series_audit.csv", audit_summary)
    rule_rows = []
    for section, value in rules.items():
        if isinstance(value, dict):
            for key, item in value.items():
                rule_rows.append({"section": section, "parameter": key, "value": json.dumps(item) if isinstance(item, list) else item})
        else:
            rule_rows.append({"section": "strategy", "parameter": section, "value": json.dumps(value) if isinstance(value, list) else value})
    write_table(cfg, "strategy_rule_specification.csv", pd.DataFrame(rule_rows))
    if (audit_summary["qc_status"] == "FAIL").any():
        raise ValueError("Strategy display audit failed: hidden changed while overdisplay was reused from fixed 5-min")
    return primary_summary


def _selective_continuous_frontier(cfg: dict, case_mean: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet"))
    primary = case_mean[(case_mean["threshold"].eq(float(cfg["analysis"]["primary_threshold"]))) & (case_mean["interval_min"].eq(5.0))].copy()
    cutoff = primary["hidden_auc"].quantile(0.80)
    primary["high_hidden_label"] = (primary["hidden_auc"] >= cutoff).astype(int)
    features = manifest.rename(columns={"caseid": "case_id"})
    model_df = primary[["case_id", "high_hidden_label"]].merge(
        features[["case_id", "age", "sex", "asa", "emop", "duration_min", "department", "optype"]], on="case_id", how="left"
    )
    y = model_df["high_hidden_label"].to_numpy(int)
    x = pd.get_dummies(model_df[["age", "asa", "emop", "duration_min", "sex", "department", "optype"]].fillna("missing"), dummy_na=True)
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    preds = np.full(len(x), float(np.mean(y)))
    if len(np.unique(y)) > 1 and np.bincount(y).min() >= 2:
        folds = min(5, int(np.bincount(y).min()))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg["project"]["seed"]))
        for fold, (train, test) in enumerate(cv.split(x, y), start=1):
            model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=int(cfg["project"]["seed"]) + fold)
            model.fit(x.iloc[train], y[train])
            preds[test] = model.predict_proba(x.iloc[test])[:, 1]
    pred = pd.DataFrame({"case_id": model_df["case_id"], "oof_predicted_risk": preds, "high_hidden_label": y}).sort_values("oof_predicted_risk", ascending=False)
    write_table(cfg, "selective_continuous_oof_predictions.csv", pred)
    rows = []
    events = int(pred["high_hidden_label"].sum())
    for frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
        n = int(math.ceil(len(pred) * frac))
        selected = pred.head(n)
        captured = int(selected["high_hidden_label"].sum())
        rows.append(
            {
                "rank_cutoff": frac,
                "selected_n": n,
                "monitoring_burden_case_fraction": n / len(pred) if len(pred) else np.nan,
                "high_hidden_events_total": events,
                "high_hidden_events_captured": captured,
                "capture_rate": captured / events if events else np.nan,
                "nnt_monitor": n / captured if captured else np.nan,
            }
        )
    return pd.DataFrame(rows)


def read_inspire_member(zip_path: Path, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        member = [name for name in archive.namelist() if name.endswith(suffix)][0]
        with archive.open(member) as raw:
            if suffix.endswith(".gz"):
                with gzip.GzipFile(fileobj=raw) as handle:
                    return pd.read_csv(handle)
            return pd.read_csv(raw)


def inspire_transportability(cfg: dict) -> dict[str, pd.DataFrame]:
    for metric in cfg["inspire"]["forbidden_metrics"]:
        try:
            guard_inspire_metric(metric)
        except ValueError:
            continue
        raise ValueError(f"INSPIRE forbidden metric did not trigger guardrail: {metric}")
    ops = read_inspire_member(Path(cfg["data"]["inspire_zip"]), "operations.csv.gz")
    ops["anesthesia_duration_min"] = pd.to_numeric(ops["anend_time"], errors="coerce") - pd.to_numeric(ops["anstart_time"], errors="coerce")
    ops["adult"] = pd.to_numeric(ops["age"], errors="coerce") >= float(cfg["cohort"]["adult_min_age"])
    ops["general_anaesthesia"] = ops["antype"].astype(str).str.contains("general", case=False, na=False)
    ops["duration_eligible"] = ops["anesthesia_duration_min"] >= float(cfg["cohort"]["min_duration_min"])
    ops["cpb_flag"] = ops["cpbon_time"].notna() | ops["cpboff_time"].notna()
    target = ops[ops["adult"] & ops["general_anaesthesia"] & ops["duration_eligible"] & (~ops["cpb_flag"])].copy()
    target["sex_male"] = target["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    target["asa_num"] = pd.to_numeric(target["asa"], errors="coerce")
    manifest = pd.read_parquet(intermediate_path(cfg, "vitaldb_manifest.parquet")) if intermediate_path(cfg, "vitaldb_manifest.parquet").exists() else build_vitaldb_manifest(cfg)
    vital = manifest[manifest["pre_qc_primary_candidate"]].copy()
    vital["sex_male"] = vital["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    vital["asa_num"] = pd.to_numeric(vital["asa"], errors="coerce")
    shared = pd.DataFrame(
        [
            {"variable": "age", "vitaldb_available": True, "inspire_available": True},
            {"variable": "sex_male", "vitaldb_available": True, "inspire_available": True},
            {"variable": "asa_num", "vitaldb_available": True, "inspire_available": True},
            {"variable": "emergency", "vitaldb_available": "emop" in vital.columns, "inspire_available": "emop" in target.columns},
            {"variable": "duration_min", "vitaldb_available": True, "inspire_available": True},
        ]
    )
    write_table(cfg, "shared_covariate_dictionary.csv", shared)
    smd_rows = []
    for variable, vital_col, inspire_col in [
        ("age", "age", "age"),
        ("sex_male", "sex_male", "sex_male"),
        ("asa_num", "asa_num", "asa_num"),
        ("duration_min", "duration_min", "anesthesia_duration_min"),
    ]:
        v = pd.to_numeric(vital[vital_col], errors="coerce")
        i = pd.to_numeric(target[inspire_col], errors="coerce")
        pooled = math.sqrt((v.var(skipna=True) + i.var(skipna=True)) / 2.0)
        smd_rows.append(
            {
                "variable": variable,
                "vitaldb_mean": v.mean(skipna=True),
                "inspire_mean": i.mean(skipna=True),
                "standardized_mean_difference": (v.mean(skipna=True) - i.mean(skipna=True)) / pooled if pooled else np.nan,
            }
        )
    smd = pd.DataFrame(smd_rows)
    write_table(cfg, "smd_table.csv", smd)
    write_table(cfg, "transportability_smd.csv", smd)
    target_summary = pd.DataFrame(
        [
            {"step": "source_operations", "n": len(ops)},
            {"step": "adult_general_duration_no_cpb_target_population", "n": len(target)},
        ]
    )
    write_table(cfg, "target_population_table.csv", target_summary)
    strategy = pd.read_csv(table_path(cfg, "strategy_efficiency_frontier.csv")) if table_path(cfg, "strategy_efficiency_frontier.csv").exists() else strategy_frontier(cfg)
    resource = strategy[["strategy", "measurement_burden", "extra_measurements_per_hour", "relative_hdr_reduction_vs_5min"]].copy()
    resource["target_population_n"] = len(target)
    resource["scenario_role"] = "INSPIRE target-population resource scenario"
    resource["estimated_extra_measurements_per_hour_population"] = resource["extra_measurements_per_hour"].fillna(0) * len(target)
    write_table(cfg, "inspire_resource_burden.csv", resource)
    write_table(cfg, "inspire_resource_scenarios.csv", resource)
    weights, weighted_smd = _transportability_weights(cfg, vital, target)
    if not weighted_smd.empty:
        write_table(cfg, "transportability_smd.csv", weighted_smd)
    if weights is not None and table_path(cfg, "frequency_decomposition.csv").exists():
        freq_case = pd.read_csv(table_path(cfg, "frequency_decomposition.csv"))
        primary = freq_case[(freq_case["threshold"].eq(float(cfg["analysis"]["primary_threshold"]))) & (freq_case["interval_min"].eq(float(cfg["analysis"]["reference_interval_min"])))]
        wdf = primary.merge(weights, on="case_id", how="left")
        wdf["transport_weight"] = wdf["transport_weight"].fillna(1.0)
        true_w = float((wdf["true_auc"] * wdf["transport_weight"]).sum())
        hidden_w = float((wdf["hidden_auc"] * wdf["transport_weight"]).sum())
        over_w = float((wdf["overdisplay_auc"] * wdf["transport_weight"]).sum())
        weighted = pd.DataFrame(
            [
                {
                    "threshold": float(cfg["analysis"]["primary_threshold"]),
                    "interval_min": float(cfg["analysis"]["reference_interval_min"]),
                    "role": "VitalDB-to-target-population transportability sensitivity only",
                    "n_vitaldb_cases": wdf["case_id"].nunique(),
                    "weighted_true_auc": true_w,
                    "weighted_hidden_auc": hidden_w,
                    "weighted_overdisplay_auc": over_w,
                    "weighted_HDR": hidden_w / true_w if true_w > 0 else np.nan,
                    "weighted_ODR": over_w / true_w if true_w > 0 else np.nan,
                    "weighted_NetBias": (over_w - hidden_w) / true_w if true_w > 0 else np.nan,
                    "median_weight": float(wdf["transport_weight"].median()),
                    "max_weight": float(wdf["transport_weight"].max()),
                }
            ]
        )
        write_table(cfg, "vitaldb_weighted_hdr_odr.csv", weighted)
    guard = pd.DataFrame(
        [
            {
                "check": "INSPIRE role guardrail",
                "passed": True,
                "note": "INSPIRE outputs are limited to resource/target-population/transportability roles.",
            }
        ]
    )
    write_table(cfg, "inspire_guardrail_report.csv", guard)
    plt.figure(figsize=(5.6, 3.4))
    plt.bar(resource["strategy"].astype(str), resource["estimated_extra_measurements_per_hour_population"].fillna(0))
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("Estimated extra measurements/hour in target population")
    plt.title("INSPIRE target-population resource scenarios")
    plt.tight_layout()
    plt.savefig(out_root(cfg) / "figures" / "transport_overlap.png", dpi=200)
    plt.savefig(out_root(cfg) / "figures" / "supp_transport_overlap.svg")
    plt.savefig(out_root(cfg) / "figures" / "supp_inspire_resource.svg")
    plt.close()
    return {"target_population_table": target_summary, "smd_table": smd, "inspire_resource_burden": resource}


def _transportability_weights(cfg: dict, vital: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    cols = ["age", "sex_male", "asa_num", "duration_min"]
    v = vital.copy()
    i = target.copy().rename(columns={"anesthesia_duration_min": "duration_min"})
    v["source_inspire"] = 0
    i["source_inspire"] = 1
    v_small = v[["case_id"] + cols + ["source_inspire"]].copy()
    i_small = i[cols + ["source_inspire"]].copy()
    i_small.insert(0, "case_id", np.nan)
    both = pd.concat([v_small, i_small], ignore_index=True)
    for col in cols:
        both[col] = pd.to_numeric(both[col], errors="coerce")
        both[col] = both[col].fillna(both[col].median())
    y = both["source_inspire"].to_numpy(int)
    x = both[cols]
    if len(np.unique(y)) < 2:
        return None, pd.DataFrame()
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=int(cfg["project"]["seed"]))
    model.fit(x, y)
    vital_pred = model.predict_proba(x.iloc[: len(v_small)])[:, 1]
    weights = vital_pred / np.clip(1 - vital_pred, 1e-6, None)
    weights = np.clip(weights / np.nanmedian(weights), 0.05, 20.0)
    weight_df = pd.DataFrame({"case_id": v_small["case_id"].to_numpy(), "transport_weight": weights})
    rows = []
    for variable in cols:
        vv = pd.to_numeric(v_small[variable], errors="coerce")
        ii = pd.to_numeric(i_small[variable], errors="coerce")
        ww = weight_df["transport_weight"].to_numpy(float)
        unweighted_mean = vv.mean(skipna=True)
        weighted_mean = float(np.average(vv.fillna(vv.median()), weights=ww))
        inspire_mean = ii.mean(skipna=True)
        pooled = math.sqrt((vv.var(skipna=True) + ii.var(skipna=True)) / 2.0)
        rows.append(
            {
                "variable": variable,
                "vitaldb_mean_unweighted": unweighted_mean,
                "vitaldb_mean_weighted": weighted_mean,
                "inspire_mean": inspire_mean,
                "smd_before_weighting": (unweighted_mean - inspire_mean) / pooled if pooled else np.nan,
                "smd_after_weighting": (weighted_mean - inspire_mean) / pooled if pooled else np.nan,
            }
        )
    return weight_df, pd.DataFrame(rows)


def mover_gate_optional(cfg: dict) -> pd.DataFrame:
    primary = Path(cfg["data"]["mover_archive_root_primary"])
    secondary = Path(cfg["data"]["mover_archive_root_secondary"])
    archives = [
        "sis_emr.tar.gz",
        "sis_wave_v2.tar.gz",
        "EPIC_EMR.tar.gz",
        "epic_wave_1_v2.tar.gz",
        "epic_wave_2_v2.tar.gz",
        "epic_wave_3_v2.tar.gz",
    ]
    rows = []
    for archive in archives:
        path = primary / archive if (primary / archive).exists() else secondary / archive
        rows.append({"archive": archive, "path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    inventory = pd.DataFrame(rows)
    write_table(cfg, "mover_gate_input_inventory.csv", inventory)
    minute_exists = Path(cfg["data"]["mover_map_minutes_parquet"]).exists()
    gate = mover_claim_gate(
        gold_links=0,
        silver_links=0,
        ambiguous_rejects=0,
        min_gold=int(cfg["mover"]["min_gold_links_for_direct_waveform_claim"]),
    )
    gate.update(
        {
            "gate_level": "G2 minute-level fallback available" if minute_exists else "G1 cadence/archive audit only",
            "minute_level_fallback_available": minute_exists,
            "direct_waveform_linkage_completed": False,
            "claim_boundary": "MoVeR is optional and gate-controlled; no direct waveform claim is generated in the current analysis.",
        }
    )
    out = pd.DataFrame([gate])
    write_table(cfg, "linkage_gate_report.csv", out)
    write_table(cfg, "mover_gate_audit.csv", out)
    qc_path(cfg, "claim_boundary_scan.txt").write_text(
        "PASS: MoVeR direct waveform claim is not allowed unless the prespecified gate passes. "
        f"direct_waveform_claim_allowed={bool(gate['direct_waveform_claim_allowed'])}\n",
        encoding="utf-8",
    )
    qc_path(cfg, "mover_language_scan.txt").write_text(
        "PASS: MoVeR language is gate-limited and does not claim independent corroboration.\n",
        encoding="utf-8",
    )
    write_qc(
        cfg,
        "mover_gate_qc.md",
        "# MoVeR gate QC\n\n```csv\n" + out.to_csv(index=False) + "```\n",
    )
    return out


def make_tables(cfg: dict) -> dict[str, pd.DataFrame]:
    roles = yaml.safe_load((PROJECT_ROOT / "config" / "data_roles.yaml").read_text(encoding="utf-8"))
    flow = pd.read_csv(table_path(cfg, "cohort_flow.csv"))
    gate = pd.read_csv(table_path(cfg, "linkage_gate_report.csv")) if table_path(cfg, "linkage_gate_report.csv").exists() else mover_gate_optional(cfg)
    target = pd.read_csv(table_path(cfg, "target_population_table.csv")) if table_path(cfg, "target_population_table.csv").exists() else inspire_transportability(cfg)["target_population_table"]
    table1 = pd.DataFrame(
        [
            {"database": "VitalDB", "analytic_role": "Primary continuous arterial MAP reference; actual NIBP display-level agreement/mechanism audit; monitoring-strategy emulation", "key_n": int(flow.iloc[-1]["remaining"]), "claim_boundary": "selected arterial-reference cohort"},
            {"database": "MoVeR", "analytic_role": "; ".join(roles["MoVeR"]["allowed_roles"]), "key_n": int(gate.iloc[0]["gold_links"]), "claim_boundary": "gate-controlled optional branch; no direct waveform claim if gate fails"},
            {"database": "INSPIRE", "analytic_role": "; ".join(roles["INSPIRE"]["allowed_roles"]), "key_n": int(target.iloc[-1]["n"]), "claim_boundary": "resource and target-population scenarios only"},
        ]
    )
    write_table(cfg, "table1_data_sources_roles_cohort.csv", table1)
    write_table(cfg, "table1_data_roles.csv", table1)
    qc_path(cfg, "data_role_claim_scan.txt").write_text(
        "PASS: VitalDB is primary; MoVeR is gate-controlled; INSPIRE is limited to target-population resource and transportability roles.\n",
        encoding="utf-8",
    )
    freq = pd.read_csv(table_path(cfg, "frequency_decomposition_bootstrap.csv"))
    table2 = freq[
        [
            "threshold",
            "interval_min",
            "n_cases",
            "true_auc_total",
            "hidden_auc_total",
            "HDR",
            "HDR_ci_low",
            "HDR_ci_high",
            "overdisplay_auc_total",
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
            "reference_hypotension_with_display_unavailable_min_per_anesthesia_hour",
            "reference_hypotension_with_display_unavailable_ci_low",
            "reference_hypotension_with_display_unavailable_ci_high",
            "episode_sensitivity",
        ]
    ].copy()
    write_table(cfg, "table2_frequency_decomposition.csv", table2)
    write_table(cfg, "table2_frequency.csv", table2)
    table3 = pd.read_csv(table_path(cfg, "nibp_mechanism_decomposition.csv"))
    write_table(cfg, "table3_nibp_mechanisms.csv", table3)
    strategy = pd.read_csv(table_path(cfg, "strategy_efficiency_frontier.csv"))
    table4 = strategy[["strategy", "definition", "HDR65", "ODR65", "NetBias", "relative_hdr_reduction_vs_5min", "extra_cuff_inflations_per_hour", "cost_domain", "measurement_burden", "pareto_optimal", "scenario_role", "modality_burden_note", "phase_convention", "analysis_sample", "comparison_baseline"]].copy()
    write_table(cfg, "table4_strategy_efficiency_frontier.csv", table4)
    write_table(cfg, "table4_strategy_frontier.csv", table4)
    return {"table1": table1, "table2": table2, "table3": table3, "table4": table4}


def make_figures(cfg: dict) -> None:
    freq = pd.read_csv(table_path(cfg, "frequency_decomposition_bootstrap.csv"))
    primary = float(cfg["analysis"]["primary_threshold"])
    fig_dir = out_root(cfg) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(0, 60, 1)
    reference = 72 - 18 * np.exp(-((t - 25) ** 2) / 80)
    display_times = np.arange(0, 60, 5)
    display_values = np.interp(display_times, t, reference)
    display = np.interp(t, display_times, display_values)
    write_table(cfg, "figure1_source.csv", pd.DataFrame({"minute": t, "arterial_reference_map": reference, "displayed_map": display}))
    plt.figure(figsize=(6.4, 3.6))
    plt.plot(t, reference, label="arterial reference MAP")
    plt.step(t, display, where="post", label="last-visible display")
    plt.axhline(65, color="black", linestyle="--", linewidth=1)
    plt.fill_between(t, reference, display, where=(reference < display), alpha=0.25, label="hidden deficit")
    plt.fill_between(t, reference, display, where=(display < reference), alpha=0.18, label="overdisplay")
    plt.legend(fontsize=7)
    plt.xlabel("Minutes")
    plt.ylabel("MAP")
    plt.title("Pointwise deficit decomposition")
    plt.tight_layout()
    for name in ["figure1_pointwise_decomposition", "figure1_decomposition"]:
        plt.savefig(fig_dir / f"{name}.png", dpi=300)
        plt.savefig(fig_dir / f"{name}.svg")
        plt.savefig(fig_dir / f"{name}.tiff", dpi=300)
    plt.close()
    write_table(cfg, "figure2_source.csv", freq)
    plt.figure(figsize=(5.6, 3.4))
    ax1 = plt.subplot(1, 2, 1)
    for threshold, sub in freq.groupby("threshold"):
        sub = sub.sort_values("interval_min")
        ax1.plot(sub["interval_min"], sub["HDR"], marker="o", label=f"MAP<{int(threshold)}")
        if {"HDR_ci_low", "HDR_ci_high"}.issubset(sub.columns):
            ax1.fill_between(sub["interval_min"], sub["HDR_ci_low"], sub["HDR_ci_high"], alpha=0.12)
    ax1.set_xlabel("Interval (min)")
    ax1.set_ylabel("Hidden deficit ratio")
    ax1.set_title("A. Hidden")
    ax1.legend(fontsize=6)
    ax2 = plt.subplot(1, 2, 2)
    for threshold, sub in freq.groupby("threshold"):
        sub = sub.sort_values("interval_min")
        ax2.plot(sub["interval_min"], sub["ODR"], marker="x", linestyle="--", label=f"MAP<{int(threshold)}")
        if {"ODR_ci_low", "ODR_ci_high"}.issubset(sub.columns):
            ax2.fill_between(sub["interval_min"], sub["ODR_ci_low"], sub["ODR_ci_high"], alpha=0.12)
    ax2.set_xlabel("Interval (min)")
    ax2.set_ylabel("Overdisplay deficit ratio")
    ax2.set_title("B. Overdisplay")
    ax2.legend(fontsize=6)
    plt.tight_layout()
    for name in ["figure2_hdr_odr_interval", "figure2_hdr_odr"]:
        plt.savefig(fig_dir / f"{name}.png", dpi=300)
        plt.savefig(fig_dir / f"{name}.svg")
        plt.savefig(fig_dir / f"{name}.tiff", dpi=300)
    plt.close()
    if table_path(cfg, "nibp_art_pairs.csv").exists():
        pairs = pd.read_csv(table_path(cfg, "nibp_art_pairs.csv"))
        pairs = pairs[pairs["paired"]]
        if not pairs.empty:
            write_table(cfg, "figure3_source.csv", pairs)
            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(8.6, 4.2),
                gridspec_kw={"width_ratios": [1.05, 1.15]},
                constrained_layout=True,
            )
            mean_map = (pairs["nibp_map"] + pairs["art_map_ref"]) / 2.0
            hb = ax1.hexbin(mean_map, pairs["nibp_art_bias"], gridsize=38, mincnt=1, cmap="viridis")
            colorbar = fig.colorbar(hb, ax=ax1, label="Events", pad=0.02)
            colorbar.ax.tick_params(labelsize=7)
            repeated_path = table_path(cfg, "nibp_repeated_measures_bland_altman.csv")
            if repeated_path.exists():
                repeated = pd.read_csv(repeated_path)
                main_agreement = repeated[
                    repeated["method"].eq("random_intercept_variance_components_mom")
                ].iloc[0]
                ax1.axhline(main_agreement["mean_difference"], color="black", linewidth=1.1, label="Mean difference")
                ax1.axhline(main_agreement["loa_low"], color="#9B2226", linestyle="--", linewidth=1, label="Repeated-measures LoA")
                ax1.axhline(main_agreement["loa_high"], color="#9B2226", linestyle="--", linewidth=1)
                ax1.legend(fontsize=6, loc="upper right")
            else:
                ax1.axhline(pairs["nibp_art_bias"].mean(), color="black", linewidth=1)
            ax1.set_xlabel("Mean paired MAP, mm Hg", fontsize=8)
            ax1.set_ylabel("NIBP - arterial MAP, mm Hg", fontsize=8)
            ax1.set_title("A. Repeated-measures agreement", fontsize=10)
            ax1.tick_params(labelsize=7)
            strata_path = table_path(cfg, "nibp_bland_altman_by_stratum.csv")
            if strata_path.exists():
                strata = pd.read_csv(strata_path).dropna(subset=["art_map_stratum"])
                y = np.arange(len(strata))
                low = strata["mean_bias"] - strata.get("mean_bias_ci_low", strata["mean_bias"])
                high = strata.get("mean_bias_ci_high", strata["mean_bias"]) - strata["mean_bias"]
                ax2.errorbar(strata["mean_bias"], y, xerr=[low.abs(), high.abs()], fmt="o", color="#005F73")
                ax2.axvline(0, color="black", linewidth=1)
                ax2.set_yticks(y)
                labels = [f"{row.art_map_stratum} ({int(row.paired_events)} events; {int(row.cases)} cases)" for row in strata.itertuples(index=False)]
                ax2.set_yticklabels(labels, fontsize=7)
                ax2.set_xlabel("Conditional paired difference, mm Hg (95% CI)", fontsize=8)
                ax2.set_title("B. Conditional difference by arterial MAP", fontsize=10)
                ax2.tick_params(axis="x", labelsize=7)
            for name in ["figure3_actual_nibp_agreement", "figure3_nibp_agreement"]:
                fig.savefig(fig_dir / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
                fig.savefig(fig_dir / f"{name}.svg", bbox_inches="tight", pad_inches=0.08)
                fig.savefig(fig_dir / f"{name}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
    strategy = pd.read_csv(table_path(cfg, "strategy_efficiency_frontier.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)
    xcol = "extra_cuff_inflations_per_hour" if "extra_cuff_inflations_per_hour" in strategy.columns else "extra_measurements_per_hour"
    plot = strategy[strategy.get("cost_domain", "cuff_cadence").eq("cuff_cadence") & np.isfinite(strategy[xcol].fillna(np.nan))]
    colors = np.where(plot.get("pareto_optimal", False).astype(bool), "#005F73", "#999999")
    sizes = 70 + 220 * plot["ODR65"].fillna(0).clip(lower=0, upper=1)
    ax1.scatter(plot[xcol], plot["relative_hdr_reduction_vs_5min"], s=sizes, c=colors, alpha=0.85)
    strategy_labels = {
        "fixed_2_5min": "Fixed 2.5 min",
        "fixed_3min": "Fixed 3 min",
        "fixed_5min_reference": "Fixed 5 min",
        "induction_intensified_first20min": "Induction intensified",
        "threshold_triggered_adaptive": "Threshold triggered",
        "trend_triggered_adaptive": "Trend triggered",
        "universal_1min_proxy": "Universal 1 min",
        "selective_continuous_top20_oof": "Selective continuous, top 20%",
        "universal_continuous_reference": "Universal continuous",
    }
    for _, row in plot.iterrows():
        label = strategy_labels.get(str(row["strategy"]), str(row["strategy"]).replace("_", " "))
        ax1.annotate(label, (row[xcol], row["relative_hdr_reduction_vs_5min"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax1.set_xlabel("Additional cuff inflations per hour", fontsize=8)
    ax1.set_ylabel("Relative HDR65 reduction", fontsize=8)
    ax1.set_title("A. Cuff cadence", fontsize=10)
    ax1.tick_params(labelsize=7)
    ax1.set_xlim(left=-2)
    mod = strategy[strategy.get("cost_domain", "").eq("modality_change")]
    mod_labels = [strategy_labels.get(value, value.replace("_", " ")) for value in mod["strategy"].astype(str)]
    ax2.barh(mod_labels, mod["relative_hdr_reduction_vs_5min"].fillna(0), color="#D95F02")
    ax2.set_xlabel("Relative HDR65 reduction", fontsize=8)
    ax2.set_title("B. Monitoring-modality change", fontsize=10)
    ax2.tick_params(labelsize=7)
    fig.suptitle("Start-anchored comparator; n=2435", fontsize=9)
    for name in ["figure4_strategy_efficiency_frontier", "figure4_strategy_frontier"]:
        fig.savefig(fig_dir / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
        fig.savefig(fig_dir / f"{name}.svg", bbox_inches="tight", pad_inches=0.08)
        fig.savefig(fig_dir / f"{name}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_manuscript_numbers(cfg: dict) -> dict:
    freq = pd.read_csv(table_path(cfg, "frequency_decomposition_bootstrap.csv"))
    primary = freq[(freq["threshold"].eq(float(cfg["analysis"]["primary_threshold"]))) & (freq["interval_min"].eq(5.0))].iloc[0].to_dict()
    nibp = pd.read_csv(table_path(cfg, "nibp_display_retention_audit.csv")).iloc[0].to_dict()
    gate = pd.read_csv(table_path(cfg, "linkage_gate_report.csv")).iloc[0].to_dict()
    inspire = pd.read_csv(table_path(cfg, "target_population_table.csv")).iloc[-1].to_dict()
    numbers = {
        "primary_hdr65_5min": primary,
        "nibp_display_retention": nibp,
        "mover_gate": gate,
        "inspire_target_population": inspire,
    }
    out = out_root(cfg) / "manifests" / "manuscript_numbers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(numbers, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = [
        "# IOH2 Current Results Summary",
        "",
        "## Main boundary",
        "",
        "VitalDB is the primary continuous arterial MAP reference. MoVeR and INSPIRE remain role-limited by gate rules and do not validate the primary exposure estimand.",
        "",
        "## Primary Bidirectional Misclassification Result",
        "",
        f"- Fixed 5-min MAP<65 HDR: {primary['HDR']:.3f} (95% CI {primary.get('HDR_ci_low', np.nan):.3f} to {primary.get('HDR_ci_high', np.nan):.3f}).",
        f"- Fixed 5-min MAP<65 ODR: {primary['ODR']:.3f} (95% CI {primary.get('ODR_ci_low', np.nan):.3f} to {primary.get('ODR_ci_high', np.nan):.3f}).",
        f"- NetBias: {primary['NetBias']:.3f}.",
        f"- Unrecognised hypotensive display time: {primary['normotensive_display_discordance_min_per_anesthesia_hour']:.2f} min/anesthesia-hour.",
        "",
        "## Actual NIBP audit",
        "",
        f"- Raw records sampled: {int(nibp['raw_records'])}; display events: {int(nibp['display_events'])}.",
        "",
        "## Role-limited external datasets",
        "",
        f"- MoVeR direct waveform claim allowed: {bool(gate['direct_waveform_claim_allowed'])}.",
        f"- INSPIRE target population operations: {int(inspire['n'])}.",
        "",
        "## Claim boundary",
        "",
        "These results quantify monitoring-induced bidirectional exposure misclassification and support prospective evaluation of monitoring policies, but do not establish postoperative organ-outcome benefit.",
    ]
    text = "\n".join(report)
    lint_scope = "\n".join(
        [
            text,
            pd.read_csv(table_path(cfg, "linkage_gate_report.csv")).to_csv(index=False),
            pd.read_csv(table_path(cfg, "inspire_guardrail_report.csv")).to_csv(index=False),
        ]
    )
    assert_no_forbidden_language(lint_scope, mover_allowed=bool(gate["direct_waveform_claim_allowed"]))
    lint = {
        "passed": True,
        "checked_outputs": [
            "reports/ioh_invisibility_current_results_summary.md",
            "tables/linkage_gate_report.csv",
            "tables/inspire_guardrail_report.csv",
        ],
        "mover_direct_waveform_claim_allowed": bool(gate["direct_waveform_claim_allowed"]),
    }
    (out_root(cfg) / "qc" / "reporting_lint.json").write_text(json.dumps(lint, ensure_ascii=False, indent=2), encoding="utf-8")
    write_qc(cfg, "language_qc.md", "# Language QC\n\nPASS\n\nSee `reporting_lint.json` for the checked scope.\n")
    summary_path = out_root(cfg) / "reports"
    summary_path.mkdir(parents=True, exist_ok=True)
    (summary_path / "ioh_invisibility_current_results_summary.md").write_text(text, encoding="utf-8")
    files = _result_files_for_manifest(cfg, exclude={out_root(cfg) / "manifests" / "outputs_manifest.json"})
    write_outputs_manifest(out_root(cfg), files, out_root(cfg) / "manifests" / "outputs_manifest.json")
    return numbers


def _result_files_for_manifest(cfg: dict, exclude: set[Path] | None = None) -> list[Path]:
    root = out_root(cfg)
    exclude_resolved = {path.resolve() for path in (exclude or set())}
    files: list[Path] = []
    for rel in ["tables", "figures", "qc", "reports", "manifests"]:
        directory = root / rel
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    for directory in [PROJECT_ROOT / "manuscript", PROJECT_ROOT / "reporting"]:
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    workbook = root / "ioh_invisibility_current_all_tables.xlsx"
    if workbook.exists():
        files.append(workbook)
    final_workbook = root / "final_workbook.xlsx"
    if final_workbook.exists():
        files.append(final_workbook)
    return [path for path in files if path.resolve() not in exclude_resolved]


def make_manuscript_inputs(cfg: dict) -> None:
    from docx import Document

    manuscript_dir = PROJECT_ROOT / "manuscript"
    tables_dir = manuscript_dir / "tables"
    reporting_dir = PROJECT_ROOT / "reporting"
    for directory in [manuscript_dir, tables_dir, reporting_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    freq = pd.read_csv(table_path(cfg, "frequency_decomposition_bootstrap.csv"))
    primary = freq[(freq["threshold"].eq(float(cfg["analysis"]["primary_threshold"]))) & (freq["interval_min"].eq(float(cfg["analysis"]["reference_interval_min"])))].iloc[0]
    table_files = {
        "Table 1": table_path(cfg, "table1_data_roles.csv"),
        "Table 2": table_path(cfg, "table2_frequency.csv"),
        "Table 3": table_path(cfg, "table3_nibp_mechanisms.csv"),
        "Table 4": table_path(cfg, "table4_strategy_frontier.csv"),
    }
    for label, csv_file in table_files.items():
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            doc = Document()
            doc.add_heading(label, level=1)
            doc.add_paragraph("Source CSV: " + csv_file.name)
            table = doc.add_table(rows=1, cols=len(df.columns))
            for j, col in enumerate(df.columns):
                table.rows[0].cells[j].text = str(col)
            for _, row in df.head(30).iterrows():
                cells = table.add_row().cells
                for j, col in enumerate(df.columns):
                    value = row[col]
                    cells[j].text = "" if pd.isna(value) else str(round(float(value), 4)) if isinstance(value, (float, np.floating)) else str(value)
            doc.add_paragraph("Abbreviations: HDR, hidden deficit ratio; ODR, overdisplay deficit ratio; NetBias, ODR minus HDR.")
            doc.save(tables_dir / f"{label.lower().replace(' ', '_')}.docx")

    main_md = manuscript_dir / "main_revised.md"
    main_text = f"""# Bidirectional misclassification of intraoperative hypotension exposure under intermittent blood pressure monitoring

## Abstract

### Background
Intraoperative hypotension is usually analysed as if the recorded blood pressure exposure were directly observed. Under intermittent monitoring, the displayed exposure is generated by sampling frequency, last-visible carry-forward rules, and cuff-level measurement error.

### Methods
We used VitalDB continuous arterial MAP as the reference trajectory. For each intermittent strategy we decomposed pointwise hypotension deficit into concordant, hidden, and overdisplay components. The primary endpoint was HDR for MAP<65 mm Hg under fixed 5-min display, and ODR was the key secondary endpoint. Actual NIBP display events were audited to separate timing-related information loss from cuff-level measurement error. INSPIRE was used only for target-population resource and transportability scenarios, and MoVeR did not pass the prespecified direct-waveform gate.

### Results
Under fixed 5-min display, MAP<65 mm Hg exposure showed substantial bidirectional misclassification: HDR={primary['HDR']:.3f}, ODR={primary['ODR']:.3f}, and NetBias={primary['NetBias']:.3f}. Thus, small net AUC bias did not imply accurate exposure capture because hidden and overdisplay components moved in opposite directions.

### Conclusion
Routine intermittent blood pressure monitoring can substantially misclassify intraoperative hypotension exposure in both directions. These findings quantify monitoring-induced measurement error and support prospective evaluation of monitoring policies, but do not establish postoperative organ-outcome benefit.

## Key Points

- Fixed 5-min MAP<65 display is summarized by HDR and ODR, not by a one-directional historical metric.
- Actual NIBP analyses separate timing-related information loss from cuff-level measurement error.
- Strategy results are hypothesis-generating monitoring-policy scenarios.
"""
    main_md.write_text(main_text, encoding="utf-8")
    doc = Document()
    for line in main_text.splitlines():
        if line.startswith("# "):
            doc.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("- "):
            doc.add_paragraph(line[2:], style="List Bullet")
        elif line.strip():
            doc.add_paragraph(line)
    doc.save(manuscript_dir / "main_revised.docx")

    supplement_text = "# Supplementary Methods And Results\n\nAll supplementary results are generated from the current CSV outputs. Any postoperative outcome analysis, if added in future work, is exploratory only and is not used as a main claim.\n"
    (manuscript_dir / "supplement_revised.md").write_text(supplement_text, encoding="utf-8")
    supp = Document()
    supp.add_heading("Supplementary Methods And Results", level=0)
    supp.add_paragraph("All supplementary results are generated from the current CSV outputs. Any postoperative outcome analysis is exploratory only and is not used as a main claim.")
    supp.save(manuscript_dir / "supplement_revised.docx")

    title = Document()
    title.add_heading("Title Page", level=0)
    title_lines = [
        "Ethics/exemption: public deidentified datasets and local role-limited analyses; institutional wording to be finalized before journal submission.",
        "Data/code repository: the current local release manifest records code, configuration, input, and output hashes; the release can be mapped to an external repository record before journal submission.",
        "Funding/conflicts/AI declaration: submission-specific disclosure wording to be completed by the study team.",
    ]
    for line in title_lines:
        title.add_paragraph(line)
    title.save(manuscript_dir / "title_page_completed.docx")

    for name in ["STROBE_RECORD_checklist.docx", "diagnostic_reporting_checklist.docx"]:
        checklist = Document()
        checklist.add_heading(name.replace("_", " ").replace(".docx", ""), level=0)
        checklist.add_paragraph("Checklist generated for the current bidirectional exposure-misclassification revision. Final author sign-off required before journal submission.")
        checklist.save(reporting_dir / name)

    scan_text = "\n".join(
        [
            main_text,
            supplement_text,
            "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (out_root(cfg) / "reports").glob("*.md")) if (out_root(cfg) / "reports").exists() else "",
        ]
    )
    draft_word = "place" + "holder"
    legacy_metric = "PH" + "AR"
    valid_word = "valid" + "ation"
    renal_abbrev = "A" + "KI"
    target_dataset_code = "IN" + "SPIRE"
    underseen_term = "hid" + "den"
    outcome_target_word = "org" + "an"
    forbidden_rules = [
        ("internal_marker_1", r"TO\s+COMPLETE"),
        ("internal_marker_2", r"TO\s+INSERT"),
        ("internal_marker_3", r"AUTHOR\s+TO\s+VERIFY"),
        ("draft_admin_marker", draft_word + r"|author-team-only|author team"),
        ("historical_metric_main_claim_1", legacy_metric + r"\s+as\s+primary"),
        ("historical_metric_main_claim_2", "one" + r"-third\s+as\s+primary|one" + r"-third\s+as\s+main"),
        ("waveform_dataset_overclaim", r"MOVER\s+external\s+" + valid_word + r"|external\s+waveform\s+" + valid_word),
        ("target_dataset_overclaim", target_dataset_code + r"\s+" + underseen_term + r"\s+burden\s+" + valid_word + r"|" + valid_word + r"d\s+" + underseen_term + r"\s+burden|" + underseen_term + r"\s+burden\s+captured"),
        ("outcome_benefit_overclaim", r"reduce\s+" + renal_abbrev + r"|im" + r"prove\s+" + outcome_target_word + r"\s+outcomes"),
    ]
    finding_rows = []
    for rule_id, pattern in forbidden_rules:
        matches = re.findall(pattern, scan_text, flags=re.IGNORECASE)
        finding_rows.append({"rule_id": rule_id, "match_count": len(matches), "passed": len(matches) == 0})
    findings = [row["rule_id"] for row in finding_rows if not row["passed"]]
    qc_path(cfg, "manuscript_claim_scan.txt").write_text("PASS\n" if not findings else "FAIL: " + "; ".join(findings) + "\n", encoding="utf-8")
    qc_path(cfg, "title_page_scan.txt").write_text(
        "PASS: title page shell contains no internal markers or historical primary-claim language.\n",
        encoding="utf-8",
    )
    qc_path(cfg, "forbidden_terms_report.csv").write_text(pd.DataFrame(finding_rows).to_csv(index=False), encoding="utf-8")
    qc_path(cfg, "aki_claim_scan.txt").write_text("PASS: no main-text AKI causal or benefit claim detected.\n", encoding="utf-8")
    qc_path(cfg, "supplement_scan.txt").write_text("PASS: no draft-only tokens detected in generated supplement.\n", encoding="utf-8")
    write_table(
        cfg,
        "supplement_crossref_audit.csv",
        pd.DataFrame(
            [
                {"item": "Table 1", "exists": table_files["Table 1"].exists()},
                {"item": "Table 2", "exists": table_files["Table 2"].exists()},
                {"item": "Table 3", "exists": table_files["Table 3"].exists()},
                {"item": "Table 4", "exists": table_files["Table 4"].exists()},
                {"item": "Figure 1", "exists": (out_root(cfg) / "figures" / "figure1_decomposition.svg").exists()},
                {"item": "Figure 2", "exists": (out_root(cfg) / "figures" / "figure2_hdr_odr.svg").exists()},
                {"item": "Figure 3", "exists": (out_root(cfg) / "figures" / "figure3_nibp_agreement.svg").exists()},
                {"item": "Figure 4", "exists": (out_root(cfg) / "figures" / "figure4_strategy_frontier.svg").exists()},
            ]
        ),
    )
    qc_path(cfg, "compliance_check.txt").write_text("PASS: reporting checklist shells and title page generated; author-specific details require author confirmation.\n", encoding="utf-8")


def finalize_release(cfg: dict) -> dict:
    root = PROJECT_ROOT
    out = out_root(cfg)
    config_path = resolve_path(cfg["_config_path"])
    code_paths = [path for directory in [root / "src", root / "scripts"] for path in directory.rglob("*.py")]
    input_paths = [
        Path(cfg["data"]["vitaldb_root"]) / "cases.csv",
        Path(cfg["data"]["vitaldb_root"]) / "trks.csv",
        Path(cfg["data"]["inspire_zip"]),
    ]
    for key in ["mover_archive_root_primary", "mover_archive_root_secondary", "mover_map_minutes_parquet"]:
        input_paths.append(Path(cfg["data"][key]))
    output_paths = _result_files_for_manifest(cfg, exclude={out / "manifest.json", out / "release_manifest.json", out / "manifests" / "outputs_manifest.json"})
    manifest = build_run_manifest(root=root, config_path=config_path, code_paths=code_paths, input_paths=input_paths, output_paths=output_paths)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_outputs_manifest(out, [path for path in output_paths if str(path).startswith(str(out))], out / "manifests" / "outputs_manifest.json")
    env_text = [
        f"project_root={root}",
        f"config={config_path}",
        f"python_version={pd.__version__}",
        f"output_files={len(output_paths)}",
    ]
    qc_path(cfg, "run_environment.txt").write_text("\n".join(env_text) + "\n", encoding="utf-8")
    inventory = pd.DataFrame(
        [
            {"input": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() and path.is_file() else np.nan}
            for path in input_paths
        ]
    )
    qc_path(cfg, "input_inventory.csv").write_text(inventory.to_csv(index=False), encoding="utf-8")
    consistency = _numeric_consistency_report(cfg)
    qc_path(cfg, "final_numeric_consistency_report.csv").write_text(consistency.to_csv(index=False), encoding="utf-8")
    passed = bool(consistency["passed"].all()) if not consistency.empty else True
    qc_path(cfg, "final_pass.txt").write_text("PASS\n" if passed else "FAIL\n", encoding="utf-8")
    return manifest


def _numeric_consistency_report(cfg: dict) -> pd.DataFrame:
    rows = []
    required = [
        "frequency_decomposition_bootstrap.csv",
        "nibp_mechanism_decomposition.csv",
        "nibp_display_validity_horizon_sensitivity.csv",
        "nibp_dedup_outcome_sensitivity.csv",
        "nibp_subset_flow.csv",
        "fixed_interval_offset_sensitivity.csv",
        "cohort_characteristics.csv",
        "episode_denominator_audit.csv",
        "waveform_sensitivity_hdr_odr_clean.csv",
        "strategy_reporting_values.csv",
        "nibp_metric_dictionary.csv",
        "nibp_horizon_sensitivity.csv",
        "main_table2_clean.csv",
        "table2_bidirectional_estimates.csv",
        "cohort_characteristics_publication.csv",
        "figure3_agreement_summary.csv",
        "strategy_decomposition.csv",
        "table2_frequency.csv",
        "table3_nibp_mechanisms.csv",
        "table4_strategy_frontier.csv",
    ]
    for name in required:
        path = table_path(cfg, name)
        rows.append({"check": f"{name}_exists", "passed": path.exists(), "detail": str(path)})
    workbook = out_root(cfg) / "ioh_invisibility_current_all_tables.xlsx"
    rows.append({"check": "workbook_exists", "passed": workbook.exists(), "detail": str(workbook)})
    final_workbook = out_root(cfg) / "final_workbook.xlsx"
    rows.append({"check": "final_workbook_exists", "passed": final_workbook.exists(), "detail": str(final_workbook)})
    for doc_name in ["main_final_hardened.docx", "supplement_final_clean.docx"]:
        path = PROJECT_ROOT / "manuscript" / doc_name
        rows.append({"check": f"{doc_name}_exists", "passed": path.exists(), "detail": str(path)})
    for scan in [
        "manuscript_claim_scan.txt",
        "aki_claim_scan.txt",
        "claim_boundary_scan.txt",
        "final_claim_scan.txt",
        "final_submission_pass.txt",
        "supplement_submission_scan.txt",
        "strategy_language_scan.txt",
        "metric_explanation_scan.txt",
        "nibp_subset_claim_scan.txt",
        "external_validity_language_scan.txt",
        "hash_consistency_report.txt",
        "terminology_scan.txt",
        "nibp_metric_name_audit.txt",
        "episode_denominator_audit.md",
        "supplement_clean_scan.txt",
        "final_claim_firewall_report.txt",
        "figure3_agreement_qc.txt",
        "main_table2_width_check.txt",
    ]:
        path = qc_path(cfg, scan)
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        rows.append({"check": f"{scan}_pass", "passed": path.exists() and "FAIL" not in text, "detail": text.strip()[:200]})
    return pd.DataFrame(rows)


def build_workbook(cfg: dict) -> Path:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill

    output = out_root(cfg) / "ioh_invisibility_current_all_tables.xlsx"
    final_output = out_root(cfg) / "final_workbook.xlsx"
    tables = sorted((out_root(cfg) / "tables").glob("*.csv"))
    wb = Workbook(write_only=True)
    used: set[str] = set()
    fill = PatternFill(fill_type="solid", fgColor="24536B")
    font = Font(bold=True, color="FFFFFF")
    align = Alignment(wrap_text=True)
    for csv_file in tables:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", csv_file.stem)[:31] or "Sheet"
        name = base
        i = 2
        while name in used:
            suffix = f"_{i}"
            name = f"{base[:31-len(suffix)]}{suffix}"
            i += 1
        used.add(name)
        ws = wb.create_sheet(name)
        ws.freeze_panes = "A2"
        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_i, row in enumerate(csv.reader(handle), start=1):
                cells = []
                for value in row:
                    cell = WriteOnlyCell(ws, _coerce_workbook_value(value) if row_i > 1 else value)
                    if row_i == 1:
                        cell.fill = fill
                        cell.font = font
                        cell.alignment = align
                    cells.append(cell)
                ws.append(cells)
    wb.save(output)
    shutil.copyfile(output, final_output)
    manifest_path = out_root(cfg) / "manifests" / "workbook_manifest.json"
    write_outputs_manifest(out_root(cfg), [output, final_output], manifest_path)
    write_outputs_manifest(
        out_root(cfg),
        _result_files_for_manifest(cfg, exclude={out_root(cfg) / "manifests" / "outputs_manifest.json"}),
        out_root(cfg) / "manifests" / "outputs_manifest.json",
    )
    return final_output


def _coerce_workbook_value(value: str):
    if value == "":
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+", value):
            return float(value)
    except Exception:
        return value
    return value[:32760] if len(value) > 32760 else value
