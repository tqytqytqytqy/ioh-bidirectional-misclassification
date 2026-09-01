from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_MANIFEST = ROOT / "outputs" / "manifests" / "outputs_manifest.json"
RELEASE_MANIFEST = ROOT / "outputs" / "manifests" / "release_manifest.json"
CHECKSUMS = ROOT / "checksums.sha256"
PROVENANCE = ROOT / "outputs" / "manifests" / "provenance.json"
RELEASE = "v1.3.0-submission"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "outputs" / "tables" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value)


def build_provenance() -> None:
    cohort = rows("repeated_surgery_audit.csv")[0]
    episode = next(
        row
        for row in rows("episode_observability_5min_overall.csv")
        if number(row["threshold"]) == 65 and number(row["interval_min"]) == 5
    )
    auc = next(
        row
        for row in rows("frequency_decomposition_bootstrap.csv")
        if number(row["threshold"]) == 65 and number(row["interval_min"]) == 5
    )
    aki_rows = rows("main_table4_exploratory_aki.csv")
    aki_presence = next(row for row in aki_rows if row["analysis"].startswith("Any hidden burden"))
    aki_clinical = next(row for row in aki_rows if row["adjustment"] == "clinical")
    aki_incremental = next(row for row in aki_rows if row["adjustment"] == "clinical_plus_reference" and row["analysis"].startswith("Absolute"))
    aki_hdr = next(row for row in aki_rows if row["analysis"].startswith("HDR65"))
    icu_rows = rows("exploratory_icu_sequential_models.csv")
    icu_clinical = next(row for row in icu_rows if row["adjustment_sequence"] == "clinical")
    icu_incremental = next(row for row in icu_rows if row["adjustment_sequence"] == "clinical_plus_reference")
    nibp = rows("nibp_case_event_flow_v7.csv")
    nibp_pool = next(row for row in nibp if row["step"] == "primary_art_coverage_processing_pool")
    nibp_paired = next(row for row in nibp if row["step"] == "paired_nibp_events_primary_window")

    write_json(
        PROVENANCE,
        {
            "analysis_version": RELEASE,
            "freeze_date": "2026-09-01",
            "primary_dataset": "VitalDB",
            "primary_analysis_unit": cohort["primary_analysis_unit"],
            "bootstrap_cluster": cohort["bootstrap_cluster"],
            "primary_cohort_cases": int(cohort["n_cases"]),
            "primary_cohort_subjects": int(cohort["n_subjects"]),
            "subjects_with_repeated_surgery": int(cohort["subjects_with_repeated_surgery"]),
            "additional_cases_from_repeated_surgery": int(cohort["additional_cases_from_repeated_surgery"]),
            "maximum_cases_per_subject": int(cohort["maximum_cases_per_subject"]),
            "map65_episodes_ge60s": int(number(episode["reference_episodes"])),
            "cases_with_map65_episodes_ge60s": int(episode["cases_with_episodes"]),
            "map65_episode_detection_probability": number(episode["episode_detection_probability"]),
            "map65_complete_miss_probability": number(episode["complete_miss_probability"]),
            "map65_first_display_ge1min_probability": number(episode["detection_with_1min_remaining_probability"]),
            "map65_first_display_ge2min_probability": number(episode["detection_with_2min_remaining_probability"]),
            "map65_reference_auc_before_first_display_fraction": number(episode["reference_auc_before_first_low_display_fraction"]),
            "map65_hdr": number(auc["HDR"]),
            "map65_odr": number(auc["ODR"]),
            "map65_net_bias": number(auc["NetBias"]),
            "map65_hdr_ci95": [number(auc["HDR_ci_low"]), number(auc["HDR_ci_high"])],
            "map65_odr_ci95": [number(auc["ODR_ci_low"]), number(auc["ODR_ci_high"])],
            "map65_net_bias_ci95": [number(auc["NetBias_ci_low"]), number(auc["NetBias_ci_high"])],
            "aki_definition": "KDIGO serum-creatinine criteria only: at least 0.3 mg/dL within 48 h or at least 1.5 times baseline within 7 d, censored at discharge",
            "aki_evaluable_cases": int(aki_presence["n_cases"]),
            "aki_events": int(aki_presence["events"]),
            "aki_hidden_presence_status": aki_presence["estimation_status"],
            "aki_principal_absolute_burden_model_cases": int(aki_clinical["n_cases"]),
            "aki_principal_rr_per_10_mmHg_min_h": number(aki_clinical["relative_risk"]),
            "aki_principal_ci95": [number(aki_clinical["ci_low"]), number(aki_clinical["ci_high"])],
            "aki_principal_p_value": number(aki_clinical["p_value"]),
            "aki_incremental_absolute_burden_rr_per_10_mmHg_min_h": number(aki_incremental["relative_risk"]),
            "aki_incremental_absolute_burden_ci95": [number(aki_incremental["ci_low"]), number(aki_incremental["ci_high"])],
            "aki_incremental_absolute_burden_p_value": number(aki_incremental["p_value"]),
            "aki_secondary_hdr65_rr_per_10_percentage_points": number(aki_hdr["relative_risk"]),
            "aki_secondary_hdr65_ci95": [number(aki_hdr["ci_low"]), number(aki_hdr["ci_high"])],
            "icu_endpoint": icu_clinical["outcome_definition"],
            "icu_endpoint_role": icu_clinical["endpoint_role"],
            "icu_endpoint_limitation": icu_clinical["endpoint_limitation"],
            "icu_clinical_model_cases": int(icu_clinical["n_cases"]),
            "icu_clinical_model_subjects": int(icu_clinical["n_subjects"]),
            "icu_clinical_model_events": int(icu_clinical["events"]),
            "icu_clinical_rr_per_10_mmHg_min_h": number(icu_clinical["relative_risk"]),
            "icu_clinical_ci95": [number(icu_clinical["ci_low"]), number(icu_clinical["ci_high"])],
            "icu_clinical_p_value": number(icu_clinical["p_value"]),
            "icu_incremental_rr_per_10_mmHg_min_h": number(icu_incremental["relative_risk"]),
            "icu_incremental_ci95": [number(icu_incremental["ci_low"]), number(icu_incremental["ci_high"])],
            "icu_incremental_p_value": number(icu_incremental["p_value"]),
            "nibp_processing_pool_cases": int(nibp_pool["n_cases"]),
            "paired_nibp_art_cases": int(nibp_paired["n_cases"]),
            "paired_nibp_art_events": int(number(nibp_paired["n_events"])),
            "nibp_pairing_window": "median arterial MAP from -30 to +30 s around each NIBP event",
            "privacy": "aggregate release only; no raw source or case-level data redistributed",
            "source_template_sha256": sha256(ROOT / "config" / "reproduction.example.yaml"),
        },
    )


def output_files() -> list[Path]:
    paths = [ROOT / "outputs" / "final_workbook.xlsx", ROOT / "outputs" / "manifests" / "provenance.json"]
    paths.extend((ROOT / "outputs" / "tables").glob("*.csv"))
    paths.extend((ROOT / "outputs" / "figures").glob("*"))
    paths.extend((ROOT / "manuscript_inputs" / "figure_source_files").glob("*.csv"))
    return sorted(path for path in paths if path.is_file())


def repository_files() -> list[Path]:
    excluded = {RELEASE_MANIFEST.resolve(), CHECKSUMS.resolve()}
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.resolve() in excluded or path.suffix == ".pyc":
            continue
        paths.append(path)
    return sorted(paths)


def build_outputs_manifest() -> None:
    build_provenance()
    write_json(
        OUTPUTS_MANIFEST,
        {
            "release": RELEASE,
            "scope": "non-identifiable aggregate outputs and figure artifacts",
            "files": [entry(path) for path in output_files()],
        },
    )


def build_release_manifest() -> None:
    files = repository_files()
    write_json(
        RELEASE_MANIFEST,
        {
            "release": RELEASE,
            "scope": "public repository inventory",
            "files": [entry(path) for path in files],
        },
    )
    checksum_lines = [f"{item['sha256']}  {item['relative_path']}" for item in [entry(path) for path in files]]
    checksum_lines.append(f"{sha256(RELEASE_MANIFEST)}  {RELEASE_MANIFEST.relative_to(ROOT).as_posix()}")
    CHECKSUMS.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("outputs", "release"))
    args = parser.parse_args()
    if args.mode == "outputs":
        build_outputs_manifest()
    else:
        build_release_manifest()


if __name__ == "__main__":
    main()
