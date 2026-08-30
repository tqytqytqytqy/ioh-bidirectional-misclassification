from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = PROJECT_ROOT / "outputs" / "tables"
OUTPUT = PROJECT_ROOT / ".artifact_work_v7" / "workbook_data.json"


SHEETS = [
    ("Cohort", "table1_primary_cohort_characteristics.csv", "Primary arterial waveform cohort characteristics"),
    ("Primary_AUC", "primary_auc_by_threshold_interval.csv", "Phase-averaged AUC decomposition by threshold and sampling interval"),
    ("Main_Table2", "main_table2_map65_5min_event_observability.csv", "Episode observability and decision-opportunity metrics at a 5-min interval"),
    ("Main_Table3", "main_table3_map65_sampling_interval_sensitivity.csv", "MAP <65 mmHg sensitivity across sampling intervals"),
    ("Main_Table4", "main_table4_exploratory_aki.csv", "Exploratory associations between counterfactual hidden hypotension and postoperative AKI"),
    ("Epi_AllIntervals", "episode_observability_by_threshold_interval.csv", "Episode observability by threshold and sampling interval"),
    ("Epi_5min_Overall", "episode_observability_5min_overall.csv", "Overall episode observability at a 5-min interval"),
    ("Epi_5min_Duration", "episode_observability_5min_by_duration.csv", "Episode observability at 5 min by episode duration"),
    ("Epi_5min_Nadir", "episode_observability_5min_by_nadir.csv", "Episode observability at 5 min by nadir category"),
    ("Epi_5min_DurNadir", "episode_observability_5min_duration_severity.csv", "Episode observability at 5 min by duration and nadir"),
    ("Epi_Consistency", "episode_observability_5min_consistency.csv", "Independent 5-min episode-analysis consistency check"),
    ("Epi_Action_QC", "episode_actionability_5min_qc.csv", "Actionability metric hierarchy, range, and confidence-interval closure checks"),
    ("NIBP_Flow", "nibp_case_event_flow_v7.csv", "All-eligible NIBP case and event flow"),
    ("NIBP_Agreement", "nibp_repeated_measures_bland_altman.csv", "Repeated-measures NIBP minus ART agreement"),
    ("NIBP_Classify", "nibp_paired_classification_clustered.csv", "Paired NIBP classification by threshold"),
    ("NIBP_Strata", "nibp_bland_altman_by_stratum.csv", "Paired differences by arterial MAP stratum"),
    ("NIBP_PairWindow", "nibp_pairing_sensitivity_grid.csv", "NIBP-ART pairing-window sensitivity"),
    ("NIBP_Dedup", "nibp_dedup_sensitivity_grid.csv", "NIBP display-event reconstruction sensitivity"),
    ("NIBP_DisplayValid", "nibp_display_validity_horizon_sensitivity.csv", "NIBP display-validity horizon sensitivity"),
    ("NIBP_Mechanism", "table3_actual_nibp_timing_cuff_decomposition.csv", "NIBP timing and cuff-level decomposition"),
    ("Fig1_Flow", "figure1_cohort_flow_source.csv", "Figure 1 cohort-flow source data"),
    ("Fig1_Schematic", "figure1_schematic_source.csv", "Figure 1 analytic schematic source data"),
    ("Fig2_Source", "figure2_episode_observability_source.csv", "Figure 2 source data"),
    ("Fig3_Source", "figure3_timing_consequences_source.csv", "Figure 3 source data"),
    ("Figure_Legends", "figure_legends_v7.csv", "Final figure titles and legends"),
    ("AKI_Gate0", "aki_outcome_gate0.csv", "Exploratory postoperative AKI feasibility gate"),
    ("AKI_Outcome", "aki_outcome_summary.csv", "Creatinine-defined postoperative AKI ascertainment and stages"),
    ("AKI_Presence", "aki_hidden_presence.csv", "Hidden-hypotension presence and binary estimability"),
    ("AKI_Sequential", "exploratory_aki_sequential_models.csv", "Sequential absolute hidden-burden models for postoperative AKI"),
    ("AKI_Models", "exploratory_aki_incremental_models.csv", "Secondary compositional and sensitivity models for postoperative AKI"),
    ("AKI_Missingness", "aki_outcome_missingness_comparison.csv", "Characteristics by postoperative AKI evaluability"),
    ("AKI_HDR_Quartiles", "aki_unadjusted_by_hdr_quartile.csv", "Unadjusted postoperative AKI risk by HDR65 quartile"),
    ("FigS1_AKI_Source", "figure_s1_exploratory_aki_forest_source.csv", "Supplementary Figure S1 source data"),
]


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def main() -> None:
    payload = {
        "metadata": {
            "title": "IOH temporal observability with exploratory postoperative AKI analysis",
            "analysis_version": "7.3-anesthesiology-aki-three-level",
            "primary_dataset": "VitalDB",
            "primary_cohort_cases": 2435,
            "nibp_processing_pool_cases": 1995,
            "episode_definition": "reference MAP below threshold for at least 60 s on the 10-s grid",
            "phase_definition": "all 10-s offsets within each sampling interval",
            "actionability_definition": "probability of first low display with at least 1 or 2 min before arterial recovery; reference AUC accrued strictly before first low display",
            "uncertainty": "case-cluster non-parametric bootstrap, 1000 replicates",
            "aki_definition": "KDIGO serum-creatinine criteria only: increase at least 0.3 mg/dL within 48 h or at least 1.5 times baseline within 7 d, censored at discharge",
            "aki_model": "post hoc sequential modified Poisson analysis with subject-cluster robust variance: absolute hidden burden unadjusted, clinically adjusted, and additionally adjusted for nonlinear total reference burden; HDR65 secondary",
            "aki_interpretation": "exploratory and noncausal; absolute hidden burden tracked AKI risk after clinical adjustment but did not add clear information beyond total reference burden",
            "nibp_pairing": "median ART MAP within -30 to +30 s of NIBP display completion",
            "privacy": "aggregate outputs only; raw datasets and case-level trajectories are not redistributed",
        },
        "sheets": [],
    }
    for sheet_name, filename, title in SHEETS:
        path = TABLE_ROOT / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        payload["sheets"].append(
            {
                "name": sheet_name,
                "source": f"outputs/tables/{filename}",
                "title": title,
                "columns": [str(column) for column in frame.columns],
                "rows": [
                    [clean_value(value) for value in row]
                    for row in frame.itertuples(index=False, name=None)
                ],
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
