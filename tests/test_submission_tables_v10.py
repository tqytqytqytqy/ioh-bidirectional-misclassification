import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


revision = _load_script("revision_v10_tables", "27_execute_anesthesiology_revision_v7.py")
figures = _load_script("revision_v10_figures", "28_make_anesthesiology_tables_figures_v7.py")


def test_cohort_characteristics_reports_female_and_asa_missing_once():
    manifest = pd.DataFrame(
        {
            "case_id": [1, 2, 3, 4],
            "age": [50, 60, 70, 80],
            "sex": ["M", "F", "Female", "Male"],
            "bmi": [20, 21, 22, 23],
            "asa": [1, 2, np.nan, 6],
            "emop": [0, 0, 1, 0],
            "preop_htn": [0, 1, 1, 0],
            "preop_dm": [0, 0, 1, 0],
            "duration_min": [120, 130, 140, 150],
            "opstart": [0, 0, 0, 0],
            "opend": [6000, 6600, 7200, 7800],
            "has_nibp_map": [1, 1, 0, 1],
        }
    )

    table = revision._cohort_characteristics(manifest, pd.Index([1, 2, 3, 4]))

    female = table[(table["characteristic"] == "Sex") & (table["level"] == "Female")]
    assert female.iloc[0]["summary"] == "2 (50.0%)"
    asa = table[table["characteristic"] == "ASA Physical Status"]
    assert asa["missing_n"].notna().sum() == 1
    assert asa["missing_n"].dropna().iloc[0] == 2
    assert asa["note"].dropna().nunique() == 1


def test_figure1_flow_source_reports_each_sequential_exclusion():
    cohort_flow = pd.DataFrame(
        {
            "step": [
                "source_cases",
                "adult",
                "general_anaesthesia",
                "eligible_surgery",
                "duration_ge_60min",
                "has_art_map",
                "art_coverage_ge_80pct",
            ],
            "remaining": [6388, 6331, 5989, 5559, 5449, 3315, 2435],
            "excluded": [0, 57, 342, 430, 110, 2134, 880],
        }
    )
    nibp_count = {
        "primary_art_coverage_processing_pool": 1995,
        "reconstructed_nibp_display_events": 1993,
        "paired_nibp_events_primary_window": 1908,
    }

    flow = figures._figure1_flow_source(cohort_flow, nibp_count)

    coverage = flow[flow["stage"] == "Primary arterial waveform cohort"].iloc[0]
    assert coverage["n_cases"] == 2435
    assert coverage["excluded_from_previous"] == 880
    assert coverage["exclusion_reason"] == "valid arterial MAP coverage <80%"
    assert (
        flow.loc[flow["branch"] == "primary", "excluded_from_previous"]
        .fillna(0)
        .sum()
        == 3953
    )
