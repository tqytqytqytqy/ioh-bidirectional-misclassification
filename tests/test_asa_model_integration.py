from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_script(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aki_analysis_frame_uses_clean_asa_and_preserves_recorded_sensitivity():
    module = _load_script(
        "33_run_exploratory_aki_analysis_v7.py", "aki_revision_v8_test"
    )
    manifest = pd.DataFrame(
        {
            "case_id": [1, 2],
            "subjectid": [101, 102],
            "age": [60, 70],
            "sex": ["M", "F"],
            "bmi": [24, 26],
            "asa": [3, 6],
            "emop": [0, 0],
            "preop_htn": [0, 1],
            "preop_dm": [0, 0],
            "duration_min": [120, 180],
            "department": ["A", "B"],
        }
    )
    outcomes = pd.DataFrame(
        {
            "case_id": [1, 2],
            "subjectid": [101, 102],
            "baseline_creatinine_mg_dl": [1.0, 1.0],
            "postop_cr_max_48h": [1.1, 1.4],
            "postop_cr_max_7d": [1.1, 1.4],
            "postop_cr_count_48h": [1, 1],
            "postop_cr_count_7d": [1, 1],
            "aki_creatinine_7d": [False, True],
            "aki_stage_creatinine": [0, 1],
            "aki_evaluable": [True, True],
        }
    )
    features = pd.DataFrame(
        {
            "case_id": [1, 2],
            "true_twa": [2.0, 3.0],
            "true_auc": [10.0, 20.0],
            "hdr": [0.5, 0.6],
            "predisplay_auc_fraction": [0.4, 0.5],
            "hidden_twa": [1.0, 2.0],
        }
    )

    frame = module._analysis_frame(manifest, outcomes, features)

    assert frame.loc[frame["case_id"].eq(1), "asa_high"].iloc[0] == 1.0
    assert pd.isna(frame.loc[frame["case_id"].eq(2), "asa_high"].iloc[0])
    assert frame.loc[frame["case_id"].eq(2), "asa_high_recorded"].iloc[0] == 1.0

    assert hasattr(module, "_recorded_asa_sensitivity_frame")
    sensitivity = module._recorded_asa_sensitivity_frame(frame)
    assert sensitivity.loc[sensitivity["case_id"].eq(2), "asa_high"].iloc[0] == 1.0
    assert pd.isna(frame.loc[frame["case_id"].eq(2), "asa_high"].iloc[0])


def test_table1_uses_valid_asa_denominator_and_reports_levels_one_to_five():
    module = _load_script(
        "27_execute_anesthesiology_revision_v7.py", "table1_revision_v8_test"
    )
    n = 6
    manifest = pd.DataFrame(
        {
            "case_id": range(1, n + 1),
            "age": [50] * n,
            "sex": ["M"] * n,
            "bmi": [25] * n,
            "asa": [1, 2, 3, 4, 5, 6],
            "emop": [0] * n,
            "preop_htn": [0] * n,
            "preop_dm": [0] * n,
            "duration_min": [120] * n,
            "opstart": [100] * n,
            "opend": [3700] * n,
            "has_nibp_map": [1] * n,
        }
    )

    table = module._cohort_characteristics(manifest, pd.Index(range(1, n + 1)))
    asa_rows = table.loc[table["characteristic"].eq("ASA Physical Status")]

    assert asa_rows["level"].tolist() == ["1", "2", "3", "4", "5"]
    assert asa_rows["nonmissing_n"].eq(5).all()
    assert asa_rows["missing_n"].notna().sum() == 1
    assert asa_rows["missing_n"].dropna().iloc[0] == 1
    assert asa_rows["summary"].tolist() == [
        "1 (20.0%)",
        "1 (20.0%)",
        "1 (20.0%)",
        "1 (20.0%)",
        "1 (20.0%)",
    ]


def test_icu_sensitivity_uses_recorded_asa_without_mutating_primary_frame():
    module = _load_script(
        "39_run_exploratory_icu_resource_use_v74.py", "icu_revision_v8_test"
    )
    assert hasattr(module, "_recorded_asa_sensitivity_frame")
    frame = pd.DataFrame(
        {
            "case_id": [1, 2],
            "asa_high": [1.0, np.nan],
            "asa_high_recorded": [1.0, 1.0],
        }
    )

    sensitivity = module._recorded_asa_sensitivity_frame(frame)

    assert sensitivity["asa_high"].tolist() == [1.0, 1.0]
    assert pd.isna(frame.loc[1, "asa_high"])
