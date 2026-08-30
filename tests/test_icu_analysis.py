from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioh.icu_analysis import assess_model_stability


def test_model_stability_passes_a_well_supported_estimate() -> None:
    result = assess_model_stability(
        {
            "converged": True,
            "events": 150,
            "n_parameters": 15,
            "relative_risk": 1.10,
            "ci_low": 1.01,
            "ci_high": 1.20,
            "standard_error": 0.04,
            "p_value": 0.03,
        }
    )

    assert result["model_stability_status"] == "PASS"
    assert result["events_per_parameter"] == 10.0


def test_model_stability_fails_without_enough_events_per_parameter() -> None:
    result = assess_model_stability(
        {
            "converged": True,
            "events": 20,
            "n_parameters": 10,
            "relative_risk": 1.10,
            "ci_low": 0.80,
            "ci_high": 1.50,
            "standard_error": 0.15,
            "p_value": 0.40,
        }
    )

    assert result["model_stability_status"] == "NOT_PASS"
    assert "events_per_parameter" in result["model_stability_reason"]


def test_model_stability_fails_nonfinite_or_extreme_interval() -> None:
    nonfinite = assess_model_stability(
        {
            "converged": True,
            "events": 150,
            "n_parameters": 10,
            "relative_risk": 1.10,
            "ci_low": np.nan,
            "ci_high": 1.50,
            "standard_error": 0.15,
            "p_value": 0.40,
        }
    )
    extreme = assess_model_stability(
        {
            "converged": True,
            "events": 150,
            "n_parameters": 10,
            "relative_risk": 1.10,
            "ci_low": 0.20,
            "ci_high": 2.50,
            "standard_error": 0.15,
            "p_value": 0.40,
        }
    )

    assert nonfinite["model_stability_status"] == "NOT_PASS"
    assert extreme["model_stability_status"] == "NOT_PASS"
    assert "ci_width_ratio" in extreme["model_stability_reason"]
