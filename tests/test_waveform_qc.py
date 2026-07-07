import numpy as np
import pandas as pd

from ioh.pipeline import _time_window_bounds, resample_median_grid
from ioh.reporting.manifest import mover_claim_gate


def test_resample_filters_implausible_map_values_before_bin_median():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([1000.0, 1000.0, 80.0])

    _, grid = resample_median_grid(times, values, start_sec=0, end_sec=10, step_sec=10, min_map=20, max_map=180)

    assert grid[0] == 80.0


def test_vitaldb_negative_interval_start_is_clamped_to_recording_zero():
    row = pd.Series({"anestart": -1667, "aneend": 6073.0, "opstart": 2413, "opend": 5413})

    window, start_sec, end_sec = _time_window_bounds(row, "anaesthetic_interval")

    assert window == "anaesthetic_interval"
    assert start_sec == 0.0
    assert end_sec == 6073.0


def test_waveform_gate_language_avoids_direct_claim_when_failed():
    gate = mover_claim_gate(gold_links=0, silver_links=0, ambiguous_rejects=0, min_gold=1000)

    assert gate["direct_waveform_claim_allowed"] is False
    assert "validation" not in gate["language"].lower()
