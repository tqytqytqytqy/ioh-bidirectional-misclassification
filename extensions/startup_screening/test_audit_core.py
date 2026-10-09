import os
import numpy as np
from audit_core import decode_waveform_time, low_plateau, median_grid, numeric_screen, run_length

def test_missing_gap_breaks_run():
    assert run_length([True, True, False, True]) == 2
    assert not low_plateau([30, 30, 30, np.nan, 30, 30, 30])

def test_low_value_not_automatically_plateau():
    assert not low_plateau([20, 30, 39, 25, 35, 31])
    assert low_plateau([30, 31, 30, 30, 30, 30])

def test_range_before_median_and_no_interpolation():
    assert np.allclose(median_grid([0, 1, 2, 20], [0, 30, 40, 80], 0, 4, 20, 180), [35, np.nan, 80, np.nan], equal_nan=True)

def test_missing_pulse_pressure_not_flag():
    x = np.repeat(30.0, 6)
    s = np.full(6, np.nan)
    f = numeric_screen(x, s, s, 0)
    assert f['low_plateau_flag'] and (not f['low_pulse_pressure_flag'])

def test_valid_pulsatile_hypotension_not_consistency_error():
    f = numeric_screen(np.repeat(35.0, 6), np.repeat(60.0, 6), np.repeat(25.0, 6), 0)
    assert f['low_plateau_flag']
    assert not f['low_pulse_pressure_flag'] and (not f['map_bounds_flag'])

def test_waveform_time_chunk_offset():
    assert np.allclose(decode_waveform_time(3, 0.002, first_index=100), [0.2, 0.202, 0.204])

def test_masked_low_values_cannot_seed_late_display():
    from run_startup_sensitivity import components
    score = components(np.r_[np.repeat(30.0, 60), np.repeat(80.0, 60)], 60)
    assert np.allclose(score[:5], 0)
    assert score[5] == 10

def test_phase_reconstruction_against_existing_implementation():
    import sys
    from pathlib import Path
    from run_startup_sensitivity import components
    src = Path(__file__).resolve().parents[2] / 'analysis_v10_subject_phase_release/src'
    sys.path.insert(0, str(src))
    from ioh.estimands.decomposition import decompose_deficit, emulate_last_visible
    v = np.array([80.0, 50.0, np.nan, 55.0, 90.0] * 20)
    t = np.arange(len(v)) * 10.0
    expected = []
    for offset in range(0, 300, 10):
        d = emulate_last_visible(t, v, 300, offset)
        p = decompose_deficit(v, d, 65, 1 / 6)
        expected.append([p[k] for k in ['true_auc', 'hidden_auc', 'overdisplay_auc', 'normotensive_display_min', 'hypotensive_display_discordance_min']])
    assert np.allclose(components(v, 0)[:5], np.mean(expected, axis=0))

def test_waveform_flat_and_missing_not_confused():
    from run_waveform_audit import bin_wave
    t = np.arange(10000) * 0.002
    v = np.r_[np.repeat(30.0, 5000), np.full(5000, np.nan)]
    b = bin_wave(t, v, 0, 0, 0.002)
    assert b.loc[0, 'wave_coverage'] == 1
    assert b.loc[0, 'wave_span_90pct'] == 0
    assert not b.loc[0, 'pulse_like']
    assert b.loc[1, 'wave_coverage'] == 0
    assert np.isnan(b.loc[1, 'wave_span_90pct'])
