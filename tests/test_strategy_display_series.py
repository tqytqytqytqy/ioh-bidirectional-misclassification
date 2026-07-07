import numpy as np

from ioh.estimands.decomposition import decompose_deficit
from ioh.pipeline import _duration_bin, build_strategy_display_series, pareto_optimal_mask, strategy_cost_metadata


def test_adaptive_strategy_builds_distinct_display_series_and_odr():
    times = np.arange(0, 16 * 60, 10, dtype=float)
    values = np.full_like(times, 78.0)
    values[(times >= 5 * 60) & (times <= 7 * 60)] = 58.0
    values[(times >= 10 * 60) & (times <= 11 * 60)] = 72.0

    fixed, fixed_meta = build_strategy_display_series(times, values, "fixed_5min_reference")
    adaptive, adaptive_meta = build_strategy_display_series(times, values, "threshold_triggered_adaptive")

    assert fixed_meta["measurement_burden_per_hour"] == 12.0
    assert adaptive_meta["trigger_count"] > 0
    assert not np.allclose(np.nan_to_num(fixed, nan=-1), np.nan_to_num(adaptive, nan=-1))

    fixed_metrics = decompose_deficit(values, fixed, threshold=65, dt_min=10 / 60)
    adaptive_metrics = decompose_deficit(values, adaptive, threshold=65, dt_min=10 / 60)

    assert adaptive_metrics["overdisplay_auc"] != fixed_metrics["overdisplay_auc"]


def test_duration_bin_boundaries_match_revision_checklist():
    assert _duration_bin(20) == "<30s"
    assert _duration_bin(30) == "30-60s"
    assert _duration_bin(60) == "1-3min"
    assert _duration_bin(180) == "3-5min"
    assert _duration_bin(300) == ">5min"


def test_pareto_frontier_prefers_lower_burden_and_higher_hdr_reduction():
    burden = np.array([0.0, 3.0, 6.0, 6.0])
    benefit = np.array([0.0, 0.2, 0.4, 0.3])

    assert pareto_optimal_mask(burden, benefit).tolist() == [True, True, True, False]


def test_continuous_monitoring_strategy_is_not_zero_cost_cuff_cadence():
    selective = strategy_cost_metadata("selective_continuous_top20_oof", cuff_measurements_per_hour=12.0)
    fixed = strategy_cost_metadata("fixed_3min", cuff_measurements_per_hour=20.0)

    assert fixed["cost_domain"] == "cuff_cadence"
    assert fixed["extra_cuff_inflations_per_hour"] == 8.0
    assert selective["cost_domain"] == "modality_change"
    assert np.isnan(selective["extra_cuff_inflations_per_hour"])
    assert selective["modality_burden_note"]
