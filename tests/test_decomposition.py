import numpy as np

from ioh.estimands.decomposition import decompose_deficit


def test_pointwise_decomposition_separates_hidden_and_overdisplay_without_net_cancellation():
    reference = np.array([70.0, 60.0, 60.0, 70.0])
    display = np.array([70.0, 70.0, 55.0, 55.0])

    out = decompose_deficit(reference, display, threshold=65.0, dt_min=1.0)

    assert out["true_auc"] == 10.0
    assert out["display_auc"] == 20.0
    assert out["hidden_auc"] == 5.0
    assert out["overdisplay_auc"] == 15.0
    assert out["concordant_auc"] == 5.0
    assert out["hidden_auc"] + out["concordant_auc"] == out["true_auc"]
    assert out["overdisplay_auc"] + out["concordant_auc"] == out["display_auc"]
    assert out["net_bias_ratio"] == 1.0


def test_missing_display_is_treated_as_no_visible_deficit_not_as_zero_map():
    reference = np.array([60.0, 70.0])
    display = np.array([np.nan, np.nan])

    out = decompose_deficit(reference, display, threshold=65.0, dt_min=1.0)

    assert out["true_auc"] == 5.0
    assert out["display_auc"] == 0.0
    assert out["hidden_auc"] == 5.0
    assert out["overdisplay_auc"] == 0.0
