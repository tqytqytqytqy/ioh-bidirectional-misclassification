from ioh.pipeline import _complete_waveform_sensitivity_row


def test_waveform_sensitivity_row_requires_numeric_outputs():
    row = _complete_waveform_sensitivity_row(
        sensitivity="coverage_ge_90pct",
        n_cases=10,
        excluded_cases=2,
        true_auc=100.0,
        hidden_auc=40.0,
        overdisplay_auc=30.0,
        display_auc=90.0,
        normotensive_discordance_min=12.0,
        hypotensive_discordance_min=8.0,
    )

    assert row["TrueAUC65"] == 100.0
    assert row["HDR65"] == 0.4
    assert row["ODR65"] == 0.3
    assert row["NetBias"] == -0.1
    assert row["normotensive_display_discordance_min"] == 12.0
    assert row["hypotensive_display_discordance_min"] == 8.0
    assert row["status"] == "computed"
