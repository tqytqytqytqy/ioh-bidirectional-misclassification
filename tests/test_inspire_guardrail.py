import pytest

from ioh.io.inspire_loader import guard_inspire_metric as guard_target_metric


def test_target_dataset_guardrail_rejects_disallowed_metrics():
    with pytest.raises(ValueError, match="continuous arterial waveform truth"):
        guard_target_metric("hidden_auc")


def test_target_dataset_guardrail_allows_resource_metrics():
    assert guard_target_metric("resource_burden") == "resource_burden"
