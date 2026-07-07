import pytest

from ioh.reporting.language import assert_no_forbidden_language
from ioh.reporting.manifest import mover_claim_gate


def test_mover_claim_gate_blocks_direct_waveform_claim_when_gold_links_are_low():
    gate = mover_claim_gate(gold_links=2, silver_links=6, ambiguous_rejects=629, min_gold=1000)

    assert gate["direct_waveform_claim_allowed"] is False
    assert "fallback" in gate["language"].lower()


def test_language_gate_rejects_forbidden_external_validation_claim():
    with pytest.raises(ValueError, match="external validation"):
        assert_no_forbidden_language("This is an external validation of hidden burden.", mover_allowed=False)
