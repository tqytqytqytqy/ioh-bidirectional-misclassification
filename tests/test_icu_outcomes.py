from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioh.icu_outcomes import build_prolonged_icu_outcome, summarize_icu_gate


def test_build_prolonged_icu_outcome_uses_ge_two_days() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [1, 2, 3, 4],
            "subjectid": [11, 12, 13, 14],
            "icu_days": [0, 1, 2, 3],
        }
    )

    result = build_prolonged_icu_outcome(frame, threshold_days=2)

    assert result["prolonged_icu_ge2d"].tolist() == [False, False, True, True]
    assert result["outcome_evaluable"].tolist() == [True, True, True, True]


def test_build_prolonged_icu_outcome_retains_missing_as_nonevaluable() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [1, 2],
            "subjectid": [11, 12],
            "icu_days": [2, None],
        }
    )

    result = build_prolonged_icu_outcome(frame, threshold_days=2)

    assert bool(result.loc[0, "prolonged_icu_ge2d"])
    assert pd.isna(result.loc[1, "prolonged_icu_ge2d"])
    assert result["outcome_evaluable"].tolist() == [True, False]


def test_build_prolonged_icu_outcome_fails_on_duplicate_cases() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [1, 1],
            "subjectid": [11, 11],
            "icu_days": [0, 2],
        }
    )

    with pytest.raises(ValueError, match="one row per case_id"):
        build_prolonged_icu_outcome(frame)


def test_build_prolonged_icu_outcome_fails_on_negative_duration() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [1],
            "subjectid": [11],
            "icu_days": [-1],
        }
    )

    with pytest.raises(ValueError, match="negative"):
        build_prolonged_icu_outcome(frame)


def test_gate_passes_structural_rules_but_preserves_endpoint_limitation() -> None:
    frame = pd.DataFrame(
        {
            "case_id": range(120),
            "subjectid": range(120),
            "icu_days": [2] * 100 + [0] * 20,
        }
    )

    gate = summarize_icu_gate(
        frame,
        threshold_days=2,
        min_completeness=0.95,
        min_events=100,
    )

    summary = gate.set_index("criterion")
    assert summary.loc["overall_gate", "status"] == "PASS_WITH_ENDPOINT_LIMITATION"
    assert summary.loc["event_count", "status"] == "PASS"
    assert "planned" in summary.loc["overall_gate", "limitation"].lower()


def test_gate_does_not_pass_when_event_count_is_too_small() -> None:
    frame = pd.DataFrame(
        {
            "case_id": range(120),
            "subjectid": range(120),
            "icu_days": [2] * 99 + [0] * 21,
        }
    )

    gate = summarize_icu_gate(frame, min_events=100)

    overall = gate.loc[gate["criterion"].eq("overall_gate")].iloc[0]
    assert overall["status"] == "NOT_PASS"
