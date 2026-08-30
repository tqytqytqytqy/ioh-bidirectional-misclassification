from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nibp_subset_characteristics_use_correct_cohort_labels_and_counts() -> None:
    path = PROJECT_ROOT / "outputs" / "tables" / "supp_nibp_subset_characteristics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    obsolete_group = "failed" + "_pairing_cases"
    assert obsolete_group not in {row["group"] for row in rows}
    counts = {
        row["group"]: int(row["n_cases"])
        for row in rows
        if row["variable"] == "age"
    }
    assert counts == {
        "primary_waveform_cohort": 2435,
        "nibp_mechanism_subset": 1993,
        "paired_agreement_cases": 1908,
        "mechanism_without_paired_agreement": 85,
    }
    assert counts["nibp_mechanism_subset"] == (
        counts["paired_agreement_cases"]
        + counts["mechanism_without_paired_agreement"]
    )


def test_timing_only_cases_have_no_paired_events() -> None:
    path = PROJECT_ROOT / "outputs" / "tables" / "supp_nibp_subset_characteristics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    row = next(
        item
        for item in rows
        if item["group"] == "mechanism_without_paired_agreement"
        and item["variable"] == "paired_events"
    )
    assert int(row["n_cases"]) == 85
    assert float(row["value"]) == 0.0
