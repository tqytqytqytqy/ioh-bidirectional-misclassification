from __future__ import annotations

import numpy as np
import pandas as pd

from ioh.aki_outcomes import (
    aggregate_invisibility_features,
    build_creatinine_aki_outcomes,
    derive_last_preoperative_creatinine,
    fit_modified_poisson,
    summarize_hidden_presence,
)


DAY = 24 * 60 * 60


def test_creatinine_aki_uses_postanesthesia_and_discharge_boundaries():
    manifest = pd.DataFrame(
        {
            "case_id": [1, 2, 3, 4, 5, 6],
            "subjectid": [101, 102, 103, 104, 105, 106],
            "aneend": [1000] * 6,
            "dis": [1000 + 10 * DAY, 1000 + 10 * DAY, 1000 + DAY] + [1000 + 10 * DAY] * 3,
            "preop_cr": [1.0, 1.0, 1.0, 1.0, 1.5, np.nan],
        }
    )
    labs = pd.DataFrame(
        [
            (1, 999, "cr", 9.0),
            (1, 1000, "cr", 8.0),
            (1, 1000 + DAY, "cr", 1.31),
            (1, 1000 + 8 * DAY, "cr", 2.0),
            (2, 1000 + 6 * DAY, "cr", 1.6),
            (3, 1000 + 2 * DAY, "cr", 2.0),
            (4, 1000 + DAY, "cr", 2.2),
            (5, 1000 + DAY, "cr", 4.6),
            (6, 1000 + DAY, "cr", 2.0),
            (1, 1000 + DAY, "hb", 4.0),
        ],
        columns=["caseid", "dt", "name", "result"],
    )

    outcome = build_creatinine_aki_outcomes(manifest, labs).set_index("case_id")

    assert outcome.loc[1, "postop_cr_max_48h"] == 1.31
    assert outcome.loc[1, "postop_cr_max_7d"] == 1.31
    assert bool(outcome.loc[1, "aki_creatinine_7d"])
    assert outcome.loc[1, "aki_stage_creatinine"] == 1

    assert bool(outcome.loc[2, "aki_creatinine_7d"])
    assert outcome.loc[2, "aki_stage_creatinine"] == 1
    assert pd.isna(outcome.loc[3, "aki_creatinine_7d"])
    assert outcome.loc[4, "aki_stage_creatinine"] == 2
    assert outcome.loc[5, "aki_stage_creatinine"] == 3
    assert pd.isna(outcome.loc[6, "aki_creatinine_7d"])


def test_stage_3_absolute_threshold_requires_an_acute_rise():
    manifest = pd.DataFrame(
        {
            "case_id": [1],
            "subjectid": [101],
            "aneend": [1000],
            "dis": [1000 + 10 * DAY],
            "preop_cr": [4.5],
        }
    )
    labs = pd.DataFrame(
        [(1, 1000 + DAY, "cr", 4.6)],
        columns=["caseid", "dt", "name", "result"],
    )

    outcome = build_creatinine_aki_outcomes(manifest, labs).iloc[0]

    assert not bool(outcome["aki_creatinine_7d"])
    assert outcome["aki_stage_creatinine"] == 0


def test_last_preoperative_creatinine_uses_latest_value_within_90_days():
    manifest = pd.DataFrame(
        {
            "case_id": [1, 2],
            "anestart": [1000, 2000],
        }
    )
    labs = pd.DataFrame(
        [
            (1, 1000 - 91 * DAY, "cr", 0.7),
            (1, 1000 - 5 * DAY, "cr", 0.8),
            (1, 1000 - DAY, "cr", 0.9),
            (1, 1001, "cr", 1.2),
            (2, 1900, "hb", 1.0),
        ],
        columns=["caseid", "dt", "name", "result"],
    )

    baseline = derive_last_preoperative_creatinine(manifest, labs).set_index("case_id")

    assert baseline.loc[1, "last_preop_cr_90d"] == 0.9
    assert pd.isna(baseline.loc[2, "last_preop_cr_90d"])


def test_invisibility_features_preserve_auc_partition_and_episode_denominators():
    frequency = pd.DataFrame(
        {
            "case_id": [1, 2],
            "threshold": [65.0, 65.0],
            "interval_min": [5.0, 5.0],
            "true_auc": [100.0, 0.0],
            "hidden_auc": [60.0, 0.0],
            "concordant_auc": [40.0, 0.0],
            "anesthesia_hours": [2.0, 1.0],
            "true_hypotension_min": [20.0, 0.0],
        }
    )
    episodes = pd.DataFrame(
        {
            "case_id": [1, 1],
            "threshold": [65.0, 65.0],
            "interval_min": [5.0, 5.0],
            "reference_episodes": [1, 2],
            "phase_count": [30, 30],
            "phase_episode_pairs": [30, 60],
            "expected_missed_episodes": [0.4, 1.0],
            "detected_with_60s_remaining_pairs": [10, 20],
            "detected_with_120s_remaining_pairs": [4, 8],
            "reference_episode_auc_phase_sum": [300.0, 600.0],
            "pre_detection_reference_auc_sum": [120.0, 330.0],
        }
    )

    features = aggregate_invisibility_features(frequency, episodes).set_index("case_id")

    assert features.loc[1, "true_twa"] == 50.0
    assert features.loc[1, "hidden_twa"] == 30.0
    assert features.loc[1, "concordant_twa"] == 20.0
    assert features.loc[1, "hdr"] == 0.6
    assert features.loc[1, "true_twa65"] == 50.0
    assert features.loc[1, "hidden_twa65"] == 30.0
    assert features.loc[1, "concordant_twa65"] == 20.0
    assert features.loc[1, "hdr65"] == 0.6
    assert np.isclose(features.loc[1, "episode_complete_miss_probability"], 1.4 / 3.0)
    assert np.isclose(features.loc[1, "episode_1min_opportunity_probability"], 30.0 / 90.0)
    assert np.isclose(features.loc[1, "episode_2min_opportunity_probability"], 12.0 / 90.0)
    assert features.loc[1, "predisplay_auc_fraction"] == 0.5
    assert pd.isna(features.loc[2, "hdr65"])
    assert pd.isna(features.loc[2, "predisplay_auc_fraction"])


def test_modified_poisson_reports_cluster_robust_relative_risk():
    rng = np.random.default_rng(20260706)
    n = 800
    exposure = rng.normal(size=n)
    baseline = rng.normal(size=n)
    probability = np.clip(np.exp(-3.2 + 0.45 * exposure + 0.15 * baseline), 0, 0.8)
    frame = pd.DataFrame(
        {
            "outcome": rng.binomial(1, probability),
            "exposure": exposure,
            "baseline": baseline,
            "subjectid": np.repeat(np.arange(n // 2), 2),
        }
    )
    frame["outcome"] = frame["outcome"].astype("boolean")

    result = fit_modified_poisson(
        frame,
        formula="outcome ~ exposure + baseline",
        term="exposure",
        cluster_column="subjectid",
    )

    assert result["n_cases"] == n
    assert result["n_subjects"] == n // 2
    assert result["events"] == int(frame["outcome"].sum())
    assert result["n_parameters"] == 3
    assert result["relative_risk"] > 1.0
    assert result["ci_low"] < result["relative_risk"] < result["ci_high"]
    assert 0.0 <= result["p_value"] <= 1.0


def test_hidden_presence_fails_closed_with_complete_separation():
    frame = pd.DataFrame(
        {
            "hidden_auc": [0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0],
            "aki": [False, False, False, True, False, True, False],
        }
    )

    summary = summarize_hidden_presence(frame).iloc[0]

    assert summary["absent_n"] == 3
    assert summary["absent_events"] == 0
    assert summary["present_n"] == 4
    assert summary["present_events"] == 2
    assert np.isinf(summary["crude_odds_ratio"])
    assert 0.0 <= summary["fisher_exact_p_value"] <= 1.0
    assert summary["estimability_status"] == "NOT_ESTIMABLE_COMPLETE_SEPARATION"
    assert "zero outcome cell" in summary["estimability_reason"]


def test_hidden_presence_is_estimable_only_with_all_four_cells_observed():
    frame = pd.DataFrame(
        {
            "hidden_auc": [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0],
            "aki": [False, True, False, False, True, False, True, False],
        }
    )

    summary = summarize_hidden_presence(frame).iloc[0]

    assert summary["absent_events"] == 1
    assert summary["absent_non_events"] == 3
    assert summary["present_events"] == 2
    assert summary["present_non_events"] == 2
    assert np.isfinite(summary["crude_odds_ratio"])
    assert summary["estimability_status"] == "ESTIMABLE"
    assert summary["estimability_reason"] == "all four exposure-outcome cells observed"
