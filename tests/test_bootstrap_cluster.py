import numpy as np
import pandas as pd

from ioh.pipeline import _frequency_population_table
from ioh.stats.bootstrap import cluster_bootstrap_mean


def test_cluster_bootstrap_samples_cases_not_events():
    df = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 2],
            "bias": [10.0, 12.0, 14.0, -2.0],
        }
    )

    out = cluster_bootstrap_mean(df, cluster_col="case_id", value_col="bias", iterations=50, seed=7)

    assert out["cluster_count"] == 2
    assert out["row_count"] == 4
    assert out["point_estimate"] == 8.5
    assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]


def test_frequency_population_table_uses_subject_clusters_without_changing_case_weighted_point():
    case_mean = pd.DataFrame(
        {
            "case_id": [1, 2, 3],
            "threshold": [65.0, 65.0, 65.0],
            "interval_min": [5.0, 5.0, 5.0],
            "true_auc": [10.0, 10.0, 10.0],
            "display_auc": [0.0, 0.0, 10.0],
            "hidden_auc": [10.0, 10.0, 0.0],
            "overdisplay_auc": [0.0, 0.0, 0.0],
            "hidden_auc_display_unavailable": [0.0, 0.0, 0.0],
            "hidden_auc_display_valid": [10.0, 10.0, 0.0],
            "true_auc_display_valid": [10.0, 10.0, 10.0],
            "display_auc_display_valid": [0.0, 0.0, 10.0],
            "normotensive_display_discordance_min": [1.0, 1.0, 0.0],
            "reference_hypotension_with_display_unavailable_min": [0.0, 0.0, 0.0],
            "hypotensive_display_discordance_min": [0.0, 0.0, 0.0],
            "anesthesia_hours": [1.0, 1.0, 1.0],
            "display_valid_min": [60.0, 60.0, 60.0],
            "display_unavailable_min": [0.0, 0.0, 0.0],
            "episode_count": [1.0, 1.0, 1.0],
            "episode_detected": [0.0, 0.0, 1.0],
        }
    )
    case_to_subject = pd.DataFrame(
        {"case_id": [1, 2, 3], "subjectid": [101, 101, 202]}
    )
    cfg = {"project": {"seed": 17}, "analysis": {"bootstrap_reps": 1000}}

    result = _frequency_population_table(
        cfg,
        case_mean,
        case_to_cluster=case_to_subject,
        cluster_column="subjectid",
    )

    row = result.iloc[0]
    assert np.isclose(row["HDR"], 2 / 3)
    assert row["n_cases"] == 3
    assert row["n_clusters"] == 2
    assert row["bootstrap_cluster"] == "subjectid"


def test_frequency_population_ci_is_invariant_to_unrelated_analysis_groups():
    rows = []
    for case_id, (true_auc, hidden_auc, subjectid) in enumerate(
        [
            (7.0, 6.0, 101),
            (11.0, 1.0, 102),
            (13.0, 9.0, 103),
            (17.0, 2.0, 104),
            (19.0, 15.0, 105),
            (23.0, 4.0, 106),
            (29.0, 12.0, 107),
            (31.0, 3.0, 108),
        ],
        start=1,
    ):
        rows.append(
            {
                "case_id": case_id,
                "threshold": 65.0,
                "interval_min": 5.0,
                "true_auc": true_auc,
                "display_auc": true_auc - hidden_auc,
                "hidden_auc": hidden_auc,
                "overdisplay_auc": 0.0,
                "hidden_auc_display_unavailable": 0.0,
                "hidden_auc_display_valid": hidden_auc,
                "true_auc_display_valid": true_auc,
                "display_auc_display_valid": true_auc - hidden_auc,
                "normotensive_display_discordance_min": 0.0,
                "reference_hypotension_with_display_unavailable_min": 0.0,
                "hypotensive_display_discordance_min": 0.0,
                "anesthesia_hours": 1.0,
                "display_valid_min": 60.0,
                "display_unavailable_min": 0.0,
                "episode_count": 1.0,
                "episode_detected": 1.0,
                "subjectid": subjectid,
            }
        )
    primary = pd.DataFrame(rows)
    unrelated = primary.assign(threshold=55.0, hidden_auc=lambda x: x["hidden_auc"] / 2)
    mapping = primary[["case_id", "subjectid"]]
    cfg = {"project": {"seed": 17}, "analysis": {"bootstrap_reps": 73}}

    standalone = _frequency_population_table(
        cfg,
        primary.drop(columns="subjectid"),
        case_to_cluster=mapping,
        cluster_column="subjectid",
    )
    combined = _frequency_population_table(
        cfg,
        pd.concat([unrelated, primary], ignore_index=True).drop(columns="subjectid"),
        case_to_cluster=mapping,
        cluster_column="subjectid",
    )

    standalone_row = standalone.loc[standalone["threshold"].eq(65.0)].iloc[0]
    combined_row = combined.loc[combined["threshold"].eq(65.0)].iloc[0]
    for column in ["HDR_ci_low", "HDR_ci_high", "NetBias_ci_low", "NetBias_ci_high"]:
        assert standalone_row[column] == combined_row[column]
