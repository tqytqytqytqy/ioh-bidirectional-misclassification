import numpy as np
import pandas as pd
import ioh.revision_v7 as revision_v7

from ioh.revision_v7 import (
    frequency_offset_to_case_episode_rows,
    phase_averaged_initial_display_decomposition,
    phase_averaged_episode_observability,
    select_nibp_candidates,
)


def test_select_nibp_candidates_all_keeps_entire_ordered_pool():
    candidates = pd.DataFrame({"case_id": [4, 2, 9, 1]})

    selected, metadata = select_nibp_candidates(candidates, "all")

    assert selected["case_id"].tolist() == [1, 2, 4, 9]
    assert metadata["candidate_cases"] == 4
    assert metadata["selected_cases"] == 4
    assert metadata["selection_method"] == "all eligible candidates"


def test_phase_averaged_observability_quantifies_detection_delay_and_remaining_time():
    times = np.arange(0, 600, 10, dtype=float)
    reference = np.full(times.shape, 75.0)
    reference[(times >= 100) & (times < 220)] = 60.0

    result = phase_averaged_episode_observability(
        times,
        reference,
        threshold=65.0,
        interval_sec=300,
        min_duration_sec=60,
        step_sec=10,
    )

    assert result["reference_episodes"] == 1
    assert result["phase_count"] == 30
    assert np.isclose(result["expected_detected_episodes"], 0.4)
    assert np.isclose(result["complete_miss_fraction"], 0.6)
    assert np.isclose(result["mean_detection_delay_sec"], 55.0)
    assert np.isclose(result["mean_remaining_reference_time_sec"], 65.0)
    assert np.isclose(result["mean_stale_display_after_recovery_sec"], 235.0)
    assert np.isclose(
        result["detection_with_1min_remaining_probability"], 7 / 30
    )
    assert np.isclose(
        result["detection_with_2min_remaining_probability"], 1 / 30
    )
    assert np.isclose(
        result["reference_auc_before_first_low_display_fraction"], 47 / 60
    )


def test_phase_averaged_observability_can_initialize_the_display_at_analytic_start():
    times = np.arange(0, 60, 10, dtype=float)
    reference = np.array([60.0, 60.0, 70.0, 70.0, 70.0, 70.0])

    unavailable = phase_averaged_episode_observability(
        times,
        reference,
        threshold=65.0,
        interval_sec=30,
        min_duration_sec=20,
        step_sec=10,
    )
    initialized = phase_averaged_episode_observability(
        times,
        reference,
        threshold=65.0,
        interval_sec=30,
        min_duration_sec=20,
        step_sec=10,
        initialize_at_start=True,
    )

    assert initialized["expected_detected_episodes"] > unavailable[
        "expected_detected_episodes"
    ]
    assert initialized["detected_phase_episode_pairs"] == 3
    assert initialized["phase_episode_pairs"] == 3


def test_initial_display_policy_sensitivity_separates_unavailable_from_valid_mismatch():
    times = np.arange(0, 60, 10, dtype=float)
    reference = np.array([60.0, 60.0, 70.0, 70.0, 70.0, 70.0])

    result = phase_averaged_initial_display_decomposition(
        times,
        reference,
        threshold=65.0,
        interval_sec=30,
        step_sec=10,
    ).set_index("initial_display_policy")

    assert set(result.index) == {
        "unavailable_before_first_scheduled_sample",
        "baseline_initialized_at_first_reference",
        "exclude_before_first_display",
    }
    assert np.isclose(
        result.loc["unavailable_before_first_scheduled_sample", "hidden_auc"],
        5 / 6,
    )
    assert np.isclose(
        result.loc["baseline_initialized_at_first_reference", "hidden_auc"],
        0.0,
    )
    assert np.isclose(
        result.loc["exclude_before_first_display", "true_auc"],
        5 / 6,
    )
    assert np.isclose(
        result.loc["exclude_before_first_display", "excluded_reference_time_min"],
        1 / 6,
    )


def test_case_phase_rows_preserve_sufficient_statistics_for_cluster_bootstrap():
    times = np.arange(0, 600, 10, dtype=float)
    reference = np.full(times.shape, 75.0)
    reference[(times >= 100) & (times < 220)] = 60.0

    rows = revision_v7.case_phase_averaged_episode_rows(
        case_id=7,
        times=times,
        reference=reference,
        thresholds=[65.0],
        intervals_min=[5.0],
        min_duration_sec=60,
        step_sec=10,
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["case_id"] == 7
    assert row["duration_bin"] == "1-<3 min"
    assert row["nadir_bin"] == "60-<65 mmHg"
    assert row["reference_episodes"] == 1
    assert row["phase_episode_pairs"] == 30
    assert row["detected_phase_episode_pairs"] == 12
    assert np.isclose(row["expected_detected_episodes"], 0.4)
    assert np.isclose(row["detection_delay_sum_sec"], 12 * 55.0)
    assert row["detected_with_60s_remaining_pairs"] == 7
    assert row["detected_with_120s_remaining_pairs"] == 1
    assert np.isclose(row["reference_episode_auc_phase_sum"], 300.0)
    assert np.isclose(row["pre_detection_reference_auc_sum"], 235.0)


def test_cluster_summary_reports_actionable_windows_and_pre_detection_auc():
    case_rows = pd.DataFrame(
        [
            {
                "case_id": 1,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1.0,
                "phase_episode_pairs": 10.0,
                "expected_detected_episodes": 0.8,
                "expected_missed_episodes": 0.2,
                "detected_phase_episode_pairs": 8.0,
                "detected_with_60s_remaining_pairs": 6.0,
                "detected_with_120s_remaining_pairs": 4.0,
                "reference_episode_auc_phase_sum": 100.0,
                "pre_detection_reference_auc_sum": 20.0,
                "detection_delay_sum_sec": 80.0,
                "remaining_reference_time_sum_sec": 400.0,
                "stale_display_after_recovery_sum_sec": 160.0,
            },
            {
                "case_id": 2,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1.0,
                "phase_episode_pairs": 10.0,
                "expected_detected_episodes": 0.0,
                "expected_missed_episodes": 1.0,
                "detected_phase_episode_pairs": 0.0,
                "detected_with_60s_remaining_pairs": 0.0,
                "detected_with_120s_remaining_pairs": 0.0,
                "reference_episode_auc_phase_sum": 50.0,
                "pre_detection_reference_auc_sum": 50.0,
                "detection_delay_sum_sec": 0.0,
                "remaining_reference_time_sum_sec": 0.0,
                "stale_display_after_recovery_sum_sec": 0.0,
            },
        ]
    )

    summary = revision_v7.summarize_episode_observability(
        case_rows,
        all_case_ids=[1, 2],
        group_columns=["threshold", "interval_min"],
        bootstrap_reps=1000,
        seed=17,
    )

    row = summary.iloc[0]
    assert np.isclose(row["detection_with_1min_remaining_probability"], 0.3)
    assert np.isclose(row["detection_with_2min_remaining_probability"], 0.2)
    assert np.isclose(
        row["reference_auc_before_first_low_display_fraction"], 70 / 150
    )
    for metric in (
        "detection_with_1min_remaining_probability",
        "detection_with_2min_remaining_probability",
        "reference_auc_before_first_low_display_fraction",
    ):
        assert row[f"{metric}_ci_low"] <= row[metric] <= row[f"{metric}_ci_high"]


def test_cluster_summary_reports_absolute_counts_and_uncertainty():
    case_rows = pd.DataFrame(
        [
            {
                "case_id": 1,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1,
                "expected_detected_episodes": 1.0,
                "expected_missed_episodes": 0.0,
                "detected_phase_episode_pairs": 1,
                "detection_delay_sum_sec": 10.0,
                "remaining_reference_time_sum_sec": 50.0,
                "stale_display_after_recovery_sum_sec": 20.0,
            },
            {
                "case_id": 2,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1,
                "expected_detected_episodes": 0.0,
                "expected_missed_episodes": 1.0,
                "detected_phase_episode_pairs": 0,
                "detection_delay_sum_sec": 0.0,
                "remaining_reference_time_sum_sec": 0.0,
                "stale_display_after_recovery_sum_sec": 0.0,
            },
        ]
    )

    summary = revision_v7.summarize_episode_observability(
        case_rows,
        all_case_ids=[1, 2],
        group_columns=["threshold", "interval_min"],
        bootstrap_reps=1000,
        seed=17,
    )

    row = summary.iloc[0]
    assert row["n_cases_total"] == 2
    assert row["cases_with_episodes"] == 2
    assert row["reference_episodes"] == 2
    assert np.isclose(row["episode_detection_probability"], 0.5)
    assert np.isclose(row["complete_miss_probability"], 0.5)
    assert row["episode_detection_ci_low"] == 0.0
    assert row["episode_detection_ci_high"] == 1.0


def test_frequency_offsets_are_reduced_to_phase_averaged_case_statistics():
    offset_rows = pd.DataFrame(
        [
            {
                "case_id": 11,
                "threshold": 65.0,
                "interval_min": 5.0,
                "offset_sec": 0.0,
                "episode_count": 2,
                "episode_detected": 1,
                "missed_episodes": 1,
                "mean_detection_delay_min": 0.5,
            },
            {
                "case_id": 11,
                "threshold": 65.0,
                "interval_min": 5.0,
                "offset_sec": 10.0,
                "episode_count": 2,
                "episode_detected": 2,
                "missed_episodes": 0,
                "mean_detection_delay_min": 1.0,
            },
        ]
    )

    rows = frequency_offset_to_case_episode_rows(offset_rows)

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["reference_episodes"] == 2
    assert row["phase_count"] == 2
    assert row["phase_episode_pairs"] == 4
    assert row["detected_phase_episode_pairs"] == 3
    assert np.isclose(row["expected_detected_episodes"], 1.5)
    assert np.isclose(row["expected_missed_episodes"], 0.5)
    assert np.isclose(row["detection_delay_sum_sec"], 150.0)
    assert np.isnan(row["remaining_reference_time_sum_sec"])


def test_cluster_summary_keeps_unavailable_timing_metrics_missing():
    case_rows = pd.DataFrame(
        [
            {
                "case_id": 1,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1,
                "expected_detected_episodes": 1.0,
                "expected_missed_episodes": 0.0,
                "detected_phase_episode_pairs": 1,
                "detection_delay_sum_sec": 10.0,
                "remaining_reference_time_sum_sec": np.nan,
                "stale_display_after_recovery_sum_sec": np.nan,
            }
        ]
    )

    summary = revision_v7.summarize_episode_observability(
        case_rows,
        all_case_ids=[1],
        group_columns=["threshold", "interval_min"],
        bootstrap_reps=10,
        seed=17,
    )

    row = summary.iloc[0]
    assert np.isclose(row["mean_detection_delay_min"], 1 / 6)
    assert np.isnan(row["mean_remaining_reference_time_min"])
    assert np.isnan(row["mean_stale_display_after_recovery_min"])


def test_cluster_bootstrap_ci_is_invariant_to_unrelated_groups():
    rows = []
    for case_id, detected in enumerate([0.0, 0.2, 0.5, 0.8, 1.0], start=1):
        for threshold in (55.0, 65.0):
            rows.append(
                {
                    "case_id": case_id,
                    "threshold": threshold,
                    "interval_min": 5.0,
                    "reference_episodes": 1.0,
                    "expected_detected_episodes": detected,
                    "expected_missed_episodes": 1.0 - detected,
                    "detected_phase_episode_pairs": detected * 10.0,
                    "detection_delay_sum_sec": detected * 30.0,
                    "remaining_reference_time_sum_sec": np.nan,
                    "stale_display_after_recovery_sum_sec": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    common = {
        "all_case_ids": range(1, 6),
        "group_columns": ["threshold", "interval_min"],
        "bootstrap_reps": 500,
        "seed": 17,
    }

    full = revision_v7.summarize_episode_observability(frame, **common)
    isolated = revision_v7.summarize_episode_observability(
        frame[frame["threshold"].eq(65.0)], **common
    )
    full_row = full[full["threshold"].eq(65.0)].iloc[0]
    isolated_row = isolated.iloc[0]

    for column in (
        "episode_detection_ci_low",
        "episode_detection_ci_high",
        "complete_miss_ci_low",
        "complete_miss_ci_high",
        "mean_detection_delay_min_ci_low",
        "mean_detection_delay_min_ci_high",
    ):
        assert full_row[column] == isolated_row[column]


def test_episode_summary_bootstraps_subjects_while_preserving_case_weighted_point_estimate():
    case_rows = pd.DataFrame(
        [
            {
                "case_id": 1,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1.0,
                "expected_detected_episodes": 1.0,
                "expected_missed_episodes": 0.0,
                "detected_phase_episode_pairs": 1.0,
                "detection_delay_sum_sec": 10.0,
                "remaining_reference_time_sum_sec": 50.0,
                "stale_display_after_recovery_sum_sec": 20.0,
            },
            {
                "case_id": 2,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1.0,
                "expected_detected_episodes": 1.0,
                "expected_missed_episodes": 0.0,
                "detected_phase_episode_pairs": 1.0,
                "detection_delay_sum_sec": 10.0,
                "remaining_reference_time_sum_sec": 50.0,
                "stale_display_after_recovery_sum_sec": 20.0,
            },
            {
                "case_id": 3,
                "threshold": 65.0,
                "interval_min": 5.0,
                "reference_episodes": 1.0,
                "expected_detected_episodes": 0.0,
                "expected_missed_episodes": 1.0,
                "detected_phase_episode_pairs": 0.0,
                "detection_delay_sum_sec": 0.0,
                "remaining_reference_time_sum_sec": 0.0,
                "stale_display_after_recovery_sum_sec": 0.0,
            },
        ]
    )
    case_to_subject = pd.DataFrame(
        {"case_id": [1, 2, 3], "subjectid": [101, 101, 202]}
    )

    summary = revision_v7.summarize_episode_observability(
        case_rows,
        all_case_ids=[1, 2, 3],
        group_columns=["threshold", "interval_min"],
        bootstrap_reps=1000,
        seed=17,
        case_to_cluster=case_to_subject,
        cluster_column="subjectid",
    )

    row = summary.iloc[0]
    assert np.isclose(row["episode_detection_probability"], 2 / 3)
    assert row["n_cases_total"] == 3
    assert row["n_clusters_total"] == 2
    assert row["bootstrap_cluster"] == "subjectid"
