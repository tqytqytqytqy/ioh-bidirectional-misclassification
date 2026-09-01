from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ioh.estimands.decomposition import decompose_deficit, emulate_last_visible
from ioh.estimands.episodes import detect_reference_episodes


def _duration_bin(duration_sec: float) -> str:
    if duration_sec < 180:
        return "1-<3 min"
    if duration_sec < 300:
        return "3-<5 min"
    return ">=5 min"


def _nadir_bin(nadir: float) -> str:
    if nadir < 55:
        return "<55 mmHg"
    if nadir < 60:
        return "55-<60 mmHg"
    if nadir < 65:
        return "60-<65 mmHg"
    return ">=65 mmHg"


def _reference_episode_auc(
    reference: np.ndarray,
    *,
    start_index: int,
    end_index: int,
    threshold: float,
    step_sec: float,
) -> float:
    values = reference[int(start_index) : int(end_index)]
    deficit = np.maximum(float(threshold) - values, 0.0)
    return float(np.nansum(deficit) * float(step_sec) / 60.0)


def phase_averaged_initial_display_decomposition(
    times,
    reference,
    *,
    threshold,
    interval_sec,
    step_sec,
) -> pd.DataFrame:
    times_arr = np.asarray(times, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    if times_arr.shape != reference_arr.shape:
        raise ValueError("times and reference must have the same shape")
    if interval_sec <= 0 or step_sec <= 0:
        raise ValueError("interval_sec and step_sec must be positive")

    offsets = np.arange(0, float(interval_sec), float(step_sec), dtype=float)
    rows: list[dict] = []
    dt_min = float(step_sec) / 60.0
    policies = (
        "unavailable_before_first_scheduled_sample",
        "baseline_initialized_at_first_reference",
        "exclude_before_first_display",
    )

    for offset_sec in offsets:
        primary_display = emulate_last_visible(
            times_arr,
            reference_arr,
            interval_sec=float(interval_sec),
            offset_sec=float(offset_sec),
        )
        initialized_display = emulate_last_visible(
            times_arr,
            reference_arr,
            interval_sec=float(interval_sec),
            offset_sec=float(offset_sec),
            initialize_at_start=True,
        )
        excluded_reference = reference_arr.copy()
        excluded = np.isfinite(excluded_reference) & ~np.isfinite(primary_display)
        excluded_reference[excluded] = np.nan

        policy_inputs = {
            policies[0]: (reference_arr, primary_display, 0.0),
            policies[1]: (reference_arr, initialized_display, 0.0),
            policies[2]: (
                excluded_reference,
                primary_display,
                float(excluded.sum()) * dt_min,
            ),
        }
        for policy, (policy_reference, policy_display, excluded_min) in policy_inputs.items():
            metrics = decompose_deficit(
                policy_reference,
                policy_display,
                threshold=float(threshold),
                dt_min=dt_min,
            )
            rows.append(
                {
                    "initial_display_policy": policy,
                    "offset_sec": float(offset_sec),
                    "excluded_reference_time_min": excluded_min,
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows)
    mean_columns = [
        column
        for column in frame.columns
        if column not in {"initial_display_policy", "offset_sec", "threshold"}
    ]
    output = (
        frame.groupby("initial_display_policy", sort=False)[mean_columns]
        .mean()
        .reset_index()
    )
    output.insert(1, "threshold", float(threshold))
    output.insert(2, "interval_min", float(interval_sec) / 60.0)
    output.insert(3, "phase_count", int(len(offsets)))
    output["anesthesia_hours"] = len(times_arr) * float(step_sec) / 3600.0
    output["episode_count"] = 0.0
    output["episode_detected"] = 0.0
    return output


def select_nibp_candidates(candidates, sample_cases):
    ordered = candidates.sort_values("case_id").copy()
    if sample_cases is None or str(sample_cases).strip().lower() == "all":
        selected = ordered
        method = "all eligible candidates"
    else:
        limit = int(sample_cases)
        if limit <= 0:
            raise ValueError("sample_cases must be 'all' or a positive integer")
        selected = ordered.head(limit).copy()
        method = f"first {limit} eligible candidates after ascending case_id ordering"
    return selected, {
        "candidate_cases": int(len(ordered)),
        "selected_cases": int(len(selected)),
        "selection_method": method,
    }


def frequency_offset_to_case_episode_rows(offset_rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "case_id",
        "threshold",
        "interval_min",
        "offset_sec",
        "episode_count",
        "episode_detected",
        "mean_detection_delay_min",
    }
    missing = sorted(required - set(offset_rows.columns))
    if missing:
        raise ValueError(f"frequency offset rows are missing columns: {missing}")

    output: list[dict] = []
    group_columns = ["case_id", "threshold", "interval_min"]
    for group_key, sub in offset_rows.groupby(group_columns, sort=False, dropna=False):
        episode_counts = pd.to_numeric(sub["episode_count"], errors="coerce")
        if episode_counts.dropna().nunique() > 1:
            raise ValueError(
                "episode_count must be invariant across offsets within a case, threshold, and interval"
            )
        if sub["offset_sec"].duplicated().any():
            raise ValueError(
                "offset_sec must be unique within a case, threshold, and interval"
            )
        reference_episodes = int(episode_counts.dropna().iloc[0])
        phase_count = int(sub["offset_sec"].nunique())
        detected = pd.to_numeric(sub["episode_detected"], errors="coerce").fillna(0.0)
        delay = pd.to_numeric(
            sub["mean_detection_delay_min"], errors="coerce"
        )
        if bool(((detected > 0) & delay.isna()).any()):
            raise ValueError(
                "mean_detection_delay_min is required when an offset detects episodes"
            )
        detected_pairs = float(detected.sum())
        expected_detected = detected_pairs / phase_count if phase_count else np.nan
        output.append(
            {
                "case_id": group_key[0],
                "threshold": float(group_key[1]),
                "interval_min": float(group_key[2]),
                "reference_episodes": reference_episodes,
                "phase_count": phase_count,
                "phase_episode_pairs": reference_episodes * phase_count,
                "detected_phase_episode_pairs": detected_pairs,
                "expected_detected_episodes": expected_detected,
                "expected_missed_episodes": (
                    reference_episodes - expected_detected
                    if phase_count
                    else np.nan
                ),
                "detection_observations": detected_pairs,
                "detection_delay_sum_sec": float(
                    (delay.fillna(0.0) * detected * 60.0).sum()
                ),
                "remaining_reference_time_sum_sec": np.nan,
                "stale_display_after_recovery_sum_sec": np.nan,
                "detected_with_60s_remaining_pairs": np.nan,
                "detected_with_120s_remaining_pairs": np.nan,
                "reference_episode_auc_phase_sum": np.nan,
                "pre_detection_reference_auc_sum": np.nan,
            }
        )
    return pd.DataFrame(output)


def phase_averaged_episode_observability(
    times,
    reference,
    *,
    threshold,
    interval_sec,
    min_duration_sec,
    step_sec,
    initialize_at_start=False,
):
    times_arr = np.asarray(times, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    if times_arr.shape != reference_arr.shape:
        raise ValueError("times and reference must have the same shape")
    if interval_sec <= 0 or step_sec <= 0:
        raise ValueError("interval_sec and step_sec must be positive")

    episodes = detect_reference_episodes(
        times_arr,
        reference_arr,
        threshold=float(threshold),
        min_duration_sec=float(min_duration_sec),
        gap_tolerance_sec=0,
    )
    offsets = np.arange(0, float(interval_sec), float(step_sec), dtype=float)
    detected_count = 0
    detection_delays: list[float] = []
    remaining_times: list[float] = []
    stale_times: list[float] = []
    detected_with_60s_remaining = 0
    detected_with_120s_remaining = 0
    reference_episode_auc_phase_sum = 0.0
    pre_detection_reference_auc_sum = 0.0

    episode_aucs = [
        _reference_episode_auc(
            reference_arr,
            start_index=episode.start_index,
            end_index=episode.end_index,
            threshold=float(threshold),
            step_sec=float(step_sec),
        )
        for episode in episodes
    ]

    for offset_sec in offsets:
        display = emulate_last_visible(
            times_arr,
            reference_arr,
            interval_sec=float(interval_sec),
            offset_sec=float(offset_sec),
            initialize_at_start=bool(initialize_at_start),
        )
        for episode_index, episode in enumerate(episodes):
            episode_auc = episode_aucs[episode_index]
            reference_episode_auc_phase_sum += episode_auc
            in_episode = (
                (times_arr >= episode.start_sec)
                & (times_arr <= episode.end_sec)
                & np.isfinite(display)
                & (display < float(threshold))
            )
            detected_indices = np.flatnonzero(in_episode)
            if len(detected_indices):
                detected_count += 1
                detection_index = int(detected_indices[0])
                detection_time = float(times_arr[detection_index])
                detection_delays.append(detection_time - episode.start_sec)
                recovery_time = episode.start_sec + episode.duration_sec
                remaining_time = max(0.0, recovery_time - detection_time)
                remaining_times.append(remaining_time)
                detected_with_60s_remaining += int(remaining_time >= 60.0)
                detected_with_120s_remaining += int(remaining_time >= 120.0)
                pre_detection_reference_auc_sum += _reference_episode_auc(
                    reference_arr,
                    start_index=episode.start_index,
                    end_index=detection_index,
                    threshold=float(threshold),
                    step_sec=float(step_sec),
                )
            else:
                pre_detection_reference_auc_sum += episode_auc
                continue

            stale_steps = 0
            recovery_index = int(np.searchsorted(times_arr, recovery_time, side="left"))
            for index in range(recovery_index, len(times_arr)):
                if not np.isfinite(reference_arr[index]) or reference_arr[index] < float(threshold):
                    break
                if not np.isfinite(display[index]) or display[index] >= float(threshold):
                    break
                stale_steps += 1
            stale_times.append(float(stale_steps) * float(step_sec))

    denominator = len(episodes) * len(offsets)
    expected_detected = detected_count / len(offsets) if len(offsets) else np.nan
    expected_missed = (
        len(episodes) - expected_detected if len(offsets) else np.nan
    )
    return {
        "reference_episodes": int(len(episodes)),
        "phase_count": int(len(offsets)),
        "phase_episode_pairs": int(denominator),
        "detected_phase_episode_pairs": int(detected_count),
        "detected_with_60s_remaining_pairs": int(detected_with_60s_remaining),
        "detected_with_120s_remaining_pairs": int(detected_with_120s_remaining),
        "reference_episode_auc_phase_sum": float(reference_episode_auc_phase_sum),
        "pre_detection_reference_auc_sum": float(pre_detection_reference_auc_sum),
        "detection_delay_sum_sec": float(np.sum(detection_delays)),
        "remaining_reference_time_sum_sec": float(np.sum(remaining_times)),
        "stale_display_after_recovery_sum_sec": float(np.sum(stale_times)),
        "expected_detected_episodes": float(expected_detected),
        "expected_missed_episodes": float(expected_missed),
        "detection_observations": int(detected_count),
        "complete_miss_fraction": (
            float(1.0 - detected_count / denominator) if denominator else np.nan
        ),
        "mean_detection_delay_sec": (
            float(np.mean(detection_delays)) if detection_delays else np.nan
        ),
        "mean_remaining_reference_time_sec": (
            float(np.mean(remaining_times)) if remaining_times else np.nan
        ),
        "mean_stale_display_after_recovery_sec": (
            float(np.mean(stale_times)) if stale_times else np.nan
        ),
        "detection_with_1min_remaining_probability": (
            float(detected_with_60s_remaining / denominator)
            if denominator
            else np.nan
        ),
        "detection_with_2min_remaining_probability": (
            float(detected_with_120s_remaining / denominator)
            if denominator
            else np.nan
        ),
        "reference_auc_before_first_low_display_fraction": (
            float(
                pre_detection_reference_auc_sum
                / reference_episode_auc_phase_sum
            )
            if reference_episode_auc_phase_sum > 0
            else np.nan
        ),
    }


def case_phase_averaged_episode_rows(
    *,
    case_id,
    times,
    reference,
    thresholds,
    intervals_min,
    min_duration_sec,
    step_sec,
) -> pd.DataFrame:
    times_arr = np.asarray(times, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    rows: list[dict] = []

    threshold_values = [float(value) for value in thresholds]
    interval_values = [float(value) for value in intervals_min]
    episodes_by_threshold: dict[float, list] = {}
    episode_keys: dict[float, list[tuple[str, str]]] = {}
    episode_aucs: dict[float, list[float]] = {}
    for threshold in threshold_values:
        episodes_by_threshold[threshold] = detect_reference_episodes(
            times_arr,
            reference_arr,
            threshold=threshold,
            min_duration_sec=float(min_duration_sec),
            gap_tolerance_sec=0,
        )
        episode_keys[threshold] = [
            (
                _duration_bin(float(episode.duration_sec)),
                _nadir_bin(
                    float(
                        np.nanmin(
                            reference_arr[episode.start_index : episode.end_index]
                        )
                    )
                ),
            )
            for episode in episodes_by_threshold[threshold]
        ]
        episode_aucs[threshold] = [
            _reference_episode_auc(
                reference_arr,
                start_index=episode.start_index,
                end_index=episode.end_index,
                threshold=threshold,
                step_sec=float(step_sec),
            )
            for episode in episodes_by_threshold[threshold]
        ]

    for interval_min in interval_values:
        interval_sec = int(round(interval_min * 60.0))
        offsets = np.arange(0, interval_sec, int(step_sec), dtype=float)
        grouped_by_threshold: dict[float, dict[tuple[str, str], dict]] = {}
        for threshold in threshold_values:
            episodes = episodes_by_threshold[threshold]
            if not episodes:
                continue
            grouped: dict[tuple[str, str], dict] = {}
            for key in episode_keys[threshold]:
                if key not in grouped:
                    grouped[key] = {
                        "case_id": case_id,
                        "threshold": threshold,
                        "interval_min": interval_min,
                        "duration_bin": key[0],
                        "nadir_bin": key[1],
                        "reference_episodes": 0,
                        "phase_count": int(len(offsets)),
                        "phase_episode_pairs": 0,
                        "detected_phase_episode_pairs": 0,
                        "detection_delay_sum_sec": 0.0,
                        "remaining_reference_time_sum_sec": 0.0,
                        "stale_display_after_recovery_sum_sec": 0.0,
                        "detected_with_60s_remaining_pairs": 0,
                        "detected_with_120s_remaining_pairs": 0,
                        "reference_episode_auc_phase_sum": 0.0,
                        "pre_detection_reference_auc_sum": 0.0,
                    }
                grouped[key]["reference_episodes"] += 1
            grouped_by_threshold[threshold] = grouped

        if not grouped_by_threshold:
            continue

        for offset_sec in offsets:
            display = emulate_last_visible(
                times_arr,
                reference_arr,
                interval_sec=interval_sec,
                offset_sec=float(offset_sec),
            )
            for threshold, grouped in grouped_by_threshold.items():
                episodes = episodes_by_threshold[threshold]
                for episode_index, episode in enumerate(episodes):
                    key = episode_keys[threshold][episode_index]
                    target = grouped[key]
                    target["phase_episode_pairs"] += 1
                    episode_auc = episode_aucs[threshold][episode_index]
                    target["reference_episode_auc_phase_sum"] += episode_auc
                    in_episode = (
                        (times_arr >= episode.start_sec)
                        & (times_arr <= episode.end_sec)
                        & np.isfinite(display)
                        & (display < threshold)
                    )
                    detected_indices = np.flatnonzero(in_episode)
                    if not len(detected_indices):
                        target["pre_detection_reference_auc_sum"] += episode_auc
                        continue
                    target["detected_phase_episode_pairs"] += 1
                    detection_index = int(detected_indices[0])
                    detection_time = float(times_arr[detection_index])
                    recovery_time = episode.start_sec + episode.duration_sec
                    target["detection_delay_sum_sec"] += detection_time - episode.start_sec
                    remaining_time = max(0.0, recovery_time - detection_time)
                    target["remaining_reference_time_sum_sec"] += remaining_time
                    target["detected_with_60s_remaining_pairs"] += int(
                        remaining_time >= 60.0
                    )
                    target["detected_with_120s_remaining_pairs"] += int(
                        remaining_time >= 120.0
                    )
                    target["pre_detection_reference_auc_sum"] += (
                        _reference_episode_auc(
                            reference_arr,
                            start_index=episode.start_index,
                            end_index=detection_index,
                            threshold=threshold,
                            step_sec=float(step_sec),
                        )
                    )
                    stale_steps = 0
                    recovery_index = int(
                        np.searchsorted(times_arr, recovery_time, side="left")
                    )
                    for index in range(recovery_index, len(times_arr)):
                        if (
                            not np.isfinite(reference_arr[index])
                            or reference_arr[index] < threshold
                            or not np.isfinite(display[index])
                            or display[index] >= threshold
                        ):
                            break
                        stale_steps += 1
                    target["stale_display_after_recovery_sum_sec"] += (
                        stale_steps * float(step_sec)
                    )

        for grouped in grouped_by_threshold.values():
            for target in grouped.values():
                phases = target["phase_count"]
                detected_pairs = target["detected_phase_episode_pairs"]
                target["expected_detected_episodes"] = detected_pairs / phases
                target["expected_missed_episodes"] = (
                    target["reference_episodes"]
                    - target["expected_detected_episodes"]
                )
                target["detection_observations"] = detected_pairs
                rows.append(target)

    return pd.DataFrame(rows)


def summarize_episode_observability(
    case_rows: pd.DataFrame,
    *,
    all_case_ids,
    group_columns,
    bootstrap_reps,
    seed,
    case_to_cluster: pd.DataFrame | None = None,
    cluster_column: str = "case_id",
) -> pd.DataFrame:
    case_rows = case_rows.copy()
    optional_columns = (
        "phase_episode_pairs",
        "detected_with_60s_remaining_pairs",
        "detected_with_120s_remaining_pairs",
        "reference_episode_auc_phase_sum",
        "pre_detection_reference_auc_sum",
    )
    for column in optional_columns:
        if column not in case_rows.columns:
            case_rows[column] = np.nan

    all_ids = pd.Index(pd.Series(list(all_case_ids)).drop_duplicates())
    output: list[dict] = []

    for group_key, sub in case_rows.groupby(list(group_columns), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_identity = "|".join(
            [str(int(seed))]
            + [
                f"{column}={value!r}"
                for column, value in zip(group_columns, group_key)
            ]
        )
        group_seed = int.from_bytes(
            hashlib.sha256(group_identity.encode("utf-8")).digest()[:8],
            byteorder="little",
            signed=False,
        )
        rng = np.random.default_rng(group_seed)
        by_case = sub.groupby("case_id", dropna=False).agg(
            reference_episodes=("reference_episodes", "sum"),
            expected_detected_episodes=("expected_detected_episodes", "sum"),
            expected_missed_episodes=("expected_missed_episodes", "sum"),
            phase_episode_pairs=("phase_episode_pairs", "sum"),
            detected_phase_episode_pairs=("detected_phase_episode_pairs", "sum"),
            detected_with_60s_remaining_pairs=(
                "detected_with_60s_remaining_pairs",
                "sum",
            ),
            detected_with_120s_remaining_pairs=(
                "detected_with_120s_remaining_pairs",
                "sum",
            ),
            reference_episode_auc_phase_sum=(
                "reference_episode_auc_phase_sum",
                "sum",
            ),
            pre_detection_reference_auc_sum=(
                "pre_detection_reference_auc_sum",
                "sum",
            ),
            detection_delay_sum_sec=("detection_delay_sum_sec", "sum"),
            remaining_reference_time_sum_sec=("remaining_reference_time_sum_sec", "sum"),
            stale_display_after_recovery_sum_sec=(
                "stale_display_after_recovery_sum_sec",
                "sum",
            ),
        ).reindex(all_ids, fill_value=0.0)

        case_arrays = {
            column: by_case[column].to_numpy(float) for column in by_case.columns
        }
        if case_to_cluster is None:
            by_cluster = by_case
            bootstrap_cluster = "case_id"
        else:
            required = {"case_id", cluster_column}
            missing_columns = required - set(case_to_cluster.columns)
            if missing_columns:
                raise ValueError(
                    f"case_to_cluster is missing columns: {sorted(missing_columns)}"
                )
            mapping = case_to_cluster[["case_id", cluster_column]].drop_duplicates()
            conflicting = mapping.groupby("case_id")[cluster_column].nunique(dropna=False)
            if bool(conflicting.gt(1).any()):
                raise ValueError("each case_id must map to exactly one bootstrap cluster")
            mapping = mapping.drop_duplicates("case_id").set_index("case_id")[cluster_column]
            cluster_ids = mapping.reindex(by_case.index)
            if bool(cluster_ids.isna().any()):
                raise ValueError("every analyzed case_id must have a bootstrap cluster")
            by_cluster = by_case.assign(
                _bootstrap_cluster=cluster_ids.to_numpy()
            ).groupby("_bootstrap_cluster", dropna=False).sum()
            bootstrap_cluster = cluster_column

        arrays = {
            column: by_cluster[column].to_numpy(float)
            for column in by_cluster.columns
        }
        reference_total = float(case_arrays["reference_episodes"].sum())
        detected_total = float(case_arrays["expected_detected_episodes"].sum())
        missed_total = float(case_arrays["expected_missed_episodes"].sum())
        detected_observations = float(
            case_arrays["detected_phase_episode_pairs"].sum()
        )

        sampled = rng.integers(
            0,
            len(by_cluster),
            size=(int(bootstrap_reps), len(by_cluster)),
        )
        boot_reference = arrays["reference_episodes"][sampled].sum(axis=1)
        boot_detected = arrays["expected_detected_episodes"][sampled].sum(axis=1)
        boot_missed = arrays["expected_missed_episodes"][sampled].sum(axis=1)
        boot_phase_episode_pairs = arrays["phase_episode_pairs"][sampled].sum(axis=1)
        boot_detection_observations = arrays["detected_phase_episode_pairs"][sampled].sum(axis=1)

        def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            return np.divide(
                numerator,
                denominator,
                out=np.full(numerator.shape, np.nan, dtype=float),
                where=denominator > 0,
            )

        def interval(values: np.ndarray) -> tuple[float, float]:
            finite = values[np.isfinite(values)]
            if not len(finite):
                return np.nan, np.nan
            return (
                float(np.quantile(finite, 0.025)),
                float(np.quantile(finite, 0.975)),
            )

        boot_detection_probability = ratio(boot_detected, boot_reference)
        boot_miss_probability = ratio(boot_missed, boot_reference)
        detection_ci = interval(boot_detection_probability)
        miss_ci = interval(boot_miss_probability)

        actionability_metrics: dict[str, float] = {}
        for metric, numerator_column in (
            (
                "detection_with_1min_remaining_probability",
                "detected_with_60s_remaining_pairs",
            ),
            (
                "detection_with_2min_remaining_probability",
                "detected_with_120s_remaining_pairs",
            ),
        ):
            available = bool(
                pd.to_numeric(sub[numerator_column], errors="coerce").notna().any()
                and pd.to_numeric(sub["phase_episode_pairs"], errors="coerce")
                .notna()
                .any()
            )
            if not available:
                actionability_metrics[metric] = np.nan
                actionability_metrics[f"{metric}_ci_low"] = np.nan
                actionability_metrics[f"{metric}_ci_high"] = np.nan
                continue
            numerator_total = float(case_arrays[numerator_column].sum())
            denominator_total = float(case_arrays["phase_episode_pairs"].sum())
            point = (
                numerator_total / denominator_total
                if denominator_total > 0
                else np.nan
            )
            boot_numerator = arrays[numerator_column][sampled].sum(axis=1)
            boot_value = ratio(boot_numerator, boot_phase_episode_pairs)
            ci_low, ci_high = interval(boot_value)
            actionability_metrics[metric] = point
            actionability_metrics[f"{metric}_ci_low"] = ci_low
            actionability_metrics[f"{metric}_ci_high"] = ci_high

        pre_detection_metric = "reference_auc_before_first_low_display_fraction"
        pre_detection_available = bool(
            pd.to_numeric(
                sub["pre_detection_reference_auc_sum"], errors="coerce"
            ).notna().any()
            and pd.to_numeric(
                sub["reference_episode_auc_phase_sum"], errors="coerce"
            ).notna().any()
        )
        if pre_detection_available:
            pre_detection_total = float(
                case_arrays["pre_detection_reference_auc_sum"].sum()
            )
            reference_auc_total = float(
                case_arrays["reference_episode_auc_phase_sum"].sum()
            )
            pre_detection_point = (
                pre_detection_total / reference_auc_total
                if reference_auc_total > 0
                else np.nan
            )
            boot_pre_detection = arrays[
                "pre_detection_reference_auc_sum"
            ][sampled].sum(axis=1)
            boot_reference_auc = arrays[
                "reference_episode_auc_phase_sum"
            ][sampled].sum(axis=1)
            pre_detection_ci = interval(
                ratio(boot_pre_detection, boot_reference_auc)
            )
        else:
            pre_detection_point = np.nan
            pre_detection_ci = (np.nan, np.nan)

        metric_specs = {
            "mean_detection_delay_min": "detection_delay_sum_sec",
            "mean_remaining_reference_time_min": "remaining_reference_time_sum_sec",
            "mean_stale_display_after_recovery_min": "stale_display_after_recovery_sum_sec",
        }
        metrics: dict[str, float] = {}
        for metric, numerator_column in metric_specs.items():
            metric_available = bool(
                pd.to_numeric(sub[numerator_column], errors="coerce").notna().any()
            )
            if not metric_available:
                metrics[metric] = np.nan
                metrics[f"{metric}_ci_low"] = np.nan
                metrics[f"{metric}_ci_high"] = np.nan
                continue
            numerator_total = float(case_arrays[numerator_column].sum())
            point = (
                numerator_total / detected_observations / 60.0
                if detected_observations > 0
                else np.nan
            )
            boot_numerator = arrays[numerator_column][sampled].sum(axis=1)
            boot_value = ratio(boot_numerator, boot_detection_observations) / 60.0
            ci_low, ci_high = interval(boot_value)
            metrics[metric] = point
            metrics[f"{metric}_ci_low"] = ci_low
            metrics[f"{metric}_ci_high"] = ci_high

        row = {column: value for column, value in zip(group_columns, group_key)}
        row.update(
            {
                "n_cases_total": int(len(all_ids)),
                "n_clusters_total": int(len(by_cluster)),
                "bootstrap_cluster": bootstrap_cluster,
                "bootstrap_reps": int(bootstrap_reps),
                "cases_with_episodes": int(
                    (case_arrays["reference_episodes"] > 0).sum()
                ),
                "reference_episodes": reference_total,
                "expected_detected_episodes": detected_total,
                "expected_missed_episodes": missed_total,
                "episode_detection_probability": (
                    detected_total / reference_total if reference_total > 0 else np.nan
                ),
                "episode_detection_ci_low": detection_ci[0],
                "episode_detection_ci_high": detection_ci[1],
                "complete_miss_probability": (
                    missed_total / reference_total if reference_total > 0 else np.nan
                ),
                "complete_miss_ci_low": miss_ci[0],
                "complete_miss_ci_high": miss_ci[1],
                **actionability_metrics,
                pre_detection_metric: pre_detection_point,
                f"{pre_detection_metric}_ci_low": pre_detection_ci[0],
                f"{pre_detection_metric}_ci_high": pre_detection_ci[1],
                **metrics,
            }
        )
        output.append(row)

    return pd.DataFrame(output)
