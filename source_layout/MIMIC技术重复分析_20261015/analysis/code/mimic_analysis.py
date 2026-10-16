"""Bounded phase-based technical replication; consumes frozen private QC arrays only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_writebytecode = True
import numpy as np
import pandas as pd

from original_decomposition import decompose_deficit
from original_events import held_display, reference_events, classify_events

SEED = 20260706
REPLICATES = 1000
CONTRACT_SHA256 = "7f44211c428cd6c54835c4395fc29fb9ae0bfe4c7756aacfb2dfd582ce48c09b"
THRESHOLDS = (65., 60., 55.)
INTERVALS = (1., 2.5, 3., 5., 10.)
SCENARIOS = {
    "primary": ("primary", "primary", "primary"),
    "baseline_initialization": ("primary", "primary", "initialize"),
    "exclude_predisplay": ("primary", "primary", "exclude_predisplay"),
    "exclude_first5min": ("primary", "primary", "exclude_first5"),
    "strict_pulsatility_fixed_primary": ("primary", "strict", "primary"),
    "numeric_coherence_fixed_primary": ("primary", "coherent", "primary"),
    "strict_pulsatility_eligible80": ("strict", "strict", "primary"),
    "numeric_coherence_eligible80": ("coherent", "coherent", "primary"),
    "earliest_eligible_hour": ("window", "primary", "primary"),
}
KEYS = ["scenario", "threshold", "interval_min"]
EVENT_KEYS = KEYS + ["gap_sec", "minimum_low_sec"]
FREQUENCY_COLUMNS = [
    "true_auc", "display_auc", "hidden_auc", "overdisplay_auc", "concordant_auc",
    "hidden_auc_display_unavailable", "hidden_auc_display_valid", "true_auc_display_valid",
    "display_auc_display_valid", "normotensive_display_discordance_min",
    "hypotensive_display_discordance_min", "reference_hypotension_with_display_unavailable_min",
    "true_hypotension_min", "display_low_min", "valid_reference_min", "display_valid_min",
    "display_unavailable_min", "original_record_hours", "analyzed_grid_hours",
]
EVENT_COLUMNS = ["episodes", "fresh", "inherited_only", "never_low", "carry_in", "first_inherited",
                 "detected", "detection_delay_sum_min"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data, private=False):
    Path(path).write_text(json.dumps(clean(data), indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    if private:
        Path(path).chmod(0o600)


def safe_ratio(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def build_reference(numeric, primary, extra=None):
    numeric = np.asarray(numeric, dtype=float)
    primary = np.asarray(primary, dtype=bool)
    if numeric.ndim != 1 or numeric.shape != primary.shape:
        raise ValueError("reference_grid_shape")
    if np.any(np.isfinite(numeric) & ((numeric < 20) | (numeric > 180))) or np.isinf(numeric).any():
        raise ValueError("numeric_outside_frozen_range")
    mask = primary.copy()
    if extra is not None:
        extra = np.asarray(extra, dtype=bool)
        if extra.shape != numeric.shape:
            raise ValueError("additional_qc_shape")
        mask &= extra
    return np.where(mask, numeric, np.nan)


def earliest_window(values, size=360, min_fraction=.8):
    finite = np.isfinite(values)
    if len(finite) < size:
        return None
    cumulative = np.r_[0, np.cumsum(finite)]
    starts = np.flatnonzero(cumulative[size:] - cumulative[:-size] >= math.ceil(size * min_fraction))
    return int(starts[0]) if len(starts) else None


def make_display(values, step_bins, phase_bins, initialize=False):
    display, held = held_display(values, step_bins, phase_bins)
    if initialize:
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite):
            first = int(finite[0])
            seed = (np.arange(len(values)) >= first) & (held < 0)
            held[seed] = first
            display[seed] = values[first]
    return display, held


def event_counts(values, display, held, events, threshold):
    result = classify_events(values, display, held, events, threshold)
    counts = np.bincount(result["category"], minlength=3)
    low = np.isfinite(display) & (display < threshold)
    delay = sum(int(np.argmax(low[s:e])) / 6 for (s, e, _), category in zip(events, result["category"]) if category != 2)
    return {"episodes": len(events), "fresh": int(counts[0]), "inherited_only": int(counts[1]),
            "never_low": int(counts[2]), "carry_in": int(result["carry_in"].sum()),
            "first_inherited": int(result["first_inherited"].sum()),
            "detected": int(counts[0] + counts[1]), "detection_delay_sum_min": delay}, result


def analyze_record(values, record_id, subject_id, duration_seconds, scenario, window=None,
                   startup="primary", thresholds=THRESHOLDS, intervals=INTERVALS):
    values = np.asarray(values, float)
    start, end = window if window is not None else (0, len(values))
    if not (0 <= start < end <= len(values)):
        raise ValueError("evaluation_window_bounds")
    reference = values[start:end].copy()
    if startup == "exclude_first5":
        reference[:max(0, 30 - start)] = np.nan
    hours = (end - start) / 360 if window is not None else float(duration_seconds) / 3600
    base = {"record_id": record_id, "subject_id": subject_id, "scenario": scenario,
            "window_start_bin": start, "window_end_bin_exclusive": end}
    fixed = {threshold: reference_events(reference, threshold, 0, 6) for threshold in thresholds}
    robust = {(gap * 10, minimum * 10): reference_events(reference, 65, gap, minimum)
              for gap in (0, 1, 2) for minimum in (6, 18, 30)} if 65 in thresholds else {}
    frequencies, event_rows, duration_rows = [], [], []
    for interval in intervals:
        step = int(round(interval * 6))
        for phase in range(step):
            full_display, full_held = make_display(values, step, phase, startup == "initialize")
            display = full_display[start:end]
            held = full_held[start:end] - start
            evaluated = reference.copy() if startup == "exclude_predisplay" else reference
            if startup == "exclude_predisplay":
                evaluated[~np.isfinite(display)] = np.nan
            valid = np.isfinite(evaluated)
            common = {**base, "interval_min": float(interval), "offset_sec": phase * 10}
            for threshold in thresholds:
                row = {**common, **decompose_deficit(evaluated, display, threshold, 1 / 6),
                       "original_record_hours": hours, "analyzed_grid_hours": (end - start) / 360,
                       "valid_reference_min": float(valid.sum() / 6),
                       "display_low_min": float(np.sum(valid & np.isfinite(display) & (display < threshold)) / 6)}
                frequencies.append(row)
                # This continuous-only sensitivity has no fixed original episode denominator.
                if startup == "exclude_predisplay":
                    continue
                events = fixed[threshold]
                counts, categories = event_counts(reference, display, held, events, threshold)
                event_base = {**common, "threshold": float(threshold), "gap_sec": 0, "minimum_low_sec": 60}
                event_rows.append({**event_base, **counts})
                lengths = np.array([event[2] for event in events], dtype=int)
                for label, selected in [("60_to_lt180s", lengths < 18),
                                         ("180_to_lt300s", (lengths >= 18) & (lengths < 30)),
                                         ("ge300s", lengths >= 30)]:
                    subset = [event for event, include in zip(events, selected) if include]
                    selected_counts, _ = event_counts(reference, display, held, subset, threshold)
                    duration_rows.append({**event_base, "duration_band": label, **selected_counts})
                if interval == 5 and threshold == 65:
                    for (gap, minimum), configuration in robust.items():
                        if gap == 0 and minimum == 60:
                            continue
                        counts, _ = event_counts(reference, display, held, configuration, 65)
                        event_rows.append({**common, "threshold": 65., "gap_sec": gap,
                                           "minimum_low_sec": minimum, **counts})
    return tuple(pd.DataFrame(rows) for rows in (frequencies, event_rows, duration_rows))


def validate_frozen_code(root):
    if sha256(root / "analysis_contract.json") != CONTRACT_SHA256:
        raise ValueError("analysis_contract_hash_mismatch")
    code = Path(__file__).resolve().parent
    provenance = json.loads((code / "provenance.json").read_text())
    for item in provenance["vendored"]:
        if sha256(code / item["copy"]) != item["sha256"]:
            raise ValueError("vendored_function_hash_mismatch")
    return provenance


def validate_ready_manifest(root, path):
    manifest = json.loads(Path(path).read_text())
    if str(manifest.get("status", "")).lower() not in {"ready", "pass", "complete", "completed"}:
        raise ValueError("qc_manifest_not_ready")
    expected = manifest.get("input_sha256", {})
    hashes = {}
    for name in ("qc_inputs.npz", "records.json"):
        actual = sha256(root / "private" / name)
        if expected.get(name) != actual:
            raise ValueError("qc_manifest_input_hash_mismatch")
        hashes[name] = actual
    return hashes


def load_inputs(root):
    records = json.loads((root / "private/records.json").read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("records_manifest_not_nonempty_list")
    inputs, selection, seen = [], [], set()
    with np.load(root / "private/qc_inputs.npz", allow_pickle=False) as arrays:
        for record in records:
            required = {"record_id", "subject_id", "duration_seconds", "adult", "full_record_inside_icu",
                        "primary_family", "original_numeric_valid_bins", "n_bins"}
            if not required.issubset(record):
                raise ValueError("record_metadata_fields_missing")
            rid = str(record["record_id"])
            if rid in seen or not rid:
                raise ValueError("record_identity_missing_or_duplicated")
            seen.add(rid)
            n = record["n_bins"]
            seconds = record["duration_seconds"]
            if not isinstance(n, int) or n < 0 or not np.isfinite(seconds) or n != math.floor(seconds / 10):
                raise ValueError("original_record_grid_length_mismatch")
            data = {}
            for prefix in ("numeric", "waveform_qc", "waveform_strict", "coherent", "waveform_mean"):
                key = f"{prefix}_{rid}"
                if key not in arrays:
                    raise ValueError("qc_array_missing")
                array = arrays[key]
                if array.shape != (n,):
                    raise ValueError("qc_array_grid_shape_mismatch")
                if prefix in {"waveform_qc", "waveform_strict", "coherent"} and array.dtype.kind != "b":
                    raise ValueError("qc_mask_not_boolean")
                data[prefix] = array
            numeric = data["numeric"].astype(float)
            if int(np.isfinite(numeric).sum()) != record["original_numeric_valid_bins"]:
                raise ValueError("original_numeric_valid_count_mismatch")
            if record["adult"] is not None and type(record["adult"]) is not bool:
                raise ValueError("adult_confirmation_not_boolean_or_null")
            if type(record["full_record_inside_icu"]) is not bool:
                raise ValueError("icu_linkage_not_boolean")
            references = {"primary": build_reference(numeric, data["waveform_qc"]),
                          "strict": build_reference(numeric, data["waveform_qc"], data["waveform_strict"]),
                          "coherent": build_reference(numeric, data["waveform_qc"], data["coherent"])}
            coverage = {variant: float(np.isfinite(values).sum() / n) if n else 0.
                        for variant, values in references.items()}
            numeric_primary = bool(record["primary_family"]) and record["original_numeric_valid_bins"] > 0
            adult = numeric_primary and record["adult"] is True
            icu = adult and record["full_record_inside_icu"] is True
            if icu and (record["subject_id"] is None or str(record["subject_id"]) == ""):
                raise ValueError("eligible_subject_identity_missing")
            long_enough = icu and n >= 360
            eligibility = {variant: bool(long_enough and np.isfinite(values).sum() >= math.ceil(.8 * n))
                           for variant, values in references.items()}
            earliest = earliest_window(references["primary"]) if long_enough else None
            paired = np.isfinite(numeric) & data["waveform_qc"] & np.isfinite(data["waveform_mean"])
            differences = np.abs(numeric[paired] - data["waveform_mean"][paired])
            selection.append({"record_id": rid, "subject_id": record["subject_id"], "n_bins": n,
                              "duration_seconds": seconds, "numeric_primary": numeric_primary,
                              "adult_confirmed": adult, "adult_icu_linked": icu, "at_least_hour": long_enough,
                              "primary_included": eligibility["primary"],
                              "strict_included": eligibility["strict"], "coherent_included": eligibility["coherent"],
                              "earliest_hour_start_bin": earliest,
                              **{f"{key}_coverage": value for key, value in coverage.items()},
                              "waveform_numeric_paired_bins": int(paired.sum()),
                              "waveform_numeric_median_absolute_difference": float(np.median(differences)) if len(differences) else np.nan})
            inputs.append({"metadata": record, "references": references, "eligibility": eligibility,
                           "earliest_window": earliest})
    frame = pd.DataFrame(selection)
    counts = {"candidate_records": len(frame), "numeric_primary_records": int(frame.numeric_primary.sum()),
              "confirmed_adult_numeric_records": int(frame.adult_confirmed.sum()),
              "confirmed_adult_icu_records": int(frame.adult_icu_linked.sum()),
              "adult_icu_at_least_hour_records": int(frame.at_least_hour.sum()),
              "primary_records": int(frame.primary_included.sum()),
              "primary_subjects": int(frame.loc[frame.primary_included, "subject_id"].nunique()),
              "strict_eligible80_records": int(frame.strict_included.sum()),
              "numeric_coherence_eligible80_records": int(frame.coherent_included.sum()),
              "earliest_hour_records": int(frame.earliest_hour_start_bin.notna().sum()),
              "earliest_hour_subjects": int(frame.loc[frame.earliest_hour_start_bin.notna(), "subject_id"].nunique())}
    return inputs, counts, frame


def cluster_sums(frame, columns, repetitions=REPLICATES, seed=SEED):
    mappings = frame[["record_id", "subject_id"]].drop_duplicates()
    if mappings.record_id.duplicated().any() or mappings.subject_id.isna().any():
        raise ValueError("record_subject_mapping_inconsistent")
    averaged = frame.groupby(["record_id", "subject_id"], sort=True)[columns].mean().reset_index()
    subjects = averaged.groupby("subject_id", sort=True)[columns].sum().to_numpy(float)
    n = len(subjects)
    indices = np.random.default_rng(seed).integers(0, n, size=(repetitions, n))
    return averaged, subjects, subjects.sum(axis=0), subjects[indices].sum(axis=1), indices


def ratio_ci(numerator, denominator):
    ratios = safe_ratio(numerator, denominator)
    valid = ratios[np.isfinite(ratios)]
    if not len(valid):
        return np.nan, np.nan, 0
    low, high = np.quantile(valid, [.025, .975])
    return float(low), float(high), len(valid)


def frequency_metrics(components):
    x = {name: components[..., index] for index, name in enumerate(FREQUENCY_COLUMNS)}
    r, d, h, o = (x[name] for name in ("true_auc", "display_auc", "hidden_auc", "overdisplay_auc"))
    absolute = h + o
    net = d - r
    residual = safe_ratio(np.abs(net), absolute)
    return {**x, "net_auc_difference": net, "absolute_auc_disagreement": absolute,
            "hidden_deficit_ratio": safe_ratio(h, r), "overdisplay_deficit_ratio": safe_ratio(o, r),
            "net_bias_ratio": safe_ratio(net, r), "absolute_auc_disagreement_ratio": safe_ratio(absolute, r),
            "residual_net_fraction_of_absolute_error": residual, "cancellation_fraction": 1 - residual,
            "reference_twa": safe_ratio(r, x["valid_reference_min"]),
            "display_twa": safe_ratio(d, x["valid_reference_min"]),
            "net_twa_difference": safe_ratio(net, x["valid_reference_min"]),
            "hidden_deficit_ratio_display_valid": safe_ratio(x["hidden_auc_display_valid"], x["true_auc_display_valid"]),
            "overdisplay_deficit_ratio_display_valid": safe_ratio(o, x["true_auc_display_valid"]),
            "reference_low_min_per_record_hour": safe_ratio(x["true_hypotension_min"], x["original_record_hours"]),
            "display_low_min_per_record_hour": safe_ratio(x["display_low_min"], x["original_record_hours"]),
            "normotensive_display_min_per_record_hour": safe_ratio(x["normotensive_display_discordance_min"], x["original_record_hours"]),
            "hypotensive_display_min_per_record_hour": safe_ratio(x["hypotensive_display_discordance_min"], x["original_record_hours"]),
            "unavailable_hypotension_min_per_record_hour": safe_ratio(x["reference_hypotension_with_display_unavailable_min"], x["original_record_hours"]),
            "absolute_state_discordance_min_per_record_hour": safe_ratio(x["normotensive_display_discordance_min"] + x["hypotensive_display_discordance_min"] + x["reference_hypotension_with_display_unavailable_min"], x["original_record_hours"])}


def event_metrics(components):
    x = {name: components[..., index] for index, name in enumerate(EVENT_COLUMNS)}
    return {**x, **{name + "_probability": safe_ratio(x[name], x["episodes"])
                    for name in ("fresh", "inherited_only", "never_low", "detected", "carry_in", "first_inherited")},
            "fresh_sampling_loss_probability": safe_ratio(x["inherited_only"] + x["never_low"], x["episodes"]),
            "mean_detection_delay_min": safe_ratio(x["detection_delay_sum_min"], x["detected"])}


def summarize(frame, keys, columns, derive):
    if frame.empty:
        return pd.DataFrame(), []
    rows, details = [], []
    for identity, group in frame.groupby(keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        averaged, subjects, total, draws, indices = cluster_sums(group, columns)
        row = {**dict(zip(keys, identity)), "n_records": len(averaged), "n_subjects": len(subjects),
               "number_of_phases": int(round(float(group.interval_min.iloc[0]) * 6)),
               "bootstrap_replicates": REPLICATES, "bootstrap_seed": SEED}
        estimates, boots = derive(total), derive(draws)
        for metric, estimate in estimates.items():
            samples = np.asarray(boots[metric], float)
            finite = samples[np.isfinite(samples)]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            row.update({metric: float(estimate), metric + "_ci_low": float(low), metric + "_ci_high": float(high),
                        metric + "_valid_bootstrap_replicates": len(finite)})
        if "episodes" in columns:
            per_record = group.groupby("record_id").episodes
            if (per_record.nunique() != 1).any():
                raise ValueError("reference_episode_denominator_varies_across_phases")
            row["reference_episodes"] = int(round(total[0]))
            row["phase_episode_pairs"] = int(group.episodes.sum())
            row["n_records_with_episodes"] = int((averaged.episodes > 0).sum())
            row["denominator_policy"] = "fixed_reference_episodes_across_phases"
        rows.append(row)
        details.append({"key": dict(zip(keys, identity)), "columns": columns, "sums": draws,
                        "indices": indices,
                        "subject_ids": np.array(sorted(group.subject_id.astype(str).unique()), dtype=str)})
    return pd.DataFrame(rows), details


def summarize_frequency(frame):
    return summarize(frame, KEYS, FREQUENCY_COLUMNS, frequency_metrics)


def summarize_events(frame, duration=False):
    return summarize(frame, EVENT_KEYS + (["duration_band"] if duration else []), EVENT_COLUMNS, event_metrics)


def distributions(frame, keys, columns, derive, level):
    rows = []
    for identity, group in frame.groupby(keys, sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        if level == "record":
            components = group.groupby("record_id", sort=True)[columns].mean().to_numpy(float)
        else:
            components = group[columns].to_numpy(float)
        for metric, values in derive(components).items():
            finite = np.asarray(values)[np.isfinite(values)]
            quantiles = np.quantile(finite, [.05, .25, .5, .75, .95]) if len(finite) else [np.nan] * 5
            rows.append({**dict(zip(keys, identity)), "level": level, "metric": metric,
                         "n_units": len(components), "n_defined": len(finite),
                         **dict(zip(["p05", "p25", "p50", "p75", "p95"], quantiles))})
    return pd.DataFrame(rows)


def check_invariants(frequency, events, durations):
    net = frequency.display_auc - frequency.true_auc
    difference = frequency.overdisplay_auc - frequency.hidden_auc
    identity_error = float(np.max(np.abs(net - difference)))
    identity_ok = np.allclose(net, difference, rtol=1e-10, atol=1e-8)
    count_ok = (events.fresh + events.inherited_only + events.never_low == events.episodes).all()
    fixed = events.groupby(EVENT_KEYS + ["record_id"]).episodes.nunique().eq(1).all()
    full_phases = frequency.groupby(KEYS + ["record_id"]).offset_sec.nunique()
    expected_phases = np.round(full_phases.index.get_level_values("interval_min").to_numpy(float) * 6).astype(int)
    phases_ok = np.array_equal(full_phases.to_numpy(), expected_phases)
    one = events.loc[(events.interval_min == 1) & (events.gap_sec == 0) & (events.minimum_low_sec == 60)]
    one_ok = bool((one.fresh == one.episodes).all())
    five = events.loc[(events.interval_min == 5) & (events.gap_sec == 0) & (events.minimum_low_sec == 300)]
    five_ok = bool((five.fresh == five.episodes).all())
    if not (identity_ok and count_ok and fixed and phases_ok and one_ok and five_ok):
        raise ValueError("analysis_arithmetic_or_phase_invariant_failed")
    return {"status": "PASS", "max_abs_D_minus_R_minus_O_plus_H": identity_error,
            "event_partition_identity": bool(count_ok), "fixed_event_denominators": bool(fixed),
            "all_absolute_phase_offsets": bool(phases_ok), "one_minute_ge60s_all_fresh": one_ok,
            "five_minute_contiguous_ge300s_all_fresh": five_ok}


def support_assessment(frequency, events):
    f = frequency.loc[(frequency.scenario == "primary") & (frequency.threshold == 65)].sort_values("interval_min")
    e = events.loc[(events.scenario == "primary") & (events.threshold == 65) & (events.gap_sec == 0)
                   & (events.minimum_low_sec == 60)].sort_values("interval_min")
    focus = f.loc[f.interval_min == 5].iloc[0].to_dict()
    event_focus = e.loc[e.interval_min == 5].iloc[0].to_dict()
    fields = ["hidden_deficit_ratio", "overdisplay_deficit_ratio", "net_bias_ratio", "net_auc_difference",
              "absolute_auc_disagreement_ratio", "cancellation_fraction", "residual_net_fraction_of_absolute_error"]
    return {"policy": "Separate descriptive criteria; no forced near-zero net difference or binary replication verdict",
            "bidirectional_magnitudes_and_net": {key: focus[key] for name in fields for key in (name, name + "_ci_low", name + "_ci_high")},
            "event_losses": {key: event_focus[key] for name in ["fresh_probability", "inherited_only_probability", "never_low_probability", "fresh_sampling_loss_probability"]
                             for key in (name, name + "_ci_low", name + "_ci_high")},
            "interval_trend_frequency": f[["interval_min", "hidden_deficit_ratio", "overdisplay_deficit_ratio", "net_bias_ratio", "absolute_auc_disagreement_ratio"]].to_dict("records"),
            "interval_trend_events": e[["interval_min", "fresh_probability", "inherited_only_probability", "never_low_probability"]].to_dict("records")}


def save_private_csv(frame, path):
    frame.to_csv(path, index=False, compression="gzip")
    path.chmod(0o600)


def run(root, ready_manifest):
    started = time.monotonic()
    provenance = validate_frozen_code(root)
    input_hashes = validate_ready_manifest(root, ready_manifest)
    inputs, counts, selection = load_inputs(root)
    if not counts["primary_records"]:
        raise ValueError("no_eligible_primary_records")
    aggregate = root / "analysis/aggregate"
    private = root / "private/analysis"
    aggregate.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    frequency_parts, event_parts, duration_parts = [], [], []
    scenario_counts = {}
    for name, (cohort, variant, startup) in SCENARIOS.items():
        n_records, subjects = 0, set()
        for item in inputs:
            included = item["earliest_window"] is not None if cohort == "window" else item["eligibility"][cohort]
            if not included:
                continue
            metadata = item["metadata"]
            start = item["earliest_window"]
            window = (start, start + 360) if cohort == "window" else None
            frequency, events, durations = analyze_record(item["references"][variant], str(metadata["record_id"]),
                                                         str(metadata["subject_id"]), metadata["duration_seconds"],
                                                         name, window=window, startup=startup)
            frequency_parts.append(frequency)
            if not events.empty:
                event_parts.append(events)
                duration_parts.append(durations)
            n_records += 1
            subjects.add(str(metadata["subject_id"]))
        scenario_counts[name] = {"records": n_records, "subjects": len(subjects), "cohort": cohort,
                                 "qc_reference": variant, "startup": startup,
                                 "event_metrics_reported": startup != "exclude_predisplay"}
    frequency = pd.concat(frequency_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    durations = pd.concat(duration_parts, ignore_index=True)
    invariants = check_invariants(frequency, events, durations)
    frequency_summary, frequency_boot = summarize_frequency(frequency)
    event_summary, event_boot = summarize_events(events)
    duration_summary, duration_boot = summarize_events(durations, duration=True)
    for level in ("record", "phase"):
        frames = [distributions(frequency, KEYS, FREQUENCY_COLUMNS, frequency_metrics, level),
                  distributions(events, EVENT_KEYS, EVENT_COLUMNS, event_metrics, level)]
        pd.concat(frames, ignore_index=True).to_csv(aggregate / f"{level}_distributions.csv", index=False)
    for frame, filename in [(frequency_summary, "frequency_summary.csv"), (event_summary, "episode_summary.csv"),
                             (duration_summary, "episode_duration_summary.csv")]:
        frame.to_csv(aggregate / filename, index=False)
    figure_source = frequency_summary.loc[frequency_summary.scenario == "primary"].merge(
        event_summary.loc[(event_summary.scenario == "primary") & (event_summary.gap_sec == 0) & (event_summary.minimum_low_sec == 60)],
        on=KEYS, suffixes=("", "_event"))
    figure_source.to_csv(aggregate / "figure_source_data.csv", index=False)
    for frame, filename in [(frequency, "frequency_by_record_phase.csv.gz"), (events, "episodes_by_record_phase.csv.gz"),
                            (durations, "episode_duration_by_record_phase.csv.gz"), (selection, "record_selection.csv.gz")]:
        save_private_csv(frame, private / filename)
    boot_arrays, boot_catalog = {}, []
    for family, details in [("frequency", frequency_boot), ("episodes", event_boot), ("duration", duration_boot)]:
        for index, detail in enumerate(details):
            prefix = f"{family}_{index}"
            boot_arrays[prefix + "_sums"] = detail["sums"]
            boot_arrays[prefix + "_indices"] = detail["indices"].astype(np.int32)
            boot_arrays[prefix + "_subject_ids"] = detail["subject_ids"]
            boot_catalog.append({"prefix": prefix, "key": detail["key"], "columns": detail["columns"]})
    np.savez_compressed(private / "bootstrap_details.npz", **boot_arrays)
    (private / "bootstrap_details.npz").chmod(0o600)
    write_json(private / "bootstrap_catalog.json", boot_catalog, private=True)
    if validate_ready_manifest(root, ready_manifest) != input_hashes:
        raise ValueError("qc_inputs_changed_during_analysis")
    write_json(aggregate / "cohort_qc_attrition.json", {"counts": counts, "scenarios": scenario_counts,
               "whole_record_denominator": "All original complete 10-second bins; no valid-only trimming",
               "window_selection": "Earliest coverage-qualified 360-bin window, never selected by hypotension",
               "waveform_mean": "Alignment diagnostic only; never used as reference or filling"})
    result = {"status": "complete", "purpose": "Adult ICU cross-context technical replication, not clinical-outcome validation",
              "clinical_adjudication": False, "source_data_modified": False, "manuscript_modified": False,
              "counts": counts, "scenarios": scenario_counts,
              "uncertainty": {"replicates": REPLICATES, "seed": SEED, "cluster": "subject_id",
                              "phase_average": "Within record before summing records within subject",
                              "ratios": "Recomputed from summed additive components in every bootstrap replicate",
                              "interval": "Nonparametric percentile 2.5th/97.5th"},
              "units": {"auc": "mmHg*min", "duration": "min", "twa": "mmHg per QC-valid reference minute",
                        "state_rates": "minutes per original ICU-record hour, including excluded startup time in whole-record hour denominator"},
              "predisplay_sensitivity": "Continuous metrics only; no phase-varying truncated episode rates reported",
              "window_display": "Built on full original QC grid before slicing, preserving absolute phase and prewindow carry-in",
              "invariants": invariants, "support_assessment": support_assessment(frequency_summary, event_summary),
              "primary_map65_5min_frequency": frequency_summary.loc[(frequency_summary.scenario == "primary") & (frequency_summary.threshold == 65) & (frequency_summary.interval_min == 5)].iloc[0].to_dict(),
              "primary_map65_5min_episodes": event_summary.loc[(event_summary.scenario == "primary") & (event_summary.threshold == 65) & (event_summary.interval_min == 5) & (event_summary.gap_sec == 0) & (event_summary.minimum_low_sec == 60)].iloc[0].to_dict(),
              "input_sha256": input_hashes, "contract_sha256": CONTRACT_SHA256,
              "code_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
              "function_provenance": provenance, "elapsed_seconds": time.monotonic() - started}
    write_json(aggregate / "results.json", result)
    return {"status": "complete", "counts": counts, "frequency_summary_rows": len(frequency_summary),
            "episode_summary_rows": len(event_summary), "elapsed_seconds": result["elapsed_seconds"],
            "results": "analysis/aggregate/results.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--ready-manifest", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not all((root / "private" / name).is_file() for name in ("qc_inputs.npz", "records.json")):
        print(json.dumps({"status": "inputs_not_ready", "code": "missing_qc_inputs"}))
        return 3
    try:
        validate_frozen_code(root)
        if args.check_inputs:
            _, counts, _ = load_inputs(root)
            print(json.dumps({"status": "inputs_valid", "counts": counts}))
            return 0
        if args.ready_manifest is None:
            print(json.dumps({"status": "inputs_not_ready", "code": "explicit_ready_manifest_required"}))
            return 3
        result = run(root, args.ready_manifest)
        print(json.dumps(clean(result), allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        # Error messages are controlled codes; never emit record IDs or private traceback contents.
        code = str(error) if isinstance(error, ValueError) else type(error).__name__
        if len(code) > 100 or any(char.isspace() for char in code):
            code = type(error).__name__
        print(json.dumps({"status": "failed", "code": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
