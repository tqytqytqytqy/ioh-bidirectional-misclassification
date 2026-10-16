"""Independent cache-based numerical verification. Never imports analysis modules."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

F13 = (
    "reference_auc", "display_auc", "hidden_auc", "overdisplay_auc",
    "hidden_unavailable_auc", "reference_low_min", "display_low_min",
    "low_reference_normal_display_min", "normal_reference_low_display_min",
    "low_reference_unavailable_min", "valid_reference_min", "display_available_min",
    "selected_minutes",
)
E6 = ("episodes", "fresh", "inherited_only", "never_low", "visible", "delay_sum_min")
COLS = F13 + E6
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SHA = "3157dfa67db7f7fbcfcbd859b8e46d978583d8851b12348ea2c71fb68dcae61b"
SEED = 20260706
GROUP_KEYS = ["source", "resolution", "scenario", "threshold", "interval_min", "event_min_sec"]
STRATUM_KEYS = ["source", "resolution", "scenario", "event_min_sec", "feature", "level"]
LEVELS = {
    "duration": ("60_to_lt180", "180_to_lt300", "300_to_lt600", "ge600"),
    "nadir": ("lt55", "55_to_lt65"),
    "preceding_normal": ("lt60", "60_to_lt300", "ge300", "unknown"),
}


def minute_reference(values):
    blocks = np.asarray(values, dtype=float)[:len(values) // 6 * 6].reshape(-1, 6)
    result = np.full(len(blocks), np.nan)
    enough = np.isfinite(blocks).sum(axis=1) >= 5
    if enough.any():
        result[enough] = np.nanmedian(blocks[enough], axis=1)
    return result


def qualifying_hours(values, dt):
    bins_per_hour = 3600 // dt
    whole = len(values) // bins_per_hour
    selected = np.zeros(len(values), dtype=bool)
    counts = np.isfinite(np.asarray(values)[:whole * bins_per_hour]).reshape(whole, bins_per_hour).sum(axis=1)
    selected[:whole * bins_per_hour] = np.repeat(counts >= bins_per_hour * .8, bins_per_hour)
    return selected


def held_display(values, step, phase):
    values = np.asarray(values, dtype=float)
    schedule = np.arange(phase, len(values), step)
    updates = schedule[np.isfinite(values[schedule])]
    at = np.searchsorted(updates, np.arange(len(values)), side="right") - 1
    origin = np.full(len(values), -1, dtype=int)
    display = np.full(len(values), np.nan)
    available = at >= 0
    origin[available] = updates[at[available]]
    display[available] = values[origin[available]]
    return display, origin


def low_runs(values, selected, dt, threshold, minimum):
    values = np.asarray(values, dtype=float)
    observed = np.asarray(selected, dtype=bool) & np.isfinite(values)
    low = observed & (values < threshold)
    edges = np.diff(np.r_[False, low, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    runs = []
    for start, end in zip(starts, ends):
        if (end - start) * dt < minimum:
            continue
        previous = start - 1
        while previous >= 0 and observed[previous] and values[previous] >= threshold:
            previous -= 1
        known = previous >= 0 and observed[previous] and values[previous] < threshold
        gap = (start - previous - 1) * dt if known else np.nan
        runs.append((int(start), int(end), float(values[start:end].min()), gap))
    return runs


def estimate(values, selected, dt, threshold=65, interval=5, minimum=180):
    values = np.asarray(values, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if dt not in (10, 60) or len(values) != len(selected):
        raise ValueError("Invalid grid or evaluation mask")
    step = interval * 60 // dt
    valid = selected & np.isfinite(values)
    deficit = np.maximum(threshold - values[valid], 0)
    low = deficit > 0
    scale = dt / 60
    events = low_runs(values, selected, dt, threshold, minimum)
    categories = []
    for start, end, nadir, gap in events:
        seconds = (end - start) * dt
        duration = ("60_to_lt180" if seconds < 180 else "180_to_lt300" if seconds < 300
                    else "300_to_lt600" if seconds < 600 else "ge600")
        preceding = ("unknown" if np.isnan(gap) else "lt60" if gap < 60
                     else "60_to_lt300" if gap < 300 else "ge300")
        categories.append((("duration", duration), ("nadir", "lt55" if nadir < 55 else "55_to_lt65"),
                           ("preceding_normal", preceding)))
    stratum = {(feature, level): np.zeros(len(E6)) for feature, levels in LEVELS.items() for level in levels}
    phase_components = []
    for phase in range(step):
        full_display, origin = held_display(values, step, phase)
        display = full_display[valid]
        available = np.isfinite(display)
        shown_deficit = np.where(available, np.maximum(threshold - display, 0), 0)
        shown_low = available & (display < threshold)
        hidden = np.maximum(deficit - shown_deficit, 0)
        over = np.maximum(shown_deficit - deficit, 0)
        measures = np.array([
            deficit.sum(), shown_deficit.sum(), hidden.sum(), over.sum(),
            deficit[~available].sum(), low.sum(), shown_low.sum(),
            (low & available & ~shown_low).sum(), (~low & shown_low).sum(),
            (low & ~available).sum(), valid.sum(), available.sum(), selected.sum(),
        ], dtype=float) * scale
        event_total = np.zeros(len(E6))
        for event_index, (start, end, _, _) in enumerate(events):
            positions = np.flatnonzero(full_display[start:end] < threshold)
            visible = len(positions) > 0
            fresh = bool(np.any(origin[start:end] >= start))
            event = np.array([1, fresh, visible and not fresh, not visible, visible,
                              positions[0] * scale if visible else 0.], dtype=float)
            event_total += event
            for key in categories[event_index]:
                stratum[key] += event / step
        phase_components.append(np.r_[measures, event_total])
    components = np.mean(phase_components, axis=0)
    expected = sum(min(1, (end - start) / step) for start, end, _, _ in events)
    return components, stratum, {
        "expected_fresh": expected,
        "analytic_error": abs(components[14] - expected),
        "phases": step,
    }


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value, private=False):
    path = Path(path)
    if private:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.chmod(temporary, 0o600 if private else 0o644)
    temporary.replace(path)


class Checks:
    def __init__(self):
        self.groups = defaultdict(lambda: {"comparisons": 0, "failures": 0, "max_abs_difference": 0.})
        self.failures = []

    def close(self, name, actual, expected, context=None, kind="component"):
        a, b = np.broadcast_arrays(np.asarray(actual, float), np.asarray(expected, float))
        atol, rtol = {"component": (1e-7, 1e-10), "count": (1e-9, 0.), "integer": (0., 0.),
                      "metric": (1e-9, 1e-9)}[kind]
        match = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        group = self.groups[name]
        group["comparisons"] += int(a.size)
        group["failures"] += int((~match).sum())
        finite = np.isfinite(a) & np.isfinite(b)
        difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        group["max_abs_difference"] = max(group["max_abs_difference"], difference)
        if not match.all() and len(self.failures) < 300:
            self.failures.append({"check": name, "context": str(context),
                                  "failed_positions": np.flatnonzero(~match).tolist()[:30],
                                  "actual": [None if not np.isfinite(x) else float(x) for x in a.reshape(-1)[~match.reshape(-1)][:30]],
                                  "expected": [None if not np.isfinite(x) else float(x) for x in b.reshape(-1)[~match.reshape(-1)][:30]]})
        return bool(match.all())

    def truth(self, name, truth, context=None):
        return self.close(name, np.asarray(truth, dtype=float), 1., context, "count")

    def public(self):
        return {name: {**value, "status": "PASS" if value["failures"] == 0 else "FAIL"}
                for name, value in sorted(self.groups.items())}


def protected_hashes(root, checks, details, label):
    manifest = json.loads((root / "qa/frozen_inputs.json").read_text())
    checks.truth("contract_hash", digest(root / "analysis_contract.json") == CONTRACT_SHA)
    checks.truth("manifest_contract_hash", manifest["contract_sha256"] == CONTRACT_SHA)
    failures = []
    count = 0
    for path, expected in manifest["protected_sha256"].items():
        count += 1
        if not Path(path).is_file() or digest(path) != expected:
            failures.append(path)
    checks.truth("protected_hashes_" + label, len(failures) == 0)
    details["protected_hash_failures_" + label] = failures
    return {"files_checked": count, "changed_or_missing": len(failures), "all_local_hashes": not failures}


def pressure_filter(values):
    result = np.array(values, dtype=float, copy=True)
    result[(result < 20) | (result > 180) | ~np.isfinite(result)] = np.nan
    return result


def numerical_records(root, source):
    project = root.parent
    sic = project / "SICdb分钟级对齐分析_20261015/private"
    if source == "VitalDB":
        sys.path.insert(0, str(sic / "deps"))
        import pyarrow.parquet as pq
        folder = project / "analysis_v10_subject_phase_release/outputs/intermediate"
        panel = pq.read_table(folder / "artmap_10s.parquet", columns=["case_id", "time_sec", "art_map"]).to_pandas()
        meta = pq.read_table(folder / "vitaldb_manifest.parquet", columns=[
            "case_id", "subjectid", "age", "primary_start_sec", "primary_end_sec"]).to_pandas().set_index("case_id")
        for case, rows in panel.groupby("case_id", sort=True):
            row = meta.loc[case]
            if not np.isfinite(row.age) or row.age < 25:
                continue
            rows = rows.sort_values("time_sec")
            times = rows.time_sec.to_numpy(float)
            if times[0] != 0 or not np.all(np.diff(times) == 10):
                raise ValueError("VitalDB frozen reference is not an origin-zero 10-second grid")
            size = int(np.floor((row.primary_end_sec - row.primary_start_sec) / 10))
            yield str(case), str(int(row.subjectid)), pressure_filter(rows.art_map.to_numpy(float)[:size])
    elif source == "MIMIC":
        folder = project / "MIMIC技术重复分析_20261015/private"
        metadata = {}
        metadata_path = root / "restricted/subgroup_metadata.csv"
        if metadata_path.exists():
            import csv
            with metadata_path.open() as handle:
                for row in csv.DictReader(handle):
                    if row["source"] in ("MIMIC", "MIMIC-IV"):
                        try:
                            metadata[row["record_id"]] = float(row.get("age", ""))
                        except (ValueError, TypeError):
                            pass
        records = json.loads((folder / "records.json").read_text())
        with np.load(folder / "qc_inputs.npz", allow_pickle=False) as archive:
            for row in records:
                if not (row["adult"] is True and row["full_record_inside_icu"] and
                        row["primary_family"] and row["n_bins"] >= 360):
                    continue
                record = str(row["record_id"])
                if np.isfinite(metadata.get(record, np.nan)) and metadata[record] < 25:
                    continue
                values = pressure_filter(archive["numeric_" + record])
                qc = archive["waveform_qc_" + record]
                if qc.dtype != bool or qc.shape != values.shape:
                    raise ValueError("Invalid upstream waveform-QC mask")
                values[~qc] = np.nan
                yield record, str(row["subject_id"]), values
    else:
        folder = sic / "sicdb"
        ready = json.loads((folder / "ready.json").read_text())
        for shard in ready["shards"]:
            with np.load(folder / shard["path"], allow_pickle=False) as archive:
                cases, subjects = archive["case_id"], archive["subject_id"]
                ages, indexes, pressures = archive["age_bin"], archive["index"], archive["map"]
                for i, case in enumerate(cases):
                    if ages[i] < 25:
                        continue
                    start, end = indexes[i:i + 2]
                    yield str(int(case)), str(int(subjects[i])), pressure_filter(pressures[start:end])


def scope_masks(source, values):
    if source == "SICdb":
        return {"primary_1min": (values, qualifying_hours(values, 60), 60, "primary", 180)}
    minute = minute_reference(values)
    native_mask, minute_mask = qualifying_hours(values, 10), qualifying_hours(minute, 60)
    common_native = np.zeros(len(values), bool)
    common_native[:len(minute) * 6] = native_mask[:len(minute) * 6] & np.repeat(minute_mask, 6)
    common_minute = common_native[:len(minute) * 6:6]
    return {
        "primary_10s": (values, native_mask, 10, "primary", 60),
        "primary_1min": (minute, minute_mask, 60, "primary", 180),
        "paired_bridge_10s": (values, common_native, 10, "paired_bridge", 180),
        "paired_bridge_1min": (minute, common_minute, 60, "paired_bridge", 180),
        "paired_native60_10s": (values, common_native, 10, "paired_native60", 60),
    }


def assignment_reference(values, offset_sec):
    values = pressure_filter(values)
    if not np.isfinite(offset_sec) or offset_sec < 0 or offset_sec >= len(values) * 60:
        return np.full(len(values), np.nan), np.zeros(len(values), bool), 0
    masked = values.copy()
    masked[np.arange(len(masked)) * 60 < offset_sec] = np.nan
    selected = qualifying_hours(values, 60) & qualifying_hours(masked, 60)
    whole_hours = len(values) // 60
    starts = np.arange(whole_hours) * 3600
    crossing = (starts < offset_sec) & (starts + 3600 > offset_sec) & selected[:whole_hours * 60:60]
    return masked, selected, int(crossing.sum())


def assignment_records(root):
    import csv
    labels = {}
    with (root / "restricted/subgroup_metadata.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "SICdb":
                labels[row["record_id"]] = row.get("surgical_label", "unknown")
    folder = root.parent / "SICdb分钟级对齐分析_20261015/private/sicdb"
    ready = json.loads((folder / "ready.json").read_text())
    for shard in ready["shards"]:
        with np.load(folder / shard["path"], allow_pickle=False) as archive:
            cases, subjects = archive["case_id"], archive["subject_id"]
            ages, indexes, pressures = archive["age_bin"], archive["index"], archive["map"]
            offsets = archive["icu_offset_sec"]
            for i, case in enumerate(cases):
                if ages[i] < 25:
                    continue
                start, end = indexes[i:i + 2]
                rid = str(int(case))
                yield rid, str(int(subjects[i])), pressure_filter(pressures[start:end]), float(offsets[i]), labels.get(rid, "unknown")


def choose_assignment_before_results(root, sample_size, checks):
    path = root / "restricted/independent_icu_selection.json"
    metadata_sha = digest(root / "restricted/subgroup_metadata.csv")
    if path.exists():
        chosen = json.loads(path.read_text())
        if (chosen["contract_sha256"] != CONTRACT_SHA or chosen["seed"] != SEED or
                chosen["sample_size"] != sample_size or chosen["metadata_sha256"] != metadata_sha):
            raise ValueError("Frozen assignment sample or source metadata changed")
        checks.truth("SICdb_assignment_selection_permissions", path.stat().st_mode & 0o777 == 0o600)
    else:
        chosen = {"contract_sha256": CONTRACT_SHA, "seed": SEED, "sample_size": sample_size,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "metadata_sha256": metadata_sha,
                  "rule": "Ascending SHA256 of seed|SICdb|record_id among cache-eligible records, separately for each scenario and targeted qualifying crossing-hour subset; no effect-based selection",
                  "scopes": {}, "eligible_record_counts": {}}
    existing = bool(chosen["scopes"])
    wanted = {rid for rows in chosen["scopes"].values() for rid in rows} if existing else None
    pools = defaultdict(list)
    counts = defaultdict(int)
    cached, grids, crossing_counts = {}, {}, {}
    for rid, pid, original, offset, surgical_label in assignment_records(root):
        if wanted is not None and rid not in wanted:
            continue
        masked, selected, crossing = assignment_reference(original, offset)
        if not selected.any():
            continue
        scenarios = ["icu_after_assignment"]
        if surgical_label in ("Elective Surgery", "Urgent Surgery"):
            scenarios.append("surgical_admission_after_icu")
        elif surgical_label == "No Surgery":
            scenarios.append("nonsurgical_admission_after_icu")
        names = scenarios + [name + "__crossing_hour" for name in scenarios if crossing]
        mappings = {name: (masked, selected, 60, name.split("__", 1)[0], 180) for name in names}
        if not existing:
            rank = hashlib.sha256(f"{SEED}|SICdb|{rid}".encode("ascii")).hexdigest()
            for name in names:
                counts[name] += 1
                pools[name].append((rank, rid))
                pools[name].sort()
                del pools[name][sample_size:]
            keep = {record for pool in pools.values() for _, record in pool}
        else:
            keep = wanted
        if rid in keep:
            cached[rid] = (pid, original)
            grids[rid] = mappings
            crossing_counts[rid] = crossing
        cached = {record: value for record, value in cached.items() if record in keep}
        grids = {record: value for record, value in grids.items() if record in keep}
        crossing_counts = {record: value for record, value in crossing_counts.items() if record in keep}
    if not existing:
        chosen["scopes"] = {name: [rid for _, rid in pool] for name, pool in sorted(pools.items())}
        chosen["eligible_record_counts"] = dict(counts)
    selected_ids = {rid for rows in chosen["scopes"].values() for rid in rows}
    checks.truth("SICdb_assignment_selected_cache_records_found", set(cached) == selected_ids)
    for scenario in ("icu_after_assignment", "surgical_admission_after_icu", "nonsurgical_admission_after_icu"):
        checks.truth("SICdb_assignment_sample_minimum", len(chosen["scopes"].get(scenario, [])) >= 8, scenario)
    for name, ids in chosen["scopes"].items():
        if name.endswith("__crossing_hour"):
            checks.truth("SICdb_assignment_targeted_crossing_sample_size", len(ids) == min(sample_size, chosen["eligible_record_counts"][name]), name)
            checks.truth("SICdb_assignment_targeted_crossing_exists", [crossing_counts[rid] > 0 for rid in ids], name)
    write_json(path, chosen, private=True)
    return chosen, cached, grids


def choose_before_results(root, sources, sample_size, checks):
    path = root / "restricted/independent_selection.json"
    if path.exists():
        chosen = json.loads(path.read_text())
        if chosen["contract_sha256"] != CONTRACT_SHA or chosen["seed"] != SEED or chosen["sample_size"] != sample_size:
            raise ValueError("Existing independently frozen selection does not match requested protocol")
        checks.truth("selection_permissions", path.stat().st_mode & 0o777 == 0o600)
    else:
        chosen = {"contract_sha256": CONTRACT_SHA, "seed": SEED, "sample_size": sample_size,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "rule": "Ascending SHA256 of seed|source|record_id, selected using cached-reference eligibility only before opening result tables",
                  "sources": {}}
    retained, identities = {}, {}
    for source in sources:
        pools = defaultdict(list)
        eligible = defaultdict(int)
        cached = {}
        identities[source] = {}
        existing = chosen["sources"].get(source)
        wanted = {rid for part in existing["scopes"].values() for rid in part} if existing else None
        for rid, pid, values in numerical_records(root, source):
            identities[source][rid] = pid
            if wanted is not None:
                if rid in wanted:
                    cached[rid] = (pid, values)
                continue
            rank = hashlib.sha256(f"{SEED}|{source}|{rid}".encode("ascii")).hexdigest()
            active = False
            for name, (_, selected, _, _, _) in scope_masks(source, values).items():
                if not selected.any():
                    continue
                eligible[name] += 1
                pools[name].append((rank, rid))
                pools[name].sort()
                del pools[name][sample_size:]
                active |= rid in {r for _, r in pools[name]}
            if active:
                cached[rid] = (pid, values)
                keep = {r for pool in pools.values() for _, r in pool}
                cached = {r: item for r, item in cached.items() if r in keep}
        if existing is None:
            chosen["sources"][source] = {"scopes": {name: [r for _, r in pool] for name, pool in sorted(pools.items())},
                                          "eligible_record_counts": dict(eligible)}
        selected_ids = {r for pool in chosen["sources"][source]["scopes"].values() for r in pool}
        checks.truth(source + "_selected_cache_records_found", set(cached) == selected_ids)
        for name, ids in chosen["sources"][source]["scopes"].items():
            checks.truth(source + "_sample_minimum", len(ids) >= 8, name)
        retained[source] = cached
    write_json(path, chosen, private=True)
    return chosen, retained, identities


def verify_identities(frame, stratum, source, checks, identities=None):
    checks.truth(source + "_source_labels", frame.source.eq(source).to_numpy())
    checks.truth(source + "_stratum_source_labels", stratum.source.eq(source).to_numpy())
    checks.truth(source + "_one_patient_per_record", frame.groupby("record_id").subject_id.nunique().to_numpy() == 1)
    if identities is not None:
        for name, table in (("components", frame), ("strata", stratum)):
            checks.truth(source + "_all_" + name + "_record_subject_links",
                         [identities.get(rid) == pid for rid, pid in zip(table.record_id, table.subject_id)])
    c = frame[list(COLS)].to_numpy(float)
    checks.truth(source + "_finite_nonnegative_components", np.isfinite(c) & (c >= -1e-9))
    checks.close(source + "_auc_identity", c[:, 1] - c[:, 0], c[:, 3] - c[:, 2])
    checks.close(source + "_time_identity", c[:, 5] - c[:, 6], c[:, 7] + c[:, 9] - c[:, 8])
    checks.close(source + "_event_partition", c[:, 13], c[:, 14:17].sum(axis=1), kind="count")
    checks.close(source + "_event_visible_partition", c[:, 17], c[:, 14] + c[:, 15], kind="count")
    checks.close(source + "_episodes_integer", c[:, 13], np.rint(c[:, 13]), kind="count")
    checks.truth(source + "_component_bounds", np.column_stack([
        c[:, 4] <= c[:, 2] + 1e-7, c[:, 2] <= c[:, 0] + 1e-7, c[:, 3] <= c[:, 1] + 1e-7,
        c[:, 6] <= c[:, 11] + 1e-7, c[:, 11] <= c[:, 10] + 1e-7,
        c[:, 10] <= c[:, 12] + 1e-7, c[:, 5] <= c[:, 10] + 1e-7,
        c[:, 13] * frame.event_min_sec.to_numpy(float) / 60 <= c[:, 5] + 1e-7,
        c[:, 18] <= c[:, 17] * frame.interval_min.to_numpy(float) + 1e-7,
    ]))
    checks.truth(source + "_unique_component_rows", ~frame.duplicated(GROUP_KEYS + ["record_id"]))
    keys = ["record_id", "subject_id", "resolution", "scenario", "event_min_sec"]
    main = frame[(frame.threshold == 65) & (frame.interval_min == 5)].set_index(keys)
    checks.truth(source + "_unique_stratum_rows", ~stratum.duplicated(keys + ["feature", "level"]))
    events = stratum[list(E6)].to_numpy(float)
    checks.truth(source + "_stratum_finite_nonnegative", np.isfinite(events) & (events >= -1e-9))
    checks.close(source + "_stratum_event_partition", events[:, 0], events[:, 1:4].sum(axis=1), kind="count")
    checks.close(source + "_stratum_visible_partition", events[:, 4], events[:, 1] + events[:, 2], kind="count")
    checks.truth(source + "_stratum_labels", [feature in LEVELS and level in LEVELS[feature]
                                               for feature, level in zip(stratum.feature, stratum.level)])
    for feature, part in stratum.groupby("feature", sort=True):
        summed = part.groupby(keys, sort=True)[list(E6)].sum()
        exists = summed.index.isin(main.index)
        checks.truth(source + "_strata_parent_exists", exists)
        if exists.all():
            checks.close(source + "_strata_feature_sums", summed.to_numpy(float),
                         main.loc[summed.index, list(E6)].to_numpy(float), feature, "count")
    feature_scenarios = ["primary", "paired_bridge", "paired_native60", "first_hour", "core60_fixed", "icu_after_assignment"]
    expected_parents = main[main.index.get_level_values("scenario").isin(feature_scenarios)].index
    all_stratum_parents = set(stratum.set_index(keys).index)
    for feature in LEVELS:
        actual_parents = stratum[stratum.feature == feature].set_index(keys).index.unique()
        checks.truth(source + "_all_feature_parent_coverage",
                     set(expected_parents).issubset(actual_parents) and set(actual_parents) == all_stratum_parents, feature)


def compare_sample(frame, stratum, source, selection, cached, checks, scope_provider=None, label=None):
    label = label or source
    counts = {}
    maximum_analytic_error = 0.
    keys = ["record_id", "resolution", "scenario", "event_min_sec"]
    target = frame[(frame.threshold == 65) & (frame.interval_min == 5)].set_index(keys)
    starget = stratum.set_index(keys + ["feature", "level"])
    for scope, ids in selection["scopes"].items():
        done = 0
        optional = scope.startswith("paired_native60")
        if optional and not frame.scenario.eq("paired_native60").any():
            continue
        for rid in ids:
            pid, original = cached[rid]
            scopes = scope_provider(rid, original) if scope_provider is not None else scope_masks(source, original)
            values, mask, dt, scenario, minimum = scopes[scope]
            resolution = "10s" if dt == 10 else "1min"
            key = (rid, resolution, scenario, minimum)
            exists = key in target.index
            checks.truth(label + "_sample_result_exists", exists, key)
            if not exists:
                continue
            row = target.loc[key]
            checks.truth(label + "_sample_patient_identity", row.subject_id == pid, key)
            independent, strata, extra = estimate(values, mask, dt, minimum=minimum)
            checks.close(label + "_sample_F13", row[list(F13)].to_numpy(float), independent[:13], key)
            checks.close(label + "_sample_E6", row[list(E6)].to_numpy(float), independent[13:], key, "count")
            checks.close(label + "_sample_analytic_baseline", independent[14], extra["expected_fresh"], key, "count")
            maximum_analytic_error = max(maximum_analytic_error, extra["analytic_error"])
            observed = starget.loc[[index for index in starget.index if index[:4] == key]]
            observed_keys = {(index[-2], index[-1]) for index in observed.index}
            checks.truth(label + "_sample_stratum_coverage", observed_keys == set(strata), key)
            for (feature, level), expected in strata.items():
                skey = key + (feature, level)
                if skey in starget.index:
                    checks.close(label + "_sample_strata_E6", starget.loc[skey, list(E6)].to_numpy(float),
                                 expected, skey, "count")
            done += 1
        counts[scope] = done
    return {"records_in_union": len(cached), "verified_records_by_scope": counts,
            "sample_analytic_max_abs_error": maximum_analytic_error}


def metric_values(components):
    c = np.asarray(components, dtype=float)
    ratio = lambda numerator, denominator: safe_divide(numerator, denominator)
    return {
        "hidden_ratio": ratio(c[..., 2], c[..., 0]),
        "overdisplay_ratio": ratio(c[..., 3], c[..., 0]),
        "net_ratio": ratio(c[..., 1] - c[..., 0], c[..., 0]),
        "reference_twa": ratio(c[..., 0], c[..., 10]),
        "display_twa": ratio(c[..., 1], c[..., 10]),
        "fresh_probability": ratio(c[..., 14], c[..., 13]),
        "inherited_probability": ratio(c[..., 15], c[..., 13]),
        "miss_probability": ratio(c[..., 16], c[..., 13]),
        "mean_visible_delay_min": ratio(c[..., 18], c[..., 17]),
        "lowref_normaldisplay_min_per_hour": ratio(60 * c[..., 7], c[..., 12]),
        "normalref_lowdisplay_min_per_hour": ratio(60 * c[..., 8], c[..., 12]),
        "unavailable_fraction": ratio(c[..., 10] - c[..., 11], c[..., 10]),
        "episodes_per100hours": ratio(6000 * c[..., 13], c[..., 12]),
    }


def safe_divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def patients_first(frame):
    ids, inverse = np.unique(frame.subject_id.to_numpy(dtype=str), return_inverse=True)
    result = np.zeros((len(ids), len(COLS)))
    np.add.at(result, inverse, frame[list(COLS)].to_numpy(float))
    return ids, result


@lru_cache(maxsize=1)
def bootstrap_weights(n):
    return np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=1000).astype(float)


def percentile_interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]) if finite.size else np.array([np.nan, np.nan])


def summary_component_frame(frame):
    import pandas as pd
    onset = frame[(frame.source == "VitalDB") & (frame.scenario == "core60_fixed")]
    allowed = set(zip(onset.record_id, onset.resolution))
    original = frame[(frame.source == "VitalDB") & (frame.scenario == "primary") &
                     (frame.threshold == 65) & (frame.interval_min == 5)]
    matched = original[[pair in allowed for pair in zip(original.record_id, original.resolution)]].copy()
    matched["scenario"] = "core60_matched_original"
    return pd.concat([frame, matched], ignore_index=True)


def aggregate_groups(frame, summary, checks, prefix, strata=False):
    keys = STRATUM_KEYS if strata else GROUP_KEYS
    metrics = ("fresh_probability", "inherited_probability", "miss_probability", "mean_visible_delay_min") if strata else tuple(metric_values(np.zeros(len(COLS))))
    indexed = summary.set_index(keys + ["estimand", "metric"])
    checks.truth(prefix + "_unique_rows", ~summary.duplicated(keys + ["estimand", "metric"]))
    seen = set()
    groups, ci_rows, point_rows = 0, 0, 0
    for key, group in frame.groupby(keys, sort=True):
        groups += 1
        if strata:
            group = group.copy()
            for column in F13:
                group[column] = 0.
        ids, patient = patients_first(group)
        total = patient.sum(axis=0)
        individual = metric_values(patient)
        pooled = metric_values(total)
        event_patients = int((patient[:, 13] > 0).sum())
        event_count = float(total[13])
        check_ci = key[2] == "primary" and (strata or key[3] == 65 and key[4] == 5)
        weights = bootstrap_weights(len(ids)) if check_ci else None
        pooled_draws = metric_values(weights @ patient) if check_ci else None
        metadata = {"n_patients": len(ids), "n_patients_with_events": event_patients,
                    "n_records": group.record_id.nunique(), "events": event_count,
                    "reference_auc": total[0], "valid_hours": total[10] / 60,
                    "selected_hours": total[12] / 60}
        for metric in metrics:
            values = individual[metric]
            defined = np.isfinite(values)
            n_defined = int(defined.sum())
            effective_n = min(n_defined, event_patients) if strata else n_defined
            status = ("CI" if effective_n >= 20 and (not strata or event_count >= 50) else
                      "POINT_ONLY" if effective_n >= 10 else "INSUFFICIENT")
            for estimand in ("pooled", "patient_equal"):
                index = tuple(key) + (estimand, metric)
                seen.add(index)
                exists = index in indexed.index
                checks.truth(prefix + "_row_exists", exists, index)
                if not exists:
                    continue
                row = indexed.loc[index]
                expected = (float(pooled[metric]) if estimand == "pooled" else
                            float(values[defined].mean()) if defined.any() else np.nan)
                if status == "INSUFFICIENT":
                    expected = np.nan
                checks.close(prefix + "_point", row.estimate, expected, index, "metric")
                checks.truth(prefix + "_status", row.status == status, index)
                checks.close(prefix + "_denominator_defined", row.n_defined, n_defined, index, "integer")
                for column, value in metadata.items():
                    if strata and column in ("reference_auc", "valid_hours", "selected_hours") and column not in row.index:
                        continue
                    kind = ("integer" if column in ("n_patients", "n_patients_with_events", "n_records") else
                            "component" if column == "reference_auc" else "count" if column == "events" else "metric")
                    checks.close(prefix + "_" + column, row[column], value, index, kind)
                point_rows += 1
                if status != "CI":
                    checks.truth(prefix + "_non_CI_missing_bounds", np.isnan(row.lo) and np.isnan(row.hi), index)
                elif check_ci:
                    draws = (pooled_draws[metric] if estimand == "pooled" else
                             safe_divide(weights @ np.where(defined, values, 0.), weights @ defined.astype(float)))
                    checks.close(prefix + "_selected_CI", [row.lo, row.hi], percentile_interval(draws), index, "metric")
                    ci_rows += 1
    checks.truth(prefix + "_no_unmatched_rows", set(indexed.index) == seen)
    return {"groups": groups, "point_rows_verified": point_rows, "selected_CI_rows_verified": ci_rows,
            "CI_scope": "All primary MAP65/5min groups" if not strata else "All primary MAP65/5min feature strata meeting frozen CI gate"}


def paired_groups(frame, summary, checks):
    selected = frame[(frame.scenario == "paired_bridge") & (frame.threshold == 65)]
    keys = ["source", "interval_min", "estimand", "metric"]
    indexed = summary.set_index(keys)
    checks.truth("paired_summary_unique_rows", ~summary.duplicated(keys))
    seen = set()
    point_rows, ci_rows, groups = 0, 0, 0
    for (source, step), group in selected.groupby(["source", "interval_min"], sort=True):
        groups += 1
        p, native = patients_first(group[group.resolution == "10s"])
        q, minute = patients_first(group[group.resolution == "1min"])
        checks.truth("paired_patient_positions", np.array_equal(p, q), (source, step))
        if not np.array_equal(p, q):
            continue
        checks.close("paired_selected_patient_hours", native[:, 12], minute[:, 12], (source, step), "count")
        old, new = metric_values(native), metric_values(minute)
        old_pooled, new_pooled = metric_values(native.sum(axis=0)), metric_values(minute.sum(axis=0))
        do_ci = step == 5
        weights = bootstrap_weights(len(p)) if do_ci else None
        if do_ci:
            old_draws, new_draws = metric_values(weights @ native), metric_values(weights @ minute)
        for metric in old:
            common = np.isfinite(old[metric]) & np.isfinite(new[metric])
            for estimand in ("pooled", "patient_equal_common_defined"):
                index = (source, step, estimand, metric)
                seen.add(index)
                exists = index in indexed.index
                checks.truth("paired_summary_row_exists", exists, index)
                if not exists:
                    continue
                row = indexed.loc[index]
                if estimand == "pooled":
                    before, after = float(old_pooled[metric]), float(new_pooled[metric])
                    delta = after - before
                    count = len(p)
                    draws = new_draws[metric] - old_draws[metric] if do_ci else None
                else:
                    count = int(common.sum())
                    before = float(old[metric][common].mean()) if count else np.nan
                    after = float(new[metric][common].mean()) if count else np.nan
                    difference = new[metric] - old[metric]
                    delta = float(difference[common].mean()) if count else np.nan
                    draws = (safe_divide(weights @ np.where(common, difference, 0.), weights @ common.astype(float))
                             if do_ci else None)
                status = "CI" if count >= 20 else "POINT_ONLY" if count >= 10 else "INSUFFICIENT"
                if status == "INSUFFICIENT":
                    before = after = delta = np.nan
                checks.close("paired_summary_points", [row.native10s, row.minute, row.minute_minus_native],
                             [before, after, delta], index, "metric")
                checks.close("paired_summary_counts", [row.n_patients, row.n_defined_common, row.event_min_sec],
                             [len(p), count, 180], index, "integer")
                checks.truth("paired_summary_status", row.status == status, index)
                if do_ci and status == "CI":
                    checks.close("paired_summary_selected_CI", [row.lo, row.hi], percentile_interval(draws), index, "metric")
                    ci_rows += 1
                elif status != "CI":
                    checks.truth("paired_summary_non_CI_missing_bounds", np.isnan(row.lo) and np.isnan(row.hi), index)
                point_rows += 1
    checks.truth("paired_summary_no_unmatched_rows", set(indexed.index) == seen)
    return {"groups": groups, "point_rows_verified": point_rows, "selected_CI_rows_verified": ci_rows,
            "CI_scope": "MAP65/5min paired resolution contrasts, patient-cluster bootstrap"}


def verify_distributions(frame, table, checks):
    indexed = table.set_index(GROUP_KEYS + ["metric"])
    checks.truth("patient_distribution_unique_rows", ~table.duplicated(GROUP_KEYS + ["metric"]))
    checked = 0
    seen = set()
    for key, group in frame[(frame.scenario == "primary") & (frame.threshold == 65) & (frame.interval_min == 5)].groupby(GROUP_KEYS, sort=True):
        ids, c = patients_first(group)
        for metric, values in metric_values(c).items():
            index = tuple(key) + (metric,)
            seen.add(index)
            exists = index in indexed.index
            checks.truth("patient_distribution_exists", exists, index)
            if not exists:
                continue
            row = indexed.loc[index]
            finite = values[np.isfinite(values)]
            quantiles = np.quantile(finite, [.05, .25, .5, .75, .95]) if finite.size else np.full(5, np.nan)
            checks.close("patient_distribution_quantiles", row[["p05", "p25", "p50", "p75", "p95"]].to_numpy(float),
                         quantiles, index, "metric")
            checks.close("patient_distribution_denominators", [row.n_patients, row.n_defined],
                         [len(ids), len(finite)], index, "integer")
            checked += 1
    checks.truth("patient_distribution_no_unmatched_rows", set(indexed.index) == seen)
    return checked


def verify_cohort_flow(frame, table, flow, checks):
    keys = ["source", "resolution", "scenario"]
    indexed = table.set_index(keys)
    checks.truth("cohort_flow_unique_rows", ~table.duplicated(keys))
    primary = frame[(frame.threshold == 65) & (frame.interval_min == 5)]
    row_keys = ["source", "record_id", "subject_id", "resolution", "scenario", "event_min_sec"]
    declared = flow.set_index(row_keys)
    component = primary.set_index(row_keys)
    checks.truth("flow_unique_record_rows", ~flow.duplicated(row_keys))
    checks.truth("flow_record_component_exists", declared.index.isin(component.index))
    if declared.index.isin(component.index).all():
        reference = component.loc[declared.index]
        checks.close("flow_record_selected_hours", declared.selected_hours.to_numpy(float),
                     reference.selected_minutes.to_numpy(float) / 60, kind="metric")
        checks.close("flow_record_valid_hours", declared.valid_hours.to_numpy(float),
                     reference.valid_reference_min.to_numpy(float) / 60, kind="metric")
    seen = set()
    for key, declared_group in flow.groupby(keys, sort=True):
        seen.add(key)
        checks.truth("cohort_flow_exists", key in indexed.index, key)
        if key not in indexed.index:
            continue
        declared_ids = declared_group.record_id.unique()
        group = primary[(primary.source == key[0]) & (primary.resolution == key[1]) &
                        (primary.scenario == key[2]) & primary.record_id.isin(declared_ids)]
        checks.truth("cohort_flow_one_row_per_record", ~group.duplicated("record_id"), key)
        row = indexed.loc[key]
        counts = [group.record_id.nunique(), group.subject_id.nunique(),
                  int(group.age_harmonization.eq("age25").sum()), int(group.age_harmonization.ne("age25").sum())]
        checks.close("cohort_flow_counts", [row.n_records, row.n_patients, row.age25_verified_records, row.age25_not_independently_verified_records],
                     counts, key, "integer")
        checks.close("cohort_flow_hours", [row.selected_hours, row.valid_hours],
                     [group.selected_minutes.sum() / 60, group.valid_reference_min.sum() / 60], key, "metric")
    checks.truth("cohort_flow_no_unmatched_rows", set(indexed.index) == seen)
    return {"groups_verified": len(seen),
            "scope": "All declared restricted-flow groups; cached SICdb sensitivity scenarios without flow rows are not asserted to be in this table.",
            "age_flag_caveat": "SICdb age25 flag means frozen encoded source-age cutoff only, not a verified true-age lower bound."}


def verify_standardization(stratum, table, gate, checks):
    bands = ["180_to_lt300", "300_to_lt600", "ge600"]
    part = stratum[(stratum.resolution == "1min") & (stratum.scenario == "primary") &
                   (stratum.feature == "duration") & stratum.level.isin(bands)]
    vital_events = part[part.source == "VitalDB"].groupby("level").episodes.sum().reindex(bands).to_numpy(float)
    target_weights = vital_events / vital_events.sum()
    sparse_gate = True
    for source in ("VitalDB", "MIMIC", "SICdb"):
        for band in bands:
            rows = part[(part.source == source) & (part.level == band)]
            patients = rows.groupby("subject_id").episodes.sum()
            sparse_gate &= bool((patients > 0).sum() >= 20 and patients.sum() >= 50)
    checks.truth("standardization_frozen_gate", gate["pass_gate"] == sparse_gate)
    checks.truth("standardization_fixed_bands", gate["bands"] == bands)
    if not sparse_gate:
        checks.truth("standardization_not_performed_if_sparse", table is None)
        return {"gate_passed": False, "rows_verified": 0}
    checks.truth("standardization_output_present", table is not None)
    if table is None:
        return {"gate_passed": True, "rows_verified": 0}
    indexed = table.set_index(["source", "metric"])
    checks.truth("standardization_unique_rows", ~table.duplicated(["source", "metric"]))
    seen = set()
    for source, group in part.groupby("source", sort=True):
        ids = sorted(group.subject_id.unique())
        components = np.stack([
            group[group.level == band].groupby("subject_id")[list(E6)].sum().reindex(ids, fill_value=0.).to_numpy(float)
            for band in bands], axis=1)
        weights = bootstrap_weights(len(ids))
        total = components.sum(axis=0)
        for metric, column in (("fresh_probability", 1), ("inherited_probability", 2), ("miss_probability", 3)):
            index = (source, metric)
            seen.add(index)
            checks.truth("standardization_row_exists", index in indexed.index, index)
            if index not in indexed.index:
                continue
            row = indexed.loc[index]
            expected = float(target_weights @ safe_divide(total[:, column], total[:, 0]))
            fractions = np.column_stack([
                safe_divide(weights @ components[:, j, column], weights @ components[:, j, 0]) for j in range(3)])
            bounds = percentile_interval(fractions @ target_weights)
            checks.close("standardization_point_CI", [row.estimate, row.lo, row.hi],
                         [expected, *bounds], index, "metric")
            checks.close("standardization_patient_count", row.n_patients, len(ids), index, "integer")
            parsed = dict(item.split(":", 1) for item in row.standard_weights.split(";"))
            checks.close("standardization_fixed_weights", [float(parsed[band]) for band in bands], target_weights, index, "metric")
    checks.truth("standardization_no_unmatched_rows", set(indexed.index) == seen)
    return {"gate_passed": True, "rows_verified": len(seen), "CI_rows_verified": len(seen),
            "interpretation": "Conditional on fixed VitalDB event-duration weights; no causal or between-database effect inference."}


def run_verification(root, sources, sample_size, select_only=False):
    import pandas as pd
    root = Path(root).resolve()
    checks = Checks()
    details = {}
    report = {
        "schema_version": "1.0-independent-cache-numerical-verification",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": CONTRACT_SHA, "verifier_sha256": digest(__file__),
        "seed": SEED, "requested_records_per_scope": sample_size,
        "frozen_input_manifest_sha256": digest(root / "qa/frozen_inputs.json"),
        "independence": {
            "imports_parent_analysis_modules": False,
            "native_waveforms_reread": False,
            "uses_frozen_upstream_QC_numerical_references": True,
            "manuscript_or_public_archive_modified": False,
            "claim_boundary": "Selected cached-reference records rederived independently; all present output rows checked for additive identities and partitions. This is not a new native-waveform audit, global validated-source claim, clinical validation, or clinical subgroup validation.",
        },
        "tolerances": {"components_atol": 1e-7, "components_rtol": 1e-10,
                       "metrics_CI_atol": 1e-9, "metrics_CI_rtol": 1e-9, "phase_counts_atol": 1e-9},
        "sources": {}, "summary_verification": {}, "pending_outputs": [],
    }
    try:
        test_result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundaryTests))
        report["boundary_tests"] = {"run": test_result.testsRun, "failures": len(test_result.failures), "errors": len(test_result.errors)}
        checks.truth("independent_boundary_tests", test_result.wasSuccessful())
        report["protected_inputs_before"] = protected_hashes(root, checks, details, "before")
        if not report["protected_inputs_before"]["all_local_hashes"]:
            raise ValueError("Protected local inputs differ from frozen manifest")
        addendum = root / "source_interpretation_addendum.json"
        if addendum.exists():
            report["source_interpretation_addendum_sha256"] = digest(addendum)
        report["age_scope"] = {
            "VitalDB": "Frozen recorded age cutoff >=25",
            "MIMIC": "Verified deidentified year-level age cutoff >=25, not birthday-exact age",
            "SICdb": "Frozen source-encoded age-code cutoff >=25; not a verified lower-bound or exact-age match",
        }
        selection, cached, identities = choose_before_results(root, sources, sample_size, checks)
        report["selection_sha256"] = digest(root / "restricted/independent_selection.json")
        assignment_sample = None
        if "SICdb" in sources:
            assignment_sample = choose_assignment_before_results(root, sample_size, checks)
        frames, strata_frames, flow_frames = [], [], []
        if not select_only:
            for source in sources:
                component_path = root / f"restricted/{source}_components.csv.gz"
                stratum_path = root / f"restricted/{source}_strata.csv.gz"
                flow_path = root / f"restricted/{source}_flow.csv.gz"
                if not component_path.exists() or not stratum_path.exists() or not flow_path.exists():
                    report["pending_outputs"].append(source + " component/strata/flow outputs")
                    continue
                before = [digest(component_path), digest(stratum_path), digest(flow_path)]
                frame = pd.read_csv(component_path, dtype={"record_id": str, "subject_id": str})
                stratum = pd.read_csv(stratum_path, dtype={"record_id": str, "subject_id": str})
                flow = pd.read_csv(flow_path, dtype={"record_id": str, "subject_id": str})
                if before != [digest(component_path), digest(stratum_path), digest(flow_path)]:
                    report["pending_outputs"].append(source + " outputs changed during verification; rerun required")
                    continue
                checks.truth(source + "_restricted_permissions", [
                    (root / "restricted").stat().st_mode & 0o777 == 0o700,
                    component_path.stat().st_mode & 0o777 == 0o600,
                    stratum_path.stat().st_mode & 0o777 == 0o600,
                    flow_path.stat().st_mode & 0o777 == 0o600])
                verify_identities(frame, stratum, source, checks, identities[source])
                sample = compare_sample(frame, stratum, source, selection["sources"][source], cached[source], checks)
                report["sources"][source] = {"component_rows": len(frame), "stratum_rows": len(stratum),
                    "record_subject_links_verified": int(frame.record_id.nunique()),
                    "component_sha256": before[0], "strata_sha256": before[1], "flow_sha256": before[2], **sample}
                if source == "SICdb" and assignment_sample is not None:
                    extra_selection, extra_cached, grids = assignment_sample
                    extra = compare_sample(frame, stratum, source, extra_selection, extra_cached, checks,
                                           scope_provider=lambda rid, values: grids[rid], label="SICdb_assignment")
                    report["sources"][source]["assignment_subgroups"] = {
                        **extra,
                        "selection_sha256": digest(root / "restricted/independent_icu_selection.json"),
                        "metadata_sha256": extra_selection["metadata_sha256"],
                        "eligible_record_counts": extra_selection["eligible_record_counts"],
                        "rule": "Mask minute bins whose start is before ICUOffset, then origin-anchored complete-hour >=48/60 finite gate; intersect primary qualifying hours. Build display from the masked full grid, not from preassignment history.",
                        "interpretation": "Qualifying hours may cross first ICU/intermediate-care transfer. This does not establish continuous ICU residence, surgical-end timing, or postoperative status.",
                        "surgical_labels": "Explicit audited source categories Elective Surgery/Urgent Surgery versus No Surgery; this numerical check does not independently validate the clinical phenotype mapping.",
                    }
                frames.append(frame)
                strata_frames.append(stratum)
                flow_frames.append(flow)
            files = {"summary": "summary.csv", "strata": "event_strata.csv", "paired": "paired_resolution.csv",
                     "distributions": "patient_distributions.csv", "cohort_flow": "cohort_flow.csv"}
            if len(frames) == 3:
                frame = pd.concat(frames, ignore_index=True)
                stratum = pd.concat(strata_frames, ignore_index=True)
                flow = pd.concat(flow_frames, ignore_index=True)
                for name, filename in files.items():
                    path = root / "aggregate" / filename
                    if not path.exists():
                        report["pending_outputs"].append(filename)
                        continue
                    sha_before = digest(path)
                    table = pd.read_csv(path)
                    if name in ("summary", "strata"):
                        outcome = aggregate_groups(stratum if name == "strata" else summary_component_frame(frame), table, checks,
                                                   "strata_summary" if name == "strata" else "summary", name == "strata")
                    elif name == "paired":
                        outcome = paired_groups(frame, table, checks)
                    elif name == "cohort_flow":
                        outcome = verify_cohort_flow(frame, table, flow, checks)
                    else:
                        outcome = {"rows_verified": verify_distributions(frame, table, checks)}
                    report["summary_verification"][name] = {**outcome, "sha256": sha_before}
                    checks.truth(name + "_output_unchanged_during_check", digest(path) == sha_before)
                gate_path = root / "aggregate/standardization_gate.json"
                standardized_path = root / "aggregate/duration_standardized.csv"
                if gate_path.exists():
                    gate_sha = digest(gate_path)
                    gate = json.loads(gate_path.read_text())
                    standard_sha = digest(standardized_path) if standardized_path.exists() else None
                    standardized = pd.read_csv(standardized_path) if standardized_path.exists() else None
                    outcome = verify_standardization(stratum, standardized, gate, checks)
                    report["summary_verification"]["duration_standardized"] = {**outcome, "sha256": standard_sha, "gate_sha256": gate_sha}
                    checks.truth("standardization_gate_unchanged_during_check", digest(gate_path) == gate_sha)
                    if standard_sha:
                        checks.truth("standardization_output_unchanged_during_check", digest(standardized_path) == standard_sha)
                else:
                    report["pending_outputs"].append("standardization_gate.json")
            else:
                report["pending_outputs"].append("All-source summaries require all three completed component tables")
        else:
            report["pending_outputs"].append("Selection-only run; output comparisons not requested in this invocation")
        report["protected_inputs_after"] = protected_hashes(root, checks, details, "after")
        checks.truth("frozen_manifest_unchanged_during_check", digest(root / "qa/frozen_inputs.json") == report["frozen_input_manifest_sha256"])
        if addendum.exists():
            checks.truth("source_interpretation_unchanged_during_check", digest(addendum) == report["source_interpretation_addendum_sha256"])
        if assignment_sample is not None:
            checks.truth("SICdb_assignment_metadata_unchanged_during_check",
                         digest(root / "restricted/subgroup_metadata.csv") == assignment_sample[0]["metadata_sha256"])
        checks.truth("verifier_code_unchanged_during_check", digest(__file__) == report["verifier_sha256"])
        for source, outcome in report["sources"].items():
            checks.truth(source + "_components_unchanged_until_end", digest(root / f"restricted/{source}_components.csv.gz") == outcome["component_sha256"])
            checks.truth(source + "_strata_unchanged_until_end", digest(root / f"restricted/{source}_strata.csv.gz") == outcome["strata_sha256"])
            checks.truth(source + "_flow_unchanged_until_end", digest(root / f"restricted/{source}_flow.csv.gz") == outcome["flow_sha256"])
    except Exception as error:
        import traceback
        details["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["execution_error_type"] = type(error).__name__
        checks.truth("verification_execution_completed", False)
    report["checks"] = checks.public()
    report["failed_comparisons"] = sum(x["failures"] for x in checks.groups.values())
    report["status"] = ("FAIL" if report["failed_comparisons"] else
                        "PARTIAL_PENDING_OUTPUTS" if report["pending_outputs"] else "PASS_SCOPED")
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    details["failures"] = checks.failures
    write_json(root / "restricted/independent_details.json", details, private=True)
    write_json(root / "qa/independent_verification.json", report)
    print(json.dumps({"status": report["status"], "failed_comparisons": report["failed_comparisons"],
                      "sources_checked": list(report["sources"]), "pending_output_groups": len(report["pending_outputs"])}))
    return 1 if report["failed_comparisons"] else 0


class IndependentBoundaryTests(unittest.TestCase):
    def test_assignment_rule_allows_crossing_hour_at_twelve_minutes_but_not_later(self):
        values = np.full(120, 80.)
        masked, selected, crossing = assignment_reference(values, 720.)
        self.assertTrue(np.isnan(masked[:12]).all())
        self.assertTrue(selected.all())
        self.assertEqual(crossing, 1)
        masked, selected, crossing = assignment_reference(values, 720.001)
        self.assertTrue(np.isnan(masked[:13]).all())
        self.assertFalse(selected[:60].any())
        self.assertTrue(selected[60:].all())
        self.assertEqual(crossing, 0)

    def test_assignment_mask_excludes_preassignment_display_and_missing_inside_counts(self):
        values = np.r_[np.full(12, 50.), np.full(48, 80.)]
        masked, selected, _ = assignment_reference(values, 720.)
        c, _, _ = estimate(masked, selected, 60)
        self.assertEqual(c[1], 0.)
        self.assertEqual(c[13], 0.)
        self.assertEqual(c[10], 48.)
        self.assertEqual(c[12], 60.)
        values[20] = np.nan
        _, selected, _ = assignment_reference(values, 720.)
        self.assertFalse(selected.any())
        for offset in (-1., np.nan, 3600.):
            _, selected, crossing = assignment_reference(np.full(60, 80.), offset)
            self.assertFalse(selected.any())
            self.assertEqual(crossing, 0)

    def test_event_strata_summary_allows_absent_not_applicable_pressure_columns(self):
        import pandas as pd
        rows = []
        for i in range(10):
            rows.append({"source": "Synthetic", "resolution": "1min", "scenario": "primary", "event_min_sec": 180,
                         "feature": "duration", "level": "ge600", "subject_id": str(i), "record_id": str(i),
                         "episodes": 1., "fresh": .6, "inherited_only": .2, "never_low": .2, "visible": .8, "delay_sum_min": .4})
        expected = []
        for estimand in ("pooled", "patient_equal"):
            for metric, point in (("fresh_probability", .6), ("inherited_probability", .2),
                                  ("miss_probability", .2), ("mean_visible_delay_min", .5)):
                expected.append({"source": "Synthetic", "resolution": "1min", "scenario": "primary", "event_min_sec": 180,
                                 "feature": "duration", "level": "ge600", "estimand": estimand, "metric": metric,
                                 "estimate": point, "lo": np.nan, "hi": np.nan, "status": "POINT_ONLY", "n_patients": 10,
                                 "n_defined": 10, "n_patients_with_events": 10, "n_records": 10, "events": 10.})
        checks = Checks()
        aggregate_groups(pd.DataFrame(rows), pd.DataFrame(expected), checks, "strata_synthetic", strata=True)
        self.assertFalse(any(group["failures"] for group in checks.groups.values()))

    def test_verifier_does_not_import_parent_analysis_modules(self):
        import ast
        tree = ast.parse(Path(__file__).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
        self.assertTrue(set(names).isdisjoint({"engine", "run_analysis", "io_sources", "summarize"}))

    def test_cohort_flow_covers_declared_rows_not_unlisted_cached_scenarios(self):
        import pandas as pd
        c, _, _ = estimate(np.full(60, 80.), np.ones(60, bool), 60)
        meta = {"source": "SICdb", "record_id": "r", "subject_id": "p", "resolution": "1min",
                "event_min_sec": 180, "threshold": 65, "interval_min": 5,
                "age_harmonization": "encoded_age25_not_exactly_matched"}
        components = pd.DataFrame([{**meta, "scenario": scenario, **dict(zip(COLS, c))}
                                   for scenario in ("primary", "baseline_initialization")])
        flow = pd.DataFrame([{**meta, "scenario": "primary", "selected_hours": 1., "valid_hours": 1.}])
        table = pd.DataFrame([{"source": "SICdb", "resolution": "1min", "scenario": "primary", "n_records": 1,
                               "n_patients": 1, "age25_verified_records": 0,
                               "age25_not_independently_verified_records": 1, "selected_hours": 1., "valid_hours": 1.}])
        checks = Checks()
        outcome = verify_cohort_flow(components, table, flow, checks)
        self.assertEqual(outcome["groups_verified"], 1)
        self.assertFalse(any(group["failures"] for group in checks.groups.values()))

    def test_matched_onset_summary_uses_record_resolution_not_all_patients(self):
        import pandas as pd
        rows = []
        for rid, resolution, scenario in (("a", "10s", "primary"), ("a", "1min", "primary"),
                                           ("b", "10s", "primary"), ("a", "10s", "core60_fixed")):
            rows.append({"source": "VitalDB", "record_id": rid, "resolution": resolution,
                         "scenario": scenario, "threshold": 65, "interval_min": 5})
        expanded = summary_component_frame(pd.DataFrame(rows))
        matched = expanded[expanded.scenario == "core60_matched_original"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0].record_id, "a")
        self.assertEqual(matched.iloc[0].resolution, "10s")

    def test_conditional_standardization_uses_fixed_vital_weights(self):
        import pandas as pd
        bands = ["180_to_lt300", "300_to_lt600", "ge600"]
        episodes = np.array([3., 6., 9.])
        fresh, inherited = np.array([1., 3., 6.]), np.array([1., 2., 2.])
        weights = episodes / episodes.sum()
        rows, expected = [], []
        for source in ("VitalDB", "MIMIC", "SICdb"):
            for patient in range(20):
                for j, band in enumerate(bands):
                    c = [episodes[j], fresh[j], inherited[j], 1., fresh[j] + inherited[j], 0.]
                    rows.append({"source": source, "resolution": "1min", "scenario": "primary", "feature": "duration",
                                 "level": band, "subject_id": str(patient), **dict(zip(E6, c))})
            for metric, numerator in (("fresh_probability", fresh), ("inherited_probability", inherited),
                                      ("miss_probability", np.ones(3))):
                point = float(weights @ (numerator / episodes))
                expected.append({"source": source, "metric": metric, "estimate": point, "lo": point, "hi": point,
                                 "n_patients": 20, "standard_weights": ";".join(f"{band}:{weight:.12g}" for band, weight in zip(bands, weights))})
        checks = Checks()
        verify_standardization(pd.DataFrame(rows), pd.DataFrame(expected), {"pass_gate": True, "bands": bands}, checks)
        self.assertFalse(any(group["failures"] for group in checks.groups.values()))

    def test_additional_source_specific_stratum_scenarios_are_not_false_failures(self):
        import pandas as pd
        values = np.array([80, 50, 50, 50, 80, 80.])
        c, strata, _ = estimate(values, np.ones(6, bool), 60)
        rows, srows = [], []
        for scenario in ("primary", "source_specific_subgroup"):
            metadata = {"source": "Synthetic", "record_id": "r", "subject_id": "s", "resolution": "1min",
                        "scenario": scenario, "event_min_sec": 180, "threshold": 65, "interval_min": 5}
            rows.append({**metadata, **dict(zip(COLS, c))})
            srows.extend({**metadata, "feature": feature, "level": level, **dict(zip(E6, z))}
                         for (feature, level), z in strata.items())
        checks = Checks()
        verify_identities(pd.DataFrame(rows), pd.DataFrame(srows), "Synthetic", checks, {"r": "s"})
        self.assertFalse(any(group["failures"] for group in checks.groups.values()))

    def test_unavailable_excludes_reference_missing_and_bootstrap_keeps_undefined(self):
        values = np.array([50., np.nan, 80.])
        components, _, _ = estimate(values, np.ones(3, bool), 60, interval=5)
        self.assertAlmostEqual(components[10], 2)
        self.assertAlmostEqual(components[11], .6)
        self.assertAlmostEqual(float(metric_values(components)["unavailable_fraction"]), .7)
        metric = np.array([np.nan, .5])
        finite = np.isfinite(metric)
        weights = np.array([[2., 0.], [0., 2.], [1., 1.]])
        draws = safe_divide(weights @ np.where(finite, metric, 0.), weights @ finite.astype(float))
        np.testing.assert_allclose(draws, [np.nan, .5, .5], equal_nan=True)

    def test_shared_hour_intersection_and_pressure_range_endpoints(self):
        values = np.full(1080, 80.)
        values[np.arange(13) * 6] = np.nan
        values[np.arange(13) * 6 + 1] = np.nan
        values[360:432] = np.nan
        values[438] = np.nan
        scopes = scope_masks("VitalDB", values)
        mask = scopes["paired_bridge_10s"][1]
        self.assertFalse(mask[:720].any())
        self.assertTrue(mask[720:].all())
        np.testing.assert_array_equal(mask, np.repeat(scopes["paired_bridge_1min"][1], 6))
        np.testing.assert_allclose(pressure_filter([19.9, 20, 180, 180.1, np.inf]),
                                   [np.nan, 20, 180, np.nan, np.nan], equal_nan=True)

    def test_randomized_identity_and_complete_run_sampling_baseline(self):
        rng = np.random.default_rng(991)
        for dt in (10, 60):
            for _ in range(12):
                values = rng.choice([50., 54., 55., 60., 65., 80., np.nan], 90)
                selected = rng.random(90) > .1
                c, strata, diagnostic = estimate(values, selected, dt, minimum=dt)
                self.assertAlmostEqual(c[1] - c[0], c[3] - c[2], places=9)
                self.assertAlmostEqual(c[5] - c[6], c[7] + c[9] - c[8], places=9)
                self.assertAlmostEqual(c[13], c[14:17].sum(), places=9)
                self.assertLess(diagnostic["analytic_error"], 1e-9)
                for feature in LEVELS:
                    np.testing.assert_allclose(sum(z for (f, _), z in strata.items() if f == feature), c[13:], atol=1e-9)

    def test_zero_event_strata_retain_all_predeclared_levels(self):
        _, strata, _ = estimate(np.full(60, 80.), np.ones(60, bool), 60)
        self.assertEqual(set(strata), {(feature, level) for feature, levels in LEVELS.items() for level in levels})
        for values in strata.values():
            np.testing.assert_array_equal(values, np.zeros(len(E6)))

    def test_patient_equal_combines_records_before_deriving_and_sorts_strings(self):
        import pandas as pd
        frame = pd.DataFrame(np.zeros((3, len(COLS))), columns=COLS)
        frame["subject_id"] = ["2", "2", "10"]
        frame["reference_auc"] = [1., 9., 10.]
        frame["hidden_auc"] = [1., 0., 5.]
        ids, values = patients_first(frame)
        self.assertEqual(ids.tolist(), ["10", "2"])
        actual = metric_values(values)["hidden_ratio"]
        np.testing.assert_allclose(actual, [.5, .1])
        self.assertAlmostEqual(actual.mean(), .3)

    def test_undefined_ratios_remain_missing_but_zero_numerator_is_defined(self):
        values = np.zeros((2, len(COLS)))
        values[1, 0] = 10
        ratios = metric_values(values)["hidden_ratio"]
        self.assertTrue(np.isnan(ratios[0]))
        self.assertEqual(ratios[1], 0)

    def test_minute_exact_five_of_six_and_no_trailing_partial_minute(self):
        values = np.r_[np.arange(6) + 60, [60, 61, 62, 63, np.nan, np.nan],
                       [60, 61, 62, 63, 64, np.nan], [50, 50]]
        self.assertEqual(len(minute_reference(values)), 3)
        np.testing.assert_allclose(minute_reference(values), [62.5, np.nan, 62], equal_nan=True)

    def test_origin_hours_exact_eighty_percent_and_no_compression(self):
        values = np.full(360 * 3 + 10, 80.)
        values[:72] = np.nan
        values[360:433] = np.nan
        chosen = qualifying_hours(values, 10)
        self.assertTrue(chosen[:360].all())
        self.assertFalse(chosen[360:720].any())
        self.assertTrue(chosen[720:1080].all())
        self.assertFalse(chosen[1080:].any())

    def test_hold_uses_full_grid_and_missing_sample_does_not_update(self):
        values = np.array([50, 80, 80, 80, 80, np.nan, 80.])
        display, origin = held_display(values, 5, 0)
        np.testing.assert_allclose(display, [50] * 7)
        np.testing.assert_array_equal(origin, [0] * 7)
        display, _ = held_display(values, 5, 3)
        self.assertTrue(np.isnan(display[:3]).all())
        self.assertEqual(display[3], 80)

    def test_hand_calculated_thirteen_components_and_six_events(self):
        values = np.array([80, 50, 50, 50, 80, 80.])
        actual, strata, diagnostics = estimate(values, np.ones(6, bool), 60)
        expected = [45, 36, 27, 18, 18, 3, 2.4, .6, 1.2, 1.2, 6, 4, 6,
                    1, .6, 0, .4, .6, .6]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(strata[("preceding_normal", "unknown")], expected[13:])
        self.assertAlmostEqual(diagnostics["expected_fresh"], .6)

    def test_preceding_gap_requires_observed_low_before_entire_normal_segment(self):
        values = np.array([50, 80, 80, 50, 50, np.nan, 80, 50, 50.])
        runs = low_runs(values, np.ones(len(values), bool), 60, 65, 120)
        self.assertEqual([(r[0], r[1]) for r in runs], [(3, 5), (7, 9)])
        self.assertEqual(runs[0][3], 120)
        self.assertTrue(np.isnan(runs[1][3]))
        selected = np.ones(len(values), bool)
        selected[1] = False
        self.assertTrue(np.isnan(low_runs(values, selected, 60, 65, 120)[0][3]))

    def test_duration_nadir_and_normal_gap_boundaries(self):
        lengths = [6, 18, 30, 60]
        values = np.r_[50., np.full(6, 80.), *[
            x for length in lengths for x in (np.full(length, 54.), np.full(30, 80.))]]
        _, strata, _ = estimate(values, np.ones(len(values), bool), 10, minimum=60)
        for level in ("60_to_lt180", "180_to_lt300", "300_to_lt600", "ge600"):
            self.assertAlmostEqual(strata[("duration", level)][0], 1)
        self.assertAlmostEqual(strata[("nadir", "lt55")][0], 4)
        self.assertAlmostEqual(strata[("preceding_normal", "60_to_lt300")][0], 1)
        self.assertAlmostEqual(strata[("preceding_normal", "ge300")][0], 3)

    def test_fresh_inherited_and_never_are_disjoint(self):
        values = np.r_[np.full(12, 50.), np.full(6, 80.), np.full(6, 50.), np.full(12, 80.)]
        component, _, _ = estimate(values, np.ones(len(values), bool), 10, minimum=60)
        self.assertEqual(component[13], 2)
        self.assertAlmostEqual(component[13], component[14:17].sum())
        self.assertGreater(component[15], 0)
        self.assertAlmostEqual(component[17], component[14] + component[15])

    def test_evaluation_gap_breaks_events_but_not_display_history(self):
        values = np.full(8, 50.)
        selected = np.array([True, True, False, False, True, True, True, True])
        runs = low_runs(values, selected, 60, 65, 180)
        self.assertEqual([(r[0], r[1]) for r in runs], [(4, 8)])
        component, _, _ = estimate(values, selected, 60)
        self.assertGreater(component[15], 0)
        self.assertAlmostEqual(component[0] - component[1], component[2] - component[3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sources", nargs="+", choices=["VitalDB", "MIMIC", "SICdb"], default=["VitalDB", "MIMIC", "SICdb"])
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], exit=True)
    if args.sample_size < 8:
        parser.error("Frozen verification requires at least eight records per available scope")
    sys.exit(run_verification(args.root, args.sources, args.sample_size, args.select_only))
