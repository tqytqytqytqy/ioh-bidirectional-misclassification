"""Independent source-time audit. Never fill missing raw timestamps."""
from __future__ import annotations

import io
import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.dont_writebytecode = True
PROJECT = Path(__file__).resolve().parents[2]
DEPS = PROJECT / "外部验证可行性审计_20261015/mimic/private/deps"
sys.path.insert(0, str(DEPS))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT / "启动段质量审计_20261007/private_case_audit"
TRACK_ROOTS = [Path(p) for p in os.environ.get('VITALDB_TRACK_ROOTS', 'data/vitaldb/tracks').split(os.pathsep)]
STRATA = ("LOW_OUTSIDE_RANGE", "LOW_IN_RANGE", "MAP40_TO_LT65", "MAP_GE65")
BIN_NUMBERS = (0, 6, 30, 59)
METRIC_TOLERANCES = {"wave_coverage": 1e-10, "wave_mean": 1e-8, "wave_p05": 1e-8,
                     "wave_p95": 1e-8, "wave_span_90pct": 1e-8, "peak_count": 0.0}
TIME_ATOL = 1e-8
SEED = 20260706


def summarize_bin(times, values, left, fs):
    t, v = np.asarray(times, float), np.asarray(values, float)
    keep = np.isfinite(t) & (t >= left) & (t < left + 10.)
    x = v[keep]
    good = np.isfinite(x)
    r = {"row_count": int(len(x)), "finite_value_count": int(good.sum()),
         "wave_coverage": float(good.sum() / (10 * fs)) if np.isfinite(fs) and fs > 0 else np.nan,
         "wave_mean": np.nan, "wave_p05": np.nan, "wave_p95": np.nan,
         "wave_span_90pct": np.nan, "peak_count": 0,
         "peak_distance_samples": int(.3 * fs) if np.isfinite(fs) and fs > 0 else None}
    if good.any():
        lo, hi = np.percentile(x[good], [5, 95], method="linear")
        r.update(wave_mean=float(np.mean(x[good])), wave_p05=float(lo), wave_p95=float(hi),
                 wave_span_90pct=float(hi - lo))
        if r["peak_distance_samples"] is None or r["peak_distance_samples"] < 1:
            r["peak_count"] = np.nan
        else:
            y = np.nan_to_num(x, nan=float(np.median(x[good])))
            r["peak_count"] = int(len(find_peaks(y, distance=r["peak_distance_samples"],
                                                 prominence=5)[0]))
    return r


def nominal_index(edge, origin, interval):
    i = max(0, int(math.floor((edge - origin) / interval)))
    while i > 0 and origin + (i - 1) * interval >= edge:
        i -= 1
    while origin + i * interval < edge:
        i += 1
    return i


def scan_csv_stream(stream, left_edges, chunksize=200000):
    if chunksize < 2:
        raise ValueError("chunksize must include the first two physical rows")
    r = {"source_rows": 0, "time_finite_rows": 0, "time_missing_rows": 0,
         "time_invalid_nonblank_rows": 0, "value_finite_rows": 0,
         "observed_time_regressions": 0, "observed_time_duplicates": 0,
         "observed_adjacent_interval_count": 0, "adjacent_interval_deviates_from_nominal": 0,
         "max_explicit_vs_rebuilt_abs_sec": 0., "explicit_vs_rebuilt_disagreement_count": 0,
         "explicit_row_indices_min": None, "explicit_row_indices_max": None}
    parts = [[] for _ in left_edges]
    diagnostics = [dict(nominal_row_count=0, nominal_rows_with_missing_time=0,
                        explicit_membership_disagreement=0, exact_left_boundary_rows=0,
                        exact_right_boundary_rows=0) for _ in left_edges]
    last_finite = last_row_time = None
    interval_min, interval_max, interval_sum = np.inf, -np.inf, 0.
    interval_sample = []
    sample_size = 0
    origin = interval = np.nan
    bounds = []
    reader = pd.read_csv(stream, chunksize=chunksize, dtype=str)
    for chunk in reader:
        if list(chunk.columns) != ["Time", "SNUADC/ART"]:
            raise ValueError("Unexpected raw track column schema")
        offset, n = r["source_rows"], len(chunk)
        t = pd.to_numeric(chunk.iloc[:, 0], errors="coerce").to_numpy(float)
        v = pd.to_numeric(chunk.iloc[:, 1], errors="coerce").to_numpy(float)
        ft = np.isfinite(t)
        fi = np.flatnonzero(ft)
        if offset == 0:
            origin = float(t[0]) if n else np.nan
            interval = float(t[1] - t[0]) if n > 1 else np.nan
            r["first_two_physical_rows_finite"] = bool(n > 1 and np.isfinite(t[:2]).all())
            r["nominal_interval_sec"] = interval
            r["nominal_origin_sec"] = origin
            r["nominal_fs_hz"] = 1 / interval if np.isfinite(interval) and interval > 0 else np.nan
            if np.isfinite(origin) and np.isfinite(interval) and interval > 0:
                bounds = [(nominal_index(e, origin, interval),
                           nominal_index(e + 10., origin, interval)) for e in left_edges]
            tokens = chunk.iloc[:4096, 1].dropna().astype(str)
            digits = tokens.str.extract(r"\.(\d+)(?:[eE].*)?$", expand=False).dropna().str.len()
            r["value_decimal_places_first_chunk_sample_max"] = int(digits.max()) if len(digits) else 0
            r["value_precision_sample_rows"] = int(len(tokens))
        r["source_rows"] += n
        r["time_finite_rows"] += int(ft.sum())
        r["time_missing_rows"] += int(chunk.iloc[:, 0].isna().sum())
        r["time_invalid_nonblank_rows"] += int((chunk.iloc[:, 0].notna().to_numpy() & ~ft).sum())
        r["value_finite_rows"] += int(np.isfinite(v).sum())
        if len(fi):
            if r["explicit_row_indices_min"] is None:
                r["explicit_row_indices_min"] = int(offset + fi[0])
            r["explicit_row_indices_max"] = int(offset + fi[-1])
            observed = t[fi]
            pairs = np.diff(np.r_[last_finite, observed]) if last_finite is not None else np.diff(observed)
            r["observed_time_regressions"] += int((pairs < 0).sum())
            r["observed_time_duplicates"] += int((pairs == 0).sum())
            last_finite = float(observed[-1])
            if bounds:
                rebuilt = origin + (offset + fi) * interval
                delta = np.abs(observed - rebuilt)
                r["max_explicit_vs_rebuilt_abs_sec"] = max(r["max_explicit_vs_rebuilt_abs_sec"], float(delta.max()))
                r["explicit_vs_rebuilt_disagreement_count"] += int((delta > TIME_ATOL).sum())
        adjacent = np.diff(t)[ft[:-1] & ft[1:]]
        if last_row_time is not None and n and ft[0] and np.isfinite(last_row_time):
            adjacent = np.r_[t[0] - last_row_time, adjacent]
        last_row_time = float(t[-1]) if n else last_row_time
        if len(adjacent):
            r["observed_adjacent_interval_count"] += len(adjacent)
            interval_min = min(interval_min, float(adjacent.min()))
            interval_max = max(interval_max, float(adjacent.max()))
            interval_sum += float(adjacent.sum())
            if np.isfinite(interval):
                r["adjacent_interval_deviates_from_nominal"] += int((np.abs(adjacent - interval) > TIME_ATOL).sum())
            take = min(len(adjacent), 100000 - sample_size)
            if take:
                interval_sample.append(adjacent[:take])
                sample_size += take
        for j, edge in enumerate(left_edges):
            actual = ft & (t >= edge) & (t < edge + 10.)
            if actual.any():
                parts[j].append((t[actual], v[actual]))
            diagnostics[j]["exact_left_boundary_rows"] += int((ft & (t == edge)).sum())
            diagnostics[j]["exact_right_boundary_rows"] += int((ft & (t == edge + 10.)).sum())
            if bounds:
                a, b = max(offset, bounds[j][0]), min(offset + n, bounds[j][1])
                if b > a:
                    diagnostics[j]["nominal_row_count"] += b - a
                    diagnostics[j]["nominal_rows_with_missing_time"] += int((~ft[a-offset:b-offset]).sum())
                explicit_nominal = (offset + fi >= bounds[j][0]) & (offset + fi < bounds[j][1])
                diagnostics[j]["explicit_membership_disagreement"] += int((actual[fi] != explicit_nominal).sum())
    count = r["observed_adjacent_interval_count"]
    r.update(all_sample_times_observed=bool(r["source_rows"] and r["time_finite_rows"] == r["source_rows"]),
             observed_adjacent_interval_min_sec=interval_min if count else np.nan,
             observed_adjacent_interval_max_sec=interval_max if count else np.nan,
             observed_adjacent_interval_mean_sec=interval_sum / count if count else np.nan,
             observed_adjacent_interval_prefix_median_sec=float(np.median(np.concatenate(interval_sample))) if sample_size else np.nan,
             observed_adjacent_interval_prefix_sample_n=sample_size)
    r["bins"] = []
    for edge, items, diagnostic in zip(left_edges, parts, diagnostics):
        t = np.concatenate([p[0] for p in items]) if items else np.array([])
        v = np.concatenate([p[1] for p in items]) if items else np.array([])
        stats = summarize_bin(t, v, edge, r.get("nominal_fs_hz", np.nan))
        stats.update(diagnostic)
        r["bins"].append(stats)
    return r


def comparison(cached, observed, tolerance):
    a, b = float(cached), float(observed)
    if np.isnan(a) and np.isnan(b):
        return {"match": True, "abs_difference": np.nan, "finite_vs_missing": False}
    finite = bool(np.isfinite(a) and np.isfinite(b))
    d = abs(a - b) if finite else np.nan
    return {"match": bool(finite and d <= tolerance), "abs_difference": d,
            "finite_vs_missing": bool(np.isfinite(a) != np.isfinite(b))}


def stratify_first_bins(wave):
    f = wave.sort_values(["case_id", "time_sec"], kind="stable").groupby("case_id", sort=True).head(1).copy()
    low = f.numeric_map < 40
    outside = (f.wave_p05 < 0) | (f.wave_p95 > 250)
    inside = (f.wave_p05 >= 0) & (f.wave_p95 <= 250)
    f["stratum"] = np.select([low & outside, low & inside,
                               (f.numeric_map >= 40) & (f.numeric_map < 65), f.numeric_map >= 65],
                              STRATA, default="UNCLASSIFIED")
    return f


def select_cases(first, per_stratum=8):
    rng, selected = np.random.default_rng(SEED), []
    for name in STRATA:
        f = first.loc[first.stratum == name].sort_values("case_id", kind="stable")
        if len(f) < per_stratum:
            raise ValueError("Insufficient stratum size; no replacement allowed")
        selected.append(f.iloc[rng.choice(len(f), size=per_stratum, replace=False)].copy())
    return pd.concat(selected, ignore_index=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        b = self.stream.read(size)
        self.digest.update(b)
        return b


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_self_tests():
    captured = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceTimeTests)
    result = unittest.TextTestRunner(stream=captured, verbosity=2).run(suite)
    return {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "passed": result.wasSuccessful()}, captured.getvalue()


def boundary_state(all_times_observed, observed_membership_disagreements):
    complete = bool(all_times_observed)
    count = int(observed_membership_disagreements)
    return {"comparison_status": "COMPARABLE" if complete else "NOT_IDENTIFIABLE_MISSING_SOURCE_TIMESTAMPS",
            "any_sample_difference": count > 0 if complete else None,
            "one_or_two_sample_difference": count in (1, 2) if complete else None,
            "invariance_verified": bool(complete and count == 0)}


def gate_change_state(raw_features_identifiable, cache_good, raw_good):
    complete = bool(raw_features_identifiable)
    return {"comparison_status": "COMPARABLE" if complete else "NOT_IDENTIFIABLE_MISSING_SOURCE_TIMESTAMPS",
            "raw_core_good": bool(raw_good) if complete else None,
            "core_gate_changed": bool(cache_good != raw_good) if complete else None,
            "invariance_verified": bool(complete and cache_good == raw_good)}


def add_boundary_gate_report(summary, details):
    """Read the completed gate table; never rerun onset or outcome statistics."""
    path = ROOT / "restricted/startup_quality_bins.csv.gz"
    before = sha256(path)
    gate = pd.read_csv(path)
    keys = ["case_id", "time_sec"]
    if gate.duplicated(keys).any():
        raise ValueError("Existing gate table has nonunique keys")
    lookup = gate.set_index(keys)
    contract = json.loads((ROOT / "analysis_contract.json").read_text())
    spec = contract["primary_bin_gate"]
    annotations = []
    for row in details.to_dict("records"):
        key = (int(row["case_id"]), float(row["cache_time_sec"]))
        old = lookup.loc[key]
        for name, atol in METRIC_TOLERANCES.items():
            if not comparison(row["cache_" + name], old[name], atol)["match"]:
                raise ValueError("Existing gate table features do not match the audited frozen cache")
        old_good = bool(old.core_good)
        complete = bool(row["literal_metrics_are_complete_bin_estimates"])
        raw_good = None
        if complete:
            m, mean = float(old.numeric_map), float(row["raw_wave_mean"])
            lo, hi = spec["finite_numeric_MAP_mmHg"]
            wlo, whi = spec["waveform_mean_mmHg"]
            slo, shi = spec["waveform_p95_minus_p05_mmHg"]
            plo, phi = spec["peak_count_per10s"]
            raw_good = bool(np.isfinite(m) and lo <= m <= hi and
                            row["raw_wave_coverage"] >= spec["waveform_coverage_at_least"] and
                            np.isfinite(mean) and wlo <= mean <= whi and
                            row["raw_wave_p05"] >= spec["waveform_p05_at_least_mmHg"] and
                            row["raw_wave_p95"] <= spec["waveform_p95_at_most_mmHg"] and
                            slo <= row["raw_wave_span_90pct"] <= shi and
                            plo <= row["raw_peak_count"] <= phi and
                            abs(mean - m) <= spec["absolute_waveform_mean_minus_numeric_MAP_at_most_mmHg"])
            s, d = float(old.sbp), float(old.dbp)
            if np.isfinite(s) and np.isfinite(d):
                pair = spec["if_both_numeric_SBP_DBP_available"]
                raw_good &= bool(pair["SBP_mmHg"][0] <= s <= pair["SBP_mmHg"][1] and
                                 pair["DBP_mmHg"][0] <= d <= pair["DBP_mmHg"][1] and
                                 pair["pulse_pressure_mmHg"][0] <= s-d <= pair["pulse_pressure_mmHg"][1] and
                                 d-pair["MAP_within_DBP_SBP_tolerance_mmHg"] <= m <=
                                 s+pair["MAP_within_DBP_SBP_tolerance_mmHg"])
        boundary = boundary_state(row["all_sample_times_observed"], row["raw_explicit_membership_disagreement"])
        change = gate_change_state(complete, old_good, raw_good)
        annotations.append({"cache_core_good_existing_table": old_good,
                            "raw_core_good": change["raw_core_good"],
                            "core_gate_changed": change["core_gate_changed"],
                            "core_gate_comparison_status": change["comparison_status"],
                            "boundary_comparison_status": boundary["comparison_status"],
                            "boundary_any_sample_difference": boundary["any_sample_difference"],
                            "boundary_one_or_two_sample_difference": boundary["one_or_two_sample_difference"],
                            "boundary_invariance_verified": boundary["invariance_verified"]})
    d = details.copy()
    for name in annotations[0]:
        d[name] = [r[name] for r in annotations]
    cases = d.drop_duplicates("case_id")
    summary["raw_time_findings"].update({
        "explicit_timestamps_per_record_distribution": {str(int(k)): int(v) for k, v in cases.time_finite_rows.value_counts().items()},
        "records_with_initial_two_and_last_timestamp_only": int(((cases.time_finite_rows == 3) &
            (cases.explicit_row_indices_min == 0) & (cases.explicit_row_indices_max == cases.source_rows-1)).sum()),
        "records_with_explicit_vs_rebuilt_difference": int((cases.explicit_vs_rebuilt_disagreement_count > 0).sum())})
    full_boundaries = d.boundary_comparison_status == "COMPARABLE"
    comparable = d.core_gate_comparison_status == "COMPARABLE"
    all_boundaries = bool(full_boundaries.all())
    all_gates = bool(comparable.all())
    summary["box_boundary_and_core_gate"] = {
        "status": "COMPARABLE" if all_boundaries and all_gates else "NOT_IDENTIFIABLE_MISSING_SOURCE_TIMESTAMPS",
        "boxes_checked": len(d), "complete_boundary_comparison_boxes": int(full_boundaries.sum()),
        "incomplete_boundary_comparison_boxes": int((~full_boundaries).sum()),
        "observed_explicit_timestamp_membership_disagreement_rows": int(d.raw_explicit_membership_disagreement.sum()),
        "one_or_two_sample_boundary_difference_present": bool(d.boundary_one_or_two_sample_difference.fillna(False).any()) if all_boundaries else None,
        "one_or_two_sample_boundary_difference_boxes": int(d.boundary_one_or_two_sample_difference.fillna(False).sum()) if all_boundaries else None,
        "any_boundary_difference_present": bool(d.boundary_any_sample_difference.fillna(False).any()) if all_boundaries else None,
        "boundary_invariance_verified": bool(all_boundaries and d.boundary_invariance_verified.all()),
        "actual_boundary_sample_count_difference_determined": all_boundaries,
        "existing_cached_core_pass_boxes": int(d.cache_core_good_existing_table.sum()),
        "existing_cached_core_fail_boxes": int((~d.cache_core_good_existing_table).sum()),
        "raw_core_gate_comparable_boxes": int(comparable.sum()),
        "raw_core_gate_unverifiable_boxes": int((~comparable).sum()),
        "core_gate_changed": bool(d.core_gate_changed.fillna(False).any()) if all_gates else None,
        "core_gate_changed_boxes": int(d.core_gate_changed.fillna(False).sum()) if all_gates else None,
        "core_gate_unchanged_boxes": int((~d.loc[comparable, "core_gate_changed"].astype(bool)).sum()) if all_gates else None,
        "core_gate_invariance_verified": bool(all_gates and not d.core_gate_changed.fillna(False).any()),
        "existing_gate_table_sha256": before,
        "existing_gate_table_unchanged": before == sha256(path),
        "existing_table_is_read_only_input": True,
        "main_statistics_or_onset_rerun": False,
        "core60_onset_effect_status": "NOT_DETERMINED; four nonconsecutive boxes cannot establish six-consecutive-bin onset invariance",
        "interpretation": "Zero disagreement among the few explicit anchors does NOT establish absence of one/two-sample boundary shifts. Missing times are NOT treated as failed core gates or as gate invariance. End-anchor drift must not be extrapolated into startup boxes without evidence."}
    if not summary["box_boundary_and_core_gate"]["existing_gate_table_unchanged"]:
        raise ValueError("Existing gate table changed during read-only comparison")
    summary["interpretation"].append(
        "The source-verification requirement is not satisfied when source times are unidentifiable. Neither cache consistency nor core-gate invariance can then be claimed; this does not by itself prove cached pressure features wrong.")
    return d


def extend_existing_report():
    """Add requested boundary/gate QA without rescanning or rerunning analyses."""
    os.umask(0o077)
    report_path = ROOT / "qa/raw_cache_check.json"
    detail_path = ROOT / "restricted/raw_cache_check_details.csv"
    log_path = ROOT / "restricted/audit_raw.internal.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        try:
            report = json.loads(report_path.read_text())
            if report["restricted_details_sha256"] != sha256(detail_path):
                raise ValueError("Details changed since the completed source audit")
            tests, test_output = run_self_tests()
            if not tests["passed"]:
                raise AssertionError("Self-tests failed")
            manifest = json.loads((ROOT / "qa/protected_input_hashes.json").read_text())
            protected_ok = all(sha256(p) == h for p, h in manifest.items())
            protected_ok &= sha256(ROOT / "analysis_contract.json") == report["frozen_contract_sha256"]
            if not protected_ok:
                raise ValueError("Frozen inputs changed before the boundary addendum")
            d = pd.read_csv(detail_path, float_precision="round_trip")
            d = add_boundary_gate_report(report, d)
            report["execution_script_sha256"] = report.get("execution_script_sha256", report["script_sha256"])
            report["script_sha256"] = sha256(Path(__file__))
            report["self_tests"] = tests
            report["boundary_addendum_utc"] = utc_now()
            report["protected_inputs_unchanged_at_boundary_addendum"] = protected_ok
            d.to_csv(detail_path, index=False, float_format="%.17g")
            report["restricted_details_sha256"] = sha256(detail_path)
            report_path.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")
            log.write(json.dumps({"utc": utc_now(), "event": "boundary_gate_addendum",
                                  "self_test_output": test_output,
                                  "aggregate": clean_json(report["box_boundary_and_core_gate"]),
                                  "execution_source_audit_not_repeated": True}, allow_nan=False) + "\n")
            print(json.dumps(clean_json(report["box_boundary_and_core_gate"])), flush=True)
            return 0
        except Exception:
            log.write(json.dumps({"utc": utc_now(), "event": "boundary_addendum_error",
                                  "error": traceback.format_exc()}) + "\n")
            print(json.dumps({"boundary_addendum": "ERROR", "see": "restricted internal log"}), flush=True)
            return 2


def run_audit():
    os.umask(0o077)
    qa_path = ROOT / "qa/raw_cache_check.json"
    detail_path = ROOT / "restricted/raw_cache_check_details.csv"
    log_path = ROOT / "restricted/audit_raw.internal.jsonl"
    if qa_path.exists() or detail_path.exists():
        raise FileExistsError("Audit outputs already exist; refusing to overwrite")
    started, clock_start = utc_now(), time.monotonic()
    log = log_path.open("a", encoding="utf-8")

    def record(event, **payload):
        log.write(json.dumps(clean_json({"utc": utc_now(), "event": event, **payload}), allow_nan=False) + "\n")
        log.flush()

    try:
        tests, test_text = run_self_tests()
        record("self_tests", result=tests, text=test_text)
        if not tests["passed"]:
            raise AssertionError("Self-tests failed")
        contract_path = ROOT / "analysis_contract.json"
        contract = json.loads(contract_path.read_text())
        if contract["grid_sec"] != 10:
            raise ValueError("Unsupported frozen grid")
        protected_manifest = ROOT / "qa/protected_input_hashes.json"
        expected = json.loads(protected_manifest.read_text())
        named = {"contract": contract_path, "protected_manifest": protected_manifest,
                 "cache": CACHE_DIR / "full_waveform_10s_metrics.csv",
                 "numeric_audit": CACHE_DIR / "startup_numeric_case_audit.csv",
                 "legacy_wave_code": CACHE_DIR.parent / "code/run_waveform_audit.py",
                 "legacy_time_code": CACHE_DIR.parent / "code/audit_core.py",
                 "legacy_scan_code": CACHE_DIR.parent / "code/scan_full_waveforms.py"}
        for i, path in enumerate(expected):
            named[f"protected_{i+1:02d}"] = Path(path)
        before = {name: sha256(p) for name, p in named.items()}
        if any(before[name] != expected[str(p)] for name, p in named.items() if str(p) in expected):
            raise ValueError("Protected input differs from pre-existing frozen hashes")
        wave = pd.read_csv(named["cache"])
        meta = pd.read_csv(named["numeric_audit"])
        if wave.duplicated(["case_id", "time_sec"]).any() or meta.case_id.duplicated().any():
            raise ValueError("Nonunique input keys")
        first = stratify_first_bins(wave)
        selected = select_cases(first)
        source = meta.set_index("case_id")
        cache_groups = {int(k): g.sort_values("time_sec", kind="stable").reset_index(drop=True)
                        for k, g in wave.groupby("case_id")}
        trks = pd.read_csv(TRACK_ROOTS[0].parent / "trks.csv", usecols=["caseid", "tname", "tid"])
        lookups = set(zip(trks.caseid.astype(int), trks.tname.astype(str), trks.tid.astype(str)))
        record("frozen_selection", selected=selected[["case_id", "stratum"]].to_dict("records"),
               seed=SEED, numpy_version=np.__version__, input_hashes=before)
        details, case_results = [], []
        for ordinal, chosen in enumerate(selected.itertuples(index=False), 1):
            case_id = int(chosen.case_id)
            m = source.loc[case_id]
            g = cache_groups[case_id]
            tid = str(m.waveform_tid)
            candidates = [root / (tid + suffix) for root in TRACK_ROOTS for suffix in (".csv.gz", ".csv")
                          if (root / (tid + suffix)).is_file()]
            base = {"case_id": case_id, "stratum": chosen.stratum, "sample_order": ordinal,
                    "waveform_tid": tid, "start_abs_sec": float(m.start_abs_sec),
                    "first_valid_rel_sec": float(m.first_valid_rel_sec),
                    "first_numeric_map": float(chosen.numeric_map),
                    "candidate_source_paths": json.dumps([str(p) for p in candidates]),
                    "candidate_source_count": len(candidates),
                    "track_identity_confirmed": (case_id, "SNUADC/ART", tid) in lookups,
                    "cache_has_exactly_60_rows": len(g) == 60}
            rel = float(m.first_valid_rel_sec) + np.arange(60) * 10.
            base["cache_all_60_relative_times_aligned"] = bool(len(g) == 60 and np.allclose(
                g.time_sec.to_numpy(float), rel, rtol=0, atol=TIME_ATOL))
            edges = [float(m.start_abs_sec) + float(m.first_valid_rel_sec) + b * 10. for b in BIN_NUMBERS]
            r = None
            try:
                if not candidates:
                    raise FileNotFoundError("No local source waveform")
                p = candidates[0]
                base["source_path"] = str(p)
                base["source_root_number"] = TRACK_ROOTS.index(p.parent) + 1
                pre = p.stat()
                with p.open("rb") as disk:
                    hashed = HashReader(disk)
                    with contextlib.ExitStack() as stack:
                        stream = stack.enter_context(gzip.GzipFile(fileobj=hashed, mode="rb")) if p.suffix == ".gz" else hashed
                        r = scan_csv_stream(stream, edges)
                    # Include any trailing bytes in the compressed-file digest.
                    while hashed.read(8 * 1024 * 1024):
                        pass
                    base["source_sha256"] = hashed.digest.hexdigest()
                post = p.stat()
                base["source_bytes"] = pre.st_size
                base["source_unchanged_during_read"] = (pre.st_size, pre.st_mtime_ns) == (post.st_size, post.st_mtime_ns)
                base.update({k: v for k, v in r.items() if k != "bins"})
                full_times = r["all_sample_times_observed"]
                regular = r["adjacent_interval_deviates_from_nominal"] == 0
                ordered = r["observed_time_regressions"] == r["observed_time_duplicates"] == 0
                fs_ok = np.isfinite(r["nominal_fs_hz"]) and int(.3 * r["nominal_fs_hz"]) >= 1
                identifiable = bool(full_times and fs_ok and
                                    base["track_identity_confirmed"] and base["source_unchanged_during_read"] and
                                    base["cache_all_60_relative_times_aligned"])
                if not full_times:
                    status = "NOT_IDENTIFIABLE_MISSING_SOURCE_TIMESTAMPS"
                elif not (regular and ordered):
                    status = "RAW_TIME_IRREGULARITY"
                elif not identifiable:
                    status = "SOURCE_OR_ALIGNMENT_CHECK_FAILED"
                else:
                    status = "COMPARABLE"
                base["raw_time_status"] = status
                for j, b in enumerate(BIN_NUMBERS):
                    row = {**base, "bin_index_zero_based": b, "bin_left_abs_sec": edges[j],
                           "bin_right_abs_sec": edges[j] + 10., "bin_interval": "[left,right)",
                           "cache_time_sec": float(g.iloc[b].time_sec),
                           "literal_metrics_are_complete_bin_estimates": identifiable}
                    obs = r["bins"][j]
                    row.update({"raw_" + k: v for k, v in obs.items()})
                    all_match = True
                    for name, tolerance in METRIC_TOLERANCES.items():
                        a = float(g.iloc[b][name])
                        cmp = comparison(a, obs[name], tolerance)
                        row["cache_" + name] = a
                        row[name + "_abs_difference"] = cmp["abs_difference"]
                        row[name + "_finite_vs_missing"] = cmp["finite_vs_missing"]
                        row[name + "_literal_match"] = cmp["match"]
                        all_match &= cmp["match"]
                    row["all_six_literal_metrics_match"] = all_match
                    if not identifiable:
                        row["validation_status"] = status
                    elif not all_match:
                        row["validation_status"] = "CACHE_MISMATCH"
                    else:
                        row["validation_status"] = "PASS" if regular and ordered else "RAW_TIME_IRREGULARITY"
                    details.append(row)
                result = {**base, "audit_read_status": "FULL_FILE_READ", "comparable": identifiable,
                          "all_four_bins_pass": all(x["validation_status"] == "PASS" for x in details[-4:])}
            except Exception as error:
                record("case_error", case_id=case_id, error=traceback.format_exc())
                status = "MISSING_SOURCE" if isinstance(error, FileNotFoundError) else "SOURCE_READ_ERROR"
                for b, edge in zip(BIN_NUMBERS, edges):
                    details.append({**base, "bin_index_zero_based": b, "bin_left_abs_sec": edge,
                                    "bin_right_abs_sec": edge + 10., "validation_status": status,
                                    "error_type": type(error).__name__})
                result = {**base, "audit_read_status": status, "comparable": False,
                          "all_four_bins_pass": False}
            case_results.append(result)
            record("case_completed", result=result)
            # Checkpoint only to the expressly authorized restricted output.
            pd.DataFrame(details).to_csv(detail_path, index=False, float_format="%.17g")
            print(json.dumps({"source_records_completed": ordinal, "planned": 32,
                              "time_identifiable_so_far": sum(x["comparable"] for x in case_results)}), flush=True)
        d, c = pd.DataFrame(details), pd.DataFrame(case_results)
        after = {name: sha256(p) for name, p in named.items()}
        preserved = before == after
        record("protected_input_hash_check", before=before, after=after, unchanged=preserved)
        read = c[c.audit_read_status == "FULL_FILE_READ"]
        metric_summary = {}
        for name in METRIC_TOLERANCES:
            column = name + "_literal_match"
            valid = d[column].notna() if column in d else pd.Series(False, index=d.index)
            mismatch = valid & ~d[column].fillna(False).astype(bool) if column in d else valid
            finite = d[name + "_abs_difference"].dropna() if name + "_abs_difference" in d else pd.Series(dtype=float)
            comparable = d.validation_status.isin(["PASS", "CACHE_MISMATCH", "RAW_TIME_IRREGULARITY"])
            metric_summary[name] = {"absolute_tolerance": METRIC_TOLERANCES[name],
                                    "literal_compared_boxes": int(valid.sum()),
                                    "literal_mismatch_boxes": int(mismatch.sum()),
                                    "definitive_comparable_mismatch_boxes": int((mismatch & comparable).sum()),
                                    "max_finite_absolute_difference": float(finite.max()) if len(finite) else None,
                                    "finite_vs_missing_boxes": int(d.get(name + "_finite_vs_missing", pd.Series(False, index=d.index)).fillna(False).sum())}
        def n_cases(column, predicate):
            return int(predicate(read[column]).sum()) if column in read else 0
        root_counts = {str(int(k)): int(v) for k, v in read.source_root_number.value_counts().items()} if len(read) else {}
        summary = {
            "schema_version": 1, "audit_started_utc": started, "audit_finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - clock_start,
            "task": "Independent stratified source-time/cache audit only; no main statistical analysis",
            "overall_status": "COMPLETE_PASS" if len(read) == 32 and c.all_four_bins_pass.all() and preserved
                              else "AUDIT_COMPLETED_WITH_UNVERIFIABLE_OR_DISCREPANT_SOURCE",
            "source_verification_pass": bool(len(read) == 32 and c.all_four_bins_pass.all() and preserved),
            "frozen_contract_sha256": before["contract"],
            "script_sha256": sha256(Path(__file__)), "protected_input_sha256_before": before,
            "protected_inputs_unchanged": preserved,
            "sampling": {"seed": SEED, "generator": "numpy.default_rng(PCG64)", "without_replacement": True,
                         "ordered_strata": list(STRATA), "ordering_within_stratum": "case_id ascending",
                         "per_stratum_planned": 8, "selected_total": len(selected),
                         "population_cache_cases": len(first), "numeric_audit_cases": len(meta),
                         "cache_absent_cases": int((~meta.case_id.isin(first.case_id)).sum()),
                         "unclassified_cache_cases": int((first.stratum == "UNCLASSIFIED").sum()),
                         "population_by_stratum": {s: int((first.stratum == s).sum()) for s in STRATA},
                         "selected_by_stratum": {s: int((selected.stratum == s).sum()) for s in STRATA},
                         "first_bin_rule": "Physical first chronological cache row; never groupby.first(skipna)",
                         "no_source_availability_resampling": True},
            "counts": {"selected_records": len(selected), "fully_read_source_records": len(read),
                       "planned_boxes": 128, "detail_rows": len(d),
                       "independently_comparable_records": int(c.comparable.sum()),
                       "passed_records": int(c.all_four_bins_pass.sum()),
                       "validation_status_boxes": {str(k): int(v) for k, v in d.validation_status.value_counts().items()},
                       "definitive_cache_mismatch_boxes": int((d.validation_status == "CACHE_MISMATCH").sum()),
                       "literal_disagreement_boxes": int((~d.all_six_literal_metrics_match.fillna(True).astype(bool)).sum()) if "all_six_literal_metrics_match" in d else 0,
                       "cache_60_row_alignment_failed_records": int((~c.cache_all_60_relative_times_aligned).sum()),
                       "source_identity_failed_records": int((~c.track_identity_confirmed).sum()),
                       "source_root_counts": root_counts},
            "methods": {"requested_boxes_zero_based": list(BIN_NUMBERS),
                        "absolute_left_edge": "start_abs_sec + first_valid_rel_sec + bin_index*10",
                        "raw_binning": "ONLY parsed finite actual CSV Time entries; left <= Time < right; no imputation, snapping, sorting or regular-time substitute",
                        "coverage": "finite timed pressure values/(10*nominal_fs); partial lower-bound diagnostic only when raw times are missing",
                        "nominal_fs": "1/(Time[physical row 1]-Time[physical row 0]); not an independently verified sampling rate when later timestamps are absent",
                        "quantiles": "numpy.percentile finite timed values, linear interpolation",
                        "peaks": "scipy.signal.find_peaks in original row order, distance=int(0.3*fs), prominence=5; NaN values replaced by within-box finite median, matching frozen peak preprocessing",
                        "time_comparison_absolute_tolerance_sec": TIME_ATOL,
                        "metric_tolerance_rationale": "Absolute float64 parsing/reduction tolerances, not clinical equivalence: pressure 1e-8 mmHg and coverage 1e-10 are far below observed source decimal quantization; exact integer peaks, rtol=0. Set before source comparisons. No tolerance shifts bin edges.",
                        "nominal_reconstruction": "Counts and explicit-row time diagnostics only; NEVER used to calculate the independently audited pressure features",
                        "full_file_time_scan": True},
            "raw_time_findings": {
                "source_rows_scanned": int(read.source_rows.sum()) if len(read) else 0,
                "explicit_finite_timestamp_rows": int(read.time_finite_rows.sum()) if len(read) else 0,
                "missing_timestamp_rows": int(read.time_missing_rows.sum()) if len(read) else 0,
                "invalid_nonblank_timestamp_rows": int(read.time_invalid_nonblank_rows.sum()) if len(read) else 0,
                "records_with_only_first_two_timestamps": int(((read.time_finite_rows == 2) & (read.explicit_row_indices_min == 0) & (read.explicit_row_indices_max == 1)).sum()) if len(read) else 0,
                "records_with_all_sample_times_observed": n_cases("all_sample_times_observed", lambda x: x.astype(bool)),
                "records_with_observed_time_regressions": n_cases("observed_time_regressions", lambda x: x > 0),
                "records_with_observed_time_duplicates": n_cases("observed_time_duplicates", lambda x: x > 0),
                "observed_adjacent_interval_count_total": int(read.observed_adjacent_interval_count.sum()) if len(read) else 0,
                "observed_adjacent_interval_min_sec": float(read.observed_adjacent_interval_min_sec.min()) if len(read) else None,
                "observed_adjacent_interval_max_sec": float(read.observed_adjacent_interval_max_sec.max()) if len(read) else None,
                "records_with_interval_deviation_from_nominal": n_cases("adjacent_interval_deviates_from_nominal", lambda x: x > 0),
                "explicit_vs_rebuilt_max_absolute_difference_sec": float(read.max_explicit_vs_rebuilt_abs_sec.max()) if len(read) else None,
                "explicit_vs_rebuilt_disagreement_rows": int(read.explicit_vs_rebuilt_disagreement_count.sum()) if len(read) else 0,
                "requested_boxes_nominal_rows_without_actual_time": int(d.raw_nominal_rows_with_missing_time.sum()) if "raw_nominal_rows_with_missing_time" in d else 0,
                "requested_boxes_observed_assignment_disagreements": int(d.raw_explicit_membership_disagreement.sum()) if "raw_explicit_membership_disagreement" in d else 0,
                "literal_timed_rows_in_requested_boxes": int(d.raw_row_count.sum()) if "raw_row_count" in d else 0,
                "exact_left_boundary_rows": int(d.raw_exact_left_boundary_rows.sum()) if "raw_exact_left_boundary_rows" in d else 0,
                "exact_right_boundary_rows": int(d.raw_exact_right_boundary_rows.sum()) if "raw_exact_right_boundary_rows" in d else 0,
                "full_record_monotonicity_and_sampling_interval_proven": bool(len(read) == 32 and read.all_sample_times_observed.all() and (read.observed_time_regressions == 0).all() and (read.observed_time_duplicates == 0).all() and (read.adjacent_interval_deviates_from_nominal == 0).all())},
            "metric_comparisons": metric_summary,
            "interpretation": [
                "Only raw rows carrying an actual finite time can be assigned independently to a requested half-open box.",
                "Missing waveform times make complete-bin coverage/pressure/peak validation unidentifiable. Literal partial-data differences are reported, not treated as proven cache corruption.",
                "Agreement at two initial anchors is algebraic by construction and does not validate missing later sample times or the reconstructed axis.",
                "No missing/inaccessible source is classified as pass. No case substitution, cache update or frozen-contract amendment.",
                "This is a 32-record/128-box stratified sample, not full-cohort or full-record waveform validation and not clinical adjudication."],
            "self_tests": tests,
            "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": __import__("scipy").__version__},
            "outputs": {"aggregate": "qa/raw_cache_check.json", "restricted_details": "restricted/raw_cache_check_details.csv",
                        "restricted_internal_log": "restricted/audit_raw.internal.jsonl"},
            "main_statistics_rerun": False, "cache_modified": False, "contract_modified": False,
            "manuscript_modified": False, "case_ids_or_case_source_paths_in_qa": False,
        }
        final_details = add_boundary_gate_report(summary, pd.DataFrame(details))
        final_details.to_csv(detail_path, index=False, float_format="%.17g")
        summary["restricted_details_sha256"] = sha256(detail_path)
        qa_path.write_text(json.dumps(clean_json(summary), indent=2, ensure_ascii=True, allow_nan=False) + "\n")
        record("audit_finished", aggregate_status=summary["overall_status"], input_preserved=preserved,
               detail_rows=len(d), elapsed_seconds=summary["elapsed_seconds"])
        print(json.dumps(clean_json({"overall_status": summary["overall_status"],
                                    "counts": summary["counts"], "protected_inputs_unchanged": preserved})), flush=True)
        return 0 if preserved and len(d) == 128 else 2
    except Exception:
        record("fatal_error", error=traceback.format_exc())
        print(json.dumps({"audit_execution": "ERROR", "see": "restricted internal log"}), flush=True)
        return 2
    finally:
        log.close()


class SourceTimeTests(unittest.TestCase):
    def test_half_open_box_uses_observed_times(self):
        r = summarize_bin([0, 9.999, 10, 20, np.nan], [10, 20, 30, 40, 999], 0, 10)
        self.assertEqual(r.get("row_count"), 2)
        self.assertAlmostEqual(r["wave_coverage"], 0.02)
        self.assertAlmostEqual(r["wave_mean"], 15)
        self.assertAlmostEqual(r["wave_p05"], 10.5)
        self.assertAlmostEqual(r["wave_p95"], 19.5)
        self.assertAlmostEqual(r["wave_span_90pct"], 9)
        self.assertEqual(r["peak_count"], 0)

    def test_irregular_times_cannot_be_replaced_by_row_number(self):
        r = summarize_bin([0, 1, 11, 12], [1, 10, 100, 200], 0, 10)
        self.assertEqual(r.get("row_count"), 2)
        self.assertAlmostEqual(r["wave_mean"], 5.5)

    def test_boundary_is_not_snapped(self):
        r = summarize_bin([np.nextafter(10., 0.), 10., np.nextafter(10., 20.)],
                          [1, 100, 200], 0, 10)
        self.assertEqual(r.get("row_count"), 1)
        self.assertEqual(r["wave_mean"], 1)

    def test_missing_times_remain_unidentifiable(self):
        r = scan_csv_stream(io.StringIO("Time,SNUADC/ART\n0,10\n1,20\n,30\n,40\n"), [0], 2)
        self.assertEqual(r.get("time_finite_rows"), 2)
        self.assertEqual(r["time_missing_rows"], 2)
        self.assertFalse(r["all_sample_times_observed"])
        self.assertEqual(r["observed_adjacent_interval_count"], 1)
        self.assertEqual(r["bins"][0]["row_count"], 2)
        self.assertEqual(r["bins"][0]["nominal_rows_with_missing_time"], 2)

    def test_time_regression_and_duplicate_across_chunks(self):
        r = scan_csv_stream(io.StringIO("Time,SNUADC/ART\n0,10\n1,20\n1,30\n0.5,40\n"), [0], 2)
        self.assertEqual(r.get("observed_time_regressions"), 1)
        self.assertEqual(r["observed_time_duplicates"], 1)
        self.assertEqual(r["observed_adjacent_interval_count"], 3)
        self.assertAlmostEqual(r["max_explicit_vs_rebuilt_abs_sec"], 2.5)

    def test_nonfinite_values_do_not_count_as_coverage(self):
        r = summarize_bin([0, 1, 2], [10, np.nan, 30], 0, 10)
        self.assertEqual(r.get("finite_value_count"), 2)
        self.assertAlmostEqual(r["wave_coverage"], 0.02)
        self.assertAlmostEqual(r["wave_mean"], 20)

    def test_nan_and_absolute_tolerance_are_explicit(self):
        self.assertEqual(comparison(1., 1. + 5e-9, 1e-8).get("match"), True)
        self.assertEqual(comparison(1., 1. + 5e-7, 1e-8)["match"], False)
        self.assertEqual(comparison(np.nan, np.nan, 1e-8)["match"], True)
        self.assertEqual(comparison(1., np.nan, 1e-8)["match"], False)

    def test_first_bin_missing_feature_is_not_replaced_by_later_value(self):
        w = pd.DataFrame({"case_id": [1, 1, 2, 3, 4, 5], "time_sec": [0, 10, 0, 0, 0, 0],
                          "numeric_map": [30, 30, 30, 30, 40, 65],
                          "wave_p05": [np.nan, -5, -1, 0, 5, 5],
                          "wave_p95": [np.nan, 50, 50, 250, 50, 100]})
        r = stratify_first_bins(w)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.set_index("case_id").stratum.to_dict(),
                         {1: "UNCLASSIFIED", 2: "LOW_OUTSIDE_RANGE", 3: "LOW_IN_RANGE",
                          4: "MAP40_TO_LT65", 5: "MAP_GE65"})

    def test_sampling_is_deterministic_without_replacement(self):
        f = pd.DataFrame({"case_id": np.arange(40), "stratum": np.repeat(
            ["LOW_OUTSIDE_RANGE", "LOW_IN_RANGE", "MAP40_TO_LT65", "MAP_GE65"], 10)})
        a = select_cases(f)
        self.assertEqual(len(a), 32)
        self.assertEqual(a.groupby("stratum").size().tolist(), [8, 8, 8, 8])
        self.assertEqual(a.case_id.nunique(), 32)
        self.assertEqual(a.case_id.tolist(), select_cases(f).case_id.tolist())

    def test_missing_times_cannot_prove_no_boundary_difference(self):
        r = boundary_state(False, 0)
        self.assertEqual(r.get("comparison_status"), "NOT_IDENTIFIABLE_MISSING_SOURCE_TIMESTAMPS")
        self.assertIsNone(r["any_sample_difference"])
        self.assertIsNone(r["one_or_two_sample_difference"])
        self.assertFalse(r["invariance_verified"])

    def test_complete_time_boundary_comparison_counts_one_or_two_samples(self):
        self.assertEqual(boundary_state(True, 0).get("any_sample_difference"), False)
        self.assertEqual(boundary_state(True, 1)["one_or_two_sample_difference"], True)
        self.assertEqual(boundary_state(True, 2)["one_or_two_sample_difference"], True)
        self.assertEqual(boundary_state(True, 3)["one_or_two_sample_difference"], False)
        self.assertEqual(boundary_state(True, 3)["any_sample_difference"], True)

    def test_unknown_raw_gate_is_not_treated_as_a_failed_or_unchanged_gate(self):
        r = gate_change_state(False, True, False)
        self.assertEqual(r.get("comparison_status"), "NOT_IDENTIFIABLE_MISSING_SOURCE_TIMESTAMPS")
        self.assertIsNone(r["raw_core_good"])
        self.assertIsNone(r["core_gate_changed"])
        self.assertFalse(r["invariance_verified"])
        self.assertTrue(gate_change_state(True, True, False)["core_gate_changed"])
        self.assertFalse(gate_change_state(True, True, True)["core_gate_changed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--extend-boundary-report", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result, output = run_self_tests()
        print(output, end="")
        raise SystemExit(0 if result["passed"] else 1)
    if args.extend_boundary_report:
        raise SystemExit(extend_existing_report())
    raise SystemExit(run_audit())
