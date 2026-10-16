"""VitalDB documented waveform encoding audit; preserve the separate literal QA."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.dont_writebytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "外部验证可行性审计_20261015/mimic/private/deps"))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

DOC_URL = "https://vitaldb.net/dataset/?query=api"
SDK_COMMIT = "d76480ff3ace7f88f59233e55dce9a7e3d6c22e4"
SDK_URL = f"https://github.com/vitaldb/vitalutils/blob/{SDK_COMMIT}/python/vitaldb-pypi/vitaldb/dataset.py"
TOLERANCES = {"wave_coverage": 1e-10, "wave_mean": 1e-8, "wave_p05": 1e-8,
              "wave_p95": 1e-8, "wave_span_90pct": 1e-8, "peak_count": 0.0}
TIME_ATOL = 1e-8
BOXES = (0, 6, 30, 59)


def parse_encoding(first_time, second_time):
    origin, interval = float(first_time), float(second_time)
    if not (np.isfinite(origin) and np.isfinite(interval) and 0 < interval <= 1):
        raise ValueError("Invalid documented waveform header")
    return {"origin_sec": origin, "interval_sec": interval, "fs_hz": 1 / interval,
            "nonzero_start_requires_review": origin != 0.}


def metrics_for_box(values, origin, interval, left):
    v = np.asarray(values, dtype=np.float64)
    times = origin + np.arange(len(v), dtype=np.float64) * interval
    indices = np.flatnonzero((times >= left) & (times < left + 10.))
    x = v[indices]
    finite = np.isfinite(x)
    fs = 1 / interval
    distance = int(.3 * fs)
    if distance < 1:
        raise ValueError("Unsupported sampling interval for frozen peak algorithm")
    r = {"row_count": len(indices), "finite_count": int(finite.sum()),
         "first_index": int(indices[0]) if len(indices) else None,
         "stop_index": int(indices[-1] + 1) if len(indices) else None,
         "wave_coverage": float(finite.sum() / (10 * fs)),
         "wave_mean": np.nan, "wave_p05": np.nan, "wave_p95": np.nan,
         "wave_span_90pct": np.nan, "peak_count": 0, "peak_distance_samples": distance}
    if finite.any():
        lo, hi = np.quantile(x[finite], [.05, .95], method="linear")
        r.update(wave_mean=float(x[finite].mean()), wave_p05=float(lo), wave_p95=float(hi),
                 wave_span_90pct=float(hi - lo))
        y = np.nan_to_num(x, nan=float(np.median(x[finite])))
        r["peak_count"] = int(len(find_peaks(y, distance=distance, prominence=5)[0]))
    return r


def compare_number(a, b, tolerance):
    a, b = float(a), float(b)
    both_nan = np.isnan(a) and np.isnan(b)
    both_finite = np.isfinite(a) and np.isfinite(b)
    delta = abs(a-b) if both_finite else np.nan
    return {"match": bool(both_nan or (both_finite and delta <= tolerance)),
            "absolute_difference": delta,
            "finite_vs_missing": bool(np.isfinite(a) != np.isfinite(b))}


def core_components(row, specification):
    s = specification
    m, mean, sbp, dbp = (float(row[k]) for k in ("numeric_map", "wave_mean", "sbp", "dbp"))
    between = lambda x, bounds: bool(bounds[0] <= x <= bounds[1])
    paired = np.isfinite(sbp) and np.isfinite(dbp)
    pair = s["if_both_numeric_SBP_DBP_available"]
    pair_ok = not paired or (between(sbp, pair["SBP_mmHg"]) and between(dbp, pair["DBP_mmHg"]) and
                            between(sbp-dbp, pair["pulse_pressure_mmHg"]) and
                            dbp-pair["MAP_within_DBP_SBP_tolerance_mmHg"] <= m <=
                            sbp+pair["MAP_within_DBP_SBP_tolerance_mmHg"])
    return {"numeric_range": bool(np.isfinite(m) and between(m, s["finite_numeric_MAP_mmHg"])),
            "coverage": bool(row["wave_coverage"] >= s["waveform_coverage_at_least"]),
            "waveform_mean": bool(np.isfinite(mean) and between(mean, s["waveform_mean_mmHg"])),
            "waveform_range": bool(row["wave_p05"] >= s["waveform_p05_at_least_mmHg"] and
                                   row["wave_p95"] <= s["waveform_p95_at_most_mmHg"]),
            "pulsatility": bool(between(row["wave_span_90pct"], s["waveform_p95_minus_p05_mmHg"]) and
                                between(row["peak_count"], s["peak_count_per10s"])),
            "numeric_wave_agreement": bool(abs(mean-m) <= s["absolute_waveform_mean_minus_numeric_MAP_at_most_mmHg"]),
            "optional_pair_coherence": bool(pair_ok)}


def prefix_count(origin, interval, exclusive_end, full_count):
    return min(int(full_count), max(2, int(math.ceil((exclusive_end-origin)/interval)) + 2))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_tests():
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(EncodingTests))
    return {"tests_run": result.testsRun, "passed": result.wasSuccessful(),
            "failures": len(result.failures), "errors": len(result.errors)}, output.getvalue()


def run_audit():
    os.umask(0o077)
    output_path = ROOT / "qa/documented_encoding_check.json"
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite an existing documented-encoding audit")
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    tests, test_output = run_tests()
    if not tests["passed"]:
        print(test_output)
        return 2
    paths = {"literal_qa": ROOT / "qa/raw_cache_check.json",
             "literal_details": ROOT / "restricted/raw_cache_check_details.csv",
             "literal_log": ROOT / "restricted/audit_raw.internal.jsonl",
             "literal_code": ROOT / "code/audit_raw.py",
             "contract": ROOT / "analysis_contract.json",
             "existing_gate_table": ROOT / "restricted/startup_quality_bins.csv.gz",
             "protected_manifest": ROOT / "qa/protected_input_hashes.json"}
    protected = json.loads(paths["protected_manifest"].read_text())
    for i, p in enumerate(protected):
        paths[f"protected_{i+1:02d}"] = Path(p)
    before = {k: sha256(p) for k, p in paths.items()}
    if any(before[k] != protected[str(p)] for k, p in paths.items() if str(p) in protected):
        raise ValueError("Frozen inputs differ from protected hashes")
    literal = json.loads(paths["literal_qa"].read_text())
    if literal["restricted_details_sha256"] != before["literal_details"]:
        raise ValueError("Literal details do not match literal QA")
    for name, tolerance in TOLERANCES.items():
        if literal["metric_comparisons"][name]["absolute_tolerance"] != tolerance:
            raise ValueError("Original tolerances changed")
    if literal["methods"]["time_comparison_absolute_tolerance_sec"] != TIME_ATOL:
        raise ValueError("Original time tolerance changed")
    d = pd.read_csv(paths["literal_details"], float_precision="round_trip")
    cases = d.drop_duplicates("case_id").sort_values("sample_order")
    if len(d) != 128 or len(cases) != 32 or d.duplicated(["case_id", "bin_index_zero_based"]).any():
        raise ValueError("Unexpected frozen audit selection")
    if sorted(cases.stratum.value_counts().tolist()) != [8, 8, 8, 8]:
        raise ValueError("Frozen selection does not contain four strata of eight")
    if not d.groupby("case_id").bin_index_zero_based.apply(lambda x: sorted(x) == list(BOXES)).all():
        raise ValueError("Frozen selection does not contain the requested boxes")
    contract = json.loads(paths["contract"].read_text())
    gate = pd.read_csv(paths["existing_gate_table"], float_precision="round_trip").set_index(["case_id", "time_sec"])
    spec = contract["primary_bin_gate"]
    results, source_checks, errors = [], [], []
    for order, case in enumerate(cases.itertuples(index=False), 1):
        try:
            p = Path(case.source_path)
            pre = p.stat()
            if pre.st_size != int(case.source_bytes):
                raise ValueError("Source file size changed since prior full scan")
            header = pd.read_csv(p, nrows=2, dtype=str, na_values=["-nan(ind)"])
            if list(header.columns) != ["Time", "SNUADC/ART"] or len(header) != 2:
                raise ValueError("Unexpected source waveform header")
            encoding = parse_encoding(header.iloc[0, 0], header.iloc[1, 0])
            if encoding["nonzero_start_requires_review"]:
                raise ValueError("Nonzero first Time requires source-specific review; no interval subtraction applied")
            origin, interval = encoding["origin_sec"], encoding["interval_sec"]
            if abs(origin-float(case.nominal_origin_sec)) > TIME_ATOL or abs(interval-float(case.nominal_interval_sec)) > TIME_ATOL:
                raise ValueError("Documented header differs from prior source scan")
            left_base = float(case.start_abs_sec) + float(case.first_valid_rel_sec)
            n = prefix_count(origin, interval, left_base + 600., int(case.source_rows))
            data = pd.read_csv(p, nrows=n, dtype=str, na_values=["-nan(ind)"])
            if len(data) != n:
                raise ValueError("Raw prefix shorter than prior source count")
            values = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(np.float64)
            post = p.stat()
            if (pre.st_size, pre.st_mtime_ns) != (post.st_size, post.st_mtime_ns):
                raise ValueError("Source changed during prefix read")
            # The previous scan retained absolute end discrepancy, not signed T_end.
            endpoint_reusable = bool(int(case.time_finite_rows) == 3 and
                                     int(case.explicit_row_indices_min) == 0 and
                                     int(case.explicit_row_indices_max) == int(case.source_rows)-1 and
                                     origin == float(case.nominal_origin_sec) == 0. and
                                     interval == float(case.nominal_interval_sec))
            source_checks.append({"origin": origin, "interval": interval, "prefix_rows": n,
                                  "full_rows": int(case.source_rows), "endpoint_reusable": endpoint_reusable,
                                  "endpoint_absolute_difference": float(case.max_explicit_vs_rebuilt_abs_sec) if endpoint_reusable else np.nan,
                                  "nominal_end_sec": origin + (int(case.source_rows)-1)*interval,
                                  "source_stat_unchanged": True})
            nominal_times = origin + np.arange(n, dtype=np.float64) * interval
            old_times = float(case.nominal_origin_sec) + np.arange(n, dtype=np.float64) * float(case.nominal_interval_sec)
            rows = d[d.case_id == case.case_id].sort_values("bin_index_zero_based")
            for row in rows.itertuples(index=False):
                b = int(row.bin_index_zero_based)
                left = left_base + b * 10.
                raw = metrics_for_box(values, origin, interval, left)
                old = gate.loc[(int(case.case_id), float(row.cache_time_sec))]
                metrics = {name: compare_number(getattr(row, "cache_"+name), raw[name], tol)
                           for name, tol in TOLERANCES.items()}
                for name, tol in TOLERANCES.items():
                    if not compare_number(getattr(row, "cache_"+name), old[name], tol)["match"]:
                        raise ValueError("Existing gate table does not match frozen cache features")
                fixed = {k: float(old[k]) for k in ("numeric_map", "sbp", "dbp")}
                original_features = {name: float(getattr(row, "cache_"+name)) for name in TOLERANCES}
                original_flags = core_components({**fixed, **original_features}, spec)
                raw_flags = core_components({**fixed, **raw}, spec)
                old_good, raw_good = all(original_flags.values()), all(raw_flags.values())
                if old_good != bool(old.core_good) or old_good != bool(row.cache_core_good_existing_table):
                    raise ValueError("Independent frozen-rule core gate differs from existing gate table")
                actual_members = (nominal_times >= left) & (nominal_times < left+10.)
                old_members = (old_times >= left) & (old_times < left+10.)
                observed_boundary_change = int(np.count_nonzero(actual_members != old_members))
                cached_rebuilt_count = int(row.raw_nominal_row_count)
                results.append({"stratum": case.stratum, "box": b, "metrics": metrics,
                                "all_metrics_match": all(x["match"] for x in metrics.values()),
                                "cache_core_good": old_good, "raw_core_good": raw_good,
                                "core_gate_changed": old_good != raw_good,
                                "component_changes": {k: original_flags[k] != raw_flags[k] for k in original_flags},
                                "boundary_changed_sample_assignments": observed_boundary_change,
                                "one_or_two_boundary_assignments_changed": observed_boundary_change in (1, 2),
                                "nominal_row_count": raw["row_count"],
                                "row_count_differs_from_prior_nominal_reconstruction": raw["row_count"] != cached_rebuilt_count,
                                "left_boundary_sample_count": int((nominal_times == left).sum()),
                                "right_boundary_sample_count": int((nominal_times == left+10.).sum()),
                                "exact_left_included": bool(np.all(actual_members[nominal_times == left])),
                                "exact_right_excluded": bool(np.all(~actual_members[nominal_times == left+10.])),
                                "peak_distance_samples": raw["peak_distance_samples"]})
        except Exception as e:
            # No case identifiers, raw paths, or error messages are emitted to aggregate QA.
            errors.append(type(e).__name__)
        if order % 4 == 0:
            print(json.dumps({"documented_encoding_records_processed": order, "planned": 32,
                              "boxes_computed": len(results), "source_errors": len(errors)}), flush=True)
    after = {k: sha256(p) for k, p in paths.items()}
    unchanged = before == after
    complete = len(results) == 128 and len(source_checks) == 32 and not errors
    all_match = bool(complete and all(r["all_metrics_match"] for r in results))
    gate_changed = sum(r["core_gate_changed"] for r in results)
    metric_summary = {}
    for name, tol in TOLERANCES.items():
        comparisons = [r["metrics"][name] for r in results]
        finite_delta = [r["absolute_difference"] for r in comparisons if np.isfinite(r["absolute_difference"])]
        metric_summary[name] = {"absolute_tolerance": tol, "relative_tolerance": 0,
                                "boxes_compared": len(comparisons),
                                "boxes_matching": sum(r["match"] for r in comparisons),
                                "boxes_mismatching": sum(not r["match"] for r in comparisons),
                                "finite_vs_missing_boxes": sum(r["finite_vs_missing"] for r in comparisons),
                                "maximum_absolute_difference": max(finite_delta) if finite_delta else None}
    endpoints = [s["endpoint_absolute_difference"] for s in source_checks if s["endpoint_reusable"]]
    output = {
        "schema_version": 1, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-clock,
        "status": "DOCUMENTED_ENCODING_SAMPLE_MATCH" if all_match and not gate_changed and unchanged else "DOCUMENTED_ENCODING_AUDIT_HAS_DIFFERENCES_OR_INCOMPLETE_SOURCE",
        "documented_encoding_sample_pass": bool(all_match and not gate_changed and unchanged),
        "source_format_interpretation": "Waveform Time is a three-value encoding: start=0, sampling interval, end. Blank Time cells are encoding, not missing waveform timing observations. Missing VALUE rows remain in place.",
        "official_sources": {
            "web_api_documentation": {"url": DOC_URL, "section": "Track data / Waveform data track", "access_date": "2026-10-15"},
            "sdk": {"url": SDK_URL+"#L68-L89", "commit": SDK_COMMIT, "commit_date": "2026-05-28T14:49:29Z",
                    "commit_subject": "vitaldb 1.6.4: VitalFile(caseid) reads .vital; deprecate trks/CSV API",
                    "raw_file_sha256_verified_at_retrieval": "2c39d6ed4e2ea7548d34db2b8363928c26c54bc36a50872d7f806a6e3a309085",
                    "precise_wave_csv_rule": [
                        "Read Time and values as float32, with '-nan(ind)' recognized as missing.",
                        "Divide Time column in-place by requested output interval, then nsamp=int(nanmax(Time/interval))+1.",
                        "Any NaN Time selects the waveform branch. If nsamp==N, return values directly.",
                        "Otherwise return values at int64(linspace(0,N-1,nsamp)) using np.take; this is index selection, not amplitude interpolation."],
                    "sdk_resampling_not_used_in_primary_check": True}},
        "selection": {"records": len(cases), "requested_boxes": 128, "boxes_zero_based": list(BOXES),
                      "same_cases_and_boxes_as_literal_audit": True, "resampled_case_selection": False,
                      "original_selection_seed": 20260706,
                      "by_stratum": {str(k): int(v) for k, v in cases.stratum.value_counts().items()},
                      "literal_details_sha256": before["literal_details"]},
        "method": {"time_axis": "t_i = first_Time + i * second_Time; second_Time is the encoded sampling interval itself, NOT generally second_Time-first_Time",
                   "all_sample_times_generated_from_documented_encoding": True,
                   "nonzero_first_time_records": sum(s["origin"] != 0 for s in source_checks),
                   "nonzero_start_policy": "Do not silently subtract first_Time from interval; stop that record for source-specific review because the documented format specifies start=0",
                   "sampling_intervals_sec": sorted(set(s["interval"] for s in source_checks)),
                   "dtype": "float64 numeric parsing and reductions to match the frozen feature computation; not SDK float32 resampling",
                   "bins": "[start_abs_sec+first_valid_rel_sec+10*b, same+10); direct >= left and < right; no epsilon, snapping or endpoint stretch",
                   "coverage": "finite VALUE count/(10*fs); keep rows with missing VALUE to preserve sample indices",
                   "quantiles": "numpy.quantile(finite_values,[0.05,0.95],method='linear')",
                   "peaks": "find_peaks with distance=int(0.3*fs), prominence=5; replace NaN VALUE with finite within-box median, preserving the frozen preprocessing",
                   "tolerances_unchanged_from_literal_audit": True,
                   "time_absolute_tolerance_sec": TIME_ATOL,
                   "no_whole_file_rescan": True,
                   "prefix_rows_read": sum(s["prefix_rows"] for s in source_checks),
                   "full_source_rows_previously_scanned": sum(s["full_rows"] for s in source_checks),
                   "current_source_full_sha256_recomputed": False,
                   "current_source_validation": "Same frozen paths and file sizes; stat unchanged during prefix reads. Full-file hashes and anchor counts retained in prior restricted audit; no redundant full-file hashing."},
        "counts": {"records_processed_successfully": len(source_checks), "boxes_computed": len(results),
                   "boxes_all_six_features_match": sum(r["all_metrics_match"] for r in results),
                   "boxes_any_feature_mismatch": sum(not r["all_metrics_match"] for r in results),
                   "source_errors": len(errors), "error_type_counts": {e: errors.count(e) for e in sorted(set(errors))}},
        "feature_comparisons": metric_summary,
        "box_boundary_comparison": {
            "comparison": "Documented nominal grid versus prior frozen nominal reconstruction; not a comparison with independently measured per-sample clock timestamps",
            "boxes_compared": len(results),
            "boxes_with_any_sample_assignment_difference": sum(r["boundary_changed_sample_assignments"] > 0 for r in results),
            "boxes_with_one_or_two_sample_assignment_differences": sum(r["one_or_two_boundary_assignments_changed"] for r in results),
            "total_sample_assignment_differences": sum(r["boundary_changed_sample_assignments"] for r in results),
            "boxes_with_different_sample_count": sum(r["row_count_differs_from_prior_nominal_reconstruction"] for r in results),
            "samples_exactly_on_left_boundaries": sum(r["left_boundary_sample_count"] for r in results),
            "samples_exactly_on_right_boundaries": sum(r["right_boundary_sample_count"] for r in results),
            "all_exact_left_boundaries_included": bool(complete and all(r["exact_left_included"] for r in results)),
            "all_exact_right_boundaries_excluded": bool(complete and all(r["exact_right_excluded"] for r in results)),
            "comparison_complete_for_requested_boxes": complete},
        "core_bin_gate_comparison": {
            "source": "Independent application of frozen contract primary_bin_gate; same frozen numeric MAP/SBP/DBP from existing gate table",
            "cached_pass_boxes": sum(r["cache_core_good"] for r in results),
            "recomputed_pass_boxes": sum(r["raw_core_good"] for r in results),
            "cached_fail_boxes": sum(not r["cache_core_good"] for r in results),
            "recomputed_fail_boxes": sum(not r["raw_core_good"] for r in results),
            "changed_boxes": gate_changed,
            "pass_to_fail_boxes": sum(r["cache_core_good"] and not r["raw_core_good"] for r in results),
            "fail_to_pass_boxes": sum(not r["cache_core_good"] and r["raw_core_good"] for r in results),
            "component_changed_boxes": {k: sum(r["component_changes"][k] for r in results) for k in (results[0]["component_changes"] if results else {})},
            "unchanged_for_all_requested_boxes": bool(complete and gate_changed == 0),
            "new_onset_or_main_statistical_analysis": False,
            "six_consecutive_bin_onset_invariance_proven": False},
        "endpoint_check": {
            "nominal_last_sample": "first_Time + (source_rows-1)*second_Time",
            "absolute_end_difference_available_records": len(endpoints),
            "absolute_difference_exceeds_unchanged_time_tolerance_records": sum(e > TIME_ATOL for e in endpoints),
            "max_absolute_end_difference_sec": max(endpoints) if endpoints else None,
            "source": "Reuse prior full-source scan's |end-anchor minus nominal last-sample time|; all 32 starts=0, intervals=0.002 and exactly three encoded Time entries, so nominal axes are identical",
            "signed_end_time_retained_in_prior_details": False,
            "end_time_or_sign_inferred": False,
            "endpoint_axis_stretching_applied": False,
            "endpoint_difference_is_not_itself_a_core_gate_change": True},
        "sdk_resampling_sensitivity": {
            "performed": False,
            "reason": "Prior restricted details retain source_rows and the ABSOLUTE end discrepancy, not end_Time or its sign. Exact float32 end/interval and nsamp cannot be determined from these retained fields. No end time was guessed and no repeated full-file scan was performed.",
            "gate_effect": None, "equivalence_to_load_trk_claimed": False},
        "preservation": {"literal_qa_unchanged": before["literal_qa"] == after["literal_qa"],
                         "literal_details_unchanged": before["literal_details"] == after["literal_details"],
                         "literal_log_unchanged": before["literal_log"] == after["literal_log"],
                         "literal_code_unchanged": before["literal_code"] == after["literal_code"],
                         "all_read_only_inputs_unchanged": unchanged,
                         "read_only_input_sha256_before": before, "read_only_input_sha256_after": after,
                         "only_new_files": ["code/audit_documented_encoding.py", "qa/documented_encoding_check.json"]},
        "scope": ["Same 32 stratified records and 128 requested boxes only; not full-cohort or full-record quality proof.",
                  "Official encoding reconstruction is not missing timestamp imputation; the literal audit remains a separate preserved description of populated CSV cells.",
                  "No main statistics, outcomes, onsets, caches, contract, or manuscript were altered.",
                  "No cross-version packed .vital/CSV constant-offset correction or clinical-time alignment validation was attempted."],
        "self_tests": tests,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__},
        "script_sha256": sha256(Path(__file__)), "qa_contains_case_identifiers_or_raw_paths": False}
    output_path.write_text(json.dumps(sanitize(output), indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    print(json.dumps(sanitize({"status": output["status"], "counts": output["counts"],
                              "gate_changed_boxes": gate_changed, "preserved": unchanged})), flush=True)
    return 0 if complete and unchanged else 2


class EncodingTests(unittest.TestCase):
    def spec(self):
        return json.loads((ROOT / "analysis_contract.json").read_text())["primary_bin_gate"]

    def good(self):
        return dict(numeric_map=28., wave_coverage=1., wave_mean=29., wave_p05=14.,
                    wave_p95=45., wave_span_90pct=31., peak_count=12, sbp=45., dbp=14.)

    def test_second_time_is_an_interval_not_a_second_sample_timestamp(self):
        r = parse_encoding(5., .002)
        self.assertEqual(r.get("interval_sec"), .002)
        self.assertEqual(r["origin_sec"], 5.)
        self.assertTrue(r["nonzero_start_requires_review"])
        self.assertEqual(parse_encoding(0., .002)["fs_hz"], 500.)

    def test_missing_time_cells_do_not_remove_value_rows(self):
        r = metrics_for_box([10, np.nan, 30, 50], 0., .1, 0.)
        self.assertEqual(r.get("row_count"), 4)
        self.assertEqual(r["finite_count"], 3)
        self.assertAlmostEqual(r["wave_coverage"], .03)
        self.assertAlmostEqual(r["wave_mean"], 30.)
        self.assertAlmostEqual(r["wave_p05"], 12.)
        self.assertAlmostEqual(r["wave_p95"], 48.)
        self.assertAlmostEqual(r["wave_span_90pct"], 36.)

    def test_half_open_boundary_includes_left_and_excludes_right(self):
        r = metrics_for_box(np.arange(201), 0., .1, 10.)
        self.assertEqual(r.get("first_index"), 100)
        self.assertEqual(r["stop_index"], 200)
        self.assertEqual(r["row_count"], 100)
        self.assertEqual(r["wave_mean"], 149.5)

    def test_nan_median_peak_preprocessing_and_distance(self):
        r = metrics_for_box([0, 0, 10, np.nan, 0, 10, 0, 0], 0., .1, 0.)
        self.assertEqual(r.get("peak_count"), 2)
        self.assertEqual(r["peak_distance_samples"], 3)

    def test_prefix_stops_just_beyond_last_requested_right_edge(self):
        self.assertEqual(prefix_count(0., .002, 600., 1000000), 300002)
        self.assertEqual(prefix_count(0., .002, 600., 200000), 200000)

    def test_absolute_tolerances_and_nan_match(self):
        self.assertEqual(compare_number(1., 1.+5e-9, 1e-8).get("match"), True)
        self.assertFalse(compare_number(1., 1.+5e-7, 1e-8)["match"])
        self.assertTrue(compare_number(np.nan, np.nan, 0)["match"])
        self.assertFalse(compare_number(1., np.nan, 1e-8)["match"])

    def test_low_pressure_can_pass_core_gate(self):
        r = core_components(self.good(), self.spec())
        self.assertEqual(r.get("numeric_range"), True)
        self.assertTrue(all(r.values()))

    def test_core_cutoffs_not_relaxed(self):
        r = self.good()
        r["wave_coverage"] = .899999
        self.assertEqual(core_components(r, self.spec()).get("coverage"), False)
        r = self.good()
        r["wave_p05"] = -1e-7
        self.assertFalse(core_components(r, self.spec())["waveform_range"])

    def test_optional_paired_pressure_rule(self):
        r = self.good()
        r["sbp"] = np.nan
        self.assertEqual(core_components(r, self.spec()).get("optional_pair_coherence"), True)
        r["sbp"] = 10.
        self.assertFalse(core_components(r, self.spec())["optional_pair_coherence"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        test_result, text = run_tests()
        print(text, end="")
        raise SystemExit(0 if test_result["passed"] else 1)
    raise SystemExit(run_audit())
