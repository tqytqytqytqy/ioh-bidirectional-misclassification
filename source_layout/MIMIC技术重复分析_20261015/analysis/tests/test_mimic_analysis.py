import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
SPEC = importlib.util.find_spec("mimic_analysis")
runner = importlib.util.module_from_spec(SPEC) if SPEC else None
if runner is not None:
    SPEC.loader.exec_module(runner)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "The bounded MIMIC runner has not been implemented")

    def test_qc_masks_without_filling_or_using_waveform_mean(self):
        values = np.array([80., 60., np.nan, 50.])
        primary = np.array([True, False, True, True])
        strict = np.array([True, True, True, False])
        coherent = np.array([False, True, True, True])
        np.testing.assert_equal(runner.build_reference(values, primary), [80., np.nan, np.nan, 50.])
        np.testing.assert_equal(runner.build_reference(values, primary, strict), [80., np.nan, np.nan, np.nan])
        np.testing.assert_equal(runner.build_reference(values, primary, coherent), [np.nan, np.nan, np.nan, 50.])
        self.assertRaises(ValueError, runner.build_reference, [19.], [True])

    def test_initial_display_and_missing_scheduled_samples(self):
        values = np.array([np.nan, 55., 80., np.nan, 70., 60., 90.])
        display, held = runner.make_display(values, 3, 0)
        np.testing.assert_equal(display, [np.nan] * 6 + [90.])
        np.testing.assert_equal(held, [-1] * 6 + [6])
        display, held = runner.make_display(values, 3, 0, initialize=True)
        np.testing.assert_equal(display, [np.nan, 55., 55., 55., 55., 55., 90.])
        np.testing.assert_equal(held, [-1, 1, 1, 1, 1, 1, 6])

    def test_missing_breaks_events_observed_normal_only_bridges(self):
        values = np.r_[np.full(3, 60.), 70., np.full(3, 60.), np.nan, np.full(6, 60.)]
        self.assertEqual(runner.reference_events(values, 65, 0, 6), [(8, 14, 6)])
        self.assertEqual(runner.reference_events(values, 65, 1, 6), [(0, 7, 6), (8, 14, 6)])
        self.assertEqual(runner.reference_events(values, 65, 2, 6), [(0, 7, 6), (8, 14, 6)])
        self.assertEqual(runner.reference_events(values, 65, 2, 7), [])

    def test_fresh_overrides_inheritance_and_never_is_distinct(self):
        values = np.r_[60., 80., np.full(6, 60.), np.full(2, 80.)]
        events = runner.reference_events(values, 65, 0, 6)
        display, held = runner.make_display(values, 30, 0)
        result = runner.classify_events(values, display, held, events, 65)
        np.testing.assert_equal(result["category"], [1])
        self.assertTrue(result["carry_in"][0])
        display, held = runner.make_display(values, 3, 0)
        result = runner.classify_events(values, display, held, events, 65)
        np.testing.assert_equal(result["category"], [0])
        self.assertTrue(result["carry_in"][0])
        display, held = runner.make_display(values, 30, 1)
        result = runner.classify_events(values, display, held, events, 65)
        np.testing.assert_equal(result["category"], [2])

    def test_decomposition_identity_and_valid_reference_denominator(self):
        values = np.array([60., 80., np.nan, 50., 70.])
        display = np.array([np.nan, 50., 20., 60., 60.])
        raw = runner.decompose_deficit(values, display, 65, 1 / 6)
        self.assertAlmostEqual(raw["display_auc"] - raw["true_auc"], raw["overdisplay_auc"] - raw["hidden_auc"])
        self.assertAlmostEqual(raw["hidden_auc_display_unavailable"], 5 / 6)
        self.assertAlmostEqual(raw["display_valid_min"] + raw["display_unavailable_min"], 4 / 6)
        self.assertAlmostEqual(raw["true_auc"], 20 / 6)

    def test_earliest_window_uses_only_coverage(self):
        values = np.r_[np.full(73, np.nan), np.full(400, 90.)]
        self.assertEqual(runner.earliest_window(values), 1)
        values[73:] = 40.
        self.assertEqual(runner.earliest_window(values), 1)
        self.assertIsNone(runner.earliest_window(np.full(359, 60.)))
        self.assertIsNone(runner.earliest_window(np.full(500, np.nan)))

    def test_all_offsets_and_absolute_window_phase_and_display_history(self):
        values = np.full(750, 80.)
        values[350:390] = 50.
        full, _, _ = runner.analyze_record(values, "case", "subject", 7500., "primary",
                                          thresholds=(65.,), intervals=(5.,))
        window, _, _ = runner.analyze_record(values, "case", "subject", 7500., "window",
                                            window=(361, 721), thresholds=(65.,), intervals=(5.,))
        self.assertEqual(len(full), 30)
        self.assertEqual(window.offset_sec.tolist(), list(range(0, 300, 10)))
        chosen = window.loc[window.offset_sec == 0].iloc[0]
        display, _ = runner.make_display(values, 30, 0)
        expected = runner.decompose_deficit(values[361:721], display[361:721], 65, 1 / 6)
        self.assertAlmostEqual(chosen.display_auc, expected["display_auc"])
        self.assertAlmostEqual(chosen.original_record_hours, 1.)
        self.assertAlmostEqual(chosen.valid_reference_min, 60.)
        self.assertNotEqual(chosen.display_auc,
                            runner.decompose_deficit(values[361:721], runner.make_display(values[361:721], 30, 0)[0], 65, 1 / 6)["display_auc"])

    def test_startup_exclusions_keep_original_schedule_and_hour_denominator(self):
        values = np.full(400, 60.)
        baseline, _, _ = runner.analyze_record(values, "case", "subject", 4005., "primary",
                                               thresholds=(65.,), intervals=(5.,))
        initialized, _, _ = runner.analyze_record(values, "case", "subject", 4005., "init",
                                                  startup="initialize", thresholds=(65.,), intervals=(5.,))
        excluded, _, _ = runner.analyze_record(values, "case", "subject", 4005., "predisplay",
                                               startup="exclude_predisplay", thresholds=(65.,), intervals=(5.,))
        first5, _, _ = runner.analyze_record(values, "case", "subject", 4005., "first5",
                                             startup="exclude_first5", thresholds=(65.,), intervals=(5.,))
        row = baseline.loc[baseline.offset_sec == 290].iloc[0]
        self.assertAlmostEqual(row.hidden_auc, 29 * 5 / 6)
        self.assertAlmostEqual(initialized.hidden_auc.sum(), 0.)
        self.assertAlmostEqual(excluded.loc[excluded.offset_sec == 290].iloc[0].valid_reference_min, 371 / 6)
        self.assertAlmostEqual(first5.valid_reference_min.iloc[0], 370 / 6)
        self.assertAlmostEqual(first5.original_record_hours.iloc[0], 4005 / 3600)
        self.assertAlmostEqual(first5.display_auc.iloc[0], 370 * 5 / 6)

    def test_event_robustness_is_cumulative_low_not_span_and_no_missing_bridge(self):
        values = np.r_[np.full(9, 60.), 70., 70., np.full(9, 60.), np.nan,
                       np.full(30, 60.), np.full(340, 80.)]
        _, events, _ = runner.analyze_record(values, "case", "subject", len(values) * 10., "primary",
                                             thresholds=(65.,), intervals=(5.,))
        phase0 = events.loc[events.offset_sec == 0]
        self.assertEqual(len(phase0), 9)
        self.assertEqual(phase0.loc[(phase0.gap_sec == 20) & (phase0.minimum_low_sec == 180), "episodes"].item(), 2)
        self.assertEqual(phase0.loc[(phase0.gap_sec == 20) & (phase0.minimum_low_sec == 300), "episodes"].item(), 1)
        self.assertEqual(phase0.loc[(phase0.gap_sec == 10) & (phase0.minimum_low_sec == 180), "episodes"].item(), 1)
        np.testing.assert_array_equal((events.fresh + events.inherited_only + events.never_low).to_numpy(), events.episodes.to_numpy())

    def test_phase_averages_then_subject_cluster_ratio_of_sums(self):
        rows = []
        for record, subject, numerators, denominator in [("a", "s1", [0., 2.], 1.),
                                                       ("b", "s1", [2., 2.], 2.),
                                                       ("c", "s2", [10., 10.], 100.)]:
            for phase, numerator in enumerate(numerators):
                rows.append(dict(record_id=record, subject_id=subject, scenario="x", threshold=65.,
                                 interval_min=1., offset_sec=phase * 10,
                                 hidden_auc=numerator, true_auc=denominator))
        frame = pd.DataFrame(rows)
        avg, subjects, totals, draws, indices = runner.cluster_sums(frame, ["hidden_auc", "true_auc"], 1000, runner.SEED)
        self.assertEqual(len(avg), 3)
        self.assertEqual(len(subjects), 2)
        self.assertAlmostEqual(totals[0] / totals[1], 13 / 103)
        expected = np.random.default_rng(20260706).integers(0, 2, size=(1000, 2))
        np.testing.assert_equal(indices, expected)
        np.testing.assert_allclose(draws, subjects[expected].sum(axis=1))
        lo, hi, n = runner.ratio_ci(draws[:, 0], draws[:, 1])
        self.assertEqual(n, 1000)
        self.assertAlmostEqual(lo, .1)
        self.assertAlmostEqual(hi, 1.)

    def test_zero_reference_auc_does_not_create_zero_ratio_or_claim_support(self):
        values = np.full(360, 90.)
        freq, events, _ = runner.analyze_record(values, "case", "subject", 3600., "primary",
                                               thresholds=(65.,), intervals=(5.,))
        summary, details = runner.summarize_frequency(freq)
        row = summary.iloc[0]
        self.assertTrue(np.isnan(row.hidden_deficit_ratio))
        self.assertEqual(row.hidden_deficit_ratio_valid_bootstrap_replicates, 0)
        self.assertEqual(row.reference_twa, 0.)
        self.assertEqual(row.valid_reference_min, 60.)
        self.assertEqual(len(details), 1)
        summary, _ = runner.summarize_events(events)
        self.assertTrue(np.isnan(summary.iloc[0].fresh_probability))

    def test_vendor_hashes_and_contract_are_frozen(self):
        provenance = json.loads((CODE / "provenance.json").read_text())
        for item in provenance["vendored"]:
            self.assertEqual(hashlib.sha256((CODE / item["copy"]).read_bytes()).hexdigest(), item["sha256"])
        self.assertEqual(runner.CONTRACT_SHA256, hashlib.sha256((HERE.parent.parent / "analysis_contract.json").read_bytes()).hexdigest())

    def test_within_record_phase_range_then_distribution_excludes_undefined_rates(self):
        spec = importlib.util.find_spec("summarize_phase_ranges")
        self.assertIsNotNone(spec, "Within-record phase ranges have not been implemented")
        ranges = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ranges)
        rows = []
        for record, denominator in [("case_a", 6), ("case_b", 0)]:
            for phase in range(30):
                fresh = phase % 7 if denominator else 0
                rows.append(dict(record_id=record, subject_id=record, scenario="primary", threshold=65.,
                                 interval_min=5., gap_sec=0, minimum_low_sec=60, offset_sec=phase * 10,
                                 episodes=denominator, fresh=fresh, never_low=denominator - fresh))
        private, aggregate = ranges.summarize_phase_ranges(pd.DataFrame(rows))
        self.assertEqual(len(private), 2)
        self.assertEqual(len(aggregate), 6)
        self.assertEqual(private.loc[private.record_id == "case_a", "fresh_probability_range"].item(), 1.)
        self.assertTrue(np.isnan(private.loc[private.record_id == "case_b", "fresh_probability_range"].item()))
        select = aggregate.loc[(aggregate.metric == "never_low_probability") & (aggregate.statistic == "within_record_range")].iloc[0]
        self.assertEqual(select.n_records, 2)
        self.assertEqual(select.n_defined, 1)
        self.assertEqual(select.p50, 1.)
        incomplete = pd.DataFrame(rows).iloc[1:]
        self.assertRaises(ValueError, ranges.summarize_phase_ranges, incomplete)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "The bounded MIMIC runner has not been implemented")
        self.temp = tempfile.TemporaryDirectory(dir=HERE, prefix="synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "private").mkdir()
        shutil.copyfile(HERE.parent.parent / "analysis_contract.json", self.root / "analysis_contract.json")

    def fixture(self):
        records, arrays = [], {}
        specs = [("z1", "s1", True, True, 400, 0), ("z2", "s1", True, True, 400, 85),
                 ("z3", "s2", True, True, 360, 0), ("z4", "s3", None, True, 360, 0),
                 ("z5", "s4", True, False, 360, 0), ("z6", "s5", True, True, 359, 0)]
        for rid, sid, adult, icu, n, bad in specs:
            numeric = np.full(n, 80.)
            numeric[60:96] = 50.
            primary = np.ones(n, dtype=bool)
            primary[:bad] = False
            strict = primary.copy()
            strict[100:170] = False
            records.append(dict(record_id=rid, subject_id=sid, duration_seconds=n * 10., adult=adult,
                                full_record_inside_icu=icu, primary_family="ABP", original_numeric_valid_bins=n, n_bins=n))
            for prefix, array in [("numeric", numeric), ("waveform_qc", primary), ("waveform_strict", strict),
                                  ("coherent", np.ones(n, dtype=bool)), ("waveform_mean", numeric + 100.)]:
                arrays[f"{prefix}_{rid}"] = array
        np.savez_compressed(self.root / "private/qc_inputs.npz", **arrays)
        (self.root / "private/records.json").write_text(json.dumps(records))
        files = {name: hashlib.sha256((self.root / "private" / name).read_bytes()).hexdigest()
                 for name in ("qc_inputs.npz", "records.json")}
        (self.root / "private/qc_ready.json").write_text(json.dumps({"status": "ready", "input_sha256": files}))
        return records

    def test_eligibility_uses_full_grid_and_window_is_separate(self):
        self.fixture()
        inputs, attrition, private = runner.load_inputs(self.root)
        self.assertEqual(attrition["candidate_records"], 6)
        self.assertEqual(attrition["primary_records"], 2)
        self.assertEqual(attrition["primary_subjects"], 2)
        self.assertEqual(attrition["earliest_hour_records"], 3)
        self.assertEqual(len(inputs), 6)
        self.assertEqual(len(inputs[1]["references"]["primary"]), 400)
        self.assertEqual(private.loc[private.record_id == "z2", "earliest_hour_start_bin"].item(), 13)
        self.assertFalse(private.loc[private.record_id == "z2", "primary_included"].item())

    def test_cli_missing_input_fails_fast_without_server_or_identifiers(self):
        result = subprocess.run([sys.executable, str(CODE / "mimic_analysis.py"), "--root", str(self.root), "--check-inputs"],
                                capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 3)
        self.assertEqual(json.loads(result.stdout)["status"], "inputs_not_ready")
        self.assertEqual(result.stderr, "")

    def test_manifest_hash_mismatch_blocks_run(self):
        self.fixture()
        (self.root / "private/qc_ready.json").write_text(json.dumps({"status": "ready", "input_sha256": {"records.json": "bad"}}))
        self.assertRaises(ValueError, runner.validate_ready_manifest, self.root, self.root / "private/qc_ready.json")

    def test_actual_cli_synthetic_all_scenarios_full_bootstrap_and_private_outputs(self):
        self.fixture()
        result = subprocess.run([sys.executable, str(CODE / "mimic_analysis.py"), "--root", str(self.root),
                                 "--ready-manifest", str(self.root / "private/qc_ready.json")],
                                capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual(status["status"], "complete")
        summary = json.loads((self.root / "analysis/aggregate/results.json").read_text())
        self.assertEqual(summary["counts"]["primary_records"], 2)
        self.assertEqual(summary["uncertainty"]["replicates"], 1000)
        self.assertEqual(summary["invariants"]["status"], "PASS")
        self.assertFalse(summary["clinical_adjudication"])
        frequency = pd.read_csv(self.root / "analysis/aggregate/frequency_summary.csv")
        self.assertEqual(set(frequency.interval_min), {1., 2.5, 3., 5., 10.})
        self.assertEqual(set(frequency.threshold), {55., 60., 65.})
        self.assertEqual(set(frequency.scenario), set(runner.SCENARIOS))
        self.assertTrue((self.root / "private/analysis/frequency_by_record_phase.csv.gz").exists())
        self.assertTrue((self.root / "private/analysis/bootstrap_details.npz").exists())
        self.assertTrue((self.root / "analysis/aggregate/figure_source_data.csv").exists())
        for path in (self.root / "analysis/aggregate").rglob("*"):
            if path.is_file():
                data = path.read_text(errors="ignore")
                for private_id in ["z1", "z2", "z3", "z4", "z5", "z6", "s1", "s2", "s3", "s4", "s5"]:
                    self.assertNotIn('"' + private_id + '"', data)
        self.assertNotIn("z1", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
