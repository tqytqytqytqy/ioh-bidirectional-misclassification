"""Regression tests for frozen treatment analysis under technical QC masks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import pandas as pd
MODULE = Path(__file__).with_name('rerun_treatment.py')

class TreatmentQCTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if MODULE.exists():
            spec = importlib.util.spec_from_file_location('rerun_treatment', MODULE)
            cls.mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.mod
            spec.loader.exec_module(cls.mod)
        else:
            cls.mod = None

    def setUp(self):
        self.assertIsNotNone(self.mod, 'QC treatment adapter has not been implemented')
        self.meta = pd.DataFrame({'case_id': [1, 2], 'subjectid': [10, 10], 'anestart': [0.0, 0.0], 'aneend': [900.0, 900.0]}).set_index('case_id')

    def events(self, cases, times):
        return pd.DataFrame({'case_id': cases, 'time_sec': times, 'drug': 'NEPI', 'kind': 'observed_rate_increase', 'record_changes': 1})

    def test_dropped_case_is_excluded_without_lookup_failure(self):
        frame, exclusions = self.mod.evaluate_events(self.events([1, 2], [400.0, 400.0]), {1: np.full(90, 60.0)}, self.meta)
        self.assertEqual(frame.case_id.tolist(), [1])
        self.assertEqual(exclusions['excluded_policy_case'], 1)
        self.assertEqual(exclusions['pre_reference_quality'], 0)

    def test_masked_pre_reference_is_not_carried_forward(self):
        values = np.full(90, 60.0)
        values[39] = np.nan
        frame, exclusions = self.mod.evaluate_events(self.events([1], [400.0]), {1: values}, self.meta)
        self.assertTrue(frame.empty)
        self.assertEqual(exclusions['pre_reference_quality'], 1)
        self.assertEqual(frame.low.dtype, np.dtype('bool'))

    def test_empty_eligible_set_has_explicit_null_estimates(self):
        frame, exclusions = self.mod.evaluate_events(self.events([2], [400.0]), {}, self.meta)
        rows = self.mod.frozen.summaries(frame, 'Primary')
        self.assertEqual(len(rows), 12)
        self.assertTrue(all((r['events'] == 0 and r['estimate'] is None for r in rows)))
        recovery, results, reasons, audit = self.mod.recover_events(frame, {})
        self.assertTrue(recovery.empty)
        self.assertTrue(all((r['estimate'] is None for r in results)))

    def test_original_global_grid_is_not_compressed_or_rephased(self):
        panel = pd.DataFrame({'case_id': 1, 'time_sec': np.arange(90) * 10.0, 'art_map': np.full(90, 75.0)})
        panel.loc[0:34, 'art_map'] = np.nan
        art = self.mod.panel_to_art(panel, self.meta)
        self.assertEqual(len(art[1]), 90)
        self.assertTrue(np.isnan(art[1][:35]).all())
        state = self.mod.core.event_state(art[1], 400.0, 30, 0)
        self.assertTrue(np.isnan(state['display']))

    def test_deleted_grid_bins_fail_closed(self):
        panel = pd.DataFrame({'case_id': [1, 1, 1], 'time_sec': [0.0, 10.0, 30.0], 'art_map': [75.0, 75.0, 75.0]})
        with self.assertRaises(ValueError):
            self.mod.panel_to_art(panel, self.meta)

    def test_incomplete_final_bin_is_not_post_support(self):
        self.meta.loc[1, 'aneend'] = 899.0
        panel = pd.DataFrame({'case_id': 1, 'time_sec': np.arange(90) * 10.0, 'art_map': np.full(90, 75.0)})
        art = self.mod.panel_to_art(panel, self.meta)
        self.assertEqual(len(art[1]), 89)
        status = self.mod.post_support(art[1], 595.0)
        self.assertEqual(status['status'], 'incomplete_post_interval')

    def test_missing_post_interval_is_exclusion_not_zero_persistence(self):
        values = np.full(90, 75.0)
        values[:40] = 60.0
        values[40:47] = np.nan
        frame, _ = self.mod.evaluate_events(self.events([1], [400.0]), {1: values}, self.meta)
        rec, results, reasons, audit = self.mod.recover_events(frame, {1: values})
        self.assertTrue(rec.empty)
        self.assertEqual(audit.iloc[0].status, 'insufficient_post_reference')
        self.assertTrue(all((r['full_post_support_events'] == 0 and r['estimate'] is None for r in results)))

    def test_eighty_percent_post_reference_boundary_is_preserved(self):
        values = np.full(90, 75.0)
        values[:40] = 60.0
        values[40:46] = np.nan
        status = self.mod.post_support(values, 400.0)
        self.assertEqual(status['status'], 'supported')
        self.assertEqual(status['post_valid_bins'], 24)

    def test_no_confirmed_recovery_is_not_zero_persistence(self):
        values = np.full(90, 60.0)
        frame, _ = self.mod.evaluate_events(self.events([1], [400.0]), {1: values}, self.meta)
        rec, results, reasons, audit = self.mod.recover_events(frame, {1: values})
        self.assertTrue(rec.empty)
        self.assertEqual(audit.iloc[0].status, 'no_confirmed_recovery')
        self.assertTrue(all((r['full_post_support_events'] == 1 and r['estimate'] is None for r in results)))

    def test_recovery_stops_at_masked_bin(self):
        values = np.full(90, 60.0)
        values[40:] = 75.0
        values[44] = np.nan
        frame, _ = self.mod.evaluate_events(self.events([1], [400.0]), {1: values}, self.meta)
        rec, results, reasons, audit = self.mod.recover_events(frame, {1: values})
        self.assertEqual(len(rec), 1)
        self.assertEqual(rec.iloc[0].recovery_delay_sec, 20.0)
        self.assertTrue((reasons.stop_reason == 'missing_reference').any())
        self.assertLessEqual(rec.iloc[0].persistence_30, 0.5)

    def test_equal_subject_filter_keeps_first_low_event_only(self):
        art = {1: np.full(90, 60.0), 2: np.full(90, 60.0)}
        frame, excluded = self.mod.evaluate_events(self.events([1, 1, 2], [400.0, 500.0, 400.0]), art, self.meta, first_subject=True)
        self.assertEqual(len(frame), 1)
        self.assertEqual(excluded['not_first_low_subject_event'], 2)

    def test_masks_cannot_modify_finite_values_or_fill_original_missingness(self):
        base = {1: np.array([np.nan, 75.0, 60.0])}
        with self.assertRaises(ValueError):
            self.mod.validate_masked_art({1: np.array([75.0, 75.0, 60.0])}, base)
        with self.assertRaises(ValueError):
            self.mod.validate_masked_art({1: np.array([np.nan, 74.0, 60.0])}, base)
        result = self.mod.validate_masked_art({1: np.array([np.nan, 75.0, np.nan])}, base)
        self.assertEqual(result['newly_masked_complete_bins'], 1)

    def test_source_preserving_comparison_detects_numeric_drift(self):
        a = pd.DataFrame({'key': ['a'], 'value': [1.0]})
        self.mod.assert_frames_match(a, a.copy(), ['key'])
        with self.assertRaises(AssertionError):
            self.mod.assert_frames_match(a, a.assign(value=1.001), ['key'])

    def test_cohort_ledger_excluded_rows_are_not_reintroduced(self):
        cohort = pd.DataFrame({'case_id': [1, 2], 'subject_id': [10, 10], 'included': [True, False]})
        self.assertEqual(self.mod.select_cohort(cohort, self.meta), {1})

    def test_cohort_ledger_unknown_inclusion_fails_closed(self):
        cohort = pd.DataFrame({'case_id': [1, 2], 'subject_id': [10, 10], 'included': [True, None]})
        with self.assertRaises(ValueError):
            self.mod.select_cohort(cohort, self.meta)

    def test_cohort_subject_mapping_is_frozen(self):
        cohort = pd.DataFrame({'case_id': [1, 2], 'subject_id': [10, 11], 'included': [True, True]})
        with self.assertRaises(AssertionError):
            self.mod.select_cohort(cohort, self.meta)

    def test_principal_summary_exposes_unrounded_estimates_and_distinct_denominators(self):
        flow = {'Policy cohort cases': 50, 'Policy cohort subjects': 48, 'Quality-eligible blocks': 20, 'Quality-eligible cases': 12, 'Quality-eligible subjects': 11, 'Low-reference blocks at adjustment': 10, 'Low-reference cases': 8, 'Low-reference subjects': 7, 'Normal-reference blocks': 10, 'Low-reference blocks with full 5-min support': 9, 'Blocks with confirmed reference recovery': 6}
        outputs = {'display': pd.DataFrame([{'scenario': 'Primary', 'interval_min': 5.0, 'category': 'normal_display', 'estimate': 41.935028, 'lower95': 37.05, 'upper95': 47.06, 'events': 10, 'cases': 8, 'subjects': 7}]), 'recovery': pd.DataFrame([{'scenario': 'Primary', 'interval_min': 5.0, 'estimate': 0.952125, 'lower95': 0.847, 'upper95': 1.048, 'median_min': 0.98, 'q1_min': 0.25, 'q3_min': 1.5, 'confirmed_recovery_events': 6, 'cases': 5, 'subjects': 4}])}
        summary = self.mod.principal_summary(outputs, flow)
        self.assertEqual(summary['eligible_adjustment_events'], 20)
        self.assertEqual(summary['low_reference_events'], 10)
        self.assertEqual(summary['confirmed_recovery_events'], 6)
        self.assertEqual(summary['display_by_interval_min']['5']['normal_display']['estimate_pct'], 41.935028)
        self.assertEqual(summary['recovery_by_interval_min']['5']['mean_persistence_min'], 0.952125)
        self.assertEqual(summary['recovery_subjects'], 4)

    def test_publication_formatter_keeps_zero_distinct_from_missing(self):
        self.assertEqual(self.mod.format_ci(0.0, 0.0, 0.0), '0.00 (0.00 to 0.00)')
        self.assertEqual(self.mod.format_ci(None, None, None), 'Not estimable')

    def test_environment_paths_are_resolved_before_frozen_import(self):
        env = dict(os.environ, IOH_PROJECT_ROOT=str(self.mod.ROOT), IOH_QC_OUTPUT_ROOT=str(self.mod.OUT / 'qa/treatment_portability_no_write'))
        code = 'import importlib.util, os; from pathlib import Path; s=importlib.util.spec_from_file_location("portable", os.environ["ADAPTER_PATH"]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m.ROOT==Path(os.environ["IOH_PROJECT_ROOT"]).resolve(); assert m.OUT==Path(os.environ["IOH_QC_OUTPUT_ROOT"]).resolve()'
        env['ADAPTER_PATH'] = str(MODULE)
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_source_path_takes_precedence_before_frozen_import(self):
        env = dict(os.environ, IOH_PROJECT_ROOT='/nonexistent/intentionally-invalid-test-source', ADAPTER_PATH=str(MODULE), EXPECTED_ROOT=str(self.mod.ROOT))
        code = 'import importlib.util, os; from pathlib import Path; s=importlib.util.spec_from_file_location("portable", os.environ["ADAPTER_PATH"]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m.ROOT==Path(os.environ["EXPECTED_ROOT"]).resolve()'
        result = subprocess.run([sys.executable, '-c', code, '--source-project', str(self.mod.ROOT)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_window_audit_detects_relevant_mask_not_just_equal_record_counts(self):
        base = {1: np.full(90, 60.0)}
        art = {1: base[1].copy()}
        art[1][39] = np.nan
        audit = self.mod.mask_window_audit(self.events([1], [400.0]), art, base)
        self.assertEqual(audit['events_with_masked_pre_reference_support'], 1)
        self.assertGreater(audit['event_phases_with_masked_active_sample'], 0)
        self.assertEqual(audit['events_with_masked_post_support'], 0)

    def test_window_audit_ignores_superseded_historical_samples(self):
        base = {1: np.full(90, 75.0)}
        art = {1: base[1].copy()}
        art[1][:5] = np.nan
        audit = self.mod.mask_window_audit(self.events([1], [400.0]), art, base)
        self.assertEqual(audit['events_with_masked_pre_reference_support'], 0)
        self.assertEqual(audit['event_phases_with_masked_active_sample'], 0)
        self.assertEqual(audit['events_with_masked_post_support'], 0)
        self.assertEqual(audit['post_phase_bins_with_masked_active_sample'], 0)
if __name__ == '__main__':
    unittest.main()
