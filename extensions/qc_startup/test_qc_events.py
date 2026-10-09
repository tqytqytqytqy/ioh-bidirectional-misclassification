"""Regression tests for the independently rerun frozen event definitions."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import pandas as pd

class QCEventTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('rerun_events.py')
        if not path.exists():
            return
        spec = importlib.util.spec_from_file_location('rerun_events', path)
        cls.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.api)

    def setUp(self):
        self.assertTrue(hasattr(self, 'api'), 'Event adapter has not been implemented')

    def analyze(self, values, thresholds=(65,), intervals=(5,), robust=False):
        values = np.asarray(values, float)
        return self.api.analyze_case(1, np.arange(len(values)) * 10 + 100, values, thresholds, intervals, robust)

    def test_sixty_second_event_does_not_include_shorter_run(self):
        f, e, d, r = self.analyze([60] * 5 + [70] + [60] * 6)
        self.assertTrue((f.episode_count == 1).all())
        self.assertEqual(r.low_duration_sec.tolist(), [60])
        self.assertTrue(np.allclose(f.true_hypotension_min, 11 / 6))
        self.assertEqual(len(e), 30)

    def test_nan_separates_events_even_with_gap_bridging(self):
        _, e, _, _ = self.analyze([60] * 3 + [np.nan] + [60] * 3, robust=True)
        self.assertEqual(e.episodes.sum(), 0)

    def test_bridging_counts_low_bins_not_elapsed_span(self):
        _, e, _, _ = self.analyze([60] * 3 + [70] * 2 + [60] * 3, robust=True)
        self.assertEqual(e[(e.gap_sec == 20) & (e.minimum_low_sec == 60)].episodes.sum(), 30)
        self.assertEqual(e[e.minimum_low_sec == 180].episodes.sum(), 0)
        self.assertEqual(e[e.gap_sec == 10].episodes.sum(), 0)

    def test_inherited_display_is_detected_without_fresh_low(self):
        f, e, _, _ = self.analyze([60, 70] + [60] * 6 + [70] * 28)
        row = e[e.offset_sec == 0].iloc[0]
        self.assertEqual([row.episodes, row.fresh, row.inherited_only, row.never_low], [1, 0, 1, 0])
        self.assertEqual(f[f.offset_sec == 0].episode_detected.iloc[0], 1)

    def test_masked_start_does_not_reset_sampling_phase(self):
        f, e, _, _ = self.analyze([np.nan] * 4 + [60] * 6 + [70] * 25)
        row = e[e.offset_sec == 0].iloc[0]
        self.assertEqual(row.never_low, 1)
        self.assertEqual(f[f.offset_sec == 0].display_unavailable_min.iloc[0], 26 / 6)
        self.assertTrue(np.allclose(f.anesthesia_hours, 35 / 360))

    def test_decomposition_conserves_auc_and_original_schema(self):
        f, _, _, _ = self.analyze([60] * 6 + [70] * 4 + [np.nan, 50, 60, 70])
        np.testing.assert_allclose(f.display_auc - f.true_auc, f.overdisplay_auc - f.hidden_auc)
        required = {'concordant_auc', 'hidden_auc_display_unavailable', 'episode_sensitivity', 'mean_detection_delay_min', 'missed_episodes', 'anesthesia_hours'}
        self.assertTrue(required.issubset(f.columns))
        np.testing.assert_allclose(f.display_valid_min + f.display_unavailable_min, 13 / 6)

    def test_irregular_or_compressed_grid_rejected(self):
        with self.assertRaises(ValueError):
            self.api.analyze_case(1, np.array([0, 10, 30]), np.array([60, 60, 60]), (65,), (5,), False)

    def test_panel_validation_rejects_replacement_or_time_shift(self):
        original = pd.DataFrame({'case_id': [1, 1, 1], 'time_sec': [0, 10, 20], 'art_map': [70.0, 60.0, 80.0]})
        changed = original.copy()
        changed.loc[1, 'art_map'] = 61
        with self.assertRaises(ValueError):
            self.api.validate_panel(changed, original)
        changed = original.iloc[1:].copy()
        with self.assertRaises(ValueError):
            self.api.validate_panel(changed, original)
        changed = original.copy()
        changed.loc[0, 'art_map'] = np.nan
        result = self.api.validate_panel(changed, original)
        self.assertEqual(result['additional_masked_bins'], 1)

    def test_subject_cluster_combines_repeated_cases_and_keeps_zero_event_subject(self):
        frame = pd.DataFrame({'case_id': [1, 2, 3], 'num': [1, 2, 0], 'den': [2, 2, 0]}).set_index('case_id')
        mapping = pd.DataFrame({'case_id': [1, 2, 3], 'subjectid': [10, 10, 20]})
        summed = self.api.subject_totals(frame, mapping, [1, 2, 3], ['num', 'den'])
        np.testing.assert_array_equal(summed.to_numpy(), [[3, 4], [0, 0]])

    def test_cohort_audit_rows_do_not_reinclude_excluded_cases(self):
        cohort = pd.DataFrame({'case_id': [1, 2, 3], 'included': [True, False, True]})
        self.assertTrue(hasattr(self.api, 'included_case_ids'), 'Inclusion adapter is missing')
        self.assertEqual(self.api.included_case_ids(cohort), {1, 3})

    def test_output_root_environment_override(self):
        target = Path(__file__).resolve().parents[1] / 'qa/events_test_output_root'
        env = dict(os.environ, IOH_PROJECT_ROOT=str(self.api.ROOT), IOH_QC_OUTPUT_ROOT=str(target))
        command = f"import importlib.util; from pathlib import Path; s=importlib.util.spec_from_file_location('events', {str(Path(__file__).with_name('rerun_events.py'))!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.OUT)"
        result = subprocess.run([sys.executable, '-c', command], env=env, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), str(target))

    def test_decomposition_population_ratios_and_original_hour_denominator(self):
        self.assertTrue(hasattr(self.api, 'summarize_decomposition'), 'Population decomposition summary is missing')
        frame = pd.DataFrame({'case_id': [1], 'true_auc': [10.0], 'display_auc': [8.0], 'hidden_auc': [5.0], 'overdisplay_auc': [3.0], 'concordant_auc': [5.0], 'normotensive_display_discordance_min': [1.0], 'hypotensive_display_discordance_min': [2.0], 'anesthesia_hours': [2.0], 'display_valid_min': [99.0], 'display_unavailable_min': [1.0]})
        mapping = pd.DataFrame({'case_id': [1], 'subjectid': [10]})
        row = self.api.summarize_decomposition(frame, mapping, [1], 65.0, 5.0)
        self.assertAlmostEqual(row['HDR'], 0.5)
        self.assertAlmostEqual(row['ODR'], 0.3)
        self.assertAlmostEqual(row['NetBias'], -0.2)
        self.assertAlmostEqual(row['NetBias_ci_low'], -0.2)
        self.assertAlmostEqual(row['absolute_state_discordance_min_per_hour'], 1.5)

    def test_event_summary_uses_phase_pairs_and_expected_counts(self):
        _, e, _, _ = self.analyze([60, 70] + [60] * 6 + [70] * 28)
        mapping = pd.DataFrame({'case_id': [1], 'subjectid': [10]})
        row = self.api.summarize_events(e, mapping, [1], 65, 5)
        self.assertEqual(row['reference_episodes'], 1)
        self.assertEqual(row['phase_episode_pairs'], 30)
        self.assertAlmostEqual(row['fresh_percent'], 20)
        self.assertAlmostEqual(row['inherited_only_percent'], 100 / 30)
        self.assertAlmostEqual(row['expected_detected_episodes'], 7 / 30)
        self.assertAlmostEqual(row['complete_miss_probability'], 23 / 30)
        self.assertAlmostEqual(row['complete_miss_ci_low'], 23 / 30)

    def test_all_nan_has_zero_events_and_zero_burden_without_fabricating_reference_time(self):
        f, e, _, r = self.analyze([np.nan] * 12)
        self.assertEqual(len(r), 0)
        self.assertEqual(e.episodes.sum(), 0)
        self.assertEqual(f.true_auc.sum(), 0)
        self.assertEqual((f.display_valid_min + f.display_unavailable_min).sum(), 0)
if __name__ == '__main__':
    unittest.main()
