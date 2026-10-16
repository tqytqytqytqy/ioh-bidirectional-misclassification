"""Synthetic numerical tests only; no human or clinical adjudication."""
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_writebytecode = True
import numpy as np
import pandas as pd
import independent_check as check


class IndependentTests(unittest.TestCase):
    def test_duration_discards_last_partial_ten_second_bin(self):
        result = check.minute_reference(np.arange(12.) + 60, duration_seconds=119)
        np.testing.assert_equal(result, [62.5])

    def test_exact_duration_retains_complete_minute(self):
        np.testing.assert_equal(check.minute_reference(np.arange(12.) + 60, 120), [62.5, 68.5])

    def test_duration_requires_adequate_original_grid(self):
        with self.assertRaises(ValueError):
            check.minute_reference(np.ones(5) * 70, 60)

    def test_five_bins_and_range_before_median(self):
        data = [20, 180, 65, 65, 65, 181, 60, 60, 60, 60, 19, np.nan]
        np.testing.assert_equal(check.minute_reference(data, 120), [65, np.nan])

    def test_complete_nonoverlapping_hours_and_48_valid(self):
        x = np.ones(179) * 70
        x[:12] = np.nan
        x[60:73] = np.nan
        mask = check.hour_selection(x)
        self.assertEqual(mask.sum(), 60)
        self.assertTrue(mask[:60].all())
        self.assertFalse(mask[60:].any())

    def test_missing_breaks_events_and_threshold_is_strict(self):
        x = np.array([64, 64, np.nan, 64, 64, 65, 64, 64, 64, 64, 64.])
        self.assertEqual(check.low_episodes(x, np.ones(11, bool), 65, 3), [(6, 11)])
        self.assertEqual(check.low_episodes(x, np.ones(11, bool), 65, 5), [(6, 11)])

    def test_adjacent_selected_hours_do_not_split_event(self):
        x = np.ones(120) * 70
        x[58:63] = 60
        self.assertEqual(check.low_episodes(x, check.hour_selection(x), 65, 5), [(58, 63)])

    def test_excluded_hour_cannot_connect_events(self):
        x = np.ones(180) * 70
        x[58:63] = 60
        mask = np.ones(180, bool)
        mask[60:120] = False
        self.assertEqual(check.low_episodes(x, mask, 65, 3), [])

    def test_display_carries_across_missing_scheduled_reading(self):
        d, updates = check.held_display(np.array([60, 70, 70, 70, 70, np.nan, 70.]), 5, 0)
        np.testing.assert_equal(d, [60] * 7)
        np.testing.assert_equal(updates, [0] * 7)

    def test_display_phase_does_not_initialize_early(self):
        d, updates = check.held_display(np.array([60, 70, 80, 90.]), 5, 3)
        np.testing.assert_equal(d, [np.nan, np.nan, np.nan, 90])
        np.testing.assert_equal(updates, [-1, -1, -1, 3])

    def test_inherited_display_not_fresh_at_selected_boundary(self):
        x = np.ones(120) * 70
        x[59:63] = 60
        x[64] = np.nan
        selected = np.zeros(120, bool)
        selected[60:] = True
        c = check.phase_components(x, selected, 65, 5, 4)
        self.assertEqual(c[check.COLS.index('event3_episodes')], 1)
        self.assertEqual(c[check.COLS.index('event3_fresh')], 0)
        self.assertEqual(c[check.COLS.index('event3_inherited_only')], 1)

    def test_hand_calculated_auc_and_event_partition(self):
        x = np.array([60, 70, 50, np.nan, 70, 60, 60, 60.])
        c = dict(zip(check.COLS, check.phase_components(x, np.ones(8, bool), 65, 5, 0)))
        expected = dict(reference_auc=35, display_auc=35, hidden_auc=10, overdisplay_auc=10,
                        hidden_unavailable_auc=0, reference_low_min=5, display_low_min=7,
                        low_reference_normal_display_min=0, normal_reference_low_display_min=2,
                        low_reference_unavailable_min=0, valid_reference_min=7,
                        display_available_min=7, selected_minutes=8,
                        event3_episodes=1, event3_fresh=1, event3_inherited_only=0,
                        event3_never_low=0, event3_visible=1, event3_delay_sum_min=0,
                        event5_episodes=0)
        for key, value in expected.items():
            self.assertEqual(c[key], value, key)

    def test_unavailable_auc_is_hidden_not_normal_display(self):
        c = dict(zip(check.COLS, check.phase_components(np.array([60, 60, 60, 70.]), np.ones(4, bool), 65, 5, 4)))
        self.assertEqual(c['hidden_auc'], 15)
        self.assertEqual(c['hidden_unavailable_auc'], 15)
        self.assertEqual(c['low_reference_unavailable_min'], 3)
        self.assertEqual(c['low_reference_normal_display_min'], 0)
        self.assertEqual(c['event3_never_low'], 1)

    def test_visible_delay_is_first_low_display_inside_episode(self):
        x = np.array([70, 60, 60, 60, 60, 60, 70.])
        c = dict(zip(check.COLS, check.phase_components(x, np.ones(7, bool), 65, 5, 4)))
        self.assertEqual(c['event5_delay_sum_min'], 3)
        self.assertEqual(c['event5_fresh'], 1)

    def test_all_integer_phases_are_averaged_components_not_ratios(self):
        x = np.array([60, 70, 70, 70, 70, 70.])
        c = dict(zip(check.COLS, check.case_components(x, np.ones(6, bool), 65, 3)))
        self.assertEqual(c['reference_auc'], 5)
        self.assertAlmostEqual(c['display_auc'], 5)
        self.assertAlmostEqual(c['hidden_auc'], 10 / 3)
        self.assertAlmostEqual(c['overdisplay_auc'], 10 / 3)

    def test_interval_one_has_zero_hidden_and_overdisplay(self):
        x = np.array([50, 70, np.nan, 60, 20, 180.])
        c = dict(zip(check.COLS, check.case_components(x, np.ones(6, bool), 65, 1)))
        self.assertEqual(c['hidden_auc'], 0)
        self.assertEqual(c['overdisplay_auc'], 0)

    def test_all_12_groups_with_all_phase_invariants(self):
        rng = np.random.default_rng(17)
        x = rng.choice([20, 54, 59, 64, 65, 70, 180, np.nan], 185)
        selected = check.hour_selection(x)
        groups = [check.case_components(x, selected, t, k) for t in (65, 60, 55) for k in (1, 3, 5, 10)]
        self.assertEqual(len(groups), 12)
        self.assertTrue(np.all(np.asarray(groups) >= 0))

    def test_derived_zero_denominator_is_undefined(self):
        m = check.metrics(np.zeros(len(check.COLS)))
        self.assertTrue(np.isnan(m['hidden_ratio']))
        self.assertTrue(np.isnan(m['event3_mean_display_delay_min']))

    def test_negative_component_rejected(self):
        c = np.zeros(len(check.COLS)); c[0] = -1
        with self.assertRaises(AssertionError):
            check.assert_components(c)

    def test_partition_violation_rejected(self):
        c = np.zeros(len(check.COLS)); c[check.COLS.index('event3_episodes')] = 1
        with self.assertRaises(AssertionError):
            check.assert_components(c)

    def test_numeric_comparison_detects_error_and_accepts_tolerance(self):
        audit = check.NumericAudit()
        audit.add(np.array([1., np.nan]), np.array([1 + 1e-8, np.nan]))
        self.assertEqual(audit.failures, 0)
        audit.add(np.array([1.]), np.array([1.001]))
        self.assertEqual(audit.failures, 1)

    def test_subject_bootstrap_sums_repeated_admissions_before_sampling(self):
        rows = []
        for subject, case, burden in [(2, 1, 10), (2, 2, 20), (9, 3, 70)]:
            c = np.zeros(len(check.COLS))
            c[0] = burden; c[1] = burden; c[10:13] = 60
            rows.append(dict(source='Synthetic', scenario='primary', threshold=65, interval_min=5,
                             subject_id=subject, case_id=case, **dict(zip(check.COLS, c))))
        report = check.primary_summary(pd.DataFrame(rows))
        row = report.iloc[0]
        self.assertEqual(row.n_subjects, 2)
        self.assertEqual(row.n_records, 3)
        self.assertEqual(row.reference_auc, 100)
        self.assertEqual(row.reference_auc_ci_low, 60)
        self.assertEqual(row.reference_auc_ci_high, 140)
        self.assertEqual(row.reference_twa, 100 / 180)
        again = check.primary_summary(pd.DataFrame(rows).iloc[::-1])
        pd.testing.assert_frame_equal(report, again)

    def test_sicdb_npz_alignment_and_range_filter(self):
        b = io.BytesIO()
        np.savez(b, case_id=[1, 2], subject_id=[11, 11], age_bin=[25, 20], index=[0, 3, 5],
                 map=[19., 20, 181, 70, 60], sbp=[90]*5, dbp=[50]*5)
        b.seek(0)
        with np.load(b, allow_pickle=False) as data:
            records = list(check.npz_records(data))
        np.testing.assert_equal(records[0]['map'], [np.nan, 20, np.nan])
        self.assertEqual(records[1]['age_bin'], 20)
        self.assertEqual(records[1]['subject_id'], 11)

    def test_sicdb_npz_bad_index_rejected(self):
        data = dict(case_id=np.array([1]), subject_id=np.array([1]), age_bin=np.array([25]),
                    index=np.array([0, 3]), map=np.ones(2))
        with self.assertRaises(ValueError):
            list(check.npz_records(data))

    def test_age_cutoff_and_qualified_window_gate(self):
        self.assertFalse(check.eligible(dict(age_bin=24.9, map=np.ones(60)*70)))
        self.assertFalse(check.eligible(dict(age_bin=25, map=np.ones(59)*70)))
        self.assertTrue(check.eligible(dict(age_bin=25, map=np.ones(60)*70)))


if __name__ == '__main__':
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    check.write_report(Path(__file__).resolve().parents[1] / 'qa/independent_tests.json', {
        'status': 'PASS' if result.wasSuccessful() else 'FAIL',
        'test_inputs': 'synthetic_only', 'human_adjudication': False,
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'elapsed_seconds': time.monotonic()-start,
    })
    sys.exit(not result.wasSuccessful())
