import unittest
import numpy as np
import treatment_core as tc


class CoreTests(unittest.TestCase):
    def test_module_has_completed_bin_classifier(self):
        self.assertTrue(hasattr(tc, 'event_state'))

    def test_grouping_uses_anchor_not_chaining(self):
        self.assertEqual(tc.block_starts([0, 50, 100], 60), [0, 2])

    def test_boundary_included(self):
        self.assertEqual(tc.block_starts([0, 60, 61], 60), [0, 2])

    def test_event_does_not_use_unfinished_bin(self):
        v = np.r_[np.full(30, 75.), 50., 50.]
        s = tc.event_state(v, 305., 30, 0)
        self.assertEqual(s['reference'], 75.)
        self.assertEqual(s['category'], 'reference_normal')

    def test_missing_reference_is_not_low(self):
        v = np.full(40, 60.); v[30] = np.nan
        self.assertIsNone(tc.event_state(v, 315., 30, 0))

    def test_unavailable_display_is_not_normal(self):
        v = np.full(40, 60.)
        s = tc.event_state(v, 305., 30, 29)
        self.assertEqual(s['category'], 'fresh_low')
        v[29] = np.nan
        s = tc.event_state(v, 315., 30, 29)
        self.assertEqual(s['category'], 'unavailable')

    def test_inherited_low_is_not_fresh(self):
        v = np.full(50, 75.); v[0] = 60.; v[30:] = 60.
        s = tc.event_state(v, 315., 30, 0)
        self.assertEqual(s['category'], 'fresh_low')
        s = tc.event_state(v, 305., 30, 0)
        self.assertEqual(s['category'], 'reference_normal')
        v[1] = 60.
        s = tc.event_state(v, 315., 30, 1)
        self.assertEqual(s['category'], 'inherited_low')

    def test_missing_breaks_current_low_run(self):
        v = np.full(50, 60.); v[29] = np.nan
        s = tc.event_state(v, 325., 30, 0)
        self.assertEqual(s['category'], 'fresh_low')
        self.assertEqual(s['low_run_start'], 30)

    def test_recovery_requires_two_normal_bins(self):
        v = np.full(80, 60.); v[31] = 75.; v[33:] = 75.
        r = tc.recovery_summary(v, 315., 30, 0)
        self.assertEqual(r['recovery_index'], 34)

    def test_recovery_stops_at_missing_reference(self):
        v = np.full(80, 60.); v[32:] = 75.; v[36] = np.nan
        r = tc.recovery_summary(v, 315., 30, 0)
        self.assertEqual(r['recovery_index'], 33)
        self.assertEqual(r['stop_reason'], 'missing_reference')
        self.assertAlmostEqual(r['persistence_min'], .5)

    def test_first_positive_is_not_start_and_gap_is_not_start(self):
        t = np.array([0., 1., 2., 20., 21., 22., 23., 24., 25., 26.])
        v = np.array([1., 1., 0., 2., 2., 2., 2., 2., 2., 2.])
        self.assertEqual(tc.transitions(t, v, 5), [])

    def test_sustained_increase(self):
        t = np.arange(12.)
        v = np.r_[np.zeros(2), np.full(10, 3.)]
        e = tc.transitions(t, v, 5)
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]['kind'], 'observed_zero_to_positive')

    def test_unstable_increase_excluded(self):
        t = np.arange(10.)
        v = np.r_[0., 2., np.zeros(8)]
        self.assertEqual(tc.transitions(t, v, 5), [])
        self.assertEqual(len(tc.transitions(t, v, 0)), 1)


if __name__ == '__main__':
    unittest.main()
