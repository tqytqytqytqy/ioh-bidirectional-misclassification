import unittest
import numpy as np
import minute_analysis as m


class MinuteTests(unittest.TestCase):
    def test_minute_median_missing_and_terminal(self):
        x = np.array([60, 62, 64, 66, 68, np.nan, 1, 2, 3, 4, np.nan, np.nan, 90.])
        np.testing.assert_allclose(m.minute_median(x), [64, np.nan], equal_nan=True)

    def test_windows_all_and_adjacent_not_split(self):
        x = np.r_[np.ones(120), np.full(60, np.nan), np.ones(59)]
        selected = m.select_hours(x)
        self.assertEqual(selected.sum(), 120)
        self.assertEqual(m.runs(selected), [(0, 120)])

    def test_coherence_only_observed_triplets(self):
        v = np.array([70., 70., 70., 70.])
        s = np.array([100., 60., np.nan, 75.]); d = np.array([50., 50., 50., 70.])
        np.testing.assert_array_equal(m.coherent_mask(v, s, d), [True, False, True, False])

    def test_constant_runs_break_on_missing(self):
        v = np.array([70., 70., 70., np.nan, 70., 70., 80.])
        np.testing.assert_array_equal(m.constant_mask(v, 3), [True, True, True, False, False, False, False])

    def test_display_carryin_and_age_cap(self):
        v = np.array([60., 80., 80., 80., 80., np.nan, 80., 80., 80., 80., 70.])
        d, h = m.display(v, 5, 0)
        self.assertEqual(d[7], 60)
        self.assertEqual(h[7], 0)
        capped, _ = m.display(v, 5, 0, age_cap=True)
        self.assertTrue(np.isnan(capped[7]))

    def test_event_missing_not_bridged_and_boundary(self):
        v = np.r_[60., 60., np.nan, 60., 60., 60., 70.]
        a, b = m.events(v, 65, 3)
        np.testing.assert_array_equal(a, [3]); np.testing.assert_array_equal(b, [6])
        selected = np.array([False, True, True, True, True, True, True])
        a, b = m.events(np.where(selected, v, np.nan), 65, 3, selected, True)
        np.testing.assert_array_equal(a, [3])

    def test_fresh_inherited_never_partition(self):
        v = np.array([60., 70., 70., 60., 60., 60., 70., 60., 60., 60., 70.])
        d, h = m.display(v, 5, 0)
        a, b = m.events(v, 65, 3)
        count = m.event_components(d, h, a, b, 65)
        self.assertEqual(count[0], 2)
        self.assertEqual(sum(count[1:4]), 2)
        self.assertEqual(count[1], 1)
        self.assertEqual(count[2], 1)

    def test_deficit_identity_and_mask(self):
        r = np.array([60., 70., 50., np.nan]); d = np.array([70., 60., np.nan, 20.])
        c = m.frequency_components(r, d, 65, np.ones(4, bool))
        z = dict(zip(m.FCOLS, c))
        self.assertEqual(z['reference_auc'], 20)
        self.assertEqual(z['display_auc'], 5)
        self.assertEqual(z['hidden_auc'], 20)
        self.assertEqual(z['overdisplay_auc'], 5)
        self.assertEqual(z['hidden_unavailable_auc'], 15)
        self.assertEqual(z['display_auc']-z['reference_auc'], z['overdisplay_auc']-z['hidden_auc'])

    def test_five_minutes_consecutive_always_fresh(self):
        v = np.r_[70., 70., 60., 60., 60., 60., 60., 70.]
        a, b = m.events(v, 65, 5)
        for phase in range(5):
            d, h = m.display(v, 5, phase)
            c = m.event_components(d, h, a, b, 65)
            self.assertEqual(c[1], c[0])


if __name__ == '__main__': unittest.main()
