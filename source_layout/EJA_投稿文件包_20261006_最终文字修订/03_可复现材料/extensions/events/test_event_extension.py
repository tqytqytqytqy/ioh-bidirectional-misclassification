import unittest
import numpy as np
from event_extension import held_display, reference_events, classify_events


class EventTests(unittest.TestCase):
    def test_nonzero_phase_and_missing_sample(self):
        v = np.array([60., 70., np.nan, 80., 90., 55., 75.])
        display, idx = held_display(v, 3, 2)
        np.testing.assert_array_equal(idx, [-1, -1, -1, -1, -1, 5, 5])
        self.assertTrue(np.isnan(display[:5]).all())
        np.testing.assert_array_equal(display[5:], [55, 55])

    def test_missing_does_not_clear_previous_display(self):
        v = np.array([60., 70., 70., np.nan, 70., 70., 80.])
        d, idx = held_display(v, 3, 0)
        np.testing.assert_array_equal(idx, [0, 0, 0, 0, 0, 0, 6])
        np.testing.assert_array_equal(d, [60, 60, 60, 60, 60, 60, 80])

    def test_bridge_only_observed_normal_gap(self):
        v = np.array([60., 60., 70., 60., 60., np.nan, 60., 60.])
        ev = reference_events(v, 65, 1, 4)
        self.assertEqual(ev, [(0, 5, 4)])
        self.assertEqual(reference_events(v, 65, 0, 4), [])

    def test_minimum_duration_counts_low_not_span(self):
        v = np.array([60., 70., 70., 60.])
        self.assertEqual(reference_events(v, 65, 2, 3), [])
        self.assertEqual(reference_events(v, 65, 2, 2), [(0, 4, 2)])

    def test_exclusive_fresh_inherited_never_categories(self):
        v = np.array([60., 70., 60., 60., 70., 60., 60., 70.])
        events = [(2, 4, 2), (5, 7, 2)]
        d, idx = held_display(v, 4, 0)
        cats = classify_events(v, d, idx, events, 65)
        np.testing.assert_array_equal(cats['category'], [1, 2])
        np.testing.assert_array_equal(cats['carry_in'], [True, False])
        d, idx = held_display(v, 3, 0)
        cats = classify_events(v, d, idx, events, 65)
        np.testing.assert_array_equal(cats['category'], [0, 0])
        np.testing.assert_array_equal(cats['first_inherited'], [True, True])


if __name__ == '__main__':
    unittest.main()
