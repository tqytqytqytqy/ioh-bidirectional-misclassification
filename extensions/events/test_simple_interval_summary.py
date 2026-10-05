import unittest

import numpy as np

try:
    from simple_interval_summary import displayed_low_minutes, pooled_twa
except ModuleNotFoundError:
    displayed_low_minutes = pooled_twa = None


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(displayed_low_minutes, 'Interval summary helper is not implemented')

    def test_display_duration_excludes_unavailable_and_missing_reference(self):
        # Five valid reference bins; two have a low display. A sixth missing bin is excluded.
        result = displayed_low_minutes(3 / 6, 1 / 6, 1 / 6, 1 / 6, 5 / 6)
        self.assertAlmostEqual(float(result), 2 / 6)

    def test_no_display_is_not_counted_as_hypotensive_display(self):
        self.assertEqual(float(displayed_low_minutes(2, 2, 0, 0, 5)), 0)

    def test_duration_reconstruction_vector(self):
        result = displayed_low_minutes([3, 2], [1, 0], [1, 1], [2, 0], [5, 5])
        np.testing.assert_allclose(result, [3, 1])

    def test_impossible_duration_raises(self):
        with self.assertRaises(ValueError):
            displayed_low_minutes(1, 0, 3, 0, 5)

    def test_pooled_twa_is_ratio_of_sums_not_mean_of_ratios(self):
        self.assertAlmostEqual(pooled_twa([10, 1000], [10, 100]), 1010 / 110)
        self.assertNotAlmostEqual(pooled_twa([10, 1000], [10, 100]), 5.5)

    def test_zero_observation_time_raises(self):
        with self.assertRaises(ValueError):
            pooled_twa([0], [0])


if __name__ == '__main__':
    unittest.main()
