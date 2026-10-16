import unittest
import numpy as np

try:
    import audit_core as core
except ModuleNotFoundError:
    core = None


class AuditCoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, "audit_core is not implemented")

    def test_systemic_aliases_only(self):
        for name in ["ABPm [mmHg]", "ARTm [mmHg]", "FAPm [mmHg]", "AoM [mmHg]"]:
            self.assertIsNotNone(core.numeric_alias(name))
        for name in ["PAPm [mmHg]", "CVP [mmHg]", "NBPm [mmHg]", "Pulse (ABP) [bpm]"]:
            self.assertIsNone(core.numeric_alias(name))
        self.assertEqual(core.wave_alias("Ao"), "Ao")
        self.assertIsNone(core.wave_alias("PAP"))

    def test_numbered_and_case_insensitive_pressure_aliases(self):
        for name in ["ART1", "ART2", "art", "AO", "ao", "abp2", "FAP1"]:
            self.assertIsNotNone(core.wave_alias(name))
        self.assertEqual(core.numeric_alias("ART1m [mmHg]"), ("ART1", "m"))
        self.assertEqual(core.numeric_alias("AOM [mmHg]"), ("Ao", "m"))
        self.assertIsNone(core.wave_alias("PAP2"))

    def test_actual_medians_no_interpolation_and_counter_clock(self):
        b, d = core.bin_medians([0, 5120, 30720, -1024, 61440], [60, 80, 90, 50, 100], 1024, 60)
        np.testing.assert_allclose(b, [70, np.nan, np.nan, 90, np.nan, np.nan], equal_nan=True)
        self.assertEqual(d["negative_time_samples"], 1)
        self.assertEqual(d["outside_header_samples"], 1)

    def test_median_then_validity_and_inclusive_bounds(self):
        b, _ = core.bin_medians([0, 1, 2, 10, 20], [-10, 40, 60, 20, 180], 1, 30)
        np.testing.assert_array_equal(core.map_valid(b), [True, True, True])
        self.assertEqual(b[0], 40)
        np.testing.assert_array_equal(core.map_valid([19.9, 20, 180, 180.1, np.nan]), [False, True, True, False, False])

    def test_threshold_no_synthetic_density(self):
        b = np.full(360, np.nan)
        b[::6] = 75
        r = core.coverage(b, 3600)
        self.assertFalse(r["whole_eligible"])
        self.assertFalse(r["window_eligible"])
        self.assertEqual(r["valid_bins"], 60)

    def test_exact_80_percent_and_short_record(self):
        b = np.r_[np.full(288, 75.), np.full(72, np.nan)]
        self.assertTrue(core.coverage(b, 3600)["whole_eligible"])
        self.assertFalse(core.coverage(b[:-1], 3599)["window_eligible"])
        self.assertTrue(core.coverage(b, 3600)["window_eligible"])

    def test_best_window_not_chosen_on_low_map(self):
        a = np.r_[np.full(360, 90.), np.full(20, np.nan), np.full(360, 50.)]
        b = np.where(a == 90, 50, np.where(a == 50, 90, a))
        ra, rb = core.coverage(a, len(a) * 10), core.coverage(b, len(b) * 10)
        self.assertEqual(ra["best_hour_start_bin"], 0)
        self.assertEqual(ra["best_hour_start_bin"], rb["best_hour_start_bin"])

    def test_primary_channel_density_only(self):
        self.assertEqual(core.choose_family({"ABP": [80, np.nan], "ART": [50, 50]}), "ART")
        self.assertEqual(core.choose_family({"ABP": [80, 80], "ART": [50, 50]}), "ABP")

    def test_coherence_missing_is_not_valid(self):
        m = [70, 70, 70, 70, 70]
        s = [120, 60, 220, 120, np.nan]
        d = [50, 50, 50, 115, 50]
        np.testing.assert_array_equal(core.coherent(m, s, d), [True, False, False, False, False])

    def test_missing_breaks_low_runs(self):
        x = [60.] * 5 + [np.nan] + [60.] * 6 + [70.]
        r = core.low_runs(x)
        self.assertEqual(r["runs_ge60s"], 1)
        self.assertEqual(r["longest_run_seconds"], 60)

    def test_age_and_interval_overlap(self):
        self.assertEqual(core.age_at_year(17, 2150, 2151), 18)
        self.assertEqual(core.overlap_seconds(0, 20, 10, 30), 10)
        self.assertEqual(core.overlap_seconds(0, 10, 10, 30), 0)

    def test_wave_flatline_and_missing(self):
        q = core.wave_quality(np.full(1000, 80.), 100)
        self.assertTrue(q["flatline_ge5s"])
        q = core.wave_quality(np.full(1000, np.nan), 100)
        self.assertEqual(q["finite_samples"], 0)
        self.assertFalse(q["automated_plausible_block"])

    def test_timestamps_duplicates_and_order(self):
        r = core.interval_summary([2, 0, 2, 5], 1)
        self.assertEqual(r["backward_steps_file_order"], 1)
        self.assertEqual(r["duplicate_timestamps"], 1)
        self.assertEqual(r["positive_interval_counts_seconds"], {"2.0": 1, "3.0": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
