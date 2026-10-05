import unittest
import numpy as np
import pandas as pd
from audit_core import (classify_legacy, clean_track, nearest_distance,
                        pair_support, rate_transitions, supported_duration, triplet_updates)


class AuditTests(unittest.TestCase):
    def test_pair_count_uses_verified_order_not_float_membership(self):
        from audit_core import paired_timer_count
        classified=pd.DataFrame({"time_sec":[0.,0.1+0.2],"reason":["initial_record","timer_same_map"]})
        pairs=pd.DataFrame({"display_time_sec":[0.,0.3],"paired":[False,True]})
        self.assertEqual(paired_timer_count(classified,pairs),1)
        pairs.loc[1,"display_time_sec"]=1.
        with self.assertRaises(ValueError):
            paired_timer_count(classified,pairs)

    def test_reconciliation_ignores_only_exact_duplicate_records(self):
        from audit_core import reconcile_records
        source=pd.DataFrame({"time_sec":[0.,2.],"value":[70.,71.]})
        frozen=pd.DataFrame({"time_sec":[0.,2.,2.],"value":[70.,71.,71.]})
        self.assertTrue(reconcile_records(source,frozen))
        frozen.loc[2,"value"]=72.
        self.assertFalse(reconcile_records(source,frozen))

    def test_timer_is_not_value_change(self):
        f = classify_legacy(np.arange(0, 301, 2), np.full(151, 70.))
        self.assertEqual(f.reason.tolist(), ["initial_record", "timer_same_map", "timer_same_map"])

    def test_map_changes_reset_timer(self):
        f = classify_legacy([0, 2, 100, 120, 220], [70, 70, 69, 69, 69])
        self.assertEqual(f.time_sec.tolist(), [0, 100, 220])

    def test_sub_mm_change_uses_legacy_threshold(self):
        f = classify_legacy([0, 2, 4], [70, 70.5, 71])
        self.assertEqual(f.time_sec.tolist(), [0, 4])

    def test_sd_can_change_without_map(self):
        m = pd.DataFrame({"time_sec": [0.,2.,4.], "value": [70.,70.,70.]})
        s = m.assign(value=[100.,101.,101.])
        d = m.assign(value=[50.,50.,50.])
        j, _ = triplet_updates(m,s,d)
        self.assertEqual(int(j.sd_only_change.sum()),1)

    def test_no_carry_across_large_alignment_gap(self):
        m = pd.DataFrame({"time_sec": [20.], "value": [70.]})
        s = pd.DataFrame({"time_sec": [0.], "value": [100.]})
        j, q = triplet_updates(m,s,s)
        self.assertEqual(q["complete_rows"],0)

    def test_actual_start_requires_observed_zero(self):
        f = pd.DataFrame({"time_sec": [0.,1.,2.,3.], "value": [5.,5.,0.,10.]})
        r = rate_transitions(f)
        self.assertEqual(r.time_sec.tolist(),[3.])

    def test_missing_gap_not_new_treatment(self):
        f = pd.DataFrame({"time_sec": [0.,30.], "value": [0.,5.]})
        self.assertTrue(rate_transitions(f).empty)

    def test_window_inclusive_and_fraction(self):
        t = np.arange(0,71,10)
        self.assertTrue(pair_support([30],t,np.ones(8))[0])
        v=np.ones(8);v[:2]=np.nan
        self.assertFalse(pair_support([30],t,v)[0])

    def test_duplicate_conflict_excluded_not_overwritten(self):
        source=pd.DataFrame({"t": [0,2,2,4], "v": [1,2,3,4]})
        original=source.copy()
        f,q=clean_track(source,0,4)
        self.assertEqual(f.time_sec.tolist(),[0,4])
        self.assertEqual(q["conflicting_timestamps"],1)
        pd.testing.assert_frame_equal(source,original)

    def test_distance_and_missing(self):
        self.assertEqual(nearest_distance([3,10],[2,5]).tolist(),[1,5])
        self.assertTrue(np.isinf(nearest_distance([1],[])[0]))

    def test_coverage_does_not_fill_long_gap(self):
        self.assertEqual(supported_duration([0,1,100],101),7.)


if __name__ == "__main__":
    unittest.main()
