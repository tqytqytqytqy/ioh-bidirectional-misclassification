import unittest
import numpy as np
from review_core import sustained_recovery, common_support, distinct_recoveries

class ReviewCoreTests(unittest.TestCase):
    def test_confirmation_at_end_of_full_normal_run(self):
        v=np.r_[np.repeat(60.,6),np.repeat(75.,50)]
        self.assertEqual(sustained_recovery(v,60,20),7)
        self.assertEqual(sustained_recovery(v,60,30),8)
        self.assertEqual(sustained_recovery(v,60,60),11)
    def test_missing_breaks_normal_run(self):
        v=np.repeat(75.,40);v[6]=np.nan
        self.assertEqual(sustained_recovery(v,60,30),9)
    def test_no_recovery_outside_search_window(self):
        v=np.r_[np.repeat(60.,35),np.repeat(75.,40)]
        self.assertIsNone(sustained_recovery(v,60,20))
    def test_incomplete_search_window_not_eligible(self):
        self.assertIsNone(sustained_recovery(np.repeat(75.,20),60,20))
    def test_common_support_and_unchanged_grid(self):
        v=np.arange(19,dtype=float)+50;v[7]=np.nan
        a,b=common_support(v)
        self.assertEqual(len(a),18);self.assertEqual(len(b),3)
        np.testing.assert_array_equal(np.isfinite(a),np.repeat(np.isfinite(b),6))
        self.assertTrue(np.isnan(a[6:12]).all())
        np.testing.assert_equal(a[:6],v[:6]);self.assertEqual(b[0],52.5)
    def test_deduplicate_case_and_confirmation_not_patient(self):
        rows=[dict(case_id=1,recovery_sec=50,adjustment_sec=20),dict(case_id=1,recovery_sec=50,adjustment_sec=10),dict(case_id=2,recovery_sec=50,adjustment_sec=10)]
        r=distinct_recoveries(rows)
        self.assertEqual(len(r),2);self.assertEqual(r[0]['adjustment_sec'],10)

if __name__=='__main__':unittest.main()
