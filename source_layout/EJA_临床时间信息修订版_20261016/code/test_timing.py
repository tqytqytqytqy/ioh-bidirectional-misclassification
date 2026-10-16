import unittest
import numpy as np
from timing import low_runs, display_indices, fresh_states, recovery_states

class TimingTests(unittest.TestCase):
    def test_confirmed_and_boundary(self):
        v=np.array([60.]*6+[70.]+[60.]*6+[np.nan]+[60.]*6)
        self.assertEqual([(a,b,c) for a,b,c in low_runs(v)],[(0,6,False),(7,13,True),(14,20,False)])
    def test_no_lookahead_and_all_phases(self):
        v=np.array([70.]*3+[60.]*6+[70.]*20)
        z=fresh_states(v,3,9,30,60)
        self.assertAlmostEqual(z['fresh'],6/30)
        self.assertAlmostEqual(z['still_low_without_low_display'],24/30)
        self.assertAlmostEqual(fresh_states(v,3,9,30,120)['recovered_without_fresh'],24/30)
    def test_deadline_and_last_low_bin(self):
        v=np.array([70.]+[60.]*12+[70.]*3)
        self.assertAlmostEqual(fresh_states(v,1,13,30,120)['fresh'],12/30)
        self.assertEqual(fresh_states(v,1,13,6,60)['fresh'],1)
    def test_inherited(self):
        v=np.array([60.,70.]+[60.]*12+[70.])
        self.assertGreater(fresh_states(v,2,14,30,60)['still_low_inherited'],0)
    def test_gap_and_end_unknown(self):
        for v in [np.array([70.]+[60.]*6),np.array([70.]+[60.]*6+[np.nan]+[70.]*20)]:
            z=fresh_states(v,1,7,30,120)
            self.assertAlmostEqual(z['unknown'],.8)
    def test_missing_scheduled_carries_and_no_reset(self):
        v=np.array([60.]+[70.]*5+[np.nan]+[70.]*8)
        h=display_indices(v,6,0)
        self.assertEqual(h[7],0)
        self.assertEqual(h[12],12)
    def test_recovery_cleared_and_persistent(self):
        v=np.array([60.]*6+[70.]*40)
        z=recovery_states(v,7,30,60)
        self.assertAlmostEqual(z['persistent_low'],6/30)
        self.assertEqual(recovery_states(v,7,6,60)['display_cleared'],1)
    def test_recurrence_priority(self):
        v=np.array([60.]*6+[70.]*2+[60.]*20)
        z=recovery_states(v,7,30,60)
        self.assertAlmostEqual(z['reference_recurred'],6/30)
    def test_recovery_early_end(self):
        v=np.array([60.]*6+[70.]*2)
        z=recovery_states(v,7,30,120)
        self.assertGreater(z['unknown'],0)
        self.assertEqual(z['persistent_low'],0)
    def test_partitions_and_discrete_baseline(self):
        for length in [6,7,12,30,31,60]:
            for start in [1,3,29,30,31]:
                v=np.array([70.]*start+[60.]*length+[70.]*40)
                for n in [6,18,30]:
                    for h in [60,120]:
                        z=fresh_states(v,start,start+length,n,h)
                        self.assertAlmostEqual(sum(z[k] for k in ['fresh','recovered_without_fresh','still_low_inherited','still_low_without_low_display','unknown']),1)
                        self.assertAlmostEqual(z['fresh'],min(length,h//10,n)/n)

if __name__=='__main__': unittest.main()
