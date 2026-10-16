import unittest
import numpy as np
try:
    import engine as e
except ImportError:
    e=None

class EngineTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(e,'shared engine must be implemented')
    def test_minute_requires_five_values_and_keeps_positions(self):
        v=np.r_[np.arange(6)+60,[60,61,62,63,np.nan,np.nan],[60,61,62,63,64,np.nan]]
        a=e.minute_median(v)
        np.testing.assert_allclose(a,[62.5,np.nan,62],equal_nan=True)
    def test_hour_selection_never_compresses_gaps(self):
        v=np.r_[np.ones(360)*80,np.full(360,np.nan),np.ones(360)*80]
        m=e.select_hours(v,10)
        self.assertEqual(m.sum(),720);self.assertFalse(m[400]);self.assertTrue(m[800])
    def test_paired_hours_are_intersection(self):
        v=np.ones(720)*80;v[np.arange(13)*6]=np.nan;v[np.arange(13)*6+1]=np.nan
        m=e.paired_hours(v)
        self.assertEqual(len(m),720);self.assertFalse(m[:360].any());self.assertTrue(m[360:].all())
    def test_display_carries_prewindow_history_without_phase_reset(self):
        v=np.array([50,80,80,80,80,90.]);d,h=e.display(v,5,0)
        self.assertEqual(d[3],50);self.assertEqual(h[3],0)
    def test_missing_scheduled_value_does_not_refresh(self):
        v=np.array([50,80,80,80,80,np.nan,80.]);d,h=e.display(v,5,0)
        self.assertEqual(d[6],50);self.assertEqual(h[6],0)
    def test_low_run_missing_break_and_prior_gap_unknown(self):
        v=np.array([50,80,80,50,50,np.nan,80,50,50.])
        a,b,nadir,gap=e.episodes(v,65,2)
        np.testing.assert_array_equal(a,[3,7]);np.testing.assert_array_equal(b,[5,9])
        self.assertEqual(gap[0],2);self.assertTrue(np.isnan(gap[1]))
    def test_nadir_and_duration_categories(self):
        v=np.array([80,60,50,60,80.]);a,b,n,g=e.episodes(v,65,3)
        self.assertEqual(n[0],50);self.assertEqual(b[0]-a[0],3)
    def test_analytic_fresh_probability(self):
        v=np.r_[80,np.full(12,50.),np.full(30,80.)]
        _,parts,_=e.analyze(v,np.ones(len(v),bool),10,65,5,60,features=True)
        self.assertAlmostEqual(parts[e.F+1]/parts[e.F],12/30)
    def test_event_partition_identity_and_auc_identity(self):
        v=np.r_[np.full(4,80.),np.full(3,50.),np.full(4,80.),np.full(2,60.),np.full(5,80.)]
        _,c,_=e.analyze(v,np.ones(len(v),bool),60,65,5,180,features=True)
        self.assertAlmostEqual(c[1]-c[0],c[3]-c[2])
        self.assertAlmostEqual(c[e.F],sum(c[e.F+1:e.F+4]))
    def test_zero_burden_and_zero_events_not_filled_with_zero(self):
        _,c,_=e.analyze(np.ones(60)*80,np.ones(60,bool),60,65,5,180)
        d=e.derive(c)
        self.assertTrue(np.isnan(d['hidden_ratio']));self.assertTrue(np.isnan(d['fresh_probability']))
    def test_reference_missing_is_not_display_unavailable_time(self):
        v=np.array([80,80,np.nan,80,80,80.])
        _,c,_=e.analyze(v,np.ones(len(v),bool),60,65,1,180)
        self.assertEqual(e.derive(c)['unavailable_fraction'],0)
    def test_boundary_complete_requires_normal_both_sides(self):
        v=np.array([50,50,80,50,50,np.nan,80,50,50,80.])
        a,b,_,_=e.episodes(v,65,2,complete=True)
        np.testing.assert_array_equal(a,[7]);np.testing.assert_array_equal(b,[9])
    def test_patient_equal_aggregates_repeat_records_first(self):
        a=np.zeros((3,e.F+6));a[:,0]=[1,9,10];a[:,2]=[1,0,5]
        ids=np.array(['a','a','b']);patients,components=e.patient_sum(ids,a)
        self.assertEqual(len(patients),2)
        ratios=e.derive(components)['hidden_ratio']
        self.assertAlmostEqual(ratios.mean(),.3)
        self.assertNotAlmostEqual(ratios.mean(),e.derive(a)['hidden_ratio'].mean())

if __name__=='__main__':unittest.main()
