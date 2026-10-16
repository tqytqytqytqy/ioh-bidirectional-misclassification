import unittest
import numpy as np
try:
    from rerun import phase_components, trimmed_values, components_to_metrics
except ImportError:
    phase_components=trimmed_values=components_to_metrics=None

class RerunTests(unittest.TestCase):
    def test_implementation_available(self):
        self.assertTrue(callable(phase_components),'Rerun estimator not implemented yet')

    def test_trim_masks_without_compressing(self):
        if trimmed_values is None:self.skipTest('absent')
        a=np.arange(40.,60.);b=trimmed_values(a,50)
        self.assertEqual(len(a),len(b));self.assertTrue(np.isnan(b[:5]).all())
        np.testing.assert_equal(b[5:],a[5:])

    def test_known_two_minute_event_is_missed_in_three_of_five_phases(self):
        if phase_components is None:self.skipTest('absent')
        v=np.full(60,90.);v[10:12]=50.
        c=phase_components(v,np.ones(60,bool),65,5,1,1)
        self.assertEqual(c['episodes'],1)
        self.assertAlmostEqual(c['never_low'],.6)
        self.assertAlmostEqual(c['fresh'],.4)

    def test_unavailable_is_not_normal_display(self):
        if phase_components is None:self.skipTest('absent')
        v=np.full(20,90.);v[1:3]=50.
        c=phase_components(v,np.ones(20,bool),65,5,1,1)
        self.assertGreater(c['reference_low_unavailable_min'],0)
        self.assertGreater(c['H_unavailable'],0)

    def test_primary_reference_unchanged_across_initializations(self):
        if phase_components is None:self.skipTest('absent')
        v=np.full(60,90.);v[0]=40
        a=phase_components(v,np.ones(60,bool),65,5,1,1)
        b=phase_components(v,np.ones(60,bool),65,5,1,1,True,True)
        self.assertEqual(a['R'],b['R']);self.assertGreater(b['D'],a['D'])
        self.assertAlmostEqual(b['D']-b['R'],b['O']-b['H'])

    def test_missing_breaks_low_event(self):
        if phase_components is None:self.skipTest('absent')
        v=np.full(60,90.);v[10:16]=50.;v[12]=np.nan
        c=phase_components(v,np.ones(60,bool),65,5,1,3)
        self.assertEqual(c['episodes'],1)

if __name__=='__main__':unittest.main()
