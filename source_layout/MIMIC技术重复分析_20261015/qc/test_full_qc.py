import unittest
import numpy as np
from full_qc_core import FlatTracker, WaveBins

class FullQCTests(unittest.TestCase):
    def test_continuous_flatline_across_chunks(self):
        f=FlatTracker(); self.assertEqual(f.consume(np.full(30,70.),0,10),[])
        spans=f.consume(np.full(31,70.),3,10)
        self.assertEqual(len(spans),1); self.assertAlmostEqual(spans[0][0],0)
        self.assertAlmostEqual(spans[0][1],6.1)
    def test_gap_does_not_join_flatline(self):
        f=FlatTracker(); f.consume(np.full(30,70.),0,10)
        self.assertEqual(f.consume(np.full(30,70.),4,10),[])
    def test_nan_breaks_flatline(self):
        f=FlatTracker(); a=np.full(81,70.); a[40]=np.nan
        self.assertEqual(f.consume(a,0,10),[])
    def test_full_bins_and_missingness(self):
        w=WaveBins(3); a=70+20*np.sin(np.arange(200)*2*np.pi/10)
        w.consume(a,0,10); r=w.finish()
        np.testing.assert_array_equal(r['pass'],[True,True,False])
        self.assertAlmostEqual(r['finite_seconds'][0],10)
    def test_flatline_masks_every_intersecting_bin(self):
        w=WaveBins(3); a=70+20*np.sin(np.arange(300)*2*np.pi/10)
        a[90:150]=70
        w.consume(a[:120],0,10); w.consume(a[120:],12,10)
        r=w.finish(); self.assertTrue(r['flatline'][0]);self.assertTrue(r['flatline'][1]);self.assertFalse(r['flatline'][2])
    def test_fractional_segment_start_alignment(self):
        w=WaveBins(3); a=70+20*np.sin(np.arange(100)*2*np.pi/10)
        w.consume(a,9.5,10); r=w.finish()
        np.testing.assert_allclose(r['observed_seconds'],[.5,9.5,0],atol=1e-8)
    def test_missing_and_out_of_range_not_passed(self):
        w=WaveBins(3);a=np.r_[np.full(100,np.nan),np.full(100,300.),70+20*np.sin(np.arange(100)*2*np.pi/10)]
        w.consume(a,0,10);r=w.finish();np.testing.assert_array_equal(r['pass'],[False,False,True])
    def test_low_pulsatility_separate_sensitivity(self):
        w=WaveBins(1);w.consume(70+np.sin(np.arange(100)*2*np.pi/10),0,10);r=w.finish()
        self.assertTrue(r['pass'][0]);self.assertFalse(r['strict'][0])
    def test_arbitrary_chunking_equals_single_pass(self):
        fs=124.945;t=np.arange(20000)/fs;a=70+20*np.sin(t*2*np.pi)
        a[580:1390]=70;a[8000:8100]=np.nan;a[14000:14800]=80
        one=WaveBins(17);one.consume(a,0,fs);x=one.finish()
        many=WaveBins(17)
        points=[0,100,600,900,1200,6000,8050,14100,14500,20000]
        for left,right in zip(points[:-1],points[1:]):many.consume(a[left:right],left/fs,fs)
        y=many.finish()
        for k in ['observed_seconds','finite_seconds','plausible_seconds','mean','range']:
            np.testing.assert_allclose(x[k],y[k],rtol=1e-12,atol=1e-8,equal_nan=True)
        for k in ['pass','strict','flatline']:np.testing.assert_array_equal(x[k],y[k])

if __name__=='__main__':unittest.main()
