import unittest
import numpy as np
import pandas as pd
try:
    from qc_onset import bin_gate, first_run, numeric_grid
except ImportError:
    bin_gate=first_run=numeric_grid=None


class GateTests(unittest.TestCase):
    def good(self, n=12):
        return pd.DataFrame(dict(numeric_map=np.full(n,28.),wave_coverage=np.ones(n),
            wave_mean=np.full(n,29.),wave_p05=np.full(n,14.),wave_p95=np.full(n,45.),
            wave_span_90pct=np.full(n,31.),peak_count=np.full(n,12),sbp=np.full(n,45.),dbp=np.full(n,14.)))

    def test_implementation_available(self):
        self.assertTrue(callable(bin_gate),'Objective gate is not implemented yet')

    def test_true_low_pressure_can_qualify(self):
        if bin_gate is None:self.skipTest('absent')
        self.assertTrue(bin_gate(self.good()).all())

    def test_high_pressure_is_not_required(self):
        if bin_gate is None:self.skipTest('absent')
        f=self.good();f.loc[0,['numeric_map','wave_mean']]=[20.,21.]
        self.assertTrue(bin_gate(f)[0])

    def test_flat_wave_rejected(self):
        if bin_gate is None:self.skipTest('absent')
        f=self.good();f['wave_span_90pct']=2;f['peak_count']=0
        self.assertFalse(bin_gate(f).any())

    def test_disagreement_rejected(self):
        if bin_gate is None:self.skipTest('absent')
        f=self.good();f['wave_mean']=60
        self.assertFalse(bin_gate(f).any())

    def test_missing_pair_allowed_but_inverted_pair_rejected(self):
        if bin_gate is None:self.skipTest('absent')
        f=self.good();f.loc[0,'sbp']=np.nan;f.loc[1,'sbp']=10.
        x=bin_gate(f);self.assertTrue(x[0]);self.assertFalse(x[1])

    def test_incomplete_coverage_rejected(self):
        if bin_gate is None:self.skipTest('absent')
        f=self.good();f['wave_coverage']=.85
        self.assertFalse(bin_gate(f).any());self.assertTrue(bin_gate(f,relaxed=True).all())

    def test_nonconsecutive_runs_do_not_join(self):
        if first_run is None:self.skipTest('absent')
        self.assertIsNone(first_run([1,1,1,0,1,1,1],6))
        self.assertEqual(first_run([0,1,1,1,1,1,1,0],6),1)

    def test_exact_duration_at_end_accepted(self):
        if first_run is None:self.skipTest('absent')
        self.assertEqual(first_run([0,0,1,1,1,1,1,1],6),2)

    def test_median_grid_preserves_time_and_range(self):
        if numeric_grid is None:self.skipTest('absent')
        out=numeric_grid([100,101,112,132],[30,40,50,500],100,4,0,300)
        np.testing.assert_allclose(out,[35,50,np.nan,np.nan],equal_nan=True)


if __name__=='__main__':unittest.main()
