import unittest
import numpy as np
import pandas as pd
import engine as e
from summarize import summarize_group,paired

class SummaryTests(unittest.TestCase):
    def test_repeated_records_then_equal_patients(self):
        rows=[]
        for j in range(10):
            for k,r,h in [(0,1,1),(1,9,0)]:
                z=np.zeros(len(e.COLS));z[0]=r;z[2]=h;z[10]=z[12]=60
                rows.append(dict(subject_id=f'a{j}',record_id=f'a{j}_{k}',**dict(zip(e.COLS,z))))
            z=np.zeros(len(e.COLS));z[0]=100;z[2]=50;z[10]=z[12]=60
            rows.append(dict(subject_id=f'b{j}',record_id=f'b{j}',**dict(zip(e.COLS,z))))
        out,_=summarize_group(pd.DataFrame(rows),['hidden_ratio'])
        a={r['estimand']:r for r in out}
        self.assertAlmostEqual(a['patient_equal']['estimate'],.3)
        self.assertAlmostEqual(a['pooled']['estimate'],51/110)
        self.assertEqual(a['patient_equal']['n_patients'],20)
        self.assertEqual(a['patient_equal']['n_records'],30)
    def test_paired_equal_requires_both_ratios_defined(self):
        rows=[]
        for j in range(20):
            for res in ['10s','1min']:
                z=np.zeros(len(e.COLS));z[0]=1;z[10]=z[12]=60
                if j>=10 or res=='1min':z[e.F]=1
                if j>=10 and res=='1min':z[e.F+1]=1
                rows.append(dict(source='Test',scenario='paired_bridge',threshold=65,interval_min=5,
                    resolution=res,subject_id=str(j),record_id=str(j),**dict(zip(e.COLS,z))))
        out=paired(pd.DataFrame(rows))
        a=out[(out.metric=='fresh_probability')&(out.estimand=='patient_equal_common_defined')].iloc[0]
        self.assertEqual(a.n_defined_common,10);self.assertAlmostEqual(a.minute_minus_native,1.)
        b=out[(out.metric=='fresh_probability')&(out.estimand=='pooled')].iloc[0]
        self.assertAlmostEqual(b.minute_minus_native,.5)

if __name__=='__main__':unittest.main()
