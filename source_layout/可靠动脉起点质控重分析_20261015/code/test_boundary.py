import unittest
import numpy as np
from boundary_sensitivity import complete_events

class BoundaryTests(unittest.TestCase):
    def test_cut_or_missing_boundaries_fail(self):
        v=np.array([50.,50.,80.,50.,50.,np.nan,50.,50.,80.,50.,50.,80.])
        candidates=[(0,2,2),(3,5,2),(6,8,2),(9,11,2)]
        self.assertEqual(complete_events(v,candidates),[(9,11,2)])

if __name__=='__main__':unittest.main()
