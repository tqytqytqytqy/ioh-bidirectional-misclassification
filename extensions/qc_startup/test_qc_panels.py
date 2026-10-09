import os
import numpy as np
import pandas as pd
from build_qc_panels import runs, eligible_mask

def frame(n=10):
    return pd.DataFrame({'numeric_map': [30.0] * n, 'wave_coverage': [1.0] * n, 'wave_span_90pct': [2.0] * n})

def test_continuity_and_gap():
    f = frame(12)
    f.loc[6, 'wave_coverage'] = 0
    assert eligible_mask(f, 6).sum() == 6
    assert eligible_mask(f, 1).sum() == 11

def test_pulsatile_or_unsupported_low_is_not_masked():
    f = frame()
    f.wave_span_90pct = 15
    assert not eligible_mask(f, 6).any()
    f.wave_span_90pct = 2
    f.wave_coverage = np.nan
    assert not eligible_mask(f, 1).any()

def test_threshold_and_missing():
    f = frame(6)
    f.numeric_map = 40
    assert not eligible_mask(f, 6).any()
    f.numeric_map = 30
    f.loc[1, 'numeric_map'] = np.nan
    assert not eligible_mask(f, 6).any()
    assert runs([True, True, False, True]) == [(0, 2), (3, 4)]
