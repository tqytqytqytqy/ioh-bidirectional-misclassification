from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import numpy as np
import pandas as pd
OUT = Path(os.environ.get('IOH_QC_OUTPUT_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', str(OUT.parent))).expanduser().resolve()
SOURCE = ROOT / 'analysis_v10_subject_phase_release/outputs/intermediate'
AUDIT = ROOT / '启动段质量审计_20261007/private_case_audit'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def runs(mask):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))

def eligible_mask(frame, min_bins):
    mask = (frame.numeric_map.lt(40) & frame.wave_coverage.ge(0.8) & frame.wave_span_90pct.le(5)).to_numpy()
    selected = np.zeros(len(frame), bool)
    for a, b in runs(mask):
        if b - a >= min_bins:
            selected[a:b] = True
    return selected

def main():
    private = OUT / 'data_restricted'
    private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 448)
    protected = {str(p): sha(p) for p in [SOURCE / 'artmap_10s.parquet', SOURCE / 'vitaldb_manifest.parquet', AUDIT / 'full_waveform_10s_metrics.csv', AUDIT / 'clinical_review_list_87_cases.csv']}
    panel = pd.read_parquet(SOURCE / 'artmap_10s.parquet')
    meta = pd.read_parquet(SOURCE / 'vitaldb_manifest.parquet').set_index('case_id')
    waves = pd.read_csv(AUDIT / 'full_waveform_10s_metrics.csv')
    review = pd.read_csv(AUDIT / 'clinical_review_list_87_cases.csv')
    codes = review.set_index('case_id').review_code.to_dict()
    numeric = pd.read_csv(AUDIT / 'startup_numeric_case_audit.csv').set_index('case_id')
    masks = {p: {} for p in ['qc60', 'qc_all']}
    intervals = []
    for cid, g in waves.groupby('case_id', sort=True):
        g = g.sort_values('time_sec').reset_index(drop=True)
        assert len(g) == 60 and np.allclose(np.diff(g.time_sec), 10)
        for policy, n in [('qc60', 6), ('qc_all', 1)]:
            select = eligible_mask(g, n)
            masks[policy][cid] = g.loc[select, 'time_sec'].to_numpy(float)
            for a, b in runs(select):
                f = g.iloc[a:b]
                intervals.append({'policy': policy, 'case_id': int(cid), 'review_code': codes.get(cid, 'Outside_87_short_flags'), 'start_sec': float(f.time_sec.iloc[0]), 'end_sec_exclusive': float(f.time_sec.iloc[-1] + 10), 'duration_sec': int((b - a) * 10), 'map_median': float(f.numeric_map.median()), 'wave_coverage_min': float(f.wave_coverage.min()), 'wave_span_90pct_max': float(f.wave_span_90pct.max()), 'reference_auc_removed': float(((65 - f.numeric_map) / 6).sum()), 'technical_action': 'MASK_IN_RULE_BASED_SENSITIVITY', 'clinical_adjudication': 'PENDING_CLINICIAN'})
    pd.DataFrame(intervals).to_csv(private / 'candidate_mask_intervals.csv', index=False)
    summary = []
    for policy in ['original', 'qc60', 'qc_all', 'startup10']:
        frames = []
        cohorts = []
        for cid, g in panel.groupby('case_id', sort=True):
            g = g.sort_values('time_sec').copy()
            orig = g.art_map.to_numpy(float)
            masked = np.zeros(len(g), bool)
            if policy in masks:
                masked = g.time_sec.isin(masks[policy].get(cid, [])).to_numpy()
            elif policy == 'startup10':
                first = int(np.flatnonzero(np.isfinite(orig))[0])
                masked[:first + 60] = True
            values = orig.copy()
            values[masked] = np.nan
            coverage = np.isfinite(values).mean()
            included = policy in ['original', 'startup10'] or coverage >= 0.8
            cohorts.append({'policy': policy, 'case_id': int(cid), 'subject_id': int(meta.loc[cid, 'subjectid']), 'original_valid_coverage': float(np.isfinite(orig).mean()), 'valid_coverage': float(coverage), 'newly_masked_bins': int((masked & np.isfinite(orig)).sum()), 'masked_reference_auc': float(np.maximum(65 - orig[masked & np.isfinite(orig)], 0).sum() / 6), 'included': bool(included), 'exclusion_reason': '' if included else 'valid_coverage_below_80_percent'})
            if included:
                g['art_map'] = values
                frames.append(g)
        coh = pd.DataFrame(cohorts)
        coh.to_csv(private / f'cohort_{policy}.csv', index=False)
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(private / f'panel_{policy}.parquet', index=False)
        keep = coh[coh.included]
        summary.append({'policy': policy, 'cases': len(keep), 'subjects': keep.subject_id.nunique(), 'affected_cases_before_coverage': int(coh.newly_masked_bins.gt(0).sum()), 'excluded_coverage_cases': int((~coh.included).sum()), 'masked_minutes_before_coverage': float(coh.newly_masked_bins.sum() / 6), 'masked_reference_auc_before_coverage': float(coh.masked_reference_auc.sum()), 'reference_auc_retained': float(np.maximum(65 - out.art_map, 0).sum() / 6), 'retained_valid_reference_hours': float(out.art_map.notna().sum() / 360), 'panel_sha256': sha(private / f'panel_{policy}.parquet')})
    pd.DataFrame(summary).to_csv(OUT / 'analysis/aggregate/cohort_qc_summary.csv', index=False)
    ints = pd.DataFrame(intervals)
    for label in ['qc60', 'qc_all']:
        v = ints[ints.policy.eq(label)]
        detail = v.groupby('case_id').apply(lambda g: '; '.join((f'{r.start_sec:g}-{r.end_sec_exclusive:g} s' for r in g.itertuples())), include_groups=False)
        review[f'{label}_candidate_intervals'] = review.case_id.map(detail).fillna('None')
    review['technical_recommendation'] = np.select([review.qc60_candidate_intervals.ne('None'), review.qc_all_candidate_intervals.ne('None'), review.pulse_like_low_cumulative_sec.ge(60)], ['Review sustained low nonpulsatile intervals; included in qc60 sensitivity', 'Review shorter or interrupted low nonpulsatile intervals; included in qc_all sensitivity', 'Pulsatile low pressure present; retain unless other technical evidence'], default='Pressure screen without sustained flat waveform corroboration; retain pending review')
    review['clinical_review_status'] = 'PENDING_CLINICIAN'
    review.to_csv(private / 'adjudication_list_87_cases.csv', index=False)
    for p, h in protected.items():
        assert sha(Path(p)) == h
    (OUT / 'qa/panels_verification.json').write_text(json.dumps({'protected_sources_unchanged': True, 'source_hashes': protected, 'protocol_sha256': sha(OUT / '方案.md'), 'policies': summary, 'clinical_adjudication': 'PENDING_CLINICIAN', 'status': 'PASS'}, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
