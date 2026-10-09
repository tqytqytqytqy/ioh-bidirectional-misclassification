from __future__ import annotations
import os
import json
import sys
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import numpy as np
import pandas as pd
from audit_core import run_length
from run_numeric_audit import DATA, OUT, locate
from run_waveform_audit import read_wave_prefix, bin_wave

def main():
    private = OUT / 'private_case_audit'
    cases = pd.read_csv(private / 'startup_numeric_case_audit.csv')
    panel = pd.read_parquet(DATA / 'artmap_10s.parquet')
    grids = {k: g.set_index('time_sec').art_map for k, g in panel.groupby('case_id')}
    old = pd.read_csv(private / 'waveform_10s_bin_metrics.csv')
    index = pd.read_csv(private / 'waveform_case_audit.csv').set_index('case_id')
    rows, bins = ([], [])
    for j, row in cases.iterrows():
        f = locate(row.waveform_tid)
        result = {'case_id': int(row.case_id), 'subject_id': int(row.subject_id), 'targeted_pressure_screen': bool(row.targeted_review), 'first_valid_rel_sec': float(row.first_valid_rel_sec)}
        if f is None:
            result['wave_status'] = 'NO_SOURCE_WAVEFORM'
            rows.append(result)
            continue
        try:
            if row.case_id in index.index:
                code = index.loc[row.case_id, 'review_code']
                wave = old[old.review_code.eq(code)].copy()
            else:
                t, v, dt = read_wave_prefix(f, row.start_abs_sec + row.first_valid_rel_sec + 600)
                wave = bin_wave(t, v, row.start_abs_sec, row.first_valid_rel_sec, dt)
                wave['numeric_map'] = wave.time_sec.map(grids[row.case_id])
            supported = wave.wave_coverage.ge(0.8)
            low = wave.numeric_map.lt(40) & wave.numeric_map.notna()
            nonpulse = low & supported & wave.wave_span_90pct.le(5)
            pulse = low & supported & wave.pulse_like
            result.update({'wave_status': 'READABLE', 'supported_bins': int(supported.sum()), 'low_numeric_bins': int(low.sum()), 'nonpulsatile_low_cumulative_sec': int(nonpulse.sum()) * 10, 'nonpulsatile_low_longest_run_sec': run_length(nonpulse) * 10, 'pulse_like_low_cumulative_sec': int(pulse.sum()) * 10, 'screened_nonpulsatile_low_reference_auc': float(((65 - wave.numeric_map[nonpulse]) / 6).sum())})
            wave = wave.drop(columns=['review_code'], errors='ignore')
            wave.insert(0, 'case_id', int(row.case_id))
            bins.append(wave)
        except Exception as e:
            result['wave_status'] = 'READ_ERROR'
            result['error_type'] = type(e).__name__
        rows.append(result)
        if (j + 1) % 200 == 0:
            pd.DataFrame(rows).to_csv(private / 'full_waveform_progress.csv', index=False)
            print(json.dumps({'all_waveform_cases_completed': j + 1, 'total': len(cases)}), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(private / 'full_waveform_case_audit.csv', index=False)
    pd.concat(bins).to_csv(private / 'full_waveform_10s_metrics.csv', index=False)
    readable = result[result.wave_status.eq('READABLE')]
    cumulative = readable.nonpulsatile_low_cumulative_sec.ge(60)
    continuous = readable.nonpulsatile_low_longest_run_sec.ge(60)
    summary = {'cohort_cases': len(result), 'wave_status': result.wave_status.value_counts().to_dict(), 'at_least60s_cumulative_nonpulsatile_low_cases': int(cumulative.sum()), 'at_least60s_continuous_nonpulsatile_low_cases': int(continuous.sum()), 'cumulative_flag_not_in_numeric_target_list_cases': int((cumulative & ~readable.targeted_pressure_screen).sum()), 'flagged_nonpulsatile_low_auc': float(readable.screened_nonpulsatile_low_reference_auc.sum()), 'flagged_nonpulsatile_low_auc_fraction_of_full_reference_auc': float(readable.screened_nonpulsatile_low_reference_auc.sum() / cases.total_reference_auc.sum()), 'clinical_adjudication': 'NOT_PERFORMED', 'primary_results_modified': False}
    (OUT / 'full_waveform_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
