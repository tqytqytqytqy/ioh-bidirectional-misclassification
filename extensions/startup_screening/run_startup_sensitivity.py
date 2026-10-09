from __future__ import annotations
import os
import json
import sys
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import numpy as np
import pandas as pd
from run_numeric_audit import DATA, OUT

def components(values, mask_start):
    v = np.asarray(values, float).copy()
    v[:mask_start] = np.nan
    n = len(v)
    idx = np.arange(n)
    finite = np.isfinite(v)
    r = np.where(finite, np.maximum(65 - v, 0), 0)
    true = r.sum() / 6
    scores = []
    for offset in range(30):
        sample = finite & ((idx - offset) % 30 == 0) & (idx >= offset)
        last = np.maximum.accumulate(np.where(sample, idx, -1))
        display = np.where(last >= 0, v[np.maximum(last, 0)], np.nan)
        visible = finite & np.isfinite(display)
        d = np.where(visible, np.maximum(65 - display, 0), 0)
        scores.append([np.maximum(r - d, 0).sum() / 6, np.where(finite, np.maximum(d - r, 0), 0).sum() / 6, (visible & (r > 0) & (d == 0)).sum() / 6, (visible & (r == 0) & (d > 0)).sum() / 6])
    avg = np.mean(scores, axis=0)
    return [true, *avg, finite.sum() / 6]

def main():
    panel = pd.read_parquet(DATA / 'artmap_10s.parquet')
    manifest = pd.read_parquet(DATA / 'vitaldb_manifest.parquet').set_index('case_id')
    original = pd.read_parquet(DATA / 'frequency_decomp_case_offset.parquet')
    expected = original[original.interval_min.eq(5) & original.threshold.eq(65)].groupby('case_id')
    expected = expected[['true_auc', 'hidden_auc', 'overdisplay_auc', 'normotensive_display_min', 'hypotensive_display_discordance_min']].mean()
    columns = ['true_auc', 'hidden_auc', 'overdisplay_auc', 'low_reference_normal_display_min', 'normal_reference_low_display_min', 'valid_reference_min']
    rows = []
    for j, (case, group) in enumerate(panel.groupby('case_id', sort=True), 1):
        v = group.art_map.to_numpy(float)
        first = int(np.flatnonzero(np.isfinite(v))[0])
        for label, start in [('Frozen primary reconstruction', 0), ('Mask anaesthesia first 10 min', 60), ('Mask first 10 min of reference startup', first + 60)]:
            scores = components(v, start)
            if start == 0:
                assert np.allclose(scores[:5], expected.loc[case].to_numpy(float), rtol=1e-09, atol=1e-08), case
            rows.append({'case_id': int(case), 'subject_id': int(manifest.loc[case, 'subjectid']), 'policy': label, 'masked_bins': start, **dict(zip(columns, scores))})
        if j % 500 == 0:
            print(json.dumps({'sensitivity_cases_completed': j, 'total': 2435}), flush=True)
    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / 'private_case_audit/startup_sensitivity_case_components.csv', index=False)
    metrics = ['HDR65', 'ODR65', 'NetBias_percent', 'low_reference_normal_display_min_per_valid_hour', 'normal_reference_low_display_min_per_valid_hour', 'reference_auc']
    rng = np.random.default_rng(20261007)
    summary = []
    for policy, df in cases.groupby('policy', sort=False):
        cluster = df.groupby('subject_id')[columns].sum().to_numpy(float)
        weights = rng.multinomial(len(cluster), np.repeat(1 / len(cluster), len(cluster)), size=1000)
        total = cluster.sum(axis=0)
        draws = weights @ cluster

        def transform(a):
            a = np.atleast_2d(a)
            return np.column_stack([a[:, 1] / a[:, 0], a[:, 2] / a[:, 0], (a[:, 2] - a[:, 1]) / a[:, 0] * 100, a[:, 3] / a[:, 5] * 60, a[:, 4] / a[:, 5] * 60, a[:, 0]])
        point, boot = (transform(total)[0], transform(draws))
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
        for k, metric in enumerate(metrics):
            summary.append({'policy': policy, 'metric': metric, 'estimate': point[k], 'ci_low': ci[0, k], 'ci_high': ci[1, k], 'cases': len(df), 'subjects': len(cluster), 'valid_reference_min': total[5]})
    out = pd.DataFrame(summary)
    out.to_csv(OUT / 'startup_sensitivity_aggregate.csv', index=False)
    note = {'primary_reconstruction_matches_all_case_components': True, 'bootstrap_reps': 1000, 'resampling_unit': 'subject', 'seed': 20261007, 'policies': 'Original 30 sampling phases retained. Masked values are removed from reference and scheduled samples before display construction. No phase reset at trimming boundary.', 'denominator_note': 'Discordance rates use retained valid-reference hours, not full anaesthesia hours; do not substitute into existing main Table 2.', 'analysis_status': 'Additional audit sensitivities, not replacement primary results', 'primary_files_modified': False}
    (OUT / 'startup_sensitivity_methods.json').write_text(json.dumps(note, indent=2) + '\n')
    print(out[['policy', 'metric', 'estimate', 'ci_low', 'ci_high']].to_json(orient='records', indent=2), flush=True)
if __name__ == '__main__':
    main()
