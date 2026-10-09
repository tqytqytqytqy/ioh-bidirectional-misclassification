from __future__ import annotations
import os
import hashlib
import json
import sys
from pathlib import Path
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import numpy as np
import pandas as pd
from audit_core import median_grid, numeric_screen
ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
OUT = Path(os.environ.get('IOH_STARTUP_AUDIT_ROOT', str(ROOT / '启动段质量审计_20261007'))).expanduser().resolve()
DATA = ROOT / 'analysis_v10_subject_phase_release/outputs/intermediate'
RAW = Path(os.environ['IOH_VITALDB_RAW_ROOT']).expanduser().resolve()
TRACK_ROOTS = [Path(p).expanduser().resolve() for p in os.environ['IOH_VITALDB_TRACK_DIRS'].split(os.pathsep) if p]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def locate(tid):
    if pd.isna(tid):
        return None
    for root in TRACK_ROOTS:
        p = root / f'{tid}.csv.gz'
        if p.is_file():
            return p
    return None

def read_numeric(tid):
    p = locate(tid)
    if p is None:
        return (np.array([]), np.array([]), None)
    df = pd.read_csv(p)
    return tuple((pd.to_numeric(df.iloc[:, i], errors='coerce').to_numpy(float) for i in range(2))) + (p,)

def main():
    private = OUT / 'private_case_audit'
    private.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(DATA / 'vitaldb_manifest.parquet').set_index('case_id')
    panel = pd.read_parquet(DATA / 'artmap_10s.parquet')
    trks = pd.read_csv(RAW / 'trks.csv')
    lookup = trks.set_index(['caseid', 'tname']).tid.to_dict()
    rows, sources = ([], [])
    for j, (case, df) in enumerate(panel.groupby('case_id', sort=True), 1):
        row = manifest.loc[case]
        t = df.time_sec.to_numpy(float)
        v = df.art_map.to_numpy(float)
        assert np.array_equal(t, np.arange(len(t)) * 10)
        start = max(0.0, float(row.anestart))
        times, vals, source = read_numeric(row.art_tid)
        rebuilt = median_grid(times, vals, start, len(v), 20, 180)
        matches = bool(np.allclose(v, rebuilt, equal_nan=True, atol=1e-10, rtol=0))
        first = int(np.flatnonzero(np.isfinite(v))[0])
        pressure = []
        for name in ['Solar8000/ART_SBP', 'Solar8000/ART_DBP']:
            rt, rv, _ = read_numeric(lookup.get((case, name)))
            pressure.append(median_grid(rt, rv, start, len(v), 0, 300))
        sbp, dbp = pressure
        flags = numeric_screen(v, sbp, dbp, first)
        deficit = np.where(np.isfinite(v), np.maximum(65 - v, 0), 0) / 6
        item = {'case_id': int(case), 'subject_id': int(row.subjectid), 'source_map_tid': row.art_tid, 'start_abs_sec': start, 'first_valid_rel_sec': first * 10, 'raw_cache_match': matches, 'first10min_valid_bins': int(np.isfinite(v[:60]).sum()), 'total_reference_auc': float(deficit.sum()), 'anaesthesia_first10_reference_auc': float(deficit[:60].sum()), 'reference_start_first10_auc': float(deficit[first:first + 60].sum()), 'waveform_tid': lookup.get((case, 'SNUADC/ART')), **flags}
        item['waveform_local'] = locate(item['waveform_tid']) is not None
        item['targeted_review'] = any((flags[f] for f in ['low_plateau_flag', 'low_pulse_pressure_flag', 'inverted_sbp_dbp_flag', 'map_bounds_flag']))
        item['raw_nonmonotonic_time'] = bool(np.any(np.diff(times[np.isfinite(times)]) < 0))
        rows.append(item)
        if source:
            sources.append({'case_id': int(case), 'role': 'ART_MBP', 'path': str(source), 'sha256': digest(source)})
        if j % 200 == 0:
            print(json.dumps({'numeric_cases_completed': j, 'total': 2435}), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(private / 'startup_numeric_case_audit.csv', index=False)
    pd.DataFrame(sources).to_csv(private / 'raw_map_source_hashes.csv', index=False)
    result.loc[result.targeted_review].to_csv(private / 'targeted_review_cases.csv', index=False)
    counts = {f: int(result[f].sum()) for f in ['raw_cache_match', 'raw_nonmonotonic_time', 'low_plateau_flag', 'large_jump_flag', 'low_pulse_pressure_flag', 'inverted_sbp_dbp_flag', 'map_bounds_flag', 'targeted_review', 'waveform_local']}
    summary = {'cases': len(result), 'subjects': int(result.subject_id.nunique()), 'counts': counts, 'first_valid_rel_sec_quantiles': result.first_valid_rel_sec.quantile([0, 0.25, 0.5, 0.75, 0.95, 1]).to_dict(), 'no_valid_map_in_first10min_cases': int(result.first10min_valid_bins.eq(0).sum()), 'anaesthesia_first10_auc_fraction': float(result.anaesthesia_first10_reference_auc.sum() / result.total_reference_auc.sum()), 'reference_start_first10_auc_fraction': float(result.reference_start_first10_auc.sum() / result.total_reference_auc.sum()), 'raw_source_root': str(RAW), 'source_panel_sha256': digest(DATA / 'artmap_10s.parquet'), 'clinical_adjudication': 'NOT_PERFORMED', 'primary_results_modified': False}
    (OUT / 'numeric_audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
