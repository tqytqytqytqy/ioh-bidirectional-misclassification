from __future__ import annotations
import os
import hashlib
import json
import sys
from pathlib import Path
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from audit_core import decode_waveform_time
from run_numeric_audit import DATA, OUT, RAW, locate, read_numeric

def read_wave_prefix(path, end_sec):
    header = pd.read_csv(path, nrows=2)
    first_time = float(header.iloc[0, 0])
    interval = float(header.iloc[1, 0]) - first_time
    count = int(np.ceil((end_sec - first_time) / interval)) + 1
    arrays, offset = ([], 0)
    for chunk in pd.read_csv(path, chunksize=200000):
        values = pd.to_numeric(chunk.iloc[:, 1], errors='coerce').to_numpy(float)
        take = min(len(values), count - offset)
        if take > 0:
            arrays.append(values[:take])
        offset += take
        if offset >= count:
            break
    v = np.concatenate(arrays) if arrays else np.array([])
    return (decode_waveform_time(len(v), interval, start=first_time), v, interval)

def bin_wave(times, values, start, first_sec, sample_interval):
    rows = []
    fs = 1 / sample_interval
    for rel in np.arange(first_sec, first_sec + 600, 10):
        i = int(np.searchsorted(times, start + rel))
        j = int(np.searchsorted(times, start + rel + 10))
        x = values[i:j]
        finite = np.isfinite(x)
        frac = finite.sum() / (10 * fs)
        if finite.any():
            q5, q95 = np.percentile(x[finite], [5, 95])
            avg = float(np.mean(x[finite]))
            std = float(np.std(x[finite]))
            y = np.nan_to_num(x, nan=float(np.median(x[finite])))
            peaks, _ = find_peaks(y, distance=int(0.3 * fs), prominence=5)
            pulse_like = bool(frac >= 0.8 and q95 - q5 >= 8 and (4 <= len(peaks) <= 40))
        else:
            q5 = q95 = avg = std = np.nan
            peaks = []
            pulse_like = False
        rows.append({'time_sec': rel, 'wave_coverage': frac, 'wave_mean': avg, 'wave_p05': q5, 'wave_p95': q95, 'wave_sd': std, 'wave_span_90pct': q95 - q5, 'pulse_like': pulse_like, 'peak_count': len(peaks)})
    return pd.DataFrame(rows)

def make_plot(code, group, raw, wave, times, values, output):
    first = float(raw.first_valid_rel_sec)
    start = float(raw.start_abs_sec)
    panel = pd.read_parquet(DATA / 'artmap_10s.parquet')
    numeric = panel[panel.case_id.eq(raw.case_id)]
    keep = (numeric.time_sec >= max(0, first - 60)) & (numeric.time_sec < first + 600)
    num = numeric[keep]
    idx = wave.wave_span_90pct.fillna(np.inf).idxmin()
    detail_rel = float(wave.loc[idx, 'time_sec'])
    dm = (times >= start + detail_rel) & (times < start + detail_rel + 5)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), gridspec_kw={'height_ratios': [1.6, 1]})
    x = wave.time_sec.to_numpy(float) / 60
    axes[0].fill_between(x, wave.wave_p05.to_numpy(float), wave.wave_p95.to_numpy(float), color='#8eb8c6', alpha=0.5, label='Waveform 5th-95th percentiles')
    axes[0].plot(x, wave.wave_mean, color='#216477', lw=1.3, label='Waveform 10 s mean')
    axes[0].plot(num.time_sec / 60, num.art_map, color='#b84855', marker='.', ms=3, lw=1, label='Monitor MAP 10 s median')
    axes[0].axhline(65, color='#555555', ls='--', lw=0.8)
    axes[0].set_ylabel('Pressure (mmHg)')
    axes[0].set_xlabel('Minutes from analytic anaesthesia start')
    axes[0].set_title(f'{code} | {group}', fontsize=12)
    axes[0].legend(fontsize=8, loc='best')
    axes[1].plot(times[dm] - start - detail_rel, values[dm], color='#216477', lw=0.7)
    axes[1].set_title(f'5 s waveform excerpt at {detail_rel / 60:.2f} min', fontsize=10)
    axes[1].set_xlabel('Seconds within excerpt')
    axes[1].set_ylabel('Pressure (mmHg)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dedede', lw=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

def main():
    private = OUT / 'private_case_audit'
    cases = pd.read_csv(private / 'startup_numeric_case_audit.csv')
    target = cases[cases.targeted_review].copy()
    target['review_group'] = 'Targeted pressure screen'
    available = cases[~cases.targeted_review & cases.waveform_tid.notna()]
    control = available[~available.large_jump_flag].sample(n=min(20, (~available.large_jump_flag).sum()), random_state=20261007).copy()
    control['review_group'] = 'Unflagged control'
    jump = available[available.large_jump_flag].sample(n=min(20, available.large_jump_flag.sum()), random_state=20261007).copy()
    jump['review_group'] = 'Jump screen sample'
    selected = pd.concat([target, control, jump]).sort_values('case_id').reset_index(drop=True)
    rows, binrows = ([], [])
    atlas = private / 'waveform_review_images'
    atlas.mkdir(exist_ok=True)
    for j, row in selected.iterrows():
        code = f'QC{j + 1:03d}'
        p = locate(row.waveform_tid)
        result = {'case_id': int(row.case_id), 'review_code': code, 'review_group': row.review_group, 'low_plateau_flag': bool(row.low_plateau_flag), 'low_pulse_pressure_flag': bool(row.low_pulse_pressure_flag), 'inverted_sbp_dbp_flag': bool(row.inverted_sbp_dbp_flag), 'first_valid_rel_sec': row.first_valid_rel_sec}
        if p is None:
            result['wave_status'] = 'NO_LOCAL_WAVEFORM'
            rows.append(result)
            continue
        try:
            times, values, interval = read_wave_prefix(p, row.start_abs_sec + row.first_valid_rel_sec + 600)
            wave = bin_wave(times, values, row.start_abs_sec, row.first_valid_rel_sec, interval)
            original = pd.read_parquet(DATA / 'artmap_10s.parquet')
            num = original[original.case_id.eq(row.case_id)].set_index('time_sec').art_map
            wave['numeric_map'] = wave.time_sec.map(num)
            supported = wave.wave_coverage.ge(0.8)
            low = wave.numeric_map.lt(40) & wave.numeric_map.notna()
            low_supported = low & supported
            nonpulse = low_supported & wave.wave_span_90pct.le(5)
            pulse_low = low_supported & wave.pulse_like
            result.update({'wave_status': 'READABLE', 'sample_interval_sec': interval, 'supported_bins': int(supported.sum()), 'low_numeric_bins': int(low.sum()), 'low_numeric_wave_supported_bins': int(low_supported.sum()), 'low_numeric_nonpulsatile_bins': int(nonpulse.sum()), 'low_numeric_pulse_like_bins': int(pulse_low.sum()), 'all_supported_nonpulsatile_bins': int((supported & wave.wave_span_90pct.le(5)).sum()), 'all_supported_pulse_like_bins': int((supported & wave.pulse_like).sum()), 'prefix_values_sha256': hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(), 'wave_source_path': str(p), 'wave_source_bytes': p.stat().st_size, 'wave_source_mtime_ns': p.stat().st_mtime_ns, 'prefix_samples': len(values)})
            if int(nonpulse.sum()) >= 6:
                result['technical_review_status'] = 'NONPULSATILE_LOW_SUPPORT_REQUIRES_REVIEW'
            elif int(pulse_low.sum()) >= 6:
                result['technical_review_status'] = 'PULSATILE_LOW_PRESENT_NOT_AUTOMATIC_ARTIFACT'
            elif int(low.sum()) and (not int(low_supported.sum())):
                result['technical_review_status'] = 'LOW_NUMERIC_WITHOUT_USABLE_WAVE_SUPPORT'
            else:
                result['technical_review_status'] = 'MIXED_OR_NO_SUSTAINED_LOW_REQUIRES_REVIEW'
            wave.insert(0, 'review_code', code)
            binrows.append(wave)
            make_plot(code, row.review_group, row, wave, times, values, atlas / f'{code}.png')
        except Exception as e:
            result['wave_status'] = 'READ_ERROR'
            result['error_type'] = type(e).__name__
        rows.append(result)
        if (j + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(private / 'waveform_audit_progress.csv', index=False)
            print(json.dumps({'waveform_cases_completed': j + 1, 'total': len(selected)}), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(private / 'waveform_case_audit.csv', index=False)
    if binrows:
        pd.concat(binrows).to_csv(private / 'waveform_10s_bin_metrics.csv', index=False)
    summary = {'cases_reviewed': len(results), 'groups': results.review_group.value_counts().to_dict(), 'wave_status': results.wave_status.value_counts().to_dict(), 'technical_status': results.technical_review_status.value_counts().to_dict(), 'clinical_adjudication': 'NOT_PERFORMED', 'screening_is_not_validated_SQI': True}
    (OUT / 'waveform_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
