"""Frozen-definition event and frequency rerun; writes only the assigned OUT files.

Run with PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=configured_dependency_directory.
The exposure stage is independently deliverable to downstream outcome analyses.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_writebytecode = True
ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
OUT = Path(os.environ.get('IOH_QC_OUTPUT_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()
ANALYSIS = ROOT / 'analysis_v10_subject_phase_release'
DATA = ANALYSIS / 'outputs/intermediate'
PACKAGE = ROOT / 'EJA_投稿文件包_20261006_最终文字修订/03_可复现材料'
EXTENSIONS = PACKAGE / 'extensions/events'
sys.path[:0] = [str(EXTENSIONS), str(ANALYSIS / 'src')]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from event_extension import held_display, reference_events, classify_events
from simple_interval_summary import displayed_low_minutes, summary_row
from ioh.estimands.decomposition import decompose_deficit, emulate_last_visible
from ioh.estimands.episodes import detect_reference_episodes
SEED = 20260706
REPLICATES = 1000
POLICIES = ('original', 'qc60', 'qc_all', 'startup10')
THRESHOLDS = (55.0, 60.0, 65.0)
INTERVALS = (1.0, 2.5, 3.0, 5.0, 10.0)
REPORT_INTERVALS = (1.0, 2.5, 5.0)
KEYS = ['case_id', 'threshold', 'interval_min', 'offset_sec']
CATEGORIES = ['fresh', 'inherited_only', 'never_low', 'carry_in', 'first_inherited']
PRIVATE = OUT / 'data_restricted'
AGG = OUT / 'analysis/aggregate'
QA = OUT / 'qa'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def write_json(path, payload):
    Path(path).write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def validate_panel(panel, original):
    columns = ['case_id', 'time_sec', 'art_map']
    if panel.duplicated(columns[:2]).any():
        raise ValueError('Duplicate case/time rows in candidate panel')
    ids = np.sort(panel.case_id.unique())
    if not set(ids).issubset(set(original.case_id.unique())):
        raise ValueError('Candidate contains non-original cases')
    before = original[original.case_id.isin(ids)][columns].sort_values(columns[:2]).reset_index(drop=True)
    after = panel[columns].sort_values(columns[:2]).reset_index(drop=True)
    if len(before) != len(after) or not before[columns[:2]].equals(after[columns[:2]]):
        raise ValueError('Candidate compressed, shifted, or otherwise changed the original grid')
    x, y = (before.art_map.to_numpy(float), after.art_map.to_numpy(float))
    unchanged = (x == y) | np.isnan(x) & np.isnan(y)
    masked = np.isfinite(x) & np.isnan(y)
    if not np.all(unchanged | masked):
        raise ValueError('Candidate changed or imputed reference values instead of masking')
    changed_ids = after.loc[masked, 'case_id'].unique().tolist()
    return {'cases': len(ids), 'panel_rows': len(after), 'additional_masked_bins': int(masked.sum()), 'changed_retained_cases': len(changed_ids), 'changed_ids': changed_ids, 'time_grid_preserved': True, 'original_values_preserved_or_masked': True}

def event_row(case_id, threshold, interval, phase, events, result, gap=0, minimum=60):
    counts = np.bincount(result['category'], minlength=3)
    return {'case_id': case_id, 'threshold': float(threshold), 'interval_min': float(interval), 'offset_sec': phase * 10.0, 'gap_sec': gap, 'minimum_low_sec': minimum, 'episodes': len(events), 'fresh': int(counts[0]), 'inherited_only': int(counts[1]), 'never_low': int(counts[2]), 'carry_in': int(result['carry_in'].sum()), 'first_inherited': int(result['first_inherited'].sum())}

def episode_metrics(display, events, threshold):
    low = np.isfinite(display) & (display < threshold)
    delays = [int(np.argmax(low[s:e])) / 6 for s, e, _ in events if np.any(low[s:e])]
    count, detected = (len(events), len(delays))
    return {'episode_count': count, 'episode_detected': detected, 'episode_sensitivity': detected / count if count else np.nan, 'mean_detection_delay_min': float(np.mean(delays)) if delays else np.nan, 'missed_episodes': count - detected}

def analyze_case(case_id, times, values, thresholds=THRESHOLDS, intervals=INTERVALS, robust=True, frequency=True, events_output=True):
    times, values = (np.asarray(times, float), np.asarray(values, float))
    if times.shape != values.shape or not len(times) or (not np.isfinite(times).all()):
        raise ValueError('Invalid case grid')
    if len(times) > 1 and (not np.allclose(np.diff(times), 10, atol=1e-10, rtol=0)):
        raise ValueError('Grid must retain the original consecutive 10-second bins')
    frows, erows, drows, rrows = ([], [], [], [])
    refs = {t: reference_events(values, t, 0, 6) for t in thresholds}
    for t, events in refs.items():
        for ordinal, (start, end, nlow) in enumerate(events):
            rrows.append({'case_id': case_id, 'threshold': float(t), 'event_index': ordinal, 'start_index': start, 'end_index_exclusive': end, 'start_sec': times[start], 'end_sec_exclusive': times[end - 1] + 10, 'low_duration_sec': nlow * 10, 'reference_event_auc': float(np.maximum(t - values[start:end], 0).sum() / 6)})
    configs = {}
    if robust and 65 in thresholds:
        for gap in (0, 1, 2):
            base = reference_events(values, 65, gap, 6)
            for minimum in (6, 18, 30):
                if gap or minimum != 6:
                    configs[gap * 10, minimum * 10] = [e for e in base if e[2] >= minimum]
    for interval in intervals:
        nphase = int(round(interval * 6))
        for phase in range(nphase):
            display, held = held_display(values, nphase, phase)
            for threshold, events in refs.items():
                if frequency:
                    row = {'case_id': case_id, 'threshold': float(threshold), 'interval_min': float(interval), 'offset_sec': phase * 10.0, 'anesthesia_hours': len(values) / 360, **decompose_deficit(values, display, threshold, 1 / 6), **episode_metrics(display, events, threshold)}
                    frows.append(row)
                if not events_output:
                    continue
                result = classify_events(values, display, held, events, threshold)
                erows.append(event_row(case_id, threshold, interval, phase, events, result))
                if threshold == 65:
                    lengths = np.asarray([e[2] for e in events])
                    for label, select in [('1 to <3 min', lengths < 18), ('3 to <5 min', (lengths >= 18) & (lengths < 30)), ('At least 5 min', lengths >= 30)]:
                        counts = np.bincount(result['category'][select], minlength=3)
                        drows.append({'case_id': case_id, 'threshold': 65.0, 'interval_min': float(interval), 'offset_sec': phase * 10.0, 'Duration': label, 'episodes': int(select.sum()), 'fresh': int(counts[0]), 'inherited_only': int(counts[1]), 'never_low': int(counts[2])})
                    if interval == 5:
                        for (gap, minimum), config in configs.items():
                            classification = classify_events(values, display, held, config, threshold)
                            erows.append(event_row(case_id, threshold, interval, phase, config, classification, gap, minimum))
    return tuple((pd.DataFrame(x) for x in (frows, erows, drows, rrows)))

def subject_totals(frame, mapping, ids, columns):
    mapping = mapping[['case_id', 'subjectid']].drop_duplicates()
    if mapping.case_id.duplicated().any():
        raise ValueError('Conflicting case-to-subject mapping')
    subjects = mapping.set_index('case_id').subjectid.reindex(ids)
    if subjects.isna().any():
        raise ValueError('Subject identity missing')
    aligned = frame.reindex(ids, fill_value=0)[columns].copy()
    aligned['subjectid'] = subjects.to_numpy()
    return aligned.groupby('subjectid', sort=True)[columns].sum()

@lru_cache(maxsize=40)
def bootstrap_indices(nsubject, threshold, interval):
    identity = f'{SEED}|threshold=np.float64({float(threshold)!r})|interval_min=np.float64({float(interval)!r})'
    derived = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
    return np.random.default_rng(derived).integers(0, nsubject, size=(REPLICATES, nsubject))

def summarize_events(frame, mapping, ids, threshold, interval):
    columns = [c for c in CATEGORIES if c in frame.columns]
    by_case = frame.groupby('case_id')[columns + ['episodes']].sum()
    by_subject = subject_totals(by_case, mapping, ids, columns + ['episodes'])
    totals = by_subject.sum()
    idx = bootstrap_indices(len(by_subject), threshold, interval)
    boot = {c: by_subject[c].to_numpy()[idx].sum(axis=1) for c in by_subject}
    denominator = float(totals.episodes)
    valid = boot['episodes'] > 0
    nphase = round(interval * 6)
    ref_count = denominator / nphase
    if abs(ref_count - round(ref_count)) > 1e-08:
        raise ValueError('Reference-event count differs across phases')
    row = {'threshold': threshold, 'interval_min': interval, 'n_cases_total': len(ids), 'n_clusters_total': len(by_subject), 'bootstrap_cluster': 'subjectid', 'bootstrap_reps': REPLICATES, 'cases_with_episodes': int((by_case.episodes > 0).sum()), 'reference_episodes': int(round(ref_count)), 'phase_episode_pairs': int(denominator), 'number_of_phases': nphase}
    for col in columns:
        frac = totals[col] / denominator if denominator else np.nan
        draws = boot[col][valid] / boot['episodes'][valid]
        ci = np.quantile(draws, [0.025, 0.975]) if len(draws) else [np.nan, np.nan]
        row.update({col + '_phase_episode_pairs': int(totals[col]), col + '_expected_n': float(totals[col] / nphase), col + '_percent': 100 * frac, col + '_lower95': 100 * ci[0], col + '_upper95': 100 * ci[1]})
    detection = totals.fresh + totals.inherited_only
    det_samples = (boot['fresh'][valid] + boot['inherited_only'][valid]) / boot['episodes'][valid]
    det_ci = np.quantile(det_samples, [0.025, 0.975]) if len(det_samples) else [np.nan, np.nan]
    row.update({'expected_detected_episodes': float(detection / nphase), 'expected_missed_episodes': float(totals.never_low / nphase), 'detected_phase_episode_pairs': int(detection), 'missed_phase_episode_pairs': int(totals.never_low), 'episode_detection_probability': detection / denominator if denominator else np.nan, 'episode_detection_ci_low': det_ci[0], 'episode_detection_ci_high': det_ci[1], 'complete_miss_probability': totals.never_low / denominator if denominator else np.nan, 'complete_miss_ci_low': row['never_low_lower95'] / 100, 'complete_miss_ci_high': row['never_low_upper95'] / 100})
    return row

def compare_numeric(new, old, keys, columns=None):
    columns = columns or [c for c in old.columns if c not in keys]
    a, b = (new.set_index(keys).sort_index(), old.set_index(keys).sort_index())
    if not a.index.equals(b.index):
        raise ValueError('Frozen comparison keys differ')
    max_errors = {}
    for col in columns:
        x, y = (a[col].to_numpy(float), b[col].to_numpy(float))
        np.testing.assert_allclose(x, y, equal_nan=True, rtol=1e-12, atol=1e-08, err_msg=f'Frozen reconciliation: {col}')
        finite = np.isfinite(x) & np.isfinite(y)
        max_errors[col] = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
    return {'rows': len(a), 'columns': len(columns), 'maximum_absolute_difference': max(max_errors.values(), default=0.0)}

def included_case_ids(cohort):
    if 'included' not in cohort:
        raise ValueError('Policy cohort must explicitly declare inclusion status')
    if cohort.included.isna().any() or not cohort.included.isin([True, False]).all():
        raise ValueError('Policy cohort has ambiguous inclusion status')
    return set(cohort.loc[cohort.included, 'case_id'])

def load_panel(policy):
    panel = pd.read_parquet(PRIVATE / f'panel_{policy}.parquet', columns=['case_id', 'time_sec', 'art_map'])
    cohort = pd.read_csv(PRIVATE / f'cohort_{policy}.csv')
    ids = np.sort(panel.case_id.unique())
    if set(ids) != included_case_ids(cohort):
        raise ValueError('Panel and policy cohort membership differ')
    return (panel, ids)

def baseline_verify(original, frozen):
    ids = np.sort(original.case_id.unique())
    selected = ids[np.linspace(0, len(ids) - 1, 30, dtype=int)]
    rebuilt = []
    reconciled_displays = 0
    for case_id, trajectory in original[original.case_id.isin(selected)].groupby('case_id', sort=True):
        g = trajectory.sort_values('time_sec')
        times, values = (g.time_sec.to_numpy(), g.art_map.to_numpy(float))
        f, _, _, _ = analyze_case(case_id, times, values, robust=False, events_output=False)
        rebuilt.append(f)
        for interval in INTERVALS:
            for phase in range(round(interval * 6)):
                display, _ = held_display(values, round(interval * 6), phase)
                old = emulate_last_visible(times, values, interval * 60, phase * 10)
                np.testing.assert_allclose(display, old, equal_nan=True, atol=0, rtol=0)
                reconciled_displays += 1
        for threshold in THRESHOLDS:
            old = detect_reference_episodes(times, values, threshold, min_duration_sec=60)
            new = reference_events(values, threshold, 0, 6)
            if [(e.start_index, e.end_index) for e in old] != [(s, e) for s, e, _ in new]:
                raise ValueError('Frozen event definition did not reconcile')
    comparison = compare_numeric(pd.concat(rebuilt), frozen[frozen.case_id.isin(selected)], KEYS)
    result = {'status': 'PASS', 'independently_recomputed_cases': len(selected), 'display_schedules_compared': reconciled_displays, 'frequency_comparison': comparison, 'original_panel_sha256': sha(PRIVATE / 'panel_original.parquet'), 'frozen_panel_sha256': sha(DATA / 'artmap_10s.parquet'), 'frequency_source_sha256': sha(DATA / 'frequency_decomp_case_offset.parquet'), 'source_definition': 'Frozen core decomposition plus >=60-s contiguous low runs'}
    if result['original_panel_sha256'] != result['frozen_panel_sha256']:
        raise ValueError('Original panel is not byte-identical to frozen source')
    write_json(QA / 'events_baseline_exposure_reconciliation.json', result)
    return result

def exposure_stage(policies, original, frozen):
    baseline = baseline_verify(original, frozen)
    readiness = {'clinical_adjudication': False, 'primary_publication_policy': 'original', 'policies': {}}
    for policy in policies:
        started = time.monotonic()
        panel, ids = load_panel(policy)
        audit = validate_panel(panel, original)
        changed = set(audit.pop('changed_ids'))
        pieces = [frozen[frozen.case_id.isin(set(ids) - changed)].copy()]
        for n, (case_id, g) in enumerate(panel[panel.case_id.isin(changed)].groupby('case_id', sort=True), 1):
            g = g.sort_values('time_sec')
            f, _, _, _ = analyze_case(case_id, g.time_sec.to_numpy(), g.art_map.to_numpy(), robust=False, events_output=False)
            pieces.append(f)
            if n % 200 == 0:
                print(f'Exposure {policy}: {n}/{len(changed)} changed cases', flush=True)
        phase = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        phase = phase[list(frozen.columns)]
        expected = len(ids) * len(THRESHOLDS) * sum((round(x * 6) for x in INTERVALS))
        if len(phase) != expected:
            raise ValueError('Incomplete case/phase coverage')
        check = phase.groupby('case_id').anesthesia_hours.agg(['min', 'max'])
        duration = original[original.case_id.isin(ids)].groupby('case_id').size() / 360
        np.testing.assert_allclose(check['min'], duration, rtol=0, atol=1e-12)
        np.testing.assert_allclose(check['max'], duration, rtol=0, atol=1e-12)
        np.testing.assert_allclose(phase.display_auc - phase.true_auc, phase.overdisplay_auc - phase.hidden_auc, atol=1e-08, rtol=1e-12)
        destination = PRIVATE / f'frequency_decomp_{policy}.parquet'
        phase.to_parquet(destination, index=False)
        audit.update({'status': 'READY', 'path': str(destination), 'sha256': sha(destination), 'rows': len(phase), 'thresholds': THRESHOLDS, 'intervals': INTERVALS, 'recomputed_retained_cases': len(changed), 'reused_unchanged_cases': len(ids) - len(changed), 'elapsed_seconds': round(time.monotonic() - started, 2), 'original_schema_preserved': pq.read_schema(destination).names == pq.read_schema(DATA / 'frequency_decomp_case_offset.parquet').names})
        readiness['policies'][policy] = audit
        write_json(QA / 'events_frequency_ready.json', readiness)
        print(f'READY frequency_decomp_{policy}.parquet: {len(ids)} cases, all five intervals and three thresholds', flush=True)
    readiness['baseline'] = baseline
    write_json(QA / 'events_frequency_ready.json', readiness)

def quantiles(frame, metrics, interval, threshold, summary):
    rows = []
    for metric in metrics:
        x = frame[metric].dropna()
        q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
        rows.append({'threshold': threshold, 'interval_min': interval, 'Metric': metric, 'Summary': summary, 'Cases': len(x), **dict(zip(['P5', 'Q1', 'Median', 'Q3', 'P95'], q))})
    return rows

def summarize_decomposition(frame, mapping, ids, threshold, interval):
    columns = ['true_auc', 'display_auc', 'hidden_auc', 'overdisplay_auc', 'concordant_auc', 'normotensive_display_discordance_min', 'hypotensive_display_discordance_min', 'anesthesia_hours', 'display_valid_min', 'display_unavailable_min']
    by_case = frame.groupby('case_id')[columns].mean()
    grouped = subject_totals(by_case, mapping, ids, columns)
    total = grouped.sum()
    material = f'frequency_population|{SEED}|{threshold:.12g}|{interval:.12g}|subjectid'
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'big')
    idx = np.random.default_rng(seed).integers(0, len(grouped), size=(REPLICATES, len(grouped)))
    boot = {c: grouped[c].to_numpy()[idx].sum(axis=1) for c in columns}
    row = {'threshold': threshold, 'interval_min': interval, 'n_cases': len(ids), 'n_clusters': len(grouped), 'bootstrap_cluster': 'subjectid', 'bootstrap_reps': REPLICATES, 'original_anesthesia_hours': total.anesthesia_hours, 'valid_reference_min': total.display_valid_min + total.display_unavailable_min}
    for col in ('true_auc', 'display_auc', 'hidden_auc', 'overdisplay_auc', 'concordant_auc'):
        row[col + '_total'] = total[col]
        row[col + '_total_ci_low'], row[col + '_total_ci_high'] = np.quantile(boot[col], [0.025, 0.975])
    specs = {'HDR': ('hidden_auc', 'true_auc'), 'ODR': ('overdisplay_auc', 'true_auc'), 'normotensive_display_discordance_min_per_anesthesia_hour': ('normotensive_display_discordance_min', 'anesthesia_hours'), 'hypotensive_display_discordance_min_per_anesthesia_hour': ('hypotensive_display_discordance_min', 'anesthesia_hours')}
    derived = {'NetBias': (total.display_auc - total.true_auc, total.true_auc, boot['display_auc'] - boot['true_auc'], boot['true_auc']), 'absolute_auc_disagreement_ratio': (total.hidden_auc + total.overdisplay_auc, total.true_auc, boot['hidden_auc'] + boot['overdisplay_auc'], boot['true_auc']), 'absolute_state_discordance_min_per_hour': (total.normotensive_display_discordance_min + total.hypotensive_display_discordance_min, total.anesthesia_hours, boot['normotensive_display_discordance_min'] + boot['hypotensive_display_discordance_min'], boot['anesthesia_hours'])}
    for name, (numerator, denominator) in specs.items():
        derived[name] = (total[numerator], total[denominator], boot[numerator], boot[denominator])
    for name, (num, den, bnum, bden) in derived.items():
        row[name] = num / den if den > 0 else np.nan
        draws = bnum[bden > 0] / bden[bden > 0]
        row[name + '_ci_low'], row[name + '_ci_high'] = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
    return row

def publication_stage(policies, mapping):
    payload = json.loads((AGG / 'events_summary.json').read_text())
    decomposition, wide_rows, interval_rows, compact_robust = ([], {}, {}, [])
    for policy in policies:
        frequency = pd.read_parquet(PRIVATE / f'frequency_decomp_{policy}.parquet')
        ids = np.sort(frequency.case_id.unique())
        decomp = pd.DataFrame([summarize_decomposition(group, mapping, ids, threshold, interval) for (threshold, interval), group in frequency.groupby(['threshold', 'interval_min'])])
        if policy == 'original':
            old = pd.read_csv(ANALYSIS / 'outputs/tables/frequency_decomposition_bootstrap.csv')
            cols = ['true_auc_total', 'display_auc_total', 'hidden_auc_total', 'overdisplay_auc_total', 'HDR', 'ODR', 'NetBias', 'HDR_ci_low', 'HDR_ci_high', 'ODR_ci_low', 'ODR_ci_high', 'NetBias_ci_low', 'NetBias_ci_high', 'normotensive_display_discordance_min_per_anesthesia_hour', 'hypotensive_display_discordance_min_per_anesthesia_hour']
            checked = compare_numeric(decomp, old, ['threshold', 'interval_min'], cols)
            write_json(QA / 'events_decomposition_reconciliation.json', {'status': 'PASS', **checked})
        decomp.insert(0, 'policy', policy)
        decomp.to_csv(AGG / f'events_decomposition_{policy}.csv', index=False)
        decomposition.append(decomp)
        primary = payload['policies'][policy]['primary_65_5min']
        robust = pd.read_csv(AGG / f'events_robustness_{policy}.csv')
        compact = []
        for _, r in robust.iterrows():
            rec = {'policy': policy, 'Normal gap (s)': int(r['Gap allowance (s)']), 'Low time >= (s)': int(r['Minimum cumulative low time (s)']), 'Episodes': int(r.reference_episodes)}
            for title, key in [('Fresh sample, % (95% confidence interval)', 'fresh'), ('Inherited only, % (95% confidence interval)', 'inherited_only'), ('No low display, % (95% confidence interval)', 'never_low')]:
                rec[title] = f"{r[key + '_percent']:.1f} ({r[key + '_lower95']:.1f} to {r[key + '_upper95']:.1f})"
            compact.append(rec)
        compact_robust.extend(compact)
        if policy == 'original':
            old_robust = pd.read_csv(PACKAGE / 'outputs/tables/Supp_Table_S25.csv')
            old_robust = old_robust.rename(columns={old_robust.columns[1]: 'Low time >= (s)'})
            old_robust['Episodes'] = old_robust.Episodes.astype(str).str.replace(',', '', regex=False).astype(int)
            pd.testing.assert_frame_equal(pd.DataFrame(compact).drop(columns='policy'), old_robust, check_dtype=False)
            old_duration = pd.read_csv(PACKAGE / 'outputs/aggregate/new_vs_inherited_by_duration.csv')
            new_duration = pd.read_csv(AGG / 'events_duration_original.csv').query('interval_min == 5')
            duration_check = compare_numeric(new_duration, old_duration, ['Duration', 'Category'])
            old_case = pd.read_csv(PACKAGE / 'outputs/aggregate/case_phase_quantiles.csv')
            new_case = pd.read_csv(AGG / 'events_case_phase_distribution_original.csv').query('threshold == 65 and interval_min == 5')
            new_case = new_case[new_case.Metric.isin(old_case.Metric)]
            case_check = compare_numeric(new_case, old_case, ['Metric', 'Summary'])
            write_json(QA / 'events_extension_reconciliation.json', {'status': 'PASS', 'all_nine_robustness_rules_publication_cells': 'EXACT', 'duration_categories_including_subject_CI': duration_check, 'case_phase_distribution': case_check})
        simple = pd.read_csv(AGG / f'events_interval_absolute_summary_{policy}.csv')
        simple = simple[simple['Threshold (mmHg)'] == 65].copy()
        simple['numeric_interval'] = pd.to_numeric(simple['Sampling interval (min)'], errors='coerce')
        reference = simple[simple.numeric_interval.isna()].iloc[0].to_dict()
        sampled = simple[simple.numeric_interval == 5].iloc[0].to_dict()
        dp = decomp.query('threshold == 65 and interval_min == 5').iloc[0].to_dict()
        payload['policies'][policy]['principal'] = {'threshold_mmHg': 65, 'sampling_interval_min': 5, 'events': primary, 'continuous_reference': reference, 'held_display': sampled, 'decomposition': dp}
        payload['policies'][policy]['intervals_MAP65'] = simple.drop(columns='numeric_interval').to_dict('records')

        def put(metric, unit, value, low=None, high=None, digits=1):
            text = f'{value:,.{digits}f}'
            if low is not None:
                text += f' ({low:,.{digits}f} to {high:,.{digits}f})'
            wide_rows.setdefault((metric, unit), {})[policy] = text
        put('Included cases', 'n', primary['n_cases_total'], digits=0)
        put('Included subjects', 'n', primary['n_clusters_total'], digits=0)
        put('Cases with qualifying reference events', 'n', primary['cases_with_episodes'], digits=0)
        put('Reference events >=60 s', 'n', primary['reference_episodes'], digits=0)
        put('Detected events, phase-averaged expected count', 'expected n', primary['expected_detected_episodes'])
        put('Missed events, phase-averaged expected count', 'expected n', primary['expected_missed_episodes'])
        for metric, stem in [('Any low display during event', 'episode_detection'), ('No low display during event', 'complete_miss')]:
            put(metric, '% (95% CI)', primary[stem + '_probability'] * 100, primary[stem + '_ci_low'] * 100, primary[stem + '_ci_high'] * 100)
        for metric, stem in [('Fresh within-event low sample', 'fresh'), ('Inherited low display only', 'inherited_only')]:
            put(metric, '% (95% CI)', primary[stem + '_percent'], primary[stem + '_lower95'], primary[stem + '_upper95'])
        put('Valid reference observation time', 'min', reference['Valid reference observation time (min)'])
        put('Original anaesthesia duration', 'hours', dp['original_anesthesia_hours'])
        for name, record in [('Reference', reference), ('Display', sampled)]:
            for metric, unit, stem, digits in [('hypotension duration', 'min (95% CI)', 'Total hypotension duration (min)', 1), ('deficit AUC', 'mmHg min (95% CI)', 'Total deficit AUC (mmHg min)', 1), ('pooled TWA deficit', 'mmHg (95% CI)', 'Pooled TWA deficit (mmHg)', 3)]:
                put(name + ' ' + metric, unit, record[stem], record[stem + ' lower 95%'], record[stem + ' upper 95%'], digits)
        for metric, stem in [('Hidden deficit fraction', 'HDR'), ('Overdisplay deficit fraction', 'ODR'), ('Net AUC bias', 'NetBias')]:
            put(metric, '% (95% CI)', dp[stem] * 100, dp[stem + '_ci_low'] * 100, dp[stem + '_ci_high'] * 100)
        stem = 'absolute_state_discordance_min_per_hour'
        put('Either-direction state discordance', 'min/anaesthesia-hour (95% CI)', dp[stem], dp[stem + '_ci_low'], dp[stem + '_ci_high'], 2)
        for _, record in simple[simple.numeric_interval.isin(REPORT_INTERVALS)].iterrows():
            interval = record.numeric_interval
            for metric, unit, stem, digits in [('Detected reference events', 'expected n (95% CI)', 'Events with low display (expected n)', 1), ('Any low display during event', '% (95% CI)', 'Events with low display (%)', 1), ('No low display during event', '% (95% CI)', 'Events missed (%)', 1), ('Displayed hypotension duration', 'min (95% CI)', 'Total hypotension duration (min)', 1), ('Displayed deficit AUC', 'mmHg min (95% CI)', 'Total deficit AUC (mmHg min)', 1), ('Displayed pooled TWA deficit', 'mmHg (95% CI)', 'Pooled TWA deficit (mmHg)', 3)]:
                key = (interval, metric, unit)
                interval_rows.setdefault(key, {})[policy] = f"{record[stem]:,.{digits}f} ({record[stem + ' lower 95%']:,.{digits}f} to {record[stem + ' upper 95%']:,.{digits}f})"
        print(f'Publication summaries {policy}: ready', flush=True)
    pd.concat(decomposition, ignore_index=True).to_csv(AGG / 'events_decomposition_comparative.csv', index=False)
    pd.DataFrame(compact_robust).to_csv(AGG / 'events_robustness_publication.csv', index=False)
    pd.DataFrame([{'Metric': key[0], 'Unit': key[1], **values} for key, values in wide_rows.items()]).to_csv(AGG / 'events_publication_S31_wide.csv', index=False)
    pd.DataFrame([{'Interval (min)': key[0], 'Metric': key[1], 'Unit': key[2], **values} for key, values in interval_rows.items()]).to_csv(AGG / 'events_publication_S34_wide.csv', index=False)
    payload['schema_version'] = 'events-qc-1.0'
    payload['code_sha256'] = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('test_qc_events.py'))}
    payload['field_dictionary'] = {'policies.<policy>.principal.events': 'Case/subject/event n and subject-bootstrap event detection/miss/fresh/inherited-only estimates and CI at MAP65, 5 min', 'policies.<policy>.principal.continuous_reference': 'Reference duration, AUC and pooled TWA; total valid reference minutes denominator', 'policies.<policy>.principal.held_display': 'Phase-averaged 5-min display duration, AUC and pooled TWA with frozen frequency-bootstrap CI', 'policies.<policy>.principal.decomposition': 'Population hidden/overdisplay fractions and net bias; ratios use total reference AUC; state discordance uses original anesthesia-hours', 'policies.<policy>.intervals_MAP65': 'Continuous reference and all five intervals with duration/AUC/TWA/event estimates and CI', 'probability': 'Fraction 0 to 1; _percent and percent-based _lower95/_upper95 use 0 to 100', 'expected_n': 'Phase-averaged count, not an integer count of individually adjudicated detected events', 'source_status': 'All clinical signoff fields remain false/pending; technical candidate sensitivity only'}
    write_json(AGG / 'events_summary.json', payload)
    write_json(QA / 'events_summary.json', payload)

def summarize_policy(policy, frequency, event, duration, mapping, ids):
    base = event[(event.gap_sec == 0) & (event.minimum_low_sec == 60)]
    rates, robustness, duration_summaries = ([], [], [])
    for (threshold, interval), group in base.groupby(['threshold', 'interval_min'], sort=True):
        rates.append(summarize_events(group, mapping, ids, threshold, interval))
    for (gap, minimum), group in event[(event.threshold == 65) & (event.interval_min == 5)].groupby(['gap_sec', 'minimum_low_sec']):
        row = summarize_events(group, mapping, ids, 65.0, 5.0)
        row.update({'Gap allowance (s)': gap, 'Minimum cumulative low time (s)': minimum, 'Reference episodes': row['reference_episodes'], 'Cases with episodes': row['cases_with_episodes'], 'Phase-episode pairs': row['phase_episode_pairs']})
        robustness.append(row)
    for (interval, label), group in duration.groupby(['interval_min', 'Duration'], sort=False):
        row = summarize_events(group, mapping, ids, 65.0, interval)
        for category in ('fresh', 'inherited_only', 'never_low'):
            duration_summaries.append({'threshold': 65.0, 'interval_min': interval, 'Duration': label, 'Reference episodes': row['reference_episodes'], 'Category': category, 'Percent': row[category + '_percent'], 'Lower 95%': row[category + '_lower95'], 'Upper 95%': row[category + '_upper95']})
    frequency = frequency.copy()
    frequency['valid_reference_min'] = frequency.display_valid_min + frequency.display_unavailable_min
    frequency['displayed_hypotension_min'] = displayed_low_minutes(frequency.true_hypotension_min, frequency.reference_hypotension_with_display_unavailable_min, frequency.normotensive_display_discordance_min, frequency.hypotensive_display_discordance_min, frequency.valid_reference_min)
    groupkeys = ['case_id', 'threshold', 'interval_min']
    case = frequency.groupby(groupkeys).mean(numeric_only=True).drop(columns='offset_sec').reset_index()
    case = case.merge(mapping[['case_id', 'subjectid']], on='case_id', validate='many_to_one')
    invariants = case.groupby(['case_id', 'threshold'])[['valid_reference_min', 'true_auc', 'true_hypotension_min', 'episode_count']].agg(lambda x: x.max() - x.min())
    if np.max(invariants.to_numpy()) > 1e-08:
        raise ValueError('Reference time or burden changed across intervals')
    rate_lookup = {(x['threshold'], x['interval_min']): x for x in rates}
    simple = []
    for threshold, g in case.groupby('threshold'):
        simple.append(summary_row(g[g.interval_min == 5], threshold, 5, reference=True))
        for interval, h in g.groupby('interval_min'):
            row = summary_row(h, threshold, interval)
            rate = rate_lookup[threshold, interval]
            row['Events with low display (%) lower 95%'] = rate['episode_detection_ci_low'] * 100
            row['Events with low display (%) upper 95%'] = rate['episode_detection_ci_high'] * 100
            row['Events missed (expected n)'] = rate['expected_missed_episodes']
            row['Events missed (%)'] = rate['complete_miss_probability'] * 100
            row['Events missed (%) lower 95%'] = rate['complete_miss_ci_low'] * 100
            row['Events missed (%) upper 95%'] = rate['complete_miss_ci_high'] * 100
            simple.append(row)
    frequency['absolute_state_discordance_min_per_hour'] = (frequency.normotensive_display_discordance_min + frequency.hypotensive_display_discordance_min) / frequency.anesthesia_hours
    frequency['low_reference_normal_display_min_per_hour'] = frequency.normotensive_display_discordance_min / frequency.anesthesia_hours
    frequency['normal_reference_low_display_min_per_hour'] = frequency.hypotensive_display_discordance_min / frequency.anesthesia_hours
    frequency['absolute_auc_disagreement_ratio'] = (frequency.hidden_auc + frequency.overdisplay_auc) / frequency.true_auc.replace(0, np.nan)
    frequency['net_auc_bias_percent'] = frequency.net_bias_ratio * 100
    metrics = ['absolute_state_discordance_min_per_hour', 'low_reference_normal_display_min_per_hour', 'normal_reference_low_display_min_per_hour', 'hidden_deficit_ratio', 'overdisplay_deficit_ratio', 'absolute_auc_disagreement_ratio', 'net_auc_bias_percent']
    distributions, cohort_phases, event_phases, private_case = ([], [], [], [])
    for (threshold, interval), phase in frequency.groupby(['threshold', 'interval_min']):
        bycase = phase.groupby('case_id')[metrics].mean()
        spread = phase.groupby('case_id')[metrics].agg(lambda s: s.max() - s.min())
        distributions += quantiles(bycase, metrics, interval, threshold, 'Phase-averaged case value')
        distributions += quantiles(spread, metrics, interval, threshold, 'Within-case phase range')
        private_case.append(bycase.reset_index().assign(threshold=threshold, interval_min=interval))
        cols = ['true_auc', 'display_auc', 'hidden_auc', 'overdisplay_auc', 'normotensive_display_discordance_min', 'hypotensive_display_discordance_min', 'anesthesia_hours']
        cohort = phase.groupby('offset_sec')[cols].sum()
        cohort['Hidden deficit percent'] = 100 * cohort.hidden_auc / cohort.true_auc
        cohort['Overdisplay deficit percent'] = 100 * cohort.overdisplay_auc / cohort.true_auc
        cohort['Net AUC bias percent'] = 100 * (cohort.display_auc - cohort.true_auc) / cohort.true_auc
        cohort['Absolute discordance min per hour'] = (cohort.normotensive_display_discordance_min + cohort.hypotensive_display_discordance_min) / cohort.anesthesia_hours
        cohort_phases.append(cohort.reset_index().assign(threshold=threshold, interval_min=interval))
        sub = base[(base.threshold == threshold) & (base.interval_min == interval)].copy()
        sub['miss_fraction'] = sub.never_low / sub.episodes.replace(0, np.nan)
        ec = sub.groupby('case_id').miss_fraction.mean().to_frame()
        er = sub.groupby('case_id').miss_fraction.agg(lambda s: s.max() - s.min()).to_frame()
        distributions += quantiles(ec, ['miss_fraction'], interval, threshold, 'Phase-averaged case value')
        distributions += quantiles(er, ['miss_fraction'], interval, threshold, 'Within-case phase range')
        ep = sub.groupby('offset_sec')[['episodes'] + CATEGORIES].sum()
        for col in CATEGORIES:
            ep[col + '_percent'] = 100 * ep[col] / ep.episodes
        event_phases.append(ep.reset_index().assign(threshold=threshold, interval_min=interval))
    pd.concat(private_case).to_parquet(PRIVATE / f'events_case_phase_averages_{policy}.parquet', index=False)
    return {'rates': pd.DataFrame(rates), 'robustness': pd.DataFrame(robustness), 'duration': pd.DataFrame(duration_summaries), 'interval_absolute_summary': pd.DataFrame(simple), 'case_phase_distribution': pd.DataFrame(distributions), 'cohort_phase_schedule': pd.concat(cohort_phases), 'event_phase_schedule': pd.concat(event_phases)}

def reconcile_original(tables, event, frequency):
    rates = tables['rates']
    old = pd.read_csv(ANALYSIS / 'outputs/tables/episode_observability_by_threshold_interval.csv')
    cols = ['reference_episodes', 'expected_detected_episodes', 'expected_missed_episodes', 'episode_detection_probability', 'episode_detection_ci_low', 'episode_detection_ci_high', 'complete_miss_probability', 'complete_miss_ci_low', 'complete_miss_ci_high', 'cases_with_episodes']
    event_check = compare_numeric(rates, old, ['threshold', 'interval_min'], cols)
    simple = tables['interval_absolute_summary']
    old_simple = pd.read_csv(PACKAGE / 'outputs/aggregate/interval_absolute_summary.csv')
    skeys = ['Threshold (mmHg)', 'Sampling interval (min)']
    simple[skeys[1]] = simple[skeys[1]].astype(str).replace({'1.0': '1', '3.0': '3', '5.0': '5', '10.0': '10'})
    old_simple[skeys[1]] = old_simple[skeys[1]].astype(str).replace({'1.0': '1', '3.0': '3', '5.0': '5', '10.0': '10'})
    numeric = [c for c in old_simple.columns if c not in skeys and pd.api.types.is_numeric_dtype(old_simple[c])]
    simple_check = compare_numeric(simple, old_simple, skeys, numeric)
    primary = rates[(rates.threshold == 65) & (rates.interval_min == 5)].iloc[0]
    if int(primary.reference_episodes) != 9187:
        raise ValueError('Frozen primary did not recover 9187 events')
    base = event[(event.threshold == 65) & (event.interval_min == 5) & (event.gap_sec == 0) & (event.minimum_low_sec == 60)]
    new = base.groupby('case_id')[['episodes'] + CATEGORIES].sum()
    old_case = pd.read_parquet(DATA / 'episode_observability_case_5min.parquet')
    old_case = old_case[old_case.threshold == 65].groupby('case_id')[['reference_episodes', 'detected_phase_episode_pairs']].sum().reindex(new.index, fill_value=0)
    np.testing.assert_array_equal(new.episodes, old_case.reference_episodes * 30)
    np.testing.assert_array_equal(new.fresh + new.inherited_only, old_case.detected_phase_episode_pairs)
    result = {'status': 'PASS', 'event_rates_and_bootstrap_ci': event_check, 'simple_interval_summary_including_ci': simple_check, 'all_case_5min_reference_and_detection_counts': 'PASS', 'baseline_primary': primary.to_dict()}
    write_json(QA / 'events_original_reconciliation.json', result)
    return result

def event_stage(policies, original, mapping):
    all_tables, summaries = ({}, {})
    original_frames = None
    for policy in policies:
        started = time.monotonic()
        panel, ids = load_panel(policy)
        audit = validate_panel(panel, original)
        changed = set(audit.pop('changed_ids'))
        if policy == 'original':
            changed = set(ids)
        elif original_frames is None:
            original_frames = tuple((pd.read_parquet(PRIVATE / f'events_{stem}_original.parquet') for stem in ('case_phase', 'duration_phase', 'reference')))
        parts = [[], [], []]
        if policy != 'original':
            for part, frame in zip(parts, original_frames):
                part.append(frame[frame.case_id.isin(set(ids) - changed)].copy())
        for n, (case_id, g) in enumerate(panel[panel.case_id.isin(changed)].groupby('case_id', sort=True), 1):
            g = g.sort_values('time_sec')
            _, events, durations, references = analyze_case(case_id, g.time_sec.to_numpy(), g.art_map.to_numpy(), frequency=False)
            for part, frame in zip(parts, (events, durations, references)):
                if len(frame):
                    part.append(frame)
            if n % 400 == 0:
                print(f'Events {policy}: {n}/{len(changed)} changed cases', flush=True)
        frames = tuple((pd.concat(x, ignore_index=True) for x in parts))
        for frame, stem in zip(frames, ('case_phase', 'duration_phase', 'reference')):
            frame.to_parquet(PRIVATE / f'events_{stem}_{policy}.parquet', index=False)
        if policy == 'original':
            original_frames = frames
        frequency = pd.read_parquet(PRIVATE / f'frequency_decomp_{policy}.parquet')
        tables = summarize_policy(policy, frequency, frames[0], frames[1], mapping, ids)
        reconciliation = reconcile_original(tables, frames[0], frequency) if policy == 'original' else None
        for name, table in tables.items():
            table.insert(0, 'policy', policy)
            table.to_csv(AGG / f'events_{name}_{policy}.csv', index=False)
            all_tables.setdefault(name, []).append(table)
        primary = tables['rates'].query('threshold == 65 and interval_min == 5').iloc[0].to_dict()
        summaries[policy] = {'primary_65_5min': primary, 'panel_audit': audit, 'elapsed_seconds': round(time.monotonic() - started, 2), 'source_sha256': {'panel': sha(PRIVATE / f'panel_{policy}.parquet'), 'frequency': sha(PRIVATE / f'frequency_decomp_{policy}.parquet')}, 'baseline_reconciliation': reconciliation}
        write_json(QA / f'events_summary_{policy}.json', summaries[policy])
        print(f"DONE events {policy}: {primary['reference_episodes']} reference episodes; detection {100 * primary['episode_detection_probability']:.6f}%", flush=True)
    for name, frames in all_tables.items():
        pd.concat(frames, ignore_index=True).to_csv(AGG / f'events_{name}_comparative.csv', index=False)
    sourcefiles = [EXTENSIONS / 'event_extension.py', EXTENSIONS / 'run_extension.py', EXTENSIONS / 'simple_interval_summary.py', ANALYSIS / 'src/ioh/estimands/decomposition.py', ANALYSIS / 'src/ioh/estimands/episodes.py']
    payload = {'status': 'COMPLETE' if tuple(policies) == POLICIES else 'PARTIAL_POLICY_RUN', 'policies': summaries, 'primary_publication_policy': 'original', 'clinical_adjudication': False, 'technical_candidate_policies_only': True, 'manuscript_modified': False, 'published_externally': False, 'definitions': {'events': 'Continuous finite MAP below threshold for at least 60 s; missing bins break runs', 'detection': 'Any low held display within reference event, including inherited low display', 'fresh_priority': 'Fresh within-event low sample takes priority over inherited-only classification', 'duration_auc': 'All valid reference bins, including low runs shorter than 60 s', 'twa': 'Pooled deficit AUC / total valid reference observation minutes; ratio of sums', 'anesthesia_hours': 'Original complete grid duration retained, not reduced valid-time duration', 'missing_scheduled_sample': 'No new sample; previously held display persists, with no phase reset', 'robustness': '5-min MAP65; observed normal gaps 0/10/20 s; cumulative low time >=60/180/300 s; missing gaps never bridged', 'bootstrap': '1000 subject-cluster percentile replicates; frozen seeds and subject ordering', 'startup10': 'Window sensitivity retaining original cases; includes true early hypotension, not artifact correction'}, 'thresholds': THRESHOLDS, 'intervals': INTERVALS, 'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sourcefiles}}
    write_json(QA / 'events_summary.json', payload)
    write_json(AGG / 'events_summary.json', payload)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'exposures', 'events', 'publication'), default='all')
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    args = parser.parse_args()
    for path in (PRIVATE, AGG, QA):
        path.mkdir(parents=True, exist_ok=True)
    original, _ = load_panel('original')
    mapping = pd.read_parquet(DATA / 'vitaldb_manifest.parquet', columns=['case_id', 'subjectid'])
    if args.stage in ('all', 'exposures'):
        frozen = pd.read_parquet(DATA / 'frequency_decomp_case_offset.parquet')
        exposure_stage(args.policies, original, frozen)
        del frozen
    if args.stage in ('all', 'events'):
        event_stage(args.policies, original, mapping)
    if args.stage in ('all', 'events', 'publication'):
        publication_stage(args.policies, mapping)
if __name__ == '__main__':
    main()
