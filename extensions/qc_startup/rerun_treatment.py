"""Reapply frozen treatment definitions to technical-QC candidate panels.

No clinical adjudication, treatment-effect estimation, or external release.
Configure local inputs with --source-project or IOH_PROJECT_ROOT, and the
QC input/output directory with --output-root or IOH_QC_OUTPUT_ROOT.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
DEFAULT_OUT = Path(__file__).resolve().parents[1]
PATH_ARGS = argparse.ArgumentParser(add_help=False)
PATH_ARGS.add_argument('--source-project', type=Path, default=Path(os.environ.get('IOH_PROJECT_ROOT', DEFAULT_OUT.parent)))
PATH_ARGS.add_argument('--output-root', type=Path, default=Path(os.environ.get('IOH_QC_OUTPUT_ROOT', DEFAULT_OUT)))
PATH_CONFIG, _ = PATH_ARGS.parse_known_args()
OUT = PATH_CONFIG.output_root.expanduser().resolve()
ROOT = PATH_CONFIG.source_project.expanduser().resolve()
FROZEN = ROOT / '治疗相关事件补充修订_20261004'
DATA = ROOT / 'analysis_v10_subject_phase_release/outputs/intermediate'
sys.path.insert(0, str(FROZEN / 'code'))
import treatment_core as core
spec = importlib.util.spec_from_file_location('frozen_treatment_analysis', FROZEN / 'code/run_analysis.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
POLICIES = ['original', 'qc60', 'qc_all', 'startup10']
BINS = [6, 15, 30]
SCENARIOS = [('Primary', 'primary', 'stable', 60, 300, 0, False), ('Merge 30 s', 'merge30', 'stable', 30, 300, 0, False), ('Merge 120 s', 'merge120', 'stable', 120, 300, 0, False), ('No stability requirement', 'no_stability', 'raw', 60, 300, 0, False), ('At least 30 s preceding low MAP', 'preceding_low30', 'stable', 60, 300, 30, False), ('First low event per subject', 'first_low_subject', 'stable', 60, 300, 0, True), ('Exclude first 10 min', 'exclude_first10', 'stable', 60, 600, 0, False)]
EVENT_COLUMNS = ['case_id', 'time_sec', 'drug', 'kind', 'record_changes', 'subjectid', 'reference', 'low', 'pre_low_sec'] + [f'{n}_{c}' for n in BINS for c in frozen.CATEGORIES]
RECOVERY_COLUMNS = ['case_id', 'subjectid', 'time_sec', 'recovery_delay_sec'] + [f'persistence_{n}' for n in BINS]

def empty_events():
    frame = pd.DataFrame(columns=EVENT_COLUMNS)
    frame['low'] = frame['low'].astype(bool)
    return frame

def panel_to_art(panel, meta, included_cases=None):
    """Preserve zero-anchored complete 10-s bins, including every masked NaN."""
    if not {'case_id', 'time_sec', 'art_map'}.issubset(panel.columns):
        raise ValueError('Panel schema does not contain the frozen reference fields')
    if panel[['case_id', 'time_sec']].isna().any().any():
        raise ValueError('Missing panel grid identifiers')
    selected = set(panel.case_id) if included_cases is None else set(included_cases)
    if selected - set(panel.case_id) or selected - set(meta.index):
        raise ValueError('An included cohort case has no panel or frozen metadata')
    art = {}
    for cid, group in panel[panel.case_id.isin(selected)].groupby('case_id', sort=True):
        group = group.sort_values('time_sec', kind='stable')
        grid = group.time_sec.to_numpy()
        if not np.allclose(grid, np.arange(len(group)) * 10.0, rtol=0, atol=1e-08):
            raise ValueError('Panel has compressed, duplicated, or shifted grid bins')
        duration = float(meta.loc[cid, 'aneend']) - max(0.0, float(meta.loc[cid, 'anestart']))
        complete = group[group.time_sec + 10 <= duration + 1e-09]
        art[int(cid)] = complete.art_map.to_numpy(dtype=float)
    return art

def validate_masked_art(art, base):
    if set(art) - set(base):
        raise ValueError('QC policy introduced a case outside the frozen cohort')
    masked = 0
    for cid, values in art.items():
        original = base[cid]
        if len(values) != len(original):
            raise ValueError('QC policy changed case duration or reference grid')
        keep = np.isfinite(values)
        if not np.array_equal(values[keep], original[keep]):
            raise ValueError('QC policy changed a retained value or filled original missingness')
        masked += int((np.isfinite(original) & ~keep).sum())
    return {'retained_cases': len(art), 'excluded_cases': len(base) - len(art), 'newly_masked_complete_bins': masked, 'global_grid_and_retained_values_unchanged': True}

def mask_window_audit(events, art, base, minimum_time=300):
    counts = dict.fromkeys(['events_checked', 'excluded_case_events', 'events_before_minimum_time_not_evaluated', 'original_reference_ineligible_events_not_evaluated', 'events_with_masked_pre_reference_support', 'events_with_masked_preceding_low_run', 'event_phases_with_masked_active_sample', 'events_with_masked_post_support', 'post_phase_bins_with_masked_active_sample'], 0)
    for row in events.itertuples():
        cid, t = (row.case_id, row.time_sec)
        if cid not in art:
            counts['excluded_case_events'] += 1
            continue
        if t < minimum_time:
            counts['events_before_minimum_time_not_evaluated'] += 1
            continue
        original = base[cid]
        state = core.event_state(original, t, 30, 0)
        if state is None:
            counts['original_reference_ineligible_events_not_evaluated'] += 1
            continue
        counts['events_checked'] += 1
        newly_masked = np.isfinite(original) & ~np.isfinite(art[cid])
        ix = state['index']
        counts['events_with_masked_pre_reference_support'] += int(newly_masked[ix - 5:ix + 1].any())
        if state['reference'] < 65:
            counts['events_with_masked_preceding_low_run'] += int(newly_masked[state['low_run_start']:ix + 1].any())
        for n in BINS:
            for phase in range(n):
                _, held = core.display_at(original, ix, n, phase)
                counts['event_phases_with_masked_active_sample'] += int(held >= 0 and newly_masked[held])
        if state['reference'] >= 65:
            continue
        available = (np.arange(len(original)) + 1) * 10
        after = np.flatnonzero((available > t + 1e-09) & (available <= t + 300 + 1e-09))
        counts['events_with_masked_post_support'] += int(newly_masked[after].any())
        for n in BINS:
            for phase in range(n):
                updates = np.where((np.arange(len(original)) % n == phase) & np.isfinite(original), np.arange(len(original)), -1)
                held = np.maximum.accumulate(updates)[after]
                counts['post_phase_bins_with_masked_active_sample'] += int(newly_masked[held[held >= 0]].sum())
    return counts

def evaluate_events(events, art, meta, minimum_time=300, require_low_sec=0, first_subject=False):
    selected = events[events.case_id.isin(art)].copy()
    exclusions = {'excluded_policy_case': len(events) - len(selected), 'before_initial_support': 0, 'pre_reference_quality': 0, 'insufficient_preceding_low': 0, 'not_first_low_subject_event': 0}
    if selected.empty:
        return (empty_events(), exclusions)
    frame, base_excluded = frozen.evaluate(selected, art, meta, minimum_time)
    exclusions.update(base_excluded)
    if frame.empty:
        return (empty_events(), exclusions)
    if require_low_sec:
        remove = frame.low & (frame.pre_low_sec < require_low_sec)
        exclusions['insufficient_preceding_low'] = int(remove.sum())
        frame = frame[~remove].copy()
    if first_subject:
        low = frame[frame.low].sort_values(['case_id', 'time_sec']).drop_duplicates('subjectid')
        exclusions['not_first_low_subject_event'] = int(frame.low.sum()) - len(low)
        frame = pd.concat([frame[~frame.low], low], ignore_index=True)
    return (frame.reindex(columns=EVENT_COLUMNS), exclusions)

def post_support(values, time_sec):
    available = (np.arange(len(values)) + 1) * 10.0
    after = np.flatnonzero((available > time_sec + 1e-09) & (available <= time_sec + 300 + 1e-09))
    valid = int(np.isfinite(values[after]).sum())
    full = bool(len(available) and available[-1] >= time_sec + 300 and (len(after) == 30))
    return {'post_bins': len(after), 'post_valid_bins': valid, 'post_valid_fraction': valid / len(after) if len(after) else None, 'status': 'incomplete_post_interval' if not full else 'insufficient_post_reference' if valid < 24 else 'supported'}

def recover_events(frame, art):
    low = frame[frame.low]
    records, phase_reasons, audits = ([], [], [])
    supported = 0
    for event in low.to_dict('records'):
        values, t = (art[event['case_id']], event['time_sec'])
        support = post_support(values, t)
        audit = {k: event[k] for k in ['case_id', 'subjectid', 'time_sec']}
        audit.update(support)
        test = core.recovery_summary(values, t, 30, 0)
        if support['status'] != 'supported':
            assert test is None
            audits.append(audit)
            continue
        assert test is not None
        supported += 1
        if test['recovery_index'] < 0:
            audit['status'] = 'no_confirmed_recovery'
            audits.append(audit)
            continue
        audit['status'] = 'confirmed_recovery'
        audits.append(audit)
        record = {k: event[k] for k in ['case_id', 'subjectid', 'time_sec']}
        record['recovery_delay_sec'] = (test['recovery_index'] + 1) * 10 - t
        for n in BINS:
            phases = [core.recovery_summary(values, t, n, p) for p in range(n)]
            assert all((x is not None and x['recovery_index'] == test['recovery_index'] for x in phases))
            record[f'persistence_{n}'] = float(np.mean([x['persistence_min'] for x in phases]))
            phase_reasons.extend(({'interval_min': n / 6, 'stop_reason': x['stop_reason']} for x in phases))
        records.append(record)
    recovery = pd.DataFrame(records, columns=RECOVERY_COLUMNS)
    results = []
    for n in BINS:
        col = f'persistence_{n}'
        ci = frozen.cluster_mean(recovery, [col])[col]
        results.append({'interval_min': n / 6, 'eligible_low_events': len(low), 'full_post_support_events': supported, 'confirmed_recovery_events': len(recovery), 'subjects': int(recovery.subjectid.nunique()), 'cases': int(recovery.case_id.nunique()), **ci, 'median_min': float(recovery[col].median()) if len(recovery) else None, 'q1_min': float(recovery[col].quantile(0.25)) if len(recovery) else None, 'q3_min': float(recovery[col].quantile(0.75)) if len(recovery) else None})
    reasons = pd.DataFrame(phase_reasons, columns=['interval_min', 'stop_reason'])
    reasons = reasons.value_counts().reset_index(name='phase_events') if len(reasons) else pd.DataFrame(columns=['interval_min', 'stop_reason', 'phase_events'])
    audit = pd.DataFrame(audits, columns=['case_id', 'subjectid', 'time_sec', 'post_bins', 'post_valid_bins', 'post_valid_fraction', 'status'])
    return (recovery, results, reasons, audit)

def assert_frames_match(actual, expected, keys):
    left = actual[list(expected.columns)].sort_values(keys, kind='stable').reset_index(drop=True)
    right = expected.sort_values(keys, kind='stable').reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=0, atol=1e-10)

def independent_verify(frame, recovery, art):
    """Reconstruct held displays with cumulative indices, independent of core loops."""
    state_checks = 0
    for row in frame.to_dict('records'):
        values, t = (art[row['case_id']], row['time_sec'])
        available = (np.arange(len(values)) + 1) * 10
        ix = np.flatnonzero(available <= t + 1e-09)[-1]
        assert np.isfinite(values[ix]) and np.isfinite(values[ix - 5:ix + 1]).sum() >= 5
        assert values[ix] == row['reference']
        for n in BINS:
            counts = dict.fromkeys(frozen.CATEGORIES, 0)
            for phase in range(n):
                if row['low']:
                    sampled = np.flatnonzero((np.arange(ix + 1) % n == phase) & np.isfinite(values[:ix + 1]))
                    breaks = np.flatnonzero(~(np.isfinite(values[:ix + 1]) & (values[:ix + 1] < 65)))
                    start = int(breaks[-1]) + 1 if len(breaks) else 0
                    held = int(sampled[-1]) if len(sampled) else -1
                    category = 'unavailable' if held < 0 else 'normal_display' if values[held] >= 65 else 'fresh_low' if held >= start else 'inherited_low'
                    counts[category] += 1
                state_checks += 1
            for category, count in counts.items():
                np.testing.assert_allclose(row[f'{n}_{category}'], count / n, rtol=0, atol=1e-12)
    for row in recovery.to_dict('records'):
        values, t = (art[row['case_id']], row['time_sec'])
        available = (np.arange(len(values)) + 1) * 10
        window = np.flatnonzero((available > t + 1e-09) & (available <= t + 300 + 1e-09))
        assert len(window) == 30 and available[-1] >= t + 300 and (np.isfinite(values[window]).sum() >= 24)
        normal = np.isfinite(values) & (values >= 65)
        candidates = window[1:][normal[window[1:]] & normal[window[1:] - 1]]
        first = int(candidates[0])
        np.testing.assert_allclose(row['recovery_delay_sec'], available[first] - t, rtol=0, atol=1e-08)
        for n in BINS:
            durations = []
            for phase in range(n):
                updates = np.where((np.arange(len(values)) % n == phase) & np.isfinite(values), np.arange(len(values)), -1)
                held = np.maximum.accumulate(updates)
                display_low = (held >= 0) & (values[np.maximum(held, 0)] < 65)
                follow = np.flatnonzero((available >= available[first]) & (available < t + 300 - 1e-09))
                valid = normal[follow] & display_low[follow]
                failure = np.flatnonzero(~valid)
                stop = int(failure[0]) if len(failure) else len(follow)
                durations.append(np.minimum(10, t + 300 - available[follow[:stop]]).sum() / 60)
            np.testing.assert_allclose(row[f'persistence_{n}'], np.mean(durations), rtol=0, atol=1e-12)
    return {'event_phase_checks': state_checks, 'recovery_events_checked': len(recovery), 'status': 'PASS'}

def frozen_paths():
    files = [DATA / 'vitaldb_manifest.parquet', DATA / 'artmap_10s.parquet', FROZEN / 'code/run_analysis.py', FROZEN / 'code/treatment_core.py', FROZEN / 'data_restricted/raw_record_changes.csv', FROZEN / 'data_restricted/stable_record_changes.csv']
    files += sorted((FROZEN / 'analysis/aggregate').glob('*.csv'))
    files += sorted((FROZEN / 'data_restricted').glob('scenario_*.csv'))
    files += [FROZEN / 'data_restricted/primary_events.csv', FROZEN / 'data_restricted/recovery_events.csv', FROZEN / 'analysis/aggregate/summary.json']
    return files

def select_cohort(cohort, meta):
    if 'case_id' not in cohort or cohort.case_id.isna().any() or cohort.case_id.duplicated().any():
        raise ValueError('Policy cohort must contain unique nonmissing case_id rows')
    if set(cohort.case_id) - set(meta.index):
        raise ValueError('Policy cohort contains cases without frozen metadata')
    if 'included' not in cohort or cohort.included.isna().any():
        raise ValueError('Policy cohort requires explicit nonmissing inclusion flags')
    flags = cohort.included.map({True: True, False: False, 'True': True, 'False': False})
    if flags.isna().any():
        raise ValueError('Unknown policy cohort inclusion flag')
    for column in ['subjectid', 'subject_id']:
        if column in cohort:
            np.testing.assert_array_equal(cohort[column].to_numpy(), meta.loc[cohort.case_id, 'subjectid'].to_numpy())
    return set(cohort.loc[flags, 'case_id'])

def load_policy(policy, meta, base, allow_frozen_original):
    panel_path = OUT / f'data_restricted/panel_{policy}.parquet'
    cohort_path = OUT / f'data_restricted/cohort_{policy}.csv'
    if policy == 'original' and allow_frozen_original and (not panel_path.exists()):
        return (base, [DATA / 'artmap_10s.parquet'], {'source': 'frozen_original_fallback'})
    if not panel_path.exists() or not cohort_path.exists():
        raise FileNotFoundError(f'Policy inputs are not ready: {policy}')
    cohort = pd.read_csv(cohort_path)
    included = select_cohort(cohort, meta)
    panel = pd.read_parquet(panel_path)
    art = panel_to_art(panel, meta, included)
    if set(panel.case_id) != included:
        raise ValueError('Panel and cohort case sets differ; explicit inclusion schema is required')
    if policy in ['qc60', 'qc_all']:
        retained = cohort[cohort.case_id.isin(included)]
        assert (retained.valid_coverage >= 0.8).all()
    return (art, [panel_path, cohort_path], {'source': 'parent_policy_panel_and_cohort'})

def run_policy(policy, art, meta, raw, stable):
    display_rows, recovery_rows, flow_rows, strata_rows, all_reasons = ([], [], [], [], [])
    restricted, verification = ({}, {})
    for scenario, slug, source, merge, minimum, pre_low, first in SCENARIOS:
        changes = stable if source == 'stable' else raw
        events = frozen.build_events(changes, merge)
        frame, excluded = evaluate_events(events, art, meta, minimum, pre_low, first)
        low = frame[frame.low]
        display_rows.extend(frozen.summaries(frame, scenario))
        rec, results, reasons, post_audit = recover_events(frame, art)
        recovery_rows.extend(({'scenario': scenario, **row} for row in results))
        reasons.insert(0, 'scenario', scenario)
        all_reasons.append(reasons)
        support_counts = post_audit.status.value_counts().to_dict()
        flow = {'Policy cohort cases': len(art), 'Policy cohort subjects': int(meta.loc[list(art), 'subjectid'].nunique()), 'Frozen raw changes': len(raw), 'Frozen stable changes': len(stable), 'Frozen adjustment blocks': len(events), 'Retained adjustment blocks': int(events.case_id.isin(art).sum()), **excluded, 'Quality-eligible blocks': len(frame), 'Quality-eligible cases': int(frame.case_id.nunique()), 'Quality-eligible subjects': int(frame.subjectid.nunique()), 'Low-reference blocks at adjustment': len(low), 'Low-reference cases': int(low.case_id.nunique()), 'Low-reference subjects': int(low.subjectid.nunique()), 'Normal-reference blocks': int((~frame.low).sum()), 'Low-reference blocks with full 5-min support': results[0]['full_post_support_events'], 'Blocks with confirmed reference recovery': len(rec), **{f'post_{k}': int(support_counts.get(k, 0)) for k in ['incomplete_post_interval', 'insufficient_post_reference', 'no_confirmed_recovery', 'confirmed_recovery']}}
        assert len(frame) + sum(excluded.values()) == len(events)
        assert sum(support_counts.values()) == len(low)
        flow_rows.extend(({'scenario': scenario, 'Metric': key, 'Count': value} for key, value in flow.items()))
        verification[scenario] = independent_verify(frame, rec, art)
        for suffix, result in [('events', frame), ('recovery', rec), ('post_support', post_audit)]:
            result.to_csv(OUT / f'data_restricted/treatment_{policy}_{slug}_{suffix}.csv', index=False)
        restricted[slug] = {'events': frame, 'recovery': rec, 'post_support': post_audit, 'flow': flow}
        if scenario == 'Primary':
            ci = frozen.cluster_mean(low, ['30_normal_display'], equal_subject=True)['30_normal_display']
            display_rows.append({'scenario': 'Equal subject weighting', 'interval_min': 5.0, 'category': 'normal_display', 'subjects': int(low.subjectid.nunique()), 'cases': int(low.case_id.nunique()), 'events': len(low), **{key: value * 100 if value is not None else None for key, value in ci.items()}})
            for variable in ['drug', 'kind']:
                for label, subset in frame.groupby(variable):
                    strata_rows.extend(({**row, 'group_variable': variable} for row in frozen.summaries(subset, label) if row['interval_min'] == 5))
        print(json.dumps({'policy': policy, 'scenario': scenario, 'events': len(frame), 'low': len(low), 'recovered': len(rec), 'verification': 'PASS'}), flush=True)
    outputs = {'display': pd.DataFrame(display_rows), 'recovery': pd.DataFrame(recovery_rows), 'flow': pd.DataFrame(flow_rows), 'strata': pd.DataFrame(strata_rows), 'recovery_stop_reasons': pd.concat(all_reasons, ignore_index=True)}
    for name, frame in outputs.items():
        frame.to_csv(OUT / f'analysis/aggregate/treatment_{name}_{policy}.csv', index=False)
    frozen.save_json(OUT / f'qa/treatment_{policy}_verification.json', verification)
    return (outputs, restricted)

def verify_baseline(outputs, restricted):
    checks = {}
    for scenario, slug, *_ in SCENARIOS:
        filename = 'primary_events.csv' if slug == 'primary' else 'scenario_' + scenario.replace(' ', '_') + '.csv'
        assert_frames_match(restricted[slug]['events'], pd.read_csv(FROZEN / 'data_restricted' / filename), ['case_id', 'time_sec'])
        checks[f'{slug}_event_records'] = 'PASS'
    expected = [('display', 'display_state_results.csv', ['scenario', 'interval_min', 'category']), ('strata', 'drug_kind_strata.csv', ['group_variable', 'scenario', 'interval_min', 'category']), ('recovery', 'recovery_results.csv', ['interval_min']), ('recovery_stop_reasons', 'recovery_stop_reasons.csv', ['interval_min', 'stop_reason'])]
    for key, filename, keys in expected:
        frame = outputs[key]
        if key in ['recovery', 'recovery_stop_reasons']:
            frame = frame[frame.scenario == 'Primary']
        assert_frames_match(frame, pd.read_csv(FROZEN / 'analysis/aggregate' / filename), keys)
        checks[key] = 'PASS'
    assert_frames_match(restricted['primary']['recovery'], pd.read_csv(FROZEN / 'data_restricted/recovery_events.csv'), ['case_id', 'time_sec'])
    checks['primary_recovery_records'] = 'PASS'
    flow = restricted['primary']['flow']
    assert (flow['Quality-eligible blocks'], flow['Low-reference blocks at adjustment'], flow['Blocks with confirmed reference recovery']) == (640, 236, 184)
    frozen_flow = json.loads((FROZEN / 'analysis/aggregate/summary.json').read_text())['flow']
    for metric in set(flow) & set(frozen_flow):
        assert flow[metric] == frozen_flow[metric]
    checks['frozen_primary_counts_640_236_184'] = 'PASS'
    checks['numeric_absolute_tolerance'] = 1e-10
    return checks

def principal_summary(outputs, flow):
    names = {'cohort_cases': 'Policy cohort cases', 'cohort_subjects': 'Policy cohort subjects', 'eligible_adjustment_events': 'Quality-eligible blocks', 'eligible_adjustment_cases': 'Quality-eligible cases', 'eligible_adjustment_subjects': 'Quality-eligible subjects', 'low_reference_events': 'Low-reference blocks at adjustment', 'low_reference_cases': 'Low-reference cases', 'low_reference_subjects': 'Low-reference subjects', 'normal_reference_events': 'Normal-reference blocks', 'full_5min_post_support_events': 'Low-reference blocks with full 5-min support', 'confirmed_recovery_events': 'Blocks with confirmed reference recovery'}
    result = {key: int(flow[value]) for key, value in names.items()}
    display, recovery = ({}, {})
    for row in outputs['display'].query('scenario == "Primary"').to_dict('records'):
        interval = f"{row['interval_min']:g}"
        display.setdefault(interval, {})[row['category']] = {'estimate_pct': row['estimate'], 'lower95_pct': row['lower95'], 'upper95_pct': row['upper95'], 'events': int(row['events']), 'cases': int(row['cases']), 'subjects': int(row['subjects'])}
    for row in outputs['recovery'].query('scenario == "Primary"').to_dict('records'):
        interval = f"{row['interval_min']:g}"
        recovery[interval] = {'mean_persistence_min': row['estimate'], 'lower95_min': row['lower95'], 'upper95_min': row['upper95'], 'median_min': row['median_min'], 'q1_min': row['q1_min'], 'q3_min': row['q3_min'], 'events': int(row['confirmed_recovery_events']), 'cases': int(row['cases']), 'subjects': int(row['subjects'])}
    result.update({'recovery_cases': recovery['5']['cases'], 'recovery_subjects': recovery['5']['subjects'], 'display_by_interval_min': display, 'recovery_by_interval_min': recovery})
    return result

def format_ci(estimate, lower, upper):
    if any((x is None or not np.isfinite(x) for x in [estimate, lower, upper])):
        return 'Not estimable'
    return f'{estimate:.2f} ({lower:.2f} to {upper:.2f})'

def publication_wide(policy_summaries):
    rows = []
    for policy, summary in policy_summaries.items():
        principal = summary['principal_numbers']
        row = {'policy': policy, 'manuscript_role': 'Primary' if policy == 'original' else 'Technical sensitivity'}
        row.update({key: value for key, value in principal.items() if key not in ['display_by_interval_min', 'recovery_by_interval_min']})
        for interval in ['1', '2.5', '5']:
            display = principal['display_by_interval_min'][interval]['normal_display']
            rec = principal['recovery_by_interval_min'][interval]
            row[f'normal_display_pct_at_{interval}min_95ci'] = format_ci(display['estimate_pct'], display['lower95_pct'], display['upper95_pct'])
            row[f'low_display_persistence_min_at_{interval}min_95ci'] = format_ci(rec['mean_persistence_min'], rec['lower95_min'], rec['upper95_min'])
        rows.append(row)
    return pd.DataFrame(rows)

def assemble_tables(all_outputs, policy_summaries):
    comparison, publication = ([], [])
    for policy, outputs in all_outputs.items():
        for scenario, slug, *_ in SCENARIOS:
            display = outputs['display'].query('scenario == @scenario and interval_min == 5 and category == "normal_display"').iloc[0]
            rec = outputs['recovery'].query('scenario == @scenario and interval_min == 5').iloc[0]
            row = {'policy': policy, 'scenario': scenario, 'cohort_cases': policy_summaries[policy]['mask_validation']['retained_cases'], 'low_events': int(display.events), 'low_subjects': int(display.subjects), 'low_cases': int(display.cases), 'normal_display_pct': display.estimate, 'normal_display_lower95': display.lower95, 'normal_display_upper95': display.upper95, 'full_post_support_events': int(rec.full_post_support_events), 'confirmed_recovery_events': int(rec.confirmed_recovery_events), 'recovery_subjects': int(rec.subjects), 'recovery_cases': int(rec.cases), 'persistence_min': rec.estimate, 'persistence_lower95': rec.lower95, 'persistence_upper95': rec.upper95}
            comparison.append(row)
        for row in outputs['display'].to_dict('records'):
            publication.append({'policy': policy, 'module': 'treatment_display', 'metric': row.pop('category'), 'unit': 'percent', **row})
        for row in outputs['recovery'].to_dict('records'):
            publication.append({'policy': policy, 'module': 'post_adjustment_recovery', 'metric': 'low_display_persistence', 'unit': 'minutes', **row})
    compare = pd.DataFrame(comparison)
    if 'original' in all_outputs:
        base = compare[compare.policy == 'original'].set_index('scenario')
        compare['normal_display_change_pp_vs_original'] = [r.normal_display_pct - base.loc[r.scenario, 'normal_display_pct'] for r in compare.itertuples()]
        compare['persistence_change_min_vs_original'] = [r.persistence_min - base.loc[r.scenario, 'persistence_min'] for r in compare.itertuples()]
    compare.to_csv(OUT / 'analysis/aggregate/treatment_policy_comparison.csv', index=False)
    pd.DataFrame(publication).to_csv(OUT / 'analysis/aggregate/treatment_publication_table.csv', index=False)
    publication_wide(policy_summaries).to_csv(OUT / 'analysis/aggregate/treatment_publication_wide.csv', index=False)
    return compare

def main():
    ap = argparse.ArgumentParser(description=__doc__, parents=[PATH_ARGS])
    ap.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    ap.add_argument('--allow-frozen-original', action='store_true', help='Baseline validation before parent panels arrive')
    args = ap.parse_args()
    if len(set(args.policies)) != len(args.policies):
        ap.error('Each policy must be requested once')
    for directory in ['analysis/aggregate', 'data_restricted', 'qa']:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    source_paths = frozen_paths() + [Path(__file__), Path(__file__).with_name('test_qc_treatment.py'), OUT / '方案.md']
    before = {str(path): frozen.sha(path) for path in source_paths}
    meta = pd.read_parquet(DATA / 'vitaldb_manifest.parquet').set_index('case_id')
    base = panel_to_art(pd.read_parquet(DATA / 'artmap_10s.parquet'), meta)
    assert len(base) == 2435
    raw = pd.read_csv(FROZEN / 'data_restricted/raw_record_changes.csv')
    stable = pd.read_csv(FROZEN / 'data_restricted/stable_record_changes.csv')
    assert len(raw) == 834 and len(stable) == 776
    assert set(raw.case_id) <= set(base) and set(stable.case_id) <= set(base)
    all_outputs, policy_summaries = ({}, {})
    all_restricted = {}
    baseline = None
    for policy in args.policies:
        art, paths, source = load_policy(policy, meta, base, args.allow_frozen_original)
        before.update({str(path): frozen.sha(path) for path in paths})
        mask_validation = validate_masked_art(art, base)
        if policy == 'original':
            assert len(art) == len(base) and mask_validation['newly_masked_complete_bins'] == 0
        if policy == 'startup10':
            assert set(art) == set(base), 'startup10 must retain the original cases'
        outputs, restricted = run_policy(policy, art, meta, raw, stable)
        all_outputs[policy] = outputs
        all_restricted[policy] = restricted
        if policy == 'original':
            baseline = verify_baseline(outputs, restricted)
            frozen.save_json(OUT / 'qa/treatment_baseline_reproduction.json', baseline)
        primary = outputs['display'].query('scenario == "Primary" and interval_min == 5 and category == "normal_display"').iloc[0]
        policy_summaries[policy] = {**source, 'mask_validation': mask_validation, 'primary_flow': restricted['primary']['flow'], 'principal_numbers': principal_summary(outputs, restricted['primary']['flow']), 'primary_normal_display_pct': primary.to_dict(), 'primary_recovery': outputs['recovery'].query('scenario == "Primary"').to_dict('records')}
        if policy in ['qc60', 'qc_all']:
            window_audit = {}
            for scenario, slug, source, merge, minimum, *_ in SCENARIOS:
                changes = stable if source == 'stable' else raw
                window_audit[scenario] = mask_window_audit(frozen.build_events(changes, merge), art, base, minimum)
            frozen.save_json(OUT / f'qa/treatment_{policy}_mask_window_audit.json', window_audit)
            policy_summaries[policy]['mask_window_audit'] = window_audit
        cohort_path = OUT / f'data_restricted/cohort_{policy}.csv'
        if cohort_path.exists():
            cohort = pd.read_csv(cohort_path)
            affected = set(cohort.loc[cohort.newly_masked_bins > 0, 'case_id'])
            removed = set(cohort.case_id) - set(art)
            baseline_events = pd.read_csv(FROZEN / 'data_restricted/primary_events.csv')
            policy_summaries[policy]['treatment_overlap_audit'] = {'affected_cases_with_frozen_raw_changes': len(affected & set(raw.case_id)), 'excluded_cases_with_frozen_raw_changes': len(removed & set(raw.case_id)), 'affected_cases_with_frozen_eligible_events': len(affected & set(baseline_events.case_id)), 'affected_cases_with_frozen_low_events': len(affected & set(baseline_events.loc[baseline_events.low, 'case_id']))}
    comparison = assemble_tables(all_outputs, policy_summaries)
    if 'original' in all_outputs:
        for policy in args.policies:
            policy_summaries[policy]['exact_aggregate_match_to_original_all_scenarios'] = {key: all_outputs[policy][key].equals(all_outputs['original'][key]) for key in ['display', 'recovery', 'strata', 'recovery_stop_reasons']}
            matches = {}
            for scenario, slug, *_ in SCENARIOS:
                matches[scenario] = {}
                for kind in ['events', 'recovery', 'post_support']:
                    actual = all_restricted[policy][slug][kind]
                    expected = all_restricted['original'][slug][kind]
                    keys = ['case_id', 'time_sec']
                    exact = actual.sort_values(keys).reset_index(drop=True).equals(expected.sort_values(keys).reset_index(drop=True))
                    matches[scenario][kind] = {'exact_match': exact, 'rows': len(actual), 'compared_fields': list(actual.columns)}
            frozen.save_json(OUT / f'qa/treatment_{policy}_event_record_comparison.json', matches)
            policy_summaries[policy]['all_event_and_recovery_fields_exactly_match_original'] = all((comparison['exact_match'] for scenario in matches.values() for comparison in scenario.values()))
    unchanged = {path: frozen.sha(Path(path)) == digest for path, digest in before.items()}
    assert all(unchanged.values()), 'An input changed during treatment reanalysis'
    frozen.save_json(OUT / 'qa/treatment_input_integrity.json', {'status': 'PASS', 'input_count': len(before), 'all_inputs_unchanged': True, 'sha256': before})
    summary = {'schema_version': 'treatment_qc_v1', 'status': 'TECHNICAL_QC_REANALYSIS_COMPLETE_FOR_REQUESTED_POLICIES', 'policies': policy_summaries, 'field_definitions': {'principal_numbers': 'Primary frozen treatment-event definition evaluated within the named reference policy', 'events': 'Merged recorded pump adjustment blocks, not individual patients or verified administrations', 'cases': 'Distinct operations', 'subjects': 'Distinct patients; bootstrap clustering unit', 'estimate_pct': 'Mean within-event proportion over all global sampling phases, multiplied by 100', 'display_denominator': 'Quality-eligible adjustment events with low preceding reference MAP', 'mean_persistence_min': 'Mean phase-averaged low-display persistence after confirmed reference recovery', 'recovery_denominator': 'Low-reference adjustment events with complete 5-min horizon, at least 80% finite reference, and confirmed recovery', 'ci': 'Two-sided percentile 95% CI from 2000 subject-cluster bootstrap replicates; fixed seed 20261004', 'wide_table': 'One row per policy; formatted estimates with 95% CI. JSON retains unrounded numeric values.'}, 'baseline_reproduction': baseline, 'seed': frozen.SEED, 'bootstrap_replicates': 2000, 'bootstrap_unit': 'subject', 'within_event_phases': 'all_global_phases_equally_weighted', 'raw_pump_transitions': 'REUSED_FROZEN_834_RAW_AND_776_STABLE_RECORDS_NOT_REREAD', 'six_original_event_sensitivities_rerun': True, 'recovery_sensitivity_extension': 'Also recomputed recovery for each of the six frozen event sensitivity definitions', 'reference_quality_adjudication': 'NOT_CLINICALLY_ADJUDICATED_TECHNICAL_CANDIDATES_ONLY', 'manuscript_primary_policy': 'original', 'qc_policy_role': 'Sensitivity analyses only pending clinical adjudication', 'clinical_notes': ['Recorded pump starts/increases are not verified administrations or treatment effects.', 'No clinician signoff is claimed.', 'Missing post support and no confirmed recovery are excluded, not assigned zero persistence.', 'QC masks preserve the frozen time zero, global sampling phases, and case durations.', 'The startup10 policy is distinct from the frozen Exclude first 10 min event sensitivity.'], 'publication': 'LOCAL_QC_CANDIDATE_TABLE_DATA_ONLY_NO_DOCUMENTS_CHANGED', 'all_inputs_unchanged': True}
    frozen.save_json(OUT / 'analysis/aggregate/treatment_summary.json', summary)
    print(comparison[comparison.scenario == 'Primary'].to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
