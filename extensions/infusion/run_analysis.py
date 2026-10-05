import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from treatment_core import transitions, block_starts, event_state, recovery_summary

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
CATEGORIES = ['normal_display', 'fresh_low', 'inherited_low', 'unavailable']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def cluster_mean(frame, columns, equal_subject=False):
    if frame.empty:
        return {c: {'estimate': None, 'lower95': None, 'upper95': None} for c in columns}
    g = frame.groupby('subjectid')[columns].agg(['sum', 'count', 'mean'])
    sums = np.column_stack([g[c]['sum'] for c in columns])
    counts = np.column_stack([g[c]['count'] for c in columns])
    if equal_subject:
        sums = sums / counts
        counts = np.ones_like(counts)
    estimate = sums.sum(0) / counts.sum(0)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(g), size=(2000, len(g)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    q = np.quantile(boot, [.025, .975], axis=0)
    return {c: {'estimate': float(estimate[i]), 'lower95': float(q[0, i]), 'upper95': float(q[1, i])}
            for i, c in enumerate(columns)}


def build_events(change_frame, merge_sec):
    rows = []
    for cid, group in change_frame.groupby('case_id', sort=True):
        group = group.sort_values(['time_sec', 'track'], kind='stable').reset_index(drop=True)
        anchors = block_starts(group.time_sec, merge_sec)
        for num, start in enumerate(anchors):
            end = anchors[num+1] if num+1 < len(anchors) else len(group)
            block = group.iloc[start:end]
            drugs = sorted(set(block.track.str.replace('Orchestra/', '', regex=False).str.replace('_RATE', '', regex=False)))
            kinds = set(block.kind)
            rows.append({'case_id': int(cid), 'time_sec': float(block.time_sec.iloc[0]),
                         'drug': '+'.join(drugs), 'kind': 'mixed' if len(kinds) > 1 else next(iter(kinds)),
                         'record_changes': len(block)})
    return pd.DataFrame(rows)


def evaluate(events, art, meta, minimum_time=300, require_low_sec=0, first_subject=False):
    out = []; excluded = {'before_initial_support': 0, 'pre_reference_quality': 0}
    for record in events.to_dict('records'):
        cid, t = record['case_id'], record['time_sec']
        values = art[cid]
        if t < minimum_time:
            excluded['before_initial_support'] += 1; continue
        base = event_state(values, t, 30, 0)
        if base is None:
            excluded['pre_reference_quality'] += 1; continue
        record.update({'subjectid': int(meta.loc[cid, 'subjectid']), 'reference': base['reference'],
                       'low': base['reference'] < 65, 'pre_low_sec': base['pre_low_sec']})
        for bins in [6, 15, 30]:
            states = [event_state(values, t, bins, phase) for phase in range(bins)]
            for category in CATEGORIES:
                record[f'{bins}_{category}'] = np.mean([s['category'] == category for s in states])
        if record['low']:
            assert np.isclose(sum(record[f'30_{c}'] for c in CATEGORIES), 1)
        if require_low_sec and record['low'] and record['pre_low_sec'] < require_low_sec:
            continue
        out.append(record)
    frame = pd.DataFrame(out)
    if first_subject and not frame.empty:
        normal = frame[~frame.low]
        low = frame[frame.low].sort_values(['case_id', 'time_sec']).drop_duplicates('subjectid')
        frame = pd.concat([normal, low], ignore_index=True)
    return frame, excluded


def summaries(frame, scenario):
    low = frame[frame.low]
    rows = []
    for bins in [6, 15, 30]:
        cols = [f'{bins}_{c}' for c in CATEGORIES]
        estimates = cluster_mean(low, cols)
        for cat in CATEGORIES:
            x = estimates[f'{bins}_{cat}']
            rows.append({'scenario': scenario, 'interval_min': bins/6, 'category': cat,
                         'subjects': low.subjectid.nunique(), 'cases': low.case_id.nunique(), 'events': len(low),
                         **{k: v*100 if v is not None else None for k, v in x.items()}})
    return rows


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--candidate', type=Path)
    ap.add_argument('--track-roots', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=Path(__file__).with_name('protocol.md'))
    args = ap.parse_args()
    ROOT=args.output.resolve()
    for name in ['analysis/aggregate', 'data_restricted', 'qa', '02_Figures', '03_Reproducibility', '01_修订稿件']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT/'data_restricted', 0o700)
    source = args.frozen/'outputs/intermediate'
    meta_path, art_path = source/'vitaldb_manifest.parquet', source/'artmap_10s.parquet'
    audit_path = args.audit/'data_restricted/observed_pump_transitions.csv'
    protected = [meta_path, art_path, audit_path]
    if args.candidate is not None:
        protected += list((args.candidate/'01_EJA_Manuscript').glob('*.docx'))
        protected += list((args.candidate/'03_Reproducibility').glob('*.json'))
        protected += list((args.candidate/'03_Reproducibility').glob('*.xlsx'))
    before = {str(p): sha(p) for p in protected}
    protocol = args.protocol
    frozen_protocol_sha = sha(protocol)
    meta = pd.read_parquet(meta_path).set_index('case_id')
    panel = pd.read_parquet(art_path)
    art = {}
    for cid, group in panel.groupby('case_id', sort=True):
        group = group.sort_values('time_sec')
        duration = float(meta.loc[cid, 'aneend'])-max(0., float(meta.loc[cid, 'anestart']))
        group = group[group.time_sec + 10 <= duration + 1e-9]
        assert np.allclose(group.time_sec, np.arange(len(group))*10)
        art[int(cid)] = group.art_map.to_numpy()
    assert len(art) == 2435
    inventory = pd.read_csv(args.track_roots[0]/'trks.csv')
    inventory = inventory[inventory.caseid.isin(art) & inventory.tname.isin(['Orchestra/NEPI_RATE', 'Orchestra/PHEN_RATE'])]
    source_hashes = {}; records = []; stable_records = []
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'audit/scripts'))
    from audit_core import clean_track
    for row in inventory.itertuples():
        filename = str(row.tid)+'.csv.gz'
        path = next((p/'tracks'/filename for p in args.track_roots if (p/'tracks'/filename).is_file()), None)
        if path is None:
            raise FileNotFoundError('Missing required pump source track')
        source_hashes[str(path)] = sha(path)
        cid = int(row.caseid)
        start = max(0., float(meta.loc[cid, 'anestart']))
        rate, q = clean_track(pd.read_csv(path, compression='gzip'), start, float(meta.loc[cid, 'aneend']))
        for sustain, dest in [(0, records), (5, stable_records)]:
            for event in transitions(rate.time_sec.to_numpy(), rate.value.to_numpy(), sustain):
                dest.append({'case_id': cid, 'track': row.tname, **event})
    raw, stable = pd.DataFrame(records), pd.DataFrame(stable_records)
    audited = pd.read_csv(audit_path)
    audited = audited[audited.track.isin(['Orchestra/NEPI_RATE', 'Orchestra/PHEN_RATE'])]
    keys = ['case_id', 'track', 'time_sec', 'kind']
    a, b = raw[keys].sort_values(keys).reset_index(drop=True), audited[keys].sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b) == 834
    assert a[['case_id', 'track', 'kind']].equals(b[['case_id', 'track', 'kind']])
    np.testing.assert_allclose(a.time_sec, b.time_sec, rtol=0, atol=1e-8)
    raw.to_csv(ROOT/'data_restricted/raw_record_changes.csv', index=False)
    stable.to_csv(ROOT/'data_restricted/stable_record_changes.csv', index=False)

    primary_events = build_events(stable, 60)
    main_frame, exclusions = evaluate(primary_events, art, meta)
    main_frame.to_csv(ROOT/'data_restricted/primary_events.csv', index=False)
    result_rows = summaries(main_frame, 'Primary')
    config = [('Merge 30 s', stable, 30, 300, 0, False), ('Merge 120 s', stable, 120, 300, 0, False),
              ('No stability requirement', raw, 60, 300, 0, False),
              ('At least 30 s preceding low MAP', stable, 60, 300, 30, False),
              ('First low event per subject', stable, 60, 300, 0, True),
              ('Exclude first 10 min', stable, 60, 600, 0, False)]
    for name, changes, merge, min_time, low_sec, first in config:
        f, _ = evaluate(build_events(changes, merge), art, meta, min_time, low_sec, first)
        f.to_csv(ROOT/'data_restricted'/('scenario_'+name.replace(' ', '_')+'.csv'), index=False)
        result_rows.extend(summaries(f, name))
    low = main_frame[main_frame.low].copy()
    equal = cluster_mean(low, ['30_normal_display'], equal_subject=True)['30_normal_display']
    result_rows.append({'scenario': 'Equal subject weighting', 'interval_min': 5., 'category': 'normal_display',
                        'subjects': low.subjectid.nunique(), 'cases': low.case_id.nunique(), 'events': len(low),
                        **{k: v*100 if v is not None else None for k, v in equal.items()}})
    estimates = pd.DataFrame(result_rows)
    estimates.to_csv(ROOT/'analysis/aggregate/display_state_results.csv', index=False)
    stratified = []
    for group_name in ['drug', 'kind']:
        for label, f in main_frame.groupby(group_name):
            for r in summaries(f, label):
                if r['interval_min'] == 5:
                    r['group_variable'] = group_name; stratified.append(r)
    pd.DataFrame(stratified).to_csv(ROOT/'analysis/aggregate/drug_kind_strata.csv', index=False)

    rec_records = []; complete_post = 0; reasons = []
    for r in low.to_dict('records'):
        values = art[r['case_id']]
        test = recovery_summary(values, r['time_sec'], 30, 0)
        if test is None:
            continue
        complete_post += 1
        if test['recovery_index'] < 0:
            continue
        rec = {'case_id': r['case_id'], 'subjectid': r['subjectid'], 'time_sec': r['time_sec'],
               'recovery_delay_sec': (test['recovery_index']+1)*10-r['time_sec']}
        for bins in [6, 15, 30]:
            all_recovery = [recovery_summary(values, r['time_sec'], bins, p) for p in range(bins)]
            rec[f'persistence_{bins}'] = np.mean([x['persistence_min'] for x in all_recovery])
            reasons.extend({'interval_min': bins/6, 'stop_reason': x['stop_reason']} for x in all_recovery)
        rec_records.append(rec)
    recovery = pd.DataFrame(rec_records)
    recovery.to_csv(ROOT/'data_restricted/recovery_events.csv', index=False)
    rec_rows = []
    for bins in [6, 15, 30]:
        col = f'persistence_{bins}'
        ci = cluster_mean(recovery, [col])[col]
        rec_rows.append({'interval_min': bins/6, 'eligible_low_events': len(low), 'full_post_support_events': complete_post,
                         'confirmed_recovery_events': len(recovery), 'subjects': recovery.subjectid.nunique(),
                         'cases': recovery.case_id.nunique(), **ci,
                         'median_min': float(recovery[col].median()),
                         'q1_min': float(recovery[col].quantile(.25)), 'q3_min': float(recovery[col].quantile(.75))})
    pd.DataFrame(rec_rows).to_csv(ROOT/'analysis/aggregate/recovery_results.csv', index=False)
    pd.DataFrame(reasons).value_counts().reset_index(name='phase_events').to_csv(ROOT/'analysis/aggregate/recovery_stop_reasons.csv', index=False)
    flow = [
        ['Fixed primary cohort cases', 2435], ['Fixed primary cohort subjects', 2380],
        ['Cases with either target pump track', raw.case_id.nunique()], ['Raw recorded increases or starts', len(raw)],
        ['Changes passing 5-s stability', len(stable)], ['Anchored 60-s adjustment blocks', len(primary_events)],
        ['Excluded before 5 min', exclusions['before_initial_support']],
        ['Excluded for preceding reference quality', exclusions['pre_reference_quality']],
        ['Quality-eligible blocks', len(main_frame)], ['Quality-eligible cases', main_frame.case_id.nunique()],
        ['Quality-eligible subjects', main_frame.subjectid.nunique()],
        ['Low-reference blocks at adjustment', len(low)], ['Low-reference cases', low.case_id.nunique()],
        ['Low-reference subjects', low.subjectid.nunique()], ['Normal-reference blocks', int((~main_frame.low).sum())],
        ['Low-reference blocks with full 5-min support', complete_post], ['Blocks with confirmed reference recovery', len(recovery)],
    ]
    pd.DataFrame(flow, columns=['Metric', 'Count']).to_csv(ROOT/'analysis/aggregate/event_flow.csv', index=False)
    primary = estimates[(estimates.scenario == 'Primary') & (estimates.interval_min == 5) &
                        (estimates.category == 'normal_display')].iloc[0].to_dict()
    precision_gate = primary['subjects'] >= 30 and primary['upper95'] - primary['lower95'] <= 20
    summary = {'protocol_sha256_before_estimation': frozen_protocol_sha,
               'seed': SEED, 'bootstrap_replicates': 2000, 'raw_reconciliation': 'PASS',
               'flow': dict(flow), 'primary': primary, 'main_text_gate': bool(precision_gate),
               'reference_quality_adjudication': 'UNRESOLVED_NOT_CERTIFIED_BY_THIS_MODULE',
               'publication': 'LOCAL_CANDIDATE_ONLY'}
    save_json(ROOT/'analysis/aggregate/summary.json', summary)
    combined = {**before, **source_hashes}
    checks = {p: sha(Path(p)) == digest for p, digest in combined.items()}
    assert all(checks.values()) and sha(protocol) == frozen_protocol_sha
    save_json(ROOT/'data_restricted/source_fingerprints.json', combined)
    save_json(ROOT/'qa/analysis_integrity.json', {'protected_files': len(before), 'raw_pump_files': len(source_hashes),
              'all_unchanged': all(checks.values()), 'protocol_unchanged': True, 'raw_reconciled_records': len(raw)})
    print(json.dumps(summary, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), flush=True)


if __name__ == '__main__':
    main()
