"""Independent minute-record verification; never imports the shared estimator.

Run with bundled Python and -B. Writes aggregate QA JSON only. Exits 0 on PASS,
1 on FAIL, 2 on PENDING_INPUTS; never waits for source inputs.
Numerical agreement is not human adjudication or clinical validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_writebytecode = True
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ['reference_auc', 'display_auc', 'hidden_auc', 'overdisplay_auc',
        'hidden_unavailable_auc', 'reference_low_min', 'display_low_min',
        'low_reference_normal_display_min', 'normal_reference_low_display_min',
        'low_reference_unavailable_min', 'valid_reference_min', 'display_available_min',
        'selected_minutes'] + [f'event{n}_{k}' for n in (3, 5) for k in
        ('episodes', 'fresh', 'inherited_only', 'never_low', 'visible', 'delay_sum_min')]
INTERVALS = (1, 3, 5, 10)
THRESHOLDS = (65, 60, 55)
ATOL, RTOL = 1e-7, 1e-9
SEED, REPLICATES = 20260706, 1000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_report(path, report):
    path = Path(path)
    if path.parent.name != 'qa' or not path.name.startswith('independent') or path.suffix != '.json':
        raise ValueError('Only qa/independent*.json may be written')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def range_clean(values):
    x = np.asarray(values, dtype=float).copy()
    x[~np.isfinite(x) | (x < 20) | (x > 180)] = np.nan
    return x


def minute_reference(ten_seconds, duration_seconds):
    if not np.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError('Invalid actual duration')
    complete = int(duration_seconds // 60)
    x = range_clean(ten_seconds)
    if len(x) < complete * 6:
        raise ValueError('Original grid is shorter than complete-duration minutes')
    blocks = x[:complete * 6].reshape(complete, 6)
    result = np.full(complete, np.nan)
    accepted = np.count_nonzero(np.isfinite(blocks), axis=1) >= 5
    if accepted.any():
        result[accepted] = np.nanmedian(blocks[accepted], axis=1)
    return result


def hour_selection(values):
    finite = np.isfinite(values)
    prefix = np.concatenate(([0], np.cumsum(finite)))
    starts = np.arange(0, len(values) - 59, 60)
    accepted = starts[(prefix[starts + 60] - prefix[starts]) >= 48]
    changes = np.zeros(len(values) + 1, dtype=np.int64)
    changes[accepted] += 1
    changes[accepted + 60] -= 1
    return np.cumsum(changes[:-1]) > 0


def eligible(record):
    return bool(record['age_bin'] >= 25 and hour_selection(record['map']).any())


def held_display(values, interval, phase):
    x = np.asarray(values, float)
    scheduled = np.arange(phase, len(x), interval)
    updates = scheduled[np.isfinite(x[scheduled])]
    display = np.full(len(x), np.nan)
    origin = np.full(len(x), -1, dtype=np.int64)
    if len(updates):
        position = np.searchsorted(updates, np.arange(len(x)), side='right') - 1
        available = position >= 0
        origin[available] = updates[position[available]]
        display[available] = x[origin[available]]
    return display, origin


def low_episodes(values, selected, threshold, minimum):
    x = np.asarray(values, float)
    indices = np.flatnonzero(selected & np.isfinite(x) & (x < threshold))
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    first = indices[np.r_[0, cuts]]
    last = indices[np.r_[cuts - 1, len(indices) - 1]] + 1
    return [(int(a), int(b)) for a, b in zip(first, last) if b - a >= minimum]


def assert_components(c):
    c = np.asarray(c, float)
    if not np.isfinite(c).all() or (c < -ATOL).any():
        raise AssertionError('Nonfinite or negative component')
    if not np.isclose(c[0] + c[3], c[1] + c[2], atol=ATOL, rtol=RTOL):
        raise AssertionError('Reference + overdisplay != display + hidden')
    for start in (13, 19):
        events, fresh, inherited, never, visible, delay = c[start:start + 6]
        if not np.isclose(events, fresh + inherited + never, atol=ATOL, rtol=RTOL):
            raise AssertionError('Fresh + inherited + never != events')
        if not np.isclose(visible, fresh + inherited, atol=ATOL, rtol=RTOL):
            raise AssertionError('Fresh + inherited != visible')
        if visible > events + ATOL or (visible == 0 and delay != 0):
            raise AssertionError('Event visibility/delay invalid')
    if c[4] > c[2] + ATOL or c[11] > c[10] + ATOL or c[10] > c[12] + ATOL:
        raise AssertionError('Minute or unavailable burden bounds violated')
    if not np.isclose(c[5], c[6] + c[7] + c[9] - c[8], atol=ATOL, rtol=RTOL):
        raise AssertionError('Low-minute partition violated')
    if c[19] > c[13] + ATOL:
        raise AssertionError('Five-minute episodes exceed three-minute episodes')


def phase_components(values, selected, threshold, interval, phase, episodes=None):
    x = np.asarray(values, float)
    selected = np.asarray(selected, bool)
    display, origin = held_display(x, interval, phase)
    valid = np.flatnonzero(selected & np.isfinite(x))
    reference = np.clip(threshold - x[valid], 0, None)
    has_display = np.isfinite(display[valid])
    displayed = np.zeros(len(valid))
    displayed[has_display] = np.clip(threshold - display[valid[has_display]], 0, None)
    difference = reference - displayed
    reference_low = reference > 0
    display_low = displayed > 0
    c = [reference.sum(), displayed.sum(), difference[difference > 0].sum(),
         -difference[difference < 0].sum(), reference[~has_display].sum(),
         reference_low.sum(), display_low.sum(),
         (reference_low & has_display & ~display_low).sum(),
         (~reference_low & display_low).sum(), (reference_low & ~has_display).sum(),
         len(valid), has_display.sum(), selected.sum()]
    low_times = np.flatnonzero(np.isfinite(display) & (display < threshold))
    fresh_times = low_times[origin[low_times] == low_times]
    for minimum in (3, 5):
        spans = episodes[minimum] if episodes is not None else low_episodes(x, selected, threshold, minimum)
        totals = np.zeros(6)
        if spans:
            bounds = np.asarray(spans)
            a, b = bounds[:, 0], bounds[:, 1]
            first_low = np.searchsorted(low_times, a)
            end_low = np.searchsorted(low_times, b)
            visible = end_low > first_low
            fresh = np.searchsorted(fresh_times, b) > np.searchsorted(fresh_times, a)
            totals[:] = [len(spans), fresh.sum(), (visible & ~fresh).sum(),
                         (~visible).sum(), visible.sum(),
                         (low_times[first_low[visible]] - a[visible]).sum()]
            if interval <= minimum and not fresh.all():
                raise AssertionError('Complete observed event lacks scheduled fresh low')
        c.extend(totals)
    c = np.asarray(c, float)
    assert_components(c)
    return c


def case_components(values, selected, threshold=65, interval=5):
    episodes = {n: low_episodes(values, selected, threshold, n) for n in (3, 5)}
    total = np.zeros(len(COLS))
    for phase in range(interval):
        total += phase_components(values, selected, threshold, interval, phase, episodes)
    averaged = total / interval
    assert_components(averaged)
    return averaged


def metrics(components):
    x = np.asarray(components, float)
    output = {name: x[..., j] for j, name in enumerate(COLS)}
    def quotient(a, b):
        a, b = np.broadcast_arrays(a, b)
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    r, d, h, o = (output[k] for k in COLS[:4])
    output.update(hidden_ratio=quotient(h, r), overdisplay_ratio=quotient(o, r),
                  net_ratio=quotient(d-r, r), absolute_disagreement_ratio=quotient(h+o, r),
                  cancellation=1-quotient(abs(d-r), h+o),
                  reference_twa=quotient(r, output['valid_reference_min']),
                  display_twa=quotient(d, output['valid_reference_min']),
                  low_reference_normal_display_min_per_hour=quotient(60*output[COLS[7]], output['selected_minutes']),
                  normal_reference_low_display_min_per_hour=quotient(60*output[COLS[8]], output['selected_minutes']))
    for n in (3, 5):
        prefix = f'event{n}_'
        for kind in ('fresh', 'inherited_only', 'never_low', 'visible'):
            output[prefix+kind+'_probability'] = quotient(output[prefix+kind], output[prefix+'episodes'])
        output[prefix+'mean_display_delay_min'] = quotient(output[prefix+'delay_sum_min'], output[prefix+'visible'])
    return output


def primary_summary(frame):
    rows = []
    for source, group in frame[frame.scenario == 'primary'].groupby('source', sort=True):
        patients = np.sort(group.subject_id.unique())
        size = len(patients)
        weights = np.random.default_rng(SEED).multinomial(size, np.ones(size)/size, size=REPLICATES)
        for (threshold, interval), records in group.groupby(['threshold', 'interval_min'], sort=True):
            patient_sums = np.zeros((size, len(COLS)))
            # Inputs are already phase-averaged admission components, not ratios.
            locations = np.searchsorted(patients, records.subject_id.to_numpy())
            np.add.at(patient_sums, locations, records[COLS].to_numpy(float))
            point = metrics(patient_sums.sum(axis=0))
            boot = metrics(weights @ patient_sums)
            row = dict(source=source, scenario='primary', threshold=int(threshold), interval_min=int(interval),
                       n_records=len(records), n_subjects=size, number_of_phases=int(interval),
                       bootstrap_replicates=REPLICATES,
                       n_records_with_3min_events=int((records.event3_episodes > 0).sum()),
                       n_records_with_5min_events=int((records.event5_episodes > 0).sum()))
            for name, value in point.items():
                defined = boot[name][np.isfinite(boot[name])]
                bounds = np.percentile(defined, [2.5, 97.5]) if len(defined) else [np.nan, np.nan]
                row[name] = float(value)
                row[name+'_ci_low'], row[name+'_ci_high'] = map(float, bounds)
            rows.append(row)
    return pd.DataFrame(rows)


class NumericAudit:
    def __init__(self):
        self.checked = 0
        self.failures = 0
        self.maximum_absolute_difference = 0.

    def add(self, actual, expected):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape:
            raise ValueError('Comparison shape mismatch')
        match = np.isclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True)
        self.checked += a.size
        self.failures += int((~match).sum())
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            self.maximum_absolute_difference = max(self.maximum_absolute_difference, float(np.max(abs(a[finite]-b[finite]))))

    def report(self):
        return dict(checked_numeric_values=self.checked, mismatched_numeric_values=self.failures,
                    maximum_absolute_difference=self.maximum_absolute_difference)


def vital_records(root):
    sys.path.insert(0, str(root/'private/deps'))
    import pyarrow.parquet as pq
    source = root.parent/'analysis_v10_subject_phase_release/outputs/intermediate'
    meta = pq.read_table(source/'vitaldb_manifest.parquet',
                         columns=['case_id', 'subjectid', 'age', 'primary_start_sec', 'primary_end_sec']).to_pandas().set_index('case_id')
    if not meta.index.is_unique:
        raise ValueError('Duplicate VitalDB metadata case')
    panel = pq.read_table(source/'artmap_10s.parquet', columns=['case_id', 'time_sec', 'art_map']).to_pandas()
    for case_id, case in panel.groupby('case_id', sort=True):
        case = case.sort_values('time_sec')
        times = case.time_sec.to_numpy(float)
        if not len(times) or times[0] != 0 or not np.allclose(times, np.arange(len(times))*10, atol=1e-7, rtol=0):
            raise ValueError('VitalDB original time grid is not zero-based ten-second native grid')
        row = meta.loc[case_id]
        duration = float(row.primary_end_sec - row.primary_start_sec)
        yield dict(case_id=int(case_id), subject_id=int(row.subjectid), age_bin=float(row.age),
                   map=minute_reference(case.art_map.to_numpy(float), duration))


def npz_records(data):
    case_ids = data['case_id']
    subjects, ages, offsets, values = (data[k] for k in ('subject_id', 'age_bin', 'index', 'map'))
    if (any(np.asarray(a).ndim != 1 for a in (case_ids, subjects, ages, offsets, values))
            or len(subjects) != len(case_ids) or len(ages) != len(case_ids) or len(offsets) != len(case_ids)+1
            or offsets[0] != 0 or offsets[-1] != len(values) or (np.diff(offsets) < 0).any()
            or not np.issubdtype(offsets.dtype, np.integer)):
        raise ValueError('Malformed SICdb NPZ offsets or record arrays')
    for j in range(len(case_ids)):
        yield dict(case_id=int(case_ids[j]), subject_id=int(subjects[j]), age_bin=float(ages[j]),
                   map=range_clean(values[offsets[j]:offsets[j+1]]))


def sic_records(root):
    base = root/'private/sicdb'
    ready = json.loads((base/'ready.json').read_text())
    if str(ready['status']).lower() not in ('ready', 'complete', 'pass'):
        raise ValueError('SICdb extraction is not ready')
    for entry in ready['shards']:
        path = base/(entry['path'] if isinstance(entry, dict) else entry)
        if isinstance(entry, dict) and digest(path) != entry['sha256']:
            raise ValueError('SICdb shard digest mismatch')
        with np.load(path, allow_pickle=False) as data:
            yield from npz_records(data)


def contract_check(root):
    c = json.loads((root/'analysis_contract.json').read_text())
    required = {'primary_age_cutoff':25, 'map_range_mmHg':[20,180],
                'vitaldb_minute_median_minimum_valid_10s_bins':5, 'window_minutes':60,
                'minimum_valid_minutes_per_window':48, 'interval_minutes':list(INTERVALS),
                'thresholds_mmHg':list(THRESHOLDS), 'event_consecutive_low_minutes':[3,5],
                'event_bridge_normal_or_missing_minutes':0}
    for key, value in required.items():
        if c.get(key) != value:
            raise ValueError('Unsupported analysis contract: '+key)
    bootstrap = c['bootstrap']
    if (bootstrap['cluster'] != 'patient' or bootstrap['replicates'] != REPLICATES
            or bootstrap['seed'] != SEED or bootstrap['interval'] != 'percentile'
            or bootstrap['average_phases_before_patient_sum'] is not True):
        raise ValueError('Unsupported bootstrap contract')


def verify_source(root, source, sample_size):
    frame = pd.read_csv(root/f'private/{source}_components.csv.gz')
    frame = frame[frame.scenario == 'primary'].copy()
    if frame.empty or not frame.source.eq(source).all() or frame.duplicated(['case_id','threshold','interval_min']).any():
        raise ValueError('Empty or duplicate primary component rows')
    if not frame.groupby('case_id').subject_id.nunique().eq(1).all():
        raise ValueError('Subject linkage differs across groups')
    table = frame[COLS].to_numpy(float)
    if not np.isfinite(table).all() or (table < -ATOL).any():
        raise ValueError('Main component table contains negative or nonfinite values')
    if not np.allclose(table[:,0]+table[:,3], table[:,1]+table[:,2], atol=ATOL, rtol=RTOL):
        raise ValueError('Main component table AUC identity failure')
    for j in (13,19):
        if (not np.allclose(table[:,j], table[:,j+1:j+4].sum(axis=1), atol=ATOL, rtol=RTOL)
                or not np.allclose(table[:,j+4], table[:,j+1:j+3].sum(axis=1), atol=ATOL, rtol=RTOL)):
            raise ValueError('Main component table event partition failure')
    expected = frame[(frame.threshold == 65) & (frame.interval_min == 5)].set_index('case_id')
    groups = set(zip(frame.threshold, frame.interval_min))
    if groups != {(t,k) for t in THRESHOLDS for k in INTERVALS} or len(frame) != len(expected)*12:
        raise ValueError('Source primary table lacks the 12 full groups')
    if not frame.groupby('case_id').size().eq(12).all():
        raise ValueError('Some source cases lack full 12 groups')
    sample = set(sorted(expected.index, key=lambda cid: hashlib.sha256(f'{source}:{int(cid)}:{SEED}'.encode()).digest())[:sample_size])
    by_group = frame.set_index(['case_id','threshold','interval_min'])
    audit_all, audit_sample = NumericAudit(), NumericAudit()
    seen, qualified = set(), set()
    subject_mismatches = 0
    sampled_count = 0
    readers = vital_records(root) if source == 'VitalDB' else sic_records(root)
    for record in readers:
        cid = record['case_id']
        if cid in seen:
            raise ValueError('Duplicate source case')
        seen.add(cid)
        if not eligible(record):
            continue
        qualified.add(cid)
        selected = hour_selection(record['map'])
        result = case_components(record['map'], selected)
        if cid in expected.index:
            subject_mismatches += int(record['subject_id'] != int(expected.loc[cid, 'subject_id']))
            audit_all.add(result, expected.loc[cid, COLS].to_numpy(float))
        if cid in sample:
            sampled_count += 1
            for t in THRESHOLDS:
                for step in INTERVALS:
                    values = result if (t,step) == (65,5) else case_components(record['map'], selected, t, step)
                    audit_sample.add(values, by_group.loc[(cid,t,step), COLS].to_numpy(float))
        if len(seen) % 1000 == 0:
            print(json.dumps(dict(independent_source=source, source_records_read=len(seen), primary_records_recalculated=len(qualified))), flush=True)
    missing = len(set(expected.index)-qualified)
    unexpected = len(qualified-set(expected.index))
    report = dict(source=source, original_records_read=len(seen), independently_eligible_records=len(qualified),
                  expected_primary_records=len(expected), all_primary_5min_65_records_checked=audit_all.checked//len(COLS),
                  all_primary_5min_65_phases_recalculated=len(qualified)*5,
                  all_primary_5min_65=audit_all.report(), deterministic_sample_records=sampled_count,
                  deterministic_sample_group_rows=sampled_count*12, sample_all_12_groups=audit_sample.report(),
                  deterministic_sample_phase_results_checked=sampled_count*sum(INTERVALS)*len(THRESHOLDS),
                  missing_eligible_records=missing, unexpected_eligible_records=unexpected,
                  subject_linkage_mismatches=subject_mismatches,
                  human_adjudication=False, clinical_validation=False,
                  main_primary_component_rows_invariant_checked=len(frame),
                  sample_selection='Smallest SHA256(source:case:20260706), up to requested count',
                  every_recalculated_phase_component_invariants_checked=True)
    report['status'] = 'PASS' if not (missing or unexpected or subject_mismatches or audit_all.failures or audit_sample.failures) else 'FAIL'
    return report, frame


def run(root=ROOT, sample_size=100):
    start = time.monotonic()
    contract_check(root)
    needed = [root/f'private/{s}_components.csv.gz' for s in ('VitalDB','SICdb')]
    needed += [root/'private/sicdb/ready.json', root/'aggregate/comparison_summary.csv']
    original = root.parent/'analysis_v10_subject_phase_release/outputs/intermediate'
    needed += [original/'artmap_10s.parquet', original/'vitaldb_manifest.parquet']
    missing = [p.name for p in needed if not p.is_file()]
    report = dict(status='PENDING_INPUTS', independent_code=True, imports_shared_estimator=False,
                  human_adjudication=False, clinical_validation=False, patient_identifiers_in_report=False,
                  tolerance=dict(absolute=ATOL, relative=RTOL), missing_inputs=missing,
                  requested_sample_records_per_source=sample_size,
                  scope='All eligible primary 5min/65 source records; deterministic sample all 12 groups; all 12 primary patient-cluster summaries per source')
    if missing:
        return report
    ready = json.loads((root/'private/sicdb/ready.json').read_text())
    if str(ready.get('status','')).lower() not in ('ready','complete','pass'):
        report['pending_reason'] = 'SICdb extraction not ready'
        return report
    files = needed + [root/'analysis_contract.json', root/'PLAN.md', Path(__file__)]
    for entry in ready['shards']:
        files.append(root/'private/sicdb'/(entry['path'] if isinstance(entry,dict) else entry))
    hashes = {str(p):digest(p) for p in files}
    source_reports, frames = [], []
    for source in ('VitalDB','SICdb'):
        result, frame = verify_source(root, source, sample_size)
        source_reports.append(result)
        frames.append(frame)
    recalculated = primary_summary(pd.concat(frames, ignore_index=True))
    expected = pd.read_csv(root/'aggregate/comparison_summary.csv')
    expected = expected[expected.scenario == 'primary']
    keys = ['source','scenario','threshold','interval_min']
    a = recalculated.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if b.index.has_duplicates or not a.index.equals(b.index) or len(a) != 24:
        raise ValueError('Primary summary keys must be 12 groups for each source')
    audit = NumericAudit()
    mismatch_columns = []
    for name in a.columns:
        if name not in b.columns:
            raise ValueError('Summary lacks column '+name)
        before = audit.failures
        audit.add(a[name].to_numpy(float), b[name].to_numpy(float))
        if before != audit.failures:
            mismatch_columns.append(name)
    unchanged = all(digest(p) == h for p,h in hashes.items())
    report.update(status='PASS' if all(s['status']=='PASS' for s in source_reports) and not audit.failures and unchanged else 'FAIL',
                  sources=source_reports, summary=dict(groups_checked=len(a), numerical_comparison=audit.report(),
                  mismatched_columns=mismatch_columns, bootstrap_cluster='patient', replicates=REPLICATES, seed=SEED,
                  summary_inputs='Main case-phase-averaged components: all primary 5min/65 rows independently source-verified; other groups source-verified on deterministic sample',
                  patient_sum_before_bootstrap=True, resampling='multinomial with sorted subject IDs', interval='percentile'),
                  input_files_unchanged_during_check=unchanged,
                  protected_sources_written=False, elapsed_seconds=time.monotonic()-start,
                  verification_code_sha256=digest(Path(__file__)),
                  analysis_contract_sha256=digest(root/'analysis_contract.json'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--sample-records', type=int, default=100)
    args = parser.parse_args()
    if args.sample_records <= 0:
        parser.error('sample-records must be positive')
    try:
        report = run(args.root, args.sample_records)
    except Exception as exc:
        # Exception messages are deliberately omitted: source errors may contain IDs.
        report = dict(status='FAIL', error_type=type(exc).__name__,
                      error='Independent verification stopped; inspect locally without publishing identifiers',
                      human_adjudication=False, patient_identifiers_in_report=False)
    write_report(args.root/'qa/independent_check.json', report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return {'PASS':0, 'FAIL':1, 'PENDING_INPUTS':2}[report['status']]


if __name__ == '__main__':
    sys.exit(main())
