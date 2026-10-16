"""Independent, aggregate-only verification. Never import the analysis estimators."""
from pathlib import Path
import sys

sys.dont_writebytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'SICdb分钟级对齐分析_20261015/private/deps'))

import argparse
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd

COLS = ['R', 'D', 'H', 'O', 'H_unavailable', 'valid_minutes',
        'unavailable_minutes', 'reference_low_unavailable_min',
        'lowref_normaldisplay_min', 'normalref_lowdisplay_min',
        'reference_low_min', 'display_low_min', 'window_minutes',
        'episodes', 'fresh', 'inherited_only', 'never_low', 'delay_sum_min']
KEYS = ['resolution', 'scenario', 'initialization', 'threshold',
        'interval_min', 'minimum_event_sec']
TOLERANCES = {
    'components': {'atol': 1e-8, 'rtol': 1e-11},
    'event_counts': {'atol': 1e-12, 'rtol': 0.},
    'metrics_and_ci': {'atol': 1e-10, 'rtol': 1e-10},
    'grid_onsets_and_flags': {'atol': 0., 'rtol': 0.},
    'numeric_cache_alignment': {'atol': 1e-10, 'rtol': 0.},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.items = {}

    def compare(self, name, actual, expected, tolerance='components'):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        item = self.items.setdefault(name, dict(status='PASS', comparisons=0,
                                               failed_comparisons=0, max_abs_error=0.,
                                               tolerance=tolerance))
        if a.shape != b.shape:
            item['status'] = 'FAIL'
            item['failed_comparisons'] += max(a.size, b.size, 1)
            item['comparisons'] += max(a.size, b.size, 1)
            return
        same = np.isclose(a, b, equal_nan=True, **TOLERANCES[tolerance])
        finite = np.isfinite(a) & np.isfinite(b)
        err = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        item['comparisons'] += int(a.size)
        item['failed_comparisons'] += int((~same).sum())
        item['max_abs_error'] = max(item['max_abs_error'], err)
        if not same.all():
            item['status'] = 'FAIL'

    def check(self, name, conditions):
        a = np.asarray(conditions, bool)
        self.compare(name, a, np.ones(a.shape), 'grid_onsets_and_flags')

    def status(self):
        return 'FAIL' if any(x['status'] == 'FAIL' for x in self.items.values()) else 'PASS'


def independent_gate(frame, relaxed=False):
    m = frame.numeric_map.to_numpy(float)
    mean = frame.wave_mean.to_numpy(float)
    span = frame.wave_p95.to_numpy(float) - frame.wave_p05.to_numpy(float)
    sbp, dbp = frame.sbp.to_numpy(float), frame.dbp.to_numpy(float)
    both = np.isfinite(sbp) & np.isfinite(dbp)
    paired = ((sbp >= 30) & (sbp <= 250) & (dbp >= 5) & (dbp <= 150)
              & (sbp - dbp >= 5) & (sbp - dbp <= 160)
              & (m >= dbp - 5) & (m <= sbp + 5))
    parts = {
        'numeric_range': np.isfinite(m) & (m >= 20) & (m <= 180),
        'coverage': frame.wave_coverage.to_numpy(float) >= (.8 if relaxed else .9),
        'waveform_mean': np.isfinite(mean) & (mean >= 20) & (mean <= 180),
        'waveform_range': ((frame.wave_p05.to_numpy(float) >= (-5 if relaxed else 0))
                           & (frame.wave_p95.to_numpy(float) <= (300 if relaxed else 250))),
        'pulsatility': ((span >= (5 if relaxed else 8)) & (span <= (180 if relaxed else 160))
                        & (frame.peak_count.to_numpy(float) >= (3 if relaxed else 4))
                        & (frame.peak_count.to_numpy(float) <= 40)),
        'numeric_wave_agreement': np.abs(mean - m) <= (15 if relaxed else 10),
        'optional_pair_coherence': ~both | paired,
    }
    return np.all(np.array(list(parts.values())), axis=0), parts


def first_qualifying(good, seconds, needed):
    consecutive = 0
    for j, value in enumerate(good):
        if j and seconds[j] - seconds[j - 1] != 10:
            consecutive = 0
        consecutive = consecutive + 1 if value else 0
        if consecutive == needed:
            return float(seconds[j - needed + 1])
    return None


def independent_display(values, step, phase, seed, seed_zero):
    v = np.asarray(values, float)
    n = len(v)
    samples = np.arange(phase, n, step)
    samples = samples[np.isfinite(v[samples])]
    ticks = np.arange(n)
    position = np.searchsorted(samples, ticks, side='right') - 1
    held = np.full(n, -1, dtype=np.int64)
    available = position >= 0
    held[available] = samples[position[available]]
    finite = np.flatnonzero(np.isfinite(v))
    if seed and (phase != 0 or seed_zero) and finite.size:
        first = int(finite[0])
        held[(ticks >= first) & (held < 0)] = first
    d = np.full(n, np.nan)
    valid = held >= 0
    d[valid] = v[held[valid]]
    return d, held


def independent_events(reference, threshold, minimum):
    ref = np.asarray(reference, float)
    low = np.isfinite(ref) & (ref < threshold)
    edges = np.diff(np.r_[False, low, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]


def independent_event_counts(display, held, events, threshold, dt):
    low = np.isfinite(display) & (display < threshold)
    fresh = inherited = never = 0
    delay = 0.
    for a, b in events:
        visible = np.flatnonzero(low[a:b])
        if not visible.size:
            never += 1
        else:
            delay += int(visible[0]) * dt
            sources = held[a:b][visible]
            if np.any((sources >= a) & (sources < b)):
                fresh += 1
            else:
                inherited += 1
    return fresh, inherited, never, delay


def independent_components(values, selected, threshold, step, dt, min_bins,
                           seed=False, seed_zero=False):
    v = np.asarray(values, float)
    selected = np.asarray(selected, bool)
    ref = np.where(selected, v, np.nan)
    valid = np.isfinite(ref)
    lowref = valid & (ref < threshold)
    r = np.zeros(len(v))
    r[valid] = np.maximum(threshold - ref[valid], 0.)
    events = independent_events(ref, threshold, min_bins)
    phase_rows = np.empty((step, len(COLS)))
    for phase in range(step):
        display, held = independent_display(v, step, phase, seed, seed_zero)
        absent = ~np.isfinite(display)
        lowdisp = ~absent & (display < threshold)
        d = np.zeros(len(v))
        d[valid & lowdisp] = threshold - display[valid & lowdisp]
        h, o = np.maximum(r - d, 0.), np.maximum(d - r, 0.)
        f, inherited, never, delay = independent_event_counts(display, held, events, threshold, dt)
        phase_rows[phase] = [r.sum() * dt, d.sum() * dt, h.sum() * dt, o.sum() * dt,
                             r[valid & absent].sum() * dt, valid.sum() * dt,
                             (valid & absent).sum() * dt, (lowref & absent).sum() * dt,
                             (lowref & ~absent & ~lowdisp).sum() * dt,
                             (valid & ~lowref & lowdisp).sum() * dt,
                             lowref.sum() * dt, (valid & lowdisp).sum() * dt,
                             selected.sum() * dt, len(events), f, inherited, never, delay]
    return phase_rows.mean(axis=0), phase_rows


def independent_minute(values, count):
    x = np.asarray(values, float)[:count * 6].reshape(count, 6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        medians = np.nanmedian(x, axis=1)
    medians[np.sum(np.isfinite(x), axis=1) < 5] = np.nan
    return medians


def independent_hours(values):
    v = np.asarray(values, float)
    selected = np.zeros(len(v), bool)
    for start in range(0, len(v) - 59, 60):
        if np.count_nonzero(np.isfinite(v[start:start + 60])) >= 48:
            selected[start:start + 60] = True
    return selected


def self_test():
    # These expectations are hand-derived, not obtained from main-code outputs.
    assert 'independent_components' in globals(), 'independent calculator missing'
    x = np.array([70., 60., 60., np.nan, 70., 50.])
    d, held = independent_display(x, 3, 1, False, False)
    np.testing.assert_equal(held, [-1, 1, 1, 1, 4, 4])
    np.testing.assert_allclose(d, [np.nan, 60, 60, 60, 70, 70], equal_nan=True)
    d, held = independent_display(x, 3, 1, True, False)
    np.testing.assert_equal(held, [0, 1, 1, 1, 4, 4])
    # A missing scheduled measurement must not erase the previous display.
    d, held = independent_display([50., 70., np.nan, 70.], 2, 0, False, False)
    np.testing.assert_equal(held, [0, 0, 0, 0])
    assert independent_events([60, 60, np.nan, 60, 60, 60], 65, 3) == [(3, 6)]
    assert first_qualifying(np.array([1, 1, 0, 1, 1, 1, 1], bool),
                            np.arange(7) * 10, 3) == 30.
    assert first_qualifying(np.ones(3, bool), np.array([0, 10, 30]), 3) is None
    low = pd.DataFrame(dict(numeric_map=[30.], wave_mean=[30.], wave_coverage=[1.],
                            wave_p05=[10.], wave_p95=[50.], wave_span_90pct=[40.],
                            peak_count=[10.], sbp=[50.], dbp=[20.]))
    assert independent_gate(low)[0][0], 'technically supported MAP<40 must qualify'
    # Fresh, inherited-only, and never-low are separate event categories.
    assert independent_event_counts(np.array([np.nan, 60., 60.]),
                                    np.array([-1, 1, 1]), [(0, 3)], 65, 1.) == (1, 0, 0, 1.)
    assert independent_event_counts(np.array([60., 60., 60.]),
                                    np.array([0, 0, 0]), [(1, 3)], 65, 1.) == (0, 1, 0, 0.)
    assert independent_event_counts(np.array([70., 70., 70.]),
                                    np.array([0, 0, 0]), [(1, 3)], 65, 1.) == (0, 0, 1, 0.)
    v = np.array([70., 60., 60., 70.])
    got, phase = independent_components(v, np.ones(4, bool), 65, 2, 1., 2)
    expected = [10, 10, 2.5, 2.5, 0, 4, .5, 0, .5, .5, 2, 2, 4, 1, 1, 0, 0, .5]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    # Keep an unavailable leading grid bin; dropping it would change absolute phases.
    a, _ = independent_components([np.nan, 60., 60., 70.], np.ones(4, bool),
                                  65, 2, 1., 2)
    assert a[12] == 4 and a[5] == 3
    m = independent_minute(np.array([60.] * 5 + [np.nan] + [70.] * 4 + [np.nan] * 2), 2)
    np.testing.assert_allclose(m, [60., np.nan], equal_nan=True)
    assert independent_hours(np.array([70.] * 48 + [np.nan] * 12 + [70.])).sum() == 60
    for phase in [0, 1, 2]:
        d, held = independent_display([np.nan, 60., 60.], 3, phase, True, True)
        assert np.isnan(d[0]) and d[1] == 60 and held[1] == 1
    d, held = independent_display([np.nan, 60., 60.], 3, 0, True, False)
    assert np.all(np.isnan(d)) and np.all(held == -1)
    return 15


def verify_invariants(audit, prefix, vectors):
    a = np.asarray(vectors, float)
    audit.check(prefix + '/finite_nonnegative', np.isfinite(a) & (a >= -1e-10))
    audit.compare(prefix + '/auc_identity', a[..., 1] - a[..., 0],
                  a[..., 3] - a[..., 2])
    audit.compare(prefix + '/event_partition', a[..., 13], a[..., 14:17].sum(axis=-1),
                  'event_counts')
    audit.compare(prefix + '/state_partition', a[..., 11],
                  a[..., 10] - a[..., 7] - a[..., 8] + a[..., 9])
    audit.check(prefix + '/bounded_minutes',
                (a[..., 6] <= a[..., 5] + 1e-10)
                & (a[..., 7] <= a[..., 6] + 1e-10)
                & (a[..., 7] <= a[..., 10] + 1e-10)
                & (a[..., 5] <= a[..., 12] + 1e-10)
                & (a[..., 4] <= a[..., 2] + 1e-8))


def scalar_metrics(vectors):
    a = np.asarray(vectors, float)

    def ratio(num, den):
        num, den = np.broadcast_arrays(num, den)
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    return {
        'hidden_ratio': ratio(a[..., 2], a[..., 0]),
        'overdisplay_ratio': ratio(a[..., 3], a[..., 0]),
        'net_ratio': ratio(a[..., 1] - a[..., 0], a[..., 0]),
        'miss_probability': ratio(a[..., 16], a[..., 13]),
        'fresh_probability': ratio(a[..., 14], a[..., 13]),
    }


def wait_for_outputs(paths, seconds, poll):
    deadline = time.monotonic() + seconds
    announced = False
    while True:
        ready = all(p.is_file() and p.stat().st_size > 0 for p in paths)
        # Avoid reading a main-process file while it is still being written.
        if ready and all(time.time() - p.stat().st_mtime >= 30 for p in paths):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if not announced:
            print(json.dumps(dict(stage='waiting_for_completed_main_outputs',
                                  poll_seconds=poll)), flush=True)
            announced = True
        time.sleep(min(poll, remaining))


def rebuild_all(audit, contract, report):
    src = PROJECT / 'analysis_v10_subject_phase_release/outputs/intermediate'
    panel = pd.read_parquet(src / 'artmap_10s.parquet')
    meta = pd.read_parquet(src / 'vitaldb_manifest.parquet').set_index('case_id')
    onsets = pd.read_csv(ROOT / 'restricted/onsets.csv').set_index('case_id')
    bins = pd.read_csv(ROOT / 'restricted/startup_quality_bins.csv.gz')
    audit.check('inputs/unique_onsets', not onsets.index.duplicated().any())
    audit.check('inputs/unique_metadata', not meta.index.duplicated().any())
    audit.check('inputs/unique_cache_ticks', not bins.duplicated(['case_id', 'time_sec']).any())
    panel_cases = set(panel.case_id.unique())
    audit.check('inputs/cohort_case_set', panel_cases == set(onsets.index))
    audit.check('inputs/cache_case_set', set(bins.case_id.unique()) <= panel_cases)
    cached = {int(c): g.sort_values('time_sec').reset_index(drop=True)
              for c, g in bins.groupby('case_id')}
    output = []
    native_phases = minute_phases = 0
    onset_counts = {rule: dict(qualified=0, map_below40=0, map_below65=0)
                    for rule in ['primary', 'core30', 'core120', 'relaxed60']}
    for number, (case, group) in enumerate(panel.groupby('case_id', sort=True), 1):
        cid = int(case)
        group = group.sort_values('time_sec')
        v = group.art_map.to_numpy(float)
        ticks = group.time_sec.to_numpy(float)
        audit.compare('grid/native_absolute_uncompressed', ticks, np.arange(len(v)) * 10.,
                      'grid_onsets_and_flags')
        audit.check('grid/original_has_finite_reference', np.isfinite(v).any())
        if cid not in onsets.index or cid not in meta.index or not np.isfinite(v).any():
            audit.check('inputs/processable_case', False)
            continue
        row = onsets.loc[cid]
        metadata = meta.loc[cid]
        sid = int(metadata.subjectid)
        audit.compare('onset/subject_link', row.subject_id, sid, 'grid_onsets_and_flags')
        first = int(np.flatnonzero(np.isfinite(v))[0])
        audit.compare('onset/first_finite_time', row.first_finite_sec, ticks[first],
                      'grid_onsets_and_flags')
        audit.compare('onset/first_finite_map', row.first_map, v[first], 'numeric_cache_alignment')
        w = cached.get(cid)
        available = w is not None
        audit.check('onset/waveform_availability', bool(row.waveform_available) == available)
        found = dict.fromkeys(['primary', 'core30', 'core120', 'relaxed60'])
        core = np.zeros(0, bool)
        if available:
            times = w.time_sec.to_numpy(float)
            audit.compare('cache/first600_seconds_complete_grid', times,
                          np.arange(first * 10, first * 10 + 600, 10, dtype=float),
                          'grid_onsets_and_flags')
            integer_times = np.isfinite(times) & (times % 10 == 0) & (times >= 0)
            audit.check('cache/integer_grid', integer_times)
            if not integer_times.all():
                continue
            indices = (times / 10).astype(int)
            aligned = np.full(len(w), np.nan)
            inside = indices < len(v)
            aligned[inside] = v[indices[inside]]
            audit.compare('cache/numeric_map_matches_raw_panel', w.numeric_map, aligned,
                          'numeric_cache_alignment')
            audit.compare('cache/span_equals_p95_minus_p05', w.wave_span_90pct,
                          w.wave_p95 - w.wave_p05, 'numeric_cache_alignment')
            core, parts = independent_gate(w)
            relaxed, _ = independent_gate(w, True)
            audit.compare('gate/core_flags', w.core_good, core, 'grid_onsets_and_flags')
            audit.compare('gate/relaxed_flags', w.relaxed_good, relaxed, 'grid_onsets_and_flags')
            for key, val in parts.items():
                audit.compare('gate/component_' + key, w['flag_' + key], val,
                              'grid_onsets_and_flags')
            for rule, mask, count in [('primary', core, 6), ('core30', core, 3),
                                       ('core120', core, 12), ('relaxed60', relaxed, 6)]:
                found[rule] = first_qualifying(mask, times, count)
            for field, val in [('core_qualifying_bins', core.sum()),
                               ('relaxed_qualifying_bins', relaxed.sum()),
                               ('paired_bins', (np.isfinite(w.sbp) & np.isfinite(w.dbp)).sum())]:
                audit.compare('onset/' + field, row[field], val, 'grid_onsets_and_flags')
        for rule, onset in found.items():
            audit.compare('onset/' + rule + '_is_first_run', row[rule + '_onset_sec'],
                          np.nan if onset is None else onset, 'grid_onsets_and_flags')
            if onset is not None:
                onset_counts[rule]['qualified'] += 1
                onset_counts[rule]['map_below40'] += int(v[int(onset / 10)] < 40)
                onset_counts[rule]['map_below65'] += int(v[int(onset / 10)] < 65)
        primary = found['primary'] is not None
        audit.check('onset/primary_pass', bool(row.primary_pass) == primary)
        specs = [('original_all', v.copy(), np.ones(len(v), bool))]
        if available:
            specs.append(('original_waveform_available', v.copy(), np.ones(len(v), bool)))
        if primary:
            specs.append(('original_matched_primary_onset', v.copy(), np.ones(len(v), bool)))
        for scenario, rule in [('primary_core60_onset', 'primary'), ('core30_onset', 'core30'),
                                ('core120_onset', 'core120'), ('relaxed60_onset', 'relaxed60')]:
            if found[rule] is not None:
                selected = ticks >= found[rule]
                specs.append((scenario, np.where(selected, v, np.nan), selected))
        if primary:
            onset = found['primary']
            selected = ticks >= onset + 60
            specs.append(('core60_confirmation_end', np.where(selected, v, np.nan), selected))
            selected = ticks >= onset
            masked = np.where(selected, v, np.nan)
            failed = indices[(~core) & (times >= onset) & (indices < len(v))]
            masked[failed] = np.nan
            specs.append(('core60_mask_later_failed_startup_bins', masked, selected))
        for scenario, values, selected in specs:
            audit.check('grid/scenario_length_preserved', len(values) == len(v) == len(selected))
            retained = np.isfinite(values)
            audit.compare('grid/retained_values_unchanged', values[retained], v[retained],
                          'grid_onsets_and_flags')
            broad = scenario in ('original_all', 'primary_core60_onset')
            intervals = contract['native_frequency']['intervals_at65_min'] if broad else [5]
            configurations = [(65, interval, 60) for interval in intervals]
            if broad:
                configurations.extend((threshold, 5, 60)
                                      for threshold in contract['native_frequency']['other_thresholds_at5min'])
            if scenario in ('original_matched_primary_onset', 'primary_core60_onset'):
                configurations.extend([(65, 5, 120), (65, 5, 240)])
            for threshold, interval, minimum in configurations:
                initializations = [False, True] if (threshold, interval, minimum) == (65, 5, 60) else [False]
                for seed in initializations:
                    vector, phases = independent_components(values, selected, threshold,
                        int(interval * 6), 1 / 6, minimum // 10, seed, False)
                    verify_invariants(audit, 'independent/native_phase', phases)
                    native_phases += len(phases)
                    output.append(dict(resolution='native10s', scenario=scenario,
                        initialization='first_valid_seed_nonzero_phase' if seed else 'unavailable',
                        threshold=threshold, interval_min=interval, minimum_event_sec=minimum,
                        case_id=cid, subject_id=sid, **dict(zip(COLS, vector))))
        if primary and float(metadata.age) >= 25:
            minutes = int(np.floor((float(metadata.primary_end_sec)
                                    - float(metadata.primary_start_sec)) / 60))
            audit.check('grid/minute_source_length', minutes * 6 <= len(v))
            if minutes * 6 > len(v):
                continue
            original_minute = independent_minute(v, minutes)
            selected = independent_hours(original_minute)
            if selected.any():
                trimmed = np.where(ticks >= found['primary'], v, np.nan)
                qualified_minute = independent_minute(trimmed, minutes)
                for scenario, values in [('minute_original_matched_fixed_hours', original_minute),
                                          ('minute_core60_fixed_hours', qualified_minute)]:
                    audit.check('grid/minute_length_preserved', len(values) == minutes)
                    for seed in [False, True]:
                        vector, phases = independent_components(values, selected, 65, 5, 1., 3, seed, True)
                        verify_invariants(audit, 'independent/minute_phase', phases)
                        minute_phases += len(phases)
                        output.append(dict(resolution='minute', scenario=scenario,
                            initialization='first_valid_seed_all_phases' if seed else 'unavailable',
                            threshold=65, interval_min=5, minimum_event_sec=180,
                            case_id=cid, subject_id=sid, **dict(zip(COLS, vector))))
        if number % 300 == 0:
            print(json.dumps(dict(stage='independent_reconstruction', completed_cases=number)), flush=True)
    report['coverage'] = dict(original_cases=len(panel_cases), original_grid_rows=len(panel),
                              independent_component_rows=len(output), native_phase_vectors=native_phases,
                              minute_phase_vectors=minute_phases, onset_rules=onset_counts,
                              component_scope='all generated native configurations and both minute scenarios; all absolute phases')
    return pd.DataFrame(output)


def compare_outputs(audit, independent, cases, summary, report):
    join = KEYS + ['case_id', 'subject_id']
    audit.check('case_table/unique_keys_main', not cases.duplicated(join).any())
    audit.check('case_table/unique_keys_independent', not independent.duplicated(join).any())
    matched = independent.merge(cases, on=join, how='outer', suffixes=('_ind', '_main'),
                                indicator=True, validate='one_to_one')
    audit.check('case_table/exact_row_key_set', matched['_merge'].eq('both'))
    both = matched[matched['_merge'].eq('both')]
    for col in COLS:
        audit.compare('case_components/' + col, both[col + '_main'], both[col + '_ind'],
                      'event_counts' if col in COLS[13:17] else 'components')
    verify_invariants(audit, 'main/all_case_rows', cases[COLS].to_numpy(float))
    verify_invariants(audit, 'main/all_summary_rows', summary[COLS].to_numpy(float))
    audit.check('summary/unique_group_keys', not summary.duplicated(KEYS).any())
    expected_keys = set(independent.groupby(KEYS).groups)
    main_keys = set(summary.groupby(KEYS).groups)
    audit.check('summary/exact_group_key_set', expected_keys == main_keys)
    indexed = summary.set_index(KEYS)
    ci_groups = []
    for keys, group in independent.groupby(KEYS, sort=True):
        if keys not in indexed.index:
            continue
        target = indexed.loc[keys]
        total = group[COLS].sum().to_numpy(float)
        audit.compare('summary/raw_component_totals', target[COLS].to_numpy(float), total)
        audit.compare('summary/n_cases', target.n_cases, len(group), 'grid_onsets_and_flags')
        audit.compare('summary/n_subjects', target.n_subjects, group.subject_id.nunique(),
                      'grid_onsets_and_flags')
        audit.compare('summary/n_cases_with_events', target.n_cases_with_events,
                      group.episodes.gt(0).sum(), 'grid_onsets_and_flags')
        point = scalar_metrics(total)
        for metric, estimate in point.items():
            audit.compare('summary/point_' + metric, target[metric], estimate, 'metrics_and_ci')
        focus = (keys[0] == 'native10s' and keys[1] in
                 ('primary_core60_onset', 'original_matched_primary_onset', 'original_all')
                 and keys[3:] == (65, 5., 60))
        if not focus:
            continue
        patients = group.groupby('subject_id', sort=True)[COLS].sum()
        n = len(patients)
        weights = np.random.default_rng(20260706).multinomial(n, np.full(n, 1 / n), size=1000)
        draws = scalar_metrics(weights @ patients.to_numpy(float))
        # Point estimates use the same patient-summed vectors as the bootstrap.
        patient_point = scalar_metrics(patients.sum().to_numpy(float))
        before = sum(x['failed_comparisons'] for x in audit.items.values())
        for metric, values in draws.items():
            defined = values[np.isfinite(values)]
            ci = np.quantile(defined, [.025, .975], method='linear') if len(defined) else np.full(2, np.nan)
            audit.compare('bootstrap/' + metric,
                          [target[metric], target[metric + '_lo'], target[metric + '_hi']],
                          [patient_point[metric], *ci], 'metrics_and_ci')
            audit.check('bootstrap/all_replicates_defined_' + metric, len(defined) == 1000)
        after = sum(x['failed_comparisons'] for x in audit.items.values())
        ci_groups.append(dict(scenario=keys[1], initialization=keys[2],
                              subjects=n, replicates=1000, metrics=5,
                              status='PASS' if before == after else 'FAIL'))
    audit.compare('bootstrap/required_groups', len(ci_groups), 6, 'grid_onsets_and_flags')
    report['bootstrap'] = dict(cluster='subjectid', seed=20260706, replicates=1000,
        phase_order='mean raw phase components per case, sum per subject, multinomial patient weights',
        percentile_method='linear 2.5th and 97.5th percentiles', groups=ci_groups)
    report['coverage'].update(compared_case_rows=len(both), summary_groups=len(summary),
                               compared_component_values=len(both) * len(COLS))


def run(args):
    began = time.monotonic()
    audit = Audit()
    report = dict(status='FAIL', method='independent raw-grid reconstruction; no main/QC/estimator imports',
                  tolerances=TOLERANCES, risks=[
        'Computational verification is not clinical adjudication, physician endorsement, or SQI validation.',
        'Waveform features and peak counts are accepted as frozen cache inputs; raw waveform feature extraction is not rederived here.',
        'Startup gates cover the cached first 600 seconds only, not full-record waveform quality.',
        'Synthetic held-display observability is not a causal clinical benefit or device-performance estimate.',
        'Confidence intervals are independently recomputed for the five prespecified metrics in six primary native groups only.',
    ])
    stage = 'self_tests'
    protected = {}
    try:
        tests = self_test()
        audit.compare('synthetic_tests/completed', tests, 15, 'grid_onsets_and_flags')
        stage = 'contract_and_input_integrity'
        contract = json.loads((ROOT / 'analysis_contract.json').read_text())
        contract_path = ROOT / 'analysis_contract.json'
        src = PROJECT / 'analysis_v10_subject_phase_release/outputs/intermediate'
        inputs = [contract_path, ROOT / 'code/rerun.py', ROOT / 'code/qc_onset.py',
                  Path(__file__), src / 'artmap_10s.parquet', src / 'vitaldb_manifest.parquet',
                  ROOT / 'restricted/onsets.csv', ROOT / 'restricted/startup_quality_bins.csv.gz']
        protected = {str(p): digest(p) for p in inputs}
        audit.check('contract/native_grid_10s', contract['grid_sec'] == 10)
        audit.check('contract/bootstrap_seed_and_replicates',
                    contract['inference']['seed'] == 20260706
                    and contract['inference']['bootstrap_replicates'] == 1000
                    and contract['inference']['cluster'] == 'subjectid')
        expected_gate = {
            'finite_numeric_MAP_mmHg': [20, 180], 'waveform_coverage_at_least': .9,
            'waveform_mean_mmHg': [20, 180], 'waveform_p05_at_least_mmHg': 0,
            'waveform_p95_at_most_mmHg': 250, 'waveform_p95_minus_p05_mmHg': [8, 160],
            'peak_count_per10s': [4, 40],
            'absolute_waveform_mean_minus_numeric_MAP_at_most_mmHg': 10,
            'if_both_numeric_SBP_DBP_available': {'SBP_mmHg': [30, 250], 'DBP_mmHg': [5, 150],
                'pulse_pressure_mmHg': [5, 160], 'MAP_within_DBP_SBP_tolerance_mmHg': 5},
        }
        audit.check('contract/implemented_frozen_primary_gate',
                    all(contract['primary_bin_gate'].get(k) == v for k, v in expected_gate.items()))
        audit.check('inputs/raw_panel_frozen_hash',
                    protected[str(src / 'artmap_10s.parquet')] == contract['base_map_panel_sha256'])
        audit.check('inputs/metadata_frozen_hash',
                    protected[str(src / 'vitaldb_manifest.parquet')] == contract['base_metadata_sha256'])
        stage = 'independent_reconstruction'
        independent = rebuild_all(audit, contract, report)
        stage = 'waiting_for_main_outputs'
        paths = [ROOT / 'restricted/case_components.csv.gz', ROOT / 'aggregate/observability_summary.csv']
        if not wait_for_outputs(paths, args.wait_seconds, args.poll_seconds):
            audit.check('main_outputs/available_before_timeout', False)
            report['risks'].append('Required main outputs were absent or not settled before timeout; comparison is incomplete.')
        else:
            stage = 'main_output_comparison'
            protected.update({str(p): digest(p) for p in paths})
            cases = pd.read_csv(paths[0])
            summary = pd.read_csv(paths[1])
            compare_outputs(audit, independent, cases, summary, report)
        stage = 'final_integrity'
        for path, frozen in protected.items():
            audit.check('integrity/inputs_and_main_code_unchanged', digest(path) == frozen)
        forbidden = [name for name in sys.modules if name in ('rerun', 'qc_onset', 'event_extension')
                     or name.startswith('ioh.estimands')]
        audit.check('independence/no_analysis_estimator_imports', not forbidden)
        report['status'] = audit.status()
    except Exception as error:
        # Do not expose per-case values, identifiers, or exception text in public QA.
        audit.check('execution/no_exception', False)
        report['execution_error'] = dict(stage=stage, exception_type=type(error).__name__)
        report['risks'].append('Execution did not finish; FAIL must not be interpreted as a completed comparison.')
    report['checks'] = audit.items
    report['comparisons'] = sum(x['comparisons'] for x in audit.items.values())
    report['failed_comparisons'] = sum(x['failed_comparisons'] for x in audit.items.values())
    report['max_abs_error'] = max((x['max_abs_error'] for x in audit.items.values()), default=0.)
    report['elapsed_seconds'] = round(time.monotonic() - began, 3)
    report['input_sha256'] = {str(Path(p).relative_to(PROJECT)): h for p, h in protected.items()}
    report['environment'] = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)
    (ROOT / 'qa/independent_verification.json').write_text(
        json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'comparisons', 'failed_comparisons',
                                            'max_abs_error', 'elapsed_seconds')}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=7200.)
    parser.add_argument('--poll-seconds', type=float, default=120.)
    options = parser.parse_args()
    if options.self_test:
        print(json.dumps(dict(status='PASS', synthetic_tests=self_test())))
    else:
        if options.wait_seconds < 0 or options.poll_seconds < 60:
            parser.error('wait must be nonnegative and poll must be at least 60 seconds')
        sys.exit(run(options))
