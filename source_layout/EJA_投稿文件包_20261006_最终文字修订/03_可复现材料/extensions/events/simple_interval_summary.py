import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from event_extension import held_display, reference_events


SEED = 20260706
REPLICATES = 1000


def displayed_low_minutes(true_low, unavailable_low, normal_discordance,
                          low_discordance, valid_reference):
    values = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in
        (true_low, unavailable_low, normal_discordance, low_discordance,
         valid_reference)])
    if any(not np.isfinite(x).all() or (x < -1e-9).any() for x in values):
        raise ValueError('Duration components must be finite and nonnegative')
    true, unavailable, false_normal, false_low, valid = values
    result = true - unavailable - false_normal + false_low
    if ((true > valid + 1e-9).any()
            or (unavailable + false_normal > true + 1e-9).any()
            or (result < -1e-9).any() or (result > valid + 1e-9).any()):
        raise ValueError('Duration components violate reference-time bounds')
    return np.clip(result, 0, valid)


def pooled_twa(area, observation_minutes):
    area, minutes = np.broadcast_arrays(np.asarray(area, dtype=float),
                                       np.asarray(observation_minutes, dtype=float))
    if (not np.isfinite(area).all() or not np.isfinite(minutes).all()
            or (area < 0).any() or (minutes < 0).any() or minutes.sum() <= 0):
        raise ValueError('Pooled TWA requires finite area and positive observation time')
    return float(area.sum() / minutes.sum())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summary_row(frame, threshold, interval, reference=False):
    grouped = frame.groupby('subjectid', sort=True)[[
        'episode_count', 'episode_detected', 'true_hypotension_min',
        'displayed_hypotension_min', 'true_auc', 'display_auc',
        'valid_reference_min']].sum()
    total = grouped.sum()
    prefix = 'true' if reference else 'display'
    count = total.episode_count if reference else total.episode_detected
    duration = total.true_hypotension_min if reference else total.displayed_hypotension_min
    area = total[prefix + '_auc']
    row = {'Threshold (mmHg)': int(threshold),
           'Sampling interval (min)': 'Continuous reference' if reference else interval,
           'Reference episodes': int(round(total.episode_count)),
           'Events with low display (expected n)': float(count),
           'Events with low display (%)': float(count / total.episode_count * 100),
           'Total hypotension duration (min)': float(duration),
           'Total deficit AUC (mmHg min)': float(area),
           'Pooled TWA deficit (mmHg)': pooled_twa([area], [total.valid_reference_min]),
           'Valid reference observation time (min)': float(total.valid_reference_min),
           'Cases': len(frame), 'Subjects': len(grouped),
           'Number of phases': 1 if reference else int(round(interval * 6))}

    # Match the frozen frequency bootstrap's subject order and deterministic draws.
    material = f'frequency_population|{SEED}|{threshold:.12g}|{interval:.12g}|subjectid'
    derived_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(derived_seed)
    idx = rng.integers(0, len(grouped), size=(REPLICATES, len(grouped)))
    columns = ['episode_count', 'episode_count' if reference else 'episode_detected',
               'true_hypotension_min' if reference else 'displayed_hypotension_min',
               prefix + '_auc', 'valid_reference_min']
    boot = np.column_stack([grouped[c].to_numpy()[idx].sum(axis=1) for c in columns])
    samples = [boot[:, 1], 100 * boot[:, 1] / boot[:, 0], boot[:, 2],
               boot[:, 3], boot[:, 3] / boot[:, 4]]
    stems = ['Events with low display (expected n)', 'Events with low display (%)',
             'Total hypotension duration (min)', 'Total deficit AUC (mmHg min)',
             'Pooled TWA deficit (mmHg)']
    for stem, values in zip(stems, samples):
        low, high = np.quantile(values, [.025, .975])
        row[stem + ' lower 95%'] = float(low)
        row[stem + ' upper 95%'] = float(high)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    source = args.analysis / 'outputs/intermediate'
    columns = ['case_id', 'threshold', 'interval_min', 'offset_sec', 'true_auc',
               'display_auc', 'hidden_auc', 'overdisplay_auc',
               'true_hypotension_min', 'display_valid_min', 'display_unavailable_min',
               'reference_hypotension_with_display_unavailable_min',
               'normotensive_display_discordance_min',
               'hypotensive_display_discordance_min', 'episode_count', 'episode_detected']
    phase = pd.read_parquet(source / 'frequency_decomp_case_offset.parquet', columns=columns)
    phase['valid_reference_min'] = phase.display_valid_min + phase.display_unavailable_min
    phase['displayed_hypotension_min'] = displayed_low_minutes(
        phase.true_hypotension_min, phase.reference_hypotension_with_display_unavailable_min,
        phase.normotensive_display_discordance_min, phase.hypotensive_display_discordance_min,
        phase.valid_reference_min)
    np.testing.assert_allclose(phase.display_auc - phase.true_auc,
                               phase.overdisplay_auc - phase.hidden_auc, atol=1e-8)
    counts = phase.groupby(['case_id', 'threshold', 'interval_min']).size()
    for (_, _, interval), number in counts.items():
        assert number == round(interval * 6)
    key = ['case_id', 'threshold', 'interval_min']
    case = phase.groupby(key, sort=True)[[c for c in phase.columns if c not in key + ['offset_sec']]].mean().reset_index()
    mapping = pd.read_parquet(source / 'vitaldb_manifest.parquet', columns=['case_id', 'subjectid'])
    case = case.merge(mapping, on='case_id', how='left', validate='many_to_one')
    assert case.subjectid.notna().all()
    assert case.case_id.nunique() == 2435 and case.subjectid.nunique() == 2380

    # Cross-interval invariants use a common reference and common valid-time denominator.
    invariant = case.groupby(['case_id', 'threshold'])[
        ['valid_reference_min', 'true_auc', 'true_hypotension_min', 'episode_count']].agg(lambda x: x.max() - x.min())
    assert np.max(invariant.to_numpy()) < 1e-8
    auc_frozen = pd.read_csv(args.analysis / 'outputs/tables/primary_auc_by_threshold_interval.csv')
    event_frozen = pd.read_csv(args.analysis / 'outputs/tables/episode_observability_by_threshold_interval.csv')
    rows = []
    for threshold, by_threshold in case.groupby('threshold', sort=True):
        reference = by_threshold[by_threshold.interval_min == 5]
        rows.append(summary_row(reference, threshold, 5, reference=True))
        for interval, group in by_threshold.groupby('interval_min', sort=True):
            row = summary_row(group, threshold, interval)
            old_auc = auc_frozen[(auc_frozen.threshold == threshold) & (auc_frozen.interval_min == interval)].iloc[0]
            old_event = event_frozen[(event_frozen.threshold == threshold) & (event_frozen.interval_min == interval)].iloc[0]
            np.testing.assert_allclose([row['Total deficit AUC (mmHg min)'], group.true_auc.sum()],
                [old_auc.display_auc_total, old_auc.true_auc_total], atol=1e-8, rtol=1e-12)
            np.testing.assert_allclose([row['Events with low display (expected n)'], row['Reference episodes'],
                                       row['Events with low display (%)'] / 100],
                [old_event.expected_detected_episodes, old_event.reference_episodes,
                 old_event.episode_detection_probability], atol=1e-10, rtol=1e-12)
            # Preserve published event-percentage intervals instead of adding Monte Carlo drift.
            row['Events with low display (%) lower 95%'] = float(old_event.episode_detection_ci_low * 100)
            row['Events with low display (%) upper 95%'] = float(old_event.episode_detection_ci_high * 100)
            rows.append(row)

    # Independently integrate the held display on 30 distributed cases and every phase.
    ids = np.sort(case.case_id.unique())
    selected = ids[np.linspace(0, len(ids) - 1, 30, dtype=int)]
    art = pd.read_parquet(source / 'artmap_10s.parquet', filters=[('case_id', 'in', selected.tolist())])
    lookup = phase.set_index(['case_id', 'threshold', 'interval_min', 'offset_sec'])
    verified = 0
    for case_id, trajectory in art.groupby('case_id', sort=True):
        trajectory = trajectory.sort_values('time_sec')
        values = trajectory.art_map.to_numpy(float)
        np.testing.assert_allclose(np.diff(trajectory.time_sec), 10)
        valid = np.isfinite(values)
        for interval in (1, 2.5, 3, 5, 10):
            step = int(interval * 6)
            for offset in range(step):
                display, _ = held_display(values, step, offset)
                for threshold in (55, 60, 65):
                    old = lookup.loc[(case_id, threshold, interval, offset * 10)]
                    low = valid & np.isfinite(display) & (display < threshold)
                    area = np.where(valid & np.isfinite(display), np.maximum(threshold - display, 0), 0).sum() / 6
                    events = reference_events(values, threshold, 0, 6)
                    detected = sum(bool(np.any(np.isfinite(display[s:e]) & (display[s:e] < threshold)))
                                   for s, e, _ in events)
                    np.testing.assert_allclose([low.sum() / 6, area, valid.sum() / 6, detected, len(events)],
                        [old.displayed_hypotension_min, old.display_auc, old.valid_reference_min,
                         old.episode_detected, old.episode_count], atol=1e-8, rtol=1e-12)
                    verified += 1
    table = pd.DataFrame(rows)
    folder = args.root / 'analysis/tables'
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / 'EJA_interval_absolute_summary.csv', index=False)
    primary = table[table['Threshold (mmHg)'] == 65]
    reader_columns = ['Sampling interval (min)', 'Events with low display (expected n)',
                      'Events with low display (%)', 'Total hypotension duration (min)',
                      'Total deficit AUC (mmHg min)', 'Pooled TWA deficit (mmHg)',
                      'Pooled TWA deficit (mmHg) lower 95%', 'Pooled TWA deficit (mmHg) upper 95%']
    primary[reader_columns].to_csv(folder / 'EJA_main_table3_simple.csv', index=False)
    result = {'source_sha256': {name: sha(source / name) for name in
        ['frequency_decomp_case_offset.parquet', 'vitaldb_manifest.parquet', 'artmap_10s.parquet']},
        'cases': 2435, 'subjects': 2380, 'thresholds_mmHg': [55, 60, 65],
        'intervals_min': [1, 2.5, 3, 5, 10], 'phase_rows': len(phase),
        'independent_integration_cases': len(selected), 'independent_scenarios_reconciled': verified,
        'published_event_counts_and_auc_exactly_reconciled': True,
        'same_valid_reference_denominator_every_row': True,
        'valid_reference_observation_minutes': float(primary.iloc[0]['Valid reference observation time (min)']),
        'twa_is_ratio_of_sums_not_mean_case_twa': True,
        'event_definition': 'Reference low-MAP runs >=60 s; any low display can be inherited',
        'duration_auc_definition': 'All valid reference bins; includes low runs shorter than 60 s',
        'initial_unavailable_display': 'Zero displayed burden; not a normal-pressure observation',
        'ci': {'method': 'Subject-cluster percentile bootstrap', 'replicates': REPLICATES,
               'base_seed': SEED, 'draw_identity': 'Frozen frequency_population hash, per threshold/interval',
               'reference_draw_identity': '5-minute frequency_population draw for each threshold',
               'event_percentage_intervals': 'Retained exactly from frozen episode-observability table'},
        'source_cohort_or_signal_modified': False, 'aki_models_refit': False}
    (args.root / 'analysis/qa/simple_interval_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(primary[reader_columns].to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
