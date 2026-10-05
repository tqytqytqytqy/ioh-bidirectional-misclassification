import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from event_extension import held_display, reference_events, classify_events

SEED = 20260706
REPLICATES = 1000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    root = args.output
    tables, figures, private, qa = [root / x for x in ('tables', 'figures', 'restricted', 'qa')]
    for p in (tables, figures, private, qa):
        p.mkdir(parents=True, exist_ok=True)
    source = args.analysis / 'outputs/intermediate'
    files = ['artmap_10s.parquet', 'vitaldb_manifest.parquet',
             'frequency_decomp_case_offset.parquet', 'episode_observability_case_5min.parquet']
    hashes = {f: sha(source / f) for f in files}
    art = pd.read_parquet(source / files[0])
    manifest = pd.read_parquet(source / files[1], columns=['case_id', 'subjectid'])
    ids = np.sort(art.case_id.unique())
    mapping = manifest.set_index('case_id').subjectid.reindex(ids)
    assert mapping.notna().all() and len(ids) == 2435
    subjects, subject_index = np.unique(mapping.to_numpy(), return_inverse=True)
    assert len(subjects) == 2380
    # Match the frozen primary episode bootstrap exactly; use common draws across new scenarios.
    # Frozen implementation used NumPy scalar repr; spell it explicitly for cross-version stability.
    group_identity = f'{SEED}|threshold=np.float64(65.0)|interval_min=np.float64(5.0)'
    group_seed = int.from_bytes(hashlib.sha256(group_identity.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(group_seed)
    draws = rng.integers(0, len(subjects), size=(REPLICATES, len(subjects)))
    weights = np.array([np.bincount(x, minlength=len(subjects)) for x in draws])

    def bootstrap_ratio(frame, columns, denominator):
        aligned = frame.reindex(ids, fill_value=0)
        arr = aligned[columns + [denominator]].to_numpy(float)
        sums = np.zeros((len(subjects), len(columns) + 1))
        np.add.at(sums, subject_index, arr)
        b = weights @ sums
        fractions = b[:, :-1] / b[:, -1:]
        estimate = arr[:, :-1].sum(0) / arr[:, -1].sum()
        return estimate, np.quantile(fractions, [.025, .975], axis=0)

    phase = pd.read_parquet(source / files[2], filters=[('threshold', '==', 65), ('interval_min', '==', 5)])
    phase['absolute_state_discordance_min_per_hour'] = (
        phase.normotensive_display_discordance_min + phase.hypotensive_display_discordance_min
    ) / phase.anesthesia_hours
    phase['low_reference_normal_display_min_per_hour'] = phase.normotensive_display_discordance_min / phase.anesthesia_hours
    phase['normal_reference_low_display_min_per_hour'] = phase.hypotensive_display_discordance_min / phase.anesthesia_hours
    phase['absolute_auc_disagreement_ratio'] = (phase.hidden_auc + phase.overdisplay_auc) / phase.true_auc.replace(0, np.nan)
    phase['net_auc_bias_percent'] = phase.net_bias_ratio * 100
    metrics = ['absolute_state_discordance_min_per_hour', 'low_reference_normal_display_min_per_hour',
               'normal_reference_low_display_min_per_hour', 'hidden_deficit_ratio',
               'overdisplay_deficit_ratio', 'absolute_auc_disagreement_ratio', 'net_auc_bias_percent']
    case = phase.groupby('case_id')[metrics].mean()
    spread = phase.groupby('case_id')[metrics].agg(lambda s: s.max() - s.min())
    distro = []
    for metric in metrics:
        for statistic, frame in [('Phase-averaged case value', case), ('Within-case phase range', spread)]:
            x = frame[metric].dropna()
            q = x.quantile([.05, .25, .5, .75, .95])
            distro.append({'Metric': metric, 'Summary': statistic, 'Cases': len(x), 'P5': q.iloc[0],
                           'Q1': q.iloc[1], 'Median': q.iloc[2], 'Q3': q.iloc[3], 'P95': q.iloc[4]})
    pd.DataFrame(distro).to_csv(tables / 'EJA_case_phase_distribution.csv', index=False)
    cohort_phase = phase.groupby('offset_sec')[['true_auc', 'hidden_auc', 'overdisplay_auc',
        'display_auc', 'normotensive_display_discordance_min', 'hypotensive_display_discordance_min', 'anesthesia_hours']].sum()
    cohort_phase['Hidden deficit percent'] = 100 * cohort_phase.hidden_auc / cohort_phase.true_auc
    cohort_phase['Overdisplay deficit percent'] = 100 * cohort_phase.overdisplay_auc / cohort_phase.true_auc
    cohort_phase['Net AUC bias percent'] = 100 * (cohort_phase.display_auc - cohort_phase.true_auc) / cohort_phase.true_auc
    cohort_phase['Absolute discordance min per hour'] = (
        cohort_phase.normotensive_display_discordance_min + cohort_phase.hypotensive_display_discordance_min
    ) / cohort_phase.anesthesia_hours
    cohort_phase.reset_index().to_csv(tables / 'EJA_cohort_phase_schedule.csv', index=False)
    case.to_parquet(private / 'case_phase_averages.parquet')
    spread.to_parquet(private / 'case_phase_ranges.parquet')

    sys.path.insert(0, str(args.analysis / 'src'))
    from ioh.estimands.decomposition import emulate_last_visible
    from ioh.estimands.episodes import detect_reference_episodes
    records, duration_records = [], []
    n_grid_reconciled = 0
    for num, (case_id, g) in enumerate(art.groupby('case_id', sort=True)):
        g = g.sort_values('time_sec')
        times, values = g.time_sec.to_numpy(), g.art_map.to_numpy()
        assert np.allclose(np.diff(times), 10)
        displays = [held_display(values, 30, p) for p in range(30)]
        # Compare all phases on a distributed sample against the original implementation.
        if num % 100 == 0 or num == len(ids) - 1:
            for p, (display, _) in enumerate(displays):
                old = emulate_last_visible(times, values, 300, p * 10)
                np.testing.assert_allclose(display, old, equal_nan=True)
                n_grid_reconciled += 1
        original = detect_reference_episodes(times, values, 65, min_duration_sec=60)
        baseline = reference_events(values, 65, 0, 6)
        assert [(e.start_index, e.end_index) for e in original] == [(s, e) for s, e, _ in baseline]
        for gap in (0, 1, 2):
            all_events = reference_events(values, 65, gap, 6)
            for minimum in (6, 18, 30):
                events = [e for e in all_events if e[2] >= minimum]
                for p, (display, held) in enumerate(displays):
                    result = classify_events(values, display, held, events, 65)
                    counts = np.bincount(result['category'], minlength=3)
                    row = {'case_id': case_id, 'gap_sec': gap * 10, 'minimum_low_sec': minimum * 10,
                        'offset_sec': p * 10, 'episodes': len(events), 'fresh': counts[0],
                        'inherited_only': counts[1], 'never_low': counts[2],
                        'carry_in': int(result['carry_in'].sum()), 'first_inherited': int(result['first_inherited'].sum())}
                    assert sum(counts) == len(events)
                    records.append(row)
                    if gap == 0 and minimum == 6:
                        lengths = np.array([e[2] for e in events])
                        for label, select in [('1 to <3 min', lengths < 18), ('3 to <5 min', (lengths >= 18) & (lengths < 30)),
                                              ('At least 5 min', lengths >= 30)]:
                            ct = np.bincount(result['category'][select], minlength=3)
                            duration_records.append({'case_id': case_id, 'Duration': label, 'episodes': int(select.sum()),
                                'fresh': ct[0], 'inherited_only': ct[1], 'never_low': ct[2]})
        if (num + 1) % 400 == 0:
            print(f'Event rules evaluated: {num + 1}/{len(ids)} cases', flush=True)

    rows = pd.DataFrame(records)
    rows.to_parquet(private / 'episode_case_phase_config.parquet', index=False)
    base = rows[(rows.gap_sec == 0) & (rows.minimum_low_sec == 60)]
    old = pd.read_parquet(source / files[3])
    old = old[old.threshold == 65].groupby('case_id')[['reference_episodes', 'detected_phase_episode_pairs']].sum().reindex(ids, fill_value=0)
    new = base.groupby('case_id')[['episodes', 'fresh', 'inherited_only']].sum().reindex(ids, fill_value=0)
    np.testing.assert_array_equal(new.episodes.to_numpy(), old.reference_episodes.to_numpy() * 30)
    np.testing.assert_allclose((new.fresh + new.inherited_only).to_numpy(), old.detected_phase_episode_pairs.to_numpy())
    assert new.episodes.sum() // 30 == 9187

    summaries = []
    cols = ['fresh', 'inherited_only', 'never_low', 'carry_in', 'first_inherited']
    for (gap, minimum), group in rows.groupby(['gap_sec', 'minimum_low_sec']):
        by_case = group.groupby('case_id')[cols + ['episodes']].sum()
        est, ci = bootstrap_ratio(by_case, cols, 'episodes')
        count = by_case.episodes.sum() // 30
        r = {'Gap allowance (s)': int(gap), 'Minimum cumulative low time (s)': int(minimum),
             'Reference episodes': int(count), 'Cases with episodes': int((by_case.episodes > 0).sum()),
             'Phase-episode pairs': int(by_case.episodes.sum())}
        for i, col in enumerate(cols):
            r[col + '_percent'] = 100 * est[i]
            r[col + '_lower95'] = 100 * ci[0, i]
            r[col + '_upper95'] = 100 * ci[1, i]
        summaries.append(r)
    robust = pd.DataFrame(summaries)
    robust.to_csv(tables / 'EJA_episode_robustness.csv', index=False)
    original_summary = pd.read_csv(args.analysis / 'outputs/tables/episode_observability_5min_overall.csv')
    original_primary = original_summary[original_summary.threshold == 65].iloc[0]
    new_primary = robust[(robust['Gap allowance (s)'] == 0) & (robust['Minimum cumulative low time (s)'] == 60)].iloc[0]
    np.testing.assert_allclose([new_primary.never_low_lower95 / 100, new_primary.never_low_upper95 / 100],
        [original_primary.complete_miss_ci_low, original_primary.complete_miss_ci_high], atol=1e-12)
    duration_summaries = []
    duration_frame = pd.DataFrame(duration_records)
    for label, group in duration_frame.groupby('Duration', sort=False):
        by_case = group.groupby('case_id')[['episodes', 'fresh', 'inherited_only', 'never_low']].sum()
        est, ci = bootstrap_ratio(by_case, ['fresh', 'inherited_only', 'never_low'], 'episodes')
        for i, col in enumerate(['fresh', 'inherited_only', 'never_low']):
            duration_summaries.append({'Duration': label, 'Reference episodes': int(by_case.episodes.sum() // 30),
                'Category': col, 'Percent': 100 * est[i], 'Lower 95%': 100 * ci[0, i], 'Upper 95%': 100 * ci[1, i]})
    duration_table = pd.DataFrame(duration_summaries)
    duration_table.to_csv(tables / 'EJA_new_vs_inherited_by_duration.csv', index=False)
    event_case = base.groupby('case_id')[cols + ['episodes']].sum()
    event_case['miss_fraction'] = event_case.never_low / event_case.episodes.replace(0, np.nan)
    case_event_quantiles = event_case.miss_fraction.dropna().quantile([.05, .25, .5, .75, .95]).to_dict()
    phase_event = base.groupby('offset_sec')[['episodes', 'fresh', 'inherited_only', 'never_low']].sum()
    for col in ['fresh', 'inherited_only', 'never_low']:
        phase_event[col + '_percent'] = 100 * phase_event[col] / phase_event.episodes
    phase_event.reset_index().to_csv(tables / 'EJA_event_phase_schedule.csv', index=False)
    by_case_phase_miss = base[base.episodes > 0].copy()
    by_case_phase_miss['miss'] = by_case_phase_miss.never_low / by_case_phase_miss.episodes
    ranges = by_case_phase_miss.groupby('case_id')['miss'].agg(lambda s: s.max() - s.min())
    phase_range_quantiles = ranges.quantile([.05, .25, .5, .75, .95]).to_dict()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), constrained_layout=True)
    for col, label, color in [(metrics[1], 'Low reference, normal display', '#0072B2'),
                             (metrics[2], 'Normal reference, low display', '#D55E00'),
                             (metrics[0], 'Either direction', '#333333')]:
        x = np.sort(case[col].to_numpy())
        axes[0].plot(x, np.arange(1, len(x) + 1) / len(x) * 100, label=label, color=color)
    axes[0].set(xlabel='Phase-averaged discordance (min / anaesthesia-hour)', ylabel='Cumulative percentage of cases', title='A  Distribution across cases')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    axes[1].hist(ranges.to_numpy() * 100, bins=np.linspace(0, 100, 21), color='#009E73', edgecolor='white')
    axes[1].set(xlabel='Within-case range of missed events across phases (pp)', ylabel='Number of cases', title='B  Sensitivity to sampling phase')
    for ext in ('png', 'pdf', 'tiff'):
        fig.savefig(figures / f'Figure_4_case_phase_distribution.{ext}', dpi=600)
    plt.close(fig)

    labels = ['1 to <3 min', '3 to <5 min', 'At least 5 min']
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    bottom = np.zeros(3)
    for col, title, color in [('fresh', 'Fresh low sample within this event', '#0072B2'),
                              ('inherited_only', 'Inherited low display only', '#E69F00'),
                              ('never_low', 'No low display during this event', '#777777')]:
        sub = duration_table[duration_table.Category == col].set_index('Duration').reindex(labels)
        x = sub.Percent.to_numpy()
        ax.bar(labels, x, bottom=bottom, color=color, label=title, width=.6)
        for i, v in enumerate(x):
            if v >= 4:
                ax.text(i, bottom[i] + v / 2, f'{v:.1f}%', ha='center', va='center', color='white' if col != 'inherited_only' else 'black')
        bottom += x
    ns = duration_table.drop_duplicates('Duration').set_index('Duration')['Reference episodes']
    ax.set_xticks(range(3), [f'{s}\nn = {ns[s]:,}' for s in labels])
    ax.set(ylabel='Phase-averaged percentage of reference events', ylim=(0, 100))
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.21), fontsize=9)
    for ext in ('png', 'pdf', 'tiff'):
        fig.savefig(figures / f'Figure_5_fresh_vs_inherited.{ext}', dpi=600, bbox_inches='tight')
    plt.close(fig)

    ref = robust[(robust['Gap allowance (s)'] == 0) & (robust['Minimum cumulative low time (s)'] == 60)].iloc[0].to_dict()
    payload = {'seed': SEED, 'bootstrap_replicates': REPLICATES, 'bootstrap_cluster': 'subjectid',
        'bootstrap_group_identity': group_identity,
        'bootstrap_baseline_ci_reconciliation': 'PASS within 1e-12',
        'cases': len(ids), 'subjects': len(subjects), 'baseline': ref,
        'case_miss_fraction_quantiles': {str(k): v for k, v in case_event_quantiles.items()},
        'within_case_phase_miss_range_quantiles': {str(k): v for k, v in phase_range_quantiles.items()},
        'phase_cohort_net_auc_bias_range_percent': [cohort_phase['Net AUC bias percent'].min(), cohort_phase['Net AUC bias percent'].max()],
        'global_phase_miss_range_percent': [phase_event.never_low_percent.min(), phase_event.never_low_percent.max()],
        'source_sha256': hashes, 'vector_display_reconciliations': n_grid_reconciled,
        'baseline_event_reconciliation': 'PASS: all 2435 cases, all 30 phases',
        'source_startup_quality_adjudicated': False,
        'new_analysis_prespecified': False, 'public_release_updated': False}
    (qa / 'analysis_summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    assert hashes == {f: sha(source / f) for f in files}
    print(json.dumps({k: payload[k] for k in ['cases', 'subjects', 'baseline', 'case_miss_fraction_quantiles', 'within_case_phase_miss_range_quantiles']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
