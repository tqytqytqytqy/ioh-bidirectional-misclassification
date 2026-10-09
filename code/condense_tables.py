from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import re
from publication_label_revision import apply as apply_publication_labels

GROUPS = {
    1: [1, 27], 2: [7, 8, 9, 10, 11, 12], 3: [2, 24], 4: [3, 4],
    5: [6], 6: [23], 7: [21, 22], 8: [25, 35], 9: [28, 29],
    10: [13, 16], 11: [14, 15], 12: [17], 13: [18, 19, 20],
    14: [30, 31], 15: [32], 16: [33],
}
OLD_TO_NEW = {old: new for new, olds in GROUPS.items() for old in olds}
ARCHIVED = {5: 'implementation consistency check',
            26: 'complete absolute interval estimates',
            34: 'complete startup interval estimates'}
TITLES = {
    1: 'Cohort construction and recorded infusion-adjustment flow',
    2: 'NIBP record classification and measurement-cycle audit',
    3: 'Episode visibility across intervals and source of low display',
    4: 'Episode visibility by nadir and duration',
    5: 'Bidirectional deficit across thresholds and sampling intervals',
    6: 'Case-level variation under a 5 min display',
    7: 'Sensitivity to clustering and initial-display convention',
    8: 'Event-definition robustness before and after startup quality checks',
    9: 'Display information at infusion adjustments and event-definition sensitivity',
    10: 'AKI ascertainment and characteristics by evaluability',
    11: 'Sequential and sensitivity associations with postoperative AKI',
    12: 'Sequential associations with postoperative ICU stay of at least 2 days',
    13: 'ASA coding audit and outcome-model sensitivity',
    14: 'Cohort retention and episode visibility under startup quality rules',
    15: 'Infusion-adjustment results under startup quality rules',
    16: 'Postoperative associations under startup quality rules',
}


def numbers_text(numbers):
    numbers = sorted(set(numbers))
    runs = []
    for n in numbers:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts = []
    for run in runs:
        if len(run) >= 3:
            parts.append(f'S{run[0]} to S{run[-1]}')
        else:
            parts.extend(f'S{n}' for n in run)
    return parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


REF_PATTERN = re.compile(r'\bTables?\s+S\d+(?:\s*(?:-|–|to|and|,)\s*S?\d+)*')


def rewrite_refs(text):
    text = text.replace('document, Supplemental Digital Content 1, Table S26',
                        'Supplemental Digital Content 2, workbook sheet Legacy_S26')
    text = text.replace('Supplemental Digital Content 1, Table S26',
                        'Supplemental Digital Content 2, workbook sheet Legacy_S26')
    def substitute(match):
        nums = []
        for a, b in re.findall(r'S?(\d+)(?:\s*(?:-|–|to)\s*S?(\d+))?', match.group()):
            nums.extend(range(int(a), int(b)+1) if b else [int(a)])
        mapped = sorted({OLD_TO_NEW[n] for n in nums if n in OLD_TO_NEW})
        parts = []
        if mapped:
            parts.append(('Table ' if len(mapped) == 1 else 'Tables ') + numbers_text(mapped))
        for n in sorted(set(nums) & ARCHIVED.keys()):
            parts.append(f'the {ARCHIVED[n]} in Supplemental Digital Content 2 (workbook sheet Legacy_S{n:02d})')
        assert parts, match.group()
        return ' and '.join(parts)
    return REF_PATTERN.sub(substitute, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    base = Path(__file__).resolve().parents[1]
    source = args.source or base / '工作记录/source_tables.json'
    dest = args.output or base / '工作记录'
    dest.mkdir(parents=True, exist_ok=True)
    old = {t['old_number']: t for t in json.loads(source.read_text())}
    assert set(old) == set(range(1, 36))
    groups = []

    def panel(number, label='', rows=None, notes=None):
        t = old[number]
        return {'label': label, 'values': copy.deepcopy(rows if rows is not None else t['values']),
                'notes': copy.deepcopy(notes if notes is not None else t['notes']),
                'source_old_tables': [number]}

    def group(n, panels, notes=None):
        groups.append({'number': n, 'name': f'Supp_Table_S{n:02d}',
                       'title': f'Supplementary Table S{n}. {TITLES[n]}',
                       'source_old_tables': GROUPS[n], 'panels': panels,
                       'notes': notes or []})

    flow = copy.deepcopy(old[1]['values'])
    flow[0][2] = 'Count, n'
    flow[0][3] = 'Excluded, n'
    for row in old[27]['values'][1:]:
        if row[0] == 'Fixed primary cohort cases':
            continue
        branch = 'Primary cohort' if row[0] == 'Fixed primary cohort subjects' else 'Infusion adjustments'
        flow.append([branch, row[0], row[1], '', ''])
    group(1, [panel(1, rows=flow, notes=old[1]['notes'] + old[27]['notes'])])
    groups[-1]['panels'][0]['source_old_tables'] = [1, 27]

    record_rows = [['Record class', 'All records, n (%)', 'Paired records, n (%)', 'Interpretation']]
    for a, b in zip(old[7]['values'][1:], old[8]['values'][1:]):
        record_rows.append([a[0], f'{a[1]} ({a[2]})', f'{b[1]} ({b[2]})', a[3]])
    record = panel(7, 'a. Algorithm-record classification', rows=record_rows,
                   notes=old[7]['notes'] + old[8]['notes'])
    record['source_old_tables'] = [7, 8]
    diagnostics = copy.deepcopy(old[9]['values'])
    diagnostics.extend([old[10]['values'][2], old[10]['values'][3], old[10]['values'][5], old[11]['values'][3]])
    diagnostic = panel(9, 'b. Source-output and update diagnostics', rows=diagnostics,
                       notes=[
                           'Recorded timestamps describe device output, not verified cuff-cycle completion. Numerical changes do not enumerate all measurements, and unchanged output can follow a true new measurement.',
                           'The paired-record comparator is the median arterial MAP within -30 to +30 s, with at least 80% valid reference coverage. It is separate from the completed pre-adjustment bins used in infusion analysis.',
                           'Full channel-level diagnostics and field definitions are retained in workbook sheets Legacy_S09 to Legacy_S12 in Supplemental Digital Content 2. MAP, mean arterial pressure; NIBP, noninvasive blood pressure; SBP, systolic blood pressure; DBP, diastolic blood pressure.'])
    diagnostic['source_old_tables'] = [9, 10, 11, 12]
    group(2, [record, diagnostic])
    group(3, [panel(2, 'a. Visibility by threshold and interval'),
              panel(24, 'b. New and inherited low display at MAP <65 mmHg and 5 min')])
    group(4, [panel(3, 'a. Reference nadir at MAP <65 mmHg'), panel(4, 'b. Episode duration')])
    group(5, [panel(6)])
    groups[-1]['panels'][0]['notes'].append('All reported estimates are followed by 95% confidence intervals, including the two state-time columns.')
    group(6, [panel(23)])

    cluster_rows = [['Measure', 'Subject bootstrap: estimate (95% CI)', 'Case bootstrap: estimate (95% CI)']]
    clustered = {}
    for domain, metric, unit, n, estimate, ci in old[21]['values'][1:]:
        clustered.setdefault((domain, metric), {})[unit] = (n, estimate, ci)
    for (domain, metric), values in clustered.items():
        assert set(values) == {'Subject', 'Case'}
        assert values['Subject'][0] == '2380' and values['Case'][0] == '2435'
        cluster_rows.append([metric.capitalize(), *(f'{values[u][1]} ({values[u][2]})' for u in ['Subject', 'Case'])])
    clusters = panel(21, 'a. Bootstrap clustering unit', rows=cluster_rows,
                     notes=old[21]['notes'] + ['Subject resampling used 2380 clusters and case resampling used 2435 clusters. Detection, miss and opportunity entries are probabilities; pre-display AUC is a pooled reference-area fraction; HDR and ODR are reference-area ratios; net AUC difference is a signed relative fraction.'])
    initials = old[22]['values']
    area_rows = [['Initial-display convention', 'Excluded time', 'HDR', 'ODR', 'Net AUC difference, %']]
    episode_rows = [['Initial-display convention', 'Eligible episodes', 'Any low display, %', 'Complete miss, %', 'Pre-display AUC, %']]
    for r in initials[1:]:
        if r[0] == 'Deficit area':
            area_rows.append([r[1], r[2], *r[6:9]])
        else:
            episode_rows.append([r[1], *r[2:6]])
    group(7, [clusters, panel(22, 'b. Initial-display convention for deficit area', rows=area_rows, notes=[]),
              panel(22, 'c. Initial-display convention for episode visibility', rows=episode_rows)])

    robust_rows = [['Normal gap / minimum low time, s', 'Fresh sample, %\nOriginal', 'Inherited only, %\nOriginal',
                    'No low display, %\nOriginal', 'No low display, %\nSustained segments',
                    'No low display, %\nAll flagged bins', 'No low display, %\nExclude startup 10 min']]
    for original, qc in zip(old[25]['values'][1:], old[35]['values'][1:]):
        assert qc[0] == f'{original[0]} / {original[1]}'
        assert qc[1].split('\n')[0] == original[5]
        assert re.search(r'n=(\d+)', qc[1]).group(1) == original[2]
        robust_rows.append([qc[0], original[3], original[4], *qc[1:]])
    robustness = panel(25, rows=robust_rows, notes=[
        'MAP <65 mmHg; 5 min display; 30 phases. Percentage entries are estimates (95% subject-cluster bootstrap CI; 1000 replicates, seed 20260706); no-low-display cells also give reference episode count n. Gap allowance bridges only finite observed normal-pressure bins, never missing reference. Minimum duration counts cumulative low-pressure time, excluding normal gaps. Each definition has a different event population; row counts are not independent cohorts. Fresh and inherited-only entries use the original population. Startup rules preserve original phases and are defined in the startup-quality methods. Longer events are less often missed; these estimates do not describe treatment effects.'])
    robustness['source_old_tables'] = [25, 35]
    group(8, [robustness])
    group(9, [panel(28, 'a. Display states and persistence by sampling interval'),
              panel(29, 'b. Fixed event-definition sensitivities at 5 min')])
    group(10, [panel(13, 'a. Ascertainment and eligibility'), panel(16, 'b. Evaluable and nonevaluable cases')])
    group(11, [panel(14, 'a. Hidden-burden presence and sequential absolute-burden models'),
               panel(15, 'b. Secondary compositional and sensitivity models')])
    group(12, [panel(17)])
    group(13, [panel(18, 'a. ASA coding audit'), panel(19, 'b. AKI coding sensitivity'), panel(20, 'c. ICU coding sensitivity')])

    qc_rows = [['Measure', *old[31]['values'][0][1:]]]
    for col, label in [(1, 'Retained cases'), (2, 'Retained subjects'),
                       (3, 'Cases affected by masking or window exclusion'), (4, 'Cases excluded for coverage'),
                       (5, 'Removed valid-reference time before coverage exclusions, min'), (6, 'Retained reference AUC, mmHg min')]:
        qc_rows.append([label, *(r[col] for r in old[30]['values'][1:])])
    qc_rows.extend(copy.deepcopy(old[31]['values'][1:]))
    quality = panel(30, rows=qc_rows, notes=[
        'In the first 600 s after first finite reference MAP, flag 10 s bins with numeric MAP <40 mmHg, waveform coverage >=80% and p95-p05 <=5 mmHg. Sustained masking requires >=60 s consecutively flagged; all-bin masking has no run-length requirement. Both reapply >=80% whole-window reference coverage. Cases without source waveforms remain. Window exclusion retains the original cohort and can remove true early hypotension. These were algorithmic sensitivity rules.',
        'Affected cases and removed valid-reference time are counted before coverage exclusions. Results use MAP <65 mmHg, a 5 min display and 30 phases; episodes require >=60 s of continuous finite low MAP. Parentheses give 95% subject-cluster percentile CIs (1000 replicates). The final three rows are mutually exclusive and sum to 100% before rounding. Original grids and phases remain; masked values cannot update sampling, and missing reference breaks episodes. AUC, area under the deficit curve; MAP, mean arterial pressure.'])
    quality['source_old_tables'] = [30, 31]
    group(14, [quality])
    group(15, [panel(32)])
    group(16, [panel(33)])
    groups[-1]['panels'][0]['values'][0][0] = 'Outcome and adjustment\nRR (95% CI)'
    assert [g['number'] for g in groups] == list(range(1, 17))
    # Rewrite notes after construction where multiple source-table notes were combined.
    for g in groups:
        for p in g['panels']:
            for row in p['values']:
                assert len(row) == len(p['values'][0]), g['number']
            p['notes'] = list(dict.fromkeys(rewrite_refs(x) for x in p['notes']))
    apply_publication_labels(groups)
    mapping = {'current_supplement_tables': 16, 'original_supplement_tables': 35,
               'old_to_new': {str(k): v for k, v in OLD_TO_NEW.items()},
               'archived_only': {str(k): {'description': v, 'sheet': f'Legacy_S{k:02d}'} for k, v in ARCHIVED.items()},
               'groups': [{'new_number': g['number'], 'title': g['title'], 'old_numbers': g['source_old_tables']} for g in groups],
               'statistics_rerun': False, 'original_primary_analysis_retained': True}
    (dest / 'condensed_tables.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2) + '\n')
    (dest / 'table_map.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    for g in groups:
        for i, p in enumerate(g['panels'], 1):
            suffix = '' if len(g['panels']) == 1 else f'_panel_{i}'
            with (dest / f"{g['name']}{suffix}.csv").open('w', newline='') as f:
                csv.writer(f).writerows(p['values'])
    print(json.dumps({'current_tables': len(groups), 'table_panels': sum(len(g['panels']) for g in groups),
                      'published_data_rows': sum(len(p['values'])-1 for g in groups for p in g['panels']),
                      'archived_only_old_tables': list(ARCHIVED)}))


if __name__ == '__main__':
    main()
