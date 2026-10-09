from __future__ import annotations
import os
import ast
import hashlib
import json
import sys
from pathlib import Path
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import numpy as np
import pandas as pd
OUT = Path(os.environ.get('IOH_QC_OUTPUT_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', str(OUT.parent))).expanduser().resolve()
pipeline_path = ROOT / 'analysis_v10_subject_phase_release/src/ioh/pipeline.py'
tree = ast.parse(pipeline_path.read_text())
node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_frequency_population_table'))
namespace = {'np': np, 'pd': pd, 'hashlib': hashlib}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(pipeline_path), 'exec'), namespace)
_frequency_population_table = namespace['_frequency_population_table']
AGG = OUT / 'analysis/aggregate'
POLICIES = ['original', 'qc60', 'qc_all', 'startup10']
LABELS = {'original': 'Original', 'qc60': 'Sustained low-amplitude segments', 'qc_all': 'All flagged low-amplitude bins', 'startup10': 'Exclude reference startup 10 min'}
SHORT = {'original': 'Original', 'qc60': 'Sustained segments', 'qc_all': 'All flagged bins', 'startup10': 'Exclude startup 10 min'}

def ci(row, estimate='estimate', lo='lower95', hi='upper95', digits=1, scale=1):
    return f'{row[estimate] * scale:.{digits}f} ({row[lo] * scale:.{digits}f} to {row[hi] * scale:.{digits}f})'

def main():
    mapping = pd.read_parquet(ROOT / 'analysis_v10_subject_phase_release/outputs/intermediate/vitaldb_manifest.parquet', columns=['case_id', 'subjectid'])
    freq = []
    for policy in POLICIES:
        phase = pd.read_parquet(OUT / f'data_restricted/frequency_decomp_{policy}.parquet')
        means = phase.groupby(['case_id', 'threshold', 'interval_min']).mean(numeric_only=True).reset_index()
        f = _frequency_population_table({'project': {'seed': 20260706}, 'analysis': {'bootstrap_reps': 1000}}, means, case_to_cluster=mapping, cluster_column='subjectid')
        f.insert(0, 'policy', policy)
        freq.append(f)
    freq = pd.concat(freq, ignore_index=True)
    freq.to_csv(AGG / 'qc_frequency_population.csv', index=False)
    old = pd.read_csv(ROOT / 'analysis_v10_subject_phase_release/outputs/tables/primary_auc_by_threshold_interval.csv')
    a = freq[freq.policy.eq('original')].set_index(['threshold', 'interval_min'])
    b = old.set_index(['threshold', 'interval_min'])
    cols = ['HDR', 'ODR', 'NetBias', 'HDR_ci_low', 'HDR_ci_high', 'ODR_ci_low', 'ODR_ci_high']
    np.testing.assert_allclose(a.loc[b.index, cols], b[cols], atol=1e-10, rtol=1e-10)
    cohorts = pd.read_csv(AGG / 'cohort_qc_summary.csv').set_index('policy')
    events = {p: pd.read_csv(AGG / f'events_rates_{p}.csv').query('threshold==65 and interval_min==5').iloc[0] for p in POLICIES}
    treatment = json.loads((AGG / 'treatment_summary.json').read_text())['policies']
    models = pd.read_csv(AGG / 'outcomes_all_models.csv')
    tables = []

    def add(n, title, values, notes, source):
        tables.append({'name': f'Supp_Table_S{n}', 'title': f'Supplementary Table S{n}. ' + title, 'values': values, 'notes': notes, 'source': source})
    rules = 'Startup means the first 600 s after the first finite arterial-reference MAP. A flagged bin has numeric MAP <40 mmHg, waveform coverage >=80%, and a 5th-to-95th percentile waveform span <=5 mmHg. Sustained segments require >=60 s continuously flagged; all flagged bins has no duration requirement. These are rule-based technical sensitivities, without completed clinical adjudication.'
    add(30, 'Cohort retention under startup signal-quality rules', [['Rule', 'Retained cases', 'Retained subjects', 'Cases with masked values', 'Coverage exclusions', 'Masked time (min)', 'Retained reference AUC (mmHg min)']] + [[SHORT[p], int(cohorts.loc[p, 'cases']), int(cohorts.loc[p, 'subjects']), int(cohorts.loc[p, 'affected_cases_before_coverage']), int(cohorts.loc[p, 'excluded_coverage_cases']), round(cohorts.loc[p, 'masked_minutes_before_coverage'], 1), round(cohorts.loc[p, 'reference_auc_retained'], 1)] for p in POLICIES], [rules, 'Masked time is counted before coverage exclusions. For the first two masking rules, whole-case reference coverage must still be >=80%. The 10 min window sensitivity retains the original cohort. Cases without a source waveform are retained. AUC, area under the deficit curve; MAP, mean arterial pressure.'], 'cohort_qc_summary.csv')
    rows = [['Measure'] + [SHORT[p] for p in POLICIES]]
    for label, key in [('Cases with qualifying episodes', 'cases_with_episodes'), ('Reference episodes', 'reference_episodes')]:
        rows.append([label] + [int(events[p][key]) for p in POLICIES])
    for label, key in [('Hidden-deficit ratio', 'HDR'), ('Overdisplay-deficit ratio', 'ODR'), ('Net AUC difference (%)', 'NetBias')]:
        rows.append([label] + [ci(freq.query('policy==@p and threshold==65 and interval_min==5').iloc[0], key, key + '_ci_low', key + '_ci_high', 3 if key != 'NetBias' else 2, 100 if key == 'NetBias' else 1) for p in POLICIES])
    for label, key in [('Episodes with no low display (%)', 'never_low'), ('Episodes with new low samples (%)', 'fresh'), ('Inherited low display only (%)', 'inherited_only')]:
        rows.append([label] + [ci(events[p], key + '_percent', key + '_lower95', key + '_upper95') for p in POLICIES])
    add(31, 'Bidirectional deficit and event visibility after startup quality checks', rows, ['MAP <65 mmHg, 5 min display, all 30 phases. Qualifying events require >=60 s of continuous finite low reference pressure. Values in parentheses are 95% subject-cluster percentile confidence intervals (1000 replicates). The last three rows are mutually exclusive and sum to 100% before rounding.', 'Rules are defined in Table S30. Original global sampling phases and anaesthesia durations are retained. Masked values cannot initialise or update the held display. Missing reference bins break episodes.'], 'qc_frequency_population.csv; events_rates_comparative.csv')
    rows = [['Measure'] + [SHORT[p] for p in POLICIES]]
    for label, key in [('Quality-eligible adjustment blocks', 'Quality-eligible blocks'), ('Low-reference adjustments', 'Low-reference blocks at adjustment'), ('Low-reference subjects', 'Low-reference subjects'), ('Confirmed recoveries', 'Blocks with confirmed reference recovery')]:
        rows.append([label] + [int(treatment[p]['primary_flow'][key]) for p in POLICIES])
    for interval in [1, 2.5, 5]:
        vals = []
        for p in POLICIES:
            d = pd.read_csv(AGG / f'treatment_display_{p}.csv')
            r = d[(d.scenario == 'Primary') & (d.interval_min == interval) & (d.category == 'normal_display')].iloc[0]
            vals.append(ci(r))
        rows.append([f'Normal display at {interval:g} min (%)'] + vals)
    rows.append(['Recovery persistence at 5 min (min)'] + [ci(next((r for r in treatment[p]['primary_recovery'] if r['interval_min'] == 5)), digits=2) for p in POLICIES])
    add(32, 'Display information at infusion adjustments under startup quality rules', rows, ['Values in parentheses are 95% subject-cluster bootstrap confidence intervals (2000 replicates). Infusion event definitions, local pre-adjustment support and recovery rules are unchanged. The denominator for display states is the low-reference adjustment count in each column.', 'Persistence averages all confirmed recovery events after phase averaging. Normal or unavailable display at recovery contributes zero; lack of recovery is not assigned zero. The duration is restricted to the remainder of the 5 min post-adjustment window. Rules are defined in Table S30.'], 'treatment_summary.json; treatment_display_*.csv; treatment_recovery_*.csv')
    rows = [['Outcome and adjustment'] + [SHORT[p] for p in POLICIES]]
    keys = []
    for family in ['aki_sequential', 'icu_sequential']:
        orig = models[(models.policy == 'original') & (models.model_family == family)]
        for _, r in orig.iterrows():
            if len(keys) >= 20:
                raise AssertionError('Unexpected model schema')
            adjustment = {'none': 'Unadjusted', 'clinical': 'Clinical adjustment', 'clinical_plus_reference': 'Clinical and total-reference-burden adjustment'}[r.adjustment_sequence]
            label = ('AKI' if family.startswith('aki') else 'ICU >=2 days') + '\n' + adjustment
            cells = []
            for p in POLICIES:
                sub = models[(models.policy == p) & (models.model_family == family) & (models.analysis == r.analysis)].iloc[0]
                cells.append(ci(sub, 'relative_risk', 'ci_low', 'ci_high', 3) + f'\nN={int(sub.n_cases)}; events={int(sub.events)}')
            rows.append([label] + cells)
            keys.append((family, r.analysis))
    add(33, 'Sequential postoperative associations under startup quality rules', rows, ['Relative risks (95% subject-cluster robust confidence intervals) per 10 mmHg min per original anaesthesia-hour greater absolute hidden deficit. Each cell gives the actual model case count and outcome event count. Clinical adjustment and nonlinear total-reference-burden adjustment use the original covariates and spline specification.', 'AKI is creatinine-defined through 7 postoperative days or discharge. ICU stay >=2 days is a resource-use endpoint; admission intent is unavailable. Models are exploratory associations. Sensitivities for ASA coding, baseline definition and other original specifications are retained in the unrounded workbook output.'], 'outcomes_all_models.csv')
    rows = [['Rule', 'Interval (min)', 'Visible events (expected n)', 'Visible events (%)', 'Low-display time (min)', 'Displayed AUC (mmHg min)', 'TWA (mmHg)']]
    for p in POLICIES:
        f = pd.read_csv(AGG / f'events_interval_absolute_summary_{p}.csv')
        f = f[f['Threshold (mmHg)'].eq(65) & pd.to_numeric(f['Sampling interval (min)'], errors='coerce').isin([1, 2.5, 5])]
        assert len(f) == 3
        for _, r in f.iterrows():
            rows.append([SHORT[p], float(r['Sampling interval (min)']), round(r['Events with low display (expected n)'], 1), round(r['Events with low display (%)'], 1), round(r['Total hypotension duration (min)'], 1), round(r['Total deficit AUC (mmHg min)'], 1), round(r['Pooled TWA deficit (mmHg)'], 3)])
    add(34, 'Absolute interval summaries under startup quality rules', rows, ['MAP <65 mmHg. Event counts require >=60 s; low duration and deficit AUC include all finite low-pressure bins, including shorter runs. TWA divides pooled AUC by retained valid-reference time within each rule, held constant across intervals. Reference time therefore differs between rules, not between intervals within a rule.', 'Expected event counts average all sampling phases. The workbook contains unrounded values and subject-bootstrap intervals for all five intervals and all three thresholds; this table selects 1, 2.5 and 5 min. AUC, area under the deficit curve; TWA, time-weighted average.'], 'events_interval_absolute_summary_comparative.csv')
    robust = {p: pd.read_csv(AGG / f'events_robustness_{p}.csv') for p in POLICIES}
    rows = [['Gap allowance / minimum low time (s)'] + [SHORT[p] for p in POLICIES]]
    for _, r in robust['original'].iterrows():
        gap, minlow = (r['Gap allowance (s)'], r['Minimum cumulative low time (s)'])
        cells = []
        for p in POLICIES:
            x = robust[p][robust[p]['Gap allowance (s)'].eq(gap) & robust[p]['Minimum cumulative low time (s)'].eq(minlow)].iloc[0]
            cells.append(ci(x, 'never_low_percent', 'never_low_lower95', 'never_low_upper95') + f'\nn={int(x.reference_episodes)}')
        rows.append([f'{int(gap)} / {int(minlow)}'] + cells)
    add(35, 'Event-definition robustness under startup quality rules', rows, ['Complete-miss percentage (95% subject-cluster bootstrap confidence interval) and reference event count n. MAP <65 mmHg, 5 min display. Gap allowance joins only finite normal-pressure bins; missing reference is never bridged. Minimum duration uses cumulative low time and excludes bridged normal bins.', 'Each row defines a different event population. All policies retain original global sampling phases; the rules are defined in Table S30.'], 'events_robustness_comparative.csv')
    (AGG / 'qc_publication_tables.json').write_text(json.dumps(tables, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for t in tables:
        pd.DataFrame(t['values'][1:], columns=t['values'][0]).to_csv(AGG / (t['name'] + '.csv'), index=False)
    (OUT / 'qa/qc_publication_tables_verification.json').write_text(json.dumps({'tables': len(tables), 'baseline_frequency_point_estimates_and_CIs_match': True, 'primary_results_replaced': False}, indent=2) + '\n')
    print(freq.query('threshold==65 and interval_min==5')[['policy', 'HDR', 'ODR', 'NetBias', 'HDR_ci_low', 'HDR_ci_high']].to_json(orient='records', indent=2))
if __name__ == '__main__':
    main()
