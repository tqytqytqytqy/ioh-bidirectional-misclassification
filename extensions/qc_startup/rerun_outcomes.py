"""Reproduce frozen AKI/ICU models under prespecified technical QC policies.

All writes are outcomes-prefixed files within the independent QC workspace.
No clinical adjudication, model selection, or external publication is performed.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode = True
OUT = Path(os.environ.get('IOH_QC_OUTPUT_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
ROOT = Path(os.environ.get('IOH_PROJECT_ROOT', str(OUT.parent))).expanduser().resolve()
FROZEN = ROOT / 'analysis_v10_subject_phase_release'
DATA = FROZEN / 'outputs/intermediate'
TABLES = FROZEN / 'outputs/tables'
SOURCE = ROOT / 'EJA_投稿文件包_20261006_最终文字修订/03_可复现材料'
PRIVATE = OUT / 'data_restricted'
AGG = OUT / 'analysis/aggregate'
QA = OUT / 'qa'
os.environ.setdefault('MPLCONFIGDIR', str(QA / 'outcomes_matplotlib'))
for path in (SOURCE / 'scripts', SOURCE / 'src', QA / 'outcomes_runtime', Path(os.environ.get('IOH_QC_DEPENDENCIES', ''))):
    sys.path.insert(0, str(path))
import numpy as np
import pandas as pd
import statsmodels
from ioh.aki_outcomes import aggregate_invisibility_features, build_creatinine_aki_outcomes, clean_asa_for_adjustment, derive_last_preoperative_creatinine, fit_modified_poisson, summarize_hidden_presence
from ioh.icu_analysis import fit_sequential_icu_models
from ioh.icu_outcomes import build_prolonged_icu_outcome, summarize_icu_gate
from ioh.estimands.decomposition import decompose_deficit, emulate_last_visible
from ioh.revision_v7 import case_phase_averaged_episode_rows
POLICIES = ('original', 'qc60', 'qc_all', 'startup10')
MODEL_FILES = {'aki_sequential': 'exploratory_aki_sequential_models.csv', 'aki_incremental': 'exploratory_aki_incremental_models.csv', 'aki_asa': 'exploratory_aki_asa_coding_sensitivity.csv', 'icu_sequential': 'exploratory_icu_sequential_models.csv', 'icu_asa': 'exploratory_icu_asa_coding_sensitivity.csv'}
EXPOSURE_COLUMNS = {'true_auc', 'hidden_auc', 'concordant_auc', 'anesthesia_hours', 'true_hypotension_min', 'true_twa', 'hidden_twa', 'concordant_twa', 'hdr', 'true_twa65', 'hidden_twa65', 'concordant_twa65', 'hdr65', 'reference_episodes', 'expected_missed_episodes', 'phase_episode_pairs', 'detected_with_60s_remaining_pairs', 'detected_with_120s_remaining_pairs', 'reference_episode_auc_phase_sum', 'pre_detection_reference_auc_sum', 'episode_complete_miss_probability', 'episode_1min_opportunity_probability', 'episode_2min_opportunity_probability', 'predisplay_auc_fraction', 'log_true_twa', 'hdr10', 'predisplay10', 'hidden_twa10'}
CHECK_FIELDS = ('n_cases', 'n_subjects', 'events', 'n_parameters', 'coefficient', 'standard_error', 'relative_risk', 'ci_low', 'ci_high', 'p_value', 'aic', 'log_likelihood', 'formula', 'term')

def load_aki_script():
    path = SOURCE / 'scripts/33_run_exploratory_aki_analysis_v7.py'
    tree = ast.parse(path.read_text())
    names = {'_analysis_frame', '_missingness_comparison'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError('Frozen source helper functions are missing')
    namespace = {'pd': pd, 'np': np, 'clean_asa_for_adjustment': clean_asa_for_adjustment}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def json_write(path, value):
    clean = json.loads(pd.Series([value]).to_json(orient='values', force_ascii=False))[0]
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def validate_panel(panel, original, policy):
    keys = ['case_id', 'time_sec']
    if panel.duplicated(keys).any():
        raise ValueError('Panel contains duplicate case-time rows')
    ids = set(panel.case_id)
    original_ids = set(original.case_id)
    if not ids <= original_ids:
        raise ValueError('Panel contains cases outside the frozen cohort')
    if policy in ('original', 'startup10') and ids != original_ids:
        raise ValueError('Original and startup10 must retain the original cases')
    left = panel.sort_values(keys).reset_index(drop=True)
    right = original[original.case_id.isin(ids)].sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError('Original time grid was dropped, shifted, or resampled')
    a, b = (left.art_map.to_numpy(float), right.art_map.to_numpy(float))
    if np.any(np.isfinite(a) & (~np.isfinite(b) | (a != b))):
        raise ValueError('Finite reference values changed instead of being masked')
    masked = np.isfinite(b) & ~np.isfinite(a)
    if policy == 'original' and masked.any():
        raise ValueError('Original policy is not identical to frozen input')
    return {'policy': policy, 'n_cases': len(ids), 'excluded_cases': len(original_ids - ids), 'n_grid_bins': len(panel), 'newly_masked_bins': int(masked.sum()), 'grid_preserved': True, 'unmasked_reference_unchanged': True}

def assert_frozen_columns(frame, original):
    columns = [c for c in original if c in frame and c not in EXPOSURE_COLUMNS]
    actual = frame.set_index('case_id').sort_index()
    expected = original.set_index('case_id').loc[actual.index]
    columns.remove('case_id')
    pd.testing.assert_frame_equal(actual[columns], expected[columns], check_dtype=False, check_exact=True, check_names=True)

def check_duration_and_reference(frame, original):
    current = frame.set_index('case_id').sort_index()
    frozen = original.set_index('case_id').loc[current.index]
    for column in ('anesthesia_hours', 'duration_hr'):
        np.testing.assert_allclose(current[column], frozen[column], atol=0, rtol=0)
    np.testing.assert_allclose(current.true_twa, current.true_auc / current.anesthesia_hours, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(current.log_true_twa, np.log1p(current.true_twa), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(current.hidden_twa10, current.hidden_auc / current.anesthesia_hours / 10, atol=1e-12, rtol=1e-12)
    changed = ~np.isclose(current.true_auc, frozen.true_auc, atol=1e-10, rtol=1e-10)
    return {'n_cases': len(current), 'original_duration_denominators_preserved': True, 'original_duration_covariate_preserved': True, 'policy_specific_reference_used_for_total_burden_adjustment': True, 'changed_reference_burden_cases': int(changed.sum()), 'exposure_duration_definition': 'original frozen 10-second grid bins * 10 / 3600', 'duration_covariate_definition': 'original frozen manifest duration_min / 60', 'total_reference_adjustment': 'bs(log(1 + policy-specific true_auc / original anesthesia_hours), df=3, degree=2)', 'valid_reference_time_not_used_as_duration_denominator': True}

def derive_case_features(args):
    case_id, times, reference, need_frequency, need_episodes = args
    frequency = []
    if need_frequency:
        displays = [emulate_last_visible(times, reference, 300, offset) for offset in range(0, 300, 10)]
        for threshold in (65.0, 60.0, 55.0):
            parts = [decompose_deficit(reference, display, threshold, 10 / 60) for display in displays]
            frequency.append({'case_id': case_id, 'threshold': threshold, 'interval_min': 5.0, 'anesthesia_hours': len(times) / 360, **{column: float(np.mean([p[column] for p in parts])) for column in ('true_auc', 'hidden_auc', 'concordant_auc', 'true_hypotension_min')}})
    episodes = case_phase_averaged_episode_rows(case_id=case_id, times=times, reference=reference, thresholds=[65.0, 60.0, 55.0], intervals_min=[5.0], min_duration_sec=60.0, step_sec=10) if need_episodes else pd.DataFrame()
    return (frequency, episodes.to_dict('records'))

def normalize_frequency(frequency):
    frequency = frequency[frequency.interval_min.eq(5.0)].copy()
    keys = ['case_id', 'threshold', 'interval_min']
    if 'offset_sec' in frequency:
        if frequency.duplicated(keys + ['offset_sec']).any():
            raise ValueError('Duplicate frequency phase rows')
        groups = frequency.groupby(keys, sort=False)
        offsets = groups.offset_sec.agg(lambda values: set(values))
        if not offsets.map(lambda values: values == set(range(0, 300, 10))).all():
            raise ValueError('Expected all 30 original 5-minute phases')
        columns = [c for c in frequency.select_dtypes(include='number') if c not in keys + ['offset_sec']]
        frequency = groups[columns].mean().reset_index()
    return frequency

def replay_panel(panel, workers=4, need_frequency=True, need_episodes=True):
    jobs = [(case, sub.sort_values('time_sec').time_sec.to_numpy(float), sub.sort_values('time_sec').art_map.to_numpy(float), need_frequency, need_episodes) for case, sub in panel.groupby('case_id', sort=True)]
    frequency, episodes = ([], [])
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (freq, epi) in enumerate(pool.map(derive_case_features, jobs, chunksize=8), 1):
            frequency.extend(freq)
            episodes.extend(epi)
            if i % 500 == 0 or i == len(jobs):
                print(f'Feature replay {i}/{len(jobs)} cases; {time.monotonic() - started:.1f}s', flush=True)
    return (pd.DataFrame(frequency), pd.DataFrame(episodes))

def verify_baseline_features(original, workers):
    frozen_frequency = pd.read_parquet(DATA / 'frequency_decomp_case_mean_v7.parquet')
    frozen_episodes = pd.read_parquet(DATA / 'episode_observability_case_5min.parquet')
    ids = np.array(sorted(original.case_id.unique()))
    sample_ids = ids[np.linspace(0, len(ids) - 1, 32, dtype=int)]
    frequency, episodes = replay_panel(original[original.case_id.isin(sample_ids)], workers)
    rows = []
    for threshold in (65.0, 60.0, 55.0):
        observed = aggregate_invisibility_features(frequency, episodes, threshold=threshold)
        expected = aggregate_invisibility_features(frozen_frequency, frozen_episodes, threshold=threshold)
        expected = expected[expected.case_id.isin(sample_ids)]
        check = compare_tables(observed, expected, f'baseline_features_{threshold}', key='case_id', fields=[c for c in observed if c != 'case_id'])
        rows.append(check)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(QA / 'outcomes_baseline_feature_verification.csv', index=False)
    if not result.passed.all():
        raise AssertionError('Independent baseline feature replay did not match frozen values')
    return (frozen_frequency, frozen_episodes)

def compare_tables(observed, expected, label, key='analysis', fields=CHECK_FIELDS):
    if observed[key].duplicated().any() or expected[key].duplicated().any():
        raise ValueError('Comparison keys must be unique')
    observed, expected = (observed.set_index(key).sort_index(), expected.set_index(key).sort_index())
    if not observed.index.equals(expected.index):
        return pd.DataFrame([{'table': label, 'field': 'row_keys', 'passed': False, 'max_absolute_difference': np.nan}])
    rows = []
    for column in fields:
        if column not in expected:
            continue
        if column not in observed:
            rows.append({'table': label, 'field': column, 'passed': False})
            continue
        a, b = (observed[column], expected[column])
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            aa, bb = (a.to_numpy(float, na_value=np.nan), b.to_numpy(float, na_value=np.nan))
            integer = column in ('n_cases', 'n_subjects', 'events', 'n_parameters')
            passed = np.allclose(aa, bb, atol=0 if integer else 1e-08, rtol=0 if integer else 1e-10, equal_nan=True)
            delta = np.abs(aa - bb)
            difference = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
        else:
            passed = a.fillna('').equals(b.fillna(''))
            difference = np.nan
        rows.append({'table': label, 'field': column, 'passed': bool(passed), 'max_absolute_difference': difference, 'n_rows': len(observed)})
    return pd.DataFrame(rows)

def publication_model_matches(estimate, displayed):
    n = int(str(displayed['Cases']).replace(',', ''))
    events = int(displayed.get('AKI', displayed.get('Events')))
    effect = displayed.get('RR (95% CI)', displayed.get('Adjusted RR (95% CI)'))
    expected_effect = f"{estimate['relative_risk']:.2f} ({estimate['ci_low']:.2f} to {estimate['ci_high']:.2f})"
    expected_p = '<0.001' if estimate['p_value'] < 0.001 else f"{estimate['p_value']:.3f}"
    return n == estimate['n_cases'] and events == estimate['events'] and (effect == expected_effect) and (str(displayed['P value']) == expected_p)

def verify_publication_baseline(results):
    specifications = {'S14': 'aki_sequential', 'S15': 'aki_incremental', 'S17': 'icu_sequential', 'S19': 'aki_asa', 'S20': 'icu_asa'}
    rows = []
    for table, family in specifications.items():
        path = SOURCE / f'outputs/tables/Supp_Table_{table}.csv'
        display = pd.read_csv(path, dtype=str)
        for published in display.to_dict('records'):
            label = published.get('Analysis', published.get('Model'))
            if label == 'Any hidden burden versus none':
                continue
            if table == 'S17':
                label = {'Unadjusted': 'Absolute hidden burden, unadjusted', 'Clinical adjusted': 'Absolute hidden burden, clinical adjusted', 'Clinical plus reference burden': 'Absolute hidden burden, clinical plus reference burden'}[label]
            if label.startswith('Exclude baseline creatinine'):
                label = 'Exclude baseline creatinine >=4 mg/dL'
            if table in ('S19', 'S20'):
                label += ' (recorded ASA sensitivity)'
            match = results[family].loc[results[family].analysis.eq(label)]
            passed = len(match) == 1 and publication_model_matches(match.iloc[0].to_dict(), published)
            rows.append({'publication_table': table, 'analysis': label, 'passed': passed})
    verified = pd.DataFrame(rows)
    verified.to_csv(QA / 'outcomes_latest_publication_verification.csv', index=False)
    if not verified.passed.all() or len(verified) != 20:
        raise AssertionError('Latest EJA formatted outcome tables differ from baseline replay')

def prepare_frozen_outcomes(manifest, script):
    labs_setting = os.environ.get('IOH_VITALDB_LABS', '')
    if not labs_setting:
        raise ValueError('Set IOH_VITALDB_LABS to the frozen official labs.csv input')
    labs_path = Path(labs_setting).expanduser()
    lab_audit = pd.read_csv(TABLES / 'aki_labs_source_audit.csv').iloc[0]
    if sha256(labs_path) != lab_audit.input_sha256:
        raise AssertionError('Laboratory input hash differs from frozen outcome ascertainment')
    labs = pd.read_csv(labs_path, usecols=['caseid', 'dt', 'name', 'result'])
    outcomes = pd.read_parquet(DATA / 'aki_creatinine_case_v7.parquet')
    rebuilt = build_creatinine_aki_outcomes(manifest, labs)
    pd.testing.assert_frame_equal(outcomes.sort_values('case_id').reset_index(drop=True), rebuilt.sort_values('case_id').reset_index(drop=True), check_dtype=False, check_exact=True)
    window48 = outcomes.copy()
    evaluable = window48.baseline_creatinine_mg_dl.gt(0) & window48.postop_cr_max_48h.notna()
    aki48 = pd.Series(pd.NA, index=window48.index, dtype='boolean')
    aki48.loc[evaluable] = (window48.postop_cr_max_48h.sub(window48.baseline_creatinine_mg_dl).ge(0.3) | window48.postop_cr_max_48h.div(window48.baseline_creatinine_mg_dl).ge(1.5)).loc[evaluable]
    window48['aki_creatinine_7d'] = aki48
    alternate_manifest = manifest.merge(derive_last_preoperative_creatinine(manifest, labs), on='case_id', validate='one_to_one')
    alternate_manifest['preop_cr'] = alternate_manifest.last_preop_cr_90d
    alternate = build_creatinine_aki_outcomes(alternate_manifest, labs)
    for name, value in (('frozen_aki', outcomes), ('frozen_aki48', window48), ('frozen_aki_lab90', alternate)):
        value.to_parquet(PRIVATE / f'outcomes_{name}.parquet', index=False)
    json_write(QA / 'outcomes_ascertainment_verification.json', {'status': 'PASS', 'laboratory_sha256': sha256(labs_path), 'primary_outcomes_exactly_reproduced': True, 'alternative_baseline_derived_once_then_frozen': True, 'outcome_definition_unchanged': True, 'clinical_signoff': False})
    return (outcomes, window48, alternate_manifest, alternate)

def select_aki_frame(spec, frames):
    label = spec['analysis']
    threshold = float(spec['map_threshold_mm_hg'])
    frame = frames[threshold].copy()
    if label == 'AKI within 48 h':
        frame = frames['48h'].copy()
    elif label == 'Latest laboratory baseline within 90 d':
        frame = frames['lab90'].copy()
    elif label == 'One case per subject':
        frame = frame.sort_values('case_id').drop_duplicates('subjectid', keep='first')
    if 'recorded ASA sensitivity' in label:
        frame['asa_high'] = frame.asa_high_recorded
    if spec['population'] == 'cases with reference AUC >0':
        frame = frame[frame.true_auc.gt(0)]
    if label == 'Pre-display AUC fraction':
        frame = frame[frame.predisplay10.notna()]
    elif label == 'At least two postoperative creatinine values':
        frame = frame[frame.postop_cr_count_7d.ge(2)]
    elif label == 'Exclude baseline creatinine >=4 mg/dL':
        frame = frame[frame.baseline_creatinine_mg_dl.lt(4.0)]
    return frame

def policy_features(policy, panel, frozen_frequency, frozen_episodes, workers):
    if policy == 'original':
        return (frozen_frequency, frozen_episodes, 'verified_frozen_original')
    frequency_path = PRIVATE / f'frequency_decomp_{policy}.parquet'
    own_frequency = PRIVATE / f'outcomes_{policy}_frequency.parquet'
    own_episodes = PRIVATE / f'outcomes_{policy}_episodes.parquet'
    need_frequency = not frequency_path.exists()
    frequency, episodes = replay_panel(panel, workers, need_frequency=need_frequency)
    origin = 'independent_core_replay' if need_frequency else 'shared_frequency_independent_episodes'
    if not need_frequency:
        frequency = pd.read_parquet(frequency_path)
    frequency.to_parquet(own_frequency, index=False)
    episodes.to_parquet(own_episodes, index=False)
    frequency = normalize_frequency(frequency)
    keys = ['case_id', 'threshold', 'interval_min']
    if frequency.duplicated(keys).any() or set(frequency.case_id) != set(panel.case_id):
        raise ValueError('Policy frequency does not match the included panel cohort')
    if set(frequency.threshold) != {55.0, 60.0, 65.0}:
        raise ValueError('All three frozen MAP thresholds are required')
    expected_hours = panel.groupby('case_id').size() / 360
    np.testing.assert_allclose(frequency.anesthesia_hours, frequency.case_id.map(expected_hours), atol=1e-12)
    return (frequency, episodes, origin)

def run_models(policy, frames, script):
    results = {}
    for name in ('aki_sequential', 'aki_incremental', 'aki_asa'):
        frozen_specs = pd.read_csv(TABLES / MODEL_FILES[name])
        rows = []
        for spec in frozen_specs.to_dict('records'):
            frame = select_aki_frame(spec, frames)
            estimate = fit_modified_poisson(frame, formula=spec['formula'], term=spec['term'], cluster_column='subjectid')
            spec.update(estimate)
            spec['positive_adverse_association_supported'] = estimate['ci_low'] > 1.0
            spec['observed_event_risk'] = estimate['events'] / estimate['n_cases']
            spec['policy'] = policy
            spec['clinical_adjudication_status'] = 'pending_not_clinically_adjudicated'
            rows.append(spec)
        results[name] = pd.DataFrame(rows)
    primary = frames[65.0].copy()
    icu = build_prolonged_icu_outcome(primary, threshold_days=2)
    primary['icu_ge2d'] = pd.to_numeric(icu.prolonged_icu_ge2d, errors='coerce')
    gate = summarize_icu_gate(primary, threshold_days=2, min_completeness=0.95, min_events=100)
    results['icu_sequential'] = fit_sequential_icu_models(primary)
    sensitivity = primary.copy()
    sensitivity['asa_high'] = sensitivity.asa_high_recorded
    results['icu_asa'] = fit_sequential_icu_models(sensitivity).iloc[1:].copy()
    results['icu_asa']['analysis'] += ' (recorded ASA sensitivity)'
    results['icu_asa']['analysis_role'] = 'ASA coding sensitivity; ' + results['icu_asa'].analysis_role
    for name in ('icu_sequential', 'icu_asa'):
        result = results[name]
        result['policy'] = policy
        result['clinical_adjudication_status'] = 'pending_not_clinically_adjudicated'
        result['observed_event_risk'] = result.events / result.n_cases
    gate_status = gate.loc[gate.criterion.eq('overall_gate'), 'status'].iloc[0]
    results['icu_sequential']['gate0_status'] = gate_status
    results['icu_sequential']['reporting_decision'] = 'ELIGIBLE_FOR_SUPPLEMENTARY_EXPLORATORY_REPORTING' if gate_status == 'PASS_WITH_ENDPOINT_LIMITATION' and results['icu_sequential'].model_stability_status.eq('PASS').all() else 'NOT_ELIGIBLE_FOR_MANUSCRIPT_REPORTING'
    for name, result in results.items():
        result.to_csv(AGG / f'outcomes_{policy}_{name}.csv', index=False)
    gate.to_csv(AGG / f'outcomes_{policy}_icu_gate.csv', index=False)
    summarize_hidden_presence(primary).to_csv(AGG / f'outcomes_{policy}_aki_presence.csv', index=False)
    script._missingness_comparison(primary).to_csv(AGG / f'outcomes_{policy}_aki_missingness.csv', index=False)
    primary.to_parquet(PRIVATE / f'outcomes_{policy}_model_frame.parquet', index=False)
    return (results, primary)

def verify_r(policy, primary, results):
    rscript = shutil.which('Rscript')
    if not rscript:
        return {'status': 'UNAVAILABLE', 'reason': 'Rscript not installed'}
    export = primary.copy()
    for outcome in ('aki', 'icu_ge2d'):
        export[outcome] = export[outcome].map({True: 'True', False: 'False'})
    frame_path = PRIVATE / f'outcomes_{policy}_r_frame.csv'
    export.to_csv(frame_path, index=False)
    aki_tables = PRIVATE / f'outcomes_{policy}_r_aki_expected.csv'
    pd.concat([results['aki_sequential'], results['aki_incremental']]).to_csv(aki_tables, index=False)
    alias = export.copy()
    alias['icu_ge2d'] = alias['aki']
    alias_path = PRIVATE / f'outcomes_{policy}_r_aki_alias_frame.csv'
    alias.to_csv(alias_path, index=False)
    tasks = [('aki', '34_verify_aki_primary_model.R', frame_path, aki_tables), ('aki_sequence', '40_verify_icu_resource_use_models_v74.R', alias_path, AGG / f'outcomes_{policy}_aki_sequential.csv'), ('icu', '40_verify_icu_resource_use_models_v74.R', frame_path, AGG / f'outcomes_{policy}_icu_sequential.csv')]
    audit = []
    for name, script, input_path, expected_path in tasks:
        output_path = QA / f'outcomes_{policy}_R_{name}.txt'
        run = subprocess.run([rscript, '--vanilla', str(SOURCE / 'scripts' / script), str(input_path), str(expected_path), str(output_path)], capture_output=True, text=True, timeout=180)
        (QA / f'outcomes_{policy}_R_{name}_run.log').write_text(run.stdout + run.stderr)
        audit.append({'verification': name, 'returncode': run.returncode, 'passed': run.returncode == 0, 'output': str(output_path)})
    return {'status': 'PASS' if all((row['passed'] for row in audit)) else 'FAIL', 'unique_models_checked': 7, 'new_QC_specific_verification': True, 'legacy_proof_not_reused': True, 'runs': audit}

def publication_outputs(models, denominators):
    report = {'schema_version': '1.0', 'primary_policy': 'original', 'clinical_adjudication_status': 'pending_not_clinically_adjudicated', 'qc_role': 'prespecified_technical_sensitivity_only', 'variance_estimator': 'subject-cluster robust sandwich', 'definitions': {'aki': 'creatinine-only KDIGO through 7 days or discharge; frozen VitalDB preop_cr baseline', 'icu': 'postoperative ICU stay >=2 days; planned versus unplanned admission unavailable', 'hidden_burden_scale': 'per 10 mm Hg*min per original anaesthesia-hour', 'hdr_scale': 'per 10 percentage points among reference AUC >0 cases', 'ci_low_ci_high': '95% confidence interval limits on the relative-risk scale', 'events': 'observed outcome events among the actual complete cases used by that model', 'outcome_risk': 'observed events divided by evaluable cases, not adjusted or causal risk', 'n_subjects': 'unique subjects defining sandwich variance clusters', 'startup10': 'whole first finite reference +600-second window removed; not artifact correction'}, 'policies': {}}
    wide_rows = {}
    for denom in denominators:
        policy = denom['policy']
        entry = {'cohort_n_cases': int(denom['n_cases']), 'cohort_n_subjects': int(denom['n_subjects']), 'outcomes': {endpoint: {'evaluable_n': int(denom[f'{endpoint}_evaluable']), 'events': int(denom[f'{endpoint}_events']), 'risk': float(denom[f'{endpoint}_risk'])} for endpoint in ('aki', 'icu')}, 'r_verification': denom['r_verification'], 'models': {}}
        selected = models[models.policy.eq(policy)]
        for row in selected.to_dict('records'):
            family = row['model_family']
            if family in ('aki_sequential', 'icu_sequential'):
                endpoint = family.split('_')[0]
                model_key = f"{endpoint}_{row['adjustment_sequence']}"
            elif family == 'aki_incremental' and row['analysis'] == 'HDR65 incremental model':
                endpoint, model_key = ('aki', 'aki_hdr_incremental')
            else:
                continue
            entry['models'][model_key] = {'analysis': row['analysis'], **{c: int(row[c]) for c in ('n_cases', 'n_subjects', 'events')}, **{c: float(row[c]) for c in ('relative_risk', 'ci_low', 'ci_high', 'p_value')}}
            wide = wide_rows.setdefault(model_key, {'endpoint': endpoint.upper(), 'model': row['analysis']})
            wide[policy] = f"{row['relative_risk']:.3f} ({row['ci_low']:.3f} to {row['ci_high']:.3f}); N={int(row['n_cases'])}; events={int(row['events'])}"
        report['policies'][policy] = entry
    wide = pd.DataFrame(wide_rows.values())
    return (report, wide)

def verify_existing_artifacts(policies):
    model_paths = [AGG / 'outcomes_all_models.csv']
    for policy in POLICIES:
        model_paths.extend((AGG / f'outcomes_{policy}_{name}.csv' for name in MODEL_FILES))
        model_paths.append(PRIVATE / f'outcomes_{policy}_model_frame.parquet')
    hashes = {str(path.relative_to(OUT)): sha256(path) for path in model_paths}
    statuses = {}
    for policy in policies:
        primary = pd.read_parquet(PRIVATE / f'outcomes_{policy}_model_frame.parquet')
        results = {name: pd.read_csv(AGG / f'outcomes_{policy}_{name}.csv') for name in MODEL_FILES}
        verification = verify_r(policy, primary, results)
        json_write(QA / f'outcomes_{policy}_R_verification.json', verification)
        statuses[policy] = verification['status']
        print(f"Existing-frame independent R verification: {policy} {verification['status']}", flush=True)
    original = pd.read_parquet(DATA / 'aki_invisibility_analysis_case_v7.parquet')
    audits = {}
    for policy in POLICIES:
        primary = pd.read_parquet(PRIVATE / f'outcomes_{policy}_model_frame.parquet')
        audits[policy] = check_duration_and_reference(primary, original)
    json_write(QA / 'outcomes_duration_reference_audit.json', audits)
    denominator_path = AGG / 'outcomes_denominators.csv'
    denominators = pd.read_csv(denominator_path)
    for policy, status in statuses.items():
        denominators.loc[denominators.policy.eq(policy), 'r_verification'] = status
    denominators.to_csv(denominator_path, index=False)
    models = pd.read_csv(AGG / 'outcomes_all_models.csv')
    publication, wide = publication_outputs(models, denominators.to_dict('records'))
    publication['duration_and_reference_checks'] = audits
    json_write(AGG / 'outcomes_summary.json', publication)
    wide.to_csv(AGG / 'outcomes_publication_comparison_wide.csv', index=False)
    comparison_path = AGG / 'outcomes_comparison.json'
    comparison = json.loads(comparison_path.read_text())
    comparison['denominators'] = denominators.to_dict('records')
    comparison['duration_and_reference_checks'] = audits
    json_write(comparison_path, comparison)
    unchanged = all((sha256(OUT / path) == digest for path, digest in hashes.items()))
    json_write(QA / 'outcomes_existing_frame_R_audit.json', {'policies_reverified': policies, 'status': statuses, 'python_models_rerun': False, 'model_frames_and_estimates_unchanged': unchanged, 'model_file_hashes': hashes})
    if not unchanged or any((status != 'PASS' for status in statuses.values())):
        raise AssertionError('Existing-frame R verification failed or model inputs changed')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-r', action='store_true')
    parser.add_argument('--verify-existing-r', nargs='+', choices=POLICIES)
    args = parser.parse_args()
    if args.verify_existing_r:
        verify_existing_artifacts(args.verify_existing_r)
        return
    if args.policies[0] != 'original':
        raise ValueError('Original baseline must be verified before QC models')
    for directory in (PRIVATE, AGG, QA):
        directory.mkdir(exist_ok=True, parents=True)
    script = load_aki_script()
    source_paths = [DATA / f for f in ('artmap_10s.parquet', 'vitaldb_manifest.parquet', 'aki_creatinine_case_v7.parquet', 'aki_invisibility_analysis_case_v7.parquet', 'frequency_decomp_case_mean_v7.parquet', 'episode_observability_case_5min.parquet')]
    source_paths += [TABLES / name for name in MODEL_FILES.values()]
    source_paths += [SOURCE / 'scripts' / name for name in ('33_run_exploratory_aki_analysis_v7.py', '39_run_exploratory_icu_resource_use_v74.py', '34_verify_aki_primary_model.R', '40_verify_icu_resource_use_models_v74.R')]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    original = pd.read_parquet(DATA / 'artmap_10s.parquet')
    manifest = pd.read_parquet(DATA / 'vitaldb_manifest.parquet')
    manifest = manifest[manifest.case_id.isin(original.case_id.unique())].copy()
    frozen_frame = pd.read_parquet(DATA / 'aki_invisibility_analysis_case_v7.parquet')
    outcomes, window48, alternate_manifest, alternate = prepare_frozen_outcomes(manifest, script)
    frozen_frequency, frozen_episodes = verify_baseline_features(original, args.workers)
    model_sets, summary, panel_audits, comparisons = ([], [], [], [])
    for policy in args.policies:
        print(f'Starting outcome models: {policy}', flush=True)
        panel_path = PRIVATE / f'panel_{policy}.parquet'
        panel = pd.read_parquet(panel_path)
        panel_audit = validate_panel(panel, original, policy)
        cohort = pd.read_csv(PRIVATE / f'cohort_{policy}.csv')
        included = cohort.included.astype(str).str.lower().isin(['true', '1'])
        if set(cohort.loc[included, 'case_id']) != set(panel.case_id):
            raise AssertionError('Cohort membership and panel cases disagree')
        ids = set(panel.case_id)
        cohort_manifest = manifest[manifest.case_id.isin(ids)].copy()
        frequency, episodes, origin = policy_features(policy, panel, frozen_frequency, frozen_episodes, args.workers)
        frames = {}
        for threshold in (65.0, 60.0, 55.0):
            features = aggregate_invisibility_features(frequency, episodes, threshold=threshold)
            features = features[features.case_id.isin(ids)]
            frames[threshold] = script._analysis_frame(cohort_manifest, outcomes, features)
            assert_frozen_columns(frames[threshold], frozen_frame)
            if threshold == 65.0:
                frames['48h'] = script._analysis_frame(cohort_manifest, window48, features)
                frames['lab90'] = script._analysis_frame(alternate_manifest[alternate_manifest.case_id.isin(ids)], alternate, features)
            frames[threshold].to_parquet(PRIVATE / f'outcomes_{policy}_features_{int(threshold)}.parquet', index=False)
        results, primary = run_models(policy, frames, script)
        if policy == 'original':
            for name, result in results.items():
                comparisons.append(compare_tables(result, pd.read_csv(TABLES / MODEL_FILES[name]), name))
            checked = pd.concat(comparisons, ignore_index=True)
            checked.to_csv(QA / 'outcomes_baseline_model_verification.csv', index=False)
            if not checked.passed.all():
                raise AssertionError('Original model results do not match frozen tables')
            verify_publication_baseline(results)
        r_result = verify_r(policy, primary, results) if args.verify_r and policy in ('original', 'qc60') else {'status': 'NOT_RUN'}
        json_write(QA / f'outcomes_{policy}_R_verification.json', r_result)
        panel_audit.update({'panel_sha256': sha256(panel_path), 'features_origin': origin, 'n_subjects': int(primary.subjectid.nunique()), 'frozen_outcomes_and_covariates': True})
        panel_audits.append(panel_audit)
        summary.append({'policy': policy, 'n_cases': len(primary), 'n_subjects': int(primary.subjectid.nunique()), 'aki_evaluable': int(primary.aki.notna().sum()), 'aki_events': int(primary.aki.sum()), 'aki_risk': float(primary.aki.mean()), 'icu_evaluable': int(primary.icu_ge2d.notna().sum()), 'icu_events': int(primary.icu_ge2d.sum()), 'icu_risk': float(primary.icu_ge2d.mean()), 'r_verification': r_result['status']})
        for name, value in results.items():
            model_sets.append(value.assign(model_family=name))
        print(json.dumps(summary[-1]), flush=True)
    all_models = pd.concat(model_sets, ignore_index=True)
    all_models.to_csv(AGG / 'outcomes_all_models.csv', index=False)
    pd.DataFrame(summary).to_csv(AGG / 'outcomes_denominators.csv', index=False)
    publication, wide = publication_outputs(all_models, summary)
    json_write(AGG / 'outcomes_summary.json', publication)
    wide.to_csv(AGG / 'outcomes_publication_comparison_wide.csv', index=False)
    pd.DataFrame(panel_audits).to_csv(QA / 'outcomes_panel_audit.csv', index=False)
    if any((sha256(path) != digest for path, digest in source_hashes.items())):
        raise AssertionError('Frozen input changed during the run')
    json_write(QA / 'outcomes_provenance.json', {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'statsmodels': statsmodels.__version__, 'source_hashes': source_hashes, 'frozen_sources_unchanged': True, 'clinical_signoff': False, 'model_selection_unchanged': True})
    selected = all_models[all_models.model_family.isin(['aki_sequential', 'icu_sequential'])]
    json_write(AGG / 'outcomes_comparison.json', {'status': 'COMPLETED_TECHNICAL_QC_SENSITIVITY', 'primary_policy': 'original', 'clinical_signoff': False, 'baseline_model_verification': 'PASS', 'variance': 'subject-cluster robust sandwich', 'interpretation': 'Exploratory noncausal associations; technical candidates, not clinical adjudication.', 'denominators': summary, 'main_models': selected[['policy', 'model_family', 'analysis', 'n_cases', 'n_subjects', 'events', 'relative_risk', 'ci_low', 'ci_high', 'p_value']].to_dict('records')})
    print(f'Completed {len(all_models)} models across {len(args.policies)} policies', flush=True)
if __name__ == '__main__':
    main()
