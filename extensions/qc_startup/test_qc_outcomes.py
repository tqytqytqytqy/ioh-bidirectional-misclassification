"""Synthetic checks for the outcomes-only QC adapter; no patient rows printed."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ADAPTER = Path(__file__).with_name('rerun_outcomes.py')

def adapter():
    assert ADAPTER.is_file(), 'Outcomes QC adapter has not been implemented'
    spec = importlib.util.spec_from_file_location('qc_outcomes', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class OutcomeAdapterTests(unittest.TestCase):

    def test_reference_adjustment_uses_new_burden_and_original_duration(self):
        mod = adapter()
        original = pd.DataFrame({'case_id': [1], 'anesthesia_hours': [2.0], 'duration_hr': [2.0], 'true_auc': [40.0]})
        revised = pd.DataFrame({'case_id': [1], 'anesthesia_hours': [2.0], 'duration_hr': [2.0], 'true_auc': [20.0], 'hidden_auc': [4.0], 'true_twa': [10.0], 'log_true_twa': [np.log1p(10.0)], 'hidden_twa10': [0.2]})
        audit = mod.check_duration_and_reference(revised, original)
        self.assertEqual(audit['changed_reference_burden_cases'], 1)
        self.assertTrue(audit['original_duration_denominators_preserved'])
        revised['log_true_twa'] = np.log1p(20.0)
        with self.assertRaises(AssertionError):
            mod.check_duration_and_reference(revised, original)

    def test_latest_formatted_publication_values_are_checked(self):
        mod = adapter()
        estimate = {'n_cases': 2240, 'events': 104, 'relative_risk': 1.097284, 'ci_low': 0.997612, 'ci_high': 1.206914, 'p_value': 0.051}
        display = {'Cases': '2,240', 'AKI': 104, 'RR (95% CI)': '1.10 (1.00 to 1.21)', 'P value': '0.051'}
        self.assertTrue(mod.publication_model_matches(estimate, display))
        display['AKI'] = 106
        self.assertFalse(mod.publication_model_matches(estimate, display))

    def test_environment_output_root_is_honored_before_paths_are_built(self):
        original = adapter()
        replacement = original.OUT / 'qa/outcomes_environment_test'
        with patch.dict(os.environ, {'IOH_PROJECT_ROOT': str(original.ROOT), 'IOH_QC_OUTPUT_ROOT': str(replacement)}):
            changed = adapter()
        self.assertEqual(changed.OUT, replacement)
        self.assertEqual(changed.PRIVATE, replacement / 'data_restricted')
        self.assertNotIn('/' + 'Users/', ADAPTER.read_text())

    def test_publication_schema_retains_numeric_ci_and_denominators(self):
        mod = adapter()
        models = pd.DataFrame([{'policy': 'original', 'model_family': 'aki_sequential', 'analysis': 'Absolute hidden burden, clinical adjusted', 'adjustment_sequence': 'clinical', 'n_cases': 100, 'n_subjects': 97, 'events': 8, 'relative_risk': 1.2, 'ci_low': 0.9, 'ci_high': 1.6, 'p_value': 0.2}])
        summary = [{'policy': 'original', 'n_cases': 110, 'n_subjects': 107, 'aki_evaluable': 105, 'aki_events': 9, 'aki_risk': 9 / 105, 'icu_evaluable': 110, 'icu_events': 11, 'icu_risk': 0.1, 'r_verification': 'PASS'}]
        report, wide = mod.publication_outputs(models, summary)
        model = report['policies']['original']['models']['aki_clinical']
        self.assertEqual(model['n_cases'], 100)
        self.assertEqual(model['events'], 8)
        self.assertEqual(model['ci_low'], 0.9)
        self.assertIn('100', wide.original.iloc[0])

    def test_source_frame_helpers_load_without_pipeline_dependencies(self):
        mod = adapter()
        source = mod.load_aki_script()
        self.assertTrue(callable(source._analysis_frame))
        self.assertTrue(callable(source._missingness_comparison))

    def test_frequency_offsets_are_averaged_before_modeling(self):
        mod = adapter()
        data = pd.DataFrame({'case_id': [1] * 30, 'threshold': [65.0] * 30, 'interval_min': [5.0] * 30, 'offset_sec': np.arange(30) * 10.0, 'true_auc': [10.0] * 30, 'hidden_auc': np.arange(30), 'concordant_auc': [5.0] * 30})
        normalized = mod.normalize_frequency(data)
        self.assertEqual(len(normalized), 1)
        self.assertAlmostEqual(normalized.hidden_auc.iloc[0], 14.5)
        with self.assertRaises(ValueError):
            mod.normalize_frequency(data.iloc[:29])

    def test_mask_preserves_original_grid_and_finite_values(self):
        mod = adapter()
        original = pd.DataFrame({'case_id': [1] * 8 + [2] * 8, 'time_sec': list(np.arange(8) * 10) * 2, 'art_map': [50.0] * 16})
        changed = original[original.case_id.eq(1)].copy()
        changed.loc[0, 'art_map'] = np.nan
        audit = mod.validate_panel(changed, original, 'qc60')
        self.assertEqual(audit['excluded_cases'], 1)
        self.assertEqual(audit['newly_masked_bins'], 1)
        with self.assertRaises(ValueError):
            mod.validate_panel(changed.iloc[1:], original, 'qc60')
        changed.loc[2, 'art_map'] = 60.0
        with self.assertRaises(ValueError):
            mod.validate_panel(changed, original, 'qc60')

    def test_startup_sensitivity_cannot_exclude_cases(self):
        mod = adapter()
        panel = pd.DataFrame({'case_id': [1, 2], 'time_sec': [0, 0], 'art_map': [60.0, 60.0]})
        with self.assertRaises(ValueError):
            mod.validate_panel(panel.iloc[:1], panel, 'startup10')

    def test_feature_replay_preserves_duration_and_auc_closure(self):
        mod = adapter()
        time = np.arange(120) * 10.0
        reference = np.full(120, 70.0)
        reference[0:6] = np.nan
        reference[8:30] = 50.0
        frequency, episodes = mod.derive_case_features((7, time, reference, True, True))
        frequency = pd.DataFrame(frequency)
        self.assertEqual(len(frequency), 3)
        np.testing.assert_allclose(frequency.anesthesia_hours, 1 / 3)
        np.testing.assert_allclose(frequency.true_auc, frequency.hidden_auc + frequency.concordant_auc)
        row = frequency.loc[frequency.threshold.eq(65)].iloc[0]
        self.assertAlmostEqual(row.true_auc, 55.0)
        self.assertGreater(row.hidden_auc, 0)
        self.assertTrue(len(episodes) > 0)

    def test_model_comparison_detects_denominator_and_ci_drift(self):
        mod = adapter()
        original = pd.DataFrame({'analysis': ['a'], 'n_cases': [100], 'events': [8], 'relative_risk': [1.2], 'ci_low': [0.9], 'ci_high': [1.6], 'formula': ['aki ~ hidden_twa10']})
        self.assertTrue(mod.compare_tables(original, original, 'synthetic').passed.all())
        changed = original.copy()
        changed.loc[0, 'n_cases'] = 99
        changed.loc[0, 'ci_low'] = 0.8
        self.assertFalse(mod.compare_tables(changed, original, 'synthetic').passed.all())

    def test_one_case_selection_precedes_reference_restriction(self):
        mod = adapter()
        frame = pd.DataFrame({'case_id': [1, 2, 3], 'subjectid': [5, 5, 6], 'true_auc': [0, 2, 3]})
        row = {'analysis': 'One case per subject', 'map_threshold_mm_hg': 65.0, 'population': 'cases with reference AUC >0'}
        selected = mod.select_aki_frame(row, {65.0: frame})
        self.assertEqual(selected.case_id.tolist(), [3])

    def test_frozen_covariates_unchanged_when_exposures_change(self):
        mod = adapter()
        frozen = pd.DataFrame({'case_id': [1, 2], 'subjectid': [1, 2], 'aki': [0.0, 1.0], 'age10': [5.0, 6.0], 'hidden_auc': [1.0, 2.0], 'true_auc': [3.0, 4.0]})
        revised = frozen.copy()
        revised['hidden_auc'] = [0.0, 1.0]
        mod.assert_frozen_columns(revised, frozen)
        revised.loc[1, 'aki'] = 0
        with self.assertRaises(AssertionError):
            mod.assert_frozen_columns(revised, frozen)
if __name__ == '__main__':
    unittest.main(verbosity=2)
