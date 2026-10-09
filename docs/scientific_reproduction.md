# Reproduction and Provenance

## Status and boundaries

This document describes the inherited scientific workflow packaged in release v1.5.0. Independent physician clinical adjudication was not performed.
Original v1.4.0 results remain primary, with QC additions reported as sensitivity
analyses. No one-command raw reproduction has been validated for this archive.
The source inputs below are intentionally not redistributed.

## Required local source tree

IOH_PROJECT_ROOT points to the user's private source project, containing:

1. analysis_v10_subject_phase_release/outputs/intermediate:
   vitaldb_manifest.parquet, artmap_10s.parquet,
   frequency_decomp_case_offset.parquet, frequency_decomp_case_mean_v7.parquet,
   episode_observability_case_5min.parquet, aki_creatinine_case_v7.parquet,
   aki_invisibility_analysis_case_v7.parquet, and other dependencies named in the
   frozen baseline scripts. The original definitions and covariates are retained.
2. analysis_v10_subject_phase_release/src and outputs/tables, including the
   sequential/incremental AKI and ICU model specification/results CSV files and
   aki_labs_source_audit.csv used for baseline reconciliation.
3. EJA_投稿文件包_20261006_最终文字修订/03_可复现材料: the frozen public package, not the candidate itself.
4. 治疗相关事件补充修订_20261004/code/run_analysis.py and treatment_core.py,
   data_restricted/raw_record_changes.csv and stable_record_changes.csv, the
   seven frozen event CSVs, recovery_events.csv, and aggregate comparison files.
   The QC treatment rerun reused 834 raw / 776 stable frozen pump transitions;
   it did not reread all raw pump tracks.
5. 启动段质量审计_20261007/private_case_audit:
   full_waveform_10s_metrics.csv, clinical_review_list_87_cases.csv and
   startup_numeric_case_audit.csv. These derive from the screening scripts.
6. Official VitalDB labs.csv supplied through IOH_VITALDB_LABS, independently
   obtained under applicable source-data terms. No laboratory rows are archived.

IOH_QC_OUTPUT_ROOT points to the private QC workspace containing the fixed
方案.md, data_restricted/panel_POLICY.parquet and cohort_POLICY.csv. The events
stage creates frequency_decomp_POLICY.parquet before the outcomes stage.
Policies are original, qc60, qc_all, startup10.

For screening reconstruction, IOH_STARTUP_AUDIT_ROOT selects a private audit
output directory. IOH_VITALDB_RAW_ROOT contains trks.csv;
IOH_VITALDB_TRACK_DIRS lists local track directories separated by os.pathsep.
The screening order is run_numeric_audit.py, run_waveform_audit.py,
scan_full_waveforms.py, prepare_review_list.py, then optional
run_startup_sensitivity.py. Generated case review material must remain private.
These reconstructed plots are not clinician adjudication or AI review evidence.
The review_images.py copy contains no embedded per-case observations. It requires
IOH_IMAGE_REVIEW_INPUT_JSON, a separately retained private JSON payload of the
previously viewed observations. It only packages those observations; it does
not perform image interpretation, reproduce the review, or certify clinicians.

## Current QC sequence

The archived copies accept local environment configuration before data imports:

    python extensions/qc_startup/build_qc_panels.py
    python extensions/qc_startup/rerun_events.py
    python extensions/qc_startup/rerun_treatment.py
    python extensions/qc_startup/rerun_outcomes.py
    python extensions/qc_startup/build_qc_publication_tables.py

Inspect each runner's help and input checks first. Do not run against incomplete
or unverified private inputs. The outcomes runner also invokes the frozen
scripts/34_verify_aki_primary_model.R and
scripts/40_verify_icu_resource_use_models_v74.R for policy-specific checks.
Public tables are derived outputs, not replacements for case-level inputs.

The workbook builder is a separate Node application, not run by archive staging. Its
public export consumes qc_publication_tables.json and baseline aggregate/table
sources. Its private clinical-review export may additionally require review
CSV files. Set CODEX_NODE_MODULES to a local installation containing
@oai/artifact-tool, jszip, xml-js and sharp. Access to that artifact runtime is
not guaranteed by this source archive. No workbook re-render is performed by
build_archive.py.

## Dependencies

Baseline Python versions are recorded in requirements.txt. QC uses Python 3.12,
NumPy, pandas, pyarrow, SciPy, statsmodels, scikit-learn, matplotlib, PyYAML and
pytest. R verification needs Rscript and the packages imported by the included
R scripts (including sandwich/lmtest as applicable). Optional private screening
document generation is outside this public package. IOH_QC_DEPENDENCIES may
point to a locally installed Python dependency directory; no runtime is bundled.
The aggregate Supplementary Figure S2 builder and archive PDF inspection also
use Pillow, pypdf and reportlab; see requirements-qc.txt. Its fixed aggregate
source CSV and caption accompany the PNG, SVG and vector PDF. Visual QA and
private-review proof images are not included.

## Code actually staged for this extension

- extensions/qc_startup/build_archive.py
- extensions/qc_startup/build_qc_panels.py
- extensions/qc_startup/build_qc_publication_tables.py
- extensions/qc_startup/build_qc_workbooks.mjs
- extensions/qc_startup/make_qc_figure.py
- extensions/qc_startup/rerun_events.py
- extensions/qc_startup/rerun_outcomes.py
- extensions/qc_startup/rerun_treatment.py
- extensions/qc_startup/review_images.py
- extensions/qc_startup/test_build_archive.py
- extensions/qc_startup/test_qc_events.py
- extensions/qc_startup/test_qc_outcomes.py
- extensions/qc_startup/test_qc_panels.py
- extensions/qc_startup/test_qc_treatment.py
- extensions/startup_screening/audit_core.py
- extensions/startup_screening/run_numeric_audit.py
- extensions/startup_screening/run_waveform_audit.py
- extensions/startup_screening/scan_full_waveforms.py
- extensions/startup_screening/prepare_review_list.py
- extensions/startup_screening/run_startup_sensitivity.py
- extensions/startup_screening/test_audit_core.py

The current baseline src/scripts/extensions/tests are copied from the supplied
v1.4.0 public package. New QC and screening copies receive path-only portability
adaptations; the private source files are unchanged. The release inventories
record relative paths and hashes; private staging receipts are not redistributed.
Syntax validation does not establish end-to-end reproduction on a new machine.

## Verification and freeze

    python code/run_synthetic_tests.py
    python code/verify_release.py

The synthetic test runner uses an isolated source layout pointing only to the
archived code and aggregate artifacts. It does not read private study inputs.
The second command verifies the frozen release inventories and file hashes.
Frozen baseline-only logs must not be read as evidence for all QC policies.
The selected new QA verdicts retain the actual policy-specific status.

The current public freeze contains outputs_manifest.json, release_manifest.json
and checksums.sha256 with noncircular inventories. Inherited assembly scripts are
retained for source-workflow reconstruction, not as a one-command re-freeze of
this published snapshot. Changes to frozen files invalidate the inventories.
