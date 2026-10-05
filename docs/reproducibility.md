# Reproducibility instructions

Baseline analytical dependencies are pinned in `requirements.txt`; the original
environment is recorded in `environment.txt`. Current release verification is
documented in `outputs/qc/`. Baseline statistical code uses Python; independent
postoperative model checks use R. Added modules use Python, NumPy and pandas.

## Public integrity and synthetic tests

Run the commands in the repository README. These verify file integrity and
analytical edge cases using synthetic data. They do not constitute a new
source-data replication or clinical signal-quality adjudication.

The workbook has 40 sheets: four main tables, 29 supplementary tables, five
detailed aggregate sheets and two documentation/summary sheets. The
`Interval_Unrounded` sheet and corresponding CSV retain full-precision
interval/threshold estimates and marginal subject-bootstrap confidence limits.

## Private source-data workflow

Obtain VitalDB independently and keep all source and case-level output outside
the repository. Copy `config/reproduction.example.yaml` to a private location,
set its data root and output path, then execute the baseline cohort, waveform,
frequency-decomposition and postoperative scripts as documented in the
Makefile. Keep the baseline `src/` modules alongside the regenerated private
`outputs/` tree. The baseline figure/table builder reflects the earlier display
audit; current publication inputs are those in `outputs/aggregate/` and the
added modules below.

```sh
python scripts/01_build_vitaldb_manifest.py --config PRIVATE_CONFIG
python scripts/02_preprocess_vitaldb_artmap.py --config PRIVATE_CONFIG
python scripts/03_emulate_frequency_decomposition.py --config PRIVATE_CONFIG
python scripts/27_execute_anesthesiology_revision_v7.py --config PRIVATE_CONFIG
python scripts/33_run_exploratory_aki_analysis_v7.py --config PRIVATE_CONFIG
python extensions/events/run_extension.py --analysis PRIVATE_BASELINE --output PRIVATE_EVENTS
python extensions/events/simple_interval_summary.py --analysis PRIVATE_BASELINE --root PRIVATE_INTERVALS
python extensions/audit/scripts/run_audit.py --frozen PRIVATE_BASELINE --output PRIVATE_AUDIT --track-roots PRIVATE_TRACK_EXPORTS
python extensions/infusion/run_analysis.py --frozen PRIVATE_BASELINE --audit PRIVATE_AUDIT --track-roots PRIVATE_TRACK_EXPORTS --output PRIVATE_INFUSION
```

The `--analysis`/`--frozen` paths must point to the private baseline tree with
regenerated intermediates and `src/ioh/`. The track roots are independently
acquired numeric MAP and pump exports. Restricted output remains private.
Infusion tests cover timing support, record merging, missingness and recovery
censoring; the analysis protocol is in `extensions/infusion/protocol.md`.

## Publication artwork

Four main figures are editable PowerPoint vector objects with geometry derived
from aggregate outputs by JavaScript. The supplementary forest figure is
Python/Matplotlib output. No generative image model produced the scientific
plots. The optional JS exporters require the `@oai/artifact-tool` runtime;
`EJA_ARTIFACT_SKILL_DIR` and `EJA_PYTHON` identify its artwork finalisation tools.
This export environment is separate from the Python statistical calculations.
Portable figure source data and the final editable objects are included.

## Interpretation

Raw data and case-level records are excluded. New source acquisition can change
software or file availability; it must be reconciled against the frozen cohort,
definitions, denominators and aggregate results before claiming reproduction.
Clinical review of arterial startup signal quality remains outside the supplied
software checks. The archive describes an emulated display on fixed recorded
trajectories, not the counterfactual treatment course under cuff-only care.
