# Temporal observability of intraoperative hypotension

This repository is the reproducibility package for a retrospective public-database study of how an emulated intermittent blood pressure display represents hypotension over time.

The release contains analysis code, a configuration template, tests, figure source data, Python/Matplotlib-generated figures, a final workbook, reproducibility manifests, and non-identifiable aggregate outputs. It does not contain submission documents.

## Scientific scope

The primary analysis uses a fixed 10-s arterial mean arterial pressure (MAP) reference trajectory and emulates intermittent displays at prespecified sampling intervals. At MAP below 65 mm Hg with a 5-min display interval, 9,187 reference episodes lasting at least 60 s were identified among 2,435 selected VitalDB cases. Complete miss occurred in 34.2% of episodes; the first displayed low value arrived with at least 1 min of reference hypotension remaining in 55.0% and at least 2 min remaining in 35.7%. The pre-display portion represented 34.9% of reference episode area under the threshold.

The supplementary cuff analysis reconstructs non-invasive blood pressure (NIBP) display events. Each event is paired to the median of valid arterial MAP samples from 30 s before through 30 s after the recorded event time.

Exploratory postoperative analyses evaluate creatinine-defined acute kidney injury (AKI) and ICU length of stay of at least 2 days. Hidden hypotension burden was associated with AKI after clinical adjustment, but the association was attenuated after nonlinear adjustment for total reference hypotension burden. The ICU analysis provides exploratory resource-use context. These observational associations do not estimate a causal effect of monitoring frequency, treatment, or organ protection.

## Data boundary

Raw VitalDB, MoVeR, and INSPIRE datasets are not redistributed. The current analytic results use VitalDB. MoVeR did not pass the prespecified direct-waveform availability gate, and no direct waveform estimate is reported from that source. INSPIRE is retained only as resource and transportability context; it is not used to estimate the temporal burden metrics.

Case-level trajectories, reconstructed display events, paired measurements, creatinine records, analysis model frames, and all other patient-level intermediates are excluded. Source datasets must be obtained independently under their applicable terms of use.

## Repository contents

- `src/ioh/`: reusable analysis modules.
- `scripts/`: pipeline entry points and independent R checks.
- `tests/`: unit and reproducibility tests.
- `config/reproduction.example.yaml`: data-path and analysis-setting template.
- `outputs/tables/`: non-identifiable aggregate CSV outputs.
- `manuscript_inputs/figure_source_files/`: source data and legends for all figures.
- `outputs/figures/`: programmatically generated PDF and PNG figures.
- `outputs/final_workbook.xlsx`: README plus all aggregate tables in separate worksheets.
- `outputs/manifests/`: provenance and frozen file-level SHA-256 inventories.

## Reproduction outline

Use Python 3.12 and install the recorded dependencies. Keep the completed private configuration outside version control.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

After obtaining VitalDB under its own terms, create a private configuration from `config/reproduction.example.yaml`, set the local source location, and run:

```bash
python scripts/01_build_vitaldb_manifest.py --config /path/to/reproduction.yaml
python scripts/02_preprocess_vitaldb_artmap.py --config /path/to/reproduction.yaml
python scripts/03_emulate_frequency_decomposition.py --config /path/to/reproduction.yaml
python scripts/27_execute_anesthesiology_revision_v7.py --config /path/to/reproduction.yaml
python scripts/28_make_anesthesiology_tables_figures_v7.py --config /path/to/reproduction.yaml
python scripts/33_run_exploratory_aki_analysis_v7.py --config /path/to/reproduction.yaml
```

The ICU resource-use script consumes a private case-level model frame generated during the preceding analysis and does not operate on the aggregate release files.

## Frozen release

- Tag: `v1.2.0-submission`
- Aggregate workbook: `outputs/final_workbook.xlsx`
- Aggregate workbook SHA-256: `e4f8c2f5e3aa2af1a320b87d2ff552c3f9d60e26c33778eb73bb61e53e693d60`
- Outputs manifest: `outputs/manifests/outputs_manifest.json`
- Outputs manifest SHA-256: `fcea2d43fef6e0b38043fcd8b30c32d03ce4772e7825388903c659e850e0badd`
- Stable Zenodo concept DOI: [10.5281/zenodo.21253616](https://doi.org/10.5281/zenodo.21253616)

The exact repository inventory is recorded in `outputs/manifests/release_manifest.json` and `checksums.sha256`.

## License and citation

Code is released under the MIT License. Dataset access remains governed by the source providers. Cite the archived release using `CITATION.cff` and the version-specific DOI displayed by Zenodo.
