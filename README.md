# Temporal observability of intraoperative hypotension

Version **v1.4.0** contains the reproducibility materials for a retrospective
waveform study of intermittent blood pressure display. The package includes
baseline cohort and outcome code, supplementary event and infusion analyses,
configuration templates, synthetic tests, figure source data, four editable
vector figures, a 40-sheet workbook and non-identifiable aggregate outputs.

## Study scope

The primary cohort contains 2435 operations from 2380 subjects. A fixed arterial
mean arterial pressure (MAP) reference is evaluated on a 10 s grid, and periodic
displays are emulated across all compatible sampling phases. At MAP below
65 mmHg, 9187 reference episodes lasting at least 60 s occurred in 1899 cases.
With a 5 min display, the phase-averaged complete-miss proportion was 34.2%.
Any low display represented 65.8% of events, while a newly acquired low value
from the current episode represented 55.9%; 9.9% had inherited low display only.

The infusion module describes display information at recorded norepinephrine
or phenylephrine starts or increases. At 236 low-reference adjustment blocks
from 83 subjects, the emulated display was normal in 41.9% of 5 min phases,
compared with 15.3% at 1 min. These are phase-averaged event estimates, not
patient-level miss rates or observations of what the treating clinician saw.
After 184 confirmed reference recoveries, restricted low-display persistence
averaged 0.95 min within the specified observation window.

Cumulative exposure and event information describe different quantities.
Initial-display conventions materially affected net deficit area: -2.4% under
the primary convention, 52.0% with baseline initialisation and 14.1% after
excluding pre-display time. Corresponding episode-miss estimates were more
stable at 31.3 to 34.2%. The results should not be used to claim uniform AUC
equivalence across display conventions.

The corrected NIBP audit treats retained numerical output as records, not
authenticated independent cuff cycles. Legacy cuff-agreement and diagnostic
estimates have been removed from the current publication outputs. Pairing in
the record audit uses the arterial MAP median within a **-30 to +30 s window**
with at least 80% valid support. This is not device validation.

Postoperative AKI and ICU analyses remain exploratory. The total-burden-adjusted
AKI relative risk was 1.04 (95% CI, 0.81 to 1.34), which did not establish an
incremental association. ICU stay reflects resource use and lacks admission
intent. Fixed recorded trajectories do not estimate a causal monitoring effect,
treatment delay, drug effect or outcome benefit. Clinical adjudication of
arterial startup signal quality was not certified by the supplementary module.

## Data boundary

**Raw VitalDB, MoVeR and INSPIRE datasets are not redistributed.**
No case-level trajectories, subject identifiers, paired measurements,
laboratory records, model frames, pump tracks or private configurations are
included. Obtain source data independently under the providers' terms.
Current temporal analyses use VitalDB. MoVeR failed the direct-waveform gate;
no direct waveform claim is made. INSPIRE is resource and transportability
context only. Submission documents and private author-review materials are
not included in this repository.

## Files

- `src/ioh/`, `scripts/`, `tests/`: baseline analytical implementation and tests.
- `extensions/events/`: new/inherited classification, case/phase distributions,
  interval summaries and event-definition robustness.
- `extensions/audit/scripts/`: cuff-record and pump-track feasibility audits.
- `extensions/infusion/`: fixed supplementary protocol, runner and tests.
- `extensions/build_*.mjs`: aggregate-based workbook and artwork exporters.
- `outputs/tables/`: current four main tables and 29 supplementary tables as CSV.
- `outputs/aggregate/`: aggregate inputs and unrounded results for all main
  figure panels, interval summaries and infusion analyses.
- `outputs/figures/`: four editable vector PPTX figures and the supplementary
  Python/Matplotlib forest figure.
- `outputs/final_workbook.xlsx`: all 33 publication tables, five detailed
  aggregate sheets and two documentation/summary sheets.
- `outputs/manifests/`: output provenance and SHA-256 inventories.

## Verification and reproduction

Use the recorded analytical dependencies in `requirements.txt`. Synthetic
tests and integrity checks do not require clinical data:

```sh
PYTHONPATH=src python -m pytest -q
python -m unittest discover -s extensions/events -p 'test_*.py'
python -m unittest discover -s extensions/audit/scripts -p 'test_*.py'
python -m unittest discover -s extensions/infusion -p 'test_*.py'
shasum -a 256 -c checksums.sha256
python scripts/41_build_public_release_manifests.py verify
```

See [reproducibility instructions](docs/reproducibility.md) for the private
source-data workflow, and [analysis definitions](docs/analysis_conventions.md)
for denominators and timing. Releasing this version did not refit source-data
analyses. The source outputs were frozen before package assembly.

## Frozen release and citation

- Release: [v1.4.0](https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v1.4.0)
- Workbook: `outputs/final_workbook.xlsx`
- Workbook SHA-256: `900a621f7c9b9ce40d87d6aa03f68f8ab97a9254539f92be08310e309432c2e6`
- Outputs manifest: `outputs/manifests/outputs_manifest.json`
- Outputs manifest SHA-256: `4294bb7e5bb7223b78614cbcc2cc212cecbbed197efc00455f7bbf906e50a55b`
- Zenodo concept DOI: [10.5281/zenodo.21253616](https://doi.org/10.5281/zenodo.21253616)

Use the version DOI displayed by Zenodo for this exact v1.4.0 release.
`CITATION.cff` and `.zenodo.json` contain the current creator order. Code is
licensed under MIT; source-dataset access remains governed by each provider.

The output manifest excludes itself and release metadata. The release manifest
excludes itself and `checksums.sha256`; the checksum list includes both
manifests and excludes itself. This prevents circular hashes.
