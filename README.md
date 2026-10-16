# Temporal information loss with intermittent blood pressure display

## Reproducibility package v2.0.1

This package accompanies a retrospective intraoperative study using VitalDB,
with cross-database technical replication in MIMIC-IV Waveform v0.1.0 and
SICdb v1.0.8. Clinical linkage used MIMIC-IV v3.1.

The analysis distinguishes a new low measurement during hypotension from a low
value retained from an earlier event. It also examines how long low values
remain displayed after confirmed recovery. Aggregate exposure, event information
and measurement freshness describe different aspects of intermittent monitoring.
Comparisons across recording resolutions and patient-weighting schemes show
where conclusions depend on the measurement and analysis conventions.

## Patch v2.0.1

Four S15 displayed event counts are corrected (1573 to 1574; 352 to 353;
342 to 343; 280 to 281). Near-integer floating-point residues had been
truncated during presentation. Frozen aggregates, effect estimates and CIs
are unchanged. All four figures now have native editable PowerPoint objects.
Figure 3B has extra lower-axis clearance for the zero estimate.
The preceding v2.0.0 / DOI 10.5281/zenodo.23414088 remains available.

## Current results

- `current/workbook/10_SDC2_Consolidated_Workbook.xlsx`: 25 worksheets aligned
  to the three main tables and Tables S1 to S21. Detailed values retain their
  source precision. Initialisation discrepancies, reporting-gate failures and
  outcome sensitivities are retained, not only matching results.
- `current/figures/`: the four current main figures and editable sources.
- `current/presentation/workbook_plan.json`: aggregate values, notes and block
  structure used to build the workbook.
- `source_layout/`: calculation code, frozen rules and aggregate intermediate
  results in their project-relative layout. Numerical caches are absent.
- `current/provenance/`: result-selection mapping, versions, source-file hashes
  and scoped verification records. The current outputs manifest is here.
  `checksums.sha256` covers the release.

Primary recovery results use 164 distinct recovery confirmation times in 78
patients. The 184 adjustment-recovery pairs are a separately labelled weighting
sensitivity. Common-valid-time resolution comparisons apply to VitalDB and
MIMIC, not to SICdb's native-minute records. These are technical comparisons of
fixed arterial trajectories, not estimates of monitoring-induced treatment delay
or patient benefit. Signal quality checks are algorithmic; independent physician
adjudication was not performed.

## Reproduction

See [the v2 reproduction guide](docs/reproducibility_v2.md) for synthetic tests,
aggregate-only rebuilding and the restricted-input sequence. The package contains
the analysis implementations; it is not a self-contained raw-data-to-manuscript
workflow. Full numerical recomputation requires separately authorised source
data, intermediate caches and historical audit dependencies. This release does
not claim a new clean-room reanalysis of every native waveform.

The root `src`, `scripts`, `extensions`, `code`, `outputs` and `presentation_inputs`
retain the preceding VitalDB pipeline and historical aggregates. Their old table
numbers and 72-sheet workbook are not current submission outputs. Use `current/`
for the current presentation. The complete 125-sheet intermediate workbook stays
in the local historical archive; its disposition is documented publicly.

## Data access and privacy

No raw recordings, patient/operation-level records, linkage keys, fitted patient
tables, private configuration, manuscripts or internal case-review materials are
included. Obtain datasets independently under their original terms:

- [VitalDB](https://vitaldb.net/dataset/)
- [MIMIC-IV Waveform v0.1.0](https://physionet.org/content/mimic4wdb/0.1.0/)
- [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
- [SICdb v1.0.8](https://physionet.org/content/sicdb/1.0.8/)

Clinical MIMIC-IV requires credentialing and a data use agreement. SICdb also
requires contributor approval of the research project. This software archive
neither grants dataset access nor constitutes an ethics approval or a provider
determination of project scope. The MIT licence covers software; source-dataset
licences and data-use restrictions remain separate.

## Versions and citation

Use the version-specific Zenodo DOI shown in the
[v2.0.1 release record](https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v2.0.1).
The preceding [v1.5.0](https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v1.5.0)
and [DOI 10.5281/zenodo.23255075](https://doi.org/10.5281/zenodo.23255075)
remain unchanged. They cover earlier VitalDB analyses, not this extension.

Frozen plans retain their original dates and local-only wording as provenance.
They were frozen before the specified additional computations, after some earlier
results had been inspected; they are not prospective registrations. Public-release
authorisation was given separately on 16 October 2026.

## Authors and acknowledgements

Qingyu Teng; Qi Li; Ziyan Gu; Yuping Yang; Junde Han; Qian Chen; Jing Zhao;
Yingya Zhao; Hui Zhang. Qingyu Teng, Qi Li and Ziyan Gu contributed equally.
Jing Zhao, Yingya Zhao and Hui Zhang are corresponding authors, with Hui Zhang
the primary correspondent. We thank Tao Xu for his guidance and comments.

The authors used OpenAI Codex/ChatGPT for language assistance, code review and
reproducibility checks. Scientific interpretation and final approval remain the
responsibility of the human authors.
