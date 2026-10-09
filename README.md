# Temporal observability of intraoperative hypotension

## Reproducibility package v1.5.0

This release contains the code and aggregate results for a retrospective waveform
study of an emulated intermittent blood pressure display. It preserves the original
analyses and the completed rule-based startup signal-quality sensitivities.

At a 5 min interval, 34.2% of qualifying reference episodes had no low display;
55.9% acquired a new low sample within the episode and 9.9% had inherited low
display only. Similar cumulative exposure therefore did not imply similar
within-episode information. Display-state analyses at recorded infusion
adjustments provide a clinical timing context, not a treatment-effect estimate.

No independent physician clinical adjudication was performed. Original analyses
remain primary and startup-QC analyses remain sensitivities. No source-data
analysis or model fitting is rerun during release assembly.

## Contents

- Four main tables and 16 supplementary tables, represented by 26 supplementary panels.
- A 72-sheet consolidated workbook, retaining current presentation and clearly named historical and unrounded aggregate results.
- Four editable main figures and two supplementary figures.
- Python, R and JavaScript analysis and presentation code, synthetic tests and configuration templates.
- File-integrity manifests, aggregate quality checks and reproduction documentation.

`outputs/tables` contains current tables. `outputs/legacy_tables` and
`presentation_inputs` preserve historical aggregate results and the numbering
map; they are not additional current manuscript tables. See `docs/table_numbering.md`.

## Data Access

Raw VitalDB, MoVeR and INSPIRE datasets are not redistributed. Source recordings,
case-level trajectories, paired measurements, laboratory rows, model frames,
private review images, private configuration and submission documents are excluded.
Obtain source data independently under their original dataset terms.
The MIT licence covers the supplied software; source-data terms remain separate.

Use `config/inputs.example.sh`, `config/reproduction.example.yaml` and the documented environment variables for
private source locations. Do not commit private inputs or local configuration.
See `docs/reproducibility.md` and `docs/data_access.md` for reproduction boundaries.

## Version And Citation

Canonical release: [v1.5.0](https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v1.5.0).
The preceding public snapshot is [v1.4.0](https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v1.4.0)
and [Zenodo 23153577](https://doi.org/10.5281/zenodo.23153577).
Use the version-specific DOI linked from the release notes and Zenodo record,
not the concept DOI, when citing this exact snapshot.

Final workbook SHA-256: 3733746c860584c799f597e714c609efeee037daa6d5c5b0682d647ae3d52171

Outputs manifest SHA-256: a37967894feae2680c66868e776e3a18d277c534be7840e1b93c478fc1c55bdc

These two hashes identify the scientific output snapshot. `checksums.sha256`
and `outputs/manifests/release_manifest.json` cover the frozen release files.

## Authors And Acknowledgements

Qingyu Teng; Qi Li; Ziyan Gu; Yuping Yang; Junde Han; Qian Chen; Jing Zhao;
Yingya Zhao; Hui Zhang. Creator metadata follows the author order confirmed
for this release. Existing version records retain their historical creators.

We thank Tao Xu for his guidance and comments on this study.
