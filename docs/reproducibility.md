# Presentation Reproduction

Release v1.5.0 packages the completed v1.5.0-algorithmic-qc analysis snapshot, with rule-based signal-quality sensitivity analysis.
The current presentation has 4 main tables, 16 supplementary tables (26 panels) and 2 supplementary figures.
This revision changes titles, notes, grouping, numbering and navigation only.
It does not rerun statistical calculations or refit models.

## Aggregate-Only Presentation Inputs

The complete 35-table source is `presentation_inputs/source_tables.json`.
The current target is `outputs/tables/condensed_tables.json` and the mapping is
`presentation_inputs/table_map.json`. Titles and notes come from these supplied
inputs, not from historical statistical table names. Retained original main
table CSVs are byte-identical. A workbook-only Main Table 2 pointer changes no
numerical cell. The current S2 caption is separate from the unchanged historical
caption in `outputs/aggregate/qc`.

Portable presentation code is `code/condense_tables.py`; inspect its help for
supported arguments. Archive copying is `code/build_condensed_archive.py`.
For aggregate-only reconstruction outside the frozen archive directory:

    python code/condense_tables.py --source presentation_inputs/source_tables.json --output ../presentation_rebuild

Neither archive staging nor hash verification establishes clinical validity.
No clean independent raw-source-to-release execution is claimed.

## Historical Scientific Reproduction

All inherited scientific code and aggregate outputs are retained unchanged.
Historical source dependencies, statistical commands and QC policy assumptions
are documented in `docs/scientific_reproduction.md` and `docs/qc_protocol.md`.
Those commands reproduce the prior scientific outputs with 35-table numbering,
not the current 16-table presentation. Do not invoke the historical archive
builder to update this presentation. Raw data and individual derivatives must
be obtained and governed separately; they are not redistributed here.

Python numerical requirements remain `requirements.txt` and
`requirements-qc.txt`. The boundary checker additionally uses PyYAML and pypdf.
No installed runtime or Node dependency directory is bundled.

## Scan and Approved Freeze

`--scan` performs one complete archive boundary scan and records fingerprints
of scanned payloads. A later freeze rechecks the fingerprints; it does not
needlessly rescan unchanged payloads. After explicit approval, `--freeze`
requires `--expected-workbook-sha256`, `--expected-condensed-sha256`, and
`--expected-s2-caption-sha256`. Changing any approved input prevents freezing.
The local final workbook QA receipt must match the current JSON and workbook
hashes, verified original-data preservation, and exact current worksheet notes.
Its aggregate-only check is `outputs/qc/workbook_presentation_checks.json`.

One current output manifest excludes all manifests. The release manifest
includes that output manifest but excludes itself and `checksums.sha256`.
The checksum list includes both manifests and excludes itself. These form a
noncircular sequence, generated only after approval. Old freeze files and `.skip`
files are not reused. `--verify` checks the inventories and hashes without upload.
