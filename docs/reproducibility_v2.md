# Reproduction of the cross-database extension

## Without patient data

Use Python 3.12 with `current/requirements.txt`. From the repository root:

```sh
python -B current/code/test_extension.py
python -B code/run_synthetic_tests.py
python -B source_layout/EJA_审稿意见逐项修订版_20261016/code/build_figures.py
node current/code/build_workbook.mjs
python -B current/code/verify_snapshot.py
```

The first two commands use synthetic inputs. The figure command rebuilds Figures
3 and 4 from aggregate CSVs. Figure 1 has an editable PowerPoint source; Figure 2
has an editable SVG and an illustrative schematic, not a patient trace. Final
figures are supplied in `current/figures/`.

The workbook builder uses `@oai/artifact-tool` and included aggregate JSON. It
does not require source DOCX files. Other users can read the XLSX or JSON directly
without that package. Rebuilt XLSX metadata may vary; logical cells, not ZIP bytes,
should be compared. The snapshot verifier checks workbook values against JSON,
privacy boundaries and scientific identities. Historical numerical QA records
describe their original execution, not a new execution during release.

## Scientific calculation chain

`source_layout` retains the original dependency layout. The following is the
computation sequence, not a claim of execution from an empty checkout:

1. The legacy VitalDB scripts produce `artmap_10s.parquet` and
   `vitaldb_manifest.parquet`. Startup-screening and infusion extensions produce
   waveform-quality and treatment-event caches.
2. The MIMIC feasibility audit, full-record waveform QC and cache finalisation
   require Waveform v0.1.0 files/checksums and MIMIC-IV v3.1 linkage tables.
3. SICdb `extract_sicdb.py` reads an authorised v1.0.8 ZIP. The minute analysis
   produces frozen reference shards and component caches.
4. Reliable-onset `qc_onset.py`, `rerun.py` and `boundary_sensitivity.py`.
5. Cross-database `audit_subgroups.py` with explicit source arguments,
   `run_analysis.py`, `summarize.py` and `independent_verify.py`.
6. Clinical-time `run_timing.py`, `summarize_timing.py` and its verifier.
7. Review `run_review.py recovery`, `run_review.py bridge` and its verifier.

Raw-location environment variables are `VITALDB_RAW_ROOT`, `VITALDB_TRACK_ROOTS`
(operating-system path separator), `MIMIC_WAVEFORM_ROOT`, `MIMIC_CLINICAL_ZIP`
and `SICDB_ZIP`. Defaults contain no user-machine locations. Numerical algorithms
and thresholds were not changed during packaging. Exact original and public
code hashes are in `current/provenance/source_file_map.json`.

## Intentionally absent inputs

- VitalDB intermediate parquet files, waveform metrics, track metadata,
  `onsets.csv`, `startup_quality_bins.csv.gz` and treatment events.
- MIMIC record/linkage metadata, `qc_inputs.npz`, `qc_ready.json` and the
  feasibility audit caches used to build full-record QC.
- SICdb authorised source ZIP, private minute shards and ready manifest,
  source-specific patient/record components.
- Cross-database `subgroup_metadata.csv`, record components and event strata.
- Individual timing/recovery records and the historical baseline referenced
  by frozen audit manifests.

Subgroup metadata are required to reproduce manuscript cohorts and surgical
context. The inherited reader returns an empty mapping if the file is absent;
do not treat that run as manuscript-equivalent. Require the input and verify
cohort denominators before a scientific rerun.

Historical freeze guards reference earlier source documents, results and
absolute-path hash manifests. They are retained for auditing and fail in a clean
checkout without those dependencies. Do not disable them and claim exact
reproduction. An independently populated environment requires a documented input
mapping, source-hash checks and matching cohort counts. No runtime dependency
installation, native data or original Word manuscript is bundled.

## Versions and estimands

Cross-database seed: 20260706, 1000 patient-cluster bootstraps. Clinical-time seed:
20261015. Distinct-recovery/common-support seed: 20261016, 2000 recovery and 1000
bridge bootstraps. Original paired-support CIs in the submitted workbook retain
the original cross-database seed; later duplicate recomputation is not substituted.

The 164 distinct recoveries, 184 adjustment-recovery pairs and restricted-mean
minutes are different estimands. Unknown follow-up remains in denominators.
Common-minute support requires all six 10-second values valid without compressing
elapsed time. It cannot supply native 10-second SICdb results. No AKI model was
refitted during release assembly.

Root `outputs/`, `presentation_inputs/`, `docs/reproducibility.md` and
`docs/scientific_reproduction.md` document the earlier VitalDB snapshot, not the
current table numbering. The 25-sheet workbook removes duplicate presentation,
not unfavourable findings. Its selection map and full historical-sheet disposition
are public; the complete 125-sheet intermediate workbook remains locally archived.
