# Release QC Report

Created UTC: 2026-07-07T15:14:13+00:00
Release tag: `v1.0.0-bja-submission`
Overall status: **PASS**

Scope: `final_public_release` only; `.git`, Python caches, pytest caches, `.DS_Store`, `checksums.sha256`, `release_qc_report.md`, and `outputs/manifests/release_manifest.json` are excluded from risk-content scans.
Checksum rule: `checksums.sha256` covers all upload files except `checksums.sha256` itself and `outputs/manifests/release_manifest.json`; `release_manifest.json` indexes upload files and includes the checksum file, but excludes itself to avoid self-referential hashes.

## Checks
- PASS: Required public release files present
  - All required files are present.
- PASS: No forbidden raw/intermediate data extensions
- PASS: No raw/restricted data directories
- PASS: No private local absolute paths
- PASS: No credential-like assignments or private keys
- PASS: No draft placeholders or author-only markers
- PASS: Scientific claim firewall preserved
  - Positive MoVeR validation, INSPIRE hidden-burden, or postoperative outcome-benefit claims were not found.
- PASS: NIBP pairing wording preserved
  - Median valid 10-s arterial MAP in the -30 to +30 s window with minimum valid fraction 0.80; not nearest single value.
- PASS: Prior public-release audit PASS files retained
  - Existing PASS audit files retained.
- PASS: Public outputs manifest matches present output files
  - 108 output files indexed.

## Claim Boundaries
- VitalDB remains the primary reference dataset.
- MoVeR is not described as direct waveform validation.
- INSPIRE is limited to resource and transportability context.
- No postoperative organ-outcome benefit is claimed.

## Public Release Decision
Private GitHub upload may proceed after this report is PASS. Public release and Zenodo archiving require final author confirmation and repository visibility review.
