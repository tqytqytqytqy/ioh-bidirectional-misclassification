# Release notes

## v1.4.0

This release contains analysis code, configuration templates, reproducibility
manifests, figure source data, final workbook, editable figures and
non-identifiable aggregate outputs.

Raw VitalDB, MoVeR and INSPIRE datasets are not redistributed.

Changes since v1.3.1:

- added new-versus-inherited low-value classification, case/phase distributions,
  common-denominator interval summaries and event-definition robustness;
- added recorded vasopressor infusion-adjustment display states and restricted
  recovery persistence, with a fixed protocol and synthetic tests;
- corrected NIBP record semantics: retained numerical output is not verified
  independent cuff cycles; legacy cuff-agreement and diagnostic outputs are
  excluded from the current publication artifacts;
- replaced the prior figures, aggregate presentation tables and workbook with
  the current four main figures, four main tables, 29 supplementary tables and
  40-sheet workbook;
- retained baseline statistical implementations and independent R checks;
- retained the current nine-creator order and created one current output
  manifest and a complete repository checksum inventory.

The release assembles previously frozen aggregate results; it does not claim
a new source-data refit, treatment-delay estimate or monitoring outcome benefit.

## v1.3.1-submission

This metadata-only patch aligns the archived reproducibility package with the final manuscript title and authorship.

Changes since v1.3.0:

- updated the package title to match the manuscript wording;
- added Tao Xu to the creator metadata between Qi Li and Hui Zhang;
- retained Hui Zhang as the final creator in the listed manuscript order;
- updated repository citation metadata and the repository-level integrity inventory;
- left all analysis code, aggregate results, figures, `outputs/final_workbook.xlsx`, and `outputs/manifests/outputs_manifest.json` unchanged.

The package does not redistribute source datasets or patient-level data.

## v1.3.0-submission

This release aligns the public reproducibility package with the subject-clustered temporal-observability manuscript revision.

Changes since v1.2.0:

- changed observational bootstrap uncertainty from case resampling to subject-cluster resampling while retaining every operation from each sampled subject;
- added a direct case-versus-subject bootstrap sensitivity analysis;
- added sensitivity analyses that initialize the display with a baseline value or exclude the first scheduled sampling interval;
- expanded the STROBE cohort-flow source data and corrected Table 1 presentation of sex and missing ASA Physical Status values;
- reran the exploratory AKI and ICU models with subject-cluster robust variance;
- replaced all aggregate tables, figure source data, figures, workbook, and integrity manifests with the current frozen outputs;
- synchronized creator metadata with the current author order;
- retained a strict redistribution boundary excluding source datasets and patient-level derivatives.

The package does not estimate a causal effect of monitoring frequency on management or postoperative outcomes.
