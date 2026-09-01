# Release notes

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
