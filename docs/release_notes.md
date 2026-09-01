# Release notes

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
