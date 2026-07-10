# Bidirectional misclassification of intraoperative hypotension exposure during intermittent blood pressure monitoring

This repository is the reproducibility package for a retrospective public-database trajectory-emulation study of bidirectional hypotension-exposure misclassification.

The release contains analysis code, configuration templates, reproducibility manifests, figure source data, the final workbook, and non-identifiable aggregate outputs.

Raw VitalDB, MoVeR, and INSPIRE datasets are not redistributed. Obtain each dataset from its original provider and comply with the applicable data-use terms.

## Primary result

In 2435 selected VitalDB arterial-reference cases, phase-averaged fixed 5-min display at MAP <65 mm Hg yielded HDR 0.597 (95% CI 0.581-0.613), ODR 0.573 (0.558-0.587), and net bias -0.024 (-0.027 to -0.022). Reference-missing time points were excluded. A valid reference with no recent displayed value contributed to the unavailable-display hidden component.

The NIBP method-comparison subset was the predefined first 500 eligible candidates in source-manifest order. Of these, 365 entered mechanism decomposition and 347 contributed paired events. Pairing used the median of valid 10-s arterial MAP samples in the -30 to +30 s window around recorded event time, with at least 80% valid coverage; it did not use the nearest single arterial value.

MoVeR failed the prespecified direct-waveform gate, so no direct waveform claim is made. INSPIRE supports target-population, transportability, and resource scenarios only. Monitoring-policy simulations are hypothesis-generating and do not establish postoperative benefit.

## Reproduce

1. Create a Python 3.12 environment from `requirements.txt`.
2. Obtain the source datasets from their providers.
3. Make a private copy of `config/config.example.yaml` outside version control and set local data paths.
4. Run `make analysis CONFIG=/path/to/private-analysis.yaml`, then `make tables`, `make figures`, and `make test` with the same `CONFIG` value.

## Frozen release

- Tag: `v1.1.0-bja-submission`
- Aggregate workbook: `outputs/final_workbook.xlsx`
- Aggregate workbook SHA-256: `b7e12b126e0ba644d1ccbf22758c1fcfdcceee03d4de4ae1ba6e0627cf55e237`
- Outputs manifest: `outputs/manifests/outputs_manifest.json`
- Outputs manifest SHA-256: `458b9aeecb751cc26ad0b819702c2d71291f16144ab3fe1d57a4d307a5fb92a9`

The exact file-level inventory is recorded in `outputs/manifests/release_manifest.json` and `checksums.sha256`.
