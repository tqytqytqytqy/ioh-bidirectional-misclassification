# Bidirectional misclassification of intraoperative hypotension exposure under intermittent blood pressure monitoring

This repository contains analysis code, configuration templates, reproducibility manifests, figure-source data, and non-identifiable aggregate outputs for the IOH2 submission.

Raw source datasets from VitalDB, MoVeR, and INSPIRE are not redistributed. Users must obtain source data from the original providers and comply with each provider's data-use terms.

Primary estimand: fixed 5-min MAP <65 mm Hg hidden deficit ratio (HDR65), with overdisplay deficit ratio (ODR65) as the key secondary endpoint. Actual NIBP pairing uses the median of valid 10-s arterial MAP samples in the -30 to +30 s window around cuff display completion, with minimum valid fraction 0.80; it is not the nearest single arterial MAP value.

## Reproduce

1. Create the Python environment from `requirements.txt`.
2. Obtain source data from the original providers.
3. Copy `config/config.example.yaml` to a private local YAML file outside git tracking and set local source-data paths.
4. Run `make clean_outputs all test CONFIG=<private-local-yaml>`.

Release tag: `v1.0.0-bja-submission`
Stable outputs manifest SHA-256: `774ef0d29e59b40bab71f78d00a46dbaf1f8b95c1ee99cceb29e310484ea582d`
Public aggregate workbook SHA-256: `d3556bbe3313fc3d0c5dc6fe60828058580b1aa82b8f77a2dabc805c64c3811d`
