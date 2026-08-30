# Reproducibility guide

## Environment

The release was checked with Python 3.12. Exact package versions are listed in `environment.txt`; installable requirements are in `requirements.txt`. The primary implementation is Python. Independent checks for the principal AKI and ICU regression models are supplied as R scripts.

## Public verification

The included tests exercise decomposition identities, episode detection, waveform quality rules, NIBP event reconstruction, AKI models, ICU models, and manifest determinism. They use synthetic data and do not require source datasets.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
```

The workbook contains a README worksheet and 35 aggregate-table worksheets. One additional supplementary cohort-characteristics CSV is supplied separately. Figure source CSVs accompany the PDF and PNG outputs. File integrity can be checked against `checksums.sha256` and the JSON manifests.

## Full regeneration

Full regeneration requires independently obtained VitalDB data and a private configuration derived from `config/reproduction.example.yaml`. Restricted source data and patient-level intermediates must remain outside the public repository.
