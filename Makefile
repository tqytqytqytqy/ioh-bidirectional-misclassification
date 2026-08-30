PYTHON ?= python3
CONFIG ?= reproduction.yaml

.PHONY: test manifest preprocess primary tables figures aki icu

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

manifest:
	$(PYTHON) scripts/01_build_vitaldb_manifest.py --config $(CONFIG)

preprocess:
	$(PYTHON) scripts/02_preprocess_vitaldb_artmap.py --config $(CONFIG)

primary:
	$(PYTHON) scripts/03_emulate_frequency_decomposition.py --config $(CONFIG)
	$(PYTHON) scripts/27_execute_anesthesiology_revision_v7.py --config $(CONFIG)

tables:
	$(PYTHON) scripts/28_make_anesthesiology_tables_figures_v7.py --config $(CONFIG)

figures: tables

aki:
	$(PYTHON) scripts/33_run_exploratory_aki_analysis_v7.py --config $(CONFIG)

icu:
	$(PYTHON) scripts/39_run_exploratory_icu_resource_use_v74.py --input $(ICU_INPUT) --output-dir $(ICU_OUTPUT)
