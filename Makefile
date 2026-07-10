CONFIG ?= config/config.example.yaml
PYTHON ?= python3

.PHONY: analysis tables figures test workbook clean

analysis:
	$(PYTHON) scripts/00_validate_config.py --config $(CONFIG)
	$(PYTHON) scripts/01_build_vitaldb_manifest.py --config $(CONFIG)
	$(PYTHON) scripts/02_preprocess_vitaldb_artmap.py --config $(CONFIG)
	$(PYTHON) scripts/03_emulate_frequency_decomposition.py --config $(CONFIG)
	$(PYTHON) scripts/04_build_nibp_display_events.py --config $(CONFIG)
	$(PYTHON) scripts/05_pair_nibp_with_artmap.py --config $(CONFIG)
	$(PYTHON) scripts/06_decompose_nibp_timing_vs_cuff.py --config $(CONFIG)
	$(PYTHON) scripts/07_strategy_frontier.py --config $(CONFIG)
	$(PYTHON) scripts/08_inspire_transportability.py --config $(CONFIG)
	$(PYTHON) scripts/09_mover_gate_optional.py --config $(CONFIG)
	$(PYTHON) scripts/18_execute_revision_v6.py --config $(CONFIG)

tables: analysis
	$(PYTHON) scripts/10_make_tables.py --config $(CONFIG)

figures: tables
	$(PYTHON) scripts/11_make_figures.py --config $(CONFIG)
	$(PYTHON) scripts/12_make_manuscript_numbers.py --config $(CONFIG)

workbook: tables
	$(PYTHON) scripts/13_build_workbook.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf outputs/intermediate outputs/logs
