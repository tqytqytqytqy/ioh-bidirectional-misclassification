from pathlib import Path

from ioh.reporting.manifest import build_run_manifest, file_sha256, write_outputs_manifest


def test_manifest_hash_is_deterministic(tmp_path: Path):
    table = tmp_path / "table.csv"
    table.write_text("a,b\n1,2\n", encoding="utf-8")

    first = file_sha256(table)
    second = file_sha256(table)
    manifest = write_outputs_manifest(tmp_path, [table], tmp_path / "manifest.json")

    assert first == second
    assert manifest["files"][0]["sha256"] == first
    assert manifest["files"][0]["relative_path"] == "table.csv"


def test_run_manifest_records_code_config_inputs_and_outputs(tmp_path: Path):
    config = tmp_path / "config.yaml"
    code = tmp_path / "pipeline.py"
    raw = tmp_path / "raw.csv"
    output = tmp_path / "outputs" / "table.csv"
    output.parent.mkdir()
    config.write_text("project: test\n", encoding="utf-8")
    code.write_text("print('ok')\n", encoding="utf-8")
    raw.write_text("id\n1\n", encoding="utf-8")
    output.write_text("a\n1\n", encoding="utf-8")

    manifest = build_run_manifest(root=tmp_path, config_path=config, code_paths=[code], input_paths=[raw], output_paths=[output])

    assert manifest["config"]["relative_path"] == "config.yaml"
    assert manifest["code"][0]["relative_path"] == "pipeline.py"
    assert manifest["inputs"][0]["relative_path"] == "raw.csv"
    assert manifest["outputs"][0]["relative_path"] == "outputs/table.csv"
    assert manifest["run"]["timestamp_utc"]


def test_run_manifest_records_external_inputs_with_absolute_path(tmp_path: Path):
    config = tmp_path / "config.yaml"
    code = tmp_path / "pipeline.py"
    output = tmp_path / "outputs" / "table.csv"
    external_root = tmp_path.parent / f"{tmp_path.name}_external"
    external_root.mkdir()
    raw = external_root / "raw.csv"
    output.parent.mkdir()
    config.write_text("project: test\n", encoding="utf-8")
    code.write_text("print('ok')\n", encoding="utf-8")
    raw.write_text("id\n1\n", encoding="utf-8")
    output.write_text("a\n1\n", encoding="utf-8")

    manifest = build_run_manifest(root=tmp_path, config_path=config, code_paths=[code], input_paths=[raw], output_paths=[output])

    assert manifest["inputs"][0]["relative_path"] is None
    assert manifest["inputs"][0]["absolute_path"] == str(raw.resolve())


def test_outputs_manifest_skips_files_outside_manifest_root(tmp_path: Path):
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    table = out_root / "table.csv"
    external = tmp_path / "manuscript" / "main.docx"
    external.parent.mkdir()
    table.write_text("a\n1\n", encoding="utf-8")
    external.write_text("not inside outputs\n", encoding="utf-8")

    manifest = write_outputs_manifest(out_root, [table, external], out_root / "manifest.json")

    assert [item["relative_path"] for item in manifest["files"]] == ["table.csv"]
