from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_MANIFEST = ROOT / "outputs" / "manifests" / "outputs_manifest.json"
RELEASE_MANIFEST = ROOT / "outputs" / "manifests" / "release_manifest.json"
CHECKSUMS = ROOT / "checksums.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_files() -> list[Path]:
    paths = [ROOT / "outputs" / "final_workbook.xlsx", ROOT / "outputs" / "manifests" / "provenance.json"]
    paths.extend((ROOT / "outputs" / "tables").glob("*.csv"))
    paths.extend((ROOT / "outputs" / "figures").glob("*"))
    paths.extend((ROOT / "manuscript_inputs" / "figure_source_files").glob("*.csv"))
    return sorted(path for path in paths if path.is_file())


def repository_files() -> list[Path]:
    excluded = {RELEASE_MANIFEST.resolve(), CHECKSUMS.resolve()}
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.resolve() in excluded or path.suffix == ".pyc":
            continue
        paths.append(path)
    return sorted(paths)


def build_outputs_manifest() -> None:
    write_json(
        OUTPUTS_MANIFEST,
        {
            "release": "v1.2.0-submission",
            "scope": "non-identifiable aggregate outputs and figure artifacts",
            "files": [entry(path) for path in output_files()],
        },
    )


def build_release_manifest() -> None:
    files = repository_files()
    write_json(
        RELEASE_MANIFEST,
        {
            "release": "v1.2.0-submission",
            "scope": "public repository inventory",
            "files": [entry(path) for path in files],
        },
    )
    checksum_lines = [f"{item['sha256']}  {item['relative_path']}" for item in [entry(path) for path in files]]
    checksum_lines.append(f"{sha256(RELEASE_MANIFEST)}  {RELEASE_MANIFEST.relative_to(ROOT).as_posix()}")
    CHECKSUMS.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("outputs", "release"))
    args = parser.parse_args()
    if args.mode == "outputs":
        build_outputs_manifest()
    else:
        build_release_manifest()


if __name__ == "__main__":
    main()
