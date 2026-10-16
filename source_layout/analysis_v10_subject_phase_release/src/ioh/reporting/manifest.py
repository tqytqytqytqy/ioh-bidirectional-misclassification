from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_outputs_manifest(root: Path, files: list[Path], output_path: Path) -> dict:
    root = Path(root)
    root_resolved = root.resolve()
    entries = []
    for path in sorted(files):
        path = Path(path)
        if not (path.exists() and path.is_file()):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            continue
        entries.append(
            {
                "relative_path": str(resolved.relative_to(root_resolved)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "files": entries
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _manifest_entry(root: Path, path: Path) -> dict:
    path = Path(path)
    resolved = path.resolve()
    root_resolved = Path(root).resolve()
    relative_path = str(resolved.relative_to(root_resolved)) if resolved.is_relative_to(root_resolved) else None
    entry = {
        "relative_path": relative_path,
        "absolute_path": str(resolved) if relative_path is None else None,
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        entry.update({"size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return entry


def build_run_manifest(
    root: Path,
    config_path: Path,
    code_paths: list[Path],
    input_paths: list[Path],
    output_paths: list[Path],
    git_commit: str | None = None,
) -> dict:
    root = Path(root)
    return {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit or "not-a-git-repository",
        },
        "config": _manifest_entry(root, Path(config_path)),
        "code": [_manifest_entry(root, Path(path)) for path in sorted(code_paths)],
        "inputs": [_manifest_entry(root, Path(path)) for path in sorted(input_paths)],
        "outputs": [_manifest_entry(root, Path(path)) for path in sorted(output_paths)],
    }


def mover_claim_gate(gold_links: int, silver_links: int, ambiguous_rejects: int, min_gold: int = 1000) -> dict:
    allowed = int(gold_links) >= int(min_gold)
    language = (
        "Direct waveform corroboration is allowed by the prespecified linkage gate."
        if allowed
        else "No direct waveform claim; MoVeR may be described only as a gate-failed waveform audit or limited fallback."
    )
    return {
        "gold_links": int(gold_links),
        "silver_links": int(silver_links),
        "ambiguous_rejects": int(ambiguous_rejects),
        "minimum_gold_required": int(min_gold),
        "direct_waveform_claim_allowed": bool(allowed),
        "language": language,
    }
