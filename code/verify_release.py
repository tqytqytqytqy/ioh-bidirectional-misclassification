"""Verify the frozen file inventories without executing source-data analyses."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_files(root):
    return [p for p in root.rglob('*') if p.is_file()
            and not any(part in {'.git', '__pycache__', '.pytest_cache'} for part in p.relative_to(root).parts)]


def verify(root):
    output_manifest = root / 'outputs/manifests/outputs_manifest.json'
    release_manifest = root / 'outputs/manifests/release_manifest.json'
    manifests = root / 'outputs/manifests'
    for filename in [output_manifest, release_manifest]:
        value = json.loads(filename.read_text())
        expected = {row['path'] for row in value['files']}
        if filename == output_manifest:
            actual = {p.relative_to(root).as_posix() for p in public_files(root / 'outputs')
                      if manifests not in p.parents}
        else:
            actual = {p.relative_to(root).as_posix() for p in public_files(root)
                      if p.relative_to(root).as_posix() not in {'checksums.sha256', 'outputs/manifests/release_manifest.json'}}
        assert expected == actual, filename.name
        for row in value['files']:
            path = (root / row['path']).resolve()
            assert path.is_relative_to(root.resolve())
            assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], row['path']
    lines = (root / 'checksums.sha256').read_text().splitlines()
    listed = set()
    for line in lines:
        checksum, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve()) and sha(path) == checksum, relative
        listed.add(relative)
    actual = {p.relative_to(root).as_posix() for p in public_files(root) if p.name != 'checksums.sha256'}
    assert listed == actual
    return {'status': 'PASS', 'verified_files': len(listed),
            'version': json.loads(release_manifest.read_text())['version']}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1])))
