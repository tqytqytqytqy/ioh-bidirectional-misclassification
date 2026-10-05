from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v1.4.0'
OUTPUT = ROOT / 'outputs/manifests/outputs_manifest.json'
RELEASE = ROOT / 'outputs/manifests/release_manifest.json'
SUMS = ROOT / 'checksums.sha256'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(path):
    return {'relative_path': path.relative_to(ROOT).as_posix(), 'size_bytes': path.stat().st_size, 'sha256': sha(path)}


def paths():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git', '__pycache__', '.pytest_cache']) and p.suffix != '.pyc')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def outputs():
    payload = [p for p in paths() if 'outputs' in p.relative_to(ROOT).parts and p not in {OUTPUT, RELEASE} and 'qc' not in p.relative_to(ROOT).parts]
    write(OUTPUT, {'release': VERSION, 'scope': 'Current non-identifiable aggregate outputs, figure source data, figures and consolidated workbook', 'files': [entry(p) for p in payload]})


def release():
    payload = [p for p in paths() if p not in {RELEASE, SUMS}]
    write(RELEASE, {'release': VERSION, 'scope': 'Exact public repository inventory', 'files': [entry(p) for p in payload]})
    SUMS.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(payload + [RELEASE])))
    verify()


def verify():
    for file in [OUTPUT, RELEASE]:
        record = json.loads(file.read_text())
        assert record['release'] == VERSION
        assert all(entry(ROOT / i['relative_path']) == i for i in record['files'])
    expected = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths() if p != SUMS}
    actual = {}
    for line in SUMS.read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert name not in actual
        actual[name] = digest
    assert actual == expected
    print(json.dumps({'status': 'PASS_EXACT_PUBLIC_INVENTORY', 'release': VERSION, 'files': len(actual), 'outputs_manifest_sha256': sha(OUTPUT), 'workbook_sha256': sha(ROOT / 'outputs/final_workbook.xlsx')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['outputs', 'release', 'verify'])
    mode = parser.parse_args().mode
    {'outputs': outputs, 'release': release, 'verify': verify}[mode]()
