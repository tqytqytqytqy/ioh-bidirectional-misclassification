"""Stage an aggregate-only presentation update without statistical execution.

Default operation only copies and documents. --scan performs one complete
boundary scan. --freeze requires explicit approval of three stable SHA-256
fingerprints and a current passing scan. No workbook writes or uploads occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.dont_write_bytecode = True
VERSION = 'v1.5.0-algorithmic-qc'
BASELINE_DOI = 'https://doi.org/10.5281/zenodo.23153577'
BASELINE_GITHUB = 'https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v1.4.0'
SOURCE_RELATIVE = '独立质控重分析_20261007/归档候选'
ARCHIVE_NAME = '03_可复现材料'
HELPER = 'extensions/qc_startup/build_archive.py'
FINAL_FILES = {'checksums.sha256', 'outputs/manifests/outputs_manifest.json',
               'outputs/manifests/release_manifest.json', 'outputs/manifests/provenance.json'}
CONTROL_FILES = {'ARCHIVE_STATUS.json', 'outputs/qc/condensed_staging_report.json',
                 'outputs/qc/archive_scan.json'} | FINAL_FILES
UPDATED_DOCUMENTS = {'README.md', 'CITATION.cff', 'codemeta.json', '.zenodo.json',
                     'Makefile', 'docs/reproducibility.md', 'docs/release_notes.md',
                     'docs/staging_inventory.json'}
APPROVAL_PATHS = {'workbook': 'outputs/final_workbook.xlsx',
                  'condensed': 'outputs/tables/condensed_tables.json',
                  'caption': 'outputs/tables/Supplementary_Figure_S2_caption.txt'}
DESCRIPTION = ('LOCAL NOT PUBLISHED. Presentation-only condensation to 16 supplementary tables '
               'and 2 supplementary figures; independent physician clinical adjudication not performed. No additional statistical rerun during presentation processing. '
               'Original v1.4.0 primary results and historical QC sensitivity outputs are retained. '
               'Aggregate results and portable code only; raw data are not redistributed.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reject_symlinks(path):
    path = Path(path)
    if any(part.is_symlink() for part in [path, *path.parents]):
        raise ValueError('Refusing symlink directory')
    if path.exists() and any(item.is_symlink() for item in path.rglob('*')):
        raise ValueError('Refusing symlink within directory')


def snapshot_tree(root):
    reject_symlinks(root)
    return [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
             'sha256': sha(path)} for path in sorted(Path(root).rglob('*')) if path.is_file()]


def load_legacy_helper(path):
    spec = importlib.util.spec_from_file_location('ioh_aggregate_archive_boundary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    return module


def copied_relative(relative):
    path = Path(relative)
    if (relative == 'outputs/final_workbook.xlsx' or relative == 'ARCHIVE_STATUS.json'
            or relative == 'checksums.sha256' or path.suffix == '.skip'
            or relative.startswith('outputs/manifests/')):
        return None
    if relative in UPDATED_DOCUMENTS:
        return 'docs/legacy/' + relative
    if relative == 'outputs/qc/archive_scan.json':
        return 'outputs/qc/legacy_archive_scan.json'
    if relative == 'outputs/qc/S2_checks.json':
        return 'outputs/qc/legacy_S2_checks.json'
    if relative.startswith('outputs/tables/'):
        return 'outputs/legacy_tables/' + relative.removeprefix('outputs/tables/')
    return relative


def table_number(table):
    value = table.get('new_number', table.get('number'))
    if value is None:
        match = re.fullmatch(r'Supp_Table_S0*(\d+)', table.get('name', ''))
        value = match.group(1) if match else None
    if isinstance(value, bool) or value is None:
        raise ValueError('Current table needs an explicit supplementary number')
    return int(value)


def current_tables(value):
    tables = value.get('tables') if isinstance(value, dict) else value
    if not isinstance(tables, list):
        raise ValueError('Condensed presentation must contain a table list')
    if len(tables) != 16 or sorted(table_number(table) for table in tables) != list(range(1, 17)):
        raise ValueError('Current table numbers must be exactly 1 through 16')
    for table in tables:
        if not table.get('title') or not isinstance(table.get('notes'), list):
            raise ValueError('Incomplete or nonrectangular condensed presentation table')
        for panel in table_panels(table):
            values = panel.get('values')
            if (not isinstance(panel.get('notes'), list) or not isinstance(values, list)
                    or not values or not values[0]
                    or not all(isinstance(row, list) and len(row) == len(values[0]) for row in values)):
                raise ValueError('Incomplete or nonrectangular condensed presentation panel')
    return sorted(tables, key=table_number)


def table_panels(table):
    panels = table.get('panels', [table])
    if not isinstance(panels, list) or not panels:
        raise ValueError('Current supplementary table has no panels')
    return panels


def csv_text(values):
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\n').writerows(values)
    return stream.getvalue()


def write_documentation(archive, tables, missing):
    counts = '4 main tables, 16 supplementary tables (26 panels) and 2 supplementary figures'
    baseline = f'[GitHub v1.4.0]({BASELINE_GITHUB}) and [Zenodo 23153577]({BASELINE_DOI})'
    titles = '\n'.join(f'- S{table_number(t)}: {t["title"]}' for t in tables)
    mapping_path = archive / 'presentation_inputs/table_map.json'
    mapping = read_json(mapping_path) if mapping_path.is_file() else {}
    groups = {group['new_number']: group for group in mapping.get('groups', [])}
    mapping_lines = ['| Current table | Historical workbook sources | Panels |', '| --- | --- | --- |']
    for table in tables:
        n = table_number(table)
        old_numbers = groups.get(n, {}).get('old_numbers', table.get('source_old_tables', []))
        old_names = ', '.join(f'Legacy_S{old:02}' for old in old_numbers) or 'Not supplied'
        mapping_lines.append(f'| S{n} | {old_names} | {len(table_panels(table))} |')
    numbering_map = '\n'.join(mapping_lines)
    pending = '\n'.join('- ' + name for name in missing) or '- No presentation inputs missing.'
    docs = {
        'README.md': f'''# Temporal observability of intraoperative hypotension

## {VERSION}: LOCAL NOT PUBLISHED

Presentation-only local candidate with {counts}.
Independent physician clinical adjudication was not performed and is not a pending requirement. No additional statistical rerun during presentation processing, model refit,
clinical validation, public release, upload or DOI assignment was performed.
Original analyses remain primary; startup-QC analyses remain sensitivities.

{baseline} identify the BASELINE ONLY, not this candidate.
There is no new public GitHub release or DOI for the condensed candidate.

## Current and Historical Contents

- `outputs/tables`: current main Tables 1-4, supplementary Tables S1-S16,
  the authoritative condensed JSON and the current S2 caption when supplied.
  Multi-panel CSVs use `_panel_1`, `_panel_2` and subsequent suffixes. One CSV
  is one panel, not necessarily one complete publication table.
- `presentation_inputs/source_tables.json`: all 35 historical publication
  tables, including original titles, notes and aggregate cells. These are
  presentation inputs, not individual records or the new table numbering.
- `presentation_inputs/table_map.json`: authoritative old-to-new numbering.
- `outputs/legacy_tables`: every original publication CSV, byte-preserved.
- `outputs/aggregate` and `outputs/aggregate/qc`: all old scientific results,
  unchanged. Old table titles and S1-S35 here are HISTORICAL OUTPUTS ONLY.
- `outputs/qc`: aggregate verification only. Inherited checks concern their
  original scope; they do not certify the new condensed documents.
  `legacy_S2_checks.json` is historical. Current `S2_checks.json` binds the new
  caption and unchanged graphics only, not a new caption layout review.
- `src`, `scripts`, `tests` and historical `extensions`: unchanged scientific
  code. Its 35-table outputs remain historical, not current publication tables.
- `code/condense_tables.py`: portable aggregate-only presentation transform.
- `outputs/final_workbook.xlsx`: independently prepared current workbook;
  staging never copies, edits or replaces it.

No raw source data, identifiable case records, restricted derivatives, model
objects, private review images, manuscripts, private notebooks, runtimes or
credentials are redistributed. Only aggregate publication figures are included.

Read `docs/table_numbering.md` for current versus legacy workbook names.
`docs/staging_inventory.json` records source and staged hashes. Actual readiness
is in `outputs/qc/condensed_staging_report.json`, not inferred from this list.

## Freeze Boundary

Staging is not freezing. Final manifests and checksums are absent until an
explicitly approved freeze. A passing full boundary scan and stable, approved
workbook, condensed JSON and S2 caption hashes are required. Freeze remains
LOCAL NOT PUBLISHED and does not authorize publication.
The final workbook QA receipt must bind the current presentation JSON, and all
current worksheet notes are independently checked. Old-note workbooks and
stale assembly receipts are rejected even if the worksheet count is correct.
''',
        'docs/table_numbering.md': f'''# Current and Historical Table Numbering

The current presentation contains {counts}.
`outputs/tables/Supp_Table_S01` through `Supp_Table_S16` use current
supplementary numbering, with `_panel_N.csv` for multi-panel tables. There are
26 panel CSVs, not 26 publication tables. Their exact titles, notes and cells are in
`outputs/tables/condensed_tables.json`. Do not interpret a historical S number
in `outputs/aggregate/qc` or old statistical code as a current citation.

## Workbook Navigation

The workbook retains all 55 original worksheets. Its 35 historical
supplementary worksheets are renamed `Legacy_S01` through `Legacy_S35`.
Sixteen new current supplementary worksheets and one mapping worksheet are
placed before the historical supplementary block, for 72 worksheets in total.
Main Tables 1-4 may precede the current supplementary block and retain all numerical
data; the Main Table 2 note points to `Legacy_S26` in the workbook.

`Legacy_S05`, `Legacy_S26` and `Legacy_S34` are WORKBOOK-ONLY detailed tables in
the current presentation, not separate current SDC publication tables. Open
`outputs/final_workbook.xlsx` and select those exact worksheet names.
Their historical CSVs and full original cells are also retained for provenance.
No historical data are deleted to achieve the 16-table presentation.

The exact correspondence and any component-level decisions are recorded in
`presentation_inputs/table_map.json`; this document does not replace that map.

## Numbering Map

{numbering_map}

## Current Titles

{titles or 'The authoritative 16-table JSON has not yet been supplied.'}

## Input Readiness

{pending}

LOCAL NOT PUBLISHED; independent physician clinical adjudication not performed. No additional statistical rerun during presentation processing.
''',
        'docs/reproducibility.md': f'''# Presentation Reproduction

Version {VERSION} is LOCAL NOT PUBLISHED, with independent physician clinical adjudication not performed.
The current presentation has {counts}.
This revision changes titles, notes, grouping, numbering and navigation only.
It does not rerun statistical calculations or refit models.

## Aggregate-Only Presentation Inputs

The complete 35-table source is `presentation_inputs/source_tables.json`.
The current target is `outputs/tables/condensed_tables.json` and the mapping is
`presentation_inputs/table_map.json`. Titles and notes come from these supplied
inputs, not from historical statistical table names. Retained original main
table CSVs are byte-identical. A workbook-only Main Table 2 pointer changes no
numerical cell. The current S2 caption is separate from the unchanged historical
caption in `outputs/aggregate/qc`.

Portable presentation code is `code/condense_tables.py`; inspect its help for
supported arguments. Archive copying is `code/build_condensed_archive.py`.
For aggregate-only reconstruction outside the frozen archive directory:

    python code/condense_tables.py --source presentation_inputs/source_tables.json --output ../presentation_rebuild

Neither archive staging nor hash verification establishes clinical validity.
No clean independent raw-source-to-release execution is claimed.

## Historical Scientific Reproduction

All inherited scientific code and aggregate outputs are retained unchanged.
Historical source dependencies, statistical commands and QC policy assumptions
are documented in `docs/legacy/docs/reproducibility.md` and `docs/qc_protocol.md`.
Those commands reproduce the prior scientific outputs with 35-table numbering,
not the current 16-table presentation. Do not invoke the historical archive
builder to update this presentation. Raw data and individual derivatives must
be obtained and governed separately; they are not redistributed here.

Python numerical requirements remain `requirements.txt` and
`requirements-qc.txt`. The boundary checker additionally uses PyYAML and pypdf.
No installed runtime or Node dependency directory is bundled.

## Scan and Approved Freeze

`--scan` performs one complete archive boundary scan and records fingerprints
of scanned payloads. A later freeze rechecks the fingerprints; it does not
needlessly rescan unchanged payloads. After explicit approval, `--freeze`
requires `--expected-workbook-sha256`, `--expected-condensed-sha256`, and
`--expected-s2-caption-sha256`. Changing any approved input prevents freezing.
The local final workbook QA receipt must match the current JSON and workbook
hashes, verified original-data preservation, and exact current worksheet notes.
Its aggregate-only check is `outputs/qc/workbook_presentation_checks.json`.

One current output manifest excludes all manifests. The release manifest
includes that output manifest but excludes itself and `checksums.sha256`.
The checksum list includes both manifests and excludes itself. These form a
noncircular sequence, generated only after approval. Old freeze files and `.skip`
files are not reused. `--verify` checks the inventories and hashes without upload.
''',
        'docs/release_notes.md': f'''# Local Condensed Candidate Notes

{VERSION}: LOCAL NOT PUBLISHED; independent physician clinical adjudication not performed.
Current presentation: {counts}. No statistical rerun or refit.
All historical scientific results and code are retained. Supplementary tables
are regrouped and renumbered from 35 to 16. Workbook-only `Legacy_S05`,
`Legacy_S26` and `Legacy_S34` remain available, with all 55 original tabs retained.
Original main Table 1-4 data and aggregate figures are unchanged; Main Table 2
has a workbook-navigation note only. Figure S2 has a separate current caption.

Baseline references only: {baseline}.
No new DOI, GitHub release, external upload or clinical signoff is implied.
''',
        'outputs/legacy_tables/README.md': '''# Historical Publication CSVs

These are unchanged historical publication CSVs, not current table numbering.
Current supplementary Tables S1-S16 are under `outputs/tables`.
All original main and supplementary CSVs are retained with original filenames.
Scientific outputs and their original S1-S35 titles remain historical.
''',
    }
    for name, text in docs.items():
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    (archive / 'Makefile').write_text(
        '.PHONY: verify\nverify:\n\tpython code/build_condensed_archive.py --verify\n', encoding='utf-8')


def update_metadata(source, archive):
    for name in ['codemeta.json', '.zenodo.json']:
        if not (source / name).exists():
            continue
        value = read_json(source / name)
        for key in ['identifier', 'datePublished', 'publication_date', 'doi', 'prereserve_doi', 'codeRepository']:
            value.pop(key, None)
        value.update({'version': VERSION.removeprefix('v') if name == 'codemeta.json' else VERSION,
                      'description': DESCRIPTION})
        if name == 'codemeta.json':
            value.update({'isBasedOn': BASELINE_DOI, 'developmentStatus': 'concept'})
        else:
            value.update({'notes': 'LOCAL NOT PUBLISHED. No candidate DOI assigned. Clinical adjudication pending. Baseline references only.',
                          'related_identifiers': [
                              {'identifier': BASELINE_DOI, 'relation': 'isDerivedFrom', 'scheme': 'url'},
                              {'identifier': BASELINE_GITHUB, 'relation': 'isDerivedFrom', 'scheme': 'url'}]})
        json_write(archive / name, value)
    if (source / 'CITATION.cff').exists():
        import yaml
        value = yaml.safe_load((source / 'CITATION.cff').read_text())
        for key in ['doi', 'date-released', 'repository-code', 'url']:
            value.pop(key, None)
        value.update({'version': VERSION.removeprefix('v'),
                      'message': DESCRIPTION + ' The references identify the baseline only.',
                      'references': [{'type': 'software', 'title': 'Baseline reproducibility release v1.4.0 only',
                                      'version': '1.4.0', 'doi': '10.5281/zenodo.23153577', 'url': BASELINE_GITHUB}]})
        (archive / 'CITATION.cff').write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding='utf-8')


def workbook_names(path):
    with zipfile.ZipFile(path) as package:
        root = ET.fromstring(package.read('xl/workbook.xml'))
    return [node.attrib['name'] for node in root.iter() if node.tag.rsplit('}', 1)[-1] == 'sheet']


def workbook_current_notes(path, tables):
    """Read aggregate worksheet strings only; do not mutate the Office package."""
    namespace = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    relation_key = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    with zipfile.ZipFile(path) as package:
        relations = ET.fromstring(package.read('xl/_rels/workbook.xml.rels'))
        targets = {item.attrib['Id']: item.attrib['Target'] for item in relations}
        workbook = ET.fromstring(package.read('xl/workbook.xml'))
        sheets = {item.attrib['name']: targets[item.attrib[relation_key]]
                  for item in workbook.findall('s:sheets/s:sheet', namespace)}
        shared = []
        if 'xl/sharedStrings.xml' in package.namelist():
            strings = ET.fromstring(package.read('xl/sharedStrings.xml'))
            shared = [''.join(item.text or '' for item in node.findall('.//s:t', namespace))
                      for node in strings.findall('s:si', namespace)]
        results = []
        for table in tables:
            name = f'Supp_Table_S{table_number(table):02}'
            target = sheets[name]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            root = ET.fromstring(package.read(target))
            strings = []
            for cell in root.findall('.//s:c', namespace):
                kind = cell.get('t')
                value = cell.find('s:v', namespace)
                if kind == 's' and value is not None:
                    strings.append(shared[int(value.text)])
                elif kind == 'inlineStr':
                    strings.append(''.join(item.text or '' for item in cell.findall('s:is//s:t', namespace)))
                elif kind == 'str' and value is not None:
                    strings.append(value.text or '')
            expected = list(table.get('notes', []))
            for panel in table_panels(table):
                expected.extend(panel.get('notes', []))
            actual = Counter(strings)
            missing = sum(max(0, count - actual[text]) for text, count in Counter(expected).items())
            results.append({'sheet': name, 'expected_note_paragraphs': len(expected),
                            'missing_note_paragraphs': missing, 'title_present': table['title'] in actual})
    return results


def workbook_receipt_checks(output, archive, source):
    paths = {key: output / 'qa/workbook' / filename for key, filename in [
        ('receipt', 'final_manifest.json'), ('verification', 'verification.json'),
        ('readback', 'independent_readback.json')]}
    missing = [key for key, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        return {'version': VERSION, 'status': 'PENDING', 'blockers': ['workbook_final_QA_receipt_missing'],
                'missing_evidence': missing, 'statistical_rerun': False}
    receipt, verification, readback = (read_json(paths[key]) for key in ['receipt', 'verification', 'readback'])
    blockers = []
    current = {'condensed_tables': output / '工作记录/condensed_tables.json',
               'table_map': output / '工作记录/table_map.json'}
    if not all(path.is_file() and not path.is_symlink() for path in current.values()):
        return {'version': VERSION, 'status': 'PENDING', 'blockers': ['current_presentation_inputs_missing'],
                'statistical_rerun': False}
    hashes = {key: sha(path) for key, path in current.items()}
    if receipt.get('schemaHashes') != hashes:
        blockers.append('workbook_receipt_current_schema_mismatch')
    staged = {'condensed_tables': archive / APPROVAL_PATHS['condensed'],
              'table_map': archive / 'presentation_inputs/table_map.json'}
    if not all(path.is_file() and sha(path) == hashes[key] for key, path in staged.items()):
        blockers.append('current_inputs_not_freshly_staged')
    workbook = archive / APPROVAL_PATHS['workbook']
    workbook_hash = sha(workbook) if workbook.is_file() else None
    if not workbook_hash or receipt.get('workbookSHA256') != workbook_hash or readback.get('sha256') != workbook_hash:
        blockers.append('workbook_final_SHA_mismatch')
    if (not str(receipt.get('status', '')).startswith('READY_') or verification.get('status') != 'PASS'
            or readback.get('status') != 'PASS' or readback.get('latestSchemaHashMatches') is not True):
        blockers.append('workbook_final_QA_not_ready')
    for key, field in [('verification', 'verificationSHA256'), ('readback', 'independentReadbackSHA256')]:
        if receipt.get(field) != sha(paths[key]):
            blockers.append('workbook_QA_evidence_SHA_mismatch')
    builder = output / 'code/build_condensed_workbook.mjs'
    if not builder.is_file() or receipt.get('builderSHA256') != sha(builder):
        blockers.append('workbook_assembly_code_SHA_mismatch')
    source_hash = sha(source / 'outputs/final_workbook.xlsx')
    if receipt.get('sourceWorkbookSHA256') != source_hash:
        blockers.append('workbook_frozen_source_SHA_mismatch')
    old_checks = verification.get('oldSheets', [])
    if len(old_checks) != 55 or not all(item.get('allNumericValuesAndTypesUnchanged') is True
                                      and item.get('allUnapprovedCellsUnchanged') is True for item in old_checks):
        blockers.append('workbook_historical_values_not_verified')
    tables = current_tables(read_json(current['condensed_tables']))
    expected_counts = {'totalSheets': 72, 'currentTables': 16,
                       'panels': sum(len(table_panels(table)) for table in tables),
                       'dataRows': sum(len(panel['values']) - 1 for table in tables for panel in table_panels(table))}
    if any(receipt.get(key) != value for key, value in expected_counts.items()):
        blockers.append('workbook_current_table_counts_mismatch')
    notes = []
    if not blockers:
        try:
            notes = workbook_current_notes(workbook, tables)
            if any(item['missing_note_paragraphs'] or not item['title_present'] for item in notes):
                blockers.append('workbook_current_notes_mismatch')
        except (OSError, KeyError, IndexError, ValueError, AttributeError, zipfile.BadZipFile, ET.ParseError):
            blockers.append('workbook_current_notes_unreadable')
    return {'version': VERSION, 'status': 'PASS' if not blockers else 'PENDING', 'blockers': sorted(set(blockers)),
            'scope': 'Current aggregate presentation schema, final workbook receipt and exact current worksheet notes',
            'schema_sha256': hashes, 'workbook_sha256': workbook_hash, 'source_workbook_sha256': source_hash,
            'evidence_sha256': {key: sha(path) for key, path in paths.items()},
            'counts': expected_counts, 'current_notes': notes, 'old_worksheets_verified': len(old_checks),
            'statistical_rerun': False, 'clinical_adjudication': 'NOT_PERFORMED'}


def workbook_layout(path, required_original_names=None):
    try:
        names = workbook_names(path)
        current = [f'Supp_Table_S{n:02}' for n in range(1, 17)]
        legacy = [f'Legacy_S{n:02}' for n in range(1, 36)]
        missing_original = sorted(set(required_original_names or []) - set(names))
        positions = [names.index(name) for name in current if name in names]
        ordered = len(positions) == 16 and positions == list(range(positions[0], positions[0] + 16))
        mains = [f'Main_Table_{n}' for n in range(1, 5)]
        leading = ordered and (positions[0] == 0 or (positions[0] == 4 and names[:4] == mains))
        mapping_position = positions[-1] + 1 if ordered else -1
        mapping_present = 0 <= mapping_position < len(names) and (
            'map' in names[mapping_position].lower() or names[mapping_position] == 'TableIndex')
        legacy_present = all(name in names for name in legacy)
        current_before_legacy = ordered and legacy_present and mapping_position < min(names.index(name) for name in legacy)
        valid = (len(names) == 72 and len(set(names)) == 72 and leading and mapping_present
                 and legacy_present and current_before_legacy and not missing_original)
        return {'status': 'PASS' if valid else 'PENDING', 'worksheet_count': len(names),
                'current_supplementary_block_ordered': ordered, 'legacy_35_present': legacy_present,
                'current_block_start': positions[0] if ordered else None,
                'current_before_legacy': current_before_legacy,
                'mapping_after_current': mapping_present,
                'missing_original_worksheets': missing_original,
                'expected_worksheet_count': 72}
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return {'status': 'PENDING', 'reason': 'Workbook missing or not a readable final Office package'}


def readiness(archive, missing, required_original_names=None):
    blockers = list(missing)
    layout = workbook_layout(archive / APPROVAL_PATHS['workbook'], required_original_names)
    if layout['status'] != 'PASS':
        blockers.append('final_workbook_72_sheet_layout')
    if not (archive / APPROVAL_PATHS['caption']).is_file():
        blockers.append('current_S2_caption')
    return blockers, layout


def stage_archive(source, output, archive, expected_count=None, caption=None):
    source, output, archive = Path(source), Path(output), Path(archive)
    for path in [source, archive]:
        reject_symlinks(path)
    source, output, archive = source.resolve(), output.resolve(), archive.resolve()
    if (archive == output or not archive.is_relative_to(output) or archive.is_relative_to(source)
            or source.is_relative_to(archive) or source == output):
        raise ValueError('Archive must be a separate child of the presentation output root')
    before = snapshot_tree(source)
    if expected_count is not None and len(before) != expected_count:
        raise ValueError('Unexpected frozen source file count')
    previous = archive / 'docs/staging_inventory.json'
    if previous.exists() and read_json(previous).get('frozen_source_files') != before:
        raise ValueError('Frozen source changed since first staging')
    work = output / '工作记录'
    sources = work / 'source_tables.json'
    if not sources.is_file() or sources.is_symlink():
        raise ValueError('Full 35-table aggregate presentation input missing or symlink')
    full = read_json(sources)
    if len(full) != 35 or sorted(table['old_number'] for table in full) != list(range(1, 36)):
        raise ValueError('Full presentation input must contain old tables 1 through 35')
    condensed, mapping, transform = (work / 'condensed_tables.json', work / 'table_map.json',
                                     output / 'code/condense_tables.py')
    missing = [label for label, path in [('condensed_tables.json', condensed),
                                        ('table_map.json', mapping), ('condense_tables.py', transform)] if not path.is_file()]
    tables = current_tables(read_json(condensed)) if condensed.is_file() else []
    archive.mkdir(parents=True, exist_ok=True)
    if any((archive / name).exists() for name in FINAL_FILES):
        raise ValueError('Existing final freeze files must not be silently invalidated')
    inventory = []

    def copy(path, name, label):
        if name == 'outputs/final_workbook.xlsx' or path.is_symlink():
            raise ValueError('Refusing workbook copy or symlink')
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        before_hash = sha(path)
        shutil.copyfile(path, destination)
        if sha(path) != before_hash or sha(destination) != before_hash:
            raise ValueError('Copy changed while staging: ' + name)
        inventory.append({'path': name, 'source': label, 'source_sha256': before_hash,
                          'staged_sha256': before_hash, 'transformation': 'none'})

    for entry in before:
        target = copied_relative(entry['path'])
        if target is not None:
            copy(source / entry['path'], target, 'frozen_source/' + entry['path'])
    for n in range(1, 5):
        name = f'outputs/tables/Main_Table_{n}.csv'
        copy(source / name, name, 'frozen_source/' + name)
    copy(sources, 'presentation_inputs/source_tables.json', 'presentation/source_tables.json')
    if mapping.is_file():
        read_json(mapping)
        copy(mapping, 'presentation_inputs/table_map.json', 'presentation/table_map.json')
    if condensed.is_file():
        copy(condensed, APPROVAL_PATHS['condensed'], 'presentation/condensed_tables.json')
        expected_csvs = set()
        for table in tables:
            panels = table_panels(table)
            for index, panel in enumerate(panels, 1):
                suffix = f'_panel_{index}' if len(panels) > 1 else ''
                path = archive / f'outputs/tables/Supp_Table_S{table_number(table):02}{suffix}.csv'
                expected_csvs.add(path.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                supplied = work / path.name
                if supplied.is_file():
                    with supplied.open(encoding='utf-8', newline='') as stream:
                        if list(csv.reader(stream)) != panel['values']:
                            raise ValueError('CSV differs from authoritative JSON: ' + path.name)
                    copy(supplied, path.relative_to(archive).as_posix(), 'presentation/' + path.name)
                else:
                    path.write_text(csv_text(panel['values']), encoding='utf-8')
        for path in (archive / 'outputs/tables').glob('Supp_Table_S*.csv'):
            if path.name not in expected_csvs:
                path.unlink()
    if transform.is_file():
        compile(transform.read_text(), '<presentation-transform>', 'exec')
        copy(transform, 'code/condense_tables.py', 'presentation/code/condense_tables.py')
    builder = Path(__file__).resolve()
    if builder != archive / 'code/build_condensed_archive.py':
        copy(builder, 'code/build_condensed_archive.py', 'presentation/code/build_condensed_archive.py')
    caption = Path(caption) if caption else work / 'qc_figure_caption.txt'
    if caption.is_file():
        copy(caption, APPROVAL_PATHS['caption'], 'presentation/qc_figure_caption.txt')
        figure_hashes = {}
        for suffix in ['png', 'pdf', 'svg']:
            relative = 'outputs/figures/Supplementary_Figure_S2_Startup_QC.' + suffix
            if not (archive / relative).is_file() or sha(archive / relative) != sha(source / relative):
                raise ValueError('Figure S2 graphic changed: ' + suffix)
            figure_hashes[suffix] = sha(archive / relative)
        caption_text = caption.read_text(encoding='utf-8')
        if not re.search(r'\bS8\b', caption_text) or not re.search(r'\bS14\b', caption_text) or not re.search(r'\bS16\b', caption_text):
            raise ValueError('Figure S2 current caption must reference S8 and S14-S16')
        if re.search(r'\bS(?:3[0-5])\b', caption_text):
            raise ValueError('Figure S2 caption has historical publication numbering')
        json_write(archive / 'outputs/qc/S2_checks.json', {
            'version': VERSION, 'status': 'PASS',
            'scope': 'Current caption references and file hashes; unchanged aggregate graphics only',
            'caption_path': APPROVAL_PATHS['caption'], 'caption_sha256': sha(caption),
            'caption_words': len(caption_text.split()), 'word_count_method': 'Whitespace-delimited tokens',
            'current_supplementary_tables': 16,
            'figure_sha256': figure_hashes, 'graphics_unchanged': True,
            'source_csv_sha256': sha(archive / 'outputs/aggregate/qc/qc_figure_source.csv'),
            'statistical_rerun': False, 'clinical_adjudication': 'NOT_PERFORMED',
            'current_caption_content_review': 'PENDING', 'current_caption_visual_review': 'NOT_PERFORMED',
            'historical_graphic_QA': 'outputs/qc/legacy_S2_checks.json',
            'historical_caption_and_document_render_evidence': 'NOT_CURRENT'})
    elif (archive / APPROVAL_PATHS['caption']).exists():
        raise ValueError('Current caption input disappeared; refusing stale staged caption')
    write_documentation(archive, tables, missing)
    update_metadata(source, archive)
    if snapshot_tree(source) != before:
        raise ValueError('Frozen source changed during staging')
    json_write(previous, {'version': VERSION, 'not_a_final_release_manifest': True,
                         'frozen_source_files': before, 'source_file_count': len(before),
                         'source_hashes_unchanged': True, 'copied_files': inventory,
                         'workbook_write': 'NEVER_PERFORMED', 'statistical_rerun': False})
    required_original_names = []
    try:
        original_names = workbook_names(source / 'outputs/final_workbook.xlsx')
        required_original_names = [re.sub(r'^Supp_Table_S(\d+)$', lambda m: f'Legacy_S{int(m[1]):02}', name)
                                   for name in original_names]
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        if expected_count is not None:
            raise ValueError('Frozen source workbook worksheet inventory is unreadable')
    blockers, layout = readiness(archive, missing, required_original_names)
    workbook_checks = workbook_receipt_checks(output, archive, source)
    blockers.extend(workbook_checks['blockers'])
    json_write(archive / 'outputs/qc/workbook_presentation_checks.json', workbook_checks)
    report = {'version': VERSION, 'state': 'STAGED_NOT_FROZEN', 'publication': 'LOCAL_NOT_PUBLISHED',
              'clinical_adjudication': 'NOT_PERFORMED', 'source_file_count': len(before),
              'source_hashes_unchanged': True, 'copied_files': len(inventory),
              'current_main_tables': 4, 'current_supplementary_tables': len(tables),
              'current_table_panels': sum(len(table_panels(table)) for table in tables),
              'supplementary_figures': 2, 'historical_presentation_input_tables': 35,
              'statistical_rerun': False, 'workbook_write': 'NEVER_PERFORMED',
              'workbook_layout': layout, 'ready_for_freeze': False,
              'required_original_worksheets': required_original_names,
              'freeze_input_blockers': blockers, 'full_boundary_scan': 'NOT_YET_RUN',
              'approved_input_sha256': {key: sha(archive / path) for key, path in APPROVAL_PATHS.items()
                                      if (archive / path).is_file()},
              'final_sha256': None, 'final_sha256_reason': 'No explicit freeze; final checksums not generated'}
    json_write(archive / 'ARCHIVE_STATUS.json', {
        'version': VERSION, 'state': report['state'], 'publication': report['publication'],
        'clinical_adjudication': 'NOT_PERFORMED', 'freeze_requires_explicit_approval': True,
        'baseline_reference_only': {'github': BASELINE_GITHUB, 'zenodo': BASELINE_DOI},
        'current_supplementary_tables': 16, 'supplementary_figures': 2, 'statistical_rerun': False})
    json_write(archive / 'outputs/qc/condensed_staging_report.json', report)
    return report


def payload_fingerprints(archive):
    return [entry for entry in snapshot_tree(archive) if entry['path'] not in CONTROL_FILES]


def scan_once(archive, helper, report):
    scan = helper.scan_archive(archive)
    scan.update({'version': VERSION, 'publication': 'LOCAL_NOT_PUBLISHED',
                 'scanned_payload_fingerprints': payload_fingerprints(archive),
                 'statistical_rerun': False})
    json_write(archive / 'outputs/qc/archive_scan.json', scan)
    report.update({'full_boundary_scan': scan['status'],
                   'ready_for_freeze': scan['status'] == 'PASS' and not report['freeze_input_blockers'],
                   'ready_is_not_freeze_authorization': True})
    json_write(archive / 'outputs/qc/condensed_staging_report.json', report)
    for name in ['outputs/qc/archive_scan.json', 'outputs/qc/condensed_staging_report.json']:
        if helper.inspect_file(archive / name, Path(name)):
            raise ValueError('Generated scan metadata failed boundary check')
    if scan['status'] != 'PASS':
        raise ValueError('Boundary scan failed: ' + json.dumps(scan['failures']))
    return report


def freeze_archive(archive, approved, helper):
    if set(approved) != set(APPROVAL_PATHS) or any(not re.fullmatch(r'[a-f0-9]{64}', str(v)) for v in approved.values()):
        raise ValueError('Freeze requires three explicitly approved stable SHA-256 fingerprints')
    report = read_json(archive / 'outputs/qc/condensed_staging_report.json')
    workbook_checks = read_json(archive / 'outputs/qc/workbook_presentation_checks.json')
    if (workbook_checks.get('status') != 'PASS' or workbook_checks.get('workbook_sha256') != approved['workbook']
            or workbook_checks.get('schema_sha256', {}).get('condensed_tables') != approved['condensed']):
        raise ValueError('Final workbook QA must match the approved current notes schema')
    for key, name in APPROVAL_PATHS.items():
        if not (archive / name).is_file() or sha(archive / name) != approved[key]:
            raise ValueError('Approved input changed: ' + key)
    figure_qa = read_json(archive / 'outputs/qc/S2_checks.json')
    if figure_qa.get('caption_sha256') != approved['caption'] or figure_qa.get('status') != 'PASS':
        raise ValueError('Current Figure S2 caption QA is missing or stale')
    for suffix, digest in figure_qa.get('figure_sha256', {}).items():
        if sha(archive / ('outputs/figures/Supplementary_Figure_S2_Startup_QC.' + suffix)) != digest:
            raise ValueError('Figure S2 graphic QA hash changed')
    blockers, layout = readiness(archive, report['freeze_input_blockers'], report.get('required_original_worksheets'))
    if blockers or layout['status'] != 'PASS':
        raise ValueError('Freeze inputs are not final: ' + ', '.join(blockers))
    scan = read_json(archive / 'outputs/qc/archive_scan.json')
    if scan['status'] != 'PASS' or scan.get('scanned_payload_fingerprints') != payload_fingerprints(archive):
        raise ValueError('Passing boundary scan is missing or payload changed after scan')
    status = read_json(archive / 'ARCHIVE_STATUS.json')
    status.update({'state': 'FROZEN_LOCAL_CANDIDATE', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                   'approved_input_sha256': approved})
    json_write(archive / 'ARCHIVE_STATUS.json', status)
    report.update({'state': 'FROZEN_LOCAL_CANDIDATE', 'ready_for_freeze': True,
                   'explicit_freeze': True, 'final_sha256_reason': 'Checksum-list digest reported by freeze verification'})
    json_write(archive / 'outputs/qc/condensed_staging_report.json', report)
    for name in ['ARCHIVE_STATUS.json', 'outputs/qc/condensed_staging_report.json']:
        if helper.inspect_file(archive / name, Path(name)):
            raise ValueError('Final status metadata failed boundary check')
    helper.write_manifests(archive)
    result = helper.verify_manifests(archive)
    return {**result, 'state': 'FROZEN_LOCAL_CANDIDATE', 'final_sha256': sha(archive / 'checksums.sha256'),
            'approved_input_sha256': approved}


def main():
    default = Path(__file__).resolve().parents[1]
    archived = default.name == ARCHIVE_NAME
    output = default.parent if archived else default
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=output.parent / SOURCE_RELATIVE)
    ap.add_argument('--output-root', type=Path, default=output)
    ap.add_argument('--archive-root', type=Path)
    ap.add_argument('--s2-caption', type=Path)
    ap.add_argument('--expected-source-count', type=int, default=311)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--freeze', action='store_true')
    modes.add_argument('--verify', action='store_true')
    ap.add_argument('--scan', action='store_true')
    for key in ['workbook', 'condensed', 's2-caption']:
        ap.add_argument('--expected-' + key + '-sha256')
    args = ap.parse_args()
    out = args.output_root.expanduser()
    archive = args.archive_root.expanduser() if args.archive_root else out / ARCHIVE_NAME
    reject_symlinks(archive)
    archive = archive.resolve()
    if args.verify:
        helper = load_legacy_helper(archive / HELPER)
        result = helper.verify_manifests(archive)
        print(json.dumps({**result, 'final_sha256': sha(archive / 'checksums.sha256')}, indent=2))
        return
    if args.freeze:
        if args.scan:
            ap.error('Run the complete scan separately before explicit freeze approval')
        helper = load_legacy_helper(archive / HELPER)
        live_checks = workbook_receipt_checks(out.resolve(), archive, args.source.expanduser().resolve())
        if live_checks['status'] != 'PASS':
            raise ValueError('Current workbook readiness changed: ' + ', '.join(live_checks['blockers']))
        caption = args.s2_caption or out / '工作记录/qc_figure_caption.txt'
        if sha(caption) != args.expected_s2_caption_sha256:
            raise ValueError('Current approved Figure S2 caption changed')
        result = freeze_archive(archive, {'workbook': args.expected_workbook_sha256,
                                          'condensed': args.expected_condensed_sha256,
                                          'caption': args.expected_s2_caption_sha256}, helper)
    else:
        result = stage_archive(args.source.expanduser(), out, archive,
                               expected_count=args.expected_source_count, caption=args.s2_caption)
        if args.scan:
            result = scan_once(archive, load_legacy_helper(archive / HELPER), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
