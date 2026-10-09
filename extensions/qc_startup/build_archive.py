"""Stage a local aggregate-only archive. Freeze only with explicit --freeze.

No network operations, uploads, ZIP creation, or workbook writes are performed.
Source code is never edited: portability changes affect archive copies only.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import yaml

sys.dont_write_bytecode = True
VERSION = 'v1.5.0-algorithmic-qc'
BASELINE_DOI = 'https://doi.org/10.5281/zenodo.23153577'
BASELINE_GITHUB = 'https://github.com/tqytqytqytqy/ioh-bidirectional-misclassification/releases/tag/v1.4.0'
BASELINE_REL = 'EJA_投稿文件包_20261006_最终文字修订/03_可复现材料'
STARTUP_REL = '启动段质量审计_20261007'
OLD_DEPS = '/' + 'tmp/ioh_startup_deps_20261006'
PRIVATE_TEXT = re.compile(r'(?:\x2fUsers\x2f|\x2fVolumes\x2f|\x2fhome\x2f|file:\x2f\x2f|(?<![A-Za-z])[A-Z]:\\)', re.I)
IDENTIFIER = re.compile(r'^(?:case_?ids?|subject_?ids?|patient_?ids?|review_code|medical_record_number|mrn|'
                        r'changed_ids|excluded_ids|image_path|image_filename|panel_path)$', re.I)
SECRET = re.compile(r'(?:sk-[A-Za-z0-9]{24,}|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
TEXT_SUFFIXES = {'.py', '.R', '.r', '.mjs', '.js', '.json', '.md', '.txt', '.yaml', '.yml',
                 '.cff', '.ini', '.sh', '.csv', '.sha256', '.svg'}
CODE_SUFFIXES = {'.py', '.R', '.r', '.mjs', '.js', '.md', '.json', '.yaml', '.yml', '.ini', '.txt'}
FORBIDDEN_PARTS = {'data_restricted', 'private_case_audit', 'outcomes_runtime', 'outcomes_matplotlib',
                   'notebooks', 'intermediate', 'raw', 'tracks', '.git', '__pycache__', '.pytest_cache'}
FORBIDDEN_SUFFIXES = {'.docx', '.doc', '.parquet', '.feather', '.pkl', '.pickle', '.rds', '.rdata',
                      '.sqlite', '.db', '.npy', '.npz', '.ipynb', '.gz', '.zip', '.pyc'}
FREEZE_FILES = ['checksums.sha256', 'outputs/manifests/outputs_manifest.json',
                'outputs/manifests/release_manifest.json', 'outputs/manifests/provenance.json']
STARTUP_SCRIPTS = ['audit_core.py', 'run_numeric_audit.py', 'run_waveform_audit.py',
                   'scan_full_waveforms.py', 'prepare_review_list.py', 'run_startup_sensitivity.py',
                   'test_audit_core.py']
QC_SCRIPTS = {'build_archive.py', 'test_build_archive.py', 'build_qc_panels.py',
              'build_qc_publication_tables.py', 'build_qc_workbooks.mjs', 'rerun_events.py',
              'rerun_treatment.py', 'rerun_outcomes.py', 'test_qc_panels.py', 'test_qc_events.py',
              'test_qc_treatment.py', 'test_qc_outcomes.py', 'review_images.py'}
QC_SCRIPTS.add('make_qc_figure.py')
QA_NAMES = {'panels_verification.json', 'qc_publication_tables_verification.json',
            'events_final_validation.json', 'events_test_summary.json',
            'events_baseline_exposure_reconciliation.json', 'events_extension_reconciliation.json',
            'events_decomposition_reconciliation.json', 'outcomes_ascertainment_verification.json',
            'treatment_baseline_reproduction.json', 'treatment_test_results.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def private_json_keys(value, trail=''):
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            if IDENTIFIER.fullmatch(str(key)):
                issues.append('identifier_key:'+trail+str(key))
            issues.extend(private_json_keys(item, trail+str(key)+'.'))
    elif isinstance(value, list):
        for item in value:
            issues.extend(private_json_keys(item, trail))
    return issues


def scrub_qa(value):
    """Keep aggregate verdicts; remove identifiers and local source-path maps."""
    if isinstance(value, dict):
        clean = {}
        omitted = 0
        for key, item in value.items():
            if (IDENTIFIER.fullmatch(str(key)) or PRIVATE_TEXT.search(str(key))
                    or key in {'source_hashes', 'sha256', 'protected', 'source_fingerprints',
                               'input_hashes_before', 'input_hashes_after', 'source_paths'}):
                omitted += 1
                continue
            clean[key] = scrub_qa(item)
        if omitted:
            clean['private_provenance_fields_omitted'] = omitted
        return clean
    if isinstance(value, list):
        return [scrub_qa(item) for item in value]
    if isinstance(value, str) and PRIVATE_TEXT.search(value):
        return '[local path omitted]'
    return value


def inspect_file(path, relative):
    relative = Path(relative)
    issues = []
    if (set(relative.parts) & FORBIDDEN_PARTS or relative.name == '.env'
            or relative.name.startswith('.env.') or relative.suffix.lower() in FORBIDDEN_SUFFIXES
            or any(part.startswith('private_') for part in relative.parts)
            or 'model_frame' in relative.name or 'model_matrix' in relative.name):
        issues.append('forbidden_path')
    if path.is_symlink():
        return issues+['symlink']
    text = None
    if relative.suffix.lower() in {'.xlsx', '.pptx', '.docx'}:
        try:
            with zipfile.ZipFile(path) as archive:
                xml = [archive.read(name).decode('utf-8') for name in archive.namelist()
                       if name.endswith(('.xml', '.rels'))]
            text = '\n'.join(xml)
            # Shared strings, inline strings and cached cell strings are all checked.
            for part in xml:
                root = ET.fromstring(part)
                for node in root.iter():
                    if node.tag.rsplit('}', 1)[-1] == 't' and IDENTIFIER.fullmatch((node.text or '').strip()):
                        issues.append('identifier_cell')
        except (zipfile.BadZipFile, UnicodeDecodeError, ET.ParseError):
            issues.append('unreadable_office_package')
    elif relative.suffix.lower() == '.pdf':
        try:
            from pypdf import PdfReader
            document = PdfReader(path)
            text = '\n'.join(page.extract_text() or '' for page in document.pages)
            text += '\n'+str(document.metadata or {})
        except Exception:
            issues.append('unreadable_pdf')
    elif relative.suffix in TEXT_SUFFIXES or relative.name in {'LICENSE', 'Makefile', '.gitignore', '.zenodo.json'}:
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            issues.append('non_utf8_text')
    elif relative.suffix.lower() != '.png':
        issues.append('unrecognized_file_type')
    if text is not None:
        if PRIVATE_TEXT.search(text):
            issues.append('local_path')
        if SECRET.search(text):
            issues.append('secret_pattern')
        if re.search(r'\bQC\d{3}\b', text):
            issues.append('private_review_code')
        if relative.suffix == '.csv':
            header = next(csv.reader(io.StringIO(text)), [])
            for value in header:
                if IDENTIFIER.fullmatch(value.strip()):
                    issues.append('identifier_column:'+value)
        elif relative.suffix == '.json':
            try:
                issues.extend(private_json_keys(json.loads(text)))
            except json.JSONDecodeError:
                issues.append('invalid_json')
    return sorted(set(issues))


def scan_archive(archive):
    failures, count = {}, 0
    for path in sorted(archive.rglob('*')):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(archive)
        if relative.as_posix() == 'outputs/qc/archive_scan.json':
            continue
        count += 1
        issues = inspect_file(path, relative)
        if issues:
            failures[relative.as_posix()] = issues
    return {'status': 'PASS' if not failures else 'FAIL', 'scanned_files': count,
            'failures': failures, 'clinical_adjudication': 'NOT_PERFORMED',
            'scope': 'Content, structured identifier fields, Office XML, local paths, secrets and forbidden files',
            'limitations': 'Automated boundary scan; not a claim of clinical validity or complete privacy certification'}


def portable_python(text, startup=False, private_review=False):
    """Use the AST only for local-path adaptation; never execute copied code."""
    tree = ast.parse(text)
    import_os = not any(isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names) for n in tree.body)
    prefix = []
    if import_os:
        prefix = [ast.Import(names=[ast.alias(name='os')])]
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == 'sys.path.insert' and OLD_DEPS in ast.unparse(node):
                new_body.extend(ast.parse("if os.environ.get('IOH_QC_DEPENDENCIES'):\n"
                                          "    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])").body)
                continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = ast.unparse(node.value)
            if name == 'ROOT' and 'IOH_PROJECT_ROOT' not in value and 'PATH_CONFIG.source_project' not in value:
                if PRIVATE_TEXT.search(value):
                    node.value = ast.parse("Path(os.environ['IOH_PROJECT_ROOT']).expanduser().resolve()", mode='eval').body
                else:
                    node.value = ast.parse(f"Path(os.environ.get('IOH_PROJECT_ROOT', str({value}))).expanduser().resolve()", mode='eval').body
            elif name == 'OUT' and 'IOH_QC_OUTPUT_ROOT' not in value and 'PATH_CONFIG.output_root' not in value:
                var = 'IOH_STARTUP_AUDIT_ROOT' if startup else 'IOH_QC_OUTPUT_ROOT'
                fallback = "ROOT / '"+STARTUP_REL+"'" if startup else value
                node.value = ast.parse(f"Path(os.environ.get('{var}', str({fallback}))).expanduser().resolve()", mode='eval').body
            elif startup and name == 'RAW':
                node.value = ast.parse("Path(os.environ['IOH_VITALDB_RAW_ROOT']).expanduser().resolve()", mode='eval').body
            elif startup and name == 'TRACK_ROOTS':
                node.value = ast.parse("[Path(p).expanduser().resolve() for p in os.environ['IOH_VITALDB_TRACK_DIRS'].split(os.pathsep) if p]", mode='eval').body
            elif private_review and name == 'STATIC':
                node.value = ast.parse("json.loads(Path(os.environ['IOH_IMAGE_REVIEW_INPUT_JSON']).read_text())", mode='eval').body
        new_body.append(node)
    tree.body = new_body
    insert_at = 0
    while insert_at < len(tree.body) and (
        isinstance(tree.body[insert_at], ast.ImportFrom) and tree.body[insert_at].module == '__future__'
        or isinstance(tree.body[insert_at], ast.Expr) and isinstance(tree.body[insert_at].value, ast.Constant)
        and isinstance(tree.body[insert_at].value.value, str)):
        insert_at += 1
    tree.body[insert_at:insert_at] = prefix

    class Dependencies(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and OLD_DEPS in node.value:
                return ast.copy_location(ast.Constant('' if node.value == OLD_DEPS else
                    node.value.replace(OLD_DEPS, 'configured_dependency_directory')), node)
            if node.value == '/'+'Users'+'/':
                return ast.copy_location(ast.BinOp(left=ast.Constant('/'), op=ast.Add(),
                                                   right=ast.Constant('Users/')), node)
            return node
    tree = Dependencies().visit(tree)
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree)+'\n'
    compile(result, '<portable-copy>', 'exec')
    return result


def portable_javascript(text):
    lines = []
    for line in text.splitlines():
        if line.startswith('const OUT ='):
            line = "const OUT = path.resolve(process.env.IOH_QC_OUTPUT_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));"
        elif line.startswith('const ROOT ='):
            line = 'const ROOT = path.resolve(process.env.IOH_PROJECT_ROOT || path.dirname(OUT));'
        elif line.startswith('const BUNDLE =') and PRIVATE_TEXT.search(line):
            line = ("const BUNDLE = process.env.CODEX_NODE_MODULES;\n"
                    "if (!BUNDLE) throw new Error('Set CODEX_NODE_MODULES to your installed Node dependency directory');")
        synthetic = '/'+'Users'+'/example/private'
        line = line.replace(repr(synthetic), "('/' + 'Users' + '/example/private')")
        lines.append(line)
    return '\n'.join(lines)+'\n'


def candidate_codemeta(original):
    result = dict(original)
    for key in ['identifier', 'datePublished']:
        result.pop(key, None)
    result.update({'version': VERSION.removeprefix('v'), 'developmentStatus': 'concept',
                   'description': 'LOCAL NOT PUBLISHED. Rule-based startup-QC sensitivity candidate; independent physician clinical adjudication not performed. '
                                  'Original v1.4.0 analyses remain the manuscript primary analyses. Aggregate results and code only.',
                   'isBasedOn': BASELINE_DOI})
    return result


def clear_freeze_files(archive):
    for relative in FREEZE_FILES:
        path = archive/relative
        if path.is_symlink():
            raise ValueError('Refusing manifest symlink')
        if path.exists():
            path.unlink()


def file_entries(archive, paths):
    return [{'path': path.relative_to(archive).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            for path in sorted(paths)]


def write_manifests(archive):
    manifests = archive/'outputs/manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    output_paths = [p for p in (archive/'outputs').rglob('*') if p.is_file() and manifests not in p.parents]
    json_write(manifests/'outputs_manifest.json', {'version': VERSION, 'publication': 'LOCAL_NOT_PUBLISHED',
               'clinical_adjudication': 'NOT_PERFORMED', 'files': file_entries(archive, output_paths)})
    all_paths = [p for p in archive.rglob('*') if p.is_file() and p.relative_to(archive).as_posix() not in
                 {'outputs/manifests/release_manifest.json', 'checksums.sha256'}]
    json_write(manifests/'release_manifest.json', {'version': VERSION, 'publication': 'LOCAL_NOT_PUBLISHED',
               'clinical_adjudication': 'NOT_PERFORMED', 'files': file_entries(archive, all_paths)})
    all_paths = [p for p in archive.rglob('*') if p.is_file() and p.name != 'checksums.sha256']
    (archive/'checksums.sha256').write_text(''.join(
        f"{sha(p)}  {p.relative_to(archive).as_posix()}\n" for p in sorted(all_paths)), encoding='utf-8')


def verify_manifests(archive):
    for filename in ['outputs_manifest.json', 'release_manifest.json']:
        value = json.loads((archive/'outputs/manifests'/filename).read_text())
        listed = {entry['path'] for entry in value['files']}
        if filename == 'outputs_manifest.json':
            actual = {p.relative_to(archive).as_posix() for p in (archive/'outputs').rglob('*')
                      if p.is_file() and archive/'outputs/manifests' not in p.parents}
        else:
            actual = {p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file()
                      and p.relative_to(archive).as_posix() not in
                      {'outputs/manifests/release_manifest.json', 'checksums.sha256'}}
        if listed != actual:
            raise ValueError('Manifest file inventory changed: '+filename)
        for entry in value['files']:
            path = (archive/entry['path']).resolve()
            if not path.is_relative_to(archive.resolve()) or sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('Manifest hash mismatch: '+entry['path'])
    checksums = {}
    for line in (archive/'checksums.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        checksums[relative] = digest
        path = (archive/relative).resolve()
        if not path.is_relative_to(archive.resolve()) or sha(path) != digest:
            raise ValueError('Checksum mismatch: '+relative)
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file() and p.name != 'checksums.sha256'}
    if set(checksums) != actual:
        raise ValueError('Checksum inventory mismatch')
    return {'status': 'PASS', 'files': len(checksums), 'version': VERSION, 'publication': 'LOCAL_NOT_PUBLISHED'}


def require_freeze_inputs(archive):
    required = ['outputs/final_workbook.xlsx', 'outputs/aggregate/qc/qc_publication_tables.json',
                'outputs/aggregate/qc/events_summary.json', 'outputs/aggregate/qc/treatment_summary.json',
                'outputs/aggregate/qc/outcomes_summary.json', 'outputs/qc/S2_checks.json',
                'outputs/aggregate/qc/qc_figure_caption.txt', 'outputs/aggregate/qc/qc_figure_source.csv']
    required.extend('outputs/figures/Supplementary_Figure_S2_Startup_QC.'+suffix
                    for suffix in ['png', 'pdf', 'svg'])
    missing = [p for p in required if not (archive/p).is_file()]
    if missing:
        raise ValueError('Freeze inputs missing: '+', '.join(missing))
    tables = json.loads((archive/required[1]).read_text())
    for table in tables:
        if not (archive/'outputs/tables'/f"{table['name']}.csv").is_file():
            raise ValueError('Missing current publication table: '+table['name'])
    figure_qa = json.loads((archive/'outputs/qc/S2_checks.json').read_text())
    if figure_qa.get('status') != 'PASS':
        raise ValueError('Figure S2 QA is not final')
    expected = {
        'outputs/aggregate/qc/qc_figure_caption.txt': figure_qa.get('caption_sha256'),
        'outputs/aggregate/qc/qc_figure_source.csv': figure_qa.get('source_csv_sha256')}
    for suffix in ['png', 'pdf', 'svg']:
        expected['outputs/figures/Supplementary_Figure_S2_Startup_QC.'+suffix] = (
            figure_qa.get('figure_sha256', {}).get(suffix))
    for relative, digest in expected.items():
        if digest != sha(archive/relative):
            raise ValueError('Figure S2 QA hash mismatch: '+relative)
    words = len((archive/'outputs/aggregate/qc/qc_figure_caption.txt').read_text().split())
    if words > 210 or words != figure_qa.get('caption_words'):
        raise ValueError('Figure S2 caption word count is not verified or exceeds 210')


def readme():
    return f"""# Temporal observability of intraoperative hypotension

## {VERSION}: LOCAL NOT PUBLISHED

This is a local archive candidate, not a released or clinically adjudicated
version. The original analysis remains primary. The qc60, qc_all and startup10
analyses are sensitivity analyses. No patient benefit, clinical safety,
treatment effect, or completed clinician signoff is implied.

The source baseline is [GitHub v1.4.0]({BASELINE_GITHUB}) and
[Zenodo 23153577]({BASELINE_DOI}). These links identify the BASELINE ONLY.
There is no DOI or public release for this candidate.

## Contents

- src, scripts, tests, and the original extensions: frozen v1.4 public code.
- extensions/qc_startup: copies of the current QC and assembly scripts.
- extensions/startup_screening: numeric/waveform screening and private-review
  reconstruction scripts. No review images or review records are distributed.
- docs/qc_protocol.md: fixed rules; independent physician clinical adjudication not performed.
- outputs/tables: baseline tables plus current QC supplementary tables.
- outputs/aggregate (frozen baseline names) and outputs/aggregate/qc: non-identifiable results.
- outputs/qc: selected aggregate verification verdicts with local paths removed.
- outputs/final_workbook.xlsx: Consolidated manuscript tables and unrounded aggregate analyses.

No raw tracks, identifiers, case-level derivatives, laboratory records, model
matrices, fitted model objects, review images, manuscripts, private notebooks,
runtime packages, secrets, or local-path manifests are included.

## Reproduction boundary

Read docs/reproducibility.md and config/inputs.example.sh. The local
reruns consumed frozen private inputs; a fresh source-raw-to-release execution
has NOT been validated in a clean independent environment. Archive staging and
hash verification are NOT new statistical analyses or clinical validation.

The assembly script defaults to staging. It never writes the workbook and
never creates a ZIP or uploads. An explicit --freeze refreshes current aggregate
results and creates the two manifests and one checksum list. Freeze remains
LOCAL NOT PUBLISHED. See ARCHIVE_STATUS.json for actual staging/freeze state.

The output manifest excludes manifests. The release manifest includes the
output manifest but excludes itself and checksums.sha256. The checksum list
includes both manifests and excludes itself. No legacy final hashes are reused.
"""


def reproduction_docs(script_names):
    names = '\n'.join('- '+name for name in script_names)
    return f"""# Reproduction and Provenance

## Status and boundaries

Version {VERSION} is LOCAL NOT PUBLISHED; clinical adjudication is pending.
Original v1.4.0 results remain primary, with QC additions reported as sensitivity
analyses. No one-command raw reproduction has been validated for this archive.
The source inputs below are intentionally not redistributed.

## Required local source tree

IOH_PROJECT_ROOT points to the user's private source project, containing:

1. analysis_v10_subject_phase_release/outputs/intermediate:
   vitaldb_manifest.parquet, artmap_10s.parquet,
   frequency_decomp_case_offset.parquet, frequency_decomp_case_mean_v7.parquet,
   episode_observability_case_5min.parquet, aki_creatinine_case_v7.parquet,
   aki_invisibility_analysis_case_v7.parquet, and other dependencies named in the
   frozen baseline scripts. The original definitions and covariates are retained.
2. analysis_v10_subject_phase_release/src and outputs/tables, including the
   sequential/incremental AKI and ICU model specification/results CSV files and
   aki_labs_source_audit.csv used for baseline reconciliation.
3. {BASELINE_REL}: the frozen public package, not the candidate itself.
4. 治疗相关事件补充修订_20261004/code/run_analysis.py and treatment_core.py,
   data_restricted/raw_record_changes.csv and stable_record_changes.csv, the
   seven frozen event CSVs, recovery_events.csv, and aggregate comparison files.
   The QC treatment rerun reused 834 raw / 776 stable frozen pump transitions;
   it did not reread all raw pump tracks.
5. {STARTUP_REL}/private_case_audit:
   full_waveform_10s_metrics.csv, clinical_review_list_87_cases.csv and
   startup_numeric_case_audit.csv. These derive from the screening scripts.
6. Official VitalDB labs.csv supplied through IOH_VITALDB_LABS, independently
   obtained under applicable source-data terms. No laboratory rows are archived.

IOH_QC_OUTPUT_ROOT points to the private QC workspace containing the fixed
方案.md, data_restricted/panel_POLICY.parquet and cohort_POLICY.csv. The events
stage creates frequency_decomp_POLICY.parquet before the outcomes stage.
Policies are original, qc60, qc_all, startup10.

For screening reconstruction, IOH_STARTUP_AUDIT_ROOT selects a private audit
output directory. IOH_VITALDB_RAW_ROOT contains trks.csv;
IOH_VITALDB_TRACK_DIRS lists local track directories separated by os.pathsep.
The screening order is run_numeric_audit.py, run_waveform_audit.py,
scan_full_waveforms.py, prepare_review_list.py, then optional
run_startup_sensitivity.py. Generated case review material must remain private.
These reconstructed plots are not clinician adjudication or AI review evidence.
The review_images.py copy contains no embedded per-case observations. It requires
IOH_IMAGE_REVIEW_INPUT_JSON, a separately retained private JSON payload of the
previously viewed observations. It only packages those observations; it does
not perform image interpretation, reproduce the review, or certify clinicians.

## Current QC sequence

The archived copies accept local environment configuration before data imports:

    python extensions/qc_startup/build_qc_panels.py
    python extensions/qc_startup/rerun_events.py
    python extensions/qc_startup/rerun_treatment.py
    python extensions/qc_startup/rerun_outcomes.py
    python extensions/qc_startup/build_qc_publication_tables.py

Inspect each runner's help and input checks first. Do not run against incomplete
or unverified private inputs. The outcomes runner also invokes the frozen
scripts/34_verify_aki_primary_model.R and
scripts/40_verify_icu_resource_use_models_v74.R for policy-specific checks.
Public tables are derived outputs, not replacements for case-level inputs.

The workbook builder is a separate Node application, not run by archive staging. Its
public export consumes qc_publication_tables.json and baseline aggregate/table
sources. Its private clinical-review export may additionally require review
CSV files. Set CODEX_NODE_MODULES to a local installation containing
@oai/artifact-tool, jszip, xml-js and sharp. Access to that artifact runtime is
not guaranteed by this source archive. No workbook re-render is performed by
build_archive.py.

## Dependencies

Baseline Python versions are recorded in requirements.txt. QC uses Python 3.12,
NumPy, pandas, pyarrow, SciPy, statsmodels, scikit-learn, matplotlib, PyYAML and
pytest. R verification needs Rscript and the packages imported by the included
R scripts (including sandwich/lmtest as applicable). Optional private screening
document generation is outside this public package. IOH_QC_DEPENDENCIES may
point to a locally installed Python dependency directory; no runtime is bundled.
The aggregate Supplementary Figure S2 builder and archive PDF inspection also
use Pillow, pypdf and reportlab; see requirements-qc.txt. Its fixed aggregate
source CSV and caption accompany the PNG, SVG and vector PDF. Visual QA and
private-review proof images are not included.

## Code actually staged for this extension

{names}

The current baseline src/scripts/extensions/tests are copied from the supplied
v1.4.0 public package. New QC and screening copies receive path-only portability
adaptations; the private source files are unchanged. See docs/staging_inventory.json
for relative source labels, source hashes, staged hashes, and transformations.
Syntax validation does not establish end-to-end reproduction on a new machine.

## Verification and freeze

    PYTHONPATH=src python -m pytest -q -p no:cacheprovider
    python extensions/qc_startup/build_archive.py --verify

The second command requires an explicitly frozen archive. Before that, staging
is the only supported package state. Tests that require private frozen helpers
need the configured private project; artifact tests need the final workbook.
Frozen baseline-only logs must not be read as evidence for all QC policies.
The selected new QA verdicts retain the actual policy-specific status.

After all current aggregates and the consolidated workbook are verified,
run the local code/build_archive.py with explicit --freeze. This
refreshes staged inputs, validates boundaries, hashes the actual workbook,
creates outputs_manifest.json then release_manifest.json then checksums.sha256,
and verifies all inventories without circular hashing. It does not upload or
make a ZIP. Any subsequent stage run invalidates the prior freeze files.
"""


def stage_archive(project, out, archive):
    baseline = project/BASELINE_REL
    inventory = []
    if archive.is_symlink() or any(p.is_symlink() for p in archive.rglob('*')):
        raise ValueError('Archive contains symlink')
    previous_inventory = archive/'docs/staging_inventory.json'
    previous = json.loads(previous_inventory.read_text()).get('staged_code_and_data', []) if previous_inventory.exists() else []
    clear_freeze_files(archive)

    def copy(source, relative, transform=None, label=None):
        if source.is_symlink() or not source.is_file():
            raise ValueError('Required source unavailable or symlink: '+(label or relative))
        before = sha(source)
        destination = archive/relative
        if destination.is_symlink():
            raise ValueError('Refusing destination symlink: '+relative)
        if relative == 'outputs/final_workbook.xlsx':
            raise ValueError('Workbook is owned by another task and must not be copied')
        content = source.read_bytes()
        if transform:
            content = transform(content.decode('utf-8')).encode('utf-8')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        if sha(source) != before:
            raise ValueError('Source changed during staging; rerun when stable: '+relative)
        inventory.append({'path': relative, 'source': label or 'baseline/'+source.relative_to(baseline).as_posix(),
                          'source_sha256': before, 'staged_sha256': sha(destination),
                          'portability_transformed': bool(transform)})

    for folder in ['src', 'scripts', 'extensions', 'tests', 'config']:
        for source in sorted((baseline/folder).rglob('*')):
            if source.is_file() and source.suffix in CODE_SUFFIXES and not (set(source.parts) & FORBIDDEN_PARTS):
                copy(source, source.relative_to(baseline).as_posix())
    for name in ['LICENSE', 'requirements.txt', 'pytest.ini', '.gitignore', 'environment.txt']:
        copy(baseline/name, name)
    for name in ['analysis_conventions.md', 'data_access.md']:
        copy(baseline/'docs'/name, 'docs/'+name)
    for folder in ['tables', 'aggregate']:
        for source in sorted((baseline/'outputs'/folder).glob('*')):
            if source.is_file() and source.suffix in {'.csv', '.json'}:
                relative = 'outputs/tables/'+source.name if folder == 'tables' else 'outputs/aggregate/'+source.name
                copy(source, relative)
    # Only these known frozen public aggregate figures are allowed as image assets.
    for source in sorted((baseline/'outputs/figures').glob('*')):
        if source.name.startswith(('Figure_1_', 'Figure_2_', 'Figure_3_', 'Figure_4_', 'Supplemental_Figure_S1_')):
            copy(source, 'outputs/figures/'+source.name)
    script_names = []
    for source in sorted((out/'code').glob('*')):
        if source.name not in QC_SCRIPTS:
            continue
        relative = 'extensions/qc_startup/'+source.name
        transform = None
        if source.name not in {'build_archive.py', 'test_build_archive.py'}:
            transform = portable_python if source.suffix == '.py' else portable_javascript
        if source.name == 'review_images.py':
            transform = lambda text: portable_python(text, private_review=True)
        copy(source, relative, transform, 'current_qc/code/'+source.name)
        script_names.append(relative)
    for name in STARTUP_SCRIPTS:
        relative = 'extensions/startup_screening/'+name
        copy(project/STARTUP_REL/'code'/name, relative,
             lambda text: portable_python(text, startup=True), 'startup_screening/code/'+name)
        script_names.append(relative)
    copy(out/'方案.md', 'docs/qc_protocol.md', label='current_qc/fixed_protocol')
    for source in sorted((out/'analysis/aggregate').glob('*')):
        if source.suffix not in {'.csv', '.json', '.txt', '.md'}:
            raise ValueError('Unrecognized aggregate file: '+source.name)
        copy(source, 'outputs/aggregate/qc/'+source.name, label='current_qc/aggregate/'+source.name)
        if re.fullmatch(r'Supp_Table_S\d+\.csv', source.name):
            copy(source, 'outputs/tables/'+source.name, label='current_qc/aggregate/'+source.name)
    for suffix in ['.png', '.svg', '.pdf']:
        name = 'Supplementary_Figure_S2_Startup_QC'+suffix
        source = out/'新版稿件/02_图件'/name
        if source.exists():
            copy(source, 'outputs/figures/'+name, label='current_qc/aggregate_figure/'+name)
    for source in sorted((out/'qa').glob('*.json')):
        if source.name in QA_NAMES or re.fullmatch(r'(?:treatment_.+(?:verification|comparison|audit)|outcomes_.+_R_verification)\.json', source.name):
            clean = scrub_qa(json.loads(source.read_text()))
            clean['public_QA_copy'] = 'Aggregate verdict only; private paths and identifiers omitted'
            json_write(archive/'outputs/qc'/source.name, clean)
    figure_qa = out/'qa/figurechecks/S2_checks.json'
    if figure_qa.exists():
        json_write(archive/'outputs/qc/S2_checks.json', scrub_qa(json.loads(figure_qa.read_text())))
    (archive/'docs').mkdir(parents=True, exist_ok=True)
    (archive/'README.md').write_text(readme(), encoding='utf-8')
    (archive/'docs/reproducibility.md').write_text(reproduction_docs(script_names), encoding='utf-8')
    (archive/'docs/release_notes.md').write_text(
        '# Local Candidate Release Notes\n\n'+VERSION+' is LOCAL NOT PUBLISHED.\n'
        'Clinical adjudication is pending. Original primary results are retained; '
        'technical-QC sensitivities are added. No original release is replaced.\n'
        'Baseline only: '+BASELINE_GITHUB+'\nBaseline only: '+BASELINE_DOI+'\n', encoding='utf-8')
    (archive/'requirements-qc.txt').write_text(
        '# Core numerical versions are retained in requirements.txt.\n'
        '-r requirements.txt\nPillow\npypdf\nreportlab\n'
        '# Node artifact dependencies are documented separately, not vendored.\n', encoding='utf-8')
    codemeta = json.loads((baseline/'codemeta.json').read_text())
    json_write(archive/'codemeta.json', candidate_codemeta(codemeta))
    zenodo = json.loads((baseline/'.zenodo.json').read_text())
    for key in ['publication_date', 'doi', 'prereserve_doi']:
        zenodo.pop(key, None)
    zenodo.update({'version': VERSION, 'description': candidate_codemeta(codemeta)['description'],
                   'notes': 'LOCAL NOT PUBLISHED. Clinical adjudication pending. No candidate DOI has been assigned. '
                            'Related identifiers refer to the v1.4.0 baseline only.',
                   'related_identifiers': [{'identifier': BASELINE_DOI, 'relation': 'isDerivedFrom', 'scheme': 'url'},
                                           {'identifier': BASELINE_GITHUB, 'relation': 'isDerivedFrom', 'scheme': 'url'}]})
    json_write(archive/'.zenodo.json', zenodo)
    citation = yaml.safe_load((baseline/'CITATION.cff').read_text())
    for key in ['doi', 'date-released']:
        citation.pop(key, None)
    citation.update({'version': VERSION.removeprefix('v'), 'message':
        'LOCAL NOT PUBLISHED candidate; independent physician clinical adjudication not performed. Baseline references are not candidate citations.',
        'references': [{'type': 'software', 'title': 'Baseline reproducibility release v1.4.0 only',
                        'version': '1.4.0', 'doi': '10.5281/zenodo.23153577', 'url': BASELINE_GITHUB}]})
    (archive/'CITATION.cff').write_text(yaml.safe_dump(citation, sort_keys=False, allow_unicode=True), encoding='utf-8')
    assert citation['authors'] == yaml.safe_load((baseline/'CITATION.cff').read_text())['authors']
    assert zenodo['creators'] == json.loads((baseline/'.zenodo.json').read_text())['creators']
    assert candidate_codemeta(codemeta)['author'] == codemeta['author']
    (archive/'config/inputs.example.sh').write_text(
        '# Example only: set these locally; never commit private configuration.\n'
        'export IOH_PROJECT_ROOT="REQUIRED_LOCAL_SOURCE_PROJECT"\n'
        'export IOH_QC_OUTPUT_ROOT="REQUIRED_LOCAL_QC_WORKSPACE"\n'
        'export IOH_STARTUP_AUDIT_ROOT="REQUIRED_LOCAL_STARTUP_AUDIT"\n'
        'export IOH_VITALDB_RAW_ROOT="REQUIRED_LOCAL_VITALDB_ROOT_WITH_TRKS_CSV"\n'
        'export IOH_VITALDB_TRACK_DIRS="REQUIRED_LOCAL_TRACK_DIRECTORIES_SEPARATED_BY_OS_PATHSEP"\n'
        'export IOH_VITALDB_LABS="REQUIRED_LOCAL_OFFICIAL_LABS_CSV"\n'
        '# Required only for the observation-packaging script, not statistical reruns:\n'
        '# export IOH_IMAGE_REVIEW_INPUT_JSON="PRIVATE_PRIOR_REVIEW_OBSERVATIONS_JSON"\n'
        '# Optional when dependencies are installed outside the active environment:\n'
        '# export IOH_QC_DEPENDENCIES="LOCAL_PYTHON_DEPENDENCIES"\n'
        '# export CODEX_NODE_MODULES="LOCAL_NODE_DEPENDENCIES"\n', encoding='utf-8')
    (archive/'Makefile').write_text(
        '.PHONY: test verify\n'
        'test:\n\tPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider\n'
        'verify:\n\tpython extensions/qc_startup/build_archive.py --verify\n', encoding='utf-8')
    current_paths = {row['path'] for row in inventory}
    for row in previous:
        relative = row['path']
        if relative in current_paths or relative == 'outputs/final_workbook.xlsx':
            continue
        path = (archive/relative).resolve()
        if not path.is_relative_to(archive.resolve()):
            raise ValueError('Invalid previous staging inventory path')
        if path.is_file():
            path.unlink()
    json_write(archive/'docs/staging_inventory.json', {'version': VERSION, 'not_a_final_release_manifest': True,
               'source_files_not_modified': True, 'workbook_copy': 'NEVER_PERFORMED',
               'staged_code_and_data': inventory})
    json_write(archive/'ARCHIVE_STATUS.json', {'version': VERSION, 'state': 'STAGED_NOT_FROZEN',
               'publication': 'LOCAL_NOT_PUBLISHED', 'clinical_adjudication': 'NOT_PERFORMED',
               'baseline_reference_only': {'github': BASELINE_GITHUB, 'zenodo': BASELINE_DOI},
               'freeze_requires_explicit_flag': True})
    report = scan_archive(archive)
    json_write(archive/'outputs/qc/archive_scan.json', report)
    if report['status'] != 'PASS':
        raise ValueError('Archive boundary scan failed: '+json.dumps(report['failures'], ensure_ascii=False))
    return {'status': 'STAGED_NOT_FROZEN', 'files': report['scanned_files'],
            'copied_sources': len(inventory), 'workbook_present': (archive/'outputs/final_workbook.xlsx').exists(),
            'publication': 'LOCAL_NOT_PUBLISHED', 'scan': report['status']}


def main():
    default_out = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-project', type=Path, default=Path(os.environ.get('IOH_PROJECT_ROOT', default_out.parent)))
    ap.add_argument('--output-root', type=Path, default=Path(os.environ.get('IOH_QC_OUTPUT_ROOT', default_out)))
    ap.add_argument('--archive-root', type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    out, project = args.output_root.expanduser().resolve(), args.source_project.expanduser().resolve()
    if args.archive_root:
        archive = args.archive_root.expanduser().resolve()
    elif args.verify and Path(__file__).parent.name == 'qc_startup':
        archive = Path(__file__).resolve().parents[2]
    else:
        archive = out/'归档候选'
    if args.verify:
        report = scan_archive(archive)
        if report['status'] != 'PASS':
            raise ValueError('Archive boundary scan failed')
        print(json.dumps(verify_manifests(archive), ensure_ascii=False))
        return
    if not archive.is_relative_to(out) or archive == out:
        ap.error('Archive must be a dedicated child directory of the QC output root')
    archive.mkdir(parents=True, exist_ok=True)
    summary = stage_archive(project, out, archive)
    if args.freeze:
        require_freeze_inputs(archive)
        status = json.loads((archive/'ARCHIVE_STATUS.json').read_text())
        status.update({'state': 'FROZEN_LOCAL_CANDIDATE', 'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
        json_write(archive/'ARCHIVE_STATUS.json', status)
        scan = scan_archive(archive)
        if scan['status'] != 'PASS':
            raise ValueError('Final boundary scan failed')
        json_write(archive/'outputs/qc/archive_scan.json', scan)
        write_manifests(archive)
        summary = {**verify_manifests(archive), 'status': 'FROZEN_LOCAL_CANDIDATE'}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
