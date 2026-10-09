"""Publication-boundary and noncircular-freeze tests using synthetic files only."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

MODULE = Path(__file__).with_name('build_archive.py')


@pytest.fixture
def api():
    assert MODULE.exists(), 'Archive builder has not been implemented'
    spec = importlib.util.spec_from_file_location('build_archive', MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_denominator_columns_are_allowed(api, tmp_path):
    file = tmp_path/'counts.csv'
    file.write_text('policy,n_cases,n_subjects,events\nqc60,100,98,12\n')
    assert api.inspect_file(file, Path('outputs/aggregate/counts.csv')) == []


def test_patient_identifier_columns_fail_closed(api, tmp_path):
    file = tmp_path/'records.csv'
    file.write_text('case_id,subjectid,estimate\n101,10,2.0\n')
    issues = api.inspect_file(file, Path('outputs/aggregate/records.csv'))
    assert any('identifier_column' in issue for issue in issues)


def test_private_json_arrays_are_removed_but_aggregate_counts_retained(api):
    private = '/'+'Users'+'/someone/private.csv'
    clean = api.scrub_qa({'n_cases': 10, 'changed_ids': [1, 2], 'output': private,
                          'source_hashes': {private: 'abcd'}, 'status': 'PASS'})
    assert clean['n_cases'] == 10 and clean['status'] == 'PASS'
    assert 'changed_ids' not in clean and 'source_hashes' not in clean
    assert private not in json.dumps(clean)


def test_private_json_key_fails_closed(api, tmp_path):
    file = tmp_path/'bad.json'
    file.write_text(json.dumps({'policy': 'qc60', 'case_ids': [1, 2]}))
    assert any('identifier_key' in issue for issue in api.inspect_file(file, Path('outputs/aggregate/bad.json')))


def test_env_and_private_directory_fail_closed(api, tmp_path):
    file = tmp_path/'fixture.txt'
    file.write_text('benign')
    assert any('forbidden_path' in issue for issue in api.inspect_file(file, Path('.env')))
    assert any('forbidden_path' in issue for issue in api.inspect_file(file, Path('qa/outcomes_runtime/a.txt')))


def test_local_home_path_in_text_is_blocked(api, tmp_path):
    file = tmp_path/'metadata.json'
    file.write_text(json.dumps({'source': '/'+'Users'+'/someone/work'}))
    assert any('local_path' in issue for issue in api.inspect_file(file, Path('metadata.json')))


def test_python_copy_honors_config_without_touching_source(api):
    text = ('from pathlib import Path\nimport sys\n'
            'sys.path.insert(0, '+repr('/'+'tmp/ioh_startup_deps_20261006')+')\n'
            'OUT = Path(__file__).resolve().parents[1]\nROOT = OUT.parent\n')
    transformed = api.portable_python(text, startup=False)
    namespace = {'__file__': '/synthetic/code/example.py'}
    import os
    old = {key: os.environ.get(key) for key in ['IOH_PROJECT_ROOT', 'IOH_QC_OUTPUT_ROOT']}
    try:
        os.environ['IOH_PROJECT_ROOT'] = '/synthetic/project'
        os.environ['IOH_QC_OUTPUT_ROOT'] = '/synthetic/qc'
        exec(compile(transformed, '<portable-copy>', 'exec'), namespace)
        assert namespace['ROOT'] == Path('/synthetic/project')
        assert namespace['OUT'] == Path('/synthetic/qc')
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert 'Path(__file__).resolve().parents[1]' in text


def test_stage_cleanup_removes_only_current_manifest_leaves(api, tmp_path):
    workbook = tmp_path/'outputs/final_workbook.xlsx'
    workbook.parent.mkdir()
    workbook.write_bytes(b'owned-by-workbook-agent')
    manifest = tmp_path/'outputs/manifests/outputs_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('{}')
    (tmp_path/'checksums.sha256').write_text('old')
    api.clear_freeze_files(tmp_path)
    assert not manifest.exists() and not (tmp_path/'checksums.sha256').exists()
    assert workbook.read_bytes() == b'owned-by-workbook-agent'


def test_metadata_preserves_author_order_and_has_no_candidate_doi(api):
    original = {'author': [{'name': 'B'}, {'name': 'A'}], 'version': '1.4.0',
                'identifier': 'old-doi', 'datePublished': '2026-10-05'}
    current = api.candidate_codemeta(original)
    assert current['author'] == original['author']
    assert current['version'] == api.VERSION.removeprefix('v')
    assert 'identifier' not in current and 'datePublished' not in current
    assert current['isBasedOn'].endswith('23153577')


def test_freeze_hashes_are_noncircular_and_tampering_is_detected(api, tmp_path):
    output = tmp_path/'outputs/aggregate'
    output.mkdir(parents=True)
    (output/'test.csv').write_text('n_cases,events\n10,2\n')
    (tmp_path/'README.md').write_text('LOCAL NOT PUBLISHED')
    api.write_manifests(tmp_path)
    assert api.verify_manifests(tmp_path)['status'] == 'PASS'
    release = json.loads((tmp_path/'outputs/manifests/release_manifest.json').read_text())
    files = {r['path'] for r in release['files']}
    assert 'outputs/manifests/outputs_manifest.json' in files
    assert 'outputs/manifests/release_manifest.json' not in files
    assert 'checksums.sha256' not in files
    (output/'test.csv').write_text('n_cases,events\n10,3\n')
    with pytest.raises(ValueError):
        api.verify_manifests(tmp_path)


def test_freeze_requires_workbook_and_current_publication_tables(api, tmp_path):
    with pytest.raises(ValueError):
        api.require_freeze_inputs(tmp_path)


@pytest.fixture
def freeze_inputs(api, tmp_path):
    aggregate = tmp_path/'outputs/aggregate/qc'
    aggregate.mkdir(parents=True)
    for name in ['events_summary', 'treatment_summary', 'outcomes_summary']:
        (aggregate/(name+'.json')).write_text('{}')
    (aggregate/'qc_publication_tables.json').write_text('[]')
    (tmp_path/'outputs/final_workbook.xlsx').write_bytes(b'synthetic workbook')
    (aggregate/'qc_figure_caption.txt').write_text('Synthetic aggregate figure caption.\n')
    (aggregate/'qc_figure_source.csv').write_text('policy,estimate\noriginal,1\n')
    figures = tmp_path/'outputs/figures'
    figures.mkdir()
    hashes = {}
    for suffix in ['png', 'pdf', 'svg']:
        figure = figures/('Supplementary_Figure_S2_Startup_QC.'+suffix)
        figure.write_bytes(b'synthetic figure')
        hashes[suffix] = api.sha(figure)
    api.json_write(tmp_path/'outputs/qc/S2_checks.json', {
        'status': 'PASS', 'caption_words': 4,
        'caption_sha256': api.sha(aggregate/'qc_figure_caption.txt'),
        'source_csv_sha256': api.sha(aggregate/'qc_figure_source.csv'),
        'figure_sha256': hashes})
    return tmp_path


def test_freeze_accepts_consistent_final_figure_evidence(api, freeze_inputs):
    api.require_freeze_inputs(freeze_inputs)


@pytest.mark.parametrize('relative', [
    'outputs/aggregate/qc/qc_figure_caption.txt',
    'outputs/aggregate/qc/qc_figure_source.csv',
    'outputs/figures/Supplementary_Figure_S2_Startup_QC.png',
    'outputs/figures/Supplementary_Figure_S2_Startup_QC.pdf',
    'outputs/figures/Supplementary_Figure_S2_Startup_QC.svg'])
def test_freeze_rejects_stale_figure_evidence(api, freeze_inputs, relative):
    with (freeze_inputs/relative).open('ab') as stream:
        stream.write(b' changed')
    with pytest.raises(ValueError, match='Figure S2 QA hash mismatch'):
        api.require_freeze_inputs(freeze_inputs)


def test_freeze_rejects_pending_figure_qa(api, freeze_inputs):
    qa = freeze_inputs/'outputs/qc/S2_checks.json'
    value = json.loads(qa.read_text())
    value['status'] = 'GENERATED_PENDING_VISUAL_REVIEW'
    api.json_write(qa, value)
    with pytest.raises(ValueError, match='Figure S2 QA is not final'):
        api.require_freeze_inputs(freeze_inputs)


def test_public_copy_removes_embedded_review_observations(api):
    text = "import json\nfrom pathlib import Path\nSTATIC = json.loads(" + repr(
        json.dumps({'reviews': {'QC'+'123': ['private observation']}})) + ")\n"
    result = api.portable_python(text, private_review=True)
    assert 'private observation' not in result and 'QC'+'123' not in result
    assert 'IOH_IMAGE_REVIEW_INPUT_JSON' in result


def test_windows_path_check_does_not_match_ordinary_escaped_prose(api, tmp_path):
    file = tmp_path/'example.py'
    file.write_text('message = "scope:\\n"\n')
    assert api.inspect_file(file, Path('extensions/example.py')) == []


def test_current_manifests_are_also_privacy_scanned(api, tmp_path):
    directory = tmp_path/'outputs/manifests'
    directory.mkdir(parents=True)
    (directory/'release_manifest.json').write_text(json.dumps({'path': '/'+'Users'+'/private/file'}))
    assert api.scan_archive(tmp_path)['status'] == 'FAIL'


def test_archive_symlinks_are_rejected_before_copy(api, tmp_path):
    archive = tmp_path/'archive'
    external = tmp_path/'external'
    archive.mkdir(); external.mkdir()
    (archive/'extensions').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='Archive contains symlink'):
        api.stage_archive(tmp_path/'source', tmp_path/'qc', archive)
    assert list(external.iterdir()) == []


def test_portable_copy_does_not_override_already_resolved_cli_paths(api, monkeypatch):
    monkeypatch.setenv('IOH_PROJECT_ROOT', '/synthetic/environment-project')
    monkeypatch.setenv('IOH_QC_OUTPUT_ROOT', '/synthetic/environment-output')
    text = ('from pathlib import Path\nfrom types import SimpleNamespace\n'
            'PATH_CONFIG = SimpleNamespace(source_project=Path("/synthetic/cli-project"), '
            'output_root=Path("/synthetic/cli-output"))\n'
            'ROOT = PATH_CONFIG.source_project.resolve()\n'
            'OUT = PATH_CONFIG.output_root.resolve()\n')
    namespace = {}
    exec(api.portable_python(text), namespace)
    assert namespace['ROOT'] == Path('/synthetic/cli-project')
    assert namespace['OUT'] == Path('/synthetic/cli-output')
