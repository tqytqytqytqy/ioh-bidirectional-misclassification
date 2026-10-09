"""Run synthetic test suites without loading private study inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def main():
    repo = Path(__file__).resolve().parents[1]
    groups = [repo / 'tests']
    groups.extend(sorted({p.parent for p in (repo / 'extensions').rglob('test_*.py')}))
    env = {k: v for k, v in os.environ.items()
           if k in {'PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
    receipts = []
    with tempfile.TemporaryDirectory(prefix='ioh-synthetic-') as temp:
        work = Path(temp)
        scaffold = work / 'source'
        baseline = scaffold / 'EJA_投稿文件包_20261006_最终文字修订/03_可复现材料'
        baseline.mkdir(parents=True)
        for name in ['scripts', 'src']:
            (baseline / name).symlink_to(repo / name, target_is_directory=True)
        treatment = scaffold / '治疗相关事件补充修订_20261004'
        treatment.mkdir()
        (treatment / 'code').symlink_to(repo / 'extensions/infusion', target_is_directory=True)
        env.update(IOH_PROJECT_ROOT=str(scaffold), IOH_QC_OUTPUT_ROOT=str(work / 'qc'),
                   IOH_STARTUP_AUDIT_ROOT=str(work / 'startup'),
                   IOH_VITALDB_RAW_ROOT=str(work / 'empty_source_recordings'),
                   IOH_VITALDB_TRACK_DIRS=str(work / 'empty_tracks'),
                   IOH_VITALDB_LABS=str(work / 'no_laboratory_data.csv'),
                   PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(work / 'matplotlib'))
        inherited_pythonpath = os.environ.get('PYTHONPATH', '')
        common = [str(repo / 'src'), str(repo), str(repo / 'extensions/events'),
                  str(repo / 'extensions/infusion')]
        for index, group in enumerate(groups):
            group_env = dict(env)
            group_env['PYTHONPATH'] = os.pathsep.join(common + [str(group), inherited_pythonpath])
            junit = work / f'suite_{index}.xml'
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                     '--basetemp', str(work / f'test_{index}'), '--junitxml', str(junit),
                                     str(group)], cwd=repo, env=group_env, capture_output=True, text=True)
            suites = ET.parse(junit).getroot().findall('testsuite')
            receipt = {'suite': group.relative_to(repo).as_posix(), 'exit_code': result.returncode}
            receipt.update({key: sum(int(s.get(key, '0')) for s in suites)
                            for key in ['tests', 'failures', 'errors', 'skipped']})
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
            if result.returncode:
                print(result.stdout)
                print(result.stderr)
    passed = all(s['exit_code'] == 0 for s in receipts)
    print(json.dumps({'status': 'PASS' if passed else 'FAIL', 'suites': receipts,
                      'scope': 'Synthetic inputs only; no source-data statistical refit.'}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
