"""Run extension tests on synthetic inputs only, in isolated processes."""
from pathlib import Path
import os, subprocess, sys, json, tempfile, re
REPO=Path(__file__).resolve().parents[2]
LAYOUT=REPO/'source_layout'
GROUPS=[
 ('跨库分辨率与异质性分析_20261015/code','test_*.py'),
 ('可靠动脉起点质控重分析_20261015/code','test_*.py'),
 ('EJA_临床时间信息修订版_20261016/code','test_timing.py'),
 ('EJA_审稿意见逐项修订版_20261016/code','test_review_core.py'),
 ('MIMIC技术重复分析_20261015/qc','test_full_qc.py'),
 ('MIMIC技术重复分析_20261015/analysis/tests','test_mimic_analysis.py'),
 ('SICdb分钟级对齐分析_20261015/code','test_*.py'),
 ('外部验证可行性审计_20261015/mimic','test_audit.py'),
]
def main():
    receipts=[]
    with tempfile.TemporaryDirectory(prefix='ioh-v2-tests-') as temp:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=temp)
        paths=[REPO/'src',REPO/'extensions/events',REPO/'extensions/infusion',LAYOUT/'MIMIC技术重复分析_20261015/analysis/code']
        env['PYTHONPATH']=os.pathsep.join([str(p) for p in paths]+[env.get('PYTHONPATH','')])
        commands=[(folder,[sys.executable,'-B','-m','unittest','discover','-s',str(LAYOUT/folder),'-p',pattern]) for folder,pattern in GROUPS]
        for rel in ['跨库分辨率与异质性分析_20261015/code/independent_verify.py','可靠动脉起点质控重分析_20261015/code/audit_raw.py','可靠动脉起点质控重分析_20261015/code/audit_documented_encoding.py']:
            commands.append((rel,[sys.executable,'-B',str(LAYOUT/rel),'--self-test']))
        for label,cmd in commands:
            result=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True)
            output=result.stdout+'\n'+result.stderr;count=re.search(r'Ran (\d+) tests?',output)
            entry={'suite':label,'exit_code':result.returncode,'tests':int(count[1]) if count else None}
            receipts.append(entry);print(json.dumps(entry),flush=True)
            if result.returncode:print(output,flush=True)
    report={'status':'PASS' if all(r['exit_code']==0 for r in receipts) else 'FAIL','suites':receipts,'tests':sum(r['tests'] or 0 for r in receipts),'scope':'Synthetic tests only; no source cohort execution or clinical adjudication.'}
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
