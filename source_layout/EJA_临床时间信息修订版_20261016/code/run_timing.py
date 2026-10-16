from pathlib import Path
import sys,json,hashlib,os
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(P/'跨库分辨率与异质性分析_20261015/code'))
from io_sources import vital_records,pd,sha
sys.path.insert(0,str(P/'治疗相关事件补充修订_20261004/code'))
from treatment_core import event_state
from timing import *

def freeze():
    for folder in ['aggregate','qa','restricted']:(R/folder).mkdir(exist_ok=True)
    (R/'restricted').chmod(0o700)
    baseline=P/'EJA_跨库语言精修版_20261015'
    files=list(baseline.joinpath('01_修订稿件').glob('*'))
    files += [P/'analysis_v10_subject_phase_release/outputs/intermediate'/x for x in ['artmap_10s.parquet','vitaldb_manifest.parquet']]
    files += [P/'可靠动脉起点质控重分析_20261015/restricted/onsets.csv',P/'独立质控重分析_20261007/data_restricted/treatment_original_primary_events.csv',P/'独立质控重分析_20261007/data_restricted/treatment_original_primary_recovery.csv',R/'analysis_contract.json']
    inputs={str(f):sha(f) for f in files if f.is_file()}
    old=json.loads((P/'可靠动脉起点质控重分析_20261015/analysis_contract.json').read_text())
    assert sha(P/'analysis_v10_subject_phase_release/outputs/intermediate/artmap_10s.parquet')==old['base_map_panel_sha256']
    assert sha(P/'analysis_v10_subject_phase_release/outputs/intermediate/vitaldb_manifest.parquet')==old['base_metadata_sha256']
    path=R/'qa/frozen_inputs.json'
    if path.exists():assert json.loads(path.read_text())['sha256']==inputs
    else:path.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'new_effects_started':False,'sha256':inputs},indent=2))
    return inputs

def private_csv(frame,name):
    path=R/'restricted'/name;frame.to_csv(path,index=False);path.chmod(0o600)

def main():
    inputs=freeze()
    therapy=pd.read_csv(P/'独立质控重分析_20261007/data_restricted/treatment_original_primary_events.csv')
    expected=pd.read_csv(P/'独立质控重分析_20261007/data_restricted/treatment_original_primary_recovery.csv')
    fresh=[];recovery=[];flow=[];tflow=[];comparisons=0
    actual_recovery=set()
    for case_no,item in enumerate(vital_records(),1):
        orig=item['values'];cid=int(item['record_id']);sid=item['subject_id'];q=item['onset']
        specs=[('original_all',orig)]
        if q['primary_pass']:
            v=orig.copy();v[:int(q['primary_onset_sec']//10)]=np.nan
            specs += [('original_matched_core60',orig),('core60',v)]
        tc=therapy[therapy.case_id.eq(cid)]
        for scenario,v in specs:
            base={'scenario':scenario,'subject_id':sid,'case_id':cid}
            runs=low_runs(v)
            flow.append(dict(base,valid_minutes=np.isfinite(v).sum()/6,low_minutes=(v<65).sum()/6,auc_mmHg_min=np.nansum(np.maximum(65-v,0))/6,events=len(runs),confirmed_events=sum(c for a,b,c in runs)))
            held={n:display_matrix(v,n) for n in [6,18,30]}
            for a,b,confirmed in runs:
                for n in [6,18,30]:
                    for h in [60,120]:
                        z=fresh_states(v,a,b,n,h,held[n])
                        assert abs(sum(z[k] for k in FRESH)-1)<1e-12
                        assert abs(z['fresh']-min(b-a,h//10,n)/n)<1e-12
                        comparisons+=1
                        fresh.append(dict(base,event_start=a*10,duration_sec=(b-a)*10,nadir=float(v[a:b].min()),confirmed=confirmed,interval_min=n/6,horizon_sec=h,**z))
            eligible=low=rec=0
            for row in tc.itertuples():
                st=event_state(v,row.time_sec,30,0)
                if row.time_sec<300 or st is None:continue
                eligible+=1
                if st['reference']>=65:continue
                low+=1
                ri=confirmed_recovery(v,row.time_sec)
                if ri is None:continue
                rec+=1
                if scenario=='original_all':actual_recovery.add((cid,float(row.time_sec)))
                for n in [6,18,30]:
                    for h in [60,120]:
                        z=recovery_states(v,ri,n,h,held[n])
                        assert abs(sum(z[k] for k in RECOVERY)-1)<1e-12
                        recovery.append(dict(base,adjustment_sec=row.time_sec,recovery_sec=(ri+1)*10,interval_min=n/6,horizon_sec=h,**z))
            if len(tc):tflow.append(dict(base,original_candidate_adjustments=len(tc),eligible=eligible,low=low,recovered=rec))
        if case_no%400==0:print(json.dumps({'cases_processed':case_no,'new_fresh_rows':len(fresh)}),flush=True)
    expectedset=set(zip(expected.case_id.astype(int),expected.time_sec.astype(float)))
    assert expectedset==actual_recovery,(len(expectedset),len(actual_recovery))
    frames={'fresh_events.csv':pd.DataFrame(fresh),'recovery_events.csv':pd.DataFrame(recovery),'case_flow.csv':pd.DataFrame(flow),'treatment_flow.csv':pd.DataFrame(tflow)}
    for name,f in frames.items():private_csv(f,name)
    summaries=[]
    for s,g in frames['case_flow.csv'].groupby('scenario'):
        summaries.append(dict(scenario=s,cases=len(g),patients=g.subject_id.nunique(),**{k:float(g[k].sum()) for k in ['valid_minutes','low_minutes','auc_mmHg_min','events','confirmed_events']}))
    pd.DataFrame(summaries).to_csv(R/'aggregate/cohort_flow.csv',index=False)
    t=[]
    for s,g in frames['treatment_flow.csv'].groupby('scenario'):
        t.append(dict(scenario=s,**{k:int(g[k].sum()) for k in ['eligible','low','recovered']},low_patients=g.loc[g.low.gt(0),'subject_id'].nunique(),recovery_patients=g.loc[g.recovered.gt(0),'subject_id'].nunique()))
    pd.DataFrame(t).to_csv(R/'aggregate/treatment_flow.csv',index=False)
    assert all(sha(Path(p))==h for p,h in inputs.items())
    (R/'qa/analysis_verification.json').write_text(json.dumps({'status':'PASS','partitions_exact':True,'discrete_phase_comparisons':comparisons,'original_recovery_events_reproduced':len(actual_recovery),'input_hashes_unchanged':True},indent=2))
    print(json.dumps({'status':'PASS','cases':len(frames['case_flow.csv']),'recovery_original':len(actual_recovery)}),flush=True)

if __name__=='__main__':main()
