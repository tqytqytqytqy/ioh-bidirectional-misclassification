"""Event-only sensitivity requiring observed normal-pressure boundaries."""
import numpy as np
import pandas as pd
from rerun import ROOT,PROJECT,ANALYSIS,COLS,components_to_metrics,trimmed_values
from rerun import held_display,reference_events,classify_events
from qc_onset import sha,write_json


def complete_events(v,events,threshold=65):
    return [(a,b,k) for a,b,k in events if a>0 and b<len(v) and np.isfinite(v[a-1])
            and v[a-1]>=threshold and np.isfinite(v[b]) and v[b]>=threshold]


def run():
    contract=ROOT/'boundary_check_contract.json';before=sha(contract)
    panel=pd.read_parquet(ANALYSIS/'outputs/intermediate/artmap_10s.parquet')
    onset=pd.read_csv(ROOT/'restricted/onsets.csv').set_index('case_id')
    frame=pd.read_csv(ROOT/'restricted/case_components.csv.gz')
    names=['original_matched_primary_onset','primary_core60_onset']
    base=frame[(frame.scenario.isin(names))&(frame.initialization=='unavailable')&
        (frame.resolution=='native10s')&(frame.threshold==65)&(frame.interval_min==5)&(frame.minimum_event_sec==60)].set_index(['scenario','case_id'])
    rows=[]
    for cid,g in panel.groupby('case_id',sort=True):
        r=onset.loc[cid]
        if not r.primary_pass:continue
        orig=g.sort_values('time_sec').art_map.to_numpy(float)
        for name in names:
            v=orig if name==names[0] else trimmed_values(orig,r.primary_onset_sec)
            all_events=reference_events(v,65,0,6);ev=complete_events(v,all_events)
            cc=[];delays=[]
            for phase in range(30):
                d,held=held_display(v,30,phase);cl=classify_events(v,d,held,ev,65)
                cc.append(np.bincount(cl['category'],minlength=3))
                low=np.isfinite(d)&(d<65)
                delays.append(sum(np.argmax(low[a:b])/6 for a,b,_ in ev if low[a:b].any()))
            old=base.loc[(name,cid)].copy();old['scenario']=name;old['case_id']=cid
            old['episodes_before_boundary_filter']=len(all_events)
            old['episodes']=len(ev);old[['fresh','inherited_only','never_low']]=np.mean(cc,axis=0)
            old['delay_sum_min']=np.mean(delays);rows.append(old.to_dict())
    f=pd.DataFrame(rows);p=ROOT/'restricted/boundary_complete_cases.csv.gz';f.to_csv(p,index=False);p.chmod(0o600)
    results=[]
    for name,g in f.groupby('scenario'):
        by=g.groupby('subject_id')[COLS].sum().sort_index();n=len(by)
        w=np.random.default_rng(20260706).multinomial(n,np.full(n,1/n),size=1000)
        m=components_to_metrics(by.sum().to_numpy(float));bm=components_to_metrics(w@by.to_numpy(float))
        out=dict(scenario=name,n_cases=len(g),n_subjects=n,episodes_before=g.episodes_before_boundary_filter.sum(),
            episodes_after=g.episodes.sum(),n_cases_with_complete_events=int(g.episodes.gt(0).sum()))
        for k in ['fresh_probability','inherited_probability','miss_probability']:
            lo,hi=np.nanquantile(bm[k],[.025,.975]);out.update({k:float(m[k]),k+'_lo':float(lo),k+'_hi':float(hi)})
        results.append(out)
    pd.DataFrame(results).to_csv(ROOT/'aggregate/boundary_complete_summary.csv',index=False)
    assert sha(contract)==before
    write_json(ROOT/'qa/boundary_check_verification.json',dict(status='PASS',contract_sha256=before,
        scenarios=len(results),frequency_components_unchanged=True,
        clinical_adjudication='NOT_PERFORMED'))
    print(pd.DataFrame(results).to_json(orient='records'))


if __name__=='__main__':run()
