"""Separate chronological replay, without using production state functions."""
from pathlib import Path
import sys,json
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(P/'跨库分辨率与异质性分析_20261015/code'))
from io_sources import vital_records,pd
f=pd.read_csv(R/'restricted/fresh_events.csv');r=pd.read_csv(R/'restricted/recovery_events.csv')
rng=np.random.default_rng(160026)
keys=f[['case_id','scenario','event_start']].drop_duplicates().sample(300,random_state=160026)
selected=set(keys.case_id)|set(r.case_id)
count=0
for item in vital_records():
    cid=int(item['record_id'])
    if cid not in selected:continue
    for scenario in ['original_all','original_matched_core60','core60']:
        if scenario!='original_all' and not item['onset']['primary_pass']:continue
        v=item['values'].copy()
        if scenario=='core60':v[:int(item['onset']['primary_onset_sec']//10)]=np.nan
        ff=f[(f.case_id==cid)&(f.scenario==scenario)]
        kk=keys[(keys.case_id==cid)&(keys.scenario==scenario)].event_start
        ff=ff[ff.event_start.isin(kk)]
        rr=r[(r.case_id==cid)&(r.scenario==scenario)]
        for kind,frame in [('fresh',ff),('recovery',rr)]:
            for row in frame.itertuples():
                n=int(row.interval_min*6);h=int(row.horizon_sec);states=[]
                for phase in range(n):
                    shown=np.nan;source=-1
                    if kind=='fresh':
                        start=int(row.event_start//10);end=start+int(row.duration_sec//10)
                        deadline=start*10+h;inherited=False;state=None
                        for j in range(min(len(v),int(deadline//10))):
                            t=(j+1)*10
                            if j% n==phase and np.isfinite(v[j]):shown=v[j];source=j
                            if t<start*10:continue
                            if t==start*10:
                                inherited=np.isfinite(shown) and shown<65;continue
                            if start<=j<end and source>=start and shown<65:state='fresh';break
                            if j>=end:
                                state='recovered_without_fresh' if np.isfinite(v[j]) and v[j]>=65 else 'unknown';break
                            inherited=inherited or (np.isfinite(shown) and shown<65 and source<start)
                        if state is None:
                            if len(v)*10<deadline:state='unknown'
                            else:state='still_low_inherited' if inherited else 'still_low_without_low_display'
                    else:
                        start=int(row.recovery_sec//10)-1;deadline=row.recovery_sec+h;state=None
                        for j in range(min(len(v),int(deadline//10))):
                            if j%n==phase and np.isfinite(v[j]):shown=v[j]
                            if j<start:continue
                            if not np.isfinite(v[j]):state='unknown';break
                            if v[j]<65:state='reference_recurred';break
                            if not np.isfinite(shown):state='unknown';break
                            if shown>=65:state='display_cleared';break
                        if state is None:state='unknown' if len(v)*10<deadline else 'persistent_low'
                    states.append(state)
                cols=['fresh','recovered_without_fresh','still_low_inherited','still_low_without_low_display','unknown'] if kind=='fresh' else ['persistent_low','display_cleared','reference_recurred','unknown']
                for k in cols:assert abs(states.count(k)/n-getattr(row,k))<1e-12,(kind,cid,k)
                count+=1
(R/'qa/independent_verification.json').write_text(json.dumps({'status':'PASS','independent_chronological_replays':count,'sampled_fresh_event_scenarios':300,'recovery_scope':'all events, scenarios, intervals and horizons'},indent=2))
print(json.dumps({'status':'PASS','independent_replays':count}))
