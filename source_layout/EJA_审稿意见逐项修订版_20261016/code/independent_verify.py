"""Independent event enumeration and bootstrap checks, with aggregate-only logs."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(P/'跨库分辨率与异质性分析_20261015/code'))
from io_sources import vital_records,pd,sha

def brute_state(v,ri,step,horizon):
    result=np.zeros(4)
    for phase in range(step):
        held=None
        for t in range(0,ri+1):
            if t%step==phase and np.isfinite(v[t]):held=float(v[t])
        state=0
        for t in range(ri,ri+horizon//10+1):
            if t>=len(v) or not np.isfinite(v[t]):state=3;break
            if t>ri and t%step==phase:held=float(v[t])
            if v[t]<65:state=2;break
            if held is None:state=3;break
            if held>=65:state=1;break
        result[state]+=1/step
    return result

f=pd.read_csv(R/'restricted/recovery_pairs.csv');metrics=['persistent_low','display_cleared','reference_recurred','unknown']
select=f.query('scenario=="original_all" and interval_min==5 and horizon_sec==120')
groups=dict(tuple(select.groupby('case_id')));comparisons=0
for item in vital_records():
    cid=int(item['record_id'])
    if cid not in groups:continue
    v=item['values']
    for r in groups[cid].itertuples():
        start=int(np.floor((r.adjustment_sec+1e-9)/10));end=int(np.floor((r.adjustment_sec+300+1e-9)/10));need=int(r.normal_sec/10)
        found=None;run=0
        for t in range(start,end):
            run=run+1 if np.isfinite(v[t]) and v[t]>=65 else 0
            if run==need:found=t;break
        assert found is not None and (found+1)*10==r.recovery_sec
        actual=brute_state(v,found,30,120)
        np.testing.assert_allclose(actual,[getattr(r,k) for k in metrics],atol=1e-12)
        comparisons+=1
u=pd.read_csv(R/'restricted/distinct_recoveries.csv',dtype={'subject_id':str})
assert not u.duplicated(['scenario','normal_sec','interval_min','horizon_sec','case_id','recovery_sec']).any()
summary=pd.read_csv(R/'aggregate/recovery_summary.csv')
for sec in [20,30,60]:
    g=u.query('scenario=="original_all" and interval_min==5 and horizon_sec==120 and normal_sec==@sec')
    grouped=[z[metrics].to_numpy() for _,z in g.groupby('subject_id',sort=True)]
    rng=np.random.default_rng(20261016);rep=[]
    for _ in range(2000):
        ix=rng.integers(0,len(grouped),len(grouped));rep.append(np.concatenate([grouped[j] for j in ix]).mean(0))
    bounds=np.quantile(rep,[.025,.975],axis=0)
    for j,m in enumerate(metrics):
        r=summary.query('scenario=="original_all" and unit=="distinct_recovery" and weight=="unit_equal" and interval_min==5 and horizon_sec==120 and normal_sec==@sec and metric==@m').iloc[0]
        np.testing.assert_allclose([r.estimate,r.lo,r.hi],[g[m].mean(),bounds[0,j],bounds[1,j]],atol=1e-12)

c=pd.read_csv(R/'restricted/bridge_components.csv.gz')
np.testing.assert_allclose(c.display_auc-c.reference_auc,c.overdisplay_auc-c.hidden_auc,atol=1e-6)
np.testing.assert_allclose(c.episodes,c.fresh+c.inherited_only+c.never_low,atol=1e-8)
support=c[c.scenario.eq('common_complete_minute')]
a=support[support.resolution.eq('10s')].set_index(['source','record_id','interval_min']).sort_index()
b=support[support.resolution.eq('1min')].set_index(['source','record_id','interval_min']).sort_index()
assert a.index.equals(b.index)
np.testing.assert_allclose(a.valid_reference_min,b.valid_reference_min,atol=1e-8)
old=pd.read_csv(P/'跨库分辨率与异质性分析_20261015/aggregate/paired_resolution.csv')
new=pd.read_csv(R/'aggregate/bridge_contrasts.csv')
matched=0
for r in old.query('estimand=="pooled"').itertuples():
    g=new.query('scenario=="original_paired" and source==@r.source and interval_min==@r.interval_min and metric==@r.metric')
    if len(g):
        np.testing.assert_allclose(g.iloc[0].estimate,r.minute_minus_native,atol=1e-10);matched+=1
frozen=json.loads((R/'qa/frozen_inputs.json').read_text())
for p,h in frozen['baseline_sha256'].items():assert sha(Path(frozen['baseline_root'])/p)==h
report=dict(status='PASS',independent_recovery_state_comparisons=comparisons,independent_bootstrap_bounds=36,unique_keys=True,common_finite_support_equal=True,bridge_auc_and_episode_identities=True,legacy_bridge_point_estimates_reproduced=matched,original_files_unchanged=True,raw_waveforms_reread=False,scope='Frozen numerical trajectories and source hashes; no independent clinical adjudication')
(R/'qa/independent_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
