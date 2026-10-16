"""Patient-cluster summaries; pooled and equal-patient estimands kept separate."""
import json,time
from functools import lru_cache
import numpy as np
import engine as e
from io_sources import ROOT,pd,save_json,save_private,freeze_inputs

KEYS=['source','resolution','scenario','threshold','interval_min','event_min_sec']
METRICS=list(e.derive(np.zeros(len(e.COLS))).keys())
EVENT_METRICS=['fresh_probability','inherited_probability','miss_probability','mean_visible_delay_min']
SEED=20260706;BOOT=1000

@lru_cache(maxsize=3)
def weights(n):return np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=BOOT).astype(float)

def interval(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]) if len(a) else (np.nan,np.nan)

def summarize_group(frame,metrics=METRICS,sparse_gate=False):
    patients,c=e.patient_sum(frame.subject_id.to_numpy(),frame[e.COLS].to_numpy(float));n=len(patients)
    w=weights(n);total=c.sum(axis=0);pooled=e.derive(total);draws=e.derive(w@c);individual=e.derive(c)
    count_events=float(total[e.F]);n_event=int((c[:,e.F]>0).sum())
    rows=[];distributions=[]
    for metric in metrics:
        x=np.asarray(individual[metric]);ok=np.isfinite(x);n_defined=int(ok.sum())
        gate_n=min(n_event,n_defined) if sparse_gate else n_defined
        status='CI' if gate_n>=20 and (not sparse_gate or count_events>=50) else ('POINT_ONLY' if gate_n>=10 else 'INSUFFICIENT')
        for estimand in ['pooled','patient_equal']:
            if estimand=='pooled':estimate=float(pooled[metric]);boot=np.asarray(draws[metric])
            else:
                estimate=float(x[ok].mean()) if ok.any() else np.nan
                boot=e.safe_ratio(w@np.where(ok,x,0),w@ok.astype(float))
            lo,hi=interval(boot) if status=='CI' else (np.nan,np.nan)
            if status=='INSUFFICIENT':estimate=np.nan
            rows.append(dict(estimand=estimand,metric=metric,estimate=estimate,lo=lo,hi=hi,status=status,
              n_patients=n,n_defined=n_defined,n_patients_with_events=n_event,n_records=frame.record_id.nunique(),
              events=count_events,reference_auc=total[0],valid_hours=total[10]/60,selected_hours=total[12]/60))
        q=np.quantile(x[ok],[.05,.25,.5,.75,.95]) if ok.any() else np.full(5,np.nan)
        distributions.append(dict(metric=metric,n_patients=n,n_defined=n_defined,**dict(zip(['p05','p25','p50','p75','p95'],q))))
    return rows,distributions

def paired(frame):
    rows=[]
    part=frame[(frame.scenario=='paired_bridge')&(frame.threshold==65)]
    for (source,step),g in part.groupby(['source','interval_min'],sort=True):
        matrices={}
        for res,x in g.groupby('resolution'):
            p,c=e.patient_sum(x.subject_id,x[e.COLS].to_numpy(float));matrices[res]=(p,c)
        p,a=matrices['10s'];q,b=matrices['1min'];assert np.array_equal(p,q)
        assert np.allclose(a[:,12],b[:,12],atol=1e-9)
        w=weights(len(p));ad,bd=e.derive(a),e.derive(b)
        ap,bp=e.derive(a.sum(axis=0)),e.derive(b.sum(axis=0))
        ab,bb=e.derive(w@a),e.derive(w@b)
        for metric in METRICS:
            for estimand in ['pooled','patient_equal_common_defined']:
                if estimand=='pooled':
                    old=float(ap[metric]);new=float(bp[metric]);diff=new-old;boot=bb[metric]-ab[metric];defined=len(p)
                else:
                    ok=np.isfinite(ad[metric])&np.isfinite(bd[metric]);defined=int(ok.sum())
                    old=float(ad[metric][ok].mean()) if ok.any() else np.nan
                    new=float(bd[metric][ok].mean()) if ok.any() else np.nan
                    diff=new-old;delta=np.where(ok,bd[metric]-ad[metric],0)
                    boot=e.safe_ratio(w@delta,w@ok.astype(float))
                status='CI' if defined>=20 else 'POINT_ONLY' if defined>=10 else 'INSUFFICIENT'
                lo,hi=interval(boot) if status=='CI' else (np.nan,np.nan)
                if status=='INSUFFICIENT':old=new=diff=np.nan
                rows.append(dict(source=source,interval_min=step,event_min_sec=180,estimand=estimand,metric=metric,
                    native10s=old,minute=new,minute_minus_native=diff,lo=lo,hi=hi,n_patients=len(p),n_defined_common=defined,status=status))
    return pd.DataFrame(rows)

def standardized(sf,wanted):
    part=sf[(sf.resolution=='1min')&(sf.scenario=='primary')&(sf.feature=='duration')&sf.level.isin(wanted)]
    target=part[part.source=='VitalDB'].groupby('level').episodes.sum().reindex(wanted)
    fixed=(target/target.sum()).to_numpy(float);rows=[]
    for source,g in part.groupby('source',sort=True):
        ids=np.sort(g.subject_id.unique());arrays=[]
        for level in wanted:
            c=g[g.level.eq(level)].groupby('subject_id')[e.ECOLS].sum().reindex(ids,fill_value=0).to_numpy(float)
            arrays.append(c)
        c=np.stack(arrays,axis=1);w=weights(len(ids));total=c.sum(axis=0)
        for metric,j in [('fresh_probability',1),('inherited_probability',2),('miss_probability',3)]:
            estimates=e.safe_ratio(total[:,j],total[:,0]);estimate=float(fixed@estimates)
            draws=np.stack([e.safe_ratio(w@c[:,k,j],w@c[:,k,0]) for k in range(len(wanted))],axis=1)
            boot=draws@fixed;lo,hi=interval(boot)
            rows.append(dict(source=source,metric=metric,estimate=estimate,lo=lo,hi=hi,n_patients=len(ids),
              standard_weights=';'.join(f'{a}:{b:.12g}' for a,b in zip(wanted,fixed)),
              interpretation='conditional on fixed VitalDB duration weights; not a causal database comparison'))
    return pd.DataFrame(rows)

def run():
    start=time.monotonic();freeze_inputs()
    frames=[];strats=[];flows=[]
    for source in ['VitalDB','MIMIC','SICdb']:
        for target,name in [(frames,'components'),(strats,'strata'),(flows,'flow')]:
            target.append(pd.read_csv(ROOT/f'restricted/{source}_{name}.csv.gz',dtype={'record_id':str,'subject_id':str}))
    frame=pd.concat(frames,ignore_index=True);sf=pd.concat(strats,ignore_index=True);flow=pd.concat(flows,ignore_index=True)
    # Compare onset masking against the same records, not the larger primary cohort.
    matched=[]
    for res,g in frame[frame.scenario.eq('core60_fixed')].groupby('resolution'):
        old=frame[frame.source.eq('VitalDB')&frame.resolution.eq(res)&frame.scenario.eq('primary')&
                  frame.threshold.eq(65)&frame.interval_min.eq(5)&frame.record_id.isin(g.record_id)].copy()
        old['scenario']='core60_matched_original';matched.append(old)
    frame=pd.concat([frame,*matched],ignore_index=True)
    rows=[];dist=[]
    for identity,g in frame.groupby(KEYS,sort=True):
        r,d=summarize_group(g);meta=dict(zip(KEYS,identity))
        rows.extend([{**meta,**x} for x in r])
        if identity[2]=='primary' and identity[3]==65 and identity[4]==5:dist.extend([{**meta,**x} for x in d])
    pd.DataFrame(rows).to_csv(ROOT/'aggregate/summary.csv',index=False)
    pd.DataFrame(dist).to_csv(ROOT/'aggregate/patient_distributions.csv',index=False)
    strata=[]
    for identity,g in sf.groupby(['source','resolution','scenario','event_min_sec','feature','level'],sort=True):
        meta=dict(zip(['source','resolution','scenario','event_min_sec','feature','level'],identity))
        for c in e.FCOLS:g[c]=0.
        r,_=summarize_group(g,EVENT_METRICS,True)
        strata.extend([{**meta,**x} for x in r])
    pd.DataFrame(strata).drop(columns=['reference_auc','valid_hours','selected_hours']).to_csv(ROOT/'aggregate/event_strata.csv',index=False)
    paired(frame).to_csv(ROOT/'aggregate/paired_resolution.csv',index=False)
    cohort=[]
    for key,g in flow.groupby(['source','resolution','scenario'],sort=True):
        cohort.append(dict(zip(['source','resolution','scenario'],key))|dict(n_records=g.record_id.nunique(),n_patients=g.subject_id.nunique(),
            selected_hours=g.selected_hours.sum(),valid_hours=g.valid_hours.sum(),
            age25_verified_records=int(g.age_harmonization.eq('age25').sum()),age25_not_independently_verified_records=int(g.age_harmonization.ne('age25').sum())))
    pd.DataFrame(cohort).to_csv(ROOT/'aggregate/cohort_flow.csv',index=False)
    # Duration standardization is gated, never rescued by merging sparse bands.
    ss=pd.DataFrame(strata)
    wanted=['180_to_lt300','300_to_lt600','ge600']
    candidate=ss[(ss.resolution=='1min')&(ss.scenario=='primary')&(ss.feature=='duration')&(ss.estimand=='pooled')&(ss.metric=='fresh_probability')]
    gate=len(candidate[candidate.level.isin(wanted)])==9 and all((candidate[candidate.level.isin(wanted)].n_patients_with_events>=20)&(candidate[candidate.level.isin(wanted)].events>=50))
    save_json(ROOT/'aggregate/standardization_gate.json',dict(pass_gate=bool(gate),
      minimum_patients=20,minimum_events=50,bands=wanted,action='conditional_duration_standardization' if gate else 'not_performed_sparse_shared_stratum',
      notes='No post-result band collapsing. Fixed VitalDB duration weights if gate passes; no pooled database effect or causal test.'))
    if gate:standardized(sf,wanted).to_csv(ROOT/'aggregate/duration_standardized.csv',index=False)
    save_json(ROOT/'qa/summary_completion.json',dict(status='PASS',component_rows=len(frame),stratum_component_rows=len(sf),
      summary_rows=len(rows),strata_summary_rows=len(strata),elapsed_sec=time.monotonic()-start,bootstrap=BOOT,seed=SEED))
    print(json.dumps(dict(stage='summarized',summary_rows=len(rows),strata_rows=len(strata),standardization_gate=bool(gate))),flush=True)

if __name__=='__main__':run()
