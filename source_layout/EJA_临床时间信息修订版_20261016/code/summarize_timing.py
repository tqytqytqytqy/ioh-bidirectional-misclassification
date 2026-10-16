from pathlib import Path
import sys,json
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parent
sys.path.insert(0,str(P/'SICdb分钟级对齐分析_20261015/private/deps'))
import pandas as pd
from timing import FRESH,RECOVERY

def estimate(g,metrics,meta):
    a=g.groupby('subject_id')[metrics].sum()
    n=g.groupby('subject_id').size().reindex(a.index).to_numpy()
    sums=a.to_numpy();means=sums/n[:,None]
    rng=np.random.default_rng(20261015)
    event=[];patient=[]
    for _ in range(20):
        ix=rng.integers(0,len(n),size=(100,len(n)))
        event.append(sums[ix].sum(axis=1)/n[ix].sum(axis=1)[:,None])
        patient.append(means[ix].mean(axis=1))
    gate='CI' if len(a)>=20 and len(g)>=50 else 'point_only' if len(a)>=10 else 'suppressed'
    rows=[]
    for w,point,reps in [('event_equal',sums.sum(0)/n.sum(),event),('patient_equal',means.mean(0),patient)]:
        lo,hi=np.quantile(np.concatenate(reps),[.025,.975],axis=0)
        for j,m in enumerate(metrics):
            rows.append(dict(meta,weight=w,metric=m,estimate=float(point[j]) if gate!='suppressed' else None,lo=float(lo[j]) if gate=='CI' else None,hi=float(hi[j]) if gate=='CI' else None,patients=len(a),cases=g.case_id.nunique(),events=len(g),status=gate))
    return rows

def main():
    output=[];contrasts=[]
    for module,metrics in [('fresh',FRESH+['unavailable_during_wait','fresh_lower','fresh_upper','not_fresh_confirmed']),('recovery',RECOVERY+['persistent_lower','persistent_upper'])]:
        f=pd.read_csv(R/'restricted'/f'{module}_events.csv')
        groups=[('confirmed_onset',f[f.confirmed])] if module=='fresh' else [('all_recoveries',f)]
        if module=='fresh':
            groups += [('all_boundary_included',f),('nadir_lt55',f[f.confirmed&f.nadir.lt(55)]),('nadir_55to65',f[f.confirmed&f.nadir.ge(55)])]
            for low,high,label in [(60,120,'duration60to119'),(120,300,'duration120to299'),(300,np.inf,'duration_ge300')]:
                groups.append((label,f[f.confirmed&f.duration_sec.ge(low)&f.duration_sec.lt(high)]))
        for population,frame in groups:
            for (s,n,h),g in frame.groupby(['scenario','interval_min','horizon_sec']):
                meta=dict(module=module,population=population,scenario=s,interval_min=n,horizon_sec=h)
                output.extend(estimate(g,metrics,meta))
            for (s,h),g in frame.groupby(['scenario','horizon_sec']):
                key=['subject_id','case_id','event_start' if module=='fresh' else 'adjustment_sec']
                a=g[g.interval_min.eq(1)].set_index(key);b=g[g.interval_min.eq(5)].set_index(key)
                assert a.index.equals(b.index)
                delta=b[metrics]-a[metrics];delta=delta.reset_index()
                meta=dict(module=module,population=population,scenario=s,interval_min='5_minus_1',horizon_sec=h)
                contrasts.extend(estimate(delta,metrics,meta))
    result=pd.DataFrame(output);delta=pd.DataFrame(contrasts)
    result.to_csv(R/'aggregate/timing_summary.csv',index=False)
    delta.to_csv(R/'aggregate/timing_paired_differences.csv',index=False)
    main=result.query("scenario=='original_all' and weight=='event_equal' and population in ['confirmed_onset','all_recoveries']")
    main.to_csv(R/'aggregate/primary_timing.csv',index=False)
    r=main.query("module=='recovery' and interval_min==5 and horizon_sec==120 and metric=='persistent_low'").iloc[0]
    decision={'treatment_main_figure_gate_pass':bool(r.patients>=30 and r.hi-r.lo<=.20),'patients':int(r.patients),'ci_width_percentage_points':float(100*(r.hi-r.lo)),'fresh_exact_discrete_baseline':True,'fresh_placement':'supplement with concise main-text result; no standalone novelty claim','MIMIC_new_timing_extension':'not added; optional under approved plan; existing full cross-database results preserved','unknown_over10percent':main.query("metric=='unknown' and estimate>.10").to_dict('records')}
    (R/'qa/presentation_decision.json').write_text(json.dumps(decision,indent=2))
    print(main.query("interval_min in [1,5] and metric in ['fresh','persistent_low','unknown','reference_recurred']").to_string(index=False))
    print(json.dumps(decision))
if __name__=='__main__':main()
