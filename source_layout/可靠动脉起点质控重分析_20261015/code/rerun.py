"""Frozen-definition observability rerun on startup-qualified reference windows."""
from pathlib import Path
import sys
import json
import time
import warnings
import numpy as np
from qc_onset import ROOT,PROJECT,sha,write_json
import pandas as pd

ANALYSIS=PROJECT/'analysis_v10_subject_phase_release'
EXT=PROJECT/'EJA_投稿文件包_20261006_最终文字修订/03_可复现材料/extensions/events'
sys.path[:0]=[str(ANALYSIS/'src'),str(EXT)]
from ioh.estimands.decomposition import decompose_deficit
from event_extension import held_display,reference_events,classify_events

COLS=['R','D','H','O','H_unavailable','valid_minutes','unavailable_minutes',
      'reference_low_unavailable_min','lowref_normaldisplay_min','normalref_lowdisplay_min',
      'reference_low_min','display_low_min','window_minutes','episodes','fresh','inherited_only','never_low','delay_sum_min']
KEYS=['resolution','scenario','initialization','threshold','interval_min','minimum_event_sec']


def trimmed_values(v,onset_sec):
    out=np.asarray(v,float).copy();out[:int(onset_sec//10)]=np.nan;return out


def phase_components(v,selected,threshold,step,dt,min_bins,initialize=False,init_zero=False,return_phases=False):
    ref=np.where(selected,v,np.nan);ev=reference_events(ref,threshold,0,min_bins)
    first=np.flatnonzero(np.isfinite(v));first=int(first[0]) if len(first) else -1
    index=np.arange(len(v));rows=[]
    for phase in range(step):
        d,held=held_display(v,step,phase)
        if initialize and (init_zero or phase>0) and first>=0:
            fill=(index>=first)&(held<0);held[fill]=first;d[fill]=v[first]
        f=decompose_deficit(ref,d,threshold,dt)
        result=classify_events(ref,d,held,ev,threshold)
        counts=np.bincount(result['category'],minlength=3)
        low=np.isfinite(d)&(d<threshold)
        delay=sum(int(np.argmax(low[a:b]))*dt for a,b,_ in ev if low[a:b].any())
        c=[f['true_auc'],f['display_auc'],f['hidden_auc'],f['overdisplay_auc'],
           f['hidden_auc_display_unavailable'],np.isfinite(ref).sum()*dt,f['display_unavailable_min'],
           f['reference_hypotension_with_display_unavailable_min'],f['normotensive_display_min'],
           f['hypotensive_display_discordance_min'],f['true_hypotension_min'],
           (np.isfinite(ref)&low).sum()*dt,selected.sum()*dt,len(ev),*counts,delay]
        if not np.isclose(c[1]-c[0],c[3]-c[2],rtol=1e-11,atol=1e-7):raise AssertionError('AUC identity')
        if c[13]!=sum(c[14:17]):raise AssertionError('Event partition')
        rows.append(c)
    out=np.asarray(rows,float)
    return (dict(zip(COLS,out.mean(axis=0))),out) if return_phases else dict(zip(COLS,out.mean(axis=0)))


def components_to_metrics(a):
    a=np.asarray(a,float);z={c:a[...,j] for j,c in enumerate(COLS)}
    def ratio(x,y):
        x,y=np.broadcast_arrays(x,y);return np.divide(x,y,out=np.full(x.shape,np.nan),where=y>0)
    r,d,h,o=[z[c] for c in ['R','D','H','O']];e=z['episodes'];t=z['valid_minutes']
    return dict(hidden_ratio=ratio(h,r),overdisplay_ratio=ratio(o,r),net_ratio=ratio(d-r,r),
        hidden_unavailable_ratio=ratio(z['H_unavailable'],r),reference_twa=ratio(r,t),display_twa=ratio(d,t),net_twa=ratio(d-r,t),
        fresh_probability=ratio(z['fresh'],e),inherited_probability=ratio(z['inherited_only'],e),miss_probability=ratio(z['never_low'],e),
        lowref_normaldisplay_min_per_hour=ratio(z['lowref_normaldisplay_min']*60,z['window_minutes']),
        normalref_lowdisplay_min_per_hour=ratio(z['normalref_lowdisplay_min']*60,z['window_minutes']),
        unavailable_fraction=ratio(z['unavailable_minutes'],t),
        reference_low_unavailable_min_per_hour=ratio(z['reference_low_unavailable_min']*60,z['window_minutes']),
        display_delay_visible_min=ratio(z['delay_sum_min'],e-z['never_low']))


def minute_values(v,n):
    x=v[:n*6].reshape(n,6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);out=np.nanmedian(x,axis=1)
    out[np.isfinite(x).sum(axis=1)<5]=np.nan
    return out


def select_hours(v):
    n=len(v)//60;out=np.zeros(len(v),bool)
    out[:n*60]=np.repeat(np.isfinite(v[:n*60]).reshape(n,60).sum(axis=1)>=48,60)
    return out


def run():
    began=time.monotonic();contract_hash=sha(ROOT/'analysis_contract.json')
    src=ANALYSIS/'outputs/intermediate';panel=pd.read_parquet(src/'artmap_10s.parquet')
    meta=pd.read_parquet(src/'vitaldb_manifest.parquet').set_index('case_id')
    onset=pd.read_csv(ROOT/'restricted/onsets.csv').set_index('case_id')
    bins=pd.read_csv(ROOT/'restricted/startup_quality_bins.csv.gz')
    bad={int(c):g.loc[~g.core_good,'time_sec'].to_numpy(int)//10 for c,g in bins.groupby('case_id')}
    rows=[];ranges=[]
    for number,(cid,g) in enumerate(panel.groupby('case_id',sort=True),1):
        cid=int(cid);original=g.sort_values('time_sec').art_map.to_numpy(float)
        index=np.arange(len(original));sid=int(meta.loc[cid,'subjectid']);row=onset.loc[cid]
        specs=[('original_all',original,np.ones(len(original),bool))]
        if row.waveform_available:specs.append(('original_waveform_available',original,np.ones(len(original),bool)))
        primary=bool(row.primary_pass)
        if primary:specs.append(('original_matched_primary_onset',original,np.ones(len(original),bool)))
        for name,col in [('primary_core60_onset','primary_onset_sec'),('core30_onset','core30_onset_sec'),
                         ('core120_onset','core120_onset_sec'),('relaxed60_onset','relaxed60_onset_sec')]:
            if np.isfinite(row[col]):
                v=trimmed_values(original,row[col]);selected=index>=int(row[col]//10)
                specs.append((name,v,selected))
        if primary:
            k=int(row.primary_onset_sec//10);v=trimmed_values(original,row.primary_onset_sec+60)
            specs.append(('core60_confirmation_end',v,index>=k+6))
            v=trimmed_values(original,row.primary_onset_sec);ix=bad.get(cid,np.array([],int));ix=ix[(ix>=k)&(ix<len(v))]
            v[ix]=np.nan;specs.append(('core60_mask_later_failed_startup_bins',v,index>=k))
        for scenario,v,selected in specs:
            broad=scenario in ['original_all','primary_core60_onset']
            configs=[(65,x,60) for x in ([1,2.5,3,5,10] if broad else [5])]
            if broad:configs += [(x,5,60) for x in [60,55]]
            if scenario in ['original_matched_primary_onset','primary_core60_onset']:
                configs += [(65,5,x) for x in [120,240]]
            for threshold,interval,minimum in configs:
                for init in ([False,True] if threshold==65 and interval==5 and minimum==60 else [False]):
                    c,pp=phase_components(v,selected,threshold,int(interval*6),1/6,int(minimum/10),init,False,True)
                    rows.append(dict(resolution='native10s',scenario=scenario,initialization='first_valid_seed_nonzero_phase' if init else 'unavailable',
                        threshold=threshold,interval_min=interval,minimum_event_sec=minimum,case_id=cid,subject_id=sid,**c))
                    if not init and threshold==65 and interval==5 and minimum==60:
                        mm=components_to_metrics(pp);d={}
                        for key in ['hidden_ratio','overdisplay_ratio','net_ratio','miss_probability','fresh_probability']:
                            x=mm[key];x=x[np.isfinite(x)];d[key+'_phase_range']=float(x.max()-x.min()) if len(x) else np.nan
                        cm=components_to_metrics(np.array([c[q] for q in COLS]));d.update({key:float(cm[key]) for key in ['hidden_ratio','overdisplay_ratio','net_ratio','miss_probability','fresh_probability']})
                        ranges.append(dict(scenario=scenario,case_id=cid,subject_id=sid,**d))
        if primary and float(meta.loc[cid,'age'])>=25:
            n=int(np.floor((float(meta.loc[cid,'primary_end_sec'])-float(meta.loc[cid,'primary_start_sec']))/60))
            ov=minute_values(original,n);selected=select_hours(ov)
            if selected.any():
                qv=minute_values(trimmed_values(original,row.primary_onset_sec),n)
                for label,v in [('minute_original_matched_fixed_hours',ov),('minute_core60_fixed_hours',qv)]:
                    for init in [False,True]:
                        c=phase_components(v,selected,65,5,1.,3,init,True)
                        rows.append(dict(resolution='minute',scenario=label,initialization='first_valid_seed_all_phases' if init else 'unavailable',
                            threshold=65,interval_min=5,minimum_event_sec=180,case_id=cid,subject_id=sid,**c))
        if number%300==0:print(json.dumps({'rerun_cases_completed':number}),flush=True)
    frame=pd.DataFrame(rows)
    for name,f in [('case_components.csv.gz',frame),('case_phase_distributions.csv.gz',pd.DataFrame(ranges))]:
        p=ROOT/'restricted'/name;f.to_csv(p,index=False);p.chmod(0o600)
    point=frame[(frame.scenario=='original_all')&(frame.initialization=='unavailable')&(frame.threshold==65)&(frame.interval_min==5)&(frame.minimum_event_sec==60)][COLS].sum()
    frozen=pd.read_csv(ANALYSIS/'outputs/tables/initial_display_policy_sensitivity.csv').iloc[0]
    for new,old in zip(['R','D','H','O'],['true_auc_total','display_auc_total','hidden_auc_total','overdisplay_auc_total']):
        assert np.isclose(point[new],frozen[old],rtol=1e-12,atol=1e-6),(new,point[new],frozen[old])
    assert point.episodes==9187
    results=[];distributions=[]
    for keys,g in frame.groupby(KEYS,sort=False):
        by=g.groupby('subject_id')[COLS].sum().sort_index();n=len(by)
        weights=np.random.default_rng(20260706).multinomial(n,np.full(n,1/n),size=1000)
        total=by.sum().to_numpy(float);draw=weights@by.to_numpy(float)
        m=components_to_metrics(total);bm=components_to_metrics(draw)
        out=dict(zip(KEYS,keys));out.update(n_cases=len(g),n_subjects=n,n_cases_with_events=int(g.episodes.gt(0).sum()),**dict(zip(COLS,total)))
        for key,val in m.items():
            vals=bm[key];vals=vals[np.isfinite(vals)];ci=np.quantile(vals,[.025,.975]) if len(vals) else [np.nan,np.nan]
            out.update({key:float(val),key+'_lo':float(ci[0]),key+'_hi':float(ci[1])})
        results.append(out)
        if keys[0]=='native10s' and keys[2]=='unavailable' and keys[3]==65 and keys[4]==5 and keys[5]==60:
            cm=components_to_metrics(g[COLS].to_numpy(float))
            for key in ['hidden_ratio','overdisplay_ratio','net_ratio','miss_probability','fresh_probability']:
                x=cm[key];x=x[np.isfinite(x)]
                if len(x):distributions.append(dict(scenario=keys[1],metric=key,n_defined=len(x),**dict(zip(['p05','p25','p50','p75','p95'],np.quantile(x,[.05,.25,.5,.75,.95])))))
    summary=pd.DataFrame(results);summary.to_csv(ROOT/'aggregate/observability_summary.csv',index=False)
    pd.DataFrame(distributions).to_csv(ROOT/'aggregate/case_distributions.csv',index=False)
    rr=pd.DataFrame(ranges);rd=[]
    for name,g in rr.groupby('scenario'):
        for key in [q for q in rr.columns if q.endswith('_phase_range')]:
            x=g[key].dropna()
            if len(x):rd.append(dict(scenario=name,metric=key,n_defined=len(x),**dict(zip(['p25','p50','p75','p95'],np.quantile(x,[.25,.5,.75,.95])))))
    pd.DataFrame(rd).to_csv(ROOT/'aggregate/within_case_phase_ranges.csv',index=False)
    paired=[]
    for init in ['unavailable','first_valid_seed_nonzero_phase']:
        part=frame[(frame.resolution=='native10s')&(frame.threshold==65)&(frame.interval_min==5)&(frame.minimum_event_sec==60)&(frame.initialization==init)]
        old=part[part.scenario=='original_matched_primary_onset'].groupby('subject_id')[COLS].sum().sort_index()
        new=part[part.scenario=='primary_core60_onset'].groupby('subject_id')[COLS].sum().reindex(old.index)
        assert new.notna().all().all()
        n=len(old);w=np.random.default_rng(20260706).multinomial(n,np.full(n,1/n),size=1000)
        om,nm=components_to_metrics(old.sum().to_numpy(float)),components_to_metrics(new.sum().to_numpy(float))
        ob,nb=components_to_metrics(w@old.to_numpy(float)),components_to_metrics(w@new.to_numpy(float))
        for key in om:
            x=nb[key]-ob[key];ci=np.nanquantile(x,[.025,.975]);paired.append(dict(initialization=init,metric=key,
                n_subjects=n,old_estimate=float(om[key]),new_estimate=float(nm[key]),difference=float(nm[key]-om[key]),lo=float(ci[0]),hi=float(ci[1])))
    pd.DataFrame(paired).to_csv(ROOT/'aggregate/paired_window_changes.csv',index=False)
    assert sha(ROOT/'analysis_contract.json')==contract_hash
    sources=json.loads((ROOT/'qa/protected_input_hashes.json').read_text())
    assert all(sha(p)==h for p,h in sources.items())
    write_json(ROOT/'qa/rerun_verification.json',dict(status='PASS',elapsed_sec=time.monotonic()-began,
        contract_sha256=contract_hash,original_point_estimates_reproduced=True,original_episodes_reproduced=9187,
        summary_groups=len(summary),case_component_rows=len(frame),original_sources_unchanged=True,
        frozen_estimators={str(p.relative_to(PROJECT)):sha(p) for p in [ANALYSIS/'src/ioh/estimands/decomposition.py',EXT/'event_extension.py']}))
    focus=summary[(summary.threshold==65)&(summary.interval_min==5)&(summary.minimum_event_sec==60)]
    print(focus[['scenario','initialization','n_cases','H','O','hidden_ratio','overdisplay_ratio','net_ratio','episodes','fresh_probability','inherited_probability','miss_probability']].to_json(orient='records'),flush=True)


if __name__=='__main__':run()
