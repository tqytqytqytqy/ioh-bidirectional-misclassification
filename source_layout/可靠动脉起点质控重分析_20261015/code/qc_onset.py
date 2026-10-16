"""Retrospective composite technical qualification; no clinical adjudication."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'SICdb分钟级对齐分析_20261015/private/deps'))
sys.dont_writebytecode=True
import pandas as pd

RAW=Path(__import__('os').environ.get('VITALDB_RAW_ROOT', 'data/vitaldb'))
TRACK_ROOTS=[Path(x) for x in __import__('os').environ.get('VITALDB_TRACK_ROOTS',str(RAW/'tracks')).split(__import__('os').pathsep)]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def locate(tid):
    if pd.isna(tid):return None
    for root in TRACK_ROOTS:
        p=root/(str(tid)+'.csv.gz')
        if p.is_file():return p
    return None


def numeric_grid(times,values,start,count,low=0,high=300):
    t=np.asarray(times,float);v=np.asarray(values,float);out=np.full(count,np.nan)
    ok=np.isfinite(t)&np.isfinite(v)&(v>=low)&(v<=high)
    bins=np.floor((t[ok]-start)/10).astype(int);keep=(bins>=0)&(bins<count)
    if keep.any():
        g=pd.DataFrame({'bin':bins[keep],'v':v[ok][keep]}).groupby('bin').v.median()
        out[g.index.to_numpy(int)]=g.to_numpy(float)
    return out


def flags(f,relaxed=False):
    n=len(f);m=f.numeric_map.to_numpy(float)
    mean=f.wave_mean.to_numpy(float);span=f.wave_span_90pct.to_numpy(float)
    s=f.sbp.to_numpy(float) if 'sbp' in f else np.full(n,np.nan)
    d=f.dbp.to_numpy(float) if 'dbp' in f else np.full(n,np.nan)
    paired=np.isfinite(s)&np.isfinite(d)
    coherent=(s>=30)&(s<=250)&(d>=5)&(d<=150)&(s-d>=5)&(s-d<=160)&(m>=d-5)&(m<=s+5)
    return dict(numeric_range=np.isfinite(m)&(m>=20)&(m<=180),
        coverage=f.wave_coverage.ge(.8 if relaxed else .9).to_numpy(),
        waveform_mean=np.isfinite(mean)&(mean>=20)&(mean<=180),
        waveform_range=(f.wave_p05.ge(-5 if relaxed else 0)&f.wave_p95.le(300 if relaxed else 250)).to_numpy(),
        pulsatility=np.isfinite(span)&(span>=(5 if relaxed else 8))&(span<=(180 if relaxed else 160))&f.peak_count.between(3 if relaxed else 4,40).to_numpy(),
        numeric_wave_agreement=np.abs(mean-m)<=(15 if relaxed else 10),
        optional_pair_coherence=~paired|coherent)


def bin_gate(f,relaxed=False):
    return np.logical_and.reduce(list(flags(f,relaxed).values()))


def first_run(mask,bins):
    x=np.asarray(mask,bool)
    if len(x)<bins:return None
    hit=np.flatnonzero(np.convolve(x.astype(int),np.ones(bins,int),'valid')==bins)
    return int(hit[0]) if len(hit) else None


def write_json(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def build():
    for name in ['restricted','aggregate','qa','交付文件']:(ROOT/name).mkdir(exist_ok=True)
    (ROOT/'restricted').chmod(0o700)
    contract=ROOT/'analysis_contract.json';contract_hash=sha(contract)
    settings=json.loads(contract.read_text())
    src=PROJECT/'analysis_v10_subject_phase_release/outputs/intermediate'
    cache=PROJECT/'启动段质量审计_20261007/private_case_audit/full_waveform_10s_metrics.csv'
    audit=cache.parent/'startup_numeric_case_audit.csv'
    manuscript=PROJECT/'EJA_VitalDB单数据库修订版_20261015/01_Main_Manuscript_EJA.docx'
    inputs=[src/'artmap_10s.parquet',src/'vitaldb_manifest.parquet',cache,audit,RAW/'trks.csv',manuscript]
    hashes={str(p):sha(p) for p in inputs}
    assert hashes[str(inputs[0])]==settings['base_map_panel_sha256']
    assert hashes[str(inputs[1])]==settings['base_metadata_sha256']
    assert hashes[str(manuscript)]==settings['base_manuscript_sha256']
    write_json(ROOT/'qa/protected_input_hashes.json',hashes)
    meta=pd.read_parquet(inputs[1]).set_index('case_id');panel=pd.read_parquet(inputs[0])
    waves=pd.read_csv(cache);waves=waves.sort_values(['case_id','time_sec'])
    grouped={int(c):g.copy().reset_index(drop=True) for c,g in waves.groupby('case_id')}
    source=pd.read_csv(audit).set_index('case_id');trks=pd.read_csv(RAW/'trks.csv')
    lookup=trks.set_index(['caseid','tname']).tid.to_dict()
    case_rows=[];allbins=[];sources=[];cohorts=[]
    for i,(cid,g) in enumerate(panel.groupby('case_id',sort=True),1):
        cid=int(cid);v=g.sort_values('time_sec').art_map.to_numpy(float);r=meta.loc[cid]
        first=int(np.flatnonzero(np.isfinite(v))[0]);sid=int(r.subjectid)
        row=dict(case_id=cid,subject_id=sid,first_finite_sec=first*10,first_map=float(v[first]),
                 waveform_available=cid in grouped,primary_onset_sec=np.nan,core30_onset_sec=np.nan,
                 core120_onset_sec=np.nan,relaxed60_onset_sec=np.nan,first_pair_available=False)
        if cid in grouped:
            w=grouped[cid]
            assert len(w)==60 and np.array_equal(w.time_sec,np.arange(first*10,first*10+600,10))
            ids=(w.time_sec.to_numpy(int)//10);valididx=ids<len(v)
            check=np.full(len(w),np.nan);check[valididx]=v[ids[valididx]]
            if not np.allclose(w.numeric_map,check,equal_nan=True,atol=1e-10,rtol=0):raise ValueError('Wave cache/reference mismatch')
            device=str(r.art_track_name).split('/')[0]
            for label,name in [('sbp','ART_SBP'),('dbp','ART_DBP')]:
                tid=lookup.get((cid,device+'/'+name));p=locate(tid)
                a=np.full(len(v),np.nan)
                if p is not None:
                    raw=pd.read_csv(p)
                    a=numeric_grid(pd.to_numeric(raw.iloc[:,0],errors='coerce'),pd.to_numeric(raw.iloc[:,1],errors='coerce'),
                                   float(r.primary_start_sec),len(v))
                    sources.append(dict(case_id=cid,role=label,path=str(p),sha256=sha(p)))
                w[label]=np.nan;w.loc[valididx,label]=a[ids[valididx]]
            core=bin_gate(w);relax=bin_gate(w,True)
            w['core_good']=core;w['relaxed_good']=relax
            for k,x in flags(w).items():w['flag_'+k]=x
            row['first_pair_available']=bool(w.sbp.notna().iloc[0] and w.dbp.notna().iloc[0])
            for key,mask,n in [('primary',core,6),('core30',core,3),('core120',core,12),('relaxed60',relax,6)]:
                onset=first_run(mask,n)
                if onset is not None:row[key+'_onset_sec']=float(w.time_sec.iloc[onset])
            row.update(core_qualifying_bins=int(core.sum()),relaxed_qualifying_bins=int(relax.sum()),
                       paired_bins=int((w.sbp.notna()&w.dbp.notna()).sum()))
            allbins.append(w)
        row['primary_pass']=bool(np.isfinite(row['primary_onset_sec']))
        row['failure_reason']='' if row['primary_pass'] else ('no_source_waveform' if not row['waveform_available'] else 'no_60s_core_run_in_startup_window')
        for key in ['primary','core30','core120','relaxed60']:
            start=row[key+'_onset_sec']
            if np.isfinite(start):
                k=int(start//10);z=v[k:];ref=np.maximum(65-z,0)
                cohorts.append(dict(rule=key,case_id=cid,subject_id=sid,onset_sec=start,delay_sec=start-first*10,
                    initial_map=float(z[0]),retained_valid_minutes=float(np.isfinite(z).sum()/6),
                    retained_duration_minutes=float(len(z)/6),retained_coverage=float(np.isfinite(z).mean()),
                    removed_reference_auc=float(np.nansum(np.maximum(65-v[:k],0))/6),
                    retained_reference_auc=float(np.nansum(ref)/6)))
        case_rows.append(row)
        if i%400==0:print(json.dumps({'startup_cases_completed':i}),flush=True)
    cases=pd.DataFrame(case_rows);bins=pd.concat(allbins,ignore_index=True);cohorts=pd.DataFrame(cohorts)
    for name,f in [('onsets.csv',cases),('startup_quality_bins.csv.gz',bins),('cohort_windows.csv',cohorts),('numeric_source_hashes.csv',pd.DataFrame(sources))]:
        p=ROOT/'restricted'/name;f.to_csv(p,index=False);p.chmod(0o600)
    flow=[dict(stage='original_cohort',cases=len(cases),subjects=cases.subject_id.nunique()),
          dict(stage='waveform_available',cases=int(cases.waveform_available.sum()),subjects=cases[cases.waveform_available].subject_id.nunique())]
    summaries=[]
    for rule,g in cohorts.groupby('rule',sort=False):
        flow.append(dict(stage=rule,cases=len(g),subjects=g.subject_id.nunique()))
        x=dict(rule=rule,cases=len(g),subjects=g.subject_id.nunique(),
            onset_MAP_lt40_cases=int(g.initial_map.lt(40).sum()),onset_MAP_lt65_cases=int(g.initial_map.lt(65).sum()),
            removed_reference_auc=float(g.removed_reference_auc.sum()),retained_reference_auc=float(g.retained_reference_auc.sum()),
            retained_valid_hours=float(g.retained_valid_minutes.sum()/60),postonset_below80percent_cases=int(g.retained_coverage.lt(.8).sum()))
        for col in ['delay_sec','initial_map','retained_duration_minutes','retained_coverage']:
            for name,q in [('min',0),('p25',.25),('p50',.5),('p75',.75),('p95',.95),('max',1)]:x[col+'_'+name]=float(g[col].quantile(q))
        summaries.append(x)
    pd.DataFrame(flow).to_csv(ROOT/'aggregate/cohort_flow.csv',index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'aggregate/onset_summary.csv',index=False)
    failures=cases.groupby('failure_reason').agg(cases=('case_id','size'),subjects=('subject_id','nunique')).reset_index()
    failures.to_csv(ROOT/'aggregate/qualification_failures.csv',index=False)
    reasons=[]
    for col in bins.filter(regex='^flag_').columns:
        reasons.append(dict(gate_component=col[5:],bins=len(bins),failed_bins=int((~bins[col]).sum()),
                            records_with_failed_bin=int(bins.loc[~bins[col],'case_id'].nunique())))
    pd.DataFrame(reasons).to_csv(ROOT/'aggregate/bin_gate_summary.csv',index=False)
    assert sha(contract)==contract_hash
    assert all(sha(p)==h for p,h in hashes.items())
    write_json(ROOT/'qa/onset_build_verification.json',dict(status='PASS',contract_sha256=contract_hash,
        source_cache_rows=len(bins),source_cache_records=bins.case_id.nunique(),
        source_panel_aligned=True,original_inputs_unchanged=True,
        waveform_missing_cases=int((~cases.waveform_available).sum()),raw_numeric_sources_checked=len(sources)))
    print(json.dumps({'status':'PASS','flow':flow,'contract_sha256':contract_hash},ensure_ascii=False),flush=True)


if __name__=='__main__':build()
