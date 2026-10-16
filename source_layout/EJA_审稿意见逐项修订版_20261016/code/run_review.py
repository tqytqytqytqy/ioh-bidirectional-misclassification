from pathlib import Path
import sys,json,hashlib,time
sys.dont_write_bytecode=True
import numpy as np
R=Path(__file__).resolve().parents[1];P=R.parent
B=P/'EJA_临床时间信息修订版_20261016'
sys.path.insert(0,str(P/'跨库分辨率与异质性分析_20261015/code'))
from io_sources import vital_records,mimic_records,subgroup_metadata,pd,sha
import engine as e
sys.path.insert(0,str(B/'code'))
from timing import display_matrix,recovery_states,RECOVERY,confirmed_recovery
sys.path.insert(0,str(P/'治疗相关事件补充修订_20261004/code'))
from treatment_core import event_state
from review_core import sustained_recovery,common_support

def save(obj,p):
    Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def freeze():
    for folder in ['aggregate','restricted','qa']:(R/folder).mkdir(exist_ok=True)
    (R/'restricted').chmod(0o700)
    original={str(p.relative_to(B)):sha(p) for p in B.rglob('*') if p.is_file() and not p.is_symlink() and 'node_modules' not in p.parts and '__pycache__' not in p.parts and p.name!='.DS_Store'}
    paths=[P/'analysis_v10_subject_phase_release/outputs/intermediate/artmap_10s.parquet',P/'analysis_v10_subject_phase_release/outputs/intermediate/vitaldb_manifest.parquet',P/'可靠动脉起点质控重分析_20261015/restricted/onsets.csv',P/'独立质控重分析_20261007/data_restricted/treatment_original_primary_events.csv',P/'MIMIC技术重复分析_20261015/private/qc_inputs.npz',P/'MIMIC技术重复分析_20261015/private/records.json',R/'WORK_LOG.md']
    frozen=dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),baseline_root=str(B),baseline_sha256=original,input_sha256={str(p):sha(p) for p in paths},seed=20261016,recovery_bootstrap=2000,bridge_bootstrap=1000,new_estimates_started=False)
    target=R/'qa/frozen_inputs.json'
    if target.exists():
        existing=json.loads(target.read_text());assert existing['baseline_sha256']==original
        for p,h in existing['input_sha256'].items():assert sha(p)==h
        return existing
    save(frozen,target);return frozen

def private(df,name):
    path=R/'restricted'/name;df.to_csv(path,index=False);path.chmod(0o600)

def estimate(g,metrics,meta,reps=2000):
    a=g.groupby('subject_id',sort=True)[metrics].sum()
    n=g.groupby('subject_id',sort=True).size().reindex(a.index).to_numpy()
    x=a.to_numpy();means=x/n[:,None];rng=np.random.default_rng(20261016)
    out=[];re=[];rp=[]
    for k in range(0,reps,100):
        ix=rng.integers(0,len(n),(min(100,reps-k),len(n)))
        re.append(x[ix].sum(1)/n[ix].sum(1)[:,None]);rp.append(means[ix].mean(1))
    for weight,point,bs in [('unit_equal',x.sum(0)/n.sum(),re),('patient_equal',means.mean(0),rp)]:
        lo,hi=np.quantile(np.concatenate(bs),[.025,.975],axis=0)
        for j,m in enumerate(metrics):out.append(dict(meta,weight=weight,metric=m,estimate=float(point[j]),lo=float(lo[j]),hi=float(hi[j]),patients=len(a),cases=int(g.case_id.nunique()),units=len(g)))
    return out

def recoveries():
    tc=pd.read_csv(P/'独立质控重分析_20261007/data_restricted/treatment_original_primary_events.csv')
    groups=dict(tuple(tc.groupby('case_id')));rows=[];flow=[];boundary=[];old_match=0
    panel=P/'analysis_v10_subject_phase_release/outputs/intermediate'
    full=pd.read_parquet(panel/'artmap_10s.parquet')
    full_valid=int(np.isfinite(full.art_map).sum());full_low=int(full.art_map.lt(65).sum())
    for i,item in enumerate(vital_records(),1):
        cid=int(item['record_id']);v=item['values'];q=item['onset']
        boundary.append([np.isfinite(v).sum(),(v<65).sum()])
        if cid not in groups:continue
        specs=[('original_all',v)]
        if q['primary_pass']:
            masked=v.copy();masked[:int(q['primary_onset_sec']//10)]=np.nan
            specs += [('matched_original',v),('core60',masked)]
        for scenario,ref in specs:
            held={n:display_matrix(ref,n) for n in [6,18,30]}
            low=0;count={k:0 for k in [20,30,60]}
            for row in groups[cid].itertuples():
                st=event_state(ref,row.time_sec,30,0)
                if row.time_sec<300 or st is None or st['reference']>=65:continue
                low+=1
                for sec in [20,30,60]:
                    ri=sustained_recovery(ref,row.time_sec,sec)
                    if sec==20:
                        assert ri==confirmed_recovery(ref,row.time_sec)
                        old_match+=1
                    if ri is None:continue
                    count[sec]+=1
                    base=dict(scenario=scenario,subject_id=item['subject_id'],case_id=cid,adjustment_sec=float(row.time_sec),normal_sec=sec,recovery_sec=(ri+1)*10)
                    for n in [6,18,30]:
                        for h in [60,120]:
                            z=recovery_states(ref,ri,n,h,held[n]);assert abs(sum(z[k] for k in RECOVERY)-1)<1e-12
                            rows.append(dict(base,interval_min=n/6,horizon_sec=h,**z))
            if low:flow.append(dict(scenario=scenario,subject_id=item['subject_id'],case_id=cid,low_adjustments=low,**{f'pairs_{k}':count[k] for k in count}))
    f=pd.DataFrame(rows);private(f,'recovery_pairs.csv');private(pd.DataFrame(flow),'recovery_flow.csv')
    keys=['scenario','normal_sec','interval_min','horizon_sec'];unique=f.sort_values(['case_id','recovery_sec','adjustment_sec']).drop_duplicates(keys+['case_id','recovery_sec'])
    private(unique,'distinct_recoveries.csv')
    results=[];contrasts=[];audit=[];metrics=RECOVERY+['persistent_lower','persistent_upper']
    for unit,frame in [('adjustment_pair',f),('distinct_recovery',unique)]:
        for key,g in frame.groupby(keys):results+=estimate(g,metrics,dict(zip(keys,key),unit=unit))
        for key,g in frame.groupby(['scenario','normal_sec','horizon_sec']):
            index=['subject_id','case_id','adjustment_sec'] if unit=='adjustment_pair' else ['subject_id','case_id','recovery_sec']
            a=g[g.interval_min.eq(1)].set_index(index).sort_index();b=g[g.interval_min.eq(5)].set_index(index).sort_index()
            assert a.index.equals(b.index)
            delta=(b[metrics]-a[metrics]).reset_index()
            contrasts+=estimate(delta,metrics,dict(zip(['scenario','normal_sec','horizon_sec'],key),unit=unit,interval_min='5_minus_1'))
    for (scenario,sec),g in f.query('interval_min==5 and horizon_sec==120').groupby(['scenario','normal_sec']):
        u=g.drop_duplicates(['case_id','recovery_sec']);counts=g.groupby(['case_id','recovery_sec']).size()
        audit.append(dict(scenario=scenario,normal_sec=int(sec),adjustment_pairs=len(g),distinct_recoveries=len(u),patients=g.subject_id.nunique(),cases=g.case_id.nunique(),shared_recovery_times=int((counts>1).sum()),excess_pairs=len(g)-len(u),max_multiplicity=int(counts.max())))
    pd.DataFrame(results).to_csv(R/'aggregate/recovery_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(R/'aggregate/recovery_contrasts.csv',index=False)
    pd.DataFrame(audit).to_csv(R/'aggregate/recovery_denominators.csv',index=False)
    b=np.array(boundary).sum(0)
    save(dict(original_valid_min=full_valid/6,completed_bin_valid_min=float(b[0]/6),difference_valid_min=float((full_valid-b[0])/6),original_low_min=full_low/6,completed_bin_low_min=float(b[1]/6),difference_low_min=float((full_low-b[1])/6),legacy_recovery_algorithm_matches=old_match),R/'aggregate/boundary_audit.json')
    assert len(f.query("scenario=='original_all' and normal_sec==20 and interval_min==5 and horizon_sec==120"))==184
    old=pd.read_csv(B/'restricted/recovery_events.csv').query('interval_min==5 and horizon_sec==120 and scenario=="original_all"')
    new=f.query('interval_min==5 and horizon_sec==120 and scenario=="original_all" and normal_sec==20')
    index=['case_id','adjustment_sec'];a=old.set_index(index).sort_index();b=new.set_index(index).sort_index()
    np.testing.assert_allclose(a[metrics].to_numpy(),b[metrics].to_numpy(),atol=1e-12)
    print(pd.DataFrame(audit).to_string(index=False),flush=True)
    print('RECOVERY PASS',flush=True)

def bridge():
    out=[];flow=[];metadata=subgroup_metadata();start=time.monotonic()
    for source,iterator in [('VitalDB',vital_records()),('MIMIC',mimic_records(metadata))]:
        for i,item in enumerate(iterator,1):
            if np.isfinite(item['age']) and item['age']<25:continue
            v=item['values'];selected=e.paired_hours(v)
            if not selected.any():continue
            a,b=common_support(v);mask=selected[:len(a)];m=mask[::6]
            oldminute=e.minute_median(v)
            assert np.array_equal(np.isfinite(a),np.repeat(np.isfinite(b),6))
            base=dict(source=source,record_id=item['record_id'],subject_id=item['subject_id'])
            flow.append(dict(base,selected_hours=mask.sum()/360,old_10s_valid_min=float((selected&np.isfinite(v)).sum()/6),old_1min_valid_min=int((m&np.isfinite(oldminute)).sum()),common_valid_min=int((m&np.isfinite(b)).sum())))
            for scenario,x,y in [('original_paired',v,oldminute),('common_complete_minute',a,b)]:
                for interval in [1,3,5,10]:
                    for dt,z,s in [(10,x,selected[:len(x)]),(60,y,m)]:
                        _,c,_=e.analyze(z,s,dt,65,interval,180)
                        out.append(dict(base,scenario=scenario,interval_min=interval,resolution='10s' if dt==10 else '1min',**dict(zip(e.COLS,c))))
            if i%400==0 or source=='MIMIC' and i%10==0:print(json.dumps(dict(source=source,records=i,elapsed_sec=round(time.monotonic()-start,1))),flush=True)
    frame=pd.DataFrame(out);private(frame,'bridge_components.csv.gz');private(pd.DataFrame(flow),'bridge_flow.csv')
    summaries=[];deltas=[];gates=[]
    for (source,scenario,n),g in frame.groupby(['source','scenario','interval_min']):
        a=g[g.resolution.eq('10s')].groupby('subject_id')[e.COLS].sum();b=g[g.resolution.eq('1min')].groupby('subject_id')[e.COLS].sum().reindex(a.index)
        x=a.to_numpy();y=b.to_numpy();rng=np.random.default_rng(20261016);bsx=[];bsy=[]
        if scenario=='common_complete_minute':np.testing.assert_allclose(x[:,e.COLS.index('valid_reference_min')],y[:,e.COLS.index('valid_reference_min')],atol=1e-8)
        for k in range(20):
            ix=rng.integers(0,len(a),(50,len(a)))
            bsx.append(x[ix].sum(1));bsy.append(y[ix].sum(1))
        dx=e.derive(x.sum(0));dy=e.derive(y.sum(0));bx=e.derive(np.concatenate(bsx));by=e.derive(np.concatenate(bsy))
        meta=dict(source=source,scenario=scenario,interval_min=n,patients=len(a),records=int(g.record_id.nunique()))
        for metric in dx:
            for resolution,point,reps in [('10s',dx[metric],bx[metric]),('1min',dy[metric],by[metric])]:
                lo,hi=np.nanquantile(reps,[.025,.975]);summaries.append(dict(meta,resolution=resolution,metric=metric,estimate=float(point),lo=float(lo),hi=float(hi)))
            lo,hi=np.nanquantile(by[metric]-bx[metric],[.025,.975]);deltas.append(dict(meta,metric=metric,estimate=float(dy[metric]-dx[metric]),lo=float(lo),hi=float(hi)))
        gates.append(dict(meta,valid_10s_min=float(x[:,e.COLS.index('valid_reference_min')].sum()),valid_1min_min=float(y[:,e.COLS.index('valid_reference_min')].sum()),episodes_10s=float(x[:,e.COLS.index('episodes')].sum()),episodes_1min=float(y[:,e.COLS.index('episodes')].sum())))
    pd.DataFrame(summaries).to_csv(R/'aggregate/bridge_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(R/'aggregate/bridge_contrasts.csv',index=False)
    pd.DataFrame(gates).to_csv(R/'aggregate/bridge_denominators.csv',index=False)
    totals=pd.DataFrame(flow).groupby('source').agg(records=('record_id','nunique'),patients=('subject_id','nunique'),selected_hours=('selected_hours','sum'),old_10s_valid_min=('old_10s_valid_min','sum'),old_1min_valid_min=('old_1min_valid_min','sum'),common_valid_min=('common_valid_min','sum')).reset_index()
    totals['removed_10s_percent']=100*(1-totals.common_valid_min/totals.old_10s_valid_min)
    totals['removed_1min_percent']=100*(1-totals.common_valid_min/totals.old_1min_valid_min)
    totals.to_csv(R/'aggregate/bridge_support_audit.csv',index=False)
    print(totals.to_string(index=False),flush=True);print('BRIDGE PASS',flush=True)

if __name__=='__main__':
    frozen=freeze()
    module=sys.argv[1] if len(sys.argv)>1 else 'all'
    if module in ['all','recovery']:recoveries()
    if module in ['all','bridge']:bridge()
    for p,h in frozen['input_sha256'].items():assert sha(p)==h
    for p,h in frozen['baseline_sha256'].items():assert sha(B/p)==h
    print('FROZEN INPUTS UNCHANGED',flush=True)
