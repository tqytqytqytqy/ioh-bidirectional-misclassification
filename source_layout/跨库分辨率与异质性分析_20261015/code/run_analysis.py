"""Execute the frozen cross-source contrasts without changing any prior artifact."""
import argparse,time,json
import numpy as np
from io_sources import ROOT,PROJECT,SIC,pd,freeze_inputs,subgroup_metadata,vital_records,mimic_records,sic_records,save_json,save_private,sha,CONTRACT
import engine as e

CONFIGS=[(65,1),(65,3),(65,5),(65,10),(60,5),(55,5)]

def run_source(source):
    frozen=freeze_inputs();start=time.monotonic();metadata=subgroup_metadata()
    auditpath=ROOT/'qa/subgroup_audit.json'
    audit=json.loads(auditpath.read_text()) if auditpath.exists() else {}
    records={'VitalDB':vital_records,'MIMIC':lambda:mimic_records(metadata),'SICdb':lambda:sic_records(metadata)}[source]()
    components=[];strata=[];flows=[];analytic=[];n=0;cache_match_max=0.;cache_match_n=0
    cache=None
    if source=='SICdb':
        cache=pd.read_csv(SIC/'private/SICdb_components.csv.gz')
        cache=cache.set_index(['case_id','scenario','threshold','interval_min'])
    def emit(item,values,selected,dt,scenario,minimum,configs=CONFIGS,features=False,**kwargs):
        if not selected.any():return
        meta=dict(source=source,record_id=item['record_id'],subject_id=item['subject_id'],
                  resolution='10s' if dt==10 else '1min',scenario=scenario,event_min_sec=minimum,
                  age_harmonization=('encoded_age25_not_exactly_matched' if source=='SICdb' else 'age25' if np.isfinite(item['age']) else 'confirmed_adult_age_unavailable'))
        for threshold,interval in configs:
            _,c,extra=e.analyze(values,selected,dt,threshold,interval,minimum,
              features=features and threshold==65 and interval==5,**kwargs)
            components.append({**meta,'threshold':threshold,'interval_min':interval,**dict(zip(e.COLS,c))})
            if threshold==65 and interval==5:
                analytic.append({**meta,'expected_fresh':extra['analytic_fresh_expected'],'error':extra['analytic_error']})
                for (feature,level),z in extra['strata'].items():
                    strata.append({**meta,'feature':feature,'level':level,**dict(zip(e.ECOLS,z))})
        flows.append({**meta,'selected_hours':selected.sum()*dt/3600,'valid_hours':(selected&np.isfinite(values)).sum()*dt/3600,
                      'reference_minutes_total':len(values)*dt/60})

    for item in records:
        n+=1
        if np.isfinite(item['age']) and item['age']<25:continue
        v=item['values'];dt=60 if source=='SICdb' else 10
        selected=e.select_hours(v,dt)
        if not selected.any():continue
        if source=='SICdb':
            cid=int(item['record_id'])
            # Reuse immutable additive components, independently rebuilding each 5-min/65 vector.
            _,c,extra=e.analyze(v,selected,60,65,5,180,features=True)
            old=cache.loc[(cid,'primary',65,5)]
            oldc=np.r_[old[e.FCOLS].to_numpy(float),old[['event3_'+x for x in e.ECOLS]].to_numpy(float)]
            assert np.allclose(c,oldc,atol=1e-7,rtol=1e-10),(source,'frozen cache differs')
            cache_match_max=max(cache_match_max,float(np.max(np.abs(c-oldc))));cache_match_n+=1
            meta=dict(source=source,record_id=item['record_id'],subject_id=item['subject_id'],resolution='1min',scenario='primary',
                      event_min_sec=180,age_harmonization='encoded_age25_not_exactly_matched')
            for threshold,interval in CONFIGS:
                row=cache.loc[(cid,'primary',threshold,interval)]
                z=np.r_[row[e.FCOLS].to_numpy(float),row[['event3_'+x for x in e.ECOLS]].to_numpy(float)]
                components.append({**meta,'threshold':threshold,'interval_min':interval,**dict(zip(e.COLS,z))})
            for (feature,level),z in extra['strata'].items():strata.append({**meta,'feature':feature,'level':level,**dict(zip(e.ECOLS,z))})
            analytic.append({**meta,'expected_fresh':extra['analytic_fresh_expected'],'error':extra['analytic_error']})
            flows.append({**meta,'selected_hours':selected.sum()/60,'valid_hours':(selected&np.isfinite(v)).sum()/60,'reference_minutes_total':len(v)})
            for scenario in ['baseline_initialization','carry_age_cap_one_interval','coherent_constant10_fixed']:
                row=cache.loc[(cid,scenario,65,5)]
                z=np.r_[row[e.FCOLS].to_numpy(float),row[['event3_'+x for x in e.ECOLS]].to_numpy(float)]
                components.append({**meta,'scenario':scenario,'threshold':65,'interval_min':5,**dict(zip(e.COLS,z))})
            row=cache.loc[(cid,'primary',65,5)]
            z=np.r_[row[e.FCOLS].to_numpy(float),row[['event5_'+x for x in e.ECOLS]].to_numpy(float)]
            components.append({**meta,'scenario':'minimum300','event_min_sec':300,'threshold':65,'interval_min':5,**dict(zip(e.COLS,z))})
            emit(item,v,selected,60,'boundary_complete',180,[(65,5)],complete=True)
            emit(item,v,e.first_hour(selected,60),60,'first_hour',180,[(65,5)],features=True)
            # ICUOffset is timing-based only; it never labels a preceding interval as surgery.
            offset=item['icu_offset_sec']
            if np.isfinite(offset) and offset>=0 and offset<len(v)*60:
                inside=np.arange(len(v))*60>=offset
                icu_hours=selected&e.select_hours(np.where(inside,v,np.nan),60)
                emit(item,np.where(inside,v,np.nan),icu_hours,60,'icu_after_assignment',180,[(65,5)],features=True)
                label=item['metadata'].get('surgical_label','unknown')
                if label in ['Elective Surgery','Urgent Surgery']:
                    emit(item,np.where(inside,v,np.nan),icu_hours,60,'surgical_admission_after_icu',180,[(65,5)],features=True)
                elif label=='No Surgery':
                    emit(item,np.where(inside,v,np.nan),icu_hours,60,'nonsurgical_admission_after_icu',180,[(65,5)],features=True)
        else:
            emit(item,v,selected,10,'primary',60,features=True)
            minute=e.minute_median(v);minute_selected=e.select_hours(minute,60)
            emit(item,minute,minute_selected,60,'primary',180,features=True)
            paired=e.paired_hours(v)
            if paired.any():
                emit(item,v,paired,10,'paired_bridge',180,[(65,i) for i in [1,3,5,10]],features=True)
                emit(item,minute,paired[:len(minute)*6:6],60,'paired_bridge',180,[(65,i) for i in [1,3,5,10]],features=True)
                emit(item,v,paired,10,'paired_native60',60,[(65,5)],features=True)
            for ref,mask,grid,minimum in [(v,selected,10,60),(minute,minute_selected,60,180)]:
                emit(item,ref,mask,grid,'baseline_initialization',minimum,[(65,5)],initialize=True)
                emit(item,ref,mask,grid,'carry_age_cap_one_interval',minimum,[(65,5)],age_cap=True)
                emit(item,ref,mask,grid,'boundary_complete',minimum,[(65,5)],complete=True)
                emit(item,ref,e.first_hour(mask,grid),grid,'first_hour',minimum,[(65,5)],features=True)
                emit(item,ref,mask,grid,'minimum300',300,[(65,5)])
                if source=='MIMIC' and item['metadata'].get('surgical_label') in ['CSURG','SURG','NSURG','VSURG']:
                    emit(item,ref,mask,grid,'stable_surgical_service',minimum,[(65,5)],features=True)
            first=v[:len(minute)*6:6].copy()
            emit(item,first,minute_selected,60,'minute_first_bin_fixed',180,[(65,5)])
            if source=='VitalDB' and item['onset']['primary_pass']:
                qualified=v.copy();qualified[:int(item['onset']['primary_onset_sec']//10)]=np.nan
                emit(item,qualified,selected,10,'core60_fixed',60,[(65,5)],features=True)
                emit(item,e.minute_median(qualified),minute_selected,60,'core60_fixed',180,[(65,5)],features=True)
        if n%500==0 or source=='MIMIC' and n%10==0:
            print(json.dumps(dict(stage='source_progress',source=source,records_scanned=n,elapsed_sec=round(time.monotonic()-start,1))),flush=True)
    for frame,name in [(pd.DataFrame(components),'components'),(pd.DataFrame(strata),'strata'),(pd.DataFrame(flows),'flow'),(pd.DataFrame(analytic),'analytic')]:
        save_private(frame,f'{source}_{name}.csv.gz')
    assert sha(ROOT/'analysis_contract.json')==CONTRACT
    summary=dict(status='PASS',source=source,records_scanned=n,component_rows=len(components),stratum_rows=len(strata),
      elapsed_sec=time.monotonic()-start,cached_SICdb_primary_records_independently_rebuilt=cache_match_n,
      cached_SICdb_component_max_abs_difference=cache_match_max,metadata_present=bool(metadata),
      analytic_fresh_max_error=max([r['error'] for r in analytic if np.isfinite(r['error'])] or [0]))
    save_json(ROOT/f'qa/{source}_completion.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['VitalDB','MIMIC','SICdb','all'],default='all')
    args=parser.parse_args()
    for source in ['VitalDB','MIMIC','SICdb'] if args.source=='all' else [args.source]:run_source(source)
