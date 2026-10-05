from __future__ import annotations

import hashlib
import json
import os
import platform
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from audit_core import (classify_legacy, clean_track, nearest_distance, pair_support,
                        rate_transitions, supported_duration, triplet_updates, reconcile_records, paired_timer_count)

ROOT = FROZEN = CANDIDATE = None
DATA_ROOTS = []


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def dump(name, obj, private=False):
    folder = ROOT / ("data_restricted" if private else "aggregate")
    folder.mkdir(parents=True,exist_ok=True)
    (folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def safe(obj):
    return json.loads(json.dumps(obj,default=lambda x: x.item() if hasattr(x,"item") else str(x)))


def run():
    for name in ["aggregate","data_restricted","qa","outputs"]:
        (ROOT/name).mkdir(exist_ok=True)
    os.chmod(ROOT/"data_restricted",0o700)
    protected = [FROZEN/"outputs/intermediate"/n for n in [
        "vitaldb_manifest.parquet","artmap_10s.parquet","nibp_raw_sample.parquet"]]
    protected += [FROZEN/"outputs/tables"/n for n in ["nibp_display_events.csv","nibp_art_pairs.csv"]]
    protected += [FROZEN/"src/ioh/qc/nibp_qc.py",FROZEN/"src/ioh/pipeline.py",
                  DATA_ROOTS[0]/"cases.csv",DATA_ROOTS[0]/"trks.csv"]
    if CANDIDATE is not None:
        manifest_file=CANDIDATE/"03_Reproducibility/outputs_manifest.json"
        protected.append(manifest_file)
        candidate_manifest=json.loads(manifest_file.read_text())
        protected += [CANDIDATE/f["path"] for f in candidate_manifest["files"]]
    before={str(p):digest(p) for p in protected}
    meta=pd.read_parquet(FROZEN/"outputs/intermediate/vitaldb_manifest.parquet").set_index("case_id")
    panel=pd.read_parquet(FROZEN/"outputs/intermediate/artmap_10s.parquet")
    primary=set(panel.case_id.unique())
    art={int(k):v.sort_values("time_sec") for k,v in panel.groupby("case_id",sort=False)}
    raw=pd.read_parquet(FROZEN/"outputs/intermediate/nibp_raw_sample.parquet")
    raw_groups={int(k):v for k,v in raw.groupby("case_id",sort=False)}
    candidate_ids=set(meta.loc[sorted(primary)].query("has_nibp_map").index)
    old=pd.read_csv(FROZEN/"outputs/tables/nibp_display_events.csv")
    old_groups={int(k):v for k,v in old.groupby("case_id",sort=False)}
    pairs=pd.read_csv(FROZEN/"outputs/tables/nibp_art_pairs.csv")
    trks=pd.read_csv(DATA_ROOTS[0]/"trks.csv")
    indexes=[]
    for root in DATA_ROOTS:
        names=set(os.listdir(root/"tracks"))
        indexes.append(names)
    trackmap={(int(r.caseid),r.tname):r.tid for r in trks.itertuples()}
    source_records={}
    def load(cid,name,start,end):
        tid=trackmap.get((cid,name))
        if tid is None:
            return pd.DataFrame(columns=["time_sec","value"]),{},None
        filename=str(tid)+".csv.gz"
        path=next((r/"tracks"/filename for r,names in zip(DATA_ROOTS,indexes) if filename in names),None)
        if path is None:
            return pd.DataFrame(columns=["time_sec","value"]),{},None
        if str(path) not in source_records:
            st=path.stat()
            source_records[str(path)]={"sha256":digest(path),"bytes":st.st_size,"mtime_ns":st.st_mtime_ns}
        df=pd.read_csv(path,compression="gzip")
        cleaned,q=clean_track(df,start,end)
        return cleaned,q,path

    nibp_rows=[]; legacy_reasons=Counter(); cadence=[]; triplet_deltas=[]
    total_adj=0; total_same=0; frozen_reconcile=0; event_reconcile=0
    proxy_pair_count=0; map_changes_by_case={}
    for i,cid in enumerate(sorted(raw_groups)):
        row=meta.loc[cid]; start=max(0.,float(row.anestart)); end=float(row.aneend)
        mapname=str(row.nibp_track_name)
        m,q,p=load(cid,mapname,start,end)
        source_n=len(m)
        frozen=raw_groups[cid].sort_values("time_sec")
        reconciled=reconcile_records(m,frozen.rename(columns={"map":"value"}))
        frozen_reconcile += int(reconciled)
        m=m[m.value.between(20,180)].copy()
        times=m.time_sec.to_numpy();values=m.value.to_numpy()
        delta=np.diff(times); vd=np.diff(values)
        positive_delta=delta[delta>0]
        cadence.extend(positive_delta.tolist())
        total_adj+=len(vd);total_same+=int(np.sum(np.abs(vd)<1))
        legacy=classify_legacy(times,values)
        oldcase=old_groups.get(cid,pd.DataFrame(columns=["display_time_sec","nibp_map"]))
        oldarr=oldcase[["display_time_sec","nibp_map"]].to_numpy()
        la=legacy[["time_sec","value"]].to_numpy()
        event_match=la.shape==oldarr.shape and np.allclose(la,oldarr,rtol=0,atol=1e-8)
        event_reconcile += int(event_match)
        legacy_reasons.update(legacy.reason)
        device=mapname.split('/')[0]
        s,sq,sp=load(cid,device+"/NIBP_SBP",start,end)
        d,dq,dp=load(cid,device+"/NIBP_DBP",start,end)
        joint,tq=triplet_updates(m,s,d)
        changes=joint.loc[joint.any_change,"time_sec"].to_numpy() if not joint.empty else np.array([])
        triplet_deltas.extend(np.diff(changes).tolist())
        actual_changes=times[np.r_[False,np.abs(vd)>=1]] if len(times) else np.array([])
        map_changes_by_case[cid]=actual_changes
        timer=legacy[legacy.reason.eq("timer_same_map")]
        supported=int(np.sum(nearest_distance(timer.time_sec,changes)<=2))
        xart=art[cid]
        paired=int(pair_support(changes,xart.time_sec.to_numpy(),xart.art_map.to_numpy()).sum())
        proxy_pair_count+=paired
        nibp_rows.append({"case_id":cid,"source_map_rows":source_n,"valid_map_rows":len(m),
            "source_frozen_reconciled":reconciled,"legacy_events_reconciled":event_match,
            "sbp_local_available":sp is not None,"dbp_local_available":dp is not None,
            "raw_step_median_sec":float(np.median(positive_delta)) if len(positive_delta) else None,
            "same_map_adjacent_rows":int(np.sum(np.abs(vd)<1)),"adjacent_rows":len(vd),
            "legacy_events":len(legacy),"legacy_initial":int(legacy.reason.eq("initial_record").sum()),
            "legacy_map_changes":int(legacy.reason.eq("map_value_change").sum()),
            "legacy_timer_same_map":len(timer),"timer_triplet_change_nearby_2s":supported,
            "timer_no_triplet_change_nearby_2s":len(timer)-supported,
            "triplet_updates_excluding_initial":len(changes),
            "triplet_sd_only_changes":int(joint.sd_only_change.sum()) if len(joint) else 0,
            "triplet_updates_after_record_gap":int((joint.any_change & joint.after_record_gap).sum()) if len(joint) else 0,
            "triplet_updates_art_window_supported":paired,**q,**tq,
            "sbp_conflicting_timestamps":sq.get("conflicting_timestamps",0),
            "dbp_conflicting_timestamps":dq.get("conflicting_timestamps",0)})
        if (i+1)%250==0:
            print(f"NIBP audited {i+1}/{len(raw_groups)} cases",flush=True)
    nq=pd.DataFrame(nibp_rows)
    empty_candidate_audit=[]
    for cid in sorted(candidate_ids-set(raw_groups)):
        row=meta.loc[cid]
        m,q,p=load(cid,str(row.nibp_track_name),max(0.,float(row.anestart)),float(row.aneend))
        empty_candidate_audit.append({"case_id":cid,"source_local_readable":p is not None,
                                      "anesthesia_window_rows":len(m)})
    pd.DataFrame(empty_candidate_audit).to_csv(ROOT/"data_restricted/empty_nibp_window_candidates.csv",index=False)
    nq.to_csv(ROOT/"data_restricted/case_nibp_diagnostics.csv",index=False)
    if frozen_reconcile != len(raw_groups) or event_reconcile != len(raw_groups):
        raise AssertionError("Source/frozen or legacy event reconciliation failed; see restricted diagnostics")
    relevant=trks[trks.tname.str.contains("PHEN|NEPI|EPI_|DOPA|DOBU|VASO",regex=True)]
    primary_med=relevant[relevant.caseid.isin(primary)]
    med_rows=[];med_transitions=[]
    for r in primary_med[primary_med.tname.str.endswith("_RATE")].itertuples():
        cid=int(r.caseid);row=meta.loc[cid];start=max(0.,float(row.anestart));end=float(row.aneend)
        duration=end-start
        rate,rq,rp=load(cid,r.tname,start,end)
        vol,vq,vp=load(cid,r.tname.replace("_RATE","_VOL"),start,end)
        valid_rate=rate[rate.value.ge(0)]
        transitions=rate_transitions(rate)
        finite_vol=vol[vol.value.ge(0)]
        dv=finite_vol.value.diff();vdtime=finite_vol.time_sec.diff()
        inc_times=finite_vol.loc[dv.gt(0) & vdtime.between(0,5,inclusive="right"),"time_sec"].to_numpy()
        xa=art[cid]
        support=pair_support(transitions.time_sec,xa.time_sec.to_numpy(),xa.art_map.to_numpy())
        volume_support=nearest_distance(transitions.time_sec,inc_times)<=60
        fresh_support=nearest_distance(transitions.time_sec,map_changes_by_case.get(cid,[]))<=30
        for j,tr in enumerate(transitions.itertuples()):
            med_transitions.append({"case_id":cid,"track":r.tname,"time_sec":float(tr.time_sec),
                "kind":tr.kind,"art_window_supported":bool(support[j]),
                "near_positive_volume_increment_60s":bool(volume_support[j]),
                "near_map_update_30s":bool(fresh_support[j])})
        med_rows.append({"case_id":cid,"track":r.tname,"rate_local":rp is not None,"volume_local":vp is not None,
            "rate_window_rows":len(rate),"rate_valid_nonnegative_rows":len(valid_rate),
            "rate_positive_rows":int(valid_rate.value.gt(0).sum()),
            "rate_supported_fraction_5s_gap_cap":supported_duration(valid_rate.time_sec,duration)/duration,
            "rate_first_observation_positive":bool(len(valid_rate) and valid_rate.value.iloc[0]>0),
            "volume_window_rows":len(vol),"volume_negative_steps":int(dv.lt(0).sum()),
            "volume_positive_steps":int(dv.gt(0).sum()),
            "observed_starts":int(transitions.kind.eq("observed_zero_to_positive").sum()),
            "observed_increases":int(transitions.kind.eq("observed_rate_increase").sum()),
            "transitions_art_window_supported":int(support.sum()),
            "transitions_volume_supported_60s":int(volume_support.sum()),
            "transitions_near_map_update_30s":int(fresh_support.sum()),
            "rate_conflicting_timestamps":rq.get("conflicting_timestamps",0),
            "rate_backward_steps_source":rq.get("backward_steps_source",0),
            "volume_conflicting_timestamps":vq.get("conflicting_timestamps",0)})
    mq=pd.DataFrame(med_rows);mt=pd.DataFrame(med_transitions)
    mq.to_csv(ROOT/"data_restricted/case_pump_diagnostics.csv",index=False)
    mt.to_csv(ROOT/"data_restricted/observed_pump_transitions.csv",index=False)
    pump_summary=[]
    for name,sub in mq.groupby("track"):
        pump_summary.append({"track":name,"primary_cases":sub.case_id.nunique(),
            "local_rate_cases":int(sub.rate_local.sum()),"positive_rate_cases":int(sub.rate_positive_rows.gt(0).sum()),
            "record_coverage_median":float(sub.rate_supported_fraction_5s_gap_cap.median()),
            "record_coverage_ge80_cases":int(sub.rate_supported_fraction_5s_gap_cap.ge(.8).sum()),
            "starts":int(sub.observed_starts.sum()),"increases":int(sub.observed_increases.sum()),
            "art_supported_transitions":int(sub.transitions_art_window_supported.sum()),
            "volume_supported_transitions":int(sub.transitions_volume_supported_60s.sum()),
            "rate_first_positive_cases":int(sub.rate_first_observation_positive.sum()),
            "volume_negative_steps":int(sub.volume_negative_steps.sum())})
    ps=pd.DataFrame(pump_summary)
    ps.to_csv(ROOT/"aggregate/pump_timing_summary.csv",index=False)
    totals=[]
    for col in ["intraop_eph","intraop_phe","intraop_epi"]:
        x=pd.to_numeric(meta.loc[sorted(primary),col],errors="coerce")
        totals.append({"field":col,"primary_cases":len(primary),"nonmissing_cases":int(x.notna().sum()),
                       "positive_total_cases":int(x.gt(0).sum()),"has_administration_timestamp":False})
    pd.DataFrame(totals).to_csv(ROOT/"aggregate/drug_total_availability.csv",index=False)
    special=trks[trks.tname.str.contains("CUFF|NIBP.*TIME|EVENT|BOLUS|MEDICATION|DRUG.*TIME",case=False,regex=True)]
    nri={"inventory_candidate_cases":len(candidate_ids),"raw_candidate_cases":len(nq),
        "candidate_cases_without_window_rows":len(empty_candidate_audit),
        "empty_window_cases_source_checked":sum(int(x["source_local_readable"] and x["anesthesia_window_rows"]==0) for x in empty_candidate_audit),
        "raw_source_frozen_reconciled_cases":frozen_reconcile,
        "exact_duplicate_frozen_window_rows":len(raw.dropna())-len(raw.dropna().drop_duplicates(["case_id","time_sec","map"])),
        "legacy_events_reconciled_cases":event_reconcile,"raw_rows":len(raw),"valid_map_rows":int(nq.valid_map_rows.sum()),
        "legacy_display_events":len(old),"paired_legacy_events":int(pairs.paired.sum()),
        "paired_legacy_cases":int(pairs.loc[pairs.paired,"case_id"].nunique()),
        "legacy_event_reasons":dict(legacy_reasons),"paired_timer_same_map":0,
        "timer_near_triplet_change":int(nq.timer_triplet_change_nearby_2s.sum()),
        "timer_no_triplet_change":int(nq.timer_no_triplet_change_nearby_2s.sum()),
        "full_triplet_available_cases":int((nq.sbp_local_available & nq.dbp_local_available).sum()),
        "triplet_complete_row_cases":int(nq.complete_rows.gt(0).sum()),
        "triplet_updates":int(nq.triplet_updates_excluding_initial.sum()),
        "triplet_sd_only_updates":int(nq.triplet_sd_only_changes.sum()),
        "triplet_updates_art_window_supported":proxy_pair_count,
        "map_adjacent_repeat_fraction":total_same/total_adj,
        "raw_positive_cadence_median_sec":float(np.median(cadence)),
        "raw_positive_cadence_2s_fraction":float(np.mean(np.isclose(cadence,2,atol=.25))),
        "triplet_update_interval_quantiles_sec":np.quantile(triplet_deltas,[.25,.5,.75]).tolist(),
        "special_marker_track_names":special.tname.unique().tolist(),
        "explicit_measurement_cycle_marker_verified":False}
    # Event identities are reconciled before attaching legacy pairing flags.
    for cid in sorted(raw_groups):
        f=raw_groups[cid];f=f[f["map"].between(20,180)].sort_values(["time_sec","map"]).drop_duplicates(["time_sec","map"])
        classified=classify_legacy(f.time_sec.to_numpy(),f["map"].to_numpy())
        p=pairs[pairs.case_id.eq(cid)]
        nri["paired_timer_same_map"]+=paired_timer_count(classified,p)
    dump("nibp_summary.json",safe(nri))
    med_any=len(set(primary_med.caseid))
    positive_cases=mq.loc[mq.rate_positive_rows.gt(0),"case_id"].nunique()
    med_summary={"primary_cases":len(primary),"any_pump_track_primary_cases":med_any,
        "any_pump_track_primary_fraction":med_any/len(primary),"positive_pump_rate_cases":int(positive_cases),
        "primary_nibp_and_pump_inventory_cases":len(set(primary_med.caseid)&set(raw_groups)),
        "observed_pump_transitions":len(mt),"observed_transition_cases":int(mt.case_id.nunique()) if len(mt) else 0,
        "art_supported_transitions":int(mt.art_window_supported.sum()) if len(mt) else 0,
        "volume_supported_transitions":int(mt.near_positive_volume_increment_60s.sum()) if len(mt) else 0,
        "transitions_near_map_update_30s":int(mt.near_map_update_30s.sum()) if len(mt) else 0,
        "rate_tracks_checked":len(mq),"missing_local_rate_tracks":int((~mq.rate_local).sum()),
        "missing_local_volume_tracks":int((~mq.volume_local).sum()),
        "bolus_timestamp_fields_in_cases":[],"actual_bolus_timing_available":False,
        "rate_change_is_actual_rescue_administration":False}
    dump("medication_summary.json",safe(med_summary))
    source_manifest=[]
    for path,initial in source_records.items():
        p=Path(path);st=p.stat()
        unchanged=st.st_size==initial["bytes"] and st.st_mtime_ns==initial["mtime_ns"] and digest(p)==initial["sha256"]
        source_manifest.append({"path":path,**initial,"unchanged_after_read":unchanged})
        if not unchanged:
            raise AssertionError("Source track changed during read-only audit")
    after={str(p):digest(p) for p in protected}
    if before!=after:
        raise AssertionError("Frozen input or candidate file changed during audit")
    dump("input_fingerprints.json",{"protected_files":before,"raw_track_files":source_manifest},private=True)
    dump("other_database_scope.json",{
        "mover_raw_data_rescanned_this_run":False,"inspire_patient_link_to_vitaldb_verified":False,
        "independent_databases_must_not_be_patient_joined":True})
    result={"audit_date":"2026-10-03","role":"READ_ONLY_FEASIBILITY_NOT_REVISED_CLINICAL_ANALYSIS",
        "actual_cuff_cycle_gate":"NOT_VERIFIED","actual_bolus_time_gate":"NO_GO_CURRENT_VITALDB_EXPORT",
        "pump_recorded_change_gate":"CONDITIONAL_PILOT_ONLY","combined_rescue_cycle_validation":"NO_GO",
        "primary_cases":len(primary),"primary_subjects":int(meta.loc[sorted(primary),"subjectid"].nunique()),
        "inputs_preserved":True,"protected_file_count":len(protected),"raw_tracks_verified":len(source_records),
        "source_timestamp_unit":"seconds_from_case_recording_start",
        "analysis_shift":"subtract_max_0_anestart_for_MAP_NIBP_and_pump",
        "pairing_window_sec":[-30,30],"min_art_valid_fraction":.8,"fresh_cuff_cycles_inferred":False,
        "clinical_outcome_models_run":False,"manuscript_changed":False,"release_changed":False,
        "runtime_python":platform.python_version()}
    dump("audit_decision.json",result)
    print(json.dumps({"completed":True,"primary_cases":len(primary),"nibp_cases":len(nq),
                      "legacy_reasons":dict(legacy_reasons),"pump_cases":med_any,"pump_transitions":len(mt),
                      "source_files_verified":len(source_records)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--track-roots',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    ROOT=args.output.resolve();FROZEN=args.frozen;CANDIDATE=args.candidate;DATA_ROOTS=args.track_roots
    ROOT.mkdir(parents=True,exist_ok=True)
    run()
