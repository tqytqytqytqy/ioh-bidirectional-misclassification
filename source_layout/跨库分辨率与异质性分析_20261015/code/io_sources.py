"""Read frozen local numerical inputs; identifiers never enter aggregate delivery."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
SIC=PROJECT/'SICdb分钟级对齐分析_20261015'
MIMIC=PROJECT/'MIMIC技术重复分析_20261015'
sys.path.insert(0,str(SIC/'private/deps'))
import pandas as pd
CONTRACT='3157dfa67db7f7fbcfcbd859b8e46d978583d8851b12348ea2c71fb68dcae61b'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def save_json(p,value,private=False):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if private:p.chmod(0o600)

def save_private(frame,name):
    p=ROOT/'restricted'/name;p.parent.mkdir(exist_ok=True,parents=True,mode=0o700)
    p.parent.chmod(0o700);frame.to_csv(p,index=False);p.chmod(0o600)

def freeze_inputs():
    for n in ['aggregate','restricted','qa','交付文件']:(ROOT/n).mkdir(exist_ok=True)
    (ROOT/'restricted').chmod(0o700)
    assert sha(ROOT/'analysis_contract.json')==CONTRACT
    marker=ROOT/'qa/frozen_inputs.json'
    if marker.exists():
        obj=json.loads(marker.read_text())
        for p,s in obj['protected_sha256'].items():assert sha(p)==s,p
        return obj
    files=[ROOT/'analysis_contract.json',
      PROJECT/'analysis_v10_subject_phase_release/outputs/intermediate/artmap_10s.parquet',
      PROJECT/'analysis_v10_subject_phase_release/outputs/intermediate/vitaldb_manifest.parquet',
      PROJECT/'可靠动脉起点质控重分析_20261015/restricted/onsets.csv',
      PROJECT/'可靠动脉起点质控重分析_20261015/analysis_contract.json',
      PROJECT/'EJA_VitalDB单数据库修订版_20261015/01_Main_Manuscript_EJA.docx',
      MIMIC/'private/records.json',MIMIC/'private/qc_inputs.npz',MIMIC/'private/qc_ready.json',
      SIC/'private/sicdb/ready.json',SIC/'private/SICdb_components.csv.gz']
    ready=json.loads((SIC/'private/sicdb/ready.json').read_text())
    for row in ready['shards']:
        p=SIC/'private/sicdb'/row['path'];assert sha(p)==row['sha256'];files.append(p)
    protected={str(p):sha(p) for p in files}
    m=json.loads((MIMIC/'private/qc_ready.json').read_text())
    for name,h in m['input_sha256'].items():
        if name in ['records.json','qc_inputs.npz']:assert sha(MIMIC/'private'/name)==h
    obj=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),contract_sha256=CONTRACT,
      effects_started=False,protected_sha256=protected,local_cache_integrity_verified=True,
      raw_external_disks_reread_this_run=False,
      scope='Upstream source extraction hashes retained; this run verifies local numerical caches, not a new full native-waveform read')
    save_json(marker,obj,True)
    print(json.dumps(dict(stage='frozen',files=len(files),contract_sha256=CONTRACT)),flush=True)
    return obj

def subgroup_metadata():
    p=ROOT/'restricted/subgroup_metadata.csv'
    if not p.exists():return {}
    f=pd.read_csv(p,dtype={'record_id':str,'subject_id':str})
    return {(r['source'],r['record_id']):r for r in f.to_dict('records')}

def vital_records():
    p=PROJECT/'analysis_v10_subject_phase_release/outputs/intermediate'
    panel=pd.read_parquet(p/'artmap_10s.parquet');meta=pd.read_parquet(p/'vitaldb_manifest.parquet').set_index('case_id')
    onset=pd.read_csv(PROJECT/'可靠动脉起点质控重分析_20261015/restricted/onsets.csv').set_index('case_id')
    for cid,g in panel.groupby('case_id',sort=True):
        row=meta.loc[cid];g=g.sort_values('time_sec')
        assert np.allclose(np.diff(g.time_sec),10)
        count=int(np.floor((row.primary_end_sec-row.primary_start_sec)/10))
        v=g.art_map.to_numpy(float)[:count].copy();v[(v<20)|(v>180)]=np.nan
        quality=onset.loc[cid].to_dict()
        yield dict(source='VitalDB',record_id=str(cid),subject_id=str(int(row.subjectid)),age=float(row.age),values=v,onset=quality)

def mimic_records(metadata=None):
    metadata=metadata or {};records=json.loads((MIMIC/'private/records.json').read_text())
    with np.load(MIMIC/'private/qc_inputs.npz',allow_pickle=False) as a:
        for r in records:
            if not(r['adult'] is True and r['full_record_inside_icu'] and r['primary_family'] and r['n_bins']>=360):continue
            rid=str(r['record_id']);v=a['numeric_'+rid].astype(float)
            v[~a['waveform_qc_'+rid]|(v<20)|(v>180)]=np.nan
            md=metadata.get(('MIMIC',rid),metadata.get(('MIMIC-IV',rid),{}))
            age=md.get('age',np.nan)
            yield dict(source='MIMIC',record_id=rid,subject_id=str(r['subject_id']),age=float(age) if age is not None else np.nan,values=v,metadata=md)

def sic_records(metadata=None):
    metadata=metadata or {};folder=SIC/'private/sicdb'
    ready=json.loads((folder/'ready.json').read_text())
    for row in ready['shards']:
        with np.load(folder/row['path'],allow_pickle=False) as a:
            for j,cid in enumerate(a['case_id']):
                left,right=a['index'][j:j+2];v=a['map'][left:right].astype(float)
                v[(v<20)|(v>180)]=np.nan;rid=str(int(cid))
                yield dict(source='SICdb',record_id=rid,subject_id=str(int(a['subject_id'][j])),age=float(a['age_bin'][j]),
                  values=v,sbp=a['sbp'][left:right].astype(float),dbp=a['dbp'][left:right].astype(float),
                  icu_offset_sec=float(a['icu_offset_sec'][j]),metadata=metadata.get(('SICdb',rid),{}))

if __name__=='__main__':freeze_inputs()
