"""Decode every systemic arterial segment; derived trajectories stay private."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
PRIOR=PROJECT/'外部验证可行性审计_20261015/mimic'
sys.path.insert(0,str(PRIOR/'private/deps'))
import hashlib
import json
import time
from collections import Counter
import numpy as np
import pandas as pd
import wfdb
from full_qc_core import WaveBins

SOURCE=Path(__import__('os').environ.get('MIMIC_WAVEFORM_ROOT', 'data/mimic4wdb-0.1.0'))
PRIVATE=ROOT/'private';PRIVATE.mkdir(exist_ok=True,mode=0o700);PRIVATE.chmod(0o700)
OUT=ROOT/'qc';START=time.time()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def save(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def log(stage,**kw):print(json.dumps({'stage':stage,'elapsed_seconds':round(time.time()-START,1),**clean(kw)}),flush=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stat(p):s=p.stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def family(name):
    n=name.upper()
    for f in ['ABP','ART','FAP','AO']:
        if n==f or (n.startswith(f) and n[len(f):].isdigit()):return 'Ao' if f=='AO' else f
    return None


def main():
    rows=json.loads((PRIOR/'private/record_audit.json').read_text())
    candidates=[r for r in rows if r['waveform']['systemic_wave_families']]
    original=np.load(PRIOR/'private/primary_and_sensitivity_bins.npz')
    input_hash=sha(PRIOR/'private/primary_and_sensitivity_bins.npz')
    old_inventory=json.loads((PRIOR/'private/source_inventory_after.json').read_text())
    expected={line.split(maxsplit=1)[1].lstrip('*'):line.split(maxsplit=1)[0] for line in (SOURCE/'SHA256SUMS.txt').read_text().splitlines() if line.strip()}
    headers={};used=set()
    for r in candidates:
        p=SOURCE/r['source_relative_record'];h=wfdb.rdheader(str(p/p.name),rd_segments=True);headers[r['record_id']]=h
        used.add(p/(p.name+'.hea'))
        nf=p/(p.name+'n.csv.gz')
        if nf.is_file():used.add(nf)
        for s in h.segments:
            if s is None:continue
            used.add(p/(s.record_name+'.hea'))
            if not s.sig_len:continue
            for j,name in enumerate(s.sig_name):
                if family(name):used.add(p/s.file_name[j])
    before={};mismatches=[]
    for p in sorted(used):
        rel=str(p.relative_to(SOURCE));info=stat(p);info['sha256']=sha(p);before[rel]=info
        if rel in expected and info['sha256']!=expected[rel]:mismatches.append(rel)
        if rel in old_inventory and 'sha256' in old_inventory[rel] and info['sha256']!=old_inventory[rel]['sha256']:mismatches.append(rel)
    save(PRIVATE/'sources_before.json',before)
    if mismatches:save(PRIVATE/'source_mismatch.json',mismatches);raise RuntimeError('Source checksum mismatch before analysis')
    log('source_verification_complete',candidate_records=len(candidates),source_files=len(before))
    arrays={};records=[];details=[];decode_errors=[];samples=Counter();family_counts=Counter();recordcounts=[]
    for ordinal,r in enumerate(candidates,1):
        p=SOURCE/r['source_relative_record'];h=headers[r['record_id']];n=int(np.floor(r['duration_seconds']/10))
        accum={};off=0.;rec_errors=0
        for segment,frames in zip(h.segments,h.seg_len):
            if segment is not None and frames>0:
                seen=set()
                for channel,name in enumerate(segment.sig_name):
                    fam=family(name)
                    if not fam:continue
                    if fam in seen:raise RuntimeError('Duplicate canonical family inside a segment requires explicit resolution')
                    seen.add(fam)
                    if segment.units[channel].lower()!='mmhg':raise RuntimeError('Unexpected arterial pressure unit')
                    acc=accum.setdefault(fam,WaveBins(n));spf=int(segment.samps_per_frame[channel]);fs=float(segment.fs*spf)
                    width=max(1,int(np.floor(segment.fs*600)))
                    for first in range(0,int(frames),width):
                        end=min(int(frames),first+width)
                        samples['expected']+=(end-first)*spf
                        try:
                            w=wfdb.rdrecord(str(p/segment.record_name),sampfrom=first,sampto=end,channels=[channel],physical=True,smooth_frames=False,return_res=64)
                            a=w.e_p_signal[0]
                            if len(a)!=(end-first)*spf:raise ValueError('Decoded sample count differs from frame count')
                            acc.consume(a,off+first/segment.fs,fs);samples['decoded']+=len(a);samples['chunks']+=1
                        except Exception as e:
                            rec_errors+=1;samples['failed_chunks']+=1
                            decode_errors.append({'record_id':r['record_id'],'segment':segment.record_name,'first_frame':first,'end_frame':end,'error_type':type(e).__name__,'error':str(e)})
            off+=frames/h.fs
        assert abs(off-r['duration_seconds'])<1e-5
        results={f:a.finish() for f,a in accum.items()}
        for f,q in results.items():
            family_counts[f]+=1
            details.append({'record_id':r['record_id'],'family':f,'n_bins':n,'wave_present_bins':int((q['observed_seconds']>0).sum()),
              'finite80_bins':int((q['finite_seconds']>=8-1e-6).sum()),'plausible80_bins':int((q['plausible_seconds']>=8-1e-6).sum()),
              'flatline_bins':int(q['flatline'].sum()),'wave_pass_bins':int(q['pass'].sum()),'wave_strict_bins':int(q['strict'].sum()),
              'samples':q['samples'],'finite_samples':q['finite_samples'],'chunks':q['chunks']})
        rid=r['record_id'];fam=r['numeric'].get('primary_family');key='primary_'+rid
        if key in original:
            numeric=original[key].astype(float);numeric[~np.isfinite(numeric)|(numeric<20)|(numeric>180)]=np.nan
            coherent=np.isfinite(original['sensitivity_'+rid])
        else:numeric=np.full(n,np.nan);coherent=np.zeros(n,bool)
        if fam in results:
            q=results[fam];wave=q['pass'];strict=q['strict'];mean=q['mean']
        else:wave=np.zeros(n,bool);strict=wave.copy();mean=np.full(n,np.nan)
        arrays.update({f'numeric_{rid}':numeric,f'waveform_qc_{rid}':wave,f'waveform_strict_{rid}':strict,f'coherent_{rid}':coherent,f'waveform_mean_{rid}':mean})
        record={'record_id':rid,'subject_id':r['subject_id'],'duration_seconds':r['duration_seconds'],
          'adult':r['clinical']['adult'],'full_record_inside_icu':r['clinical']['full_record_inside_icu'],
          'primary_family':fam,'original_numeric_valid_bins':int(np.isfinite(numeric).sum()),'n_bins':n,'decode_error_chunks':rec_errors}
        records.append(record)
        qv=np.isfinite(numeric)&wave
        recordcounts.append({'record_id':rid,'subject_id':r['subject_id'],'adult':r['clinical']['adult'],
          'full_record_inside_icu':r['clinical']['full_record_inside_icu'],'n_bins':n,
          'numeric_valid_bins':int(np.isfinite(numeric).sum()),'wave_qc_valid_bins':int(qv.sum()),
          'strict_valid_bins':int((np.isfinite(numeric)&strict).sum()),'coherent_valid_bins':int((qv&coherent).sum()),
          'whole_eligible':bool(n>=360 and qv.sum()/n>=.8),'errors':rec_errors})
        log('record_complete',completed_records=ordinal,total_records=len(candidates),decoded_samples=samples['decoded'],failed_chunks=samples['failed_chunks'])
    np.savez_compressed(PRIVATE/'qc_inputs.npz',**arrays);save(PRIVATE/'records.json',records)
    pd.DataFrame(details).to_csv(PRIVATE/'wave_family_qc.csv',index=False);pd.DataFrame(recordcounts).to_csv(PRIVATE/'qc_record_counts.csv',index=False)
    save(PRIVATE/'decode_errors.json',decode_errors)
    after={};changes=[]
    for rel in before:
        p=SOURCE/rel;info=stat(p);info['sha256']=sha(p);after[rel]=info
        if info!=before[rel]:changes.append(rel)
    save(PRIVATE/'sources_after.json',after)
    if changes:raise RuntimeError('Source changed during full waveform QC')
    adult=[x for x in recordcounts if x['adult'] is True and x['full_record_inside_icu']]
    agg={'candidate_records':len(candidates),'systemic_family_record_counts':dict(family_counts),'decoding':dict(samples),
        'primary_numeric_family_records':sum(r['primary_family'] is not None for r in records),
        'confirmed_adult_icu_records_with_numeric':sum(r['primary_family'] is not None for r in records if r['adult'] is True and r['full_record_inside_icu']),
        'adult_icu_whole_eligible_after_qc':sum(r['whole_eligible'] for r in adult),
        'wave_family_totals':{k:int(pd.DataFrame(details)[k].sum()) for k in ['n_bins','wave_present_bins','finite80_bins','plausible80_bins','flatline_bins','wave_pass_bins','wave_strict_bins','samples']},
        'all_systemic_segments_attempted':True,'all_expected_samples_decoded':samples['expected']==samples['decoded'],
        'clinical_adjudication':False,'source_files_hashed_before_after':len(before),'source_changes':0,'source_manifest_mismatches':0,
        'prior_numeric_cache_sha256':input_hash,'qc_input_sha256':sha(PRIVATE/'qc_inputs.npz'),
        'code_sha256':sha(Path(__file__)),'core_code_sha256':sha(OUT/'full_qc_core.py'),
        'elapsed_seconds':time.time()-START,'wfdb_version':wfdb.__version__}
    save(OUT/'full_qc_aggregate.json',agg);save(OUT/'QC_READY.json',{'complete':True,'all_records':len(records),'input_sha256':agg['qc_input_sha256'],'decode_failures':samples['failed_chunks']})
    log('full_qc_complete',**agg)


if __name__=='__main__':main()
