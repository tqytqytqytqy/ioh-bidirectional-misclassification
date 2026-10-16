from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1];P=R/'private';O=R/'qc'
records=json.loads((P/'records.json').read_text());a=np.load(P/'qc_inputs.npz')
full=json.loads((O/'full_qc_aggregate.json').read_text())
assert full['all_expected_samples_decoded'] and full['decoding'].get('failed_chunks',0)==0
checks={};totals=[];delta=[];xall=[];yall=[];coverage=[]
for r in records:
    rid=r['record_id'];v=a['numeric_'+rid];w=a['waveform_qc_'+rid];s=a['waveform_strict_'+rid];c=a['coherent_'+rid];mean=a['waveform_mean_'+rid]
    assert all(len(x)==r['n_bins'] for x in [v,w,s,c,mean])
    assert (s<=w).all()
    if not (r['adult'] is True and r['full_record_inside_icu'] and r['primary_family']):continue
    ok=np.isfinite(v);coverage.append({'original':ok.mean(),'qc':(ok&w).mean(),'strict':(ok&s).mean()})
    for label,mask in [('numeric_only',ok),('wave_qc',ok&w),('wave_strict',ok&s),('numeric_coherence',ok&w&c)]:
        whole=len(mask)>=360 and mask.mean()>=.8
        window=len(mask)>=360 and np.convolve(mask.astype(np.int64),np.ones(360,dtype=np.int64),'valid').max()>=288
        totals.append({'variant':label,'whole':int(whole),'window':int(window),'valid_bins':int(mask.sum()),'total_bins':len(mask)})
    both=ok&w&np.isfinite(mean)
    delta.append(v[both]-mean[both]);xall.append(v[both]);yall.append(mean[both])
d=np.concatenate(delta);x=np.concatenate(xall);y=np.concatenate(yall)
z=pd.DataFrame(totals).groupby('variant',sort=False).sum().reset_index();z['valid_hours']=z.valid_bins/360
z.to_csv(O/'cohort_and_coverage.csv',index=False)
pd.DataFrame(coverage).to_csv(P/'adult_coverage_private.csv',index=False)
hist,edges=np.histogram(d,bins=np.arange(-50,50.5,.5))
pd.DataFrame({'lower_mmHg':edges[:-1],'upper_mmHg':edges[1:],'bins':hist}).to_csv(O/'numeric_wave_difference_histogram.csv',index=False)
coverage_hist=[]
for field in ['original','qc','strict']:
    count,edge=np.histogram([r[field] for r in coverage],bins=np.linspace(0,1,21))
    coverage_hist.extend({'variant':field,'lower':edge[i],'upper':edge[i+1],'records':int(n)} for i,n in enumerate(count))
pd.DataFrame(coverage_hist).to_csv(O/'record_coverage_histogram.csv',index=False)
diagnostic={'comparison_bins':len(d),'numeric_minus_wave_mean_median_mmHg':np.median(d),
 'difference_q025_mmHg':np.quantile(d,.025),'difference_q975_mmHg':np.quantile(d,.975),
 'absolute_difference_median_mmHg':np.median(abs(d)),'absolute_difference_p95_mmHg':np.quantile(abs(d),.95),
 'fraction_absolute_difference_gt5':np.mean(abs(d)>5),'fraction_absolute_difference_gt10':np.mean(abs(d)>10),
 'pooled_correlation_descriptive_only':np.corrcoef(x,y)[0,1],
 'scope':'Diagnostic comparison of monitor numeric10s median and physical waveform10s mean, not an ISO device validation or independent observations'}
diag={k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in diagnostic.items()}
(O/'numeric_wave_alignment.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n')
checks={'all64records_completed':len(records)==64,'all_expected_samples_decoded':full['all_expected_samples_decoded'],
 'no_source_changes':full['source_changes']==0,'no_manifest_mismatch':full['source_manifest_mismatches']==0,
 'strict_masks_nested_in_primary':True,'all_array_lengths_match_original_grid':True,
 'no_clinical_adjudication_claim':not full['clinical_adjudication']}
(O/'qc_verification.json').write_text(json.dumps({'all_passed':all(checks.values()),'checks':checks},indent=2)+'\n')
print(json.dumps({'all_passed':all(checks.values()),'cohorts':z.to_dict('records'),'alignment':diag},ensure_ascii=False))
