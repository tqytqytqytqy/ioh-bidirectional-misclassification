"""Independent primary analysis without importing the replication engine."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
records = json.loads((ROOT / 'private/records.json').read_text())
parts = []
durations = []
with np.load(ROOT / 'private/qc_inputs.npz', allow_pickle=False) as z:
    for rec in records:
        rid = str(rec['record_id'])
        values = np.where(z['waveform_qc_' + rid], z['numeric_' + rid], np.nan)
        if not (rec['adult'] is True and rec['full_record_inside_icu'] and
                rec['primary_family'] and len(values) >= 360 and
                np.isfinite(values).sum() >= np.ceil(.8 * len(values))):
            continue
        duration = rec['duration_seconds'] / 3600
        durations.append((duration, np.isfinite(values).sum() / 360))
        valid = np.isfinite(values)
        for threshold in [65., 60., 55.]:
            low = valid & (values < threshold)
            transitions = np.diff(np.r_[False, low, False].astype(np.int8))
            starts = np.flatnonzero(transitions == 1)
            ends = np.flatnonzero(transitions == -1)
            events = [(s, e) for s, e in zip(starts, ends) if e - s >= 6]
            r = np.where(valid, np.maximum(threshold - values, 0), 0)
            for interval in [1., 2.5, 3., 5., 10.]:
                step = int(interval * 6)
                for phase in range(step):
                    sampled = np.arange(phase, len(values), step)
                    sampled = sampled[valid[sampled]]
                    idx = np.searchsorted(sampled, np.arange(len(values)), side='right') - 1
                    display = np.full(len(values), np.nan)
                    available = idx >= 0
                    display[available] = values[sampled[idx[available]]]
                    d = np.where(valid & available, np.maximum(threshold - display, 0), 0)
                    sample_low = sampled[values[sampled] < threshold]
                    fresh = inherited = never = 0
                    for start, end in events:
                        position = np.searchsorted(sample_low, start)
                        if position < len(sample_low) and sample_low[position] < end:
                            fresh += 1
                        elif np.any(display[start:end] < threshold):
                            inherited += 1
                        else:
                            never += 1
                    parts.append(dict(record_id=rid, subject_id=str(rec['subject_id']), threshold=threshold,
                                      interval_min=interval, offset_sec=phase * 10,
                                      true_auc=r.sum()/6, display_auc=d.sum()/6,
                                      hidden_auc=np.maximum(r-d,0).sum()/6,
                                      overdisplay_auc=np.maximum(d-r,0).sum()/6,
                                      true_hypotension_min=low.sum()/6,
                                      display_low_min=(valid & (display<threshold)).sum()/6,
                                      normotensive_display_discordance_min=(low & (display>=threshold)).sum()/6,
                                      hypotensive_display_discordance_min=(valid & ~low & (display<threshold)).sum()/6,
                                      original_record_hours=duration,valid_reference_min=valid.sum()/6,
                                      episodes=len(events),fresh=fresh,inherited_only=inherited,never_low=never))

frame = pd.DataFrame(parts)
metrics = ['true_auc','display_auc','hidden_auc','overdisplay_auc','true_hypotension_min','display_low_min',
           'normotensive_display_discordance_min','hypotensive_display_discordance_min',
           'original_record_hours','valid_reference_min','episodes','fresh','inherited_only','never_low']
summaries = []
for (threshold,interval), group in frame.groupby(['threshold','interval_min']):
    averaged = group.groupby(['record_id','subject_id'])[metrics].mean().reset_index()
    subjects = averaged.groupby('subject_id')[metrics].sum().to_numpy()
    totals = subjects.sum(axis=0)
    ind = np.random.default_rng(20260706).integers(0,len(subjects),(1000,len(subjects)))
    draws = subjects[ind].sum(axis=1)
    out = dict(threshold=threshold,interval_min=interval,n_records=len(averaged),n_subjects=len(subjects))
    out.update(dict(zip(metrics,totals)))
    for name,num,den in [('hidden_deficit_ratio',2,0),('overdisplay_deficit_ratio',3,0),
                         ('fresh_probability',11,10),('inherited_only_probability',12,10),
                         ('never_low_probability',13,10),('reference_twa',0,9),('display_twa',1,9)]:
        out[name] = totals[num]/totals[den]
        bounds = np.quantile(draws[:,num]/draws[:,den],[.025,.975])
        out[name+'_ci_low'],out[name+'_ci_high'] = bounds
    out['net_bias_ratio']=(totals[1]-totals[0])/totals[0]
    bounds=np.quantile((draws[:,1]-draws[:,0])/draws[:,0],[.025,.975])
    out['net_bias_ratio_ci_low'],out['net_bias_ratio_ci_high']=bounds
    summaries.append(out)

summary = pd.DataFrame(summaries)
outdir=ROOT/'qc'
summary.to_csv(outdir/'independent_primary_results.csv',index=False)
checks={}
aggregate=ROOT/'analysis/aggregate'
if (aggregate/'frequency_summary.csv').exists():
    f=pd.read_csv(aggregate/'frequency_summary.csv').query("scenario=='primary'")
    e=pd.read_csv(aggregate/'episode_summary.csv').query("scenario=='primary' and gap_sec==0 and minimum_low_sec==60")
    for original,cols,label in [(f,metrics[:10]+['hidden_deficit_ratio','overdisplay_deficit_ratio','net_bias_ratio','reference_twa','display_twa'],'frequency'),
                                (e,metrics[10:]+['fresh_probability','inherited_only_probability','never_low_probability'],'events')]:
        merged=summary.merge(original,on=['threshold','interval_min'],suffixes=('_independent','_engine'))
        assert len(merged)==15
        for col in cols:
            for suffix in (['','_ci_low','_ci_high'] if col.endswith(('ratio','probability','twa')) else ['']):
                field=col+suffix
                checks[label+'_'+field]=bool(np.allclose(merged[field+'_independent'],merged[field+'_engine'],rtol=1e-10,atol=1e-8,equal_nan=True))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
result=dict(status='PASS' if checks else 'independent_results_ready',checks=checks,
            n_records=len(durations),original_hours=sum(x[0] for x in durations),
            valid_hours=sum(x[1] for x in durations),
            record_hours_median=float(np.median([x[0] for x in durations])),
            record_hours_min=float(min(x[0] for x in durations)),record_hours_max=float(max(x[0] for x in durations)),
            method='Independent searchsorted sampling and run-length event detection; no replication engine imports')
(outdir/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
