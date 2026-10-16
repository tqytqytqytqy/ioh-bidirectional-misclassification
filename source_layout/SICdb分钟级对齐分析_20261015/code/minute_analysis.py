"""Shared native-minute estimators; no interpolation or patient outcome models."""
from pathlib import Path
import hashlib
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd

sys.dont_writebytecode = True
ROOT = Path(__file__).resolve().parents[1]
FCOLS = ['reference_auc', 'display_auc', 'hidden_auc', 'overdisplay_auc',
         'hidden_unavailable_auc', 'reference_low_min', 'display_low_min',
         'low_reference_normal_display_min', 'normal_reference_low_display_min',
         'low_reference_unavailable_min', 'valid_reference_min', 'display_available_min',
         'selected_minutes']
ECOLS = ['episodes', 'fresh', 'inherited_only', 'never_low', 'visible', 'delay_sum_min']
COLS = FCOLS + [f'event{n}_{c}' for n in (3, 5) for c in ECOLS]
KEYS = ['source', 'scenario', 'threshold', 'interval_min']
SEED = 20260706
BOOT = 1000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write_json(path, data, private=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if private: Path(path).chmod(0o600)


def runs(mask):
    e = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def minute_median(values):
    a = np.asarray(values, float)
    n = len(a)//6
    a = a[:n*6].reshape(n, 6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = np.nanmedian(a, axis=1)
    v[np.isfinite(a).sum(axis=1) < 5] = np.nan
    return v


def select_hours(values):
    n = len(values)//60
    mask = np.zeros(len(values), bool)
    if n: mask[:n*60] = np.repeat(np.isfinite(values[:n*60]).reshape(n, 60).sum(axis=1) >= 48, 60)
    return mask


def coherent_mask(v, sbp, dbp):
    paired = np.isfinite(sbp) & np.isfinite(dbp)
    good = ((sbp >= 30) & (sbp <= 250) & (dbp >= 10) & (dbp <= 150) &
            (sbp-dbp >= 10) & (sbp-dbp <= 160) & (dbp <= v) & (v <= sbp))
    return ~paired | good


def constant_mask(values, minimum):
    a = np.asarray(values)
    good = np.isfinite(a)
    equal = good[1:] & good[:-1] & (a[1:] == a[:-1])
    edges = np.r_[0, np.flatnonzero(~equal)+1, len(a)]
    lengths = np.diff(edges)
    return np.repeat((lengths >= minimum) & good[edges[:-1]], lengths)


def display(values, step, phase, initialize=False, age_cap=False):
    values = np.asarray(values, float)
    update = np.full(len(values), -1, dtype=np.int64)
    scheduled = np.arange(phase, len(values), step)
    observed = scheduled[np.isfinite(values[scheduled])]
    update[observed] = observed
    held = np.maximum.accumulate(update)
    if initialize:
        good = np.flatnonzero(np.isfinite(values))
        if len(good): held[(np.arange(len(values)) >= good[0]) & (held < 0)] = good[0]
    out = np.full(len(values), np.nan)
    available = held >= 0
    if age_cap: available &= np.arange(len(values))-held < step
    out[available] = values[held[available]]
    return out, held


def events(values, threshold, minimum, selected=None, complete_edges=False):
    low = np.isfinite(values) & (values < threshold)
    e = np.diff(np.r_[False, low, False].astype(np.int8))
    a, b = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    keep = b-a >= minimum
    if complete_edges:
        if selected is None: raise ValueError('edge filter requires selected mask')
        safea, safeb = np.maximum(a-1, 0), np.minimum(b, len(values)-1)
        keep &= (a > 0) & (b < len(values)) & selected[safea] & selected[safeb]
    return a[keep], b[keep]


def event_components(d, held, a, b, threshold):
    low = np.isfinite(d) & (d < threshold)
    fresh = low & (held == np.arange(len(d)))
    cum = np.r_[0, np.cumsum(low)]
    fcum = np.r_[0, np.cumsum(fresh)]
    visible = cum[b] > cum[a]
    f = fcum[b] > fcum[a]
    if not len(a): return np.zeros(len(ECOLS))
    nextlow = np.minimum.accumulate(np.where(low, np.arange(len(d)), len(d))[::-1])[::-1]
    delay = np.sum(nextlow[a[visible]]-a[visible])
    return np.array([len(a), f.sum(), (visible & ~f).sum(), (~visible).sum(), visible.sum(), delay], float)


def frequency_components(reference, d, threshold, selected):
    valid = selected & np.isfinite(reference)
    seen = valid & np.isfinite(d)
    r = np.where(valid, np.maximum(threshold-reference, 0), 0)
    z = np.where(seen, np.maximum(threshold-d, 0), 0)
    h, o = np.maximum(r-z, 0), np.maximum(z-r, 0)
    return np.array([r.sum(), z.sum(), h.sum(), o.sum(), h[valid & ~seen].sum(),
                     (valid & (reference < threshold)).sum(), (seen & (d < threshold)).sum(),
                     (seen & (reference < threshold) & (d >= threshold)).sum(),
                     (seen & (reference >= threshold) & (d < threshold)).sum(),
                     (valid & ~seen & (reference < threshold)).sum(), valid.sum(), seen.sum(), selected.sum()], float)


def derive(x):
    x = np.asarray(x)
    z = {k: x[..., j] for j, k in enumerate(COLS)}
    def ratio(a, b):
        a, b = np.broadcast_arrays(a, b)
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    r, d, h, o = [z[k] for k in ('reference_auc', 'display_auc', 'hidden_auc', 'overdisplay_auc')]
    net = d-r
    z.update(hidden_ratio=ratio(h, r), overdisplay_ratio=ratio(o, r), net_ratio=ratio(net, r),
             absolute_disagreement_ratio=ratio(h+o, r), cancellation=1-ratio(np.abs(net), h+o),
             reference_twa=ratio(r, z['valid_reference_min']), display_twa=ratio(d, z['valid_reference_min']),
             low_reference_normal_display_min_per_hour=ratio(z['low_reference_normal_display_min']*60, z['selected_minutes']),
             normal_reference_low_display_min_per_hour=ratio(z['normal_reference_low_display_min']*60, z['selected_minutes']))
    for n in (3, 5):
        den = z[f'event{n}_episodes']
        for key in ('fresh', 'inherited_only', 'never_low', 'visible'):
            z[f'event{n}_{key}_probability'] = ratio(z[f'event{n}_{key}'], den)
        z[f'event{n}_mean_display_delay_min'] = ratio(z[f'event{n}_delay_sum_min'], z[f'event{n}_visible'])
    return z


def analyze_case(values, selected, intervals, thresholds, initialize=False, age_cap=False, edges=False):
    reference = np.where(selected, values, np.nan)
    ev = {(t, n): events(reference, t, n, selected, edges) for t in thresholds for n in (3, 5)}
    results, phases = [], []
    for step in intervals:
        prepared = [display(values, step, phase, initialize, age_cap) for phase in range(step)]
        for t in thresholds:
            components = []
            for d, held in prepared:
                c = np.r_[frequency_components(values, d, t, selected),
                          *(event_components(d, held, *ev[t, n], t) for n in (3, 5))]
                components.append(c)
            a = np.array(components)
            if not np.allclose(a[:, 1]-a[:, 0], a[:, 3]-a[:, 2], rtol=1e-10, atol=1e-7): raise ValueError('decomposition identity')
            for n in (3, 5):
                j = len(FCOLS) + (0 if n == 3 else len(ECOLS))
                if not np.all(a[:, j] == a[:, j+1:j+4].sum(axis=1)): raise ValueError('event partition')
                if step <= n and not np.all(a[:, j] == a[:, j+1]): raise ValueError('complete sampling of long event')
            results.append((t, step, a.mean(axis=0)))
            if t == 65 and step == 5: phases = a
    return results, phases


def vital_inputs():
    sys.path.insert(0, str(ROOT/'private/deps'))
    source = ROOT.parent/'analysis_v10_subject_phase_release/outputs/intermediate'
    panelpath, metapath = source/'artmap_10s.parquet', source/'vitaldb_manifest.parquet'
    hashes = {p.name: sha(p) for p in (panelpath, metapath)}
    panel, meta = pd.read_parquet(panelpath), pd.read_parquet(metapath).set_index('case_id')
    records = []
    for cid, g in panel.groupby('case_id', sort=True):
        g = g.sort_values('time_sec')
        if not np.allclose(np.diff(g.time_sec), 10): raise ValueError('VitalDB ten second grid')
        row = meta.loc[cid]
        complete_minutes = int(np.floor((float(row.primary_end_sec)-float(row.primary_start_sec))/60))
        v = g.art_map.to_numpy(float)[:complete_minutes*6]
        v[(v < 20) | (v > 180)] = np.nan
        records.append({'source': 'VitalDB', 'case_id': int(cid), 'subject_id': int(row.subjectid),
                        'age_bin': float(row.age), 'map': minute_median(v), 'first10': v[:len(v)//6*6:6].copy()})
    write_json(ROOT/'private/vitaldb_input_hashes.json', hashes, True)
    return records


def sic_inputs():
    root = ROOT/'private/sicdb'
    ready = json.loads((root/'ready.json').read_text())
    if str(ready['status']).lower() not in ('ready', 'complete', 'pass'): raise ValueError('SICdb extraction not ready')
    for entry in ready['shards']:
        name = entry['path'] if isinstance(entry, dict) else entry
        p = root/name
        if isinstance(entry, dict) and sha(p) != entry['sha256']: raise ValueError('SICdb shard hash')
        with np.load(p, allow_pickle=False) as a:
            data = {name:a[name] for name in ['case_id','subject_id','age_bin','index','map','sbp','dbp']}
        for k, cid in enumerate(data['case_id']):
            l, r = data['index'][k:k+2]
            yield {'source': 'SICdb', 'case_id': int(cid), 'subject_id': int(data['subject_id'][k]),
                   'age_bin': float(data['age_bin'][k]), 'map': data['map'][l:r].astype(float),
                   'sbp': data['sbp'][l:r].astype(float), 'dbp': data['dbp'][l:r].astype(float)}


def scenarios(item):
    raw = item['map']; v = raw.copy()
    v[~np.isfinite(v) | (v < 20) | (v > 180)] = np.nan
    primary = select_hours(v)
    age = item['age_bin']
    if age >= 25 and primary.any():
        yield 'primary', v, primary, False, False, False
        yield 'baseline_initialization', v, primary, True, False, False
        yield 'carry_age_cap_one_interval', v, primary, False, True, False
        yield 'boundary_complete_events', v, primary, False, False, True
        if item['source'] == 'SICdb':
            coherence = coherent_mask(v, item['sbp'], item['dbp'])
            const10, const30 = constant_mask(v, 10), constant_mask(v, 30)
            for name, good in [('coherent_fixed', coherence), ('constant10_fixed', ~const10),
                               ('constant30_fixed', ~const30), ('coherent_constant10_fixed', coherence & ~const10)]:
                changed = np.where(good, v, np.nan)
                yield name, changed, primary, False, False, False
            changed = np.where(coherence & ~const10, v, np.nan)
            reselected = select_hours(changed)
            if reselected.any(): yield 'coherent_constant10_reselected', changed, reselected, False, False, False
        else:
            changed = item['first10']
            changed_selection = select_hours(changed)
            if changed_selection.any(): yield 'vitaldb_first10s_bin', changed, changed_selection, False, False, False
    if age >= 20 and primary.any(): yield 'age20', v, primary, False, False, False
    if item['source'] == 'VitalDB' and primary.any(): yield 'vitaldb_all_original_adults', v, primary, False, False, False


def summarize(frame):
    rows = []; boots = {}; distribution = []
    for (source, scenario), part in frame.groupby(['source', 'scenario'], sort=True):
        ids = np.sort(part.subject_id.unique())
        n = len(ids)
        weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=BOOT).astype(float)
        for (threshold, interval), g in part.groupby(['threshold', 'interval_min'], sort=True):
            sums = g.groupby('subject_id')[COLS].sum().reindex(ids, fill_value=0).to_numpy(float)
            estimates = derive(sums.sum(axis=0)); draws = weights @ sums
            bm = derive(draws)
            row = dict(source=source, scenario=scenario, threshold=int(threshold), interval_min=int(interval),
                       n_records=len(g), n_subjects=n, number_of_phases=int(interval), bootstrap_replicates=BOOT)
            for k, value in estimates.items():
                finite = bm[k][np.isfinite(bm[k])]
                lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
                row.update({k:float(value), k+'_ci_low':float(lo), k+'_ci_high':float(hi)})
            row['n_records_with_3min_events'] = int(g.event3_episodes.gt(0).sum())
            row['n_records_with_5min_events'] = int(g.event5_episodes.gt(0).sum())
            rows.append(row)
            tag = f'{source}__{scenario}__{threshold}__{interval}'
            boots[tag] = draws
            if scenario == 'primary' and threshold == 65 and interval == 5:
                d = derive(g[COLS].to_numpy(float))
                for k in ['hidden_ratio', 'overdisplay_ratio', 'net_ratio', 'event3_fresh_probability',
                          'event3_inherited_only_probability', 'event3_never_low_probability']:
                    finite = d[k][np.isfinite(d[k])]
                    distribution.append({'source':source, 'level':'admission_phase_average', 'metric':k,
                                         'n_defined':len(finite), **dict(zip(['p05','p25','p50','p75','p95'], np.quantile(finite,[.05,.25,.5,.75,.95])))})
                total = sums.sum(axis=0); loo = derive(total-sums)
                for k in ['hidden_ratio', 'overdisplay_ratio', 'net_ratio', 'event3_fresh_probability']:
                    finite = loo[k][np.isfinite(loo[k])]
                    distribution.append({'source':source,'level':'leave_one_patient_out','metric':k,'n_defined':len(finite),
                                         'minimum':float(finite.min()),'maximum':float(finite.max()),
                                         'maximum_absolute_change':float(np.max(np.abs(finite-estimates[k])))})
        print(json.dumps({'bootstrap_complete':source,'scenario':scenario,'n_subjects':n}), flush=True)
    np.savez_compressed(ROOT/'private/bootstrap_sums.npz', **boots)
    (ROOT/'private/bootstrap_sums.npz').chmod(0o600)
    return pd.DataFrame(rows), pd.DataFrame(distribution)


def run(sources=('VitalDB','SICdb')):
    start = time.monotonic()
    for name in ['aggregate', 'private', 'qa']: (ROOT/name).mkdir(exist_ok=True)
    (ROOT/'private').chmod(0o700)
    before = sha(ROOT/'analysis_contract.json')
    manuscript = ROOT.parent/'EJA_VitalDB单数据库修订版_20261015/01_Main_Manuscript_EJA.docx'
    original = sha(manuscript)
    if original != json.loads((ROOT/'analysis_contract.json').read_text())['original_manuscript_sha256']: raise ValueError('manuscript baseline changed')
    for source in sources:
        rows = []; phases = []; qc = []
        iterator = vital_inputs() if source == 'VitalDB' else sic_inputs()
        for count, item in enumerate(iterator, 1):
            raw = item['map']; v = np.where(np.isfinite(raw) & (raw >= 20) & (raw <= 180), raw, np.nan)
            selected = select_hours(v)
            q = {'case_id':item['case_id'],'subject_id':item['subject_id'],'age_bin':item['age_bin'],
                 'complete_minutes':len(v),'valid_range_minutes':int(np.isfinite(v).sum()),
                 'selected_minutes':int(selected.sum()),'selected_valid_minutes':int((selected & np.isfinite(v)).sum()),
                 'selected_runs':len(runs(selected)), 'implausible_map_minutes':int((np.isfinite(raw) & ((raw < 20) | (raw > 180))).sum())}
            if source == 'SICdb':
                observed = np.isfinite(v) & np.isfinite(item['sbp']) & np.isfinite(item['dbp'])
                q.update(paired_triplet_minutes=int(observed.sum()), incoherent_triplet_minutes=int((observed & ~coherent_mask(v,item['sbp'],item['dbp'])).sum()),
                         constant10_minutes=int(constant_mask(v,10).sum()),constant30_minutes=int(constant_mask(v,30).sum()))
            qc.append(q)
            for name, values, mask, init, cap, edges in scenarios(item):
                result, pp = analyze_case(values, mask, [1,3,5,10] if name == 'primary' else [5],
                                          [65,60,55] if name == 'primary' else [65], init, cap, edges)
                for t, step, c in result:
                    rows.append({'source':source,'scenario':name,'threshold':t,'interval_min':step,
                                 'case_id':item['case_id'],'subject_id':item['subject_id'],**dict(zip(COLS,c))})
                if name == 'primary' and len(pp):
                    dd = derive(pp)
                    for k in ['hidden_ratio','overdisplay_ratio','net_ratio','event3_fresh_probability','event3_never_low_probability']:
                        finite = dd[k][np.isfinite(dd[k])]
                        phases.append({'source':source,'case_id':item['case_id'],'metric':k,'n_defined':len(finite),
                                       'minimum':finite.min() if len(finite) else np.nan,'maximum':finite.max() if len(finite) else np.nan,
                                       'phase_range':np.ptp(finite) if len(finite) else np.nan})
            if count % 1000 == 0: print(json.dumps({'processed_source':source,'records':count,'elapsed_seconds':round(time.monotonic()-start)}),flush=True)
        for name, data in [('components',rows),('quality',qc),('phase_ranges',phases)]:
            p = ROOT/f'private/{source}_{name}.csv.gz'
            pd.DataFrame(data).to_csv(p,index=False,compression={'method':'gzip','mtime':0}); p.chmod(0o600)
        print(json.dumps({'source_complete':source,'records':count,'case_scenario_rows':len(rows)}), flush=True)
    if not all((ROOT/f'private/{s}_components.csv.gz').exists() for s in ['VitalDB','SICdb']):
        if before != sha(ROOT/'analysis_contract.json') or original != sha(manuscript): raise ValueError('protected files modified')
        print(json.dumps({'partial_source_processing_complete':list(sources),'awaiting_other_source':True}),flush=True)
        return
    frame = pd.concat([pd.read_csv(ROOT/f'private/{s}_components.csv.gz') for s in ['VitalDB','SICdb']], ignore_index=True)
    summary, dist = summarize(frame)
    summary.to_csv(ROOT/'aggregate/comparison_summary.csv',index=False)
    dist.to_csv(ROOT/'aggregate/distributions_and_influence.csv',index=False)
    phase_rows = []
    for source in ['VitalDB','SICdb']:
        p = pd.read_csv(ROOT/f'private/{source}_phase_ranges.csv.gz')
        for metric, g in p.groupby('metric'):
            v = g.phase_range.dropna().to_numpy()
            phase_rows.append({'source':source,'metric':metric,'n_defined':len(v),**dict(zip(['p05','p25','p50','p75','p95'],np.quantile(v,[.05,.25,.5,.75,.95])))})
    pd.DataFrame(phase_rows).to_csv(ROOT/'aggregate/within_admission_phase_ranges.csv',index=False)
    if before != sha(ROOT/'analysis_contract.json') or original != sha(manuscript): raise ValueError('protected contract or manuscript modified')
    write_json(ROOT/'qa/main_engine_completion.json', {'status':'PASS','elapsed_seconds':time.monotonic()-start,
               'analysis_contract_sha256':before,'manuscript_sha256':original,'manuscript_unchanged':True,
               'code_sha256':sha(Path(__file__)),'summary_rows':len(summary),'uncertainty':'1000 patient-cluster multinomial bootstrap percentile intervals, seed20260706'})


if __name__ == '__main__': run(tuple(sys.argv[1:]) or ('VitalDB','SICdb'))
