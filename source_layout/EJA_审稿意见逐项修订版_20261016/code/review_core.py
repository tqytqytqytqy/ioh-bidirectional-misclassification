"""Focused review sensitivities; preserve elapsed time and missing intervals."""
import numpy as np

def sustained_recovery(values, time_sec, normal_sec):
    v=np.asarray(values,float)
    width=int(normal_sec//10)
    if width<2 or width*10!=normal_sec:raise ValueError('whole 10 s bins required')
    available=(np.arange(len(v))+1)*10
    after=np.flatnonzero((available>time_sec+1e-9)&(available<=time_sec+300+1e-9))
    if len(after)!=30 or available[-1]<time_sec+300 or np.isfinite(v[after]).mean()<.8:return None
    normal=np.isfinite(v)&(v>=65)
    for k in range(width-1,len(after)):
        j=int(after[k])
        if normal[j-width+1:j+1].all():return j
    return None

def common_support(values):
    v=np.asarray(values,float)[:len(values)//6*6].copy()
    blocks=v.reshape(-1,6);complete=np.isfinite(blocks).all(axis=1)
    blocks[~complete]=np.nan
    minute=np.full(len(blocks),np.nan)
    minute[complete]=np.median(blocks[complete],axis=1)
    return v,minute

def distinct_recoveries(rows):
    out={}
    for row in sorted(rows,key=lambda x:(x['case_id'],x['recovery_sec'],x['adjustment_sec'])):
        out.setdefault((row['case_id'],row['recovery_sec']),row)
    return list(out.values())
