"""Completed-bin timing states; all times remain on the original 10-s grid."""
import numpy as np

FRESH=['fresh','recovered_without_fresh','still_low_inherited','still_low_without_low_display','unknown']
RECOVERY=['persistent_low','display_cleared','reference_recurred','unknown']

def low_runs(values,minimum=6):
    v=np.asarray(values,float);low=np.isfinite(v)&(v<65)
    edges=np.diff(np.r_[False,low,False].astype(int))
    return [(int(a),int(b),bool(a>0 and np.isfinite(v[a-1]) and v[a-1]>=65))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=minimum]

def display_indices(values,step,phase):
    v=np.asarray(values,float);ix=np.arange(len(v))
    return np.maximum.accumulate(np.where((ix%step==phase)&np.isfinite(v),ix,-1))

def display_matrix(v,n):
    return np.array([display_indices(v,n,p) for p in range(n)])

def fresh_states(values,start,end,step,horizon,held=None):
    v=np.asarray(values,float);held=display_matrix(v,step) if held is None else held
    deadline=start+horizon//10
    last=min(end,deadline)-1
    phases=np.arange(step)
    first=start+(phases-start)%step
    got=first<=last
    initial=held[:,start-1] if start else np.full(step,-1)
    inherited=(initial>=0)&(v[np.maximum(initial,0)]<65)
    unavailable=initial<0
    state=np.where(inherited,2,3)
    if end<deadline:
        if end<len(v) and np.isfinite(v[end]) and v[end]>=65:
            state[:]=1
        else:
            state[:]=4
    state[got]=0
    out={k:float(np.mean(state==j)) for j,k in enumerate(FRESH)}
    out['unavailable_during_wait']=float(np.mean(unavailable))
    out['fresh_lower']=out['fresh'];out['fresh_upper']=out['fresh']+out['unknown']
    out['not_fresh_confirmed']=sum(out[k] for k in FRESH[1:4])
    return out

def recovery_states(values,recovery_index,step,horizon,held=None):
    v=np.asarray(values,float);held=display_matrix(v,step) if held is None else held
    stop=recovery_index+horizon//10
    state=np.full(step,-1,int)
    for j in range(recovery_index,stop+1):
        active=state<0
        if not active.any():break
        if j>=len(v) or not np.isfinite(v[j]):
            state[active]=3;break
        if v[j]<65:
            state[active]=2;break
        h=held[:,j]
        state[active&(h<0)]=3
        state[active&(h>=0)&(v[np.maximum(h,0)]>=65)]=1
    state[state<0]=0
    out={k:float(np.mean(state==j)) for j,k in enumerate(RECOVERY)}
    out['persistent_lower']=out['persistent_low'];out['persistent_upper']=out['persistent_low']+out['unknown']
    return out

def confirmed_recovery(v,time_sec):
    available=(np.arange(len(v))+1)*10
    after=np.flatnonzero((available>time_sec+1e-9)&(available<=time_sec+300+1e-9))
    if len(after)!=30 or available[-1]<time_sec+300 or np.isfinite(v[after]).mean()<.8:return None
    for j in after[1:]:
        if np.isfinite(v[j-1:j+1]).all() and np.all(v[j-1:j+1]>=65):return int(j)
    return None
