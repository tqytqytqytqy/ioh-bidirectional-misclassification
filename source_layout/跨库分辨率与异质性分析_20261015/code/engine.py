"""Shared grid-preserving observability estimators; no clinical artifact labels."""
import warnings
import numpy as np

FCOLS=['reference_auc','display_auc','hidden_auc','overdisplay_auc','hidden_unavailable_auc',
       'reference_low_min','display_low_min','low_reference_normal_display_min',
       'normal_reference_low_display_min','low_reference_unavailable_min',
       'valid_reference_min','display_available_min','selected_minutes']
ECOLS=['episodes','fresh','inherited_only','never_low','visible','delay_sum_min']
COLS=FCOLS+ECOLS;F=len(FCOLS)
FEATURE_LEVELS={'duration':['60_to_lt180','180_to_lt300','300_to_lt600','ge600'],
                'nadir':['lt55','55_to_lt65'],
                'preceding_normal':['lt60','60_to_lt300','ge300','unknown']}

def minute_median(values):
    v=np.asarray(values,float);v=v[:len(v)//6*6].reshape(-1,6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);a=np.nanmedian(v,axis=1)
    a[np.isfinite(v).sum(axis=1)<5]=np.nan
    return a

def select_hours(values,dt):
    size=int(3600/dt);n=len(values)//size;m=np.zeros(len(values),bool)
    if n:m[:n*size]=np.repeat(np.isfinite(values[:n*size]).reshape(n,size).sum(axis=1)>=.8*size,size)
    return m

def paired_hours(values):
    v=np.asarray(values);n=len(v)//360*360
    a=select_hours(v[:n],10);b=np.repeat(select_hours(minute_median(v[:n]),60),6)
    out=np.zeros(len(v),bool);out[:n]=a&b
    return out

def first_hour(selected,dt):
    out=np.zeros(len(selected),bool);where=np.flatnonzero(selected)
    if len(where):out[where[0]:where[0]+int(3600/dt)]=selected[where[0]:where[0]+int(3600/dt)]
    return out

def display(values,step,phase,initialize=False,age_cap=False):
    v=np.asarray(values);update=np.full(len(v),-1,dtype=np.int64)
    planned=np.arange(phase,len(v),step);good=planned[np.isfinite(v[planned])]
    update[good]=good;held=np.maximum.accumulate(update)
    if initialize:
        finite=np.flatnonzero(np.isfinite(v))
        if len(finite):held[(np.arange(len(v))>=finite[0])&(held<0)]=finite[0]
    seen=held>=0
    if age_cap:seen&=np.arange(len(v))-held<step
    out=np.full(len(v),np.nan);out[seen]=v[held[seen]]
    return out,held

def episodes(reference,threshold,minimum,complete=False):
    v=np.asarray(reference);finite=np.isfinite(v);low=finite&(v<threshold)
    edges=np.diff(np.r_[False,low,False].astype(np.int8))
    a=np.flatnonzero(edges==1);b=np.flatnonzero(edges==-1)
    if not len(a):return a,b,np.array([],float),np.array([],float)
    nadir=np.minimum.reduceat(np.where(low,v,np.inf),a)
    normal=finite&(v>=threshold)
    last_bad=np.maximum.accumulate(np.where(~normal,np.arange(len(v)),-1))
    prior=last_bad[np.maximum(a-1,0)]
    known=(a>0)&normal[np.maximum(a-1,0)]&(prior>=0)&low[np.maximum(prior,0)]
    gap=np.where(known,a-prior-1,np.nan).astype(float)
    keep=b-a>=minimum
    if complete:
        keep&=(a>0)&(b<len(v))&normal[np.maximum(a-1,0)]&normal[np.minimum(b,len(v)-1)]
    return a[keep],b[keep],nadir[keep],gap[keep]

def event_phase(d,held,a,b,threshold,dt):
    if not len(a):return np.zeros(6),np.empty((0,6))
    low=np.isfinite(d)&(d<threshold);fresh=low&(held==np.arange(len(d)))
    count=np.r_[0,np.cumsum(low)];new=np.r_[0,np.cumsum(fresh)]
    visible=count[b]>count[a];f=new[b]>new[a]
    nxt=np.minimum.accumulate(np.where(low,np.arange(len(d)),len(d))[::-1])[::-1]
    delay=np.where(visible,(nxt[a]-a)*dt/60,0)
    rows=np.column_stack([np.ones(len(a)),f,visible&~f,~visible,visible,delay]).astype(float)
    return rows.sum(axis=0),rows

def frequency(reference,d,threshold,selected,dt):
    valid=selected&np.isfinite(reference);seen=valid&np.isfinite(d)
    r=np.where(valid,np.maximum(threshold-reference,0),0)
    z=np.where(seen,np.maximum(threshold-d,0),0)
    h=np.maximum(r-z,0);o=np.maximum(z-r,0)
    return np.array([r.sum(),z.sum(),h.sum(),o.sum(),h[valid&~seen].sum(),
        (valid&(reference<threshold)).sum(),(seen&(d<threshold)).sum(),
        (seen&(reference<threshold)&(d>=threshold)).sum(),
        (seen&(reference>=threshold)&(d<threshold)).sum(),
        (valid&~seen&(reference<threshold)).sum(),valid.sum(),seen.sum(),selected.sum()],float)*dt/60

def analyze(values,selected,dt,threshold,interval,min_sec,features=False,initialize=False,age_cap=False,complete=False):
    v=np.asarray(values,float);selected=np.asarray(selected,bool)
    if v.shape!=selected.shape:raise ValueError('grid mismatch')
    step=int(round(interval*60/dt))
    if not np.isclose(step*dt,interval*60):raise ValueError('interval incompatible with reference grid')
    r=np.where(selected,v,np.nan)
    a,b,nadir,gap=episodes(r,threshold,int(np.ceil(min_sec/dt)),complete)
    groups={}
    if features:
        codes={'duration':np.searchsorted([180,300,600],(b-a)*dt,side='right'),
               'nadir':np.where(nadir<55,0,1),
               'preceding_normal':np.where(np.isnan(gap),3,np.searchsorted([60,300],np.nan_to_num(gap)*dt,side='right'))}
        groups={(feature,label):np.zeros(6) for feature,labels in FEATURE_LEVELS.items() for label in labels}
    phases=[]
    for phase in range(step):
        d,held=display(v,step,phase,initialize,age_cap)
        ec,rows=event_phase(d,held,a,b,threshold,dt)
        phases.append(np.r_[frequency(v,d,threshold,selected,dt),ec])
        if features:
            for feature,labels in FEATURE_LEVELS.items():
                for code,label in enumerate(labels):groups[feature,label]+=rows[codes[feature]==code].sum(axis=0)/step
    phases=np.array(phases);c=phases.mean(axis=0)
    if not np.allclose(phases[:,1]-phases[:,0],phases[:,3]-phases[:,2],rtol=1e-10,atol=1e-7):raise ValueError('AUC identity')
    if not np.allclose(phases[:,F],phases[:,F+1:F+4].sum(axis=1),atol=1e-9,rtol=0):raise ValueError('event partition')
    expected=float(np.minimum(1,(b-a)/step).sum())
    error=abs(c[F+1]-expected) if not initialize else np.nan
    if not initialize and error>1e-8:raise ValueError('periodic fresh sampling identity')
    return phases,c,dict(strata=groups,analytic_fresh_expected=expected,analytic_error=error,episodes=len(a))

def safe_ratio(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)

def derive(x):
    x=np.asarray(x);z={k:x[...,j] for j,k in enumerate(COLS)}
    r=z['reference_auc'];n=z['episodes'];m=z['valid_reference_min'];h=z['selected_minutes']
    return dict(hidden_ratio=safe_ratio(z['hidden_auc'],r),overdisplay_ratio=safe_ratio(z['overdisplay_auc'],r),
      net_ratio=safe_ratio(z['display_auc']-r,r),reference_twa=safe_ratio(r,m),display_twa=safe_ratio(z['display_auc'],m),
      fresh_probability=safe_ratio(z['fresh'],n),inherited_probability=safe_ratio(z['inherited_only'],n),
      miss_probability=safe_ratio(z['never_low'],n),mean_visible_delay_min=safe_ratio(z['delay_sum_min'],z['visible']),
      lowref_normaldisplay_min_per_hour=safe_ratio(z['low_reference_normal_display_min']*60,h),
      normalref_lowdisplay_min_per_hour=safe_ratio(z['normal_reference_low_display_min']*60,h),
      unavailable_fraction=safe_ratio(m-z['display_available_min'],m),
      episodes_per100hours=safe_ratio(n*6000,h))

def patient_sum(ids,x):
    ids=np.asarray(ids).astype(str);unique,inverse=np.unique(ids,return_inverse=True)
    out=np.zeros((len(unique),np.asarray(x).shape[1]));np.add.at(out,inverse,x)
    return unique,out
