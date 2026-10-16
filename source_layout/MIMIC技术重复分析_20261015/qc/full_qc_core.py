"""Time-preserving automatic arterial waveform QC; no clinical adjudication."""
import numpy as np


class FlatTracker:
    def __init__(self):
        self.last_time=None; self.last_value=np.nan; self.run_start=None; self.last_fs=None

    def consume(self, values, start, fs):
        a=np.asarray(values,float)
        if not len(a):return []
        continuous=self.last_time is not None and self.last_fs==fs and abs(start-self.last_time-1/fs)<1e-5
        d=np.zeros(len(a),bool)
        if continuous:d[0]=np.isfinite(self.last_value) and np.isfinite(a[0]) and abs(a[0]-self.last_value)<=.01
        d[1:]=np.isfinite(a[1:])&np.isfinite(a[:-1])&(np.abs(np.diff(a))<=.01)
        edge=np.diff(np.r_[False,d,False].astype(np.int8)); ss=np.flatnonzero(edge==1);ee=np.flatnonzero(edge==-1)
        spans=[]; trailing=None
        for s,e in zip(ss,ee):
            first=self.run_start if s==0 and continuous else start+(int(s)-1)/fs
            if first is None:first=self.last_time
            last=start+(int(e)-1)/fs
            if last-first>=5-1e-8:spans.append((float(first),float(last+1/fs)))
            if e==len(a):trailing=float(first)
        self.last_time=float(start+(len(a)-1)/fs);self.last_value=float(a[-1]);self.last_fs=fs
        self.run_start=trailing if trailing is not None else self.last_time
        return spans


class WaveBins:
    def __init__(self,n):
        self.n=n;self.observed=np.zeros(n);self.finite=np.zeros(n);self.plausible=np.zeros(n);self.total=np.zeros(n)
        self.minimum=np.full(n,np.inf);self.maximum=np.full(n,-np.inf);self.flat=np.zeros(n,bool)
        self.tracker=FlatTracker();self.samples=0;self.finite_samples=0;self.chunks=0

    def consume(self,values,start,fs):
        a=np.asarray(values,float);self.samples+=len(a);self.finite_samples+=int(np.isfinite(a).sum());self.chunks+=1
        for left,right in self.tracker.consume(a,start,fs):
            lo=max(0,int(np.floor(left/10+1e-9)));hi=min(self.n,int(np.ceil(right/10-1e-9)))
            if hi>lo:self.flat[lo:hi]=True
        t=start+np.arange(len(a))/fs;idx=np.floor(t/10+1e-9).astype(np.int64)
        keep=(idx>=0)&(idx<self.n)
        if not keep.any():return
        idx,a=idx[keep],a[keep];first=int(idx[0]);local=idx-first;length=int(local[-1])+1
        finite=np.isfinite(a);valid=finite&(a>=20)&(a<=250);sl=slice(first,first+length)
        self.observed[sl]+=np.bincount(local,minlength=length)/fs
        self.finite[sl]+=np.bincount(local,weights=finite.astype(float),minlength=length)/fs
        self.plausible[sl]+=np.bincount(local,weights=valid.astype(float),minlength=length)/fs
        self.total[sl]+=np.bincount(local,weights=np.where(finite,a,0),minlength=length)/fs
        begins=np.r_[0,np.flatnonzero(np.diff(idx))+1];binids=idx[begins]
        lo=np.minimum.reduceat(np.where(finite,a,np.inf),begins);hi=np.maximum.reduceat(np.where(finite,a,-np.inf),begins)
        self.minimum[binids]=np.minimum(self.minimum[binids],lo);self.maximum[binids]=np.maximum(self.maximum[binids],hi)

    def finish(self):
        if (self.observed>10.05).any():raise ValueError('Overlapping same-family waveform time exceeds expected bin duration')
        finite=self.finite>=8-1e-6;plausible=self.plausible>=8-1e-6
        good=finite&plausible&~self.flat
        span=self.maximum-self.minimum
        strict=good&(span>=10)&(span<=150)
        mean=np.divide(self.total,self.finite,out=np.full(self.n,np.nan),where=self.finite>0)
        return {'pass':good,'strict':strict,'mean':mean,'observed_seconds':self.observed,'finite_seconds':self.finite,
                'plausible_seconds':self.plausible,'flatline':self.flat,'range':span,
                'samples':self.samples,'finite_samples':self.finite_samples,'chunks':self.chunks}
