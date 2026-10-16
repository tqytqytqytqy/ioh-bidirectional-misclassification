from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];P=R.parent;B=P/'EJA_临床时间信息修订版_20261016'
sys.path.insert(0,str(B/'qa/plot_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(P/'SICdb分钟级对齐分析_20261015/private/deps'))
import pandas as pd
O=R/'02_图件';O.mkdir(exist_ok=True);(O/'preview').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':10,'svg.fonttype':'none','pdf.fonttype':42})
C=['#237A8C','#B14A38','#5D6970','#A89034'];rows=[]
def save(fig,name):
    fig.savefig(O/f'{name}.png',dpi=600,facecolor='white')
    fig.savefig(O/f'{name}.svg',facecolor='white')
    fig.savefig(O/'preview'/f'{name}.pdf',facecolor='white');plt.close(fig)
def err(ax,r,x,y,vertical=False,color=C[0],marker='o',label=None):
    if vertical:ax.errorbar(x,100*r.estimate,yerr=[[100*(r.estimate-r.lo)],[100*(r.hi-r.estimate)]],fmt=marker,color=color,capsize=3,label=label)
    else:ax.errorbar(100*r.estimate,y,xerr=[[100*(r.estimate-r.lo)],[100*(r.hi-r.estimate)]],fmt=marker,color=color,capsize=3,label=label)
f=pd.read_csv(R/'aggregate/recovery_summary.csv')
base=f.query('scenario=="original_all" and unit=="distinct_recovery" and weight=="unit_equal"')
original=pd.read_csv(P/'治疗相关事件补充修订_20261004/analysis/aggregate/display_state_results.csv').query('scenario=="Primary" and category=="normal_display"').sort_values('interval_min')
fig,axes=plt.subplots(2,2,figsize=(7.8,7.2),layout='constrained',gridspec_kw={'height_ratios':[1,1.12]});a,b,c,d=axes.ravel()
for r in original.itertuples():
    a.errorbar(r.interval_min,r.estimate,yerr=[[r.estimate-r.lower95],[r.upper95-r.estimate]],fmt='o',color=C[1],capsize=3)
    rows.append(dict(figure=3,panel='A',interval_min=r.interval_min,estimate=r.estimate/100,lo=r.lower95/100,hi=r.upper95/100,source='Original infusion adjustment states'))
a.set(xticks=[1,2.5,5],xlim=(.5,5.5),ylim=(0,60),xlabel='Sampling interval (min)',ylabel='Normal display during low MAP (%)',title='A  At infusion adjustment')
for j,n in enumerate([1,3,5]):
    g=base.query('normal_sec==20 and metric=="persistent_low" and interval_min==@n').sort_values('horizon_sec')
    b.errorbar(g.horizon_sec+(j-1)*2,100*g.estimate,yerr=[100*(g.estimate-g.lo),100*(g.hi-g.estimate)],marker='o',color=C[j],capsize=3,label=f'{n} min')
    rows.extend(dict(figure=3,panel='B',**r) for r in g.to_dict('records'))
b.set(xticks=[60,120],xlim=(45,135),ylim=(-2,60),xlabel='Time since recovery confirmation (s)',ylabel='Persistent low display (%)',title='B  After distinct recovery');b.legend(frameon=False,fontsize=8)
left=np.zeros(3)
for m,label,col in [('persistent_low','Persistent low',C[1]),('display_cleared','Display cleared',C[0]),('reference_recurred','Recurrence first',C[3]),('unknown','Unknown','#B9BFC3')]:
    g=base.query('normal_sec==20 and horizon_sec==120 and metric==@m').sort_values('interval_min');vals=100*g.estimate.to_numpy()
    c.barh(range(3),vals,left=left,color=col,height=.5,label=label)
    for j,v in enumerate(vals):
        if v>=10:c.text(left[j]+v/2,j,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if m in ['persistent_low','display_cleared'] else '#111111')
    left+=vals;rows.extend(dict(figure=3,panel='C',**r) for r in g.to_dict('records'))
c.set(yticks=range(3),yticklabels=['1 min','3 min','5 min'],xlim=(0,100),ylim=(2.7,-.7),xlabel='164 recoveries, phase-averaged (%)',title='C  First transitions by 120 s')
c.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,fontsize=7.5,columnspacing=.9,handlelength=1.2)
for j,sec in enumerate([20,30,60]):
    r=base.query('normal_sec==@sec and interval_min==5 and horizon_sec==120 and metric=="persistent_low"').iloc[0]
    err(d,r,0,j,color=C[1]);d.text(34,j,f'n={r.units}',fontsize=8,va='center');rows.append(dict(figure=3,panel='D',**r.to_dict()))
d.set(yticks=range(3),yticklabels=['20 s','30 s','60 s'],ylim=(2.7,-.7),xlim=(0,43),xticks=[0,10,20,30,40],xlabel='Persistent through 120 s (%)',ylabel='Required normal pressure',title='D  Recovery definition (5 min)')
save(fig,'Figure_3_Infusion_recovery')

original=pd.read_csv(P/'跨库分辨率与异质性分析_20261015/aggregate/paired_resolution.csv')
common=pd.read_csv(R/'aggregate/bridge_contrasts.csv')
summ=pd.read_csv(B/'03_方案与结果来源/aggregate/summary.csv')
fig=plt.figure(figsize=(7.8,6.6),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1]);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
for ax,scenario,panel,title in [(a,'original_paired','A','A  Original paired hours'),(b,'common_complete_minute','B','B  Identical valid time')]:
    for si,source in enumerate(['VitalDB','MIMIC']):
        for mi,m in enumerate(['hidden_ratio','overdisplay_ratio','miss_probability']):
            if scenario=='original_paired':
                r=original.query('source==@source and interval_min==5 and estimand=="pooled" and metric==@m').iloc[0].copy();r['estimate']=r.minute_minus_native
            else:r=common.query('source==@source and interval_min==5 and scenario==@scenario and metric==@m').iloc[0]
            err(ax,r,0,mi+(si-.5)*.18,color=C[si],label=source if mi==0 else None)
            rows.append(dict(figure=4,panel=panel,source=source,scenario=scenario,metric=m,estimate=r.estimate,lo=r.lo,hi=r.hi))
    ax.axvline(0,color='#999999',lw=.8);ax.set(yticks=range(3),yticklabels=['Hidden area','Overdisplay area','No low display'],xlabel='Minute minus 10 s (pp)',title=title,xlim=(-19,7),ylim=(2.65,-.6));ax.legend(frameon=False,loc='lower left',fontsize=8)
for si,source in enumerate(['VitalDB','MIMIC','SICdb']):
    for wi,w in enumerate(['pooled','patient_equal']):
        r=summ.query('source==@source and resolution=="1min" and scenario=="primary" and threshold==65 and interval_min==5 and event_min_sec==180 and estimand==@w and metric=="hidden_ratio"').iloc[0]
        err(c,r,0,si+(wi-.5)*.2,color=C[wi],marker='o' if wi==0 else 's',label=('Pooled' if wi==0 else 'Patient equal') if si==0 else None)
        rows.append(dict(figure=4,panel='C',**r.to_dict()))
c.set(yticks=range(3),yticklabels=['VitalDB','MIMIC','SICdb'],xlim=(0,70),ylim=(2.6,-.6),xlabel='Hidden area / reference area (%)',title='C  Patient weighting at minute resolution');c.legend(frameon=False,loc='lower right',fontsize=8)
save(fig,'Figure_4_Resolution_weighting')
pd.DataFrame(rows).to_csv(R/'aggregate/figure_sources.csv',index=False)
print(json.dumps(dict(figures_updated=2,generative_imaging=False,source_rows=len(rows))))
