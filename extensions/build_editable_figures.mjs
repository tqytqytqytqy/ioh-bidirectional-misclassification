import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
import os from 'node:os';
const require=createRequire(process.env.RUNTIME_NODE_MODULES ? path.join(process.env.RUNTIME_NODE_MODULES,'..','runtime.js') : import.meta.url);
const {Presentation,PresentationFile}=await import(require.resolve('@oai/artifact-tool'));
const OUT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const BUILD=await fs.mkdtemp(path.join(os.tmpdir(),'eja-figure-build-'));
const SKILL=process.env.EJA_ARTIFACT_SKILL_DIR;
const PYTHON=process.env.EJA_PYTHON;
if(!SKILL||!PYTHON)throw new Error('Set EJA_ARTIFACT_SKILL_DIR and EJA_PYTHON for the editable-PowerPoint authoring runtime.');
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const D=JSON.parse(await fs.readFile(path.join(OUT,'outputs/aggregate/figure_data.json'),'utf8'));
const W=628,FONT='Arial',BLACK='#242424',BLUE='#0072B2',ORANGE='#D55E00',GRAY='#8C8C8C',YELLOW='#E2C23A';
let s;
const box=(x,y,w,h,fill='none',stroke='none',sw=0,geometry='rect')=>s.shapes.add({geometry,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:sw}});
function text(t,x,y,w,h,size=13,bold=false,color=BLACK,align='left'){
 const a=box(x,y,w,h);a.text=t;a.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,autoFit:'none',insets:{top:0,right:0,bottom:0,left:0}};return a;
}
const line=(x,y,w,h,color=BLACK)=>box(x,y,Math.max(.8,w),Math.max(.8,h),color);
const dot=(x,y,r=4,color=BLUE)=>box(x-r,y-r,r*2,r*2,color,'none',0,'ellipse');
function key(items,x,y){for(const [name,c] of items){box(x,y+4,11,11,c);text(name,x+17,y,260,22,12);y+=23;}}
function axis(x,y,w,h,max,ticks,xlabels,xs,title){
 line(x,y,w,1);line(x,y-h,1,h);
 for(const t of ticks){const yy=y-h*t/max;line(x-4,yy,4,1);text(String(t),x-41,yy-8,34,19,12,false,BLACK,'right');if(t>0)line(x+1,yy,w-1,.8,'#DEDEDE');}
 text(title,x-8,y-h-37,w+16,26,13,true);
 xlabels.forEach((l,i)=>text(l,xs[i]-29,y+7,58,22,12,false,BLACK,'center'));
}
async function save(fig,height,name){
 const staging=path.join(BUILD,'outputs/figures',name);await fs.mkdir(staging,{recursive:true});
 const receipts=path.join(BUILD,'figure_validation');await fs.mkdir(receipts,{recursive:true});
 const revision=Date.now();
 const candidate=path.join(staging,`candidate_${revision}.pptx`),final=path.join(staging,`final_${revision}.pptx`);
 await (await PresentationFile.exportPptx(fig)).save(candidate);
 await finalizePresentation({workspaceDir:BUILD,candidatePath:candidate,finalPath:final,pythonExecutable:PYTHON,
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
 layoutArgs:['--expected-slide-size-emu',`${W*9525},${height*9525}`,'--folio-mode','none'],
 receiptPath:path.join(receipts,name+'_'+revision+'.json')});
 await fs.mkdir(path.join(OUT,'outputs/figures'),{recursive:true});
 await fs.copyFile(final,path.join(OUT,'outputs/figures',name+'.pptx'));
 const png=await fig.export({slide:s,format:'png',scale:2});
 await fs.writeFile(path.join(OUT,'outputs/figures',name+'.png'),new Uint8Array(await png.arrayBuffer()));
}
function create(h){const p=Presentation.create({slideSize:{width:W,height:h}});s=p.slides.add();s.background.fill='#FFFFFF';return p;}

// Figure 1: separate cohort and management-event flows, with explicit units.
{
 const p=create(680);text('a  Primary waveform cohort',8,4,300,30,15,true);text('b  Infusion-adjustment subset',329,4,290,30,15,true);
 const stages=[['VitalDB source\n6388 cases',58],['Adults\n6331 cases',143],['General anaesthesia\n5989 cases',228],['Eligible surgery groups\n5559 cases',313],['Anaesthesia duration ≥60 min\n5449 cases',398],['Candidate arterial MAP\n3315 cases',483],['Primary waveform cohort\n2435 cases / 2380 subjects',579]];
 const reasons=['57 aged <18 yr','342 without general anaesthesia','430 cardiac, obstetric or transplant','110 with duration <60 min','2134 without candidate MAP track','880 with valid MAP coverage <80%'];
 stages.forEach(([label,y],i)=>{box(10,y,285,47,'#FAFAFA','#555555',.8);text(label,17,y+3,271,40,13,i===6,BLACK,'center');
  if(i<6){line(150,y+47,1,stages[i+1][1]-y-47,'#666666');box(10,y+51,286,25,'#FFFFFF');text(reasons[i],10,y+55,286,19,11,false,BLACK,'center');}});
 const events=[['Target pump records\n133 cases',58],['Observed starts or increases\n834 rate changes',157],['Quality-eligible adjustment blocks\n640 events / 124 cases / 120 subjects',256],['Reference MAP <65 mmHg\n236 events / 85 cases / 83 subjects',355],['Full post-window support\n234 events',454],['Confirmed reference recovery\n184 events / 80 cases / 78 subjects',553]];
 events.forEach(([label,y],i)=>{box(331,y,285,58,'#FAFAFA','#555555',.8);text(label,337,y+5,273,50,13,i===3,BLACK,'center');if(i<5)line(473,y+58,1,41,'#666666');});
 text('Low-reference states use 236 events.\nPost-recovery persistence uses 184 events.',333,625,280,44,12);
 s.speakerNotes.textFrame.setText('Aggregate flow counts from Supplementary Tables S1 and S27. Sequential cohort exclusions are mutually exclusive. Editable vector shapes, no patient-level data.');
 await save(p,680,'Figure_1_Cohort_and_infusion_flow');
}
{
 const p=create(590);const rows=D.intervals.slice(2),xs=rows.map((r,i)=>80+i*99);
 text('a  Reference episodes with any low display',8,2,600,26,15,true);
 axis(66,245,535,185,100,[0,25,50,75,100],rows.map(r=>r[0]),xs,'Events represented (%)');
 const det=rows.map(r=>Number(r[2]));det.forEach((v,i)=>{box(xs[i]-18,245-185*v/100,36,185*v/100,BLUE);text(v.toFixed(1),xs[i]-30,245-185*v/100-24,60,21,12,false,BLACK,'center');});
 text('Sampling interval (min)',200,275,240,22,13,false,BLACK,'center');
 text('b  Aggregate exposure relative to the continuous reference',8,313,610,26,15,true);
 const ref=D.intervals[1],num=t=>Number(String(t).replaceAll(',','').replaceAll(' ',''));
 const dur=rows.map(r=>100*num(r[3])/num(ref[3])),auc=rows.map(r=>100*num(r[4])/num(ref[4]));
 text('Percentage of the reference total',58,342,551,26,13,true);
 rows.forEach((r,i)=>text(r[0],xs[i]-29,529,58,22,12,false,BLACK,'center'));
 // The plotted range is explicitly 94-100; it is not a zero-origin exposure axis.
 line(66,379,1,143);line(66,522,535,1);
 for(const t of [94,96,98,100]){const y=522-(t-94)/6*143;line(66,y,535,.8,'#DDDDDD');text(String(t),27,y-8,32,19,12,false,BLACK,'right');}
 for(let i=0;i<xs.length;i++){
  const y1=522-(dur[i]-94)/6*143,y2=522-(auc[i]-94)/6*143;
  if(i){lineSegment(xs[i-1],522-(dur[i-1]-94)/6*143,xs[i],y1,GRAY);lineSegment(xs[i-1],522-(auc[i-1]-94)/6*143,xs[i],y2,ORANGE);}
  dot(xs[i],y1,3.5,GRAY);dot(xs[i],y2,3.5,ORANGE);
 }
 box(87,570,12,12,GRAY);text('Low-display duration',105,565,200,22,12);box(340,570,12,12,ORANGE);text('Deficit AUC',358,565,150,22,12);
 s.speakerNotes.textFrame.setText('Data from Main Table 2; reference duration 39908 min and AUC 262.222 x 10^3 mmHg min. Panel b range is 94-100% with labelled ticks. No phase-level independence implied.');
 await save(p,590,'Figure_2_Visibility_and_exposure');
}
function lineSegment(x1,y1,x2,y2,color){
 // Narrow editable segments avoid rasterising quantitative traces.
 const length=Math.hypot(x2-x1,y2-y1),steps=Math.max(1,Math.ceil(length/2));
 for(let i=0;i<steps;i++)box(x1+(x2-x1)*i/steps,y1+(y2-y1)*i/steps,1.8,1.8,color);
}
{
 const p=create(640);text('a  Source of low display within each reference event',8,2,610,28,15,true);
 const groups=[{label:'All events',n:9187,vals:[55.9,9.9,34.2]},...['1 to <3 min','3 to <5 min','At least 5 min'].map(label=>{
  const r=D.sources_by_duration.filter(r=>r.Duration===label);return{label,n:+r[0]['Reference episodes'],vals:['fresh','inherited_only','never_low'].map(c=>+r.find(t=>t.Category===c).Percent)};})];
 groups.forEach((g,i)=>{let x=167;const y=56+i*62;text(g.label+'\n'+g.n+' episodes',8,y-2,150,48,13);g.vals.forEach((v,k)=>{if(v>0){box(x,y,435*v/100,35,[BLUE,GRAY,ORANGE][k]);if(v>8)text(v.toFixed(1)+'%',x+1,y+6,435*v/100-2,24,12,false,'#FFFFFF','center');x+=435*v/100;}});});
 text('Inherited low only: 4.9%',167,217,260,20,11,false,GRAY);
 [0,25,50,75,100].forEach(t=>text(String(t),167+435*t/100-20,291,40,21,12,false,BLACK,'center'));
 text('Phase-averaged percentage of reference events',167,314,435,25,13,false,BLACK,'center');
 key([['New low sample in current event',BLUE],['Inherited low display only',GRAY],['No low display',ORANGE]],167,345);
 text('b  Case-level miss proportion',8,427,300,28,14,true);text('c  Within-case phase range',328,427,292,28,14,true);
 const plot=(q,x,y,w,unit,full)=>{const X=v=>x+w*v/100;
  line(x,y+31,w,1);[0,25,50,75,100].forEach(t=>{line(X(t),y+29,1,5);text(String(t),X(t)-18,y+38,36,20,12,false,BLACK,'center');});
  if(full){line(X(q.P5),y,X(q.P95)-X(q.P5),1.3,GRAY);line(X(q.P5),y-5,1,10,GRAY);line(X(q.P95),y-5,1,10,GRAY);}
  line(X(q.Q1),y-3,X(q.Q3)-X(q.Q1),6,BLUE);dot(X(q.Median),y,4,BLACK);
  text(q.Median.toFixed(1)+' ['+q.Q1.toFixed(1)+' to '+q.Q3.toFixed(1)+']',x-3,y-31,w+6,25,13,false,BLACK,'center');
  text(unit,x-3,y+68,w+6,34,12,false,BLACK,'center');};
 plot(D.case_miss,37,510,250,'Missed events per case (%)',true);plot(D.phase_range,345,510,250,'Maximum-minus-minimum (percentage points)',false);
 text('1899 cases; point = median; thick line = interquartile range',28,609,574,25,12,false,BLACK,'center');
 s.speakerNotes.textFrame.setText('Panel a: overall sources from S24 and duration-stratified aggregates. Panels b and c: quantile summaries from S23 notes. Thin whiskers in b are 5th-95th percentiles, not bootstrap confidence intervals.');
 await save(p,640,'Figure_3_Sampling_source_and_heterogeneity');
}
{
 const p=create(500);text('a  Display state at adjustment',8,4,300,26,14,true);text('b  Low display after recovery',329,4,292,26,14,true);
 text('236 low-reference events / 83 subjects',8,34,301,22,12);text('184 recovery events / 78 subjects',329,34,293,22,12);
 const xs=[86,171,256],right=[382,467,552],interval=[1,2.5,5];
 axis(55,331,247,227,100,[0,25,50,75,100],['1','2.5','5'],xs,'Phase-averaged events (%)');
 for(let j=0;j<3;j++){let y=331;for(const [cat,col] of [['normal_display',ORANGE],['inherited_low',GRAY],['fresh_low',BLUE],['unavailable',YELLOW]]){
  const v=+D.display_states.find(r=>+r.interval_min===interval[j]&&r.category===cat).estimate;const height=v*227/100;y-=height;box(xs[j]-20,y,40,height,col);
  if(v>9)text(v.toFixed(1),xs[j]-22,y+height/2-9,44,20,12,false,'#FFFFFF','center');}}
 axis(358,331,245,227,1.2,[0,.3,.6,.9,1.2],['1','2.5','5'],right,'Mean restricted persistence (min)');
 for(let j=0;j<3;j++){
  const r=D.recovery.find(r=>+r.interval_min===interval[j]),Y=v=>331-227*v/1.2;
  line(right[j],Y(+r.upper95),1.2,Y(+r.lower95)-Y(+r.upper95),BLUE);
  line(right[j]-5,Y(+r.upper95),10,1.2,BLUE);line(right[j]-5,Y(+r.lower95),10,1.2,BLUE);dot(right[j],Y(+r.estimate),4,BLUE);
  text((+r.estimate).toFixed(2),right[j]-24,Y(+r.upper95)-26,48,22,13,false,BLACK,'center');
 }
 text('Sampling interval (min)',102,366,425,23,13,false,BLACK,'center');
 key([['Normal display',ORANGE],['New low sample in current low run',BLUE],['Inherited low display only',GRAY],['Unavailable display',YELLOW]],43,402);
 text('Recovery window: remainder of 5 min\nafter adjustment; bars are 95% CI.',337,417,269,55,12);
 s.speakerNotes.textFrame.setText('Infusion module unrounded aggregate data, Table 4 / S28. Distinct state and recovery denominators. Editable vector artwork; not an analysis of rescue indication, treatment delay or drug effectiveness.');
 await save(p,500,'Figure_4_Infusion_adjustment_information');
}
console.log('Four fully editable aggregate-data figures exported and validated.');
