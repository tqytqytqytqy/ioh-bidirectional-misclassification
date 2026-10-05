import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import os from 'node:os';
const require=createRequire(process.env.CODEX_NODE_MODULES
 ? path.join(process.env.CODEX_NODE_MODULES,'..','runtime.js') : import.meta.url);
const {Workbook,SpreadsheetFile}=await import(require.resolve('@oai/artifact-tool'));
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const inputs=JSON.parse(await fs.readFile(path.join(root,'outputs/aggregate/workbook_content.json'),'utf8'));
const wb=Workbook.create();
const font={name:'Arial',size:10,color:'#202020'};
const label=(name)=>({interval_min:'Interval (min)',normal_display:'Normal display',fresh_low:'New low value',
 inherited_low:'Inherited low only',lower95:'Lower 95% CI',upper95:'Upper 95% CI',estimate:'Estimate',
 full_post_support_events:'Full post-window support events',confirmed_recovery_events:'Confirmed recovery events',
 median_min:'Median (min)',q1_min:'Q1 (min)',q3_min:'Q3 (min)'}[name]||name.replaceAll('_',' '));
function base(name,title,rows,cols){
 const s=wb.worksheets.add(name); s.showGridLines=false;
 const range=s.getRangeByIndexes(0,0,rows+6,cols);
 range.format.font=font;range.format.verticalAlignment='center';range.format.rowHeight=28;
 s.getRange('A2').values=[[title]];s.getRange('A2').format.font={...font,size:14,bold:true};
 return s;
}
const overview=base('Infusion_Summary','Display information at infusion adjustments',16,8);
overview.tabColor='#414B56';
overview.getRange('A3').values=[['236 low-reference events in 85 cases from 83 subjects; phase-averaged descriptive estimates']];
const readme=base('ReadMe','Revision scope and definitions',16,2);
readme.tabColor='#808080';
const metadata=[
 ['Version','v1.4.0 reproducibility package, 5 October 2026; frozen aggregate results'],
 ['Primary cohort','2435 cases / 2380 subjects; reference trajectories and postoperative models unchanged'],
 ['Infusion subset','133 target-pump cases; 640 quality-eligible adjustment blocks in 124 cases / 120 subjects'],
 ['Primary denominator','236 low-reference events in 85 cases / 83 subjects; each event averaged across sampling phases'],
 ['Timing','Use completed 10-s medians; exclude first 5 min and require preceding local reference support'],
 ['Inference','2000 subject-cluster percentile bootstrap replicates; seed 20261004; phases are not independent samples'],
 ['Recovery denominator','184 confirmed recoveries in 80 cases / 78 subjects, from 234 events with post-window support'],
 ['NIBP correction','Tables S7-S12 are record audits. Legacy timer records are not authenticated independent cuff measurements'],
 ['Publication tables','All 4 main tables and 29 supplementary tables; Interval_Unrounded adds 55/60/65 mmHg estimates and uncertainty'],
 ['Scope','Descriptive treatment-related context; no rescue-time, drug-effect or monitoring-benefit estimate'],
 ['Reference quality','Startup signal-quality clinical adjudication remains unresolved; this module did not certify it'],
 ['Archives','Release v1.4.0 includes baseline and supplementary modules; cite its version-specific Zenodo DOI'],
 ['Data handling','Raw VitalDB data and case-level trajectories are not redistributed'],
 ['Source','VitalDB v1.0.0; https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md; frozen aggregate outputs'],
];
readme.getRange('A4:B17').values=metadata;
readme.getRange('A4:A17').format.columnWidth=26;
readme.getRange('B4:B17').format.columnWidth=100;
readme.getRange('A4:B17').format.wrapText=true;
readme.getRange('A4:B17').format.rowHeight=38;
readme.getRange('A4:A17').format.font={...font,bold:true};
const sheetMap=new Map();
const qa=await fs.mkdtemp(path.join(os.tmpdir(),'eja-workbook-qa-'));
await fs.mkdir(qa,{recursive:true});
for(const input of inputs){
 const rows=input.values.length,cols=input.values[0].length;
 const s=base(input.name,input.title,rows,cols); sheetMap.set(input.name,s);
 const matrix=input.values.map((r,i)=>i===0?r.map(label):r);
 s.getRangeByIndexes(3,0,rows,cols).values=matrix;
 const area=s.getRangeByIndexes(3,0,rows,cols);
 area.format.wrapText=true; area.format.columnWidth=23;area.format.rowHeight=34;
 s.getRangeByIndexes(3,0,1,cols).format={fill:'#414B56',font:{...font,bold:true,color:'#FFFFFF'},
  wrapText:true,rowHeight:72,horizontalAlignment:'center',verticalAlignment:'center'};
 s.getRangeByIndexes(4,0,rows-1,cols).format.borders={bottom:{style:'thin',color:'#E5E5E5'}};
 if(input.name.startsWith('Supp_Table')||input.name.startsWith('Main_Table')){
  const widest=Math.max(...matrix.slice(1).map(r=>String(r[0]??'').length));
  s.getRangeByIndexes(3,0,rows,1).format.columnWidth=Math.min(48,Math.max(24,widest*.55));
  s.getRangeByIndexes(4,0,rows-1,cols).format.rowHeight=58;
  if(input.name==='Supp_Table_S12') s.getRangeByIndexes(3,1,rows,1).format.columnWidth=75;
  if(input.name==='Supp_Table_S27') s.getRangeByIndexes(3,0,rows,1).format.columnWidth=56;
  if(input.name==='Supp_Table_S28') s.getRangeByIndexes(3,1,rows,cols-1).format.columnWidth=26;
  if(input.name==='Supp_Table_S29') s.getRangeByIndexes(3,0,rows,1).format.columnWidth=49;
 }else{
  s.getRangeByIndexes(4,0,rows-1,cols).setNumberFormat('0.000');
  if(input.name==='Infusion_Unrounded'){
   s.getRangeByIndexes(3,0,rows,1).format.columnWidth=44;
   s.getRangeByIndexes(3,2,rows,1).format.columnWidth=24;
   s.getRangeByIndexes(4,3,rows-1,3).setNumberFormat('#,##0');
  }
 }
 s.freezePanes.freezeRows(4);
 let noteRow=rows+5;
 for(const note of input.notes||[]){
  const nr=s.getRangeByIndexes(noteRow,0,1,cols); nr.merge(); nr.values=[[note]];
  nr.format={font:{...font,size:9,color:'#555555'},wrapText:true,verticalAlignment:'center',
   rowHeight:Math.max(40,Math.ceil(note.length/(cols*24))*15)};
  noteRow++;
 }
 const source=s.getRangeByIndexes(noteRow,0,1,cols); source.merge();
 source.values=[['Source: '+input.source]];source.format.font={...font,color:'#555555',italic:true};
 const render=await wb.render({sheetName:s.name,range:`A1:${String.fromCharCode(64+cols)}${Math.min(noteRow+2,32)}`,scale:1,format:'png'});
 await fs.writeFile(path.join(qa,s.name+'.png'),new Uint8Array(await render.arrayBuffer()));
}
const headers=['Interval (min)','Normal display (%)','Lower 95% CI','Upper 95% CI','New low value (%)','Inherited low only (%)','Unavailable (%)','Recovery persistence (min)'];
overview.getRange('A5:H5').values=[headers];
overview.getRange('A5:H5').format={fill:'#414B56',font:{...font,bold:true,color:'#FFFFFF'},rowHeight:56,
 horizontalAlignment:'center',verticalAlignment:'center',wrapText:true};
overview.getRange('A5:H8').format.columnWidth=20;
overview.getRange('A6:A8').values=[[1],[2.5],[5]];
overview.getRange('A6:H8').setNumberFormat('0.0');
overview.getRange('H6:H8').setNumberFormat('0.00');
const detailed=inputs.find(t=>t.name==='Infusion_Unrounded');
const recovery=inputs.find(t=>t.name==='Recovery_Unrounded');
for(let i=0;i<3;i++){
 const interval=[1,2.5,5][i],r=i+6;
 const locate=(category)=>detailed.values.findIndex((v,k)=>k>0&&v[0]==='Primary'&&v[1]===interval&&v[2]===category)+4;
 const primaryRow=locate('normal_display');
 const recRow=recovery.values.findIndex((v,k)=>k>0&&v[0]===interval)+4;
 overview.getRange(`B${r}:H${r}`).formulas=[[
  `=Infusion_Unrounded!G${primaryRow}`,`=Infusion_Unrounded!H${primaryRow}`,`=Infusion_Unrounded!I${primaryRow}`,
  `=Infusion_Unrounded!G${locate('fresh_low')}`,`=Infusion_Unrounded!G${locate('inherited_low')}`,
  `=Infusion_Unrounded!G${locate('unavailable')}`,`=Recovery_Unrounded!G${recRow}`
 ]];
}
overview.getRange('A11').values=[['Recovery persistence: 184 events from 78 subjects; observed remainder of a 5-min window, not time to drug effect.']];
overview.getRange('A12').values=[['Normal display is emulated; it does not establish that clinicians missed hypotension or delayed treatment.']];
wb.recalculate();
for(const [sheet,range] of [[overview,'A1:H13'],[readme,'A1:B18']]){
 const render=await wb.render({sheetName:sheet.name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(qa,sheet.name+'.png'),new Uint8Array(await render.arrayBuffer()));
}
const inspect=await wb.inspect({kind:'table',range:'Infusion_Summary!A5:H8',include:'values,formulas',maxChars:2500});
await fs.writeFile(path.join(qa,'workbook_summary.ndjson'),inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'formula error scan'});
await fs.writeFile(path.join(qa,'workbook_formula_scan.ndjson'),errors.ndjson);
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(root,'outputs/final_workbook.xlsx'));
console.log(JSON.stringify({sheets:inputs.length+2,publicationTables:33,output:'outputs/final_workbook.xlsx'}));
