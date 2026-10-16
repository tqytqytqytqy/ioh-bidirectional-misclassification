import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
import {verifyS15} from './count_integrity.mjs';
const R=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const plan=JSON.parse(await fs.readFile(path.join(R,'presentation/workbook_plan.json'),'utf8'));
verifyS15(plan,
  JSON.parse(await fs.readFile(path.join(R,'provenance/workbook_layout.json'),'utf8')),
  JSON.parse(await fs.readFile(path.join(R,'provenance/s15_count_correction.json'),'utf8')));
const wb=Workbook.create();
const layout=[];
const out=process.argv[2] ?? path.join(R,'workbook/10_SDC2_Consolidated_Workbook.xlsx');
await fs.mkdir(path.dirname(out),{recursive:true});
const previews=process.argv[3] ?? path.join(R,'rebuild_previews');await fs.mkdir(previews,{recursive:true});
const colName=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
function wrappedLines(value,width){
  const capacity=Math.max(8,Math.floor((width-12)/6.8));
  return String(value??'').split('\n').reduce((total,line)=>{
    let lines=1,used=0;
    for(const word of line.split(/\s+/)){
      if(!word)continue;
      if(used&&used+1+word.length<=capacity)used+=1+word.length;
      else{
        if(used){lines++;used=0;}
        lines+=Math.floor((word.length-1)/capacity);
        used=((word.length-1)%capacity)+1;
      }
    }
    return total+lines;
  },0);
}
for(const s of plan){
  const sh=wb.worksheets.add(s.name);sh.showGridLines=false;
  const nc=Math.max(2,...s.blocks.map(b=>b.rows[0].length));
  const widths=Array.from({length:nc},(_,c)=>s.name==='00_Readme'?(c===0?230:1000):(c===0?(nc>10?190:215):135));
  if(s.name==='Table_1')widths[2]=230;
  if(s.name==='S01_Cohort'){widths[1]=300;widths[4]=280;}
  let r=0;
  const mergedText=(text,size=11,bold=false,color='#222222',span=nc)=>{
    const at=r;sh.mergeCells(`${colName(0)}${r+1}:${colName(span-1)}${r+1}`);
    const a=sh.getCell(r,0);a.values=[[text]];a.format.font={name:'Arial',size,bold,color};a.format.wrapText=true;a.format.verticalAlignment='center';
    const charsPerLine=Math.max(65,span*17);sh.getRangeByIndexes(r,0,1,span).format.rowHeight=Math.max(size+13,(text.split('\n').reduce((n,t)=>n+Math.ceil(t.length/charsPerLine),0))*15+10);
    r++;return at;
  };
  const firstWidth=s.blocks[0].rows[0].length;
  mergedText(s.title,14,true,'#222222',firstWidth);mergedText('Current submission results | 16 October 2026',10,false,'#555555',firstWidth);r++;
  const sec=[];
  for(let bi=0;bi<s.blocks.length;bi++){
    const b=s.blocks[bi];const labelRow=mergedText(b.label,11,true,'#222222',b.rows[0].length);const start=r;
    const w=b.rows[0].length;const full=sh.getRangeByIndexes(r,0,b.rows.length,w);
    full.values=b.rows;
    full.format.font={name:'Arial',size:10,color:'#222222'};full.format.wrapText=true;full.format.verticalAlignment='center';full.format.rowHeight=38;
    const head=sh.getRangeByIndexes(r,0,1,w);head.format.font={name:'Arial',size:10,bold:true,color:'#111111'};
    head.format.rowHeight=Math.max(70,...b.rows[0].map((v,c)=>wrappedLines(v,widths[c])*15+10));head.format.borders={top:{style:'medium',color:'#000000'},bottom:{style:'thin',color:'#000000'}};
    sh.getRangeByIndexes(r+b.rows.length-1,0,1,w).format.borders={bottom:{style:'medium',color:'#000000'}};
    for(let c=0;c<w;c++){
      const vals=b.rows.slice(1).map(x=>x[c]);const nums=vals.filter(v=>typeof v==='number');
      if(nums.length){
        const col=sh.getRangeByIndexes(r+1,c,b.rows.length-1,1);
        col.setNumberFormat(nums.every(Number.isInteger)?'0':'0.0000');
        col.format.horizontalAlignment='center';
      }
    }
    const repeatedNoteRows=[];
    // Merged source footnotes can appear once per column after extraction.
    for(let j=1;j<b.rows.length;j++){
      const vals=b.rows[j].filter(v=>v!==null&&v!=='');
      const repeated=vals.length>1&&vals.every(v=>v===vals[0])&&typeof vals[0]==='string'&&vals[0].length>60;
      if((vals.length===1&&typeof vals[0]==='string')||repeated){
        const rr=r+j;sh.mergeCells(`A${rr+1}:${colName(w-1)}${rr+1}`);
        sh.getRangeByIndexes(rr,0,1,w).values=[[vals[0],...Array(w-1).fill(null)]];
        sh.getRangeByIndexes(rr,0,1,w).format.rowHeight=Math.max(30,wrappedLines(vals[0],widths.slice(0,w).reduce((a,b)=>a+b,0))*15+10);
        if(repeated)repeatedNoteRows.push(j);
      } else{
        const lines=Math.max(...b.rows[j].map((v,c)=>typeof v==='string'?wrappedLines(v,widths[c]):1));
        sh.getRangeByIndexes(r+j,0,1,w).format.rowHeight=Math.max(32,Math.min(400,lines*15+10));
      }
    }
    r+=b.rows.length;
    if(b.note)mergedText(b.note,10,false,'#444444');r+=2;
    sec.push({label:b.label,headerRow:start+1,range:`A${start+1}:${colName(w-1)}${start+b.rows.length}`,rows:b.rows.length,cols:w,labelRow:labelRow+1,repeatedNoteRows});
  }
  if(s.notes){mergedText('Notes for the publication table',11,true);for(const t of [...new Set(s.notes.trim().split('\n'))])mergedText(t,10);}
  for(let c=0;c<nc;c++)sh.getRangeByIndexes(0,c,r,1).format.columnWidthPx=widths[c];
  sh.freezePanes.freezeRows(sec[0].headerRow);sh.freezePanes.freezeColumns(1);
  sh.tabColor=s.name==='00_Readme'?'#333333':s.name.startsWith('Table')?'#2B6A5E':'#687C8F';
  layout.push({name:s.name,rows:r,cols:nc,sections:sec});
}
await wb.recalculate();
await fs.writeFile(path.join(previews,'artifact_inspection.json'),JSON.stringify(await wb.inspect({kind:'sheet',include:'id,name',maxChars:10000}),null,2));
await (await SpreadsheetFile.exportXlsx(wb)).save(out);
await fs.writeFile(path.join(previews,'workbook_layout.json'),JSON.stringify(layout,null,2));
console.log('EXPORTED',out,layout.length);
for(const s of layout){
  const sections=[s.sections[0],...(s.sections.length>1?[s.sections.at(-1)]:[])];
  for(let k=0;k<sections.length;k++){
    const b=sections[k];const top=k===0?1:b.labelRow;
    const range=`A${top}:${colName(b.cols-1)}${Math.min(b.headerRow+6,b.headerRow+b.rows-1)}`;
    const blob=await wb.render({sheetName:s.name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(previews,`${s.name}_${k?'detail':'head'}.png`),new Uint8Array(await blob.arrayBuffer()));
  }
  console.log('RENDERED',s.name);
}
const releasePreview=await wb.render({sheetName:'00_Readme',range:'A18:B21',scale:1,format:'png'});
await fs.writeFile(path.join(previews,'00_Readme_release.png'),new Uint8Array(await releasePreview.arrayBuffer()));
