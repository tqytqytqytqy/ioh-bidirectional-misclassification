import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export function checkedInteger(value) {
  const x=Number(value);
  if (!Number.isFinite(x) || x<0 || Math.abs(x-Math.round(x))>1e-7) throw new Error(`Not an integer event count: ${value}`);
  return Math.round(x);
}

export function verifyS15(plan, layout, evidence) {
  const spec=plan.find(s=>s.name==='S15_Cross_Event_Strata');
  const positions=layout.find(s=>s.name===spec.name);
  const cell=address=>{
    const [,col,row]=address.match(/^([A-Z]+)(\d+)$/);
    const c=[...col].reduce((n,k)=>n*26+k.charCodeAt(0)-64,0)-1;
    for(let i=0;i<positions.sections.length;i++) {
      const j=Number(row)-positions.sections[i].headerRow;
      if(j>=0 && j<spec.blocks[i].rows.length) return spec.blocks[i].rows[j][c];
    }
    throw new Error(`Unmapped cell ${address}`);
  };
  for(const fix of evidence.changes) {
    const expected=checkedInteger(cell(fix.detail_cell_unchanged));
    assert.equal(expected,fix.verified_integer_count);
    assert.equal(Number(cell(fix.cell)),expected);
  }
  for(const check of evidence.cross_dimension_checks) {
    const [start,end]=check.range.split(':').map(x=>Number(x.slice(1)));
    let sum=0;
    for(let r=start;r<=end;r++) sum+=checkedInteger(cell(`D${r}`));
    assert.equal(sum,check.independent_frozen_total,check.source_reference+' '+check.feature);
  }
  return {correctedCells:evidence.changes.length,partitionChecks:evidence.cross_dimension_checks.length};
}

if(process.argv[1]===fileURLToPath(import.meta.url)) {
  const R=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
  for(const [x,n] of [[1573.9999999999998,1574],[352.99999999999994,353],[342.99999999999994,343],[280.99999999999994,281],[0,0]]) assert.equal(checkedInteger(x),n);
  for(const x of [-1,NaN,Infinity,0.5,42.01]) assert.throws(()=>checkedInteger(x));
  const read=p=>JSON.parse(fs.readFileSync(path.join(R,p),'utf8'));
  console.log(JSON.stringify({status:'PASS',syntheticChecks:10,...verifyS15(read('presentation/workbook_plan.json'),read('provenance/workbook_layout.json'),read('provenance/s15_count_correction.json'))}));
}
