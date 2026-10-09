import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';

const OUT = path.resolve(process.env.IOH_QC_OUTPUT_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
const ROOT = path.resolve(process.env.IOH_PROJECT_ROOT || path.dirname(OUT));
const OLD = path.join(ROOT, 'EJA_投稿文件包_20261006_最终文字修订');
const OLDOUT = path.join(OLD, '03_可复现材料/outputs');
const PRIVATE = path.join(OUT, 'data_restricted');
const QA = path.join(OUT, 'qa/workbook');
const BUNDLE = process.env.CODEX_NODE_MODULES;
if (!BUNDLE) throw new Error('Set CODEX_NODE_MODULES to your installed Node dependency directory');
const require = createRequire(path.join(BUNDLE, '..', 'runtime.js'));
const {Workbook, SpreadsheetFile} = await import(require.resolve('@oai/artifact-tool'));
const JSZip = require('jszip');
const {xml2js, js2xml} = require('xml-js');
const sharp = require('sharp');
const FONT = {name: 'Arial', size: 10, color: '#202020'};
const ERROR = /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A|#NUM!|#NULL!|#SPILL!|#CALC!/;
const PRIVATE_HEADER = /^(?:(?:case|subject|patient)[ _]?id|review_code|image_filename|image_path|panel_path)$/i;
const LOCAL_PATH = /(?:\/Users\/|\/Volumes\/|\/home\/|file:\/\/|[A-Z]:\\)/i;
const numeric = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;
const commaNumber = /^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$/;
const specs = new Map();
const provenance = new Map();

const digest = x => crypto.createHash('sha256').update(x).digest('hex');
const normalize = x => String(x ?? '').replace(/\s+/g, ' ').trim();
const numericTokens = x => normalize(x).replace(/[−–]/g, '-').replace(/(?<=\d)[, ](?=\d{3}(?:\D|$))/g, '').match(/\d+(?:\.\d+)?/g) || [];
const col = n => {let s = ''; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s;};
const visibleLength = x => [...String(x ?? '')].reduce((n, c) => n + (c.charCodeAt(0) > 255 ? 1.75 : 1), 0);
const safeSource = x => String(x || '').split(/[;/]/).filter(Boolean).at(-1) || 'Aggregate source';
const fail = message => {throw new Error(message);};
async function read(file) {const data = await fs.readFile(file); provenance.set(file, digest(data)); return data;}
async function exists(file) {try {await fs.access(file); return true;} catch {return false;}}

function typed(value, header = '') {
  if (value == null || value === '') return null;
  if (typeof value !== 'string') return value;
  if (PRIVATE_HEADER.test(header) || /(?:^|_)(?:id|code|filename|path|sha256)(?:$|_)/i.test(header)) return value;
  if (/^(true|false)$/i.test(value)) return value.toLowerCase() === 'true';
  if ((numeric.test(value) || commaNumber.test(value)) && !/^0\d/.test(value)) return Number(value.replaceAll(',', ''));
  return value;
}

function numberFormat(original, value) {
  if (typeof value !== 'number') return '@';
  if (typeof original === 'number') return Number.isInteger(value) ? '#,##0' : '0.000000000000';
  if (/e[+-]?\d+/i.test(original)) return '0.000000E+00';
  const decimals = (String(original).split('.')[1] || '').length;
  return (String(original).includes(',') ? '#,##0' : '0') + (decimals ? '.' + '0'.repeat(Math.min(decimals, 14)) : '');
}

async function csv(file) {
  const text = (await read(file)).toString('utf8').replace(/^\uFEFF/, '');
  const w = await Workbook.fromCSV(text, {sheetName: 'Source'});
  const values = w.worksheets.getItemAt(0).getUsedRange().values;
  while (values.length && values.at(-1).every(x => x == null || x === '')) values.pop();
  assert(values.length > 1, `Empty source: ${path.basename(file)}`);
  const headers = values[0].map(x => String(x ?? ''));
  assert(new Set(headers).size === headers.length, `Duplicate headers: ${path.basename(file)}`);
  return {headers, rows: values.slice(1).map(row => Object.fromEntries(headers.map((h, i) => [h, row[i] == null ? '' : String(row[i])]))),
    values: values.map(r => headers.map((_, i) => r[i] == null ? '' : String(r[i])))};
}

function assertPublic(input) {
  assert(!input.values[0].some(h => PRIVATE_HEADER.test(String(h).trim())), `Restricted header in public table ${input.name}`);
  for (const value of [input.title, input.source, ...(input.notes || []), ...input.values.flat()]) {
    assert(!LOCAL_PATH.test(String(value ?? '')), `Local path in public table ${input.name}`);
  }
}

function textNode(node) {
  if (node.type === 'text') return node.text;
  if (node.name === 'w:tab') return '\t';
  if (node.name === 'w:br') return '\n';
  return (node.elements || []).map(textNode).join('');
}

async function docxTables(file, supplementary = false) {
  const zip = await JSZip.loadAsync(await read(file));
  const tree = xml2js(await zip.file('word/document.xml').async('string'), {compact: false});
  const document = tree.elements.find(e => e.name === 'w:document');
  const body = document.elements.find(e => e.name === 'w:body');
  const tables = [];
  let current;
  for (const node of body.elements || []) {
    if (node.name === 'w:p') {
      const text = textNode(node).trim();
      const match = text.match(supplementary ? /^Supplementary Table S(\d+)\./ : /^Table (\d+)\./);
      if (match) {
        current = {name: supplementary ? `Supp_Table_S${match[1].padStart(2, '0')}` : `Main_Table_${match[1]}`, title: text, notes: []};
        tables.push(current);
      } else if (/^(?:Supplementary )?Figure\s/.test(text)) current = undefined;
      else if (current?.values && text && !/^Table \d+ \(continued\)/.test(text)) current.notes.push(text);
    } else if (node.name === 'w:tbl' && current) {
      const matrix = (node.elements || []).filter(e => e.name === 'w:tr').map(row =>
        (row.elements || []).filter(e => e.name === 'w:tc').map(cell =>
          (cell.elements || []).filter(e => e.name === 'w:p').map(textNode).join('\n')));
      if (current.values) {
        assert.equal(current.name, 'Main_Table_3', 'Unexpected split publication table');
        assert.equal(matrix.length, current.values.length, 'Table 3 panels have different row counts');
        current.values = current.values.map((row, i) => {
          assert.deepEqual(row.slice(0, 2).map(normalize), matrix[i].slice(0, 2).map(normalize), 'Table 3 panel keys mismatch');
          return [...row, ...matrix[i].slice(2)];
        });
      } else current.values = matrix;
    }
  }
  return tables.filter(t => t.values);
}

async function baseline() {
  const inputs = JSON.parse((await read(path.join(OLDOUT, 'aggregate/workbook_content.json'))).toString('utf8'));
  const latest = [];
  for (const file of ['03_Table_1_EJA.docx', '04_Table_2_EJA.docx', '05_Table_3_EJA.docx', '06_Table_4_EJA.docx']) {
    latest.push(...await docxTables(path.join(OLD, '01_投稿文件', file)));
  }
  latest.push(...await docxTables(path.join(OLD, '01_投稿文件/07_Supplemental_Digital_Content_1_EJA.docx'), true));
  assert.equal(latest.length, 33, 'Latest Word must contain 4 main and 29 supplementary tables');
  const refreshed = [];
  const wordPresentationDifferences = [];
  for (const input of inputs) {
    input.source = safeSource(input.source);
    if (/^(?:Main_Table_|Supp_Table_)/.test(input.name)) {
      const source = await csv(path.join(OLDOUT, 'tables', input.name + '.csv'));
      assert.deepEqual(source.values, input.values.map(r => r.map(v => String(v ?? ''))), `Frozen CSV/JSON mismatch: ${input.name}`);
      const doc = latest.find(t => t.name === input.name);
      assert(doc, `Missing Word table: ${input.name}`);
      assert.equal(doc.values.length, input.values.length, `Latest Word row count changed: ${input.name}`);
      let presentationDifferences = 0;
      for (let i = 0; i < input.values.length; i++) {
        assert.equal(doc.values[i].length, input.values[i].length, `Latest Word column count changed: ${input.name}`);
        for (let j = 0; j < input.values[i].length; j++) {
          if (/^[+\-−<≥>≤]?\d/.test(normalize(input.values[i][j]))) {
            assert.deepEqual(numericTokens(doc.values[i][j]), numericTokens(input.values[i][j]), `Latest Word numeric result changed: ${input.name} row ${i+1} col ${j+1}`);
          }
          if (normalize(doc.values[i][j]) !== normalize(input.values[i][j])) presentationDifferences++;
        }
      }
      if (presentationDifferences) wordPresentationDifferences.push({name: input.name, cells: presentationDifferences, handling: 'Frozen CSV values retained'});
      if (input.title !== doc.title || JSON.stringify(input.notes || []) !== JSON.stringify(doc.notes)) refreshed.push(input.name);
      input.title = doc.title;
      input.notes = doc.notes;
    }
  }
  return {inputs, refreshed, wordPresentationDifferences};
}

function addTable(wb, input, {kind = 'publication', headerRow = 4, editable = [], validations = {}, formulas = [], widths: suppliedWidths} = {}) {
  assert(!specs.get(wb)?.some(s => s.name === input.name), `Duplicate worksheet: ${input.name}`);
  assert(input.name.length <= 31 && !/[\\/?*:[\]]/.test(input.name), `Invalid worksheet name: ${input.name}`);
  const columns = input.values[0].length;
  const values = input.values.map((r, i) => input.values[0].map((h, j) => i === 0
    ? (['aggregate', 'private'].includes(kind) ? String(r[j] ?? '').replaceAll('_', ' ') : String(r[j] ?? '')) : typed(r[j], h)));
  assert(values.every(r => r.length === columns), 'Rectangular values required');
  const sheet = wb.worksheets.add(input.name);
  sheet.showGridLines = false;
  if (kind === 'qc') sheet.tabColor = '#386856';
  if (kind === 'clinical') sheet.tabColor = '#A47718';
  const bottom = headerRow + values.length + (input.notes || []).length + 4;
  const area = sheet.getRangeByIndexes(0, 0, bottom, columns);
  area.format = {font: FONT, rowHeight: 28, verticalAlignment: 'center'};
  const data = sheet.getRangeByIndexes(headerRow - 1, 0, values.length, columns);
  data.values = values;
  data.format.wrapText = true;
  const widths = suppliedWidths || input.values[0].map((h, j) => {
    const max = Math.max(...input.values.slice(1).map(r => visibleLength(r[j])));
    if (kind === 'publication') return Math.max(j === 0 ? 24 : 19, Math.min(j === 0 ? 52 : 56, max * 0.6));
    if (/(?:notes|recommendation|reason|interpretation|summary|limitation|action|intervals|evidence|uncertainty|assessment_scope)/i.test(h)) return 56;
    if (/sha256/i.test(h)) return 56;
    if (/(?:image|filename|path)/i.test(h)) return 48;
    if (PRIVATE_HEADER.test(h)) return 17;
    return Math.min(36, Math.max(19, visibleLength(h) * 0.7));
  });
  let presentationColumns = 1, presentationWidth = widths[0];
  while (presentationColumns < columns && presentationWidth + widths[presentationColumns] <= 175) presentationWidth += widths[presentationColumns++];
  const titleRange = sheet.getRangeByIndexes(1, 0, 1, presentationColumns);
  titleRange.merge(); titleRange.values = [[input.title]];
  titleRange.format = {font: {...FONT, size: 14, bold: true}, wrapText: true, rowHeight: 46};
  for (let j = 0; j < columns; j++) {
    sheet.getRangeByIndexes(headerRow - 1, j, values.length, 1).format.columnWidth = widths[j];
    const formats = input.values.slice(1).map((r, i) => {
      const v = values[i + 1][j];
      if (kind === 'qc' && typeof v === 'number') {
        const header = String(input.values[0][j]);
        if (/TWA/i.test(header)) return ['0.000'];
        if (/time|AUC|Visible events/i.test(header)) return [/%/.test(header) ? '0.0' : '#,##0.0'];
        return [Number.isInteger(v) ? '#,##0' : '0.###'];
      }
      return [['aggregate', 'private'].includes(kind) && typeof v === 'number' && !Number.isInteger(v)
        ? (Math.abs(v) > 0 && Math.abs(v) < 0.000001 ? '0.000000E+00' : '0.000000') : numberFormat(r[j], v)];
    });
    if (formats.length) sheet.getRangeByIndexes(headerRow, j, formats.length, 1).format.numberFormat = formats;
  }
  const header = sheet.getRangeByIndexes(headerRow - 1, 0, 1, columns);
  header.format = {fill: '#414B56', font: {...FONT, color: '#FFFFFF', bold: true}, wrapText: true,
    horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 72,
    borders: {insideVertical: {style: 'thin', color: '#FFFFFF'}}};
  for (let i = 1; i < values.length; i++) {
    const lines = Math.max(...values[i].map((v, j) => String(v ?? '').split('\n').reduce((n, line) => n + Math.max(1, Math.ceil(visibleLength(line) / Math.max(8, widths[j] - 2))), 0)));
    sheet.getRangeByIndexes(headerRow - 1 + i, 0, 1, columns).format.rowHeight = Math.max(kind === 'publication' ? 46 : 34, lines * 15 + 12);
  }
  if (values.length > 1) sheet.getRangeByIndexes(headerRow, 0, values.length - 1, columns).format.borders = {insideHorizontal: {style: 'thin', color: '#E5E5E5'}};
  for (const j of editable) {
    sheet.getRangeByIndexes(headerRow, j, values.length - 1, 1).format.fill = '#FFF2CC';
    sheet.getRangeByIndexes(headerRow - 1, j, 1, 1).format.fill = '#80651D';
  }
  for (const [j, choices] of Object.entries(validations)) sheet.getRangeByIndexes(headerRow, Number(j), values.length - 1, 1).dataValidation = {rule: {type: 'list', values: choices}};
  for (const {column, expressions} of formulas) sheet.getRangeByIndexes(headerRow, column, expressions.length, 1).formulas = expressions.map(v => [v]);
  if (values.length > 15) sheet.freezePanes.freezeRows(headerRow);
  let noteRow = headerRow + values.length + 1;
  for (const note of input.notes || []) {
    const range = sheet.getRangeByIndexes(noteRow - 1, 0, 1, presentationColumns);
    range.merge(); range.values = [[note]];
    range.format = {font: {...FONT, color: '#555555'}, wrapText: true, rowHeight: Math.max(38, Math.ceil(visibleLength(note) / (presentationWidth - 5)) * 16 + 16)};
    noteRow++;
  }
  const sourceRange = sheet.getRangeByIndexes(noteRow - 1, 0, 1, presentationColumns);
  sourceRange.merge(); sourceRange.values = [['Source: ' + input.source]];
  sourceRange.format = {font: {...FONT, color: '#555555', italic: true}, wrapText: true, rowHeight: 34};
  if (!specs.has(wb)) specs.set(wb, []);
  specs.get(wb).push({name: input.name, title: input.title, values, original: input.values, headerRow, bottom: noteRow, columns, widths, editable, formulas, kind, hyperlinks: []});
  return sheet;
}

async function publicWorkbook(old, qc) {
  const wb = Workbook.create();
  const overview = {name: 'QC_Scope', title: 'Reference-quality sensitivity reanalysis', source: 'Frozen publication tables and QC aggregate outputs', values: [
    ['Component', 'Scope'],
    ['Original primary analysis', 'All 4 main tables and 29 supplementary tables are retained with unchanged table values. Titles and notes follow the final submission Word files.'],
    ['QC sensitivity analyses', 'Six new publication tables, S30-S35, are added as rule-based technical sensitivities. There are 4 main and 35 supplementary tables. They do not replace the original primary analyses or establish clinically adjudicated artefacts.'],
    ['Clinical adjudication', 'Pending actual clinician review and signature. AI-assisted technical review is not independent human adjudication.'],
    ['qc_all', 'Computational sensitivity across all eligible short-flag intervals. Cases outside the 87-case review set were not added to the clinical review set.'],
    ['startup10', 'Excludes the first 600 s from the first valid reference. This may remove real early hypotension and is not an artefact-correction analysis.'],
    ['Data handling', 'Aggregate data only. No case or subject identifiers, review images or machine-local paths. This is a local archive candidate, not a public release.'],
    ['Numerical precision', 'Publication display strings remain as supplied. Numeric cells and unrounded aggregate estimates retain source precision; display precision does not round stored values.']
  ]};
  addTable(wb, overview, {kind: 'qc', widths: [28, 114]});
  const baselineTables = old.inputs.filter(t => /^(Main_Table_|Supp_Table_)/.test(t.name));
  assert.equal(baselineTables.length, 33);
  for (const input of [...baselineTables, ...qc]) {
    const copy = {...input, source: safeSource(input.source)};
    assertPublic(copy);
    addTable(wb, copy, {kind: baselineTables.includes(input) ? 'publication' : 'qc'});
  }
  for (const input of old.inputs.filter(t => !baselineTables.includes(t))) {
    assertPublic(input); addTable(wb, input, {kind: 'aggregate'});
  }
  const names = new Set(specs.get(wb).map(s => s.name));
  const aggregateMap = [
    ['cohort_qc_summary.csv', 'QC_Cohort_Unrounded'],
    ['events_rates_comparative.csv', 'QC_Event_Rates'],
    ['events_decomposition_comparative.csv', 'QC_AUC_Decomposition'],
    ['events_interval_absolute_summary_comparative.csv', 'QC_Interval_Absolute'],
    ['events_duration_comparative.csv', 'QC_Event_Duration'],
    ['events_robustness_comparative.csv', 'QC_Event_Robustness'],
    ['outcomes_denominators.csv', 'QC_Outcome_Denominators'],
    ['outcomes_all_models.csv', 'QC_Outcome_Models'],
    ['treatment_policy_comparison.csv', 'QC_Treatment_Comparison'],
    ['treatment_publication_table.csv', 'QC_Treatment_Unrounded']
  ];
  const aggregateSources = [];
  for (const [filename, sheetName] of aggregateMap) {
    const data = await csv(path.join(OUT, 'analysis/aggregate', filename));
    assert(!data.headers.some(h => PRIVATE_HEADER.test(h)), `Refusing case-level aggregate input ${filename}`);
    const keep = data.headers.map((h, j) => /(?:sha256|_path|_filename|r_verification)$/i.test(h) ? -1 : j).filter(j => j >= 0);
    const name = sheetName;
    assert(!names.has(name), `Duplicate aggregate worksheet ${name}`);
    const input = {name, title: filename.replace(/\.csv$/, '').replaceAll('_', ' '), values: data.values.map(r => keep.map(j => r[j])),
      notes: ['Unrounded aggregate source values. Policy labels distinguish the frozen original from rule-based QC sensitivities. These rows are not human clinical adjudication.'], source: filename};
    assertPublic(input); addTable(wb, input, {kind: 'aggregate'}); names.add(name); aggregateSources.push(filename);
  }
  return {wb, aggregateSources};
}

async function privateWorkbook(review, intervals, ai) {
  assert.equal(review.rows.length, 87, 'Expected exactly 87 original review cases');
  const cases = new Set(review.rows.map(r => r.case_id));
  const codes = new Set(review.rows.map(r => r.review_code));
  assert.equal(cases.size, 87); assert.equal(codes.size, 87);
  const aiKey = ai.headers.includes('review_code') ? 'review_code' : 'case_id';
  assert(ai.headers.includes(aiKey), 'AI review needs an explicit review_code or case_id');
  const aiIndex = new Map(ai.rows.map(r => [r[aiKey], r]));
  assert.equal(aiIndex.size, 87, 'AI technical review must be complete and unique for the 87 cases');
  assert(ai.rows.every(r => ai.headers.every(h => String(r[h]).trim() !== '')), 'AI technical review fields must be complete');
  for (const r of review.rows) assert(aiIndex.has(r[aiKey]), 'Unmatched original review case in AI technical review');
  const aiCounts = Object.fromEntries(['VISIBLE_NONPULSATILE_LOW', 'PLAUSIBLE_PULSATILE_LOW', 'UNCLEAR'].map(k => [k, ai.rows.filter(r => r.technical_visual_assessment === k).length]));
  assert.deepEqual(Object.values(aiCounts), [70, 3, 14], 'AI technical category counts changed');
  const priorityOne = review.rows.filter(r => Number(r.priority) === 1).length;
  assert.equal(priorityOne, 28, 'Priority-1 count changed');
  const fields = ['confirmed_artifact_start_rel_sec', 'confirmed_artifact_end_rel_sec', 'trusted_pulsatile_start_rel_sec', 'reviewer_and_date', 'review_notes'];
  assert(review.rows.every(r => fields.every(f => !r[f])), 'Existing clinical entries must not be overwritten by a blank template');
  const inside = intervals.rows.filter(r => cases.has(r.case_id));
  const outside = intervals.rows.filter(r => !cases.has(r.case_id));
  assert(outside.every(r => r.policy === 'qc_all'), 'Unexpected non-qc_all interval outside 87-case review');
  const qcAllCases = new Set(intervals.rows.filter(r => r.policy === 'qc_all').map(r => r.case_id)).size;
  assert.equal(qcAllCases, 355, 'qc_all case count changed; confirm scope before creating workbook');
  const wb = Workbook.create();
  const imageRoot = path.join(ROOT, '启动段质量审计_20261007/private_case_audit/waveform_review_images');
  const imageCopies = path.join(PRIVATE, 'waveform_review_images');
  await fs.mkdir(imageCopies, {recursive: true, mode: 0o700});
  const rows = [];
  for (const r of review.rows) {
    const imageFile = path.isAbsolute(r.image_filename) ? r.image_filename : path.join(imageRoot, path.basename(r.image_filename));
    assert(await exists(imageFile), 'Review image missing for an original review case');
    const sourceHash = digest(await read(imageFile));
    assert.equal(sourceHash, aiIndex.get(r[aiKey]).image_sha256, 'AI image hash differs from original review image');
    const imageCopy = path.join(imageCopies, path.basename(imageFile));
    if (await exists(imageCopy)) assert.equal(digest(await fs.readFile(imageCopy)), sourceHash, 'Do not overwrite a changed image copy');
    else await fs.copyFile(imageFile, imageCopy);
    assert.equal(digest(await fs.readFile(imageCopy)), sourceHash, 'Copied image hash mismatch');
    await fs.chmod(imageCopy, 0o600);
    const relativeImage = path.relative(PRIVATE, imageCopy).split(path.sep).join('/');
    rows.push([r.review_code, r.case_id, r.subject_id, '待临床裁决', null, null, null, null, null, null, null, relativeImage, r.priority, r.qc60_candidate_intervals, r.qc_all_candidate_intervals]);
  }
  const clinical = {name: '临床裁决_87例', title: '原87例临床裁决填写表', source: 'adjudication_list_87_cases.csv',
    values: [['复核编号', 'case_id', 'subject_id', '当前状态', '实际临床裁决', '确认伪影起点 (s)', '确认伪影终点 (s，右开)', '可信脉动起点 (s)', '实际裁决医师', '实际裁决日期', '实际临床意见', '原图相对链接', 'priority', 'qc60候选区间 (s)', 'qc_all候选区间 (s)'], ...rows],
    notes: ['黄色栏仅由实际临床医师填写，当前全部空白。AI技术复核、计算规则和预填技术建议不能代替独立临床裁决。',
      '时间均保留源文件的相对秒单位；不凭图形推算补填时间。候选区间不是已确认伪影区间。图像相对路径以本工作簿所在目录为起点。']};
  const clinicalSheet = addTable(wb, clinical, {kind: 'clinical', editable: [4,5,6,7,8,9,10], widths: [17,15,15,24,25,21,24,21,23,23,55,48,15,52,58],
    validations: {4: ['保留', '掩蔽确认伪影区间', '证据不足，继续复核']}});
  clinicalSheet.getRange('F5:H91').setNumberFormat('0.0');
  clinicalSheet.getRange('J5:J91').setNumberFormat('yyyy-mm-dd');
  // Completion never certifies a human review: entered records still need signature verification.
  const expressions = review.rows.map((_, i) => {
    const n = i + 5;
    return `=IF(AND(E${n}<>"",I${n}<>"",J${n}<>""),IF(E${n}="掩蔽确认伪影区间",IF(AND(ISNUMBER(F${n}),ISNUMBER(G${n}),G${n}>F${n}),"已填写，待签署核验","区间待补全"),"已填写，待签署核验"),"待临床裁决")`;
  });
  clinicalSheet.getRangeByIndexes(4, 3, 87, 1).formulas = expressions.map(x => [x]);
  specs.get(wb)[0].formulas.push({column: 3, expressions});
  const entryRange = clinicalSheet.getRange('E5:K5');
  let workflowTests = 0;
  try {
    for (const [entry, expectedStatus] of [
      [['保留', null, null, null, null, null, null], '待临床裁决'],
      [['保留', null, null, null, 'QA_TEST', 36526, null], '已填写，待签署核验'],
      [['掩蔽确认伪影区间', null, null, null, 'QA_TEST', 36526, null], '区间待补全'],
      [['掩蔽确认伪影区间', 0, 10, null, 'QA_TEST', 36526, null], '已填写，待签署核验'],
      [['掩蔽确认伪影区间', 10, 0, null, 'QA_TEST', 36526, null], '区间待补全']
    ]) {
      entryRange.values = [entry];
      assert.equal(clinicalSheet.getRange('D5').values[0][0], expectedStatus, 'Clinical workflow status test failed');
      workflowTests++;
    }
  } finally {entryRange.values = [[null, null, null, null, null, null, null]];}
  assert.equal(clinicalSheet.getRange('D5').values[0][0], '待临床裁决');
  clinicalSheet.getRange('L5:L91').format.font = {...FONT, color: '#0563C1'};
  specs.get(wb)[0].hyperlinks = rows.map((r, i) => ({cell: `L${i+5}`, target: r[11]}));
  for (const r of ai.rows) if (r.image_path && r.image_sha256) {
    const imageFile = path.isAbsolute(r.image_path) ? r.image_path : path.resolve(PRIVATE, r.image_path);
    assert.equal(digest(await fs.readFile(imageFile)), r.image_sha256, 'AI-reviewed image hash does not match its source');
  }
  const aiValues = [ai.headers, ...review.rows.map(r => ai.headers.map(h => {
    const value = aiIndex.get(r[aiKey])[h];
    return h === 'image_path' ? `waveform_review_images/${path.basename(value)}` : value;
  }))];
  addTable(wb, {name: 'AI技术复核_87例', title: 'AI辅助图像技术复核，非临床裁决', source: 'image_technical_review.csv', values: aiValues,
    notes: ['完整保留输入的AI技术复核字段，image_path转换为相对本工作簿的路径。任何建议、分类和不确定性均不代表实际麻醉医生已裁决。']}, {kind: 'private'});
  for (const [name, title, selected, note] of [
    ['候选区间_87例', '原87例中的规则性候选掩蔽区间', inside, '仅列原87例对应的候选区间。qc60与qc_all可能包含重叠区间，不能相加解释为独立异常。'],
    ['额外qc_all_仅计算', '原87例之外的qc_all计算性敏感性区间', outside, `qc_all总计${qcAllCases}例，其中本表为原87例之外的${new Set(outside.map(r=>r.case_id)).size}例、${outside.length}段。这些额外病例不在87例图像复核与临床裁决清单内。`]
  ]) addTable(wb, {name, title, source: 'candidate_mask_intervals.csv', values: [intervals.headers, ...selected.map(r => intervals.headers.map(h => r[h]))],
    notes: [note, 'technical_action与clinical_adjudication为源程序标签。PENDING_CLINICIAN表示未完成临床裁决。各区间起点包含、终点不包含，时间为相对秒。']}, {kind: 'private'});
  addTable(wb, {name: '原筛查_87例', title: '原87例筛查与技术建议，源数据保留', source: 'adjudication_list_87_cases.csv', values: review.values,
    notes: ['原始clinical_review_status及空白临床字段按源文件保留；本表不是实际临床填写入口。实际裁决填写在第一张工作表黄色栏。']}, {kind: 'private'});
  addTable(wb, {name: '使用说明', title: '受限工作簿范围与裁决边界', source: 'QC reanalysis inputs', values: [
    ['项目', '说明'], ['使用范围', '仅限本地受限研究资料。不得复制至公开归档包。'], ['复核范围', '原87例保持不变；额外短标记仅作计算性敏感性分析。'],
    ['AI短片段未见可信脉动', aiCounts.VISIBLE_NONPULSATILE_LOW], ['AI短片段可见可能脉动', aiCounts.PLAUSIBLE_PULSATILE_LOW], ['AI短片段无法明确判断', aiCounts.UNCLEAR], ['优先级1例数', priorityOne],
    ['分类限制', '上述70/3/14为所示短片段的AI技术分类，不是整例、每个候选区间或伪影的临床金标准。全部87例仍待实际临床裁决。'],
    ['临床输入', '第一张工作表黄色单元格全部留空，供实际医师填写。填入后仅显示待签署核验，不自动认定临床裁决完成。'],
    ['AI输入', '第二张工作表完整保留image_technical_review.csv的技术字段；AI结论和置信程度不等同于临床签署。'],
    ['区间', '候选掩蔽区间按qc60与qc_all分开标记，允许重叠。不得据此宣称全部低血压为伪影。'],
    ['图像', '87张指定PNG的原样副本放在waveform_review_images，链接相对本工作簿目录。副本与源图逐文件SHA-256相同。'],
    ['时间坐标', '候选区间保留源数据的相对秒坐标，起点包含、终点不包含。图像摘录分钟数是标题近似读数，不可替代精确候选区间边界。'],
    ['公开归档', 'final_workbook.xlsx仅包含原发表表和聚合敏感性结果。未进行任何对外发布。']
  ]}, {kind: 'private', widths: [28, 105]});
  return {wb, counts: {reviewCases: 87, aiReviewRows: ai.rows.length, candidateIntervals: intervals.rows.length, intervalsWithin87: inside.length,
    qcAllCases, extraQcAllCases: new Set(outside.map(r => r.case_id)).size, extraQcAllIntervals: outside.length, blankClinicalCells: 87 * 7,
    relativeImageHyperlinks: 87, copiedImages: 87, imageCopyHashesMatch: true, aiTechnicalCategories: aiCounts, priorityOne, clinicalWorkflowTests: workflowTests}};
}

function getElements(node, name) {return (node?.elements || []).filter(x => x.name?.split(':').at(-1) === name);}
function rootElement(xml, name) {return getElements(xml2js(xml, {compact: false}), name)[0];}

async function addNativeHyperlinks(file, expected) {
  const targets = expected.filter(s => s.hyperlinks.length);
  if (!targets.length) return;
  const zip = await JSZip.loadAsync(await fs.readFile(file));
  const book = rootElement(await zip.file('xl/workbook.xml').async('string'), 'workbook');
  const rel = rootElement(await zip.file('xl/_rels/workbook.xml.rels').async('string'), 'Relationships');
  const sheets = getElements(getElements(book, 'sheets')[0], 'sheet');
  for (const spec of targets) {
    const sheet = sheets.find(s => s.attributes.name === spec.name);
    const relationship = getElements(rel, 'Relationship').find(r => r.attributes.Id === sheet.attributes['r:id']);
    const part = relationship.attributes.Target.startsWith('/') ? relationship.attributes.Target.slice(1) : path.posix.join('xl', relationship.attributes.Target);
    const tree = xml2js(await zip.file(part).async('string'), {compact: false});
    const root = getElements(tree, 'worksheet')[0];
    root.attributes['xmlns:r'] = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
    const prefix = root.name.includes(':') ? root.name.split(':')[0] + ':' : '';
    const relPart = path.posix.join(path.posix.dirname(part), '_rels', path.posix.basename(part) + '.rels');
    const relTree = zip.file(relPart) ? xml2js(await zip.file(relPart).async('string'), {compact: false})
      : {elements: [{type: 'element', name: 'Relationships', attributes: {xmlns: 'http://schemas.openxmlformats.org/package/2006/relationships'}, elements: []}]};
    const relRoot = getElements(relTree, 'Relationships')[0];
    const links = {type: 'element', name: prefix + 'hyperlinks', elements: []};
    for (const [i, link] of spec.hyperlinks.entries()) {
      assert(!path.isAbsolute(link.target) && !LOCAL_PATH.test(link.target), 'Image links must be relative');
      const id = `rIdQCImage${i+1}`;
      assert(!relRoot.elements.some(r => r.attributes?.Id === id), 'Hyperlink relationship collision');
      links.elements.push({type: 'element', name: prefix + 'hyperlink', attributes: {ref: link.cell, 'r:id': id}});
      relRoot.elements.push({type: 'element', name: 'Relationship', attributes: {Id: id, Type: 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', Target: link.target, TargetMode: 'External'}});
    }
    const afterLinks = new Set(['printOptions', 'pageMargins', 'pageSetup', 'headerFooter', 'rowBreaks', 'colBreaks', 'customProperties', 'cellWatches', 'ignoredErrors', 'smartTags', 'drawing', 'legacyDrawing', 'legacyDrawingHF', 'picture', 'oleObjects', 'controls', 'webPublishItems', 'tableParts', 'extLst']);
    const index = root.elements.findIndex(e => afterLinks.has(e.name?.split(':').at(-1)));
    root.elements.splice(index < 0 ? root.elements.length : index, 0, links);
    zip.file(part, js2xml(tree)); zip.file(relPart, js2xml(relTree));
  }
  await fs.writeFile(file, await zip.generateAsync({type: 'nodebuffer', compression: 'DEFLATE'}));
}

async function inspectExport(file, expected, restricted) {
  const zip = await JSZip.loadAsync(await fs.readFile(file));
  const book = rootElement(await zip.file('xl/workbook.xml').async('string'), 'workbook');
  const rel = rootElement(await zip.file('xl/_rels/workbook.xml.rels').async('string'), 'Relationships');
  const targets = new Map(getElements(rel, 'Relationship').map(r => [r.attributes.Id, r.attributes.Target]));
  const sharedFile = zip.file('xl/sharedStrings.xml');
  const shared = sharedFile ? getElements(rootElement(await sharedFile.async('string'), 'sst'), 'si').map(textNode) : [];
  const results = [];
  for (const sheet of getElements(getElements(book, 'sheets')[0], 'sheet')) {
    const spec = expected.find(s => s.name === sheet.attributes.name);
    assert(spec, 'Unexpected exported worksheet');
    let target = targets.get(sheet.attributes['r:id']);
    target = target.startsWith('/') ? target.slice(1) : path.posix.join('xl', target);
    const xml = await zip.file(target).async('string');
    const root = rootElement(xml, 'worksheet');
    assert(!getElements(root, 'sheetProtection').length, 'Clinical editing must not be blocked by sheet protection');
    const cells = new Map(); let numbers = 0, strings = 0, booleans = 0, formulas = 0;
    for (const row of getElements(getElements(root, 'sheetData')[0], 'row')) for (const c of getElements(row, 'c')) {
      const type = c.attributes.t;
      assert(type !== 'e', `Exported formula error in ${spec.name}`);
      let value = textNode(getElements(c, 'v')[0] || {elements: []});
      if (type === 's') {value = shared[Number(value)]; strings++;}
      else if (type === 'inlineStr') {value = textNode(getElements(c, 'is')[0]); strings++;}
      else if (type === 'b') {value = value === '1'; booleans++;}
      else if (type === 'str') strings++;
      else if (value !== '') {value = Number(value); numbers++;}
      else value = null;
      const formula = getElements(c, 'f')[0]; if (formula) formulas++;
      cells.set(c.attributes.r, {value, formula: formula ? textNode(formula) : null, type});
    }
    for (let i = 0; i < spec.values.length; i++) for (let j = 0; j < spec.columns; j++) {
      const cell = cells.get(`${col(j)}${spec.headerRow + i}`);
      const expectedValue = spec.values[i][j];
      const formulaSpec = i > 0 && spec.formulas.find(f => f.column === j);
      if (formulaSpec) {
        assert.equal(cell?.formula, formulaSpec.expressions[i - 1].replace(/^=/, ''), `Formula mismatch in ${spec.name}`);
        if (spec.kind === 'clinical' && j === 3) assert.equal(cell.value, '待临床裁决');
      } else if (typeof expectedValue === 'number') {
        assert.equal(typeof cell?.value, 'number', `Number exported as text in ${spec.name}`);
        assert(Math.abs(cell.value - expectedValue) <= Math.max(1e-13, Math.abs(expectedValue) * 5e-14), `Numeric mismatch in ${spec.name}`);
        if (Number.isInteger(expectedValue)) assert.equal(cell.value, expectedValue, `Count mismatch in ${spec.name}`);
      } else assert.equal(cell?.value ?? null, expectedValue, `Value mismatch in ${spec.name} ${col(j)}${spec.headerRow+i}`);
      if (spec.editable.includes(j) && i > 0) assert.equal(cell?.value ?? null, null, 'Clinical fields must remain blank');
    }
    assert.equal(cells.get('A2')?.value, spec.title, 'Exported title mismatch');
    const nativeLinks = getElements(getElements(root, 'hyperlinks')[0], 'hyperlink');
    assert.equal(nativeLinks.length, spec.hyperlinks.length, 'Native hyperlink count mismatch');
    if (nativeLinks.length) {
      const relPart = path.posix.join(path.posix.dirname(target), '_rels', path.posix.basename(target) + '.rels');
      const sheetRelationships = rootElement(await zip.file(relPart).async('string'), 'Relationships');
      for (const expectedLink of spec.hyperlinks) {
        const link = nativeLinks.find(l => l.attributes.ref === expectedLink.cell);
        const r = getElements(sheetRelationships, 'Relationship').find(r => r.attributes.Id === link?.attributes['r:id']);
        assert.equal(r?.attributes.Target, expectedLink.target, 'Relative hyperlink target mismatch');
        assert(await exists(path.resolve(PRIVATE, r.attributes.Target)), 'Copied hyperlink target missing');
      }
    }
    if (!restricted) {
      assert(![...cells.values()].some(c => LOCAL_PATH.test(String(c.value ?? ''))), 'Public local-path leak');
      assert(!getElements(root, 'hyperlinks').length, 'Public workbook must not link to private paths');
    }
    results.push({sheet: spec.name, dataRows: spec.values.length - 1, columns: spec.columns, numericCells: numbers, textCells: strings, booleanCells: booleans, formulas, hyperlinks: nativeLinks.length, valueAndTypeCheck: 'PASS'});
  }
  assert.equal(results.length, expected.length);
  if (!restricted) for (const [name, entry] of Object.entries(zip.files)) {
    if (!entry.dir && /\.(?:xml|rels)$/.test(name)) assert(!LOCAL_PATH.test(await entry.async('string')), 'Machine-local path in public OOXML');
  }
  return results;
}

async function renderAll(wb, directory) {
  await fs.mkdir(directory, {recursive: true});
  const renders = [];
  for (const spec of specs.get(wb)) {
    const batches = [];
    let start = 0, width = 0;
    for (let j = 0; j < spec.columns; j++) {
      if (width + spec.widths[j] > 175 && j > start) {batches.push([start, j - 1]); start = j; width = 0;}
      width += spec.widths[j];
    }
    batches.push([start, spec.columns - 1]);
    for (let b = 0; b < batches.length; b++) {
      const [a, z] = batches[b];
      const range = `${col(a)}${b === 0 ? 1 : spec.headerRow}:${col(z)}${Math.min(spec.bottom, spec.headerRow + 7)}`;
      const blob = await wb.render({sheetName: spec.name, range, scale: 1.3, format: 'png'});
      const filename = `${spec.name}__${b+1}.png`;
      await fs.writeFile(path.join(directory, filename), new Uint8Array(await blob.arrayBuffer()));
      renders.push({sheet: spec.name, range, filename});
    }
    if (spec.bottom > spec.headerRow + 7) {
      const begin = Math.max(spec.headerRow + 1, spec.headerRow + spec.values.length - 3);
      const range = `A${begin}:${col(batches[0][1])}${spec.bottom}`;
      const blob = await wb.render({sheetName: spec.name, range, scale: 1, format: 'png'});
      const filename = `${spec.name}__tail.png`;
      await fs.writeFile(path.join(directory, filename), new Uint8Array(await blob.arrayBuffer()));
      renders.push({sheet: spec.name, range, filename});
    }
    console.log(JSON.stringify({renderedSheet: spec.name}));
  }
  const thumbnails = [];
  for (const r of renders) {
    const image = sharp(path.join(directory, r.filename));
    const meta = await image.metadata();
    const buffer = await image.resize({width: 720, height: 540, fit: 'inside', withoutEnlargement: true}).toBuffer();
    const resized = await sharp(buffer).metadata();
    thumbnails.push({buffer, width: resized.width, height: resized.height, sourceWidth: meta.width, sourceHeight: meta.height, ...r});
  }
  const contacts = [];
  for (let offset = 0; offset < thumbnails.length; offset += 6) {
    const group = thumbnails.slice(offset, offset + 6);
    const composites = group.map((r, i) => ({input: r.buffer, left: (i % 2) * 740 + 10, top: Math.floor(i / 2) * 560 + 10}));
    const filename = `contact_${String(offset / 6 + 1).padStart(2,'0')}.png`;
    await sharp({create: {width: 1480, height: 560 * Math.ceil(group.length / 2), channels: 3, background: '#FFFFFF'}}).composite(composites).png().toFile(path.join(directory, filename));
    contacts.push({filename, views: group.map(r => r.filename)});
  }
  return {renders, contacts};
}

async function finish(wb, file, directory, restricted) {
  wb.recalculate();
  for (const spec of specs.get(wb)) {
    const found = await wb.inspect({kind: 'table', range: `'${spec.name}'!A${spec.headerRow}:${col(Math.min(5, spec.columns - 1))}${Math.min(spec.headerRow + 3, spec.headerRow + spec.values.length - 1)}`, include: 'values,formulas', maxChars: 2000});
    // Private cell contents are never written to QA logs or stdout.
    assert(found.ndjson, 'Artifact inspection returned no data');
  }
  const scan = await wb.inspect({kind: 'match', searchTerm: ERROR.source, options: {useRegex: true, maxResults: 300}, summary: 'Final formula error scan'});
  await fs.mkdir(directory, {recursive: true});
  await fs.writeFile(path.join(directory, 'formula_scan.ndjson'), scan.ndjson);
  const staged = path.join(directory, path.basename(file) + '.pending');
  const output = await SpreadsheetFile.exportXlsx(wb);
  await output.save(staged);
  // The supported artifact API has no range hyperlink setter; add only native link relationships.
  await addNativeHyperlinks(staged, specs.get(wb));
  if (restricted) await fs.chmod(staged, 0o600);
  const exported = await inspectExport(staged, specs.get(wb), restricted);
  const visual = await renderAll(wb, path.join(directory, 'renders'));
  await fs.writeFile(path.join(directory, 'verification.json'), JSON.stringify({status: 'PASS_STRUCTURAL', file: path.basename(file), sha256: digest(await fs.readFile(staged)), sheets: exported,
    allSheetsRendered: true, visualReview: 'PENDING_HUMAN_AGENT_IMAGE_INSPECTION', ...visual}, null, 2) + '\n');
  return {file: path.basename(file), staged, sheets: exported.length, numericCells: exported.reduce((n, s) => n + s.numericCells, 0), formulas: exported.reduce((n, s) => n + s.formulas, 0), renders: visual.renders.length};
}

function selfTest() {
  assert.equal(typed('0'), 0); assert.equal(typed(''), null); assert.equal(typed('00017', 'case_id'), '00017');
  assert.equal(typed('87'), 87); assert.equal(typed('2,435'), 2435); assert.equal(typed('0.125'), 0.125);
  assert.equal(typed('5 [3, 8]'), '5 [3, 8]'); assert.equal(typed('False'), false);
  assert.equal(col(0), 'A'); assert.equal(col(25), 'Z'); assert.equal(col(26), 'AA');
  assert.throws(() => assertPublic({name:'Bad',title:'Bad',values:[['case_id'],[123]]}));
  assert.throws(() => assertPublic({name:'Bad',title:'Bad',values:[['Source'],[('/' + 'Users' + '/example/private')]]}));
  assertPublic({name:'Good',title:'Aggregate',values:[['Cases','Estimate'],[87,0.125]],source:'qc.csv'});
  assert.equal(numberFormat('0.000', 0), '0.000');
  assert.deepEqual(numericTokens('18 381'), numericTokens('18,381'));
  assert.deepEqual(numericTokens('<1 × 10−12'), numericTokens('<1 × 10-12'));
  console.log(JSON.stringify({selfTests: 'PASS', inputGate: 'Missing required inputs prevent both exports'}));
}

async function main() {
  const wantPublic = !process.argv.includes('--private-only');
  const wantPrivate = !process.argv.includes('--public-only');
  assert(wantPublic || wantPrivate, 'Select one target or neither target flag');
  const required = [path.join(PRIVATE, 'adjudication_list_87_cases.csv'), path.join(PRIVATE, 'candidate_mask_intervals.csv'),
    path.join(PRIVATE, 'image_technical_review.csv'), path.join(OUT, 'analysis/aggregate/qc_publication_tables.json')];
  const missing = [];
  for (const f of (wantPrivate ? required : [required[3]])) if (!await exists(f)) missing.push(path.relative(OUT, f));
  if (process.argv.includes('--check-inputs')) {console.log(JSON.stringify({ready: !missing.length, missing})); return;}
  assert(!missing.length, 'Required inputs not ready: ' + missing.join(', '));
  const privateOutput = path.join(PRIVATE, '临床裁决清单_87例.xlsx');
  const publicOutput = path.join(OUT, '归档候选/outputs/final_workbook.xlsx');
  const refreshHashes = new Map();
  for (const [wanted, file, qaDir] of [[wantPrivate, privateOutput, path.join(PRIVATE, 'workbook_qa')], [wantPublic, publicOutput, path.join(QA, 'public')]]) {
    if (!wanted || !await exists(file)) continue;
    if (!process.argv.includes('--refresh-generated')) fail('Workbook exists. Preserve edits before a controlled refresh.');
    const verified = JSON.parse(await fs.readFile(path.join(qaDir, 'verification.json'), 'utf8'));
    const hash = digest(await fs.readFile(file));
    assert.equal(hash, verified.sha256, 'Workbook changed since generation. Refusing to overwrite possible clinician edits.');
    refreshHashes.set(file, hash);
  }
  await fs.mkdir(QA, {recursive: true});
  const runtimeLink = path.join(QA, 'node_modules');
  if (!await exists(runtimeLink)) await fs.symlink(BUNDLE, runtimeLink, 'dir');
  const old = wantPublic ? await baseline() : {inputs: [], refreshed: [], wordPresentationDifferences: []};
  const qcPayload = JSON.parse((await read(required[3])).toString('utf8'));
  const qc = Array.isArray(qcPayload) ? qcPayload : qcPayload.tables;
  assert(Array.isArray(qc) && qc.length > 0, 'QC publication tables must be nonempty');
  assert.equal(qc.length, 6, 'Expected six new QC publication tables, S30-S35');
  assert.deepEqual(qc.map(t => t.name).sort(), Array.from({length: 6}, (_, i) => `Supp_Table_S${30+i}`).sort(), 'QC publication tables must be S30-S35');
  for (const input of qc) {
    assert(input.name && input.title && input.values?.length > 1, 'Malformed QC publication table');
    assert(!old.inputs.some(t => t.name === input.name), 'QC tables cannot replace baseline worksheets');
    assertPublic(input);
  }
  let privateBook, publicBook, privateResult, publicResult;
  if (wantPrivate) {
    const review = await csv(required[0]);
    const intervals = await csv(required[1]);
    const ai = await csv(required[2]);
    privateBook = await privateWorkbook(review, intervals, ai);
    privateResult = await finish(privateBook.wb, privateOutput, path.join(PRIVATE, 'workbook_qa'), true);
  }
  if (wantPublic) {
    publicBook = await publicWorkbook(old, qc);
    publicResult = await finish(publicBook.wb, publicOutput, path.join(QA, 'public'), false);
  }
  for (const [file, hash] of provenance) assert.equal(digest(await fs.readFile(file)), hash, `Source changed during build: ${path.basename(file)}`);
  for (const [file, hash] of refreshHashes) assert.equal(digest(await fs.readFile(file)), hash, 'Workbook changed during build; refusing overwrite');
  if (privateResult) {await fs.rename(privateResult.staged, privateOutput); delete privateResult.staged;}
  if (publicResult) {await fs.mkdir(path.dirname(publicOutput), {recursive: true}); await fs.rename(publicResult.staged, publicOutput); delete publicResult.staged;}
  const manifestFile = path.join(QA, 'build_manifest.json');
  const previous = await exists(manifestFile) ? JSON.parse(await fs.readFile(manifestFile, 'utf8')) : {};
  const outputs = [...(previous.outputs || []).filter(o => o.file !== privateResult?.file && o.file !== publicResult?.file), ...[privateResult, publicResult].filter(Boolean)];
  await fs.writeFile(manifestFile, JSON.stringify({...previous, status: 'PASS_STRUCTURAL_PENDING_VISUAL_REVIEW', outputs,
    ...(publicResult ? {baselinePublicationTables: 33, qcPublicationTables: qc.length, baselineValuesUnchanged: true, latestWordMetadataRefreshed: old.refreshed,
      wordPresentationDifferences: old.wordPresentationDifferences, aggregateSources: publicBook.aggregateSources} : {}),
    ...(privateBook?.counts || {}), sourceHashes: {...previous.sourceHashes, ...Object.fromEntries([...provenance].map(([f,h]) => [path.relative(ROOT,f),h]))},
    clinicalAdjudication: 'PENDING_ACTUAL_CLINICIAN', externalPublication: false}, null, 2) + '\n');
  console.log(JSON.stringify({outputs, originalTables: 33, qcTables: qc.length, ...(privateBook?.counts || {})}));
}

if (process.argv.includes('--self-test')) selfTest();
else await main();
