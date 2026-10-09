"""Package previously viewed image observations, with read-only source checks.

This is not an image classifier and does not perform clinical adjudication.
--write exclusively creates the two owned evidence outputs.
--check (the default) verifies existing outputs without writing any files.
"""
import os
import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import struct
from zoneinfo import ZoneInfo
ROOT = Path(os.environ['IOH_PROJECT_ROOT']).expanduser().resolve()
OUT = Path(os.environ.get('IOH_QC_OUTPUT_ROOT', str(ROOT / '独立质控重分析_20261007'))).expanduser().resolve()
SOURCE_DIR = ROOT / '启动段质量审计_20261007' / 'private_case_audit'
SOURCE_CSV = SOURCE_DIR / 'clinical_review_list_87_cases.csv'
IMAGE_DIR = SOURCE_DIR / 'waveform_review_images'
CSV_OUT = OUT / 'data_restricted' / 'image_technical_review.csv'
JSON_OUT = OUT / 'qa' / 'image_review_summary.json'
STATIC = json.loads(Path(os.environ['IOH_IMAGE_REVIEW_INPUT_JSON']).read_text())
REVIEWED_BY = 'AI technical review'
CLINICAL_STATUS = 'PENDING_CLINICIAN'
COMMON_UNCERTAINTY = 'Five-second excerpt only; full >=60-second interval, exact coverage, quantiles and synchronized MAP eligibility are not validated. No clinical cause/signoff is assigned.'
CATEGORY_ZH = {'VISIBLE_NONPULSATILE_LOW': '所示低压片段无明确持续动脉搏动形态；不等同于已确认伪差', 'PLAUSIBLE_PULSATILE_LOW': '所示低压片段保留可信搏动形态；绝对压力校准仍未验证', 'UNCLEAR': '所示图像不足以明确判断目标低压片段的搏动形态'}
FIELDS = ['review_code', 'priority', 'image_filename', 'image_sha256', 'image_path', 'view_order', 'excerpt_min_from_image', 'assessment_scope', 'technical_visual_assessment', 'technical_visual_assessment_zh', 'evidence', 'uncertainty', 'target_low_map_excerpt_alignment', 'mask_interval_validation', 'clinical_artifact_confirmed', 'interval_action', 'reviewed_by', 'clinical_review_status']

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def validate_sources():
    manifest = STATIC['manifest']
    require(sha256(SOURCE_CSV) == manifest['source_list_sha256'], 'Source review-list hash changed; stop without overwriting.')
    with SOURCE_CSV.open(encoding='utf-8-sig', newline='') as handle:
        current = [{key: row[key] for key in ('review_code', 'priority', 'image_filename')} for row in csv.DictReader(handle)]
    expected = [{key: row[key] for key in ('review_code', 'priority', 'image_filename')} for row in manifest['rows']]
    require(current == expected, 'Exact source row/order mapping mismatch.')
    codes = [row['review_code'] for row in current]
    require(len(codes) == len(set(codes)) == 87, 'Expected 87 unique source codes.')
    require(set(STATIC['reviews']) == set(codes), 'Incomplete/extra observations.')
    require(set(STATIC['alignment_flags']) == set(codes), 'Alignment flags mismatch.')
    viewed = STATIC['viewed_order']
    require(len(viewed) == len(set(viewed)) == 87 and set(viewed) == set(codes), 'Viewing record must include all 87 images exactly once.')
    priority_one = [row['review_code'] for row in current if row['priority'] == '1']
    require(len(priority_one) == 28, 'Expected 28 priority-1 cases.')
    require(set(viewed[:28]) == set(priority_one), 'Priority-1 images were not all first in viewing record.')
    require(Counter((row['priority'] for row in current)) == {'1': 28, '2': 59}, 'Priority counts changed.')
    dimensions = Counter()
    for row in manifest['rows']:
        filename = row['image_filename']
        require(Path(filename).name == filename and filename.endswith('.png'), 'Unsafe or non-PNG image filename.')
        path = IMAGE_DIR / filename
        require(sha256(path) == row['sha256'], 'Source PNG hash changed: ' + row['review_code'])
        with path.open('rb') as handle:
            header = handle.read(24)
        require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'Unexpected PNG header: ' + row['review_code'])
        width, height = struct.unpack('>II', header[16:24])
        dimensions[str(width) + 'x' + str(height)] += 1
    return dict(dimensions)

def build_rows():
    order = {code: index + 1 for index, code in enumerate(STATIC['viewed_order'])}
    rows = []
    for source in STATIC['manifest']['rows']:
        code = source['review_code']
        review = STATIC['reviews'][code]
        require(review['category'] in CATEGORY_ZH, 'Unknown morphology category.')
        require(review['evidence'].strip() and review['uncertainty'].strip(), 'Empty case-specific evidence or uncertainty: ' + code)
        rows.append({'review_code': code, 'priority': source['priority'], 'image_filename': source['image_filename'], 'image_sha256': source['sha256'], 'image_path': str(IMAGE_DIR / source['image_filename']), 'view_order': str(order[code]), 'excerpt_min_from_image': format(review['excerpt_min'], '.2f'), 'assessment_scope': 'DISPLAYED_5S_EXCERPT_PLUS_OVERVIEW_NOT_WHOLE_CASE', 'technical_visual_assessment': review['category'], 'technical_visual_assessment_zh': CATEGORY_ZH[review['category']], 'evidence': review['evidence'], 'uncertainty': review['uncertainty'] + ' ' + COMMON_UNCERTAINTY, 'target_low_map_excerpt_alignment': STATIC['alignment_flags'][code], 'mask_interval_validation': 'NOT_ESTABLISHED_FROM_IMAGE', 'clinical_artifact_confirmed': 'NOT_ADJUDICATED', 'interval_action': 'NO_MASK_AUTHORIZED_BY_THIS_REVIEW', 'reviewed_by': REVIEWED_BY, 'clinical_review_status': CLINICAL_STATUS})
    return rows

def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8-sig')

def build_summary(rows, dimensions, csv_hash, generated_at):
    result = copy.deepcopy(STATIC['summary_spec'])
    counts = Counter((row['technical_visual_assessment'] for row in rows))
    by_priority = {priority: dict(Counter((row['technical_visual_assessment'] for row in rows if row['priority'] == priority))) for priority in ('1', '2')}
    categories = {category: [row['review_code'] for row in rows if row['technical_visual_assessment'] == category] for category in CATEGORY_ZH}
    flags = sorted(set((row['target_low_map_excerpt_alignment'] for row in rows)))
    result.update({'schema_version': '1.0', 'output_generated_at': generated_at, 'timestamp_meaning': 'Actual artifact generation time; not a human clinical review date', 'clinical_review_date': None, 'source_list_path': str(SOURCE_CSV), 'source_image_directory': str(IMAGE_DIR), 'source_list_sha256_before_review': STATIC['manifest']['source_list_sha256'], 'source_list_sha256_at_verification': sha256(SOURCE_CSV), 'counts': {'source_cases': len(rows), 'actual_pngs_individually_viewed': len(STATIC['viewed_order']), 'unique_review_codes': len(set((row['review_code'] for row in rows))), 'clinical_pending': len(rows), 'human_clinical_signoffs': 0, 'clinically_confirmed_artifact_cases': 0, 'clinically_confirmed_artifact_cases_count_note': 'Zero signoffs, not evidence that no artifacts exist', 'morphology': dict(counts), 'by_priority': by_priority, 'image_dimensions': dimensions}, 'case_codes_by_morphology': categories, 'alignment_limitations': {flag: {'count': sum((row['target_low_map_excerpt_alignment'] == flag for row in rows)), 'review_codes': [row['review_code'] for row in rows if row['target_low_map_excerpt_alignment'] == flag]} for flag in flags}, 'viewed_order': STATIC['viewed_order'], 'source_image_manifest': STATIC['manifest']['rows'], 'outputs': {'case_review_csv': str(CSV_OUT), 'case_review_csv_sha256': csv_hash, 'summary_json': str(JSON_OUT), 'packaging_script': str(Path(__file__).resolve()), 'packaging_script_sha256': sha256(Path(__file__)), 'case_observations_language': 'English; category labels also provided in Chinese'}, 'verification': {'exact_source_list_mapping_and_order': True, 'all_87_png_hashes_unchanged_since_pre_review_baseline': True, 'source_list_hash_unchanged_since_pre_review_baseline': True, 'all_87_observations_complete': True, 'priority_28_viewed_first': True, 'all_clinical_statuses_pending': True, 'no_patient_or_subject_id_columns_copied_to_output': True, 'no_artifact_interval_boundaries_or_human_signoff_assigned': True, 'write_policy': 'Exclusive creation; refuse to overwrite either evidence output'}})
    return result

def check_outputs():
    dimensions = validate_sources()
    rows = build_rows()
    expected_csv = csv_bytes(rows)
    require(CSV_OUT.read_bytes() == expected_csv, 'CSV differs from frozen reviewed observations.')
    with CSV_OUT.open(encoding='utf-8-sig', newline='') as handle:
        actual = list(csv.DictReader(handle))
    require(actual == rows, 'CSV round-trip mismatch.')
    require(all((row['reviewed_by'] == REVIEWED_BY and row['clinical_review_status'] == CLINICAL_STATUS for row in actual)), 'Reviewer/status mismatch.')
    require(not {'case_id', 'subject_id', 'reviewer_and_date'} & set(actual[0]), 'Unexpected identifying or human-signoff column.')
    summary = json.loads(JSON_OUT.read_text(encoding='utf-8'))
    parsed_time = datetime.fromisoformat(summary['output_generated_at'])
    require(parsed_time.utcoffset() is not None, 'Generation timestamp has no timezone.')
    expected_summary = build_summary(rows, dimensions, hashlib.sha256(expected_csv).hexdigest(), summary['output_generated_at'])
    require(summary == expected_summary, 'Summary differs from reproducible evidence.')
    print(json.dumps({'status': 'PASS', 'reviewed_images': 87, 'priority_first': 28, 'morphology_counts': summary['counts']['morphology'], 'clinical_pending': 87, 'source_hashes_unchanged': 88, 'csv_roundtrip_and_frozen_content': 'PASS', 'summary_reconstruction': 'PASS', 'case_review_csv_sha256': sha256(CSV_OUT)}, ensure_ascii=True))

def write_outputs():
    require(not CSV_OUT.exists() and (not JSON_OUT.exists()), 'Owned output already exists; use --check. Overwrite is forbidden.')
    dimensions = validate_sources()
    rows = build_rows()
    content = csv_bytes(rows)
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    summary = build_summary(rows, dimensions, hashlib.sha256(content).hexdigest(), timestamp)
    with CSV_OUT.open('xb') as handle:
        CSV_OUT.chmod(384)
        handle.write(content)
    with JSON_OUT.open('x', encoding='utf-8') as handle:
        JSON_OUT.chmod(384)
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    check_outputs()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--write', action='store_true', help='Create the two owned outputs exclusively.')
    action.add_argument('--check', action='store_true', help='Read-only verification (default).')
    args = parser.parse_args()
    if args.write:
        write_outputs()
    else:
        check_outputs()
if __name__ == '__main__':
    main()
