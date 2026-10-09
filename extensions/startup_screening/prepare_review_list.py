import os
import sys
if os.environ.get('IOH_QC_DEPENDENCIES'):
    sys.path.insert(0, os.environ['IOH_QC_DEPENDENCIES'])
import pandas as pd
from run_numeric_audit import OUT, locate
from run_waveform_audit import read_wave_prefix, bin_wave, make_plot

def main():
    p = OUT / 'private_case_audit'
    numeric = pd.read_csv(p / 'startup_numeric_case_audit.csv')
    wave = pd.read_csv(p / 'full_waveform_case_audit.csv')
    old = pd.read_csv(p / 'waveform_case_audit.csv')
    mapping = old.set_index('case_id').review_code.to_dict()
    work = numeric.merge(wave[['case_id', 'nonpulsatile_low_cumulative_sec', 'nonpulsatile_low_longest_run_sec', 'pulse_like_low_cumulative_sec']], on='case_id')
    work = work[work.targeted_review | work.nonpulsatile_low_cumulative_sec.ge(60)].copy()
    assert len(work) == 87
    extra = []
    next_code = 116
    for _, row in work.iterrows():
        if row.case_id in mapping:
            continue
        code = f'QC{next_code:03d}'
        next_code += 1
        source = locate(row.waveform_tid)
        t, v, interval = read_wave_prefix(source, row.start_abs_sec + row.first_valid_rel_sec + 600)
        b = bin_wave(t, v, row.start_abs_sec, row.first_valid_rel_sec, interval)
        make_plot(code, 'Additional waveform screen', row, b, t, v, p / 'waveform_review_images' / f'{code}.png')
        mapping[row.case_id] = code
        extra.append({'case_id': int(row.case_id), 'review_code': code, 'review_group': 'Additional waveform screen'})
    if extra:
        pd.DataFrame(extra).to_csv(p / 'additional_waveform_image_index.csv', index=False)
    work['review_code'] = work.case_id.map(mapping)
    work['priority'] = work.nonpulsatile_low_longest_run_sec.ge(60).map({True: 1, False: 2})
    work['clinical_review_status'] = 'NOT_ADJUDICATED'
    work['confirmed_artifact_start_rel_sec'] = ''
    work['confirmed_artifact_end_rel_sec'] = ''
    work['trusted_pulsatile_start_rel_sec'] = ''
    work['reviewer_and_date'] = ''
    work['review_notes'] = ''
    work['image_filename'] = work.review_code + '.png'
    keep = ['review_code', 'priority', 'case_id', 'subject_id', 'first_valid_rel_sec', 'low_plateau_flag', 'low_pulse_pressure_flag', 'inverted_sbp_dbp_flag', 'nonpulsatile_low_cumulative_sec', 'nonpulsatile_low_longest_run_sec', 'pulse_like_low_cumulative_sec', 'image_filename', 'clinical_review_status', 'confirmed_artifact_start_rel_sec', 'confirmed_artifact_end_rel_sec', 'trusted_pulsatile_start_rel_sec', 'reviewer_and_date', 'review_notes']
    work.sort_values(['priority', 'review_code'])[keep].to_csv(p / 'clinical_review_list_87_cases.csv', index=False)
    print({'clinical_review_cases': len(work), 'priority1_cases': int(work.priority.eq(1).sum()), 'additional_diagnostic_images': len(extra)})
if __name__ == '__main__':
    main()
