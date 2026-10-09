"""Apply publication wording without changing statistical estimates."""
GENERAL = ('Run end is the end of the observed low-pressure run, which may be caused by '
           'missing reference or recording end rather than confirmed recovery. '
           'Post-run persistence includes only observed normal-reference time with low display. '
           'Missing-data or recording endpoints contribute zero observed time, not confirmed '
           'zero clinical persistence. Timing means condition on represented phase-episode pairs.')
EARLY = ('Early-adjustment exclusion removes adjustments before 10 min from analytic-window '
         'start without masking the reference trajectory; it differs from startup-window '
         'masking measured from the first finite reference MAP.')
STARTUP = ('Startup-window exclusion masks reference values for 600 s from the first finite '
           'reference MAP and then re-evaluates event support; it is not the early-adjustment '
           'exclusion from analytic-window start in Table S9b.')


def apply(groups):
    for group in groups:
        for panel in group['panels']:
            if group['number'] == 4:
                panel['values'] = [[value.replace('before recovery', 'before run end').replace('Stale low display, min', 'Observed post-run persistence, min') for value in row]
                                   for row in panel['values']]
            elif group['number'] == 9:
                panel['values'] = [[value.replace('Exclude first 10 min', 'Exclude adjustments before 10 min')
                                    for value in row] for row in panel['values']]
            elif group['number'] == 16:
                panel['values'] = [[value.replace('N=', 'n=') for value in row] for row in panel['values']]
        if group['number'] == 4 and GENERAL not in group['panels'][1]['notes']:
            group['panels'][1]['notes'].append(GENERAL)
        if group['number'] == 9 and EARLY not in group['panels'][1]['notes']:
            group['panels'][1]['notes'].append(EARLY)
        if group['number'] == 15 and STARTUP not in group['panels'][0]['notes'][0]:
            group['panels'][0]['notes'][0] += ' ' + STARTUP
