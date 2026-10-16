"""Record-level pump adjustment and display analysis; no treatment-effect inference."""
import numpy as np


def block_starts(times, merge_sec):
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) < 0) or merge_sec < 0:
        raise ValueError('Unsorted times or invalid merge window')
    anchors = []
    for i, time in enumerate(times):
        if not anchors or time > times[anchors[-1]] + merge_sec + 1e-9:
            anchors.append(i)
    return anchors


def transitions(times, values, sustain_sec=5):
    times, values = np.asarray(times), np.asarray(values)
    out = []
    for i in range(1, len(times)):
        before, after = values[i-1:i+1]
        if (not np.isfinite([before, after]).all() or min(before, after) < 0
                or not 0 < times[i] - times[i-1] <= 5 or after <= before + 1e-8):
            continue
        if sustain_sec:
            j = i
            while j < len(times)-1 and times[j] < times[i] + sustain_sec - 1e-9:
                j += 1
                if (times[j] - times[j-1] > 5 or not np.isfinite(values[j])
                        or values[j] <= before + 1e-8):
                    break
            if (times[j] < times[i] + sustain_sec - 1e-9
                    or np.any(~np.isfinite(values[i:j+1]))
                    or np.any(values[i:j+1] <= before + 1e-8)
                    or np.any(np.diff(times[i:j+1]) > 5)):
                continue
        out.append({'time_sec': float(times[i]),
                    'kind': 'observed_zero_to_positive' if before == 0 else 'observed_rate_increase',
                    'previous_rate': float(before), 'new_rate': float(after)})
    return out


def display_at(values, index, step_bins, phase_bins):
    if not 0 <= phase_bins < step_bins or step_bins < 1:
        raise ValueError('Invalid sampling phase')
    sampled = np.arange(phase_bins, index + 1, step_bins)
    valid = sampled[np.isfinite(values[sampled])]
    if not len(valid):
        return np.nan, -1
    held = int(valid[-1])
    return float(values[held]), held


def event_state(values, time_sec, step_bins, phase_bins, threshold=65):
    values = np.asarray(values, dtype=float)
    index = int(np.floor((time_sec + 1e-9) / 10)) - 1
    if index < 5 or index >= len(values):
        return None
    if not np.isfinite(values[index]) or np.isfinite(values[index-5:index+1]).sum() < 5:
        return None
    reference = float(values[index])
    display, held = display_at(values, index, step_bins, phase_bins)
    start = index
    if reference < threshold:
        while start and np.isfinite(values[start-1]) and values[start-1] < threshold:
            start -= 1
    if reference >= threshold:
        category = 'reference_normal'
    elif not np.isfinite(display):
        category = 'unavailable'
    elif display >= threshold:
        category = 'normal_display'
    elif held >= start:
        category = 'fresh_low'
    else:
        category = 'inherited_low'
    return {'reference': reference, 'display': display, 'held_index': held, 'index': index,
            'low_run_start': start, 'pre_low_sec': (index-start+1)*10 if reference < threshold else 0,
            'category': category}


def recovery_summary(values, time_sec, step_bins, phase_bins, threshold=65, window_sec=300):
    values = np.asarray(values, dtype=float)
    available = (np.arange(len(values)) + 1) * 10.
    horizon = time_sec + window_sec
    after = np.flatnonzero((available > time_sec + 1e-9) & (available <= horizon + 1e-9))
    if (not len(after) or available[-1] < horizon or len(after) != window_sec // 10
            or np.isfinite(values[after]).mean() < .8):
        return None
    recovered = [int(j) for j in after[1:]
                 if np.isfinite(values[j-1:j+1]).all() and np.all(values[j-1:j+1] >= threshold)]
    if not recovered:
        return {'recovery_index': -1, 'persistence_min': np.nan, 'stop_reason': 'no_confirmed_recovery'}
    first = recovered[0]
    seconds, reason = 0., 'window_end'
    for j in range(first, len(values)):
        if available[j] >= horizon - 1e-9:
            break
        if not np.isfinite(values[j]):
            reason = 'missing_reference'; break
        if values[j] < threshold:
            reason = 'recurrent_low_reference'; break
        display, _ = display_at(values, j, step_bins, phase_bins)
        if not np.isfinite(display):
            reason = 'unavailable_display'; break
        if display >= threshold:
            reason = 'normal_display'; break
        seconds += min(10., horizon - available[j])
    return {'recovery_index': first, 'persistence_min': seconds/60, 'stop_reason': reason}
