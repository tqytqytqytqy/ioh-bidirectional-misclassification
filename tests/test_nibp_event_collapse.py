import pandas as pd

from ioh.qc.nibp_qc import collapse_nibp_display_events


def test_collapse_nibp_display_events_keeps_first_value_change_and_long_gap_same_value():
    raw = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 1, 1, 1],
            "time_sec": [0, 10, 20, 110, 130, 150],
            "map": [80, 80, 80, 80, 80, 79],
        }
    )

    events = collapse_nibp_display_events(raw, same_value_hold_sec=90, min_gap_new_event_sec=120)

    assert events[["display_time_sec", "nibp_map"]].to_dict("records") == [
        {"display_time_sec": 0, "nibp_map": 80},
        {"display_time_sec": 130, "nibp_map": 80},
        {"display_time_sec": 150, "nibp_map": 79},
    ]


def test_collapse_nibp_display_events_filters_implausible_values():
    raw = pd.DataFrame({"case_id": [1, 1, 1], "time_sec": [0, 60, 120], "map": [10, 80, 190]})

    events = collapse_nibp_display_events(raw, plausible_range=(20, 180))

    assert len(events) == 1
    assert events.iloc[0]["nibp_map"] == 80
