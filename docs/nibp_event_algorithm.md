# Actual NIBP Display-Event Algorithm

Raw NIBP monitor records are grouped by case, filtered to physiologic MAP values, sorted by monitor time, and collapsed into display-level events. The primary de-duplication rule keeps the first value, keeps value changes, collapses repeated identical values held for less than 90 seconds, and requires at least 120 seconds before a repeated same-value event is treated as a new display event.

The displayed trajectory is built by forward-filling the last visible cuff value from each display event onto the 10-second arterial reference grid. The configured primary display-validity horizon is 10 minutes; after this horizon, the displayed value is set to missing until the next display event. No-visible intervals are reported separately and are handled as no displayed deficit in the decomposition. Sensitivity analyses compare no maximum horizon and a 5-minute maximum display-validity horizon.
