# Analysis conventions

## Case duration and window

Eligibility uses anaesthesia duration, calculated as `(aneend-anestart)/60`. The primary trajectory is the recorded portion of the anaesthetic interval from `max(0, anestart)` to `aneend`. Surgery time is a sensitivity window only.

## Missing reference and unavailable display

Reference-missing grid points are excluded from the reference denominator and both directional components. If the reference is valid but the display is unavailable under the applicable horizon, displayed deficit is zero and concurrent reference deficit is assigned to the unavailable-display hidden component. Display-valid conditional estimates are reported separately.

## NIBP event time and pairing

The database timestamp is described as recorded event time because cuff-cycle completion semantics could not be verified. Pairing uses the median valid arterial MAP in the -30 to +30 s window, requiring at least 80% valid coverage.

## Strategy phase

Strategy-frontier estimates use a start-anchored offset-0 comparator in the same 2435-case cohort. Phase-averaged estimates remain the primary fixed-interval analysis.
