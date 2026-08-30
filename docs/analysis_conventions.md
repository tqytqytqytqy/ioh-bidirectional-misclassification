# Analysis conventions

## Time base and reference

- The arterial MAP reference is evaluated on a fixed 10-s grid.
- The primary threshold is MAP below 65 mm Hg.
- The main episode analysis requires a reference episode duration of at least 60 s.
- Fixed-interval displays use last observation carried forward according to the prespecified phase convention.

## Temporal observability

- Complete miss: no displayed value below the threshold during a reference hypotension episode.
- Detection delay: elapsed time from reference episode onset to the first displayed low value.
- Pre-display AUC: reference area below the threshold accrued before the first displayed low value; for a completely missed episode, this equals the full episode AUC.
- Decision-opportunity metrics characterize the measurement process. They do not establish that a clinical action would have occurred.

## NIBP pairing

Each reconstructed NIBP display event is paired to the median of valid arterial MAP values from -30 through +30 s around the recorded event time. Sensitivity analyses evaluate alternative pairing and de-duplication settings.

## Postoperative outcomes

AKI is creatinine-defined. Sequential modified Poisson models report risk ratios with 95% confidence intervals. The incremental model adds nonlinear adjustment for total reference hypotension burden. ICU length of stay of at least 2 days is a frozen exploratory resource-use endpoint; VitalDB cannot distinguish planned from unplanned ICU admission.

All outcome analyses are observational associations and do not estimate causal effects of monitoring frequency or treatment.
