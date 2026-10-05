# Analysis conventions

The reference is the median arterial MAP on a 10 s grid, with physiological
range 20 to 180 mmHg and at least 80% valid coverage. The analytic window uses
anaesthesia start/end, intersected with recording availability, rather than
surgical start/end. The cohort has 2435 operations from 2380 subjects.

## Deficit and display

At each valid point, hidden deficit is the positive reference-minus-display
deficit and overdisplay is the reverse; ratios use summed reference deficit
area. Missing reference points enter neither numerator nor denominator.
Before the first scheduled sample at nonzero offsets, the primary convention
has no displayed deficit and counts concurrent reference deficit as hidden.
Baseline initialisation and pre-display exclusion are explicit sensitivities.
Their materially different AUC partitioning is reported rather than treating
near-zero net area as a convention-independent finding.

## Episodes and measurement source

Qualifying episodes have at least 60 s of contiguous valid MAP below 65 mmHg.
Detection means any low display during the reference episode. Exclusive
phase-episode categories are new low sample within the current episode,
inherited low only, and no low display. New sampling takes priority. An
inherited value present at onset is separately reported and can overlap with
new sampling. Case quantiles give each case equal weight. Phases are averaged,
not treated as independent observations. Subject resampling retains repeated
operations. Gap sensitivities bridge observed normal values, never missing MAP.

Interval tables use a common reference-time denominator. Displayed duration
and AUC include low runs shorter than the episode minimum. Expected detected
event counts can be non-integers because they average deterministic phases.

## Recorded infusion adjustments

Norepinephrine or phenylephrine starts/increases require adjacent records
within 5 s and a sustained increase for 5 s. Changes are merged in anchored
60 s windows. The first 5 min is excluded and preceding reference support is
required. Completed 10 s bins prevent post-adjustment samples entering
pre-adjustment classification. There are 236 low-reference blocks in 85 cases
from 83 subjects. Bootstrap uncertainty uses 2000 subject replicates, seed
20261004. Restricted persistence uses 184 confirmed recoveries and ends at
normal display, missing reference, recurrent low pressure or 5 min after
adjustment. It is not a drug-effect estimate or authenticated rescue time.

## NIBP records

Independent cuff-cycle markers were not verified. Retained numerical records
cannot establish independent cuff measurements. The audit pairs records with
the arterial MAP median from -30 to +30 s, requiring at least 80% support.
Legacy cuff-agreement and diagnostic classification outputs are not retained
as current clinical results. Baseline reconstruction code is preserved to
reproduce the legacy record audit, not to authenticate cuff cycles.

## Postoperative outcomes

Creatinine-only AKI uses values after anaesthesia end until discharge or 7 days.
Sequential modified Poisson models use subject-cluster variance and nonlinear
total reference burden adjustment. There are 2246 evaluable cases/106 AKI
events; the complete-case models use 2242 cases/104 events. The binary contrast
is not estimated because the no-hidden group has zero AKI events. ICU stay of
at least 2 days is resource use; its models use 2248 cases/153 events and lack
admission intent. Associations do not estimate monitoring or treatment effects.
