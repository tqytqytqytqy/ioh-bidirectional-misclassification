# IOH2 Estimand Dictionary

Primary endpoint: `HDR65_fixed5`. Key secondary endpoint: `ODR65_fixed5`.

For each time point, `R=max(0, threshold-reference_MAP)` and `D=max(0, threshold-displayed_MAP)`. The pointwise components are `concordant=min(R,D)`, `hidden=max(0,R-D)`, and `overdisplay=max(0,D-R)`. Population ratios use aggregate `TrueAUC` denominators. Historical one-directional visibility metrics are not primary endpoints.
