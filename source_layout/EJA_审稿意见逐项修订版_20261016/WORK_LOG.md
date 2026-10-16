# EJA review-response revision

Scope: preserve the 2026-10-16 clinical-timing version; create a separate, local revision. No upload, public release, new DOI or submission.

## Locked analysis decisions

1. The infusion-adjustment analysis retains its original unit. The revised recovery-specific main estimate counts each (case, recovery-confirmation time) once, using the earliest eligible adjustment only to identify a duplicate. This unit is chosen before new estimates, not for statistical significance. All adjustment-weighted estimates remain available.
2. Retain 20 s of consecutive normal completed 10 s bins as the reference recovery definition; add 30 s and 60 s. Use the same candidate low-reference infusion adjustments, original 300 s search window and 80% coverage requirement. Start the 60/120 s follow-up at confirmation of the entire normal run. Do not retrospectively backdate the start. Deduplicate separately for each definition. Missing values break a normal run. No future 120 s completeness restriction.
3. Estimate all four first-transition states, unknown bounds, 1/3/5 min display intervals, both follow-up horizons, adjustment-equal, distinct-recovery-equal and patient-equal weights. Retain the objective-start matched analyses. Use 2000 patient-cluster bootstrap replicates, seed 20261016. Report all definitions, not only consistent estimates.
4. Common-support bridge: retain original paired whole-hour positions and age rules. On the entire original time grid, retain only minutes with six finite 10 s values. Set all six values to missing otherwise and derive the minute median on exactly those complete minutes. Apply the same input support to scheduled display updates and reference scoring without compressing time. Do not reset display at hour boundaries. Compare 10 s and 1 min with identical valid support, 180 s minimum episodes, all phases, thresholds 65 mmHg, intervals 1/3/5/10 min. Original paired-hour bridge remains the comparator. Use 1000 patient-cluster paired bootstrap replicates, seed 20261016; report valid-time loss and denominators. SICdb has no native 10 s data and receives no fabricated bridge.
5. Keep prior absolute exposure tables on their frozen original bin grid. State explicitly that completed-bin timing excludes incomplete terminal bins; provide a numerical boundary audit. Do not silently replace historical denominators.
6. Present temporal information as the clinical contribution, not improved treatment or outcomes. Qualify net-AUC similarity by the primary initialization convention. Retain all prior inconsistent results and source-role limits.
7. Ethics approval date and chair require evidence. Search existing local source evidence and official publications; never infer them from an approval number. If unavailable, provide an honest draft response and retain a clearly identified submission blocker outside the manuscript.

## Execution

- [ ] Freeze source hashes, add focused tests and new analysis code.
- [ ] Recovery deduplication and sustained-normal sensitivities.
- [ ] Common-valid-minute paired resolution bridge.
- [ ] Independently verify results and retained source hashes.
- [ ] Update manuscript, Table 3, figures, supplement, Cover Letter, checklist and workbook.
- [ ] Render and inspect updated documents/figures; freeze a local manifest and report remaining blockers.

Ruling: Recovery-specific results are now recovery-equal, while adjustment-weighted results remain explicit sensitivity outputs. This corrects the unit without concealing the previous estimand.
Ruling: The common-support bridge uses complete six-bin minutes, rather than accepting five-bin medians, so both resolutions have exactly the same finite temporal support. The cost is some discarded valid 10 s data, which is quantified.
