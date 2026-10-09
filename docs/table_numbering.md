# Current and Historical Table Numbering

The current presentation contains 4 main tables, 16 supplementary tables (26 panels) and 2 supplementary figures.
`outputs/tables/Supp_Table_S01` through `Supp_Table_S16` use current
supplementary numbering, with `_panel_N.csv` for multi-panel tables. There are
26 panel CSVs, not 26 publication tables. Their exact titles, notes and cells are in
`outputs/tables/condensed_tables.json`. Do not interpret a historical S number
in `outputs/aggregate/qc` or old statistical code as a current citation.

## Workbook Navigation

The workbook retains all 55 original worksheets. Its 35 historical
supplementary worksheets are renamed `Legacy_S01` through `Legacy_S35`.
Sixteen new current supplementary worksheets and one mapping worksheet are
placed before the historical supplementary block, for 72 worksheets in total.
Main Tables 1-4 may precede the current supplementary block and retain all numerical
data; the Main Table 2 note points to `Legacy_S26` in the workbook.

`Legacy_S05`, `Legacy_S26` and `Legacy_S34` are WORKBOOK-ONLY detailed tables in
the current presentation, not separate current SDC publication tables. Open
`outputs/final_workbook.xlsx` and select those exact worksheet names.
Their historical CSVs and full original cells are also retained for provenance.
No historical data are deleted to achieve the 16-table presentation.

The exact correspondence and any component-level decisions are recorded in
`presentation_inputs/table_map.json`; this document does not replace that map.

## Numbering Map

| Current table | Historical workbook sources | Panels |
| --- | --- | --- |
| S1 | Legacy_S01, Legacy_S27 | 1 |
| S2 | Legacy_S07, Legacy_S08, Legacy_S09, Legacy_S10, Legacy_S11, Legacy_S12 | 2 |
| S3 | Legacy_S02, Legacy_S24 | 2 |
| S4 | Legacy_S03, Legacy_S04 | 2 |
| S5 | Legacy_S06 | 1 |
| S6 | Legacy_S23 | 1 |
| S7 | Legacy_S21, Legacy_S22 | 3 |
| S8 | Legacy_S25, Legacy_S35 | 1 |
| S9 | Legacy_S28, Legacy_S29 | 2 |
| S10 | Legacy_S13, Legacy_S16 | 2 |
| S11 | Legacy_S14, Legacy_S15 | 2 |
| S12 | Legacy_S17 | 1 |
| S13 | Legacy_S18, Legacy_S19, Legacy_S20 | 3 |
| S14 | Legacy_S30, Legacy_S31 | 1 |
| S15 | Legacy_S32 | 1 |
| S16 | Legacy_S33 | 1 |

## Current Titles

- S1: Supplementary Table S1. Cohort construction and recorded infusion-adjustment flow
- S2: Supplementary Table S2. NIBP record classification and measurement-cycle audit
- S3: Supplementary Table S3. Episode visibility across intervals and source of low display
- S4: Supplementary Table S4. Episode visibility by nadir and duration
- S5: Supplementary Table S5. Bidirectional deficit across thresholds and sampling intervals
- S6: Supplementary Table S6. Case-level variation under a 5 min display
- S7: Supplementary Table S7. Sensitivity to clustering and initial-display convention
- S8: Supplementary Table S8. Event-definition robustness before and after startup quality checks
- S9: Supplementary Table S9. Display information at infusion adjustments and event-definition sensitivity
- S10: Supplementary Table S10. AKI ascertainment and characteristics by evaluability
- S11: Supplementary Table S11. Sequential and sensitivity associations with postoperative AKI
- S12: Supplementary Table S12. Sequential associations with postoperative ICU stay of at least 2 days
- S13: Supplementary Table S13. ASA coding audit and outcome-model sensitivity
- S14: Supplementary Table S14. Cohort retention and episode visibility under startup quality rules
- S15: Supplementary Table S15. Infusion-adjustment results under startup quality rules
- S16: Supplementary Table S16. Postoperative associations under startup quality rules

## Input Readiness

- No presentation inputs missing.

Frozen release snapshot. Independent physician clinical adjudication was not performed. This editorial/presentation revision retained the statistical results without model refitting; publication status is verified in the external version record.
