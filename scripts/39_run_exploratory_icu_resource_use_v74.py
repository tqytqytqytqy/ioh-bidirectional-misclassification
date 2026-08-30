from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ioh.icu_analysis import fit_sequential_icu_models
from ioh.icu_outcomes import build_prolonged_icu_outcome, summarize_icu_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.input)
    outcome = build_prolonged_icu_outcome(source, threshold_days=2)
    gate = summarize_icu_gate(
        source,
        threshold_days=2,
        min_completeness=0.95,
        min_events=100,
    )
    model_frame = source.copy()
    model_frame["icu_ge2d"] = pd.to_numeric(
        outcome["prolonged_icu_ge2d"], errors="coerce"
    )
    models = fit_sequential_icu_models(model_frame)

    gate_pass = (
        gate.loc[gate["criterion"].eq("overall_gate"), "status"].iloc[0]
        == "PASS_WITH_ENDPOINT_LIMITATION"
    )
    models_pass = bool(models["model_stability_status"].eq("PASS").all())
    reporting_decision = (
        "ELIGIBLE_FOR_SUPPLEMENTARY_EXPLORATORY_REPORTING"
        if gate_pass and models_pass
        else "NOT_ELIGIBLE_FOR_MANUSCRIPT_REPORTING"
    )
    models["gate0_status"] = gate.loc[
        gate["criterion"].eq("overall_gate"), "status"
    ].iloc[0]
    models["reporting_decision"] = reporting_decision

    gate.to_csv(args.output_dir / "icu_outcome_gate0.csv", index=False)
    models.to_csv(
        args.output_dir / "exploratory_icu_sequential_models.csv", index=False
    )
    model_columns = [
        "case_id",
        "subjectid",
        "icu_ge2d",
        "hidden_twa10",
        "log_true_twa",
        "age10",
        "male",
        "bmi5",
        "asa_high",
        "emop",
        "preop_htn",
        "preop_dm",
        "baseline_cr05",
        "duration_hr",
        "department",
    ]
    model_frame[model_columns].to_csv(
        args.output_dir / "icu_model_frame_v94_internal.csv", index=False
    )

    event_row = gate.loc[gate["criterion"].eq("event_count")].iloc[0]
    lines = [
        "# Postoperative ICU Outcome Gate 0 and Sequential Models",
        "",
        "## Frozen endpoint",
        "",
        "- Definition: postoperative ICU length of stay >=2 days (`icu_days >=2`).",
        "- Role: exploratory postoperative resource-use outcome.",
        "- Limitation: VitalDB does not distinguish planned from unplanned ICU admission.",
        "- The endpoint was selected before model fitting; reporting eligibility is not based on P values.",
        "",
        "## Gate 0",
        "",
        f"- Source cases: {len(source):,}.",
        f"- Evaluable cases: {int(event_row['denominator']):,}.",
        f"- Events: {int(event_row['value']):,} ({float(event_row['proportion']):.1%}).",
        f"- Gate status: {models['gate0_status'].iloc[0]}.",
        "",
        "## Sequential models",
        "",
    ]
    for row in models.itertuples(index=False):
        lines.append(
            f"- {row.analysis}: N={row.n_cases:,}; events={row.events:,}; "
            f"RR {row.relative_risk:.3f} (95% CI {row.ci_low:.3f} to "
            f"{row.ci_high:.3f}); P={row.p_value:.4f}; stability={row.model_stability_status}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- {reporting_decision}",
            "- AKI remains the sole emphasized clinical outcome; this ICU endpoint must remain supplementary and exploratory.",
            "- No causal or unplanned-admission interpretation is permitted.",
            "",
        ]
    )
    (args.output_dir / "icu_outcome_qc.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
