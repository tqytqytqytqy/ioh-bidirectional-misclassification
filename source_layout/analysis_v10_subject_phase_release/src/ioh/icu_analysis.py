from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


BASE_COVARIATES = (
    "age10 + male + bmi5 + asa_high + emop + preop_htn + preop_dm + "
    "baseline_cr05 + duration_hr + C(department)"
)
REFERENCE_BURDEN = "bs(log_true_twa, df=3, degree=2, include_intercept=False)"
COVARIATE_DESCRIPTION = (
    "age, sex, body mass index, ASA Physical Status >=3, emergency surgery, "
    "hypertension, diabetes, baseline creatinine, anaesthesia duration, and "
    "surgical department"
)


def assess_model_stability(
    estimate: dict,
    *,
    min_events_per_parameter: float = 5.0,
    max_ci_width_ratio: float = 10.0,
) -> dict:
    """Apply result-independent numerical stability criteria to one model."""

    events = float(estimate["events"])
    parameters = float(estimate["n_parameters"])
    events_per_parameter = events / parameters if parameters > 0 else np.nan
    numeric_fields = (
        "relative_risk",
        "ci_low",
        "ci_high",
        "standard_error",
        "p_value",
    )
    finite = all(np.isfinite(float(estimate[field])) for field in numeric_fields)
    positive_interval = bool(
        finite
        and float(estimate["relative_risk"]) > 0
        and float(estimate["ci_low"]) > 0
        and float(estimate["ci_high"]) > 0
    )
    ci_width_ratio = (
        float(estimate["ci_high"]) / float(estimate["ci_low"])
        if positive_interval
        else np.nan
    )

    checks = {
        "converged": bool(estimate.get("converged", False)),
        "finite_estimate": finite,
        "positive_confidence_interval": positive_interval,
        "events_per_parameter": bool(
            np.isfinite(events_per_parameter)
            and events_per_parameter >= min_events_per_parameter
        ),
        "ci_width_ratio": bool(
            np.isfinite(ci_width_ratio) and ci_width_ratio <= max_ci_width_ratio
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "events_per_parameter": events_per_parameter,
        "ci_width_ratio": ci_width_ratio,
        "model_stability_status": "PASS" if not failed else "NOT_PASS",
        "model_stability_reason": (
            "all analysis-plan-defined numerical stability criteria passed"
            if not failed
            else "failed: " + ", ".join(failed)
        ),
        "stability_min_events_per_parameter": min_events_per_parameter,
        "stability_max_ci_width_ratio": max_ci_width_ratio,
    }


def fit_modified_poisson(
    frame: pd.DataFrame,
    *,
    formula: str,
    term: str,
    cluster_column: str = "subjectid",
) -> dict:
    """Fit a Poisson log-link model with subject-cluster robust variance."""

    if cluster_column not in frame.columns:
        raise ValueError(f"cluster column is missing: {cluster_column}")
    working = frame.reset_index(drop=True).copy()
    for column in working.columns:
        if isinstance(working[column].dtype, pd.BooleanDtype):
            working[column] = working[column].astype("Float64").astype(float)
    model = smf.glm(formula=formula, data=working, family=sm.families.Poisson())
    row_labels = pd.Index(model.data.row_labels)
    groups = working.loc[row_labels, cluster_column]
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    if term not in fitted.params:
        raise ValueError(f"model term is missing: {term}")
    ci = fitted.conf_int().loc[term]
    outcome = pd.Series(fitted.model.endog)
    estimate = {
        "formula": formula,
        "term": term,
        "n_cases": int(fitted.nobs),
        "n_subjects": int(pd.Series(groups).nunique()),
        "events": int(outcome.sum()),
        "n_parameters": int(len(fitted.params)),
        "converged": bool(fitted.converged),
        "coefficient": float(fitted.params[term]),
        "standard_error": float(fitted.bse[term]),
        "relative_risk": float(np.exp(fitted.params[term])),
        "ci_low": float(np.exp(ci.iloc[0])),
        "ci_high": float(np.exp(ci.iloc[1])),
        "p_value": float(fitted.pvalues[term]),
        "aic": float(fitted.aic),
        "log_likelihood": float(fitted.llf),
    }
    estimate.update(assess_model_stability(estimate))
    return estimate


def fit_sequential_icu_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the frozen three-level hidden-burden association sequence."""

    required = {
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
        "subjectid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ICU model frame is missing columns: {missing}")

    specifications = [
        (
            "Absolute hidden burden, unadjusted",
            "total association, unadjusted",
            "none",
            "icu_ge2d ~ hidden_twa10",
            "none",
        ),
        (
            "Absolute hidden burden, clinical adjusted",
            "exploratory resource-use association",
            "clinical",
            f"icu_ge2d ~ {BASE_COVARIATES} + hidden_twa10",
            "none",
        ),
        (
            "Absolute hidden burden, clinical plus reference burden",
            "incremental association beyond total reference burden",
            "clinical_plus_reference",
            (
                f"icu_ge2d ~ {BASE_COVARIATES} + {REFERENCE_BURDEN} + "
                "hidden_twa10"
            ),
            (
                "quadratic B-spline (3 df) of log(1 + reference AUC per "
                "anaesthesia-hour)"
            ),
        ),
    ]
    rows: list[dict] = []
    for label, role, sequence, formula, reference_adjustment in specifications:
        estimate = fit_modified_poisson(
            frame,
            formula=formula,
            term="hidden_twa10",
            cluster_column="subjectid",
        )
        estimate.update(
            {
                "analysis": label,
                "analysis_role": role,
                "outcome_definition": "postoperative ICU length of stay >=2 days",
                "endpoint_role": "exploratory postoperative resource-use outcome",
                "endpoint_limitation": (
                    "VitalDB does not distinguish planned from unplanned ICU admission"
                ),
                "exposure": (
                    "hidden AUC per anaesthesia-hour at MAP <65 mm Hg"
                ),
                "exposure_scale": "per 10 mm Hg*min per anaesthesia-hour",
                "population": (
                    "all endpoint-evaluable complete cases, including zero "
                    "reference AUC"
                ),
                "adjustment_sequence": sequence,
                "reference_burden_adjustment": reference_adjustment,
                "covariate_adjustment": (
                    "none" if sequence == "none" else COVARIATE_DESCRIPTION
                ),
                "variance_estimator": "subject-cluster robust sandwich",
                "causal_boundary": (
                    "Exploratory association under counterfactual display emulation; "
                    "not a causal effect of monitoring frequency"
                ),
            }
        )
        rows.append(estimate)
    return pd.DataFrame(rows)
