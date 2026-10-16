from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    cluster_col: str,
    value_col: str,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    cluster_means = df.groupby(cluster_col, dropna=True)[value_col].mean().dropna()
    values = cluster_means.to_numpy(dtype=float)
    point = float(df[value_col].mean())
    if len(values) == 0:
        return {"point_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "cluster_count": 0, "row_count": len(df)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(int(iterations), len(values)))
    reps = values[idx].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    return {
        "point_estimate": point,
        "ci_low": float(np.quantile(reps, alpha / 2.0)),
        "ci_high": float(np.quantile(reps, 1.0 - alpha / 2.0)),
        "cluster_count": int(len(values)),
        "row_count": int(len(df)),
    }


def case_bootstrap_ratio(
    case_table: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    case_col: str,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    grouped = case_table.groupby(case_col, dropna=True)[[numerator_col, denominator_col]].sum().reset_index()
    num = grouped[numerator_col].to_numpy(dtype=float)
    den = grouped[denominator_col].to_numpy(dtype=float)
    point = float(num.sum() / den.sum()) if den.sum() > 0 else np.nan
    if len(grouped) == 0:
        return {"point_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "cluster_count": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(grouped), size=(int(iterations), len(grouped)))
    num_rep = num[idx].sum(axis=1)
    den_rep = den[idx].sum(axis=1)
    reps = np.divide(num_rep, den_rep, out=np.full_like(num_rep, np.nan, dtype=float), where=den_rep > 0)
    alpha = 1.0 - float(confidence_level)
    return {
        "point_estimate": point,
        "ci_low": float(np.nanquantile(reps, alpha / 2.0)),
        "ci_high": float(np.nanquantile(reps, 1.0 - alpha / 2.0)),
        "cluster_count": int(len(grouped)),
    }

