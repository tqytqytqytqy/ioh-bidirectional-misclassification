import pandas as pd

from ioh.stats.bootstrap import cluster_bootstrap_mean


def test_cluster_bootstrap_samples_cases_not_events():
    df = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 2],
            "bias": [10.0, 12.0, 14.0, -2.0],
        }
    )

    out = cluster_bootstrap_mean(df, cluster_col="case_id", value_col="bias", iterations=50, seed=7)

    assert out["cluster_count"] == 2
    assert out["row_count"] == 4
    assert out["point_estimate"] == 8.5
    assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]
