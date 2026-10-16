"""Summarize within-record phase ranges from this run's private episode table."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_writebytecode = True
import numpy as np
import pandas as pd

from mimic_analysis import sha256, write_json


def summarize_phase_ranges(frame):
    selected = frame.loc[(frame.scenario == "primary") & (frame.threshold == 65)
                         & (frame.interval_min == 5) & (frame.gap_sec == 0)
                         & (frame.minimum_low_sec == 60)]
    if selected.empty:
        raise ValueError("primary_map65_5min_phase_table_missing")
    rows = []
    for record_id, group in selected.groupby("record_id", sort=True):
        if (len(group) != 30 or set(group.offset_sec) != set(range(0, 300, 10))
                or group.episodes.nunique() != 1 or group.subject_id.nunique() != 1):
            raise ValueError("within_record_phase_grid_or_denominator_inconsistent")
        denominator = float(group.episodes.iloc[0])
        row = {"record_id": record_id, "subject_id": group.subject_id.iloc[0],
               "reference_episodes": int(denominator), "number_of_phases": 30}
        for name in ("fresh", "never_low"):
            probability = group[name].to_numpy(float) / denominator if denominator > 0 else np.full(30, np.nan)
            finite = probability[np.isfinite(probability)]
            minimum, maximum = (float(finite.min()), float(finite.max())) if len(finite) else (np.nan, np.nan)
            row.update({name + "_probability_min": minimum, name + "_probability_max": maximum,
                        name + "_probability_range": maximum - minimum})
        rows.append(row)
    private = pd.DataFrame(rows)
    aggregate_rows = []
    for name in ("fresh", "never_low"):
        for statistic in ("min", "max", "range"):
            values = private[name + "_probability_" + statistic].dropna().to_numpy(float)
            quantiles = np.quantile(values, [.05, .25, .5, .75, .95]) if len(values) else [np.nan] * 5
            aggregate_rows.append({"scenario": "primary", "threshold": 65., "interval_min": 5.,
                                   "gap_sec": 0, "minimum_low_sec": 60,
                                   "metric": name + "_probability", "statistic": "within_record_" + statistic,
                                   "n_records": len(private), "n_defined": len(values),
                                   "n_records_without_reference_episodes": int((private.reference_episodes == 0).sum()),
                                   "mean": float(values.mean()) if len(values) else np.nan,
                                   "minimum": float(values.min()) if len(values) else np.nan,
                                   "maximum": float(values.max()) if len(values) else np.nan,
                                   **dict(zip(["p05", "p25", "p50", "p75", "p95"], quantiles))})
    return private, pd.DataFrame(aggregate_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    results = root / "analysis/aggregate/results.json"
    source = root / "private/analysis/episodes_by_record_phase.csv.gz"
    if json.loads(results.read_text()).get("status") != "complete":
        raise ValueError("core_analysis_not_complete")
    private, aggregate = summarize_phase_ranges(pd.read_csv(source))
    aggregate_path = root / "analysis/aggregate/within_record_phase_range_summary.csv"
    private_path = root / "private/analysis/within_record_phase_ranges.csv.gz"
    aggregate.to_csv(aggregate_path, index=False)
    private.to_csv(private_path, index=False, compression="gzip")
    private_path.chmod(0o600)
    write_json(root / "analysis/aggregate/phase_range_manifest.json",
               {"status": "complete", "source": "Current completed core analysis; no QC input changes or reanalysis",
                "records": len(private), "aggregate_rows": len(aggregate),
                "core_results_sha256": sha256(results), "source_phase_table_sha256": sha256(source),
                "code_sha256": sha256(Path(__file__)), "aggregate_sha256": sha256(aggregate_path),
                "undefined_rate_policy": "Records with no reference episode retained privately; rates and ranges undefined, not zero"})
    print(json.dumps({"status": "complete", "records": len(private), "aggregate_rows": len(aggregate)}))


if __name__ == "__main__":
    main()
