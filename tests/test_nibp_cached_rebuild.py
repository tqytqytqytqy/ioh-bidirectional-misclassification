from __future__ import annotations

import hashlib

import pandas as pd

import ioh.pipeline as pipeline


def test_nibp_events_can_be_rebuilt_from_an_explicit_frozen_raw_sample(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "outputs"
    (output_dir / "intermediate").mkdir(parents=True)
    manifest = pd.DataFrame(
        {
            "case_id": [1, 2],
            "pre_qc_primary_candidate": [True, True],
            "has_nibp_map": [True, True],
            "nibp_tid": ["unused-1", "unused-2"],
            "age": [50, 60],
            "sex": ["M", "F"],
            "asa": [2, 3],
            "emop": [0, 1],
            "duration_min": [120.0, 90.0],
        }
    )
    manifest.to_parquet(output_dir / "intermediate" / "vitaldb_manifest.parquet")
    pd.DataFrame(
        {"case_id": [1, 2], "time_sec": [0.0, 0.0], "art_map": [80.0, 75.0]}
    ).to_parquet(output_dir / "intermediate" / "artmap_10s.parquet")
    cached = tmp_path / "frozen_nibp_raw.parquet"
    pd.DataFrame(
        {
            "case_id": [1, 1, 1, 2],
            "time_sec": [0.0, 10.0, 130.0, 0.0],
            "map": [80.0, 80.0, 80.0, 70.0],
        }
    ).to_parquet(cached)
    expected_hash = hashlib.sha256(cached.read_bytes()).hexdigest()
    cfg = {
        "project": {"output_dir": str(output_dir), "seed": 7},
        "nibp": {
            "require_primary_art_coverage": True,
            "sample_cases": "all",
            "same_value_hold_sec": 90,
            "min_gap_new_event_sec": 120,
            "plausible_range": [20, 180],
        },
    }

    events = pipeline.build_nibp_display_events(cfg, raw_sample_path=cached)

    assert len(events) == 3
    audit = pd.read_csv(output_dir / "tables" / "nibp_display_retention_audit.csv")
    assert audit.loc[0, "input_mode"] == "frozen_derived_raw_sample"
    assert audit.loc[0, "input_sha256"] == expected_hash
    assert audit.loc[0, "raw_records"] == 4


def test_pair_events_returns_a_stable_schema_for_no_display_events():
    events = pd.DataFrame(columns=["case_id", "display_time_sec", "nibp_map"])
    panel = pd.DataFrame(columns=["case_id", "time_sec", "art_map"])

    pairs = pipeline._pair_events(events, panel, (-30, 30), 0.8, 10)

    assert pairs.columns.tolist() == [
        "case_id",
        "display_time_sec",
        "nibp_map",
        "window_before_sec",
        "window_after_sec",
        "art_points",
        "expected_art_points",
        "valid_fraction",
        "art_map_ref",
        "nibp_art_bias",
        "paired",
    ]
