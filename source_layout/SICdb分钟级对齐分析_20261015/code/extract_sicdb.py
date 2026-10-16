"""Read-only SICdb arterial minute extraction; no effect estimation.

All identifiers and trajectories stay under private/sicdb (0700/0600).
The compressed *whole* hourly member, not a selected subset, is hashed.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(os.environ.get('SICDB_ZIP', 'data/sicdb-1.0.8.zip'))
DATA_IDS = {701: "sbp", 702: "dbp", 703: "map"}
DT = np.dtype([("case", "<i8"), ("offset", "<f8"), ("raw", "<f4", (60,))])
EXPECTED_KEYS = {
    "map", "sbp", "dbp", "index", "case_id", "subject_id", "age_bin",
    "icu_offset_sec", "duration_sec",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value, private=False):
    path = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600 if private else 0o644)
    if private:
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.bytes = 0

    def read(self, n=-1):
        block = self.stream.read(n)
        self.hash.update(block)
        self.bytes += len(block)
        return block


def decode_hour(payload, counts=None):
    """Decode 60 little-endian float32 positions; source zero means null."""
    if not isinstance(payload, str) or not payload:
        return None, "missing"
    if not payload.startswith("0x"):
        return None, "invalid_prefix"
    if len(payload) != 482:
        return None, "wrong_length"
    try:
        binary = bytes.fromhex(payload[2:])
    except ValueError:
        return None, "invalid_hex"
    if len(binary) != 240:
        return None, "wrong_length"
    values = np.frombuffer(binary, dtype="<f4").copy()
    zero = values == 0
    if counts is not None:
        counts["zero_as_null_minutes"] += int(zero.sum())
        counts["source_nan_minutes"] += int(np.isnan(values).sum())
        counts["source_infinite_minutes"] += int(np.isinf(values).sum())
        counts["finite_nonzero_minutes_before_timestamp_qc"] += int((np.isfinite(values) & ~zero).sum())
    values[zero] = np.nan
    return values, "ok"


def minute_grid(offsets, raw, duration_sec):
    """Keep complete origin-zero minutes; finite conflicts become missing.

    Missing duplicates never erase a finite observation. Equal finite duplicates
    collapse; any unequal finite values at a timestamp reject that timestamp.
    Range checks are intentionally not applied to this raw extraction.
    """
    if not np.isfinite(duration_sec) or duration_sec < 0:
        raise ValueError("Invalid source duration")
    n = int(np.floor(duration_sec / 60))
    if n > 5_000_000:
        raise ValueError("Unexpected duration; refusing silent large allocation")
    grid = np.full(n, np.nan, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (len(offsets), 60):
        raise ValueError("Hour rows must contain exactly 60 native minute positions")
    counts = Counter({
        key: 0 for key in (
            "hour_rows", "source_minute_positions", "source_nan_records",
            "source_infinite_records", "source_finite_records",
            "invalid_offset_hour_rows", "offgrid_minute_positions", "offgrid_finite_records",
            "outside_complete_grid_minute_positions", "outside_complete_grid_finite_records",
            "duplicate_timestamp_positions_including_missing",
            "duplicate_timestamp_excess_records_including_missing",
            "finite_duplicate_timestamp_positions", "exact_duplicate_finite_excess_records",
            "conflicting_finite_timestamps", "finite_records_at_conflicting_timestamps",
            "retained_finite_minutes",
        )
    })
    counts.update(hour_rows=len(offsets), source_minute_positions=raw.size,
                  source_nan_records=int(np.isnan(raw).sum()),
                  source_infinite_records=int(np.isinf(raw).sum()),
                  source_finite_records=int(np.isfinite(raw).sum()),
                  invalid_offset_hour_rows=int((~np.isfinite(offsets)).sum()))
    if not len(offsets):
        return grid, counts
    tt = (offsets[:, None] + np.arange(60, dtype=np.float64) * 60).ravel()
    vv = raw.ravel()
    finite_time = np.isfinite(tt)
    finite_value = np.isfinite(vv)
    aligned = np.zeros(len(tt), dtype=bool)
    aligned[finite_time] = np.remainder(tt[finite_time], 60) == 0
    outside = aligned & ((tt < 0) | (tt >= n * 60))
    offgrid = finite_time & ~aligned
    counts["offgrid_minute_positions"] = int(offgrid.sum())
    counts["offgrid_finite_records"] = int((offgrid & finite_value).sum())
    counts["outside_complete_grid_minute_positions"] = int(outside.sum())
    counts["outside_complete_grid_finite_records"] = int((outside & finite_value).sum())
    accepted_time = aligned & ~outside
    valid_t = tt[accepted_time]
    if len(valid_t):
        _, repetitions = np.unique(valid_t, return_counts=True)
        counts["duplicate_timestamp_positions_including_missing"] = int((repetitions > 1).sum())
        counts["duplicate_timestamp_excess_records_including_missing"] = int((repetitions - 1).sum())
    keep = accepted_time & finite_value
    tt, vv = tt[keep], vv[keep]
    if not len(tt):
        return grid, counts
    order = np.argsort(tt, kind="stable")
    tt, vv = tt[order], vv[order]
    unique, first, repetitions = np.unique(tt, return_index=True, return_counts=True)
    minimum = np.minimum.reduceat(vv, first)
    maximum = np.maximum.reduceat(vv, first)
    conflict = minimum != maximum
    counts["finite_duplicate_timestamp_positions"] = int((repetitions > 1).sum())
    counts["exact_duplicate_finite_excess_records"] = int((repetitions[~conflict] - 1).sum())
    counts["conflicting_finite_timestamps"] = int(conflict.sum())
    counts["finite_records_at_conflicting_timestamps"] = int(repetitions[conflict].sum())
    grid[(unique[~conflict] / 60).astype(np.int64)] = minimum[~conflict]
    counts["retained_finite_minutes"] = int(np.isfinite(grid).sum())
    return grid, counts


def case_quality(map_values, sbp, dbp, contract):
    """Availability and prespecified quality counts only; never alter arrays."""
    counts = Counter(total_complete_minutes=len(map_values))
    coherence = contract["coherence"]
    for name, values, limits in (
        ("map", map_values, contract["map_range_mmHg"]),
        ("sbp", sbp, coherence["sbp_range"]),
        ("dbp", dbp, coherence["dbp_range"]),
    ):
        finite = np.isfinite(values)
        inside = finite & (values >= limits[0]) & (values <= limits[1])
        counts[name + "_finite_minutes"] = int(finite.sum())
        counts[name + "_missing_minutes"] = int((~finite).sum())
        counts[name + "_range_valid_minutes"] = int(inside.sum())
        counts[name + "_out_of_range_finite_minutes"] = int((finite & ~inside).sum())
    mf = np.isfinite(map_values)
    both = np.isfinite(sbp) & np.isfinite(dbp)
    concurrent = mf & both
    pp = sbp - dbp
    sbp_ok = (sbp >= coherence["sbp_range"][0]) & (sbp <= coherence["sbp_range"][1])
    dbp_ok = (dbp >= coherence["dbp_range"][0]) & (dbp <= coherence["dbp_range"][1])
    pp_ok = (pp >= coherence["pulse_pressure_range"][0]) & (pp <= coherence["pulse_pressure_range"][1])
    ordered = (dbp <= map_values) & (map_values <= sbp)
    coherent = concurrent & sbp_ok & dbp_ok & pp_ok
    if coherence["require_dbp_le_map_le_sbp"]:
        coherent &= ordered
    counts["sbp_dbp_both_finite_minutes"] = int(both.sum())
    counts["all_three_finite_minutes"] = int(concurrent.sum())
    counts["coherent_all_three_minutes"] = int(coherent.sum())
    counts["coherence_flagged_map_minutes"] = int((concurrent & ~coherent).sum())
    counts["map_finite_without_both_sbp_dbp_minutes"] = int((mf & ~both).sum())
    counts["concurrent_sbp_range_failure_minutes"] = int((concurrent & ~sbp_ok).sum())
    counts["concurrent_dbp_range_failure_minutes"] = int((concurrent & ~dbp_ok).sum())
    counts["concurrent_pulse_pressure_failure_minutes"] = int((concurrent & ~pp_ok).sum())
    counts["concurrent_order_failure_minutes"] = int((concurrent & ~ordered).sum())
    valid_map = mf & (map_values >= contract["map_range_mmHg"][0]) & (map_values <= contract["map_range_mmHg"][1])
    counts["range_valid_map_coherence_flagged_minutes"] = int((valid_map & both & ~coherent).sum())
    window = int(contract["window_minutes"])
    nwin = len(map_values) // window
    valid_per_window = valid_map[:nwin * window].reshape(nwin, window).sum(axis=1)
    qualified = valid_per_window >= contract["minimum_valid_minutes_per_window"]
    counts["complete_origin_60min_windows"] = nwin
    counts["range_qualified_origin_60min_windows"] = int(qualified.sum())
    counts["range_valid_minutes_in_qualified_windows"] = int(valid_per_window[qualified].sum())
    return counts


def _member(z, basename):
    names = [name for name in z.namelist() if name == basename or name.endswith("/" + basename)]
    if len(names) != 1:
        raise ValueError("Expected exactly one source member: " + basename)
    return names[0]


def _source_state(source):
    st = source.stat()
    return {"sha256": sha256(source), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _load_cases(z):
    with z.open(_member(z, "cases.csv.gz")) as member:
        with gzip.GzipFile(fileobj=member) as gz:
            cases = pd.read_csv(gz, usecols=[
                "CaseID", "PatientID", "TimeOfStay", "ICUOffset", "AgeOnAdmission",
            ])
    for name in cases.columns:
        cases[name] = pd.to_numeric(cases[name], errors="raise")
    if not cases.CaseID.is_unique:
        raise ValueError("Nonunique source case metadata")
    for name in ("CaseID", "PatientID", "AgeOnAdmission"):
        v = cases[name].to_numpy(dtype=np.float64)
        if not (np.isfinite(v) & (v == np.floor(v))).all():
            raise ValueError("Missing or nonintegral identifier/age-bin metadata")
        cases[name] = cases[name].astype(np.int64)
    duration = cases.TimeOfStay.to_numpy(dtype=np.float64)
    if not (np.isfinite(duration) & (duration >= 0)).all():
        raise ValueError("Missing or negative source duration metadata")
    return cases.sort_values("CaseID", kind="stable").reset_index(drop=True)


def _scan(z, spool, known, retained, prior, chunk_rows, log):
    scan = Counter(all_hour_rows_scanned=0, unknown_case_selected_hour_rows=0,
                   under_60min_selected_hour_rows=0)
    decoding = {did: Counter() for did in DATA_IDS}
    member_name = _member(z, "data_float_h.csv.gz")
    last_log = time.monotonic()
    with ExitStack() as stack:
        streams = {}
        for did in DATA_IDS:
            fd = os.open(spool / (str(did) + ".bin"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            streams[did] = stack.enter_context(os.fdopen(fd, "wb"))
        hr = HashedReader(stack.enter_context(z.open(member_name)))
        gz = stack.enter_context(gzip.GzipFile(fileobj=hr))
        for chunk in pd.read_csv(
            gz, usecols=["CaseID", "DataID", "Offset", "rawdata"], chunksize=chunk_rows,
            dtype={"CaseID": "int64", "DataID": "int64", "Offset": "float64", "rawdata": "object"},
        ):
            scan["all_hour_rows_scanned"] += len(chunk)
            selected = chunk[chunk.DataID.isin(DATA_IDS)]
            scan["unknown_case_selected_hour_rows"] += int((~selected.CaseID.isin(known)).sum())
            scan["under_60min_selected_hour_rows"] += int((selected.CaseID.isin(known) & ~selected.CaseID.isin(retained)).sum())
            for did, rows in selected.groupby("DataID", sort=False):
                did = int(did)
                count = decoding[did]
                count["selected_hour_rows"] += len(rows)
                records = np.empty(len(rows), dtype=DT)
                records["case"] = rows.CaseID.to_numpy()
                records["offset"] = rows.Offset.to_numpy()
                for j, payload in enumerate(rows.rawdata):
                    values, status = decode_hour(payload, count)
                    count[status + "_hour_rows"] += 1
                    records["raw"][j] = np.nan if values is None else values
                keep = rows.CaseID.isin(retained).to_numpy()
                records[keep].tofile(streams[did])
                count["retained_case_hour_rows_spooled"] += int(keep.sum())
            if time.monotonic() - last_log >= 30:
                log("scan_progress", all_hour_rows_scanned=scan["all_hour_rows_scanned"])
                last_log = time.monotonic()
        gz.close()
        while hr.read(8 * 1024 * 1024):
            pass
        result = dict(scan)
        result.update({
            "whole_hourly_member": member_name,
            "compressed_member_bytes": hr.bytes,
            "compressed_member_sha256": hr.hash.hexdigest(),
            "zip_member_bytes_equal": hr.bytes == z.getinfo(member_name).file_size,
            "prior_member_sha256_equal": hr.hash.hexdigest() == prior["compressed_member_sha256"],
            "prior_member_bytes_equal": hr.bytes == prior["compressed_member_bytes"],
            "prior_all_hour_rows_equal": scan["all_hour_rows_scanned"] == prior["counts"]["all_hour_rows_scanned"],
        })
    return result, {str(did): dict(c) for did, c in decoding.items()}


def _open_spool(spool):
    data = {}
    for did in DATA_IDS:
        path = spool / (str(did) + ".bin")
        if path.stat().st_size % DT.itemsize:
            raise ValueError("Incomplete restricted spool record")
        a = np.memmap(path, dtype=DT, mode="r") if path.stat().st_size else np.empty(0, dtype=DT)
        order = np.argsort(a["case"], kind="stable")
        sorted_cases = a["case"][order]
        unique, first, count = np.unique(sorted_cases, return_index=True, return_counts=True)
        bounds = {int(c): (int(i), int(i + n)) for c, i, n in zip(unique, first, count)}
        data[did] = (a, order, bounds)
    return data


def _write_shard(private, number, rows, signals):
    lengths = [len(a) for a in signals["map"]]
    index = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    arrays = {name: np.concatenate(signals[name]).astype(np.float32, copy=False) for name in DATA_IDS.values()}
    arrays.update({
        "index": index.astype(np.int64, copy=False),
        "case_id": np.asarray([r.CaseID for r in rows], dtype=np.int64),
        "subject_id": np.asarray([r.PatientID for r in rows], dtype=np.int64),
        "age_bin": np.asarray([r.AgeOnAdmission for r in rows], dtype=np.int64),
        "icu_offset_sec": np.asarray([r.ICUOffset for r in rows], dtype=np.float64),
        "duration_sec": np.asarray([r.TimeOfStay for r in rows], dtype=np.float64),
    })
    path = private / ("shard_%05d.npz" % number)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    # Reopen the actual compressed artifact without pickle, checking all arrays.
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != EXPECTED_KEYS:
            raise ValueError("Unexpected shard schema")
        for name, expected in arrays.items():
            actual = saved[name]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected, equal_nan=True):
                raise ValueError("Shard roundtrip mismatch")
        if np.any(np.diff(saved["case_id"]) <= 0):
            raise ValueError("Shard cases are not strictly sorted")
        if not np.array_equal(np.diff(saved["index"]), np.floor(saved["duration_sec"] / 60).astype(np.int64)):
            raise ValueError("Shard complete-minute boundary mismatch")
    if path.stat().st_mode & 0o777 != 0o600:
        raise ValueError("Restricted shard permissions mismatch")
    return {"path": path.name, "sha256": sha256(path), "cases": len(rows),
            "minutes": int(index[-1]), "bytes": path.stat().st_size}


def _build_shards(cases, spool, private, contract, cases_per_shard, log):
    data = _open_spool(spool)
    timestamp_qc = {str(did): Counter() for did in DATA_IDS}
    total = Counter()
    flow = Counter(retained_cases=len(cases))
    strata = {"all_retained": Counter(), "primary_age25": Counter(), "sensitivity_age20": Counter()}
    subjects = {key: set() for key in strata}
    qualifying_subjects = {key: set() for key in strata}
    shards, shard_rows = [], []
    signals = {name: [] for name in DATA_IDS.values()}
    last_log = time.monotonic()
    for row in cases.itertuples(index=False):
        grids = {}
        for did, name in DATA_IDS.items():
            a, order, bounds = data[did]
            lo, hi = bounds.get(int(row.CaseID), (0, 0))
            records = a[order[lo:hi]]
            grid, qc = minute_grid(records["offset"], records["raw"], row.TimeOfStay)
            timestamp_qc[str(did)].update(qc)
            grids[name] = grid
            signals[name].append(grid)
        stats = case_quality(grids["map"], grids["sbp"], grids["dbp"], contract)
        total.update(stats)
        for name in DATA_IDS.values():
            flow["retained_cases_with_finite_" + name] += int(stats[name + "_finite_minutes"] > 0)
            flow["retained_cases_without_finite_" + name] += int(stats[name + "_finite_minutes"] == 0)
        flow["retained_cases_with_all_three_concurrent"] += int(stats["all_three_finite_minutes"] > 0)
        for label, include in (
            ("all_retained", True),
            ("primary_age25", row.AgeOnAdmission >= contract["primary_age_cutoff"]),
            ("sensitivity_age20", row.AgeOnAdmission >= contract["sensitivity_age_cutoff"]),
        ):
            if include:
                s = strata[label]
                s["cases"] += 1
                s["cases_with_finite_map"] += int(stats["map_finite_minutes"] > 0)
                s["cases_without_finite_map"] += int(stats["map_finite_minutes"] == 0)
                s["cases_with_range_valid_map"] += int(stats["map_range_valid_minutes"] > 0)
                s["cases_with_qualified_origin_60min_window"] += int(stats["range_qualified_origin_60min_windows"] > 0)
                s.update(stats)
                subjects[label].add(int(row.PatientID))
                if stats["range_qualified_origin_60min_windows"] > 0:
                    qualifying_subjects[label].add(int(row.PatientID))
        shard_rows.append(row)
        if len(shard_rows) == cases_per_shard:
            shards.append(_write_shard(private, len(shards), shard_rows, signals))
            shard_rows = []
            signals = {name: [] for name in DATA_IDS.values()}
        if time.monotonic() - last_log >= 30:
            log("shard_progress", cases_completed=strata["all_retained"]["cases"], shards_written=len(shards))
            last_log = time.monotonic()
    if shard_rows:
        shards.append(_write_shard(private, len(shards), shard_rows, signals))
    for label in strata:
        strata[label]["subjects"] = len(subjects[label])
        strata[label]["subjects_with_qualified_origin_60min_window"] = len(qualifying_subjects[label])
    flow["retained_subjects"] = len(subjects["all_retained"])
    for a, _, _ in data.values():
        if isinstance(a, np.memmap):
            a._mmap.close()
    return shards, dict(flow), dict(total), {k: dict(v) for k, v in timestamp_qc.items()}, {k: dict(v) for k, v in strata.items()}


def run_extraction(source, root, prior_scan, cases_per_shard=100, chunk_rows=75000):
    source, root, prior_scan = Path(source), Path(root), Path(prior_scan)
    if not 1 <= cases_per_shard <= 1000 or chunk_rows < 1:
        raise ValueError("Invalid extraction chunk settings")
    contract_path = root / "analysis_contract.json"
    contract = json.loads(contract_path.read_text())
    prior = json.loads(prior_scan.read_text())
    if not prior.get("complete"):
        raise ValueError("Prior full hourly scan is not complete")
    private = root / "private/sicdb"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    if (private / "ready.json").exists() or list(private.glob("shard_*.npz")):
        raise FileExistsError("Existing extraction artifacts preserved; use a clean authorized output")
    (root / "qa").mkdir(exist_ok=True)
    (root / "aggregate").mkdir(exist_ok=True)
    start = time.monotonic()
    log_path = root / "qa/extraction.log"

    def log(phase, **values):
        entry = {"phase": phase, "elapsed_seconds": round(time.monotonic() - start, 2), **values}
        line = json.dumps(entry, allow_nan=False)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    provenance = {
        "contract_sha256": sha256(contract_path), "plan_sha256": sha256(root / "PLAN.md"),
        "extractor_sha256": sha256(Path(__file__)),
        "tests_sha256": sha256(Path(__file__).with_name("test_extract_sicdb.py")),
        "prior_scan_sha256": sha256(prior_scan),
    }
    before = _source_state(source)
    save_json(root / "qa/extraction_source_before.json", before)
    log("source_before_full_hash_complete", source_sha256=before["sha256"])
    after = None
    try:
        prior_source_path = prior_scan.with_name("source_before.json")
        if prior_source_path.exists():
            prior_source = json.loads(prior_source_path.read_text())
            if before["sha256"] != prior_source["sha256"]:
                raise ValueError("Full source checksum does not match prior source")
            provenance["prior_full_source_sha256_equal"] = True
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=private) as temporary:
            spool = Path(temporary)
            spool.chmod(0o700)
            with zipfile.ZipFile(source, "r") as z:
                source_cases = _load_cases(z)
                cases = source_cases[source_cases.TimeOfStay >= 3600].copy()
                metadata_flow = {
                    "source_cases": len(source_cases), "source_subjects": int(source_cases.PatientID.nunique()),
                    "source_cases_under_60_minutes": int((source_cases.TimeOfStay < 3600).sum()),
                }
                scan, decoding = _scan(z, spool, source_cases.CaseID, cases.CaseID, prior, chunk_rows, log)
            save_json(root / "qa/extraction_scan.json", {"scan": scan, "decoding": decoding})
            if not all(scan[key] for key in (
                "zip_member_bytes_equal", "prior_member_sha256_equal", "prior_member_bytes_equal", "prior_all_hour_rows_equal",
            )):
                raise ValueError("Whole hourly member prior scan checksum/row-count mismatch")
            log("whole_hourly_member_verified", all_hour_rows_scanned=scan["all_hour_rows_scanned"],
                compressed_member_sha256=scan["compressed_member_sha256"])
            shards, flow, quality, timestamp_qc, strata = _build_shards(
                cases, spool, private, contract, cases_per_shard, log)
        after = _source_state(source)
        preservation = {
            "before": before, "after": after,
            "full_hash_equal": before["sha256"] == after["sha256"],
            "bytes_mtime_equal": (before["bytes"], before["mtime_ns"]) == (after["bytes"], after["mtime_ns"]),
            "raw_source_opened_read_only": True,
        }
        save_json(root / "qa/extraction_source_preservation.json", preservation)
        if not preservation["full_hash_equal"] or not preservation["bytes_mtime_equal"]:
            raise ValueError("Source preservation check failed")
        if sha256(contract_path) != provenance["contract_sha256"]:
            raise ValueError("Frozen extraction contract changed during extraction")
        provenance["plan_sha256_after"] = sha256(root / "PLAN.md")
        provenance["plan_progress_changed_during_extraction"] = provenance["plan_sha256_after"] != provenance["plan_sha256"]
        if sum(s["cases"] for s in shards) != len(cases) or sum(s["minutes"] for s in shards) != quality["total_complete_minutes"]:
            raise ValueError("Shards do not cover all retained source cases/minutes")
        qc = {
            "status": "ready", "stage": "extraction and quality counts only; no effect estimates",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source), "provenance": provenance,
            "source_preservation": preservation, "scan": scan, "decoding": decoding,
            "flow": {**metadata_flow, **flow}, "timestamp_qc_retained_cases": timestamp_qc,
            "availability_range_and_concurrent_coherence": quality,
            "primary_age25_rough_counts": strata["primary_age25"],
            "sensitivity_age20_rough_counts": strata["sensitivity_age20"],
            "all_retained_rough_counts": strata["all_retained"],
            "rules": {
                "signals": {str(did): name for did, name in DATA_IDS.items()},
                "minute_origin_sec": 0, "minute_step_sec": 60,
                "complete_minutes": "floor(TimeOfStay/60); preserve positions, no interpolation",
                "zero_and_nonfinite": "source zero and nonfinite values stored as NaN; finite implausible values retained",
                "timestamp": "outside complete grid/offgrid rejected; unequal finite duplicates reject timestamp; missing plus finite retains finite",
                "duplicate_counts_include": "all decoded native positions, including missing; finite conflicts separately counted",
                "coherence": contract["coherence"], "map_range_mmHg": contract["map_range_mmHg"],
                "range_and_coherence_counts_do_not_modify_raw_arrays": True,
                "coherence_failure_components_overlap": True,
                "rough_windows": "nonoverlapping origin-zero complete 60-minute windows; range-valid MAP >=48 minutes",
            },
            "shard_count": len(shards), "shard_total_bytes": sum(s["bytes"] for s in shards),
            "all_shards_roundtrip_verified_allow_pickle_false": True,
            "restricted_directory_mode": "0700", "restricted_file_mode": "0600",
            "aggregate_contains_identifiers": False, "effect_estimation_run": False,
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        aggregate_path = root / "aggregate/sicdb_extraction_qc.json"
        save_json(aggregate_path, qc)
        ready = {
            "status": "ready", "schema_version": 1, "relative_to": "private/sicdb",
            "shards": shards, "cases": len(cases), "subjects": flow["retained_subjects"],
            "minutes": quality["total_complete_minutes"], "source_sha256": before["sha256"],
            "whole_hourly_member_sha256": scan["compressed_member_sha256"],
            **provenance,
            "aggregate_qc_path": "aggregate/sicdb_extraction_qc.json",
            "aggregate_qc_sha256": sha256(aggregate_path),
            "array_schema": {"map": "concatenated float32", "sbp": "concatenated float32",
                             "dbp": "concatenated float32", "index": "int64, ncases+1",
                             "case_id": "int64", "subject_id": "int64", "age_bin": "int64",
                             "icu_offset_sec": "float64", "duration_sec": "float64"},
            "case_sort": "strictly ascending across all shards", "minute_origin_sec": 0,
            "primary_age_cutoff": contract["primary_age_cutoff"],
            "sensitivity_age_cutoff": contract["sensitivity_age_cutoff"],
            "effect_estimation_run": False,
        }
        temporary_ready = private / ".ready.tmp.json"
        save_json(temporary_ready, ready, private=True)
        temporary_ready.replace(private / "ready.json")
        log("ready", cases=ready["cases"], subjects=ready["subjects"], minutes=ready["minutes"],
            shards=len(shards), source_preserved=True, whole_hour_member_prior_checksum_verified=True)
        return ready
    except Exception as exc:
        if after is None:
            after = _source_state(source)
        save_json(root / "qa/extraction_failure.json", {
            "status": "failed", "exception_type": type(exc).__name__,
            "source_before": before, "source_after": after,
            "full_hash_equal": before["sha256"] == after["sha256"],
            "ready_written": False,
        })
        log("failed", exception_type=type(exc).__name__, ready_written=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prior-scan", type=Path)
    parser.add_argument("--cases-per-shard", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=75000)
    args = parser.parse_args()
    prior = args.prior_scan or args.root.parent / "外部验证可行性审计_20261015/sicdb/scan_complete.json"
    run_extraction(args.source, args.root, prior, args.cases_per_shard, args.chunk_rows)


if __name__ == "__main__":
    main()
