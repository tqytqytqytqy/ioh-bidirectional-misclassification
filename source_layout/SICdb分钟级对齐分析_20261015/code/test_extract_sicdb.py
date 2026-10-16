"""Synthetic-only tests for the SICdb source extraction boundary."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np


SCRIPT = Path(__file__).with_name("extract_sicdb.py")
CONTRACT = {
    "primary_age_cutoff": 25,
    "sensitivity_age_cutoff": 20,
    "map_range_mmHg": [20, 180],
    "window_minutes": 60,
    "minimum_valid_minutes_per_window": 48,
    "coherence": {
        "sbp_range": [30, 250], "dbp_range": [10, 150],
        "pulse_pressure_range": [10, 160],
        "require_dbp_le_map_le_sbp": True,
        "mask_only_when_sbp_and_dbp_both_observed": True,
    },
}


def encoded(values):
    return "0x" + struct.pack("<60f", *values).hex()


def load_module():
    spec = importlib.util.spec_from_file_location("extract_sicdb_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            return
        cls.m = load_module()

    def setUp(self):
        self.assertTrue(SCRIPT.exists(), "SICdb extractor implementation is not present yet")

    def test_little_endian_zero_missing_and_native_positions(self):
        values = [80.5, 0, -0.0, -1, 300] + [75] * 55
        a, status = self.m.decode_hour(encoded(values))
        self.assertEqual(status, "ok")
        self.assertEqual(a.dtype, np.dtype("float32"))
        self.assertEqual(a.shape, (60,))
        np.testing.assert_equal(a[:5], [80.5, np.nan, np.nan, -1, 300])
        grid, qc = self.m.minute_grid([120], a[None, :], 600)
        np.testing.assert_equal(grid[:7], [np.nan, np.nan, 80.5, np.nan, np.nan, -1, 300])
        self.assertEqual(qc["outside_complete_grid_finite_records"], 52)

    def test_decoder_rejects_missing_bad_prefix_length_and_hex(self):
        for raw, expected in [(None, "missing"), ("", "missing"),
                              ("00" + "0" * 480, "invalid_prefix"),
                              ("0x00", "wrong_length"),
                              ("0x" + "g" * 480, "invalid_hex")]:
            with self.subTest(expected=expected):
                a, status = self.m.decode_hour(raw)
                self.assertIsNone(a)
                self.assertEqual(status, expected)

    def test_complete_minute_boundaries_and_offgrid_offsets(self):
        raw = np.full((4, 60), 77, dtype=np.float32)
        grid, qc = self.m.minute_grid([-60, 3540, 30, np.nan], raw, 3659)
        self.assertEqual(len(grid), 60)
        self.assertEqual(int(np.isfinite(grid).sum()), 60)
        self.assertEqual(qc["invalid_offset_hour_rows"], 1)
        self.assertEqual(qc["offgrid_finite_records"], 60)
        self.assertEqual(qc["outside_complete_grid_finite_records"], 60)
        self.assertEqual(qc["exact_duplicate_finite_excess_records"], 0)
        np.testing.assert_equal(grid, np.full(60, 77, dtype=np.float32))

    def test_duplicate_conflict_is_missing_and_cannot_be_resurrected(self):
        raw = np.full((3, 60), np.nan, dtype=np.float32)
        raw[:, 0] = [80, 81, 80]
        raw[:, 1] = [70, 70, 70]
        raw[:, 2] = [np.nan, 60, np.nan]
        grid, qc = self.m.minute_grid([0, 0, 0], raw, 3600)
        np.testing.assert_equal(grid[:4], [np.nan, 70, 60, np.nan])
        self.assertEqual(qc["conflicting_finite_timestamps"], 1)
        self.assertEqual(qc["exact_duplicate_finite_excess_records"], 2)
        self.assertEqual(qc["duplicate_timestamp_positions_including_missing"], 60)
        self.assertEqual(qc["duplicate_timestamp_excess_records_including_missing"], 120)

    def test_nonfinite_normalized_but_finite_implausible_not_filtered(self):
        raw = np.full((1, 60), np.nan, dtype=np.float32)
        raw[0, :6] = [19, 20, 180, 181, np.inf, -np.inf]
        grid, qc = self.m.minute_grid([0], raw, 3600)
        np.testing.assert_equal(grid[:6], [19, 20, 180, 181, np.nan, np.nan])
        self.assertEqual(qc["source_infinite_records"], 2)
        stats = self.m.case_quality(grid, np.full(60, np.nan), np.full(60, np.nan), CONTRACT)
        self.assertEqual(stats["map_finite_minutes"], 4)
        self.assertEqual(stats["map_range_valid_minutes"], 2)
        self.assertEqual(stats["map_out_of_range_finite_minutes"], 2)

    def test_coherence_only_flags_observed_sbp_and_dbp(self):
        m = np.array([80, 80, 80, 80, 80, 80, 80], dtype=np.float32)
        s = np.array([120, 70, 120, np.nan, 260, 90, 250], dtype=np.float32)
        d = np.array([60, 60, 90, 60, 60, 85, 10], dtype=np.float32)
        stats = self.m.case_quality(m, s, d, CONTRACT)
        self.assertEqual(stats["all_three_finite_minutes"], 6)
        self.assertEqual(stats["coherent_all_three_minutes"], 1)
        self.assertEqual(stats["coherence_flagged_map_minutes"], 5)
        self.assertEqual(stats["map_finite_without_both_sbp_dbp_minutes"], 1)

    def test_rough_windows_are_origin_anchored_and_nonoverlapping(self):
        m = np.full(179, np.nan, dtype=np.float32)
        m[:48] = 80
        m[60:107] = 80
        m[120:] = 80
        stats = self.m.case_quality(m, m + 40, m - 20, CONTRACT)
        self.assertEqual(stats["complete_origin_60min_windows"], 2)
        self.assertEqual(stats["range_qualified_origin_60min_windows"], 1)
        self.assertEqual(stats["range_valid_minutes_in_qualified_windows"], 48)

    def test_end_to_end_scans_all_rows_retains_missing_map_and_secures_shards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "synthetic.zip"
            cases = (
                "CaseID,PatientID,TimeOfStay,ICUOffset,AgeOnAdmission\n"
                "3,30,3600,0,25\n1,10,3659,120,30\n"
                "2,10,7200,3600,20\n4,40,3599,0,45\n"
            )
            a = encoded([80] + [0] * 59)
            s = encoded([120] * 60)
            d = encoded([60] * 60)
            other = encoded([95] * 60)
            rows = (
                "CaseID,DataID,Offset,rawdata\n"
                f"1,703,0,{a}\n1,701,0,{s}\n1,702,0,{d}\n"
                f"2,703,3600,{a}\n4,703,0,{a}\n"
                f"3,706,0,{other}\n999,703,0,{a}\n"
                "2,703,0,0x00\n"
            )
            compressed = gzip.compress(rows.encode(), mtime=0)
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("fixture/cases.csv.gz", gzip.compress(cases.encode(), mtime=0))
                z.writestr("fixture/data_float_h.csv.gz", compressed)
            prior = root / "prior.json"
            prior.write_text(json.dumps({
                "complete": True, "compressed_member_sha256": hashlib.sha256(compressed).hexdigest(),
                "compressed_member_bytes": len(compressed),
                "counts": {"all_hour_rows_scanned": 8},
            }))
            out = root / "output"
            out.mkdir()
            (out / "PLAN.md").write_text("Synthetic extraction test only\n")
            (out / "analysis_contract.json").write_text(json.dumps(CONTRACT))
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            original_build = self.m._build_shards

            def build_with_parallel_progress(*args, **kwargs):
                result = original_build(*args, **kwargs)
                (out / "PLAN.md").write_text("Synthetic extraction test only\nProgress updated by main task\n")
                return result

            with mock.patch.object(self.m, "_build_shards", side_effect=build_with_parallel_progress):
                result = self.m.run_extraction(source, out, prior, cases_per_shard=2, chunk_rows=3)
            self.assertEqual(result["status"], "ready")
            qc = json.loads((out / "aggregate/sicdb_extraction_qc.json").read_text())
            self.assertEqual(qc["scan"]["all_hour_rows_scanned"], 8)
            self.assertTrue(qc["source_preservation"]["full_hash_equal"])
            self.assertTrue(qc["scan"]["prior_member_sha256_equal"])
            self.assertEqual(qc["flow"]["retained_cases"], 3)
            self.assertEqual(qc["flow"]["retained_cases_without_finite_map"], 1)
            self.assertEqual(qc["primary_age25_rough_counts"]["cases"], 2)
            self.assertEqual(qc["decoding"]["703"]["wrong_length_hour_rows"], 1)
            self.assertEqual(qc["scan"]["unknown_case_selected_hour_rows"], 1)
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            private = out / "private/sicdb"
            ready = json.loads((private / "ready.json").read_text())
            self.assertEqual(ready["status"], "ready")
            self.assertTrue(ready["plan_progress_changed_during_extraction"])
            ids = []
            for shard in ready["shards"]:
                path = private / shard["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), shard["sha256"])
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(set(data.files), {
                        "map", "sbp", "dbp", "index", "case_id", "subject_id",
                        "age_bin", "icu_offset_sec", "duration_sec",
                    })
                    ids.extend(data["case_id"].tolist())
                    self.assertEqual(data["index"].dtype, np.dtype("int64"))
                    self.assertEqual(len(data["index"]), len(data["case_id"]) + 1)
                    for key in ("map", "sbp", "dbp"):
                        self.assertEqual(data[key].dtype, np.dtype("float32"))
                        self.assertEqual(len(data[key]), data["index"][-1])
                    for key in ("case_id", "subject_id", "age_bin"):
                        self.assertEqual(data[key].dtype, np.dtype("int64"))
                    for key in ("icu_offset_sec", "duration_sec"):
                        self.assertEqual(data[key].dtype, np.dtype("float64"))
                    if 3 in data["case_id"]:
                        pos = data["case_id"].tolist().index(3)
                        self.assertTrue(np.isnan(data["map"][data["index"][pos]:data["index"][pos+1]]).all())
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(ids, [1, 2, 3])
            self.assertEqual(private.stat().st_mode & 0o777, 0o700)
            self.assertEqual((private / "ready.json").stat().st_mode & 0o777, 0o600)
            self.assertNotIn("case_id", json.dumps(qc))
            self.assertNotIn("subject_id", json.dumps(qc))

    def test_wrong_prior_checksum_fails_closed_without_ready(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "synthetic.zip"
            member = gzip.compress(b"CaseID,DataID,Offset,rawdata\n")
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("fixture/cases.csv.gz", gzip.compress(
                    b"CaseID,PatientID,TimeOfStay,ICUOffset,AgeOnAdmission\n1,10,3600,0,30\n"))
                z.writestr("fixture/data_float_h.csv.gz", member)
            prior = root / "prior.json"
            prior.write_text(json.dumps({"complete": True, "compressed_member_sha256": "bad",
                                         "compressed_member_bytes": len(member),
                                         "counts": {"all_hour_rows_scanned": 0}}))
            out = root / "output"
            out.mkdir()
            (out / "PLAN.md").write_text("Synthetic test\n")
            (out / "analysis_contract.json").write_text(json.dumps(CONTRACT))
            with self.assertRaisesRegex(ValueError, "prior.*checksum"):
                self.m.run_extraction(source, out, prior)
            self.assertFalse((out / "private/sicdb/ready.json").exists())

    def test_parallel_contract_change_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "synthetic.zip"
            member = gzip.compress(b"CaseID,DataID,Offset,rawdata\n")
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("fixture/cases.csv.gz", gzip.compress(
                    b"CaseID,PatientID,TimeOfStay,ICUOffset,AgeOnAdmission\n1,10,3600,0,30\n"))
                z.writestr("fixture/data_float_h.csv.gz", member)
            prior = root / "prior.json"
            prior.write_text(json.dumps({"complete": True,
                                         "compressed_member_sha256": hashlib.sha256(member).hexdigest(),
                                         "compressed_member_bytes": len(member),
                                         "counts": {"all_hour_rows_scanned": 0}}))
            out = root / "output"
            out.mkdir()
            (out / "PLAN.md").write_text("Synthetic test\n")
            (out / "analysis_contract.json").write_text(json.dumps(CONTRACT))
            original_build = self.m._build_shards

            def build_with_changed_contract(*args, **kwargs):
                result = original_build(*args, **kwargs)
                changed = dict(CONTRACT, map_range_mmHg=[0, 999])
                (out / "analysis_contract.json").write_text(json.dumps(changed))
                return result

            with mock.patch.object(self.m, "_build_shards", side_effect=build_with_changed_contract):
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    self.m.run_extraction(source, out, prior)
            self.assertFalse((out / "private/sicdb/ready.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
