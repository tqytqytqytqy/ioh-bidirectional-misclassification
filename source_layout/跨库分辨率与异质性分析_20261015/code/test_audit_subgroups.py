"""Synthetic-only metadata audit tests; no patient identifiers in assertions."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.dont_writebytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = None
        if (HERE / "audit_subgroups.py").exists():
            import audit_subgroups
            cls.a = audit_subgroups

    def setUp(self):
        self.assertIsNotNone(self.a, "metadata auditor implementation is missing")

    def test_contract_hash_mismatch_stops_before_metadata(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as d:
            p = Path(d) / "analysis_contract.json"
            p.write_text('{"age":"test"}')
            with self.assertRaisesRegex(ValueError, "contract_hash_mismatch"):
                self.a.read_contract(p, "0" * 64)
            self.assertEqual(self.a.read_contract(p, hashlib.sha256(p.read_bytes()).hexdigest())["age"], "test")

    def test_mimic_year_age_threshold_and_missing_are_not_adult_imputation(self):
        self.assertEqual(self.a.mimic_age({"anchor_age": "24", "anchor_year": "2150"}, "2151-02-03T00:00:00", 25), (25, True))
        self.assertEqual(self.a.mimic_age({"anchor_age": "24", "anchor_year": "2150"}, "2150-02-03T00:00:00", 24), (24, False))
        self.assertEqual(self.a.mimic_age(None, None, None), (None, None))
        with self.assertRaisesRegex(ValueError, "mimic_age_disagrees"):
            self.a.mimic_age({"anchor_age": "24", "anchor_year": "2150"}, "2151-02-03T00:00:00", 24)

    def test_sicdb_bin_does_not_claim_exact_age_or_verified_lower_bound(self):
        self.assertEqual(self.a.sicdb_age(25), (None, None, True))
        self.assertEqual(self.a.sicdb_age(20), (None, None, False))
        with self.assertRaisesRegex(ValueError, "sicdb_age_bin_invalid"):
            self.a.sicdb_age(24)

    def test_reference_decoding_checks_field_and_unknown_codes_fail_closed(self):
        refs = {"3126": {"ReferenceName": "SurgicalAdmissionType", "ReferenceValue": "Elective Surgery"},
                "1": {"ReferenceName": "Other", "ReferenceValue": "Elective Surgery"}}
        self.assertEqual(self.a.decode_reference("3126", "SurgicalAdmissionType", refs), "Elective Surgery")
        self.assertEqual(self.a.decode_reference("999", "SurgicalAdmissionType", refs), "unknown")
        self.assertEqual(self.a.decode_reference("1", "SurgicalAdmissionType", refs), "unknown")

    def test_stable_service_is_not_postoperative_and_cross_record_changes_are_unknown(self):
        rows = [{"transfertime": "2150-01-01 00:00:00", "curr_service": "SURG"},
                {"transfertime": "2150-01-02 12:00:00", "curr_service": "MED"}]
        self.assertEqual(self.a.service_label(rows, "2150-01-02 10:00:00", 3600), ("SURG", "stable_source_service"))
        self.assertEqual(self.a.service_label(rows, "2150-01-02 11:30:00", 3600), ("unknown", "service_changes_within_record"))
        self.assertEqual(self.a.service_label(rows, "2149-12-31 00:00:00", 3600)[0], "unknown")
        self.assertEqual(self.a.postoperative_label("SURG", 600, "Elective Surgery"), "unknown")

    def test_icu_offset_uses_fully_after_minute_not_truncated_offset(self):
        self.assertEqual(self.a.first_full_icu_minute(60, 300), 1)
        self.assertEqual(self.a.first_full_icu_minute(61, 300), 2)
        self.assertEqual(self.a.first_full_icu_minute(0, 300), 0)
        self.assertIsNone(self.a.first_full_icu_minute(None, 300))
        self.assertIsNone(self.a.first_full_icu_minute(301, 300))
        self.assertIsNone(self.a.first_full_icu_minute(-1, 300))

    def test_identity_mismatch_and_duplicate_linkage_fail_without_emitting_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate_record_key"):
            self.a.unique_index([{"k": "synthetic"}, {"k": "synthetic"}], "k")
        with self.assertRaisesRegex(ValueError, "subject_linkage_mismatch"):
            self.a.check_subject("1", "2")

    def test_shard_hash_mismatch_prevents_loading(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as d:
            p = Path(d) / "shard.npz"
            p.write_bytes(b"not a cache")
            with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
                self.a.checked_hash(p, "0" * 64)

    def test_fractional_or_float_identifiers_are_not_silently_cast(self):
        import numpy as np
        validate = getattr(self.a, "validate_sicdb_arrays", None)
        self.assertTrue(callable(validate), "strict metadata validator is missing")
        good = {"case_id": np.array([1]), "subject_id": np.array([2]), "age_bin": np.array([25]),
                "icu_offset_sec": np.array([61.]), "duration_sec": np.array([3600.])}
        validate(good, 1)
        for field in ("case_id", "subject_id", "age_bin"):
            bad = dict(good)
            bad[field] = np.array([25.1])
            with self.assertRaisesRegex(ValueError, "sicdb_shard_integer_metadata_required"):
                validate(bad, 1)
        with self.assertRaisesRegex(ValueError, "sicdb_shard_metadata_shape_mismatch"):
            validate(good, 2)

    def test_ambiguous_simultaneous_services_are_unknown(self):
        rows = [{"transfertime": "2150-01-01 00:00:00", "curr_service": "SURG"},
                {"transfertime": "2150-01-01 00:00:00", "curr_service": "MED"}]
        self.assertEqual(self.a.service_label(rows, "2150-01-01 00:00:00", 3600),
                         ("unknown", "ambiguous_service_at_record_start"))

    def test_restricted_writer_rejects_symlinks_without_touching_target(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as d:
            target = Path(d) / "existing.txt"
            target.write_text("keep")
            link = Path(d) / "metadata.csv"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "restricted_output_symlink"):
                self.a.write_restricted(link, [])
            self.assertEqual(target.read_text(), "keep")

    def test_restricted_writer_permissions_schema_and_nullable_boolean(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as d:
            p = Path(d) / "restricted" / "subgroup_metadata.csv"
            row = dict(source="SICdb", record_id="synthetic", subject_id="synthetic-subject", age=None,
                       age25_verified=None, icu_offset_sec=61, surgical_label="Elective Surgery",
                       postoperative_label="unknown", eligibility_basis={"contract_age_bin_ge25": True})
            self.a.write_restricted(p, [row])
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            self.assertEqual(p.parent.stat().st_mode & 0o777, 0o700)
            with p.open() as f:
                result = list(csv.DictReader(f))
            self.assertEqual(result[0]["age25_verified"], "")
            self.assertTrue(json.loads(result[0]["eligibility_basis"])["contract_age_bin_ge25"])
            self.assertEqual(list(result[0]), ["source", "record_id", "subject_id", "age", "age25_verified", "icu_offset_sec", "surgical_label", "postoperative_label", "eligibility_basis"])

    def test_aggregate_counts_do_not_publish_identity_arrays(self):
        rows = [dict(source="MIMIC", record_id="private-a", subject_id="private-p", age=25,
                     age25_verified=True, icu_offset_sec=-60, surgical_label="SURG",
                     postoperative_label="unknown", eligibility_basis={"base_eligible": True}),
                dict(source="MIMIC", record_id="private-b", subject_id="private-p", age=None,
                     age25_verified=None, icu_offset_sec=None, surgical_label="unknown",
                     postoperative_label="unknown", eligibility_basis={"base_eligible": False})]
        result = self.a.aggregate(rows)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["subjects"], 1)
        self.assertEqual(result["age25_verified"], {"true": 1, "false": 0, "unknown": 1})
        self.assertNotIn("private-", json.dumps(result))

    def test_end_to_end_synthetic_sources_hashes_labels_and_no_input_changes(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=HERE.parent) as d:
            root = Path(d)
            src = root / "synthetic_inputs"
            src.mkdir()
            contract = root / "analysis_contract.json"
            contract.write_text('{"age":"frozen","subgroup_gate":{"failure":"NO-GO"}}')
            def save(name, data):
                p = src / name
                p.write_text(json.dumps(data))
                return p
            record = dict(record_id="synthetic-record", subject_id=1, adult=True,
                          full_record_inside_icu=True, primary_family="ABP", original_numeric_valid_bins=360)
            records = save("records.json", [record])
            linkage = save("linkage.json", [dict(record_id="synthetic-record", subject_id=1, hadm_id=10,
                        base_datetime="2151-01-01T12:00:00", duration_seconds=3600,
                        clinical=dict(age_year_level=25, adult=True, full_record_inside_icu=True))])
            clinical = src / "clinical.zip"
            with zipfile.ZipFile(clinical, "w") as z:
                for name, text in {
                    "patients": "subject_id,anchor_age,anchor_year\n1,24,2150\n",
                    "icustays": "subject_id,hadm_id,stay_id,intime,outtime\n1,10,100,2151-01-01 00:00:00,2151-01-03 00:00:00\n",
                    "services": "subject_id,hadm_id,transfertime,curr_service\n1,10,2151-01-01 00:00:00,SURG\n",
                    "procedures_icd": "subject_id,hadm_id,chartdate,icd_code,icd_version\n1,10,2151-01-01,0000,9\n",
                }.items():
                    z.writestr("mimic/hosp/" + name + ".csv.gz", gzip.compress(text.encode()))
            status = save("source_status.json", {"clinical_archive": str(clinical), "clinical_members": {}})
            qc = src / "qc.npz"
            np.savez(qc, placeholder=[1])
            qc_ready = save("qc_ready.json", {"status": "ready", "input_sha256": {
                "records.json": hashlib.sha256(records.read_bytes()).hexdigest(),
                "qc_inputs.npz": hashlib.sha256(qc.read_bytes()).hexdigest()}})
            sic = src / "sic.zip"
            with zipfile.ZipFile(sic, "w") as z:
                z.writestr("sic/cases.csv.gz", gzip.compress(b"CaseID,PatientID,TimeOfStay,ICUOffset,AgeOnAdmission,SurgicalAdmissionType\n1,2,3600,61,25,3126\n"))
                z.writestr("sic/d_references.csv.gz", gzip.compress(b"ReferenceGlobalID,ReferenceName,ReferenceValue\n3126,SurgicalAdmissionType,Elective Surgery\n"))
            shard = src / "shard.npz"
            np.savez(shard, case_id=[1], subject_id=[2], age_bin=[25], icu_offset_sec=[61.], duration_sec=[3600.])
            ready = save("ready.json", {"status": "ready", "cases": 1, "subjects": 1, "minute_origin_sec": 0,
                    "source_sha256": hashlib.sha256(sic.read_bytes()).hexdigest(),
                    "shards": [{"path": "shard.npz", "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(), "cases": 1}]})
            paths = dict(mimic_records=records, mimic_linkage=linkage, mimic_source_status=status,
                         mimic_qc=qc, mimic_qc_ready=qc_ready, sicdb_ready=ready, sicdb_zip=sic)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
            result = self.a.run_audit(root, paths, hashlib.sha256(contract.read_bytes()).hexdigest())
            self.assertEqual(result["sources"]["MIMIC"]["age25_verified"]["true"], 1)
            self.assertEqual(result["sources"]["SICdb"]["surgical_labels"], {"Elective Surgery": {"records": 1, "subjects": 1}})
            self.assertEqual(result["gates"]["postoperative_timing"]["status"], "NO-GO")
            self.assertTrue(result["verification"]["source_files_unchanged"])
            for p, digest in before.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            with (root / "restricted/subgroup_metadata.csv").open() as stream:
                private_rows = list(csv.DictReader(stream))
            self.assertEqual(private_rows[0]["surgical_label"], "SURG")
            self.assertEqual(private_rows[0]["icu_offset_sec"], "-43200.0")
            self.assertEqual(private_rows[1]["age"], "")
            self.assertFalse(json.loads(private_rows[1]["eligibility_basis"])["exact_age_available"])
            self.assertEqual(private_rows[1]["postoperative_label"], "unknown")
            self.assertNotIn("synthetic-record", (root / "qa/subgroup_audit.json").read_text())

    def test_public_audit_is_compact_with_complete_provenance_restricted(self):
        publish = getattr(self.a, "publish_audit", None)
        self.assertTrue(callable(publish), "aggregate/provenance publisher is missing")
        with tempfile.TemporaryDirectory(dir=HERE.parent) as d:
            root = Path(d)
            (root / "qa").mkdir()
            old = {"source_hashes": [{"role": "frozen shard", "path": "/Users/private/shard.npz", "sha256": "a" * 64}],
                   "legacy_evidence": "preserve-exactly"}
            (root / "qa/subgroup_audit.json").write_text(json.dumps(old))
            full = {"source_hashes": old["source_hashes"], "sources": {}, "gates": {}, "source": {}, "status": "completed"}
            compact = publish(root, full)
            public = (root / "qa/subgroup_audit.json").read_text()
            self.assertNotIn("/Users/", public)
            self.assertNotIn("source_hashes", compact)
            provenance = root / "restricted/subgroup_source_provenance.json"
            preserved = json.loads(provenance.read_text())
            self.assertEqual(preserved["source_hashes"], old["source_hashes"])
            self.assertEqual(preserved["legacy_public_audit_snapshot"], old)
            self.assertEqual(provenance.stat().st_mode & 0o777, 0o600)
            self.assertEqual(compact["source_provenance"]["sha256"], hashlib.sha256(provenance.read_bytes()).hexdigest())
            publish(root, full)
            self.assertEqual(json.loads(provenance.read_text())["legacy_public_audit_snapshot"], old)


if __name__ == "__main__":
    unittest.main()
