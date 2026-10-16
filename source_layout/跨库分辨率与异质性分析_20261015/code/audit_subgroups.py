"""Read-only source metadata audit. No trajectory effects or clinical phenotyping."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

sys.dont_writebytecode = True

CONTRACT_SHA256 = "3157dfa67db7f7fbcfcbd859b8e46d978583d8851b12348ea2c71fb68dcae61b"
FIELDS = ["source", "record_id", "subject_id", "age", "age25_verified", "icu_offset_sec",
          "surgical_label", "postoperative_label", "eligibility_basis"]
SURGICAL_SERVICES = {"CSURG", "NSURG", "ORTHO", "PSURG", "SURG", "TSURG", "VSURG"}
KNOWN_SERVICES = SURGICAL_SERVICES | {"CMED", "DENT", "ENT", "EYE", "GU", "GYN", "MED",
                                     "NB", "NBB", "NMED", "OBS", "OMED", "PSYCH", "TRAUM"}
DOCUMENTATION = [
    {"url": "https://physionet.org/content/sicdb/1.0.8/", "version": "1.0.8; published 2024-09-10",
     "verified": "ICUOffset is seconds to first ICU bed assignment; referenced codes require d_references."},
    {"url": "https://www.sicdb.com/Documentation/SICdb_Documentation", "version": "oldid=345; checked 2026-10-15",
     "verified": "First transfer can be to intensive or intermediate care. Age documentation does not unambiguously specify lower-bound bin encoding."},
    {"url": "https://mimic.mit.edu/docs/iv/modules/hosp/patients.html", "version": "checked 2026-10-15; local data 3.1",
     "verified": "anchor_age is age in anchor_year; values above 89 are top-coded to 91."},
    {"url": "https://mimic.mit.edu/docs/iv/modules/hosp/services.html", "version": "checked 2026-10-15; local data 3.1",
     "verified": "curr_service records treating service after transfertime; surgical service is not completed surgery or postoperative timing."},
    {"url": "https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd.html", "version": "checked 2026-10-15; local data 3.1",
     "verified": "Hospital-billed procedures have date-only chartdate. No unapproved surgical-code phenotype is introduced."},
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_hash(path, expected=None):
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError("source_hash_mismatch")
    return digest


def read_contract(path, expected=CONTRACT_SHA256):
    if sha256(path) != expected:
        raise ValueError("contract_hash_mismatch")
    return json.loads(Path(path).read_text())


def unique_index(rows, key):
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError("duplicate_record_key")
        result[value] = row
    return result


def check_subject(a, b):
    if str(a) != str(b):
        raise ValueError("subject_linkage_mismatch")


def as_time(value):
    return datetime.fromisoformat(str(value))


def mimic_age(patient, base, stored):
    if patient is None or base is None:
        return None, None
    age = int(patient["anchor_age"]) + as_time(base).year - int(patient["anchor_year"])
    if not 0 <= age <= 130:
        raise ValueError("mimic_age_invalid")
    if stored is not None and float(stored) != age:
        raise ValueError("mimic_age_disagrees")
    return age, age >= 25


def sicdb_age(age_bin):
    age_bin = float(age_bin)
    if not math.isfinite(age_bin) or age_bin < 0 or age_bin > 90 or age_bin % 5:
        raise ValueError("sicdb_age_bin_invalid")
    # Preserve the frozen encoded threshold, without inventing bin boundaries.
    return None, None, age_bin >= 25


def decode_reference(code, field, references):
    row = references.get(str(code))
    if not row or row.get("ReferenceName") != field:
        return "unknown"
    value = row.get("ReferenceValue", "").strip()
    return "unknown" if value in {"", ".", "Unknown"} else value


def service_label(rows, base, duration):
    if base is None:
        return "unknown", "record_datetime_unavailable"
    start = as_time(base)
    end = start + timedelta(seconds=float(duration))
    dated = sorted((as_time(r["transfertime"]), r["curr_service"]) for r in rows if r.get("transfertime"))
    before = [(t, value) for t, value in dated if t <= start]
    if not before:
        return "unknown", "no_service_at_record_start"
    last = before[-1][0]
    current = {value for t, value in before if t == last}
    if len(current) != 1:
        return "unknown", "ambiguous_service_at_record_start"
    label = next(iter(current))
    if label not in KNOWN_SERVICES:
        return "unknown", "undocumented_service_code"
    if any(value != label for t, value in dated if start < t < end):
        return "unknown", "service_changes_within_record"
    return label, "stable_source_service"


def postoperative_label(service=None, icu_offset=None, admission_category=None):
    return "unknown"


def first_full_icu_minute(offset, duration):
    if offset is None:
        return None
    offset, duration = float(offset), float(duration)
    if not math.isfinite(offset) or offset < 0 or offset > duration:
        return None
    first = math.ceil(offset / 60)
    return first if (first + 1) * 60 <= duration else None


def aggregate(rows):
    result = {"records": len(rows), "subjects": len({str(r["subject_id"]) for r in rows}),
              "age_available_records": sum(r["age"] is not None for r in rows),
              "age25_verified": {name: sum(r["age25_verified"] is value for r in rows)
                                 for name, value in (("true", True), ("false", False), ("unknown", None))},
              "base_eligible_records": sum(r["eligibility_basis"].get("base_eligible", False) for r in rows)}
    for field, key in (("surgical_label", "surgical_labels"), ("postoperative_label", "postoperative_labels")):
        result[key] = {label: {"records": sum(r[field] == label for r in rows),
                              "subjects": len({str(r["subject_id"]) for r in rows if r[field] == label})}
                       for label in sorted({r[field] for r in rows})}
    return result


def write_restricted(path, rows):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("restricted_output_symlink")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in FIELDS}
            output["eligibility_basis"] = json.dumps(output["eligibility_basis"], sort_keys=True, ensure_ascii=True, allow_nan=False)
            if output["age25_verified"] is not None:
                output["age25_verified"] = str(output["age25_verified"]).lower()
            writer.writerow(output)


def publish_audit(root, full_report):
    root = Path(root)
    qa = root / "qa"
    qa.mkdir(exist_ok=True)
    public_path = qa / "subgroup_audit.json"
    private = root / "restricted/subgroup_source_provenance.json"
    if private.is_symlink() or private.parent.is_symlink():
        raise ValueError("restricted_output_symlink")
    if not private.resolve().is_relative_to(root.resolve()) or not public_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("audit_output_outside_root")
    private.parent.mkdir(exist_ok=True, mode=0o700)
    private.parent.chmod(0o700)
    legacy = None
    if private.exists():
        legacy = json.loads(private.read_text()).get("legacy_public_audit_snapshot")
    if public_path.exists():
        previous = json.loads(public_path.read_text())
        if "source_hashes" in previous:
            legacy = previous
    provenance = dict(full_report)
    if legacy is not None:
        provenance["legacy_public_audit_snapshot"] = legacy
    fd = os.open(private, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(provenance, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    compact = {key: value for key, value in full_report.items() if key != "source_hashes"}
    compact["sources"] = json.loads(json.dumps(compact.get("sources", {})))
    for source in compact["sources"].values():
        details = source.get("details", {})
        details.pop("clinical_member_hashes", None)
        details.pop("source_member_hashes", None)
    compact["source_provenance"] = {"restricted_file": private.name, "sha256": sha256(private),
                                     "source_entries": len(full_report.get("source_hashes", []))}
    content = json.dumps(compact, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    if "/Users/" in content or "/Volumes/" in content:
        raise ValueError("public_audit_contains_absolute_source_path")
    public_path.write_text(content)
    return compact


def default_paths(root):
    parent = root.parent
    prior = parent / "外部验证可行性审计_20261015/mimic"
    mimic = parent / "MIMIC技术重复分析_20261015"
    return dict(mimic_records=mimic / "private/records.json", mimic_qc=mimic / "private/qc_inputs.npz",
                mimic_qc_ready=mimic / "private/qc_ready.json", mimic_linkage=prior / "private/record_audit.json",
                mimic_source_status=prior / "source_status.json", mimic_linkage_code=prior / "audit_mimic.py",
                mimic_qc_code=mimic / "qc/run_full_qc.py",
                sicdb_ready=parent / "SICdb分钟级对齐分析_20261015/private/sicdb/ready.json",
                sicdb_zip=parent.parent / "salzburg-intensive-care-database-sicdb-a-freely-accessible-intensive-care-database-1.0.8.zip")


def read_member(archive, suffix, fingerprints):
    names = [n for n in archive.namelist() if n.endswith("/" + suffix)]
    if len(names) != 1:
        raise ValueError("archive_member_missing_or_ambiguous")
    name = names[0]
    blob = archive.read(name)
    info = archive.getinfo(name)
    fingerprints[suffix] = {"archive_member": name, "member_bytes": len(blob),
                            "zip_crc32": f"{info.CRC:08x}",
                            "sha256_compressed_member": hashlib.sha256(blob).hexdigest()}
    return blob


def csv_gzip(blob):
    return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(blob)), encoding="utf-8-sig"))


def validate_sicdb_arrays(data, expected_count):
    count = len(data["case_id"])
    if count != expected_count or any(value.shape != (count,) for value in data.values()):
        raise ValueError("sicdb_shard_metadata_shape_mismatch")
    if any(data[key].dtype.kind not in "iu" for key in ("case_id", "subject_id", "age_bin")):
        raise ValueError("sicdb_shard_integer_metadata_required")


def load_mimic(paths, track):
    records = json.loads(paths["mimic_records"].read_text())
    unique_index(records, "record_id")
    linkage = unique_index(json.loads(paths["mimic_linkage"].read_text()), "record_id")
    status = json.loads(paths["mimic_source_status"].read_text())
    ready = json.loads(paths["mimic_qc_ready"].read_text())
    if ready.get("status") != "ready":
        raise ValueError("mimic_qc_not_ready")
    for name, key in (("records.json", "mimic_records"), ("qc_inputs.npz", "mimic_qc")):
        checked_hash(paths[key], ready["input_sha256"][name])
    wanted = {str(r["subject_id"]) for r in records}
    linked = []
    for record in records:
        row = linkage.get(str(record["record_id"]))
        if row is None:
            raise ValueError("frozen_record_linkage_missing")
        check_subject(row["subject_id"], record["subject_id"])
        if record["adult"] is not row["clinical"]["adult"] or record["full_record_inside_icu"] is not row["clinical"]["full_record_inside_icu"]:
            raise ValueError("frozen_eligibility_linkage_mismatch")
        linked.append((record, row))
    archive_path = Path(status["clinical_archive"])
    fingerprints, patients, stays, services, procedures = {}, {}, defaultdict(list), defaultdict(list), defaultdict(list)
    available = archive_path.is_file()
    if available:
        track(archive_path, "MIMIC clinical archive; member hashes only", full_hash=False)
        with zipfile.ZipFile(archive_path) as archive:
            for name in ("patients.csv.gz", "icustays.csv.gz", "services.csv.gz", "procedures_icd.csv.gz"):
                blob = read_member(archive, name, fingerprints)
                expected = status.get("clinical_members", {}).get(name, {}).get("sha256_compressed_member")
                if expected is not None and fingerprints[name]["sha256_compressed_member"] != expected:
                    raise ValueError("clinical_member_prior_hash_mismatch")
                for row in csv_gzip(blob):
                    if row["subject_id"] not in wanted:
                        continue
                    if name == "patients.csv.gz":
                        if row["subject_id"] in patients:
                            raise ValueError("duplicate_patient_key")
                        patients[row["subject_id"]] = row
                    else:
                        key = row["subject_id"], row["hadm_id"]
                        {"icustays.csv.gz": stays, "services.csv.gz": services, "procedures_icd.csv.gz": procedures}[name][key].append(row)
    output = []
    service_reasons = Counter()
    procedure_counts = Counter()
    topcoded = 0
    for record, row in linked:
        subject = str(record["subject_id"])
        pair = subject, str(row["hadm_id"])
        clinical = row["clinical"]
        age, age25 = mimic_age(patients.get(subject), row.get("base_datetime"), clinical.get("age_year_level"))
        base_eligible = (record["adult"] is True and record["full_record_inside_icu"] is True
                         and bool(record["primary_family"]) and record["original_numeric_valid_bins"] > 0)
        if age is not None and clinical["adult"] != (age >= 18):
            raise ValueError("adult_age_disagrees")
        topcode = patients.get(subject, {}).get("anchor_age") == "91"
        topcoded += topcode
        label, reason = service_label(services[pair], row.get("base_datetime"), row["duration_seconds"])
        service_reasons[reason] += 1
        offset = None
        if row.get("base_datetime"):
            base = as_time(row["base_datetime"])
            end = base + timedelta(seconds=float(row["duration_seconds"]))
            containing = [s for s in stays[pair] if as_time(s["intime"]) <= base and as_time(s["outtime"]) >= end]
            if len(containing) == 1:
                offset = (as_time(containing[0]["intime"]) - base).total_seconds()
            elif base_eligible and available:
                raise ValueError("frozen_icu_record_not_uniquely_contained")
            proc = procedures[pair]
            procedure_counts["records_with_billed_procedures"] += bool(proc)
            procedure_counts["records_with_procedure_dates_before_waveform_day"] += any(as_time(p["chartdate"]).date() < base.date() for p in proc)
            procedure_counts["records_with_same_day_procedure_dates"] += any(as_time(p["chartdate"]).date() == base.date() for p in proc)
        basis = {"base_eligible": base_eligible, "parent_selection_intersection_required": True,
                 "adult": record["adult"], "full_record_inside_icu": record["full_record_inside_icu"],
                 "numeric_family_present": bool(record["primary_family"]),
                 "age_kind": "deidentified_year_level_topcoded" if topcode else "deidentified_year_level",
                 "age_source_field": "clinical.age_year_level", "age_raw_patients_verified": age is not None,
                 "age_formula": "anchor_age + waveform_start_year - anchor_year",
                 "service_basis": reason, "surgical_service_context": label in SURGICAL_SERVICES,
                 "surgery_performed_verified": False, "postoperative_timing_verified": False,
                 "icu_offset_basis": "linked_icustays.intime_minus_waveform_start" if offset is not None else "unknown"}
        output.append(dict(source="MIMIC", record_id=record["record_id"], subject_id=record["subject_id"], age=age,
                           age25_verified=age25, icu_offset_sec=offset, surgical_label=label,
                           postoperative_label=postoperative_label(label, offset), eligibility_basis=basis))
    info = {"raw_clinical_archive_available": available, "clinical_archive_full_hashed": False,
            "clinical_member_hashes": fingerprints, "service_basis_counts": dict(service_reasons),
            "procedure_availability_only": dict(procedure_counts), "topcoded_age_records": topcoded,
            "age_formula_checks": sum(r["age"] is not None for r in output),
            "selection": "Inventory only; intersect frozen parent-selected record IDs. No trajectory selection rerun."}
    return output, info


def load_sicdb(paths, track):
    import numpy as np
    ready = json.loads(paths["sicdb_ready"].read_text())
    if ready.get("status") != "ready" or ready.get("minute_origin_sec") != 0:
        raise ValueError("sicdb_cache_not_ready_or_origin_nonzero")
    digest = track(paths["sicdb_zip"], "SICdb original local ZIP")
    if digest != ready["source_sha256"]:
        raise ValueError("sicdb_zip_frozen_hash_mismatch")
    fingerprints = {}
    with zipfile.ZipFile(paths["sicdb_zip"]) as archive:
        cases = unique_index(list(csv_gzip(read_member(archive, "cases.csv.gz", fingerprints))), "CaseID")
        refs = unique_index(list(csv_gzip(read_member(archive, "d_references.csv.gz", fingerprints))), "ReferenceGlobalID")
        if any(n.endswith("/Documentation.pdf") for n in archive.namelist()):
            read_member(archive, "Documentation.pdf", fingerprints)
    rows, seen = [], set()
    offset_status = Counter()
    age_bins = Counter()
    mapping = {}
    cardiac = Counter()
    metadata_checks = 0
    last = None
    for descriptor in ready["shards"]:
        path = (paths["sicdb_ready"].parent / descriptor["path"]).resolve()
        if not path.is_relative_to(paths["sicdb_ready"].parent.resolve()):
            raise ValueError("shard_path_outside_cache")
        track(path, "SICdb frozen shard", expected=descriptor["sha256"])
        with np.load(path, allow_pickle=False) as cache:
            data = {k: cache[k] for k in ("case_id", "subject_id", "age_bin", "icu_offset_sec", "duration_sec")}
        validate_sicdb_arrays(data, descriptor["cases"])
        count = len(data["case_id"])
        for j in range(count):
            case = int(data["case_id"][j])
            if case in seen or (last is not None and case <= last):
                raise ValueError("sicdb_case_order_or_duplicates")
            seen.add(case)
            last = case
            row = cases.get(str(case))
            if row is None:
                raise ValueError("sicdb_case_not_in_source")
            subject = int(data["subject_id"][j])
            check_subject(subject, row["PatientID"])
            age_bin = int(data["age_bin"][j])
            duration = float(data["duration_sec"][j])
            offset_value = float(data["icu_offset_sec"][j])
            original_offset = None if row["ICUOffset"] in {"", "NULL"} else float(row["ICUOffset"])
            offset = offset_value if math.isfinite(offset_value) else None
            if (age_bin != int(row["AgeOnAdmission"]) or duration != float(row["TimeOfStay"])
                    or offset != original_offset):
                raise ValueError("sicdb_source_cache_metadata_mismatch")
            metadata_checks += 1
            age, age25, encoded_pass = sicdb_age(age_bin)
            age_bins[str(age_bin)] += 1
            first = first_full_icu_minute(offset, duration)
            if offset is None:
                state = "missing"
            elif offset < 0 or offset > duration:
                state = "outside_record"
            elif first is None:
                state = "no_complete_minute_after_icu_assignment"
            else:
                state = "usable"
            offset_status[state] += 1
            raw_code = row.get("SurgicalAdmissionType", "")
            label = decode_reference(raw_code, "SurgicalAdmissionType", refs)
            mapping[raw_code] = {"source_value": refs.get(raw_code, {}).get("ReferenceValue"),
                                 "reference_name": refs.get(raw_code, {}).get("ReferenceName"),
                                 "output_label": label}
            for field in ("HeartSurgeryBeginOffset", "HeartSurgeryEndOffset"):
                if row.get(field) not in (None, "", "NULL"):
                    cardiac[field + "_nonmissing_records"] += 1
            basis = {"base_eligible": encoded_pass, "parent_selection_intersection_required": True,
                     "age_kind": "five_year_source_encoding_not_verified_lower_bound", "source_age_bin": age_bin,
                     "contract_age_bin_ge25": encoded_pass, "exact_age_available": False,
                     "age25_lower_bound_verified": False, "icu_offset_status": state,
                     "first_fully_at_or_after_icu_minute": first,
                     "surgical_source_field": "cases.SurgicalAdmissionType", "surgical_source_code": raw_code,
                     "surgical_reference_decoded": raw_code in refs and refs[raw_code].get("ReferenceName") == "SurgicalAdmissionType",
                     "surgery_performed_timing_verified": False, "postoperative_timing_verified": False}
            rows.append(dict(source="SICdb", record_id=case, subject_id=subject, age=age, age25_verified=age25,
                             icu_offset_sec=offset, surgical_label=label, postoperative_label=postoperative_label(None, offset, label),
                             eligibility_basis=basis))
    if len(rows) != ready["cases"] or len({r["subject_id"] for r in rows}) != ready["subjects"]:
        raise ValueError("sicdb_ready_population_mismatch")
    info = {"original_case_records": len(cases), "cache_shards_verified": len(ready["shards"]),
            "source_metadata_rows_exactly_matched": metadata_checks, "source_member_hashes": fingerprints,
            "source_age_bin_record_counts": dict(sorted(age_bins.items(), key=lambda pair: int(pair[0]))),
            "icu_offset_status_records": dict(offset_status), "surgical_admission_code_map": mapping,
            "cardiac_timing_fields_availability_only": dict(cardiac),
            "selection": "All frozen cache records; encoded age >=25 is preserved, not silently replaced with >=30."}
    return rows, info


def run_audit(root, paths=None, expected_contract_hash=CONTRACT_SHA256):
    root = Path(root).resolve()
    contract_path = root / "analysis_contract.json"
    read_contract(contract_path, expected_contract_hash)
    paths = {k: Path(v) for k, v in (paths or default_paths(root)).items()}
    tracked = {}
    hashes = []
    def track(path, role, expected=None, full_hash=True):
        path = Path(path)
        stat = path.stat()
        digest = checked_hash(path, expected) if full_hash else None
        tracked[path] = (digest, stat.st_size, stat.st_mtime_ns)
        hashes.append({"role": role, "path": str(path), "sha256": digest, "bytes": stat.st_size,
                       "verification": "full_sha256" if full_hash else "archive_stat_and_selected_member_sha256"})
        return digest
    track(contract_path, "Frozen analysis contract", expected_contract_hash)
    for key, path in paths.items():
        if key != "sicdb_zip":
            track(path, key)
    mimic, mimic_info = load_mimic(paths, track)
    sicdb, sicdb_info = load_sicdb(paths, track)
    rows = mimic + sicdb
    if len({(r["source"], str(r["record_id"])) for r in rows}) != len(rows):
        raise ValueError("output_duplicate_source_record")
    # Rehash actual audit inputs; the large MIMIC archive is member-hashed, stat-guarded.
    for path, (digest, size, mtime) in tracked.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime) or (digest is not None and sha256(path) != digest):
            raise ValueError("source_changed_during_audit")
    output = root / "restricted/subgroup_metadata.csv"
    write_restricted(output, rows)
    grouped = {}
    for name, data, info in (("MIMIC", mimic, mimic_info), ("SICdb", sicdb, sicdb_info)):
        grouped[name] = {**aggregate(data), "base_eligible_inventory": aggregate([r for r in data if r["eligibility_basis"]["base_eligible"]]),
                         "details": info}
    code = Path(__file__).resolve()
    report = {"status": "completed_metadata_audit", "contract_sha256": expected_contract_hash,
              "generated_utc": datetime.now(timezone.utc).isoformat(), "effect_estimation_performed": False,
              "physician_adjudication_performed": False, "sources": grouped,
              "gates": {
                  "MIMIC_age25": {"status": "GO" if all(r["age25_verified"] is not None for r in mimic if r["eligibility_basis"]["base_eligible"]) else "NO-GO",
                                  "scope": "Verified deidentified year-level age, not exact age; intersect parent-selected records."},
                  "SICdb_encoded_age_cutoff": {"status": "GO_SOURCE_CODE_ONLY", "scope": "Frozen AgeOnAdmission encoding >=25; no threshold change."},
                  "SICdb_true_age_lower_bound": {"status": "NO-GO", "reason": "Official PDF p5 and online schema say rounded +/-5 years; five-year grouping text alone does not prove each code is its lower bound. No complete age-matching claim."},
                  "SICdb_ICU": {"status": "GO", "scope": "Mask minute bins with minute_start_sec < ICUOffset as missing, then retain complete origin-zero hour blocks with >=48/60 finite post-offset minutes, intersected with the original qualifying-hour mask. A crossing hour may qualify; pre-offset bins remain missing. Require a finite nonnegative ICUOffset within the retained minute grid. Do not re-anchor hours at ICUOffset. This represents first ICU/intermediate-care assignment, not uninterrupted ICU residence or surgery timing."},
                  "surgical_context": {"status": "GO_SOURCE_CATEGORIES_ONLY", "scope": "SICdb SurgicalAdmissionType labels and MIMIC stable treating-service codes; source-specific context, not cross-source equivalent operations."},
                  "surgical_admission_icu": {"status": "GO", "source": "SICdb", "labels": ["Elective Surgery", "Urgent Surgery"],
                                             "scope": "Source-defined surgical admission category, not verified intraoperative or postoperative time. Mask minute bins with minute_start_sec < ICUOffset as missing, then retain complete origin-zero hour blocks with >=48/60 finite post-offset minutes, intersected with the original qualifying-hour mask. A crossing hour may qualify; pre-offset bins remain missing. Require a finite nonnegative ICUOffset within the retained minute grid. Do not re-anchor hours at ICUOffset."},
                  "no_surgery_icu": {"status": "GO", "source": "SICdb", "labels": ["No Surgery"],
                                     "scope": "Separate source-defined No Surgery category; unknown is not No Surgery. Mask minute bins with minute_start_sec < ICUOffset as missing, then retain complete origin-zero hour blocks with >=48/60 finite post-offset minutes, intersected with the original qualifying-hour mask. A crossing hour may qualify; pre-offset bins remain missing. Require a finite nonnegative ICUOffset within the retained minute grid. Do not re-anchor hours at ICUOffset."},
                  "surgical_service": {"status": "GO", "source": "MIMIC", "labels": ["CSURG", "SURG", "NSURG", "VSURG"],
                                       "scope": "Stable treating service over the entire waveform record only; not proof an operation occurred or finished."},
                  "postoperative_timing": {"status": "NO-GO", "reason": "No generally applicable, source-verified surgery-end linkage. SURG, pre-ICU status and billed procedure presence are insufficient."},
                  "subgroup_effects": {"status": "NOT_ASSESSED", "reason": "Metadata audit only. Parent must intersect selected windows and apply frozen patient/event reporting gates."}},
              "scope_revision_note": {"revision_date":"2026-10-15","change_type":"metadata_audit_scope_text_only","reason":"Align audit wording with the frozen minute-bin masking and 80-percent origin-anchored hour gate; the superseded whole-hour-start requirement was not the implemented rule.","superseded_scopes":{"SICdb_ICU":"Minutes fully at/after first ICU/intermediate-care assignment, using ceil(ICUOffset/60); not uninterrupted ICU residence or surgery timing.","surgical_admission_icu":"Source-defined surgical admission category, not verified intraoperative or postoperative time. Intersect complete origin-zero qualifying hours with hour_start_sec >= ICUOffset and hour_end_sec <= duration; require 0 <= ICUOffset <= duration.","no_surgery_icu":"Separate source-defined No Surgery category; apply same valid-offset/full-hour constraints. Unknown is not No Surgery."},"effects_rerun":False,"effect_data_or_threshold_changes":False,"metadata_unchanged":True},
              "source": {
                  "MIMIC": {"record_key": "record_id", "record_key_type": "string", "age_field": "age",
                            "age_kind": "deidentified_year_level_not_exact", "surgical_label_field": "surgical_label",
                            "group_key": "surgical_service", "included_labels": ["CSURG", "SURG", "NSURG", "VSURG"],
                            "stable_entire_record_required": True, "postoperative_supported": False},
                  "SICdb": {"record_key": "record_id", "record_key_type": "string", "age_field": None,
                            "age_bin_source": "frozen_shard.age_bin", "encoded_age_bin_cutoff": 25,
                            "surgical_label_field": "surgical_label", "group_key": "surgical_admission_icu",
                            "included_labels": ["Elective Surgery", "Urgent Surgery"], "reference_labels": ["No Surgery"],
                            "icu_offset_field": "icu_offset_sec", "invalid_offset_records": sicdb_info["icu_offset_status_records"].get("outside_record", 0),
                            "icu_context": "first intensive_or_intermediate_care_assignment", "postoperative_supported": False}},
              "metadata_schema": {"fields": FIELDS, "boolean_encoding": "true/false/empty; empty means unknown, never false",
                                  "age": "MIMIC deidentified year-level age (top coding documented in basis); SICdb null, source_age_bin in eligibility_basis",
                                  "eligibility_basis": "JSON object in CSV cell; base_eligible is metadata-only, NOT selected analysis cohort",
                                  "surgical_label": "Exact SICdb reference value or MIMIC stable curr_service; unknown for missing/ambiguous/undocumented codes",
                                  "postoperative_label": "unknown: no postoperative phenotype approved or verified",
                                  "icu_offset_sec": "SICdb seconds from MetaVision origin; MIMIC linked ICU intime minus waveform origin, generally negative; not interchangeable origins"},
              "limitations": [
                  "MIMIC 64 rows are the frozen candidate inventory, not 50 selected people; no signal coverage selection is recreated here.",
                  "Age precision differs by source. SICdb age25_verified is deliberately null, even when the frozen source code cutoff passes.",
                  "SICdb ICUOffset is first ICU or intermediate-care transfer, not hospital admission, incision, surgery end or continuous ICU occupancy.",
                  "SICdb source surgical categories may be unknown; no diagnosis-name keyword inference or imported phenotype is used.",
                  "MIMIC surgical service does not establish completed surgery. No-day-level procedure date is promoted to exact surgery end.",
                  "Heart-surgery offsets exist for a subset but cannot establish a broad postoperative subgroup without source-compatible timing verification.",
                  "MIMIC original ZIP is not fully hashed; selected clinical gzip members are SHA256-checked, including prior patients/icustays fingerprints; archive stat is unchanged.",
                  "Official SICdb online documentation has stale/inconsistent table details; the local release dictionary is authoritative for categorical decoding.",
                  "All aggregates refer to metadata availability, not clinical outcome incidence or effect estimates."],
              "documentation": DOCUMENTATION, "source_hashes": hashes,
              "verification": {"source_files_unchanged": True, "source_hashes_recomputed_at_end": True,
                               "records_are_unique_by_source": True, "output_ids_restricted": True,
                               "raw_trajectories_loaded": False, "protected_analysis_files_edited": False},
              "outputs": {"restricted_metadata": {"path": "restricted/subgroup_metadata.csv", "sha256": sha256(output),
                                                  "rows": len(rows), "directory_mode": "0700", "file_mode": "0600"}},
              "auditor_code_sha256": sha256(code),
              "tests_sha256": sha256(code.with_name("test_audit_subgroups.py"))}
    return publish_audit(root, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = run_audit(args.root)
        print(json.dumps({"status": result["status"], "records": result["outputs"]["restricted_metadata"]["rows"],
                          "effect_estimation_performed": False, "gates": {k: v["status"] for k, v in result["gates"].items()}}))
    except Exception as error:
        # Exceptions from clinical readers may contain IDs: expose class only.
        print(json.dumps({"status": "NO-GO", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
