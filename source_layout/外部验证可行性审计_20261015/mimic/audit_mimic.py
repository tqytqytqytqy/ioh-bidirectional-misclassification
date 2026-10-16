"""Read-only MIMIC-IV Waveform 0.1.0 source feasibility audit.

All writes are confined to this script's directory. No remote data calls.
"""
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PRIVATE = OUT / "private"
PRIVATE.mkdir(exist_ok=True, mode=0o700)
os.chmod(PRIVATE, 0o700)
os.environ["MPLCONFIGDIR"] = str(PRIVATE / "mplconfig")
sys.path.insert(0, str(PRIVATE / "deps"))

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import platform
import re
import traceback
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import wfdb

from audit_core import (age_at_year, bin_medians, choose_family, coherent, coverage,
                        family_rank, interval_summary, low_runs, map_valid,
                        mask_runs, numeric_alias, overlap_seconds, wave_alias,
                        wave_quality)

SOURCE = Path(__import__('os').environ.get('MIMIC_WAVEFORM_ROOT', 'data/mimic4wdb-0.1.0'))
ARCHIVE = Path(__import__('os').environ.get('MIMIC_CLINICAL_ZIP', 'data/mimic-iv-3.1.zip'))
MAIN_BASELINE_USER_SUPPLIED = "88119204d3bb5c5155f2b5eeac18b7f44974cc89d3794c4b1b210eeeac783de7"


def save_json(path, value):
    path = path.resolve()
    if not path.is_relative_to(OUT):
        raise ValueError("Output outside allowed audit directory")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not path.resolve().is_relative_to(OUT):
        raise ValueError("Output outside allowed audit directory")
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def stat(path):
    s = path.stat()
    return {"size":s.st_size, "mtime_ns":s.st_mtime_ns, "mode":s.st_mode}


def inventory():
    patient_dirs = [SOURCE / x.strip() for x in (SOURCE / "RECORDS").read_text().splitlines() if x.strip()]
    records = sorted(p for d in patient_dirs for p in d.iterdir()
                     if p.is_dir() and (p / (p.name + ".hea")).is_file())
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file() and not any(x.startswith(".") for x in p.relative_to(SOURCE).parts))
    return patient_dirs, records, files


def source_snapshot(files, hash_files):
    result = {}
    for i, p in enumerate(files):
        row = stat(p)
        if p in hash_files:
            row["sha256"] = sha(p)
        result[str(p.relative_to(SOURCE))] = row
        if i and i % 1000 == 0:
            print(json.dumps({"stage":"source_snapshot", "files_completed":i}), flush=True)
    return result


def clinical_tables():
    data, fingerprints = {}, {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in ("patients.csv.gz", "icustays.csv.gz"):
            member = next(i for i in z.infolist() if i.filename.endswith("/" + name))
            content = z.read(member)  # ZipFile checks member CRC on complete read.
            fingerprints[name] = {"archive_member":member.filename, "member_bytes":len(content),
                                  "zip_crc32":f"{member.CRC:08x}",
                                  "sha256_compressed_member":hashlib.sha256(content).hexdigest()}
            data[name] = pd.read_csv(io.BytesIO(content), compression="gzip")
    patients = data["patients.csv.gz"].set_index("subject_id")
    stays = data["icustays.csv.gz"]
    for c in ("intime", "outtime"):
        stays[c] = pd.to_datetime(stays[c])
    return patients, stays, fingerprints


def clinical_link(header, subject, hadm, patients, stays):
    base = header.base_datetime
    duration = header.sig_len / header.fs
    r = {"subject_found":bool(subject in patients.index), "hadm_has_icu_stay":False,
         "record_datetime_available":base is not None, "age_year_level":None,
         "adult":None, "temporal_icu_overlap_seconds":0., "temporal_icu_overlap_stays":0,
         "full_record_inside_icu":False, "icu_intervals_seconds":[],
         "nearest_icu_gap_seconds":None}
    if subject in patients.index and base is not None:
        p = patients.loc[subject]
        r["age_year_level"] = age_at_year(p.anchor_age, p.anchor_year, base.year)
        r["adult"] = bool(r["age_year_level"] >= 18)
    candidate = stays[(stays.subject_id == subject) & (stays.hadm_id == hadm)]
    r["hadm_has_icu_stay"] = bool(len(candidate))
    r["matched_icu_stay_ids"] = candidate.stay_id.astype(int).tolist()
    if base is None:
        return r
    gaps = []
    for row in candidate.itertuples():
        a, b = (row.intime.to_pydatetime()-base).total_seconds(), (row.outtime.to_pydatetime()-base).total_seconds()
        overlap = overlap_seconds(0, duration, a, b)
        gaps.append(max(a-duration, -b, 0.))
        if overlap > 0:
            r["temporal_icu_overlap_stays"] += 1
            r["icu_intervals_seconds"].append([max(0.,a),min(duration,b)])
    r["temporal_icu_overlap_seconds"] = union_seconds(r["icu_intervals_seconds"])
    r["full_record_inside_icu"] = bool(r["temporal_icu_overlap_seconds"] >= duration - 1e-5)
    r["nearest_icu_gap_seconds"] = min(gaps) if gaps else None
    return r


def union_seconds(intervals):
    total = 0.
    end = None
    for a,b in sorted(intervals):
        if end is None or a >= end:
            total += b-a
        elif b > end:
            total += b-end
        end = b if end is None else max(b,end)
    return total


def fully_in_icu_bins(n, intervals):
    a = np.arange(n)*10.
    mask = np.zeros(n, dtype=bool)
    for x,y in intervals:
        mask |= (a >= x) & (a+10 <= y)
    return mask


def annotate_intervals(counter, result):
    counter.update(result["positive_interval_counts_seconds"])


def waveform_audit(p, h):
    channels = defaultdict(list)
    pressure = defaultdict(list)
    rates = Counter()
    offset = 0.
    for s, frames in zip(h.segments, h.seg_len):
        if s is not None and s.sig_len > 0:
            for i,(name,unit) in enumerate(zip(s.sig_name,s.units)):
                if unit.lower() == "mmhg":
                    pressure[name].append([offset, offset+frames/h.fs])
                family = wave_alias(name)
                if family:
                    channels[family].append((s,i,offset,name))
                    rates[(name, float(s.fs*s.samps_per_frame[i]))] += 1
        offset += frames/h.fs
    blocks = []
    for family, choices in sorted(channels.items()):
        lengths = np.asarray([s.sig_len/s.fs for s,_,_,_ in choices])
        cumulative = np.cumsum(lengths)
        for q in (.1,.5,.9):
            target = q*cumulative[-1]
            k = int(np.searchsorted(cumulative,target,side="right"))
            s, i, off, name = choices[min(k,len(choices)-1)]
            relative = target - (cumulative[k-1] if k else 0)
            width = min(s.sig_len, max(1,int(round(60*s.fs))))
            start = int(np.clip(round((relative-30)*s.fs),0,s.sig_len-width))
            row = {"family":family,"signal_name":name,"quantile":q,
                   "segment":s.record_name,"record_start_seconds":off+start/s.fs,
                   "frame_fs":float(s.fs),"samples_per_frame":int(s.samps_per_frame[i]),
                   "sampling_hz":float(s.fs*s.samps_per_frame[i]),"units":s.units[i],
                   "decoded":False}
            try:
                w = wfdb.rdrecord(str(p/s.record_name),sampfrom=start,sampto=start+width,
                                  channels=[i],physical=True,smooth_frames=False)
                a = w.e_p_signal[0]
                row.update(wave_quality(a,row["sampling_hz"]))
                row["decoded"] = True
                row["sampled_seconds"] = len(a)/row["sampling_hz"]
                row["decoded_block_sha256_float64"] = hashlib.sha256(np.asarray(a,dtype="<f8").tobytes()).hexdigest()
            except Exception as ex:
                row["error_type"] = type(ex).__name__
                row["private_error"] = str(ex)
            blocks.append(row)
    intervals = [ab for lines in pressure.values() for ab in lines]
    arterial_intervals = [[off,off+s.sig_len/s.fs] for choices in channels.values() for s,_,off,_ in choices]
    return {"systemic_wave_families":sorted(channels), "pressure_channel_names":sorted(pressure),
            "pressure_channel_hours":{name:union_seconds(v)/3600 for name,v in pressure.items()},
            "systemic_wave_header_hours_union":union_seconds(arterial_intervals)/3600,
            "any_pressure_header_hours_union":union_seconds(intervals)/3600,
            "sampling_rate_segment_counts":[{"signal":n,"hz":f,"segments":v} for (n,f),v in sorted(rates.items())],
            "qc_blocks":blocks}


def numeric_audit(p, h, link, all_intervals, primary_intervals):
    f = p/(p.name+"n.csv.gz")
    n = int(h.sig_len/h.fs/10)
    duration = h.sig_len/h.fs
    result = {"numeric_exists":f.is_file(),"numeric_readable":False,"rows":0,
              "numeric_map_families":[],"primary_family":None,
              "unclassified_pressure_columns":[],"nbp_mean_present":False}
    empty = np.full(n,np.nan)
    if not f.is_file():
        return result,empty,empty
    frame = pd.read_csv(f)
    result.update(numeric_readable=True,rows=int(len(frame)))
    ticks = pd.to_numeric(frame["time"],errors="coerce").to_numpy(dtype=float)
    result["invalid_time_rows"] = int((~np.isfinite(ticks)).sum())
    result["all_rows_timestamp_audit"] = interval_summary(ticks,h.counter_freq)
    annotate_intervals(all_intervals,result["all_rows_timestamp_audit"])
    result["negative_time_rows"] = int((ticks < 0).sum())
    result["past_header_rows"] = int((ticks/h.counter_freq >= duration).sum())
    columns = defaultdict(dict)
    for c in frame.columns:
        alias = numeric_alias(c)
        if alias:
            fam,kind = alias
            if kind in columns[fam]:
                raise ValueError("Duplicate canonical pressure channel")
            columns[fam][kind] = c
        elif "[mmhg]" in c.lower() and not re.match(r"^(PAP|CVP|ICP|NBP|CO2|EtCO2|UVP)",c,re.I):
            result["unclassified_pressure_columns"].append(c)
    result["nbp_mean_present"] = "NBPm [mmHg]" in frame.columns
    bins = {}
    details = {}
    for fam, cols in sorted(columns.items()):
        if "m" not in cols:
            continue
        vals = pd.to_numeric(frame[cols["m"]],errors="coerce").to_numpy(dtype=float)
        b, raw = bin_medians(ticks,vals,h.counter_freq,duration)
        bins[fam] = b
        sb = np.full(n,np.nan)
        db = np.full(n,np.nan)
        if "s" in cols:
            sb,_ = bin_medians(ticks,pd.to_numeric(frame[cols["s"]],errors="coerce").to_numpy(dtype=float),h.counter_freq,duration)
        if "d" in cols:
            db,_ = bin_medians(ticks,pd.to_numeric(frame[cols["d"]],errors="coerce").to_numpy(dtype=float),h.counter_freq,duration)
        c_ok = coherent(b,sb,db)
        paired = map_valid(b) & np.isfinite(sb) & np.isfinite(db)
        stats = interval_summary(ticks[np.isfinite(vals)],h.counter_freq)
        details[fam] = {"column":cols["m"],"raw":raw,"coverage":coverage(b,duration),
                        "timestamps":stats,"sensitivity":coverage(b,duration,c_ok),
                        "triplet_assessable_bins":int(paired.sum()),
                        "triplet_failed_bins":int((paired & ~c_ok).sum()),
                        "sbp_dbp_columns_present":bool("s" in cols and "d" in cols)}
        details[fam]["numeric_flatline"] = numeric_flatline(b)
    result["aliases"] = details
    result["numeric_map_families"] = sorted(bins)
    result["multiple_map_families"] = len(bins)>1
    primary = choose_family(bins)
    result["primary_family"] = primary
    if primary is None:
        return result,empty,empty
    b = bins[primary]
    detail = details[primary]
    c = np.asarray(b).copy()
    # Sensitivity uses the same primary alias, never substitutes a different family.
    cols = columns[primary]
    if "s" in cols and "d" in cols:
        sb,_ = bin_medians(ticks,pd.to_numeric(frame[cols["s"]],errors="coerce").to_numpy(dtype=float),h.counter_freq,duration)
        db,_ = bin_medians(ticks,pd.to_numeric(frame[cols["d"]],errors="coerce").to_numpy(dtype=float),h.counter_freq,duration)
        c[~coherent(b,sb,db)] = np.nan
    else:
        c[:] = np.nan
    result["primary_coverage"] = detail["coverage"]
    result["primary_sensitivity"] = detail["sensitivity"]
    result["primary_triplet_assessable_bins"] = detail["triplet_assessable_bins"]
    result["primary_triplet_failed_bins"] = detail["triplet_failed_bins"]
    result["primary_flatline"] = detail["numeric_flatline"]
    annotate_intervals(primary_intervals,detail["timestamps"])
    union_mask = np.any(np.vstack([map_valid(x) for x in bins.values()]),axis=0)
    result["alias_union_coverage_only_not_reference"] = coverage(np.where(union_mask,70.,np.nan),duration)
    icumask = fully_in_icu_bins(n,link["icu_intervals_seconds"])
    result["full_bins_inside_icu"] = int(icumask.sum())
    result["valid_bins_inside_icu"] = int((icumask & map_valid(b)).sum())
    result["icu_only_coverage"] = coverage(np.where(icumask,b,np.nan),duration)
    # Require the entire 360-bin interval inside ICU, not merely the valid samples.
    result["icu_fully_contained_hour_eligible"] = False
    result["icu_best_hour_start_bin"] = None
    result["icu_eligible_sliding_hour_windows"] = 0
    if n >= 360:
        ci = np.r_[0,np.cumsum(icumask)]
        cv = np.r_[0,np.cumsum(icumask & map_valid(b))]
        full = (ci[360:]-ci[:-360]) == 360
        counts = cv[360:]-cv[:-360]
        qualifying = full & (counts>=288)
        result["icu_fully_contained_hour_eligible"] = bool(qualifying.any())
        result["icu_eligible_sliding_hour_windows"] = int(qualifying.sum())
        if qualifying.any():
            masked = np.where(full,counts,-1)
            result["icu_best_hour_start_bin"] = int(np.argmax(masked))
    result["low_map_feasibility_all"] = low_runs(b)
    start = detail["coverage"]["best_hour_start_bin"]
    if detail["coverage"]["window_eligible"]:
        result["low_map_feasibility_best_eligible_hour"] = low_runs(b[start:start+360])
    start_icu = result["icu_best_hour_start_bin"]
    if start_icu is not None:
        result["low_map_feasibility_best_icu_hour"] = low_runs(b[start_icu:start_icu+360])
    return result,b,c


def numeric_flatline(b):
    z = np.asarray(b,dtype=float)
    same = np.isfinite(z[1:]) & np.isfinite(z[:-1]) & (np.abs(np.diff(z)) <= .01)
    starts,ends = mask_runs(same)
    longest = max(ends-starts+1,default=0)*10
    return {"longest_unchanged_bin_seconds":int(longest),"unchanged_ge5min_flag":bool(longest>=300),
            "meaning":"repeated numeric display; not proof of physical transducer flatline"}


def quantiles(a):
    a = np.asarray(a,dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n":0}
    return {"n":int(len(a)), **{k:float(v) for k,v in zip(("min","p01","p05","p25","p50","p75","p95","p99","max"),np.quantile(a,[0,.01,.05,.25,.5,.75,.95,.99,1]))}}


def summarize_intervals(counts):
    items = sorted((float(k),v) for k,v in counts.items())
    n = sum(v for _,v in items)
    if not n:
        return {"n":0}
    cum = np.cumsum([v for _,v in items])
    vals = [v for v,_ in items]
    q = {f"p{int(p*100):02}":vals[min(int(np.searchsorted(cum,max(1,p*n))),len(vals)-1)] for p in (.01,.05,.25,.5,.75,.95,.99)}
    return {"n":int(n),"distinct_intervals":len(items),"min":vals[0],"max":vals[-1],**q,
            "fraction_le2s":sum(v for x,v in items if x<=2)/n,
            "fraction_le10s":sum(v for x,v in items if x<=10)/n,
            "fraction_ge60s":sum(v for x,v in items if x>=60)/n,
            "top_intervals":[{"seconds":float(k),"count":int(v)} for k,v in counts.most_common(12)]}


def aggregate(rows, arrays, sensitivity, all_intervals, primary_intervals):
    nums = [r for r in rows if r["numeric"].get("primary_family")]
    def subjects(rs):
        return len({r["subject_id"] for r in rs})
    def summ(rs, mode="all"):
        rs = list(rs)
        bins_count = 0
        low = Counter()
        low_records = 0
        union_by_subject = defaultdict(list)
        for r in rs:
            b = arrays[r["record_id"]]
            if mode == "best_hour":
                start = r["numeric"]["primary_coverage"]["best_hour_start_bin"]
                b = b[start:start+360]
                off = start*10
            elif mode == "icu_hour":
                start = r["numeric"]["icu_best_hour_start_bin"]
                b = b[start:start+360]
                off = start*10
            else:
                off = 0
            ok = map_valid(b)
            bins_count += int(ok.sum())
            l = low_runs(b)
            low.update({k:v for k,v in l.items() if k != "longest_run_seconds"})
            low_records += l["runs_ge60s"] > 0
            if r["base_datetime"]:
                base = dt.datetime.fromisoformat(r["base_datetime"]).timestamp()+off
                ss,ee = mask_runs(ok)
                union_by_subject[r["subject_id"]].extend((base+s*10,base+e*10) for s,e in zip(ss,ee))
        return {"records":len(rs),"unique_subjects":subjects(rs),
                "header_record_hours":sum(r["duration_seconds"] for r in rs)/3600,
                "selected_grid_hours":len(rs) if mode!="all" else sum(len(arrays[r["record_id"]]) for r in rs)/360,
                "valid_bin_hours":bins_count/360,
                "subject_deduplicated_valid_bin_hours":sum(union_seconds(v) for v in union_by_subject.values())/3600,
                "records_with_observed_low_run_ge60s":int(low_records),
                "low_map_availability_counts":dict(low)}
    whole = [r for r in nums if r["numeric"]["primary_coverage"]["whole_eligible"]]
    window = [r for r in nums if r["numeric"]["primary_coverage"]["window_eligible"]]
    adult = [r for r in nums if r["clinical"]["adult"] is True]
    icu = [r for r in nums if r["numeric"]["icu_fully_contained_hour_eligible"]]
    adult_icu = [r for r in icu if r["clinical"]["adult"] is True]
    cohorts = {"all_numeric_systemic_map":summ(nums),"whole_record_eligible":summ(whole),
               "at_least_one_eligible_hour_records":summ(window),
               "best_eligible_hour_only":summ(window,"best_hour"),
               "adult_numeric_systemic_map":summ(adult),
               "adult_eligible_hour_records":summ(r for r in window if r["clinical"]["adult"] is True),
               "fully_icu_contained_eligible_hour_records":summ(icu),
               "adult_fully_icu_contained_eligible_hour_records":summ(adult_icu),
               "adult_best_icu_hour_only":summ(adult_icu,"icu_hour"),
               "adult_whole_record_eligible_and_full_icu":summ(r for r in whole if r["clinical"]["adult"] is True and r["clinical"]["full_record_inside_icu"])}
    aliases = []
    for fam in sorted({f for r in nums for f in r["numeric"]["aliases"]},key=family_rank):
        rs = [r for r in nums if fam in r["numeric"]["aliases"]]
        ds = [r["numeric"]["aliases"][fam] for r in rs]
        aliases.append({"family":fam,"records_with_column":len(rs),"unique_subjects":subjects(rs),
                        "records_with_finite_samples":sum(d["raw"]["finite_samples"]>0 for d in ds),
                        "raw_finite_samples":sum(d["raw"]["finite_samples"] for d in ds),
                        "raw_outside_20_180_samples":sum(d["raw"]["raw_map_outside_20_180_samples"] for d in ds),
                        "negative_time_samples":sum(d["raw"]["negative_time_samples"] for d in ds),
                        "outside_header_samples":sum(d["raw"]["outside_header_samples"] for d in ds),
                        "valid_bins":sum(d["coverage"]["valid_bins"] for d in ds),
                        "valid_bin_hours":sum(d["coverage"]["valid_bin_hours"] for d in ds),
                        "whole_record_eligible":sum(d["coverage"]["whole_eligible"] for d in ds),
                        "at_least_one_hour_eligible":sum(d["coverage"]["window_eligible"] for d in ds),
                        "selected_primary_records":sum(r["numeric"]["primary_family"]==fam for r in nums)})
    blocks = [b for r in rows for b in r["waveform"]["qc_blocks"]]
    dec = [b for b in blocks if b["decoded"]]
    qcrecs = [r for r in rows if r["waveform"]["qc_blocks"]]
    plausible = [r for r in qcrecs if any(b.get("automated_plausible_block",False) for b in r["waveform"]["qc_blocks"])]
    names = sorted({n for r in rows for n in r["waveform"]["pressure_channel_names"]})
    excluded = {"PAP","CVP","ICP","CO2","UVP"}
    channels = [{"name":n,"classification":"systemic" if wave_alias(n) else "explicitly_excluded" if n.upper() in excluded else "unclassified_not_assumed_systemic",
                 "records":sum(n in r["waveform"]["pressure_channel_names"] for r in rows),
                 "unique_subjects":subjects(r for r in rows if n in r["waveform"]["pressure_channel_names"]),
                 "header_channel_hours":sum(r["waveform"]["pressure_channel_hours"].get(n,0) for r in rows)} for n in names]
    rates = Counter()
    for r in rows:
        for v in r["waveform"]["sampling_rate_segment_counts"]:
            rates[(v["signal"],v["hz"])] += v["segments"]
    ages = [r["clinical"]["age_year_level"] for r in rows if r["clinical"]["age_year_level"] is not None]
    out = {"status":"completed_local_read_only_feasibility_audit_not_validation",
           "inventory":{"patient_directories":198,"records":len(rows),"unique_subjects":subjects(rows),
                        "numeric_files":sum(r["numeric"]["numeric_exists"] for r in rows),
                        "numeric_readable":sum(r["numeric"]["numeric_readable"] for r in rows),
                        "all_record_header_hours":sum(r["duration_seconds"] for r in rows)/3600,
                        "systemic_numeric_map_records":len(nums),"systemic_numeric_map_unique_subjects":subjects(nums),
                        "multi_map_alias_records":sum(r["numeric"].get("multiple_map_families",False) for r in rows),
                        "nbp_mean_column_records":sum(r["numeric"]["nbp_mean_present"] for r in rows)},
           "cohorts":cohorts,"numeric_aliases":aliases,
           "primary_coverage":{"full_bins":sum(len(a) for a in arrays.values()),
                               "valid_bins":sum(int(map_valid(a).sum()) for a in arrays.values()),
                               "missing_bins":sum(int((~np.isfinite(a)).sum()) for a in arrays.values()),
                               "invalid_present_bins":sum(int((np.isfinite(a) & ~map_valid(a)).sum()) for a in arrays.values()),
                               "valid_map_values_mmHg":quantiles(np.concatenate([a[map_valid(a)] for a in arrays.values()])),
                               "whole_record_valid_fraction_distribution":quantiles([r["numeric"]["primary_coverage"]["valid_fraction"] for r in nums]),
                               "best_hour_valid_fraction_distribution":quantiles([r["numeric"]["primary_coverage"]["best_hour_valid_fraction"] for r in nums]),
                               "eligible_nonoverlapping_record_start_hour_windows":sum(r["numeric"]["primary_coverage"]["eligible_nonoverlap_hour_windows"] for r in nums),
                               "primary_records_numeric_unchanged_ge5min_flag":sum(r["numeric"]["primary_flatline"]["unchanged_ge5min_flag"] for r in nums)},
           "coherence_sensitivity":{"rule":"MAP20-180 and 20<=DBP<=MAP<=SBP<=250 and PP10-150; automated, not adjudicated",
                                    "assessable_primary_bins":sum(r["numeric"]["primary_triplet_assessable_bins"] for r in nums),
                                    "failed_assessable_bins":sum(r["numeric"]["primary_triplet_failed_bins"] for r in nums),
                                    "valid_bins":sum(int(map_valid(a).sum()) for a in sensitivity.values()),
                                    "valid_bin_hours":sum(int(map_valid(a).sum()) for a in sensitivity.values())/360,
                                    "whole_record_eligible_records":sum(r["numeric"]["primary_sensitivity"]["whole_eligible"] for r in nums),
                                    "one_hour_eligible_records":sum(r["numeric"]["primary_sensitivity"]["window_eligible"] for r in nums),
                                    "one_hour_eligible_unique_subjects":subjects(r for r in nums if r["numeric"]["primary_sensitivity"]["window_eligible"])},
           "clinical_linkage":{"tables_read":["patients.csv.gz","icustays.csv.gz"],
                               "subject_matched_records":sum(r["clinical"]["subject_found"] for r in rows),
                               "subject_matched_unique_subjects":subjects(r for r in rows if r["clinical"]["subject_found"]),
                               "same_subject_hadm_has_icu_records":sum(r["clinical"]["hadm_has_icu_stay"] for r in rows),
                               "temporal_icu_overlap_records":sum(r["clinical"]["temporal_icu_overlap_seconds"]>0 for r in rows),
                               "temporal_icu_overlap_unique_subjects":subjects(r for r in rows if r["clinical"]["temporal_icu_overlap_seconds"]>0),
                               "full_record_inside_icu_records":sum(r["clinical"]["full_record_inside_icu"] for r in rows),
                               "summed_icu_overlap_hours":sum(r["clinical"]["temporal_icu_overlap_seconds"] for r in rows)/3600,
                               "adult_records":sum(r["clinical"]["adult"] is True for r in rows),
                               "adult_unique_subjects":subjects(r for r in rows if r["clinical"]["adult"] is True),
                               "age_unknown_records":sum(r["clinical"]["adult"] is None for r in rows),
                               "age_year_level_distribution_record_weighted":quantiles(ages),
                               "clock_nonoverlap_is_not_proven_nonlinkage":True,
                               "operating_room_time_proven":False},
           "timestamp_intervals":{"all_numeric_row_timestamps_seconds":summarize_intervals(all_intervals),
                                  "primary_map_present_sample_timestamps_seconds":summarize_intervals(primary_intervals),
                                  "records_with_backward_row_steps":sum(r["numeric"].get("all_rows_timestamp_audit",{}).get("backward_steps_file_order",0)>0 for r in rows),
                                  "negative_time_rows":sum(r["numeric"].get("negative_time_rows",0) for r in rows),
                                  "past_header_rows":sum(r["numeric"].get("past_header_rows",0) for r in rows)},
           "waveform_qc":{"scope":"systematic 10/50/90% of available arterial segment time, <=60s/block; automated only; no full-record certification",
                          "systemic_header_records":len(qcrecs),"systemic_header_unique_subjects":subjects(qcrecs),
                          "systemic_header_hours_union_within_record":sum(r["waveform"]["systemic_wave_header_hours_union"] for r in rows),
                          "attempted_blocks":len(blocks),"decoded_blocks":len(dec),"failed_blocks":len(blocks)-len(dec),
                          "decoded_records":sum(any(b["decoded"] for b in r["waveform"]["qc_blocks"]) for r in qcrecs),
                          "sampled_hours":sum(b["sampled_seconds"] for b in dec)/3600,
                          "decoded_samples":sum(b["samples"] for b in dec),
                          "finite_samples":sum(b["finite_samples"] for b in dec),
                          "automated_plausible_blocks":sum(b["automated_plausible_block"] for b in dec),
                          "records_with_any_automated_plausible_block":len(plausible),
                          "records_all_attempted_blocks_automated_plausible":sum(all(b.get("automated_plausible_block",False) for b in r["waveform"]["qc_blocks"]) for r in qcrecs),
                          "flatline_ge5s_blocks":sum(b["flatline_ge5s"] for b in dec),
                          "amplitude_lt80percent_blocks":sum(b["amplitude_20_250_fraction"]<.8 for b in dec),
                          "finite_lt80percent_blocks":sum(b["finite_fraction"]<.8 for b in dec),
                          "pressure_channel_inventory":channels,
                          "sampling_hz_segment_inventory":[{"signal":n,"hz":f,"segments":v} for (n,f),v in sorted(rates.items())]},
           "boundaries":{"no_interpolation":True,"no_reference_locf":True,
                         "no_selection_on_low_MAP":True,"no_hypothesis_test":True,
                         "no_cuff_freshness_claim":True,"no_intraoperative_validation_claim":True,
                         "no_manuscript_access_or_edit":True,
                         "user_supplied_manuscript_hash_not_independently_verified":MAIN_BASELINE_USER_SUPPLIED}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-only",action="store_true",help="Skip waveform decode for diagnostics; not final audit")
    args = parser.parse_args()
    start = dt.datetime.now(dt.timezone.utc)
    print(json.dumps({"stage":"started","utc":start.isoformat()}),flush=True)
    pdirs, paths, files = inventory()
    hash_files = {p for p in files if p.suffix in (".hea",".gz") or p.parent == SOURCE}
    before = source_snapshot(files,hash_files)
    save_json(PRIVATE/"source_inventory_before.json",before)
    archive_before = stat(ARCHIVE)
    patients, stays, clinical_fingerprints = clinical_tables()
    rows, arrays, sensitivity = [], {}, {}
    all_intervals, primary_intervals = Counter(),Counter()
    for index,p in enumerate(paths,1):
        h = wfdb.rdheader(str(p/p.name),rd_segments=True)
        comments = "\n".join(h.comments)
        subject = int(re.search(r"\bsubject_id\s+(\d+)",comments)[1])
        hm = re.search(r"\bhadm_id\s+(\d+)",comments)
        hadm = int(hm[1]) if hm else None
        link = clinical_link(h,subject,hadm,patients,stays)
        num,b,c = numeric_audit(p,h,link,all_intervals,primary_intervals)
        wave = waveform_audit(p,h) if not args.numeric_only else {"qc_blocks":[],"pressure_channel_names":[],"pressure_channel_hours":{},"systemic_wave_header_hours_union":0,"sampling_rate_segment_counts":[]}
        row = {"record_id":p.name,"subject_id":subject,"hadm_id":hadm,
               "source_relative_record":str(p.relative_to(SOURCE)),
               "duration_seconds":h.sig_len/h.fs,"frame_hz":float(h.fs),"counter_hz":float(h.counter_freq),
               "base_datetime":h.base_datetime.isoformat() if h.base_datetime else None,
               "clinical":link,"numeric":num,"waveform":wave}
        rows.append(row)
        if num.get("primary_family"):
            arrays[p.name] = b
            sensitivity[p.name] = c
        if index%10 == 0 or index==len(paths):
            save_json(PRIVATE/"record_audit.json",rows)
            print(json.dumps({"stage":"records","completed":index,"total":len(paths),
                              "map_records_so_far":len(arrays),
                              "primary_eligible_hour_records_so_far":sum(r["numeric"].get("primary_coverage",{}).get("window_eligible",False) for r in rows),
                              "wave_blocks_decoded_so_far":sum(b["decoded"] for r in rows for b in r["waveform"]["qc_blocks"])}),flush=True)
    np.savez_compressed(PRIVATE/"primary_and_sensitivity_bins.npz",**{k:v for rid,a in arrays.items() for k,v in (("primary_"+rid,a),("sensitivity_"+rid,sensitivity[rid]))})
    result = aggregate(rows,arrays,sensitivity,all_intervals,primary_intervals)
    result["inventory"]["patient_directories"] = len(pdirs)
    result["run"] = {"started_utc":start.isoformat(),"wave_decode_skipped":args.numeric_only,
                     "python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"wfdb":wfdb.__version__,"platform":platform.platform()}
    save_json(OUT/"aggregate.json",result)
    print(json.dumps({"stage":"first_complete_aggregate","inventory":result["inventory"],
                      "primary_usable_hour":result["cohorts"]["best_eligible_hour_only"],
                      "adult_icu_usable_hour":result["cohorts"]["adult_best_icu_hour_only"],
                      "clinical_linkage":result["clinical_linkage"],
                      "wave_qc":{"attempted":result["waveform_qc"]["attempted_blocks"],"decoded":result["waveform_qc"]["decoded_blocks"]}}),flush=True)
    save_csv(OUT/"aggregate_cohorts.csv",[{"cohort":k,**{kk:vv for kk,vv in v.items() if not isinstance(vv,dict)}} for k,v in result["cohorts"].items()])
    save_csv(OUT/"aggregate_numeric_aliases.csv",result["numeric_aliases"])
    save_csv(OUT/"aggregate_pressure_channels.csv",result["waveform_qc"]["pressure_channel_inventory"])
    save_csv(OUT/"aggregate_wave_sampling_rates.csv",result["waveform_qc"]["sampling_hz_segment_inventory"])
    save_csv(OUT/"aggregate_timestamp_intervals.csv",[{"stream":label,"seconds":float(k),"intervals":v} for label,co in (("all_rows",all_intervals),("primary_MAP_present_samples",primary_intervals)) for k,v in sorted(co.items(),key=lambda item:float(item[0]))])
    save_csv(PRIVATE/"record_audit.csv",[{"record_id":r["record_id"],"subject_id":r["subject_id"],"hadm_id":r["hadm_id"],"duration_hours":r["duration_seconds"]/3600,
                                        "adult":r["clinical"]["adult"],"icu_overlap_hours":r["clinical"]["temporal_icu_overlap_seconds"]/3600,
                                        "full_record_in_icu":r["clinical"]["full_record_inside_icu"],"primary_family":r["numeric"].get("primary_family"),
                                        **r["numeric"].get("primary_coverage",{}),
                                        "icu_usable_hour":r["numeric"].get("icu_fully_contained_hour_eligible",False)} for r in rows])
    save_csv(PRIVATE/"waveform_qc_blocks.csv",[{"record_id":r["record_id"],"subject_id":r["subject_id"],**b} for r in rows for b in r["waveform"]["qc_blocks"]])
    after = source_snapshot(files,hash_files)
    save_json(PRIVATE/"source_inventory_after.json",after)
    # Re-enumeration also detects additions/deletions in scope.
    _,_,files_after = inventory()
    same_set = set(files) == set(files_after)
    changes = [name for name in before if before[name] != after.get(name)]
    supplied = {}
    for line in (SOURCE/"SHA256SUMS.txt").read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts)==2:
            supplied[parts[1].lstrip(" *")] = parts[0]
    compared = [name for name,x in before.items() if "sha256" in x and name in supplied]
    mismatch = [name for name in compared if before[name]["sha256"] != supplied[name]]
    # Re-read only the two small clinical members, never scan large event tables.
    _,_,clinical_after = clinical_tables()
    checked_digest = hashlib.sha256(json.dumps({name:before[name]["sha256"] for name in sorted(before) if "sha256" in before[name]},sort_keys=True).encode()).hexdigest()
    status = {"waveform_source_root":str(SOURCE),"clinical_archive":str(ARCHIVE),
              "source_file_count_stat_checked":len(files),"source_bytes_stat_checked":sum(x["size"] for x in before.values()),
              "source_file_set_unchanged":same_set,"source_stat_or_selected_hash_changed_files":len(changes),
              "hashed_header_numeric_metadata_files":len(hash_files),
              "hash_checked_file_bytes":sum(before[str(p.relative_to(SOURCE))]["size"] for p in hash_files),
              "hashes_compared_to_supplied_manifest":len(compared),"manifest_mismatches":len(mismatch),
              "selected_source_fingerprint_sha256":checked_digest,
              "root_RECORDS_sha256":sha(SOURCE/"RECORDS"),"root_SHA256SUMS_sha256":sha(SOURCE/"SHA256SUMS.txt"),
              "all_waveform_DAT_files_full_hashed":False,"clinical_archive_full_hashed":False,
              "clinical_archive_stat_unchanged":archive_before == stat(ARCHIVE),
              "clinical_archive_stat":archive_before,"clinical_members":clinical_fingerprints,
              "clinical_members_hash_unchanged":clinical_fingerprints == clinical_after,
              "scope_status":"stat_all_and_SHA256_headers_numerics_metadata_and_clinical_members_only",
              "manuscript_baseline_user_supplied_not_accessed":MAIN_BASELINE_USER_SUPPLIED,
              "no_publication_upload_or_manuscript_edit":True,
              "finished_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
    save_json(OUT/"source_status.json",status)
    result["source_status"] = status
    result["run"]["finished_utc"] = status["finished_utc"]
    result["run"]["elapsed_seconds"] = (dt.datetime.now(dt.timezone.utc)-start).total_seconds()
    save_json(OUT/"aggregate.json",result)
    save_json(PRIVATE/"source_changes_or_mismatch.json",{"changed_files":changes,"manifest_mismatches":mismatch})
    print(json.dumps({"stage":"completed","source_unchanged_within_checks":not changes and same_set,
                      "manifest_mismatches":len(mismatch),"elapsed_seconds":result["run"]["elapsed_seconds"]}),flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (PRIVATE/"last_error.log").write_text(traceback.format_exc())
        print(json.dumps({"stage":"failed","detail":"private/last_error.log"}),flush=True)
        sys.exit(1)
