"""Frozen cohorts and resumable read-only released-tool benchmarking (Python 3.6+)."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
VERSION = "published_amp_six_by_six_v1.2"
DATASETS = ["AMPlify", "tsAMP_I", "c_AMPs_benchmark_resplit", "HMD_AMP", "AmPEP_raw_source", "ampir_raw_mature90"]
TOOLS = ["Macrel", "AMPlify", "ampir", "tsAMP_I", "HMD_AMP", "c_AMPs"]
COMPONENTS = ["Macrel", "AMPlify", "ampir", "tsAMP_I", "HMD_AMP", "camps_keras", "camps_bert"]
EXPECTED_TEST = dict(zip(DATASETS, [(1534,700,834), (15951,6284,9667), (5968,92,5876),
                                   (1110,512,598), (16810,265,16545), (583,271,312)]))
FIELDS = ["sequence_id", "sequence", "group_id", "split", "label"]
NATIVE_FIELDS = FIELDS + ["probability", "prediction"]
FINAL_FIELDS = FIELDS + ["score", "prediction", "score_definition"]


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(str(path), encoding="utf-8-sig") as f:
        return json.load(f)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write("\n")
    os.replace(str(temp), str(path))


def read_tsv(path):
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(str(temp), str(path))


def verify_files(root, files):
    root = Path(root).resolve()
    require(bool(files), "Empty checksum manifest")
    for name, expected in files.items():
        path = (root / name).resolve()
        path.relative_to(root)
        require(sha(path) == expected, "Artifact checksum mismatch: " + str(path))


def package_checks():
    files = {}
    for line in (HERE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        files[name] = digest
    verify_files(HERE, files)
    return files


def support(tool, row):
    n = len(row["sequence"])
    if tool == "AMPlify":
        return 2 <= n <= 200
    if tool in ("Macrel", "ampir"):
        return n >= 10
    if tool in ("tsAMP_I", "HMD_AMP"):
        return 1 <= n <= 1022
    if tool == "c_AMPs":
        # Full stored BERT input includes CLS+SEP. Do not silently truncate.
        return 1 <= n <= 126
    raise ValueError(tool)


def load_cohorts(plan, expected=EXPECTED_TEST):
    root = Path(plan["source_work"]).resolve()
    manifest = root / "dataset_complete.json"
    require(sha(manifest) == plan["source_manifest_sha256"], "Source dataset manifest differs from completed six-source experiment")
    stamp = read_json(manifest)
    require(stamp["version"] == "external_plm_holdout_v2.2", "Wrong frozen source version")
    verify_files(root, stamp["files"])
    cohorts = {}
    for dataset in DATASETS:
        all_rows = read_tsv(root / "04_datasets" / dataset / "all.tsv")
        ids, sequences = set(), set()
        for r in all_rows:
            seq = r["sequence"]
            require(seq and set(seq) <= set("ACDEFGHIKLMNPQRSTVWY"), "Invalid source sequence")
            require(r["sequence_id"] == "s_" + hashlib.sha256(seq.encode("ascii")).hexdigest(), "Source sequence hash mismatch")
            require(int(r["label"]) in (0,1) and r["split"] in ("train","validation","test"), "Bad source label/split")
            require(int(r["length"]) == len(seq) and bool(r["group_id"]), "Bad length/group metadata")
            require(r["sequence_id"] not in ids and seq not in sequences, "Duplicate source sequences")
            ids.add(r["sequence_id"]); sequences.add(seq)
        for split in ("train", "validation", "test"):
            require(read_tsv(root / "04_datasets" / dataset / (split + ".tsv")) ==
                    [r for r in all_rows if r["split"] == split], "Frozen all/split mismatch: " + dataset)
        rows = [{k: r[k] for k in FIELDS} for r in all_rows if r["split"] == "test"]
        counts = (len(rows), sum(int(r["label"]) == 1 for r in rows), sum(int(r["label"]) == 0 for r in rows))
        require(counts == expected[dataset], "Unexpected original TEST counts: " + dataset)
        cohorts[dataset] = rows
    return cohorts


def smoke_checks(plan):
    root = Path(plan["camps_smoke"]).resolve()
    stamp = read_json(root / "SMOKE_COMPLETE.json")
    require(stamp["version"] == "camps_recovery_smoke_v1.0" and stamp["smoke_complete"], "c_AMPs CPU smoke did not pass")
    verify_files(root, stamp["files"])
    for name in ("lstm", "attention", "bert"):
        rows = read_json(root / (name + ".json"))["rows"]
        require(len(rows) == 8, "Wrong c_AMPs smoke coverage")
    return sha(root / "SMOKE_COMPLETE.json")


def make_unions(cohorts):
    unions = {}
    coverage = []
    for tool in TOOLS:
        by_id = {}
        for dataset in DATASETS:
            selected = [r for r in cohorts[dataset] if support(tool, r)]
            coverage.append({"dataset": dataset, "tool": tool, "original_test_n": len(cohorts[dataset]),
                             "supported_n": len(selected), "positive": sum(int(r["label"]) == 1 for r in selected),
                             "negative": sum(int(r["label"]) == 0 for r in selected),
                             "excluded_n": len(cohorts[dataset])-len(selected)})
            for r in selected:
                if r["sequence_id"] in by_id:
                    require(by_id[r["sequence_id"]] == r, "Cross-source label/group conflict: " + r["sequence_id"])
                by_id[r["sequence_id"]] = r
        unions[tool] = [by_id[k] for k in sorted(by_id)]
    return unions, coverage


def prepared_checks(work):
    stamp = read_json(Path(work) / "prepared.json")
    verify_files(work, stamp["files"])
    return stamp


def prepare(plan, cohorts):
    work = Path(plan["work"]).resolve()
    source = Path(plan["source_work"]).resolve()
    require(work != source and work not in source.parents and source not in work.parents, "Unsafe output/source nesting")
    inputs = {"version": VERSION, "plan": plan, "code": package_checks(), "smoke_complete_sha256": smoke_checks(plan)}
    if (work / "prepared.json").exists():
        require(prepared_checks(work)["inputs"] == inputs, "Frozen plan/source/code changed; use new version")
        return
    require(not work.exists() or not any(p for p in work.iterdir() if p.name == "prepared.json" or not p.name.startswith(".")),
            "Nonempty unfinished destination; preserve it and use a new directory")
    work.mkdir(parents=True, exist_ok=True)
    unions, coverage = make_unions(cohorts)
    files = []
    for dataset in DATASETS:
        path = work / "01_inputs" / dataset / "test.tsv"
        write_tsv(path, cohorts[dataset], FIELDS)
        files.append(path)
    jobs = []
    for component in COMPONENTS:
        tool = "c_AMPs" if component.startswith("camps_") else component
        rows = unions[tool]
        query = work / "02_union" / component / "queries.tsv"
        write_tsv(query, rows, FIELDS)
        files.append(query)
        size = plan["chunk_sizes"][component]
        for offset in range(0, len(rows), size):
            selected = rows[offset:offset+size]
            key = component + "__{:06d}".format(offset // size)
            path = work / "02_union" / component / (key + ".tsv")
            write_tsv(path, selected, FIELDS); files.append(path)
            jobs.append({"key": key, "component": component, "query": str(path),
                         "query_sha256": sha(path), "n": len(selected)})
    write_tsv(work / "01_inputs/coverage.tsv", coverage, list(coverage[0]))
    files.append(work / "01_inputs/coverage.tsv")
    protocol = {
        "version": VERSION, "datasets": DATASETS, "tools": TOOLS, "expected_cells": 36,
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "source_policy": "byte-identical completed v2.2 TEST; no re-splitting, no training, no source/cache/model writes",
        "ampir_dataset": "raw_mature90 fixed resplit; NOT separately recovered official v1.1 train/test",
        "deduplication": "union of TEST queries per tool; shared sequence scored once then mapped back to each dataset",
        "support": {"AMPlify": "2..200", "Macrel": ">=10", "ampir": ">=10",
                    "tsAMP_I": "1..1022", "HMD_AMP": "1..1022", "c_AMPs": "1..126 no BERT truncation"},
        "published_thresholds": "native decisions; ampir >=0.5 declared benchmark reference (not author universal cutoff)",
        "amplify_output_validation": "native ensemble and submodel TSV mean checked with absolute tolerance 5e-7 for FP32/serialization; native scores and decisions unchanged",
        "recovery_policy": "verified committed v1.1 chunks copied with provenance into new version; failed/uncommitted attempts never imported",
        "c_AMPs": "all 3 probabilities strictly >0.5; full submodel outputs retained; no native continuous ensemble score invented",
        "c_AMPs_extension": "min of 3 probabilities is a separately named experimental ranking score, NOT an author-declared probability",
        "panels": ["per-tool supported TEST", "same TEST IDs supported by all six tools", "common TEST restricted to 10..49 aa"],
        "ranking": "no hyperparameter/threshold selection using TEST, no accuracy claims for diagnostic controls",
        "limitations": ["not independent released-tool leaderboard; training/source overlap possible",
                        "no full homology-independence claim; negative-source and M-start bias retained",
                        "raw resplit c_AMPs/HMD/AmPEP/ampir are not official external held-out test sets",
                        "AMPlify and tsAMP original test memberships subject to prior cleaning",
                        "c_AMPs runtime support up to126 does not prove deployment validity; original short-peptide domain differs",
                        "ampir >=60 and Macrel >=100 are extrapolation panels, not mature-short-AMP validation",
                        "datasets share sequences; cells are not statistically independent; point estimates, no significance claim",
                        "c_AMPs CPU smoke is runtime sanity, not exact 2021-runtime parity proof"]
    }
    dump(work / "protocol.json", protocol); files.append(work / "protocol.json")
    dump(work / "jobs.json", jobs); files.append(work / "jobs.json")
    dump(work / "prepared.json", {"inputs": inputs, "files": {p.relative_to(work).as_posix(): sha(p) for p in files}})
    print("FROZEN SIX-DATASET / SIX-TOOL PREPARATION COMPLETE", flush=True)


def job_inputs(work, job):
    return {"version": VERSION, "job": job, "protocol_sha256": sha(Path(work) / "protocol.json"),
            "prepared_sha256": sha(Path(work) / "prepared.json")}


def completed_job(work, job):
    marker = Path(work) / "03_jobs" / job["key"] / "complete.json"
    if not marker.exists():
        return None
    stamp = read_json(marker)
    require(stamp["inputs"] == job_inputs(work, job), "Completed chunk belongs to a different protocol")
    attempt = (marker.parent / stamp["attempt"]).resolve()
    attempt.relative_to(marker.parent.resolve())
    verify_files(attempt, stamp["files"])
    return attempt


def commit_job(work, job, attempt):
    parent = Path(work) / "03_jobs" / job["key"]
    require(not (parent / "complete.json").exists(), "Do not overwrite committed chunk")
    files = {p.relative_to(attempt).as_posix(): sha(p) for p in attempt.rglob("*")
             if p.is_file() and "tmp" not in p.relative_to(attempt).parts and p.name != "console.log"}
    require("predictions.tsv" in files and "identity.json" in files, "Missing worker artifacts")
    dump(parent / "complete.json", {"inputs": job_inputs(work, job), "attempt": attempt.name, "files": files})


def align(records, expected, component):
    require(len(records) == len(expected), "Prediction coverage mismatch")
    index = {}
    for r in records:
        require(r["sequence_id"] not in index, "Duplicate prediction ID")
        index[r["sequence_id"]] = r
    require(set(index) == {r["sequence_id"] for r in expected}, "Missing/extra prediction IDs")
    result = []
    for r in expected:
        p = index[r["sequence_id"]]
        require(all(str(p[k]) == str(r[k]) for k in FIELDS), "ID-aligned metadata changed")
        names = ("lstm", "attention") if component == "camps_keras" else ("probability",)
        for name in names:
            val = float(p[name])
            require(math.isfinite(val) and 0 <= val <= 1, "Invalid submodel score")
        if component not in ("camps_keras", "camps_bert"):
            require(int(p["prediction"]) in (0,1), "Invalid native decision")
        result.append(p)
    return result


def collect_component(work, component, jobs):
    result, identity = {}, None
    for job in [j for j in jobs if j["component"] == component]:
        attempt = completed_job(work, job)
        require(attempt is not None, "Missing committed chunk: " + job["key"])
        current = read_json(attempt / "identity.json")
        if identity is None:
            identity = current
        require(current == identity, "Runtime/checkpoint drift between chunks: " + component)
        expected = read_tsv(job["query"])
        require(sha(job["query"]) == job["query_sha256"], "Query changed")
        rows = align(read_tsv(attempt / "predictions.tsv"), expected, component)
        for r in rows:
            require(r["sequence_id"] not in result, "Duplicated union result")
            result[r["sequence_id"]] = r
    expected = read_tsv(Path(work) / "02_union" / component / "queries.tsv")
    require(set(result) == {r["sequence_id"] for r in expected}, "Union coverage mismatch")
    return result
