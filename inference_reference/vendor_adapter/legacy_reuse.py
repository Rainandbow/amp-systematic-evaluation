"""Verified v1.1 committed predictions -> new v1.2 tree; never modify old results."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid
import core as c


@contextmanager
def old_guard(root):
    import fcntl
    with (Path(root)/".scheduler.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
        yield


def inventory(plan, check_model_files=False):
    spec = plan.get("recovery")
    if not spec:
        return [],[]
    root = Path(spec["work"]).resolve()
    target = Path(plan["work"]).resolve()
    c.require(root!=target and root not in target.parents and target not in root.parents,"Unsafe recovery nesting")
    package = Path(spec["package"])
    manifest = package/"SHA256SUMS.txt"
    c.require(c.sha(manifest)==spec["package_manifest_sha256"],"Old package manifest differs from pinned v1.1 release")
    code = dict((name,digest) for digest,name in
                [line.split("  ",1) for line in manifest.read_text().splitlines()])
    c.verify_files(package,code)
    with old_guard(root):
        status = c.read_json(root/"status.json")
        c.require(status["status"] in ("failed_or_interrupted","complete") and not status.get("active"),
                  "Previous benchmark is not safely stopped")
        prepared = c.read_json(root/"prepared.json")
        c.verify_files(root,prepared["files"])
        expected_plan = dict(plan)
        expected_plan.pop("recovery",None)
        expected_plan.update(version=spec["version"],work=spec["work"])
        c.require(prepared["inputs"]["plan"]==expected_plan,"Original plan differs; refuse unsafe reuse")
        c.require(prepared["inputs"]["code"]==code and prepared["inputs"]["version"]==spec["version"],
                  "Original preparation code/version mismatch")
        c.require(prepared["inputs"]["smoke_complete_sha256"]==c.smoke_checks(plan),"CPU smoke changed since v1.1")
        jobs = c.read_json(root/"jobs.json")
        c.require(len(jobs)==spec["expected_total_chunks"],"Unexpected original total chunk count")
        ready = []
        checked_models = {}
        for job in jobs:
            marker = root/"03_jobs"/job["key"]/"complete.json"
            if not marker.exists():
                continue  # Native TSV alone is NOT proof of committed inference.
            stamp = c.read_json(marker)
            expected_inputs = {"version":spec["version"],"job":job,
                "protocol_sha256":c.sha(root/"protocol.json"),"prepared_sha256":c.sha(root/"prepared.json")}
            c.require(stamp["inputs"]==expected_inputs,"Original committed chunk protocol mismatch")
            attempt = (marker.parent/stamp["attempt"]).resolve()
            attempt.relative_to(marker.parent.resolve())
            c.verify_files(attempt,stamp["files"])
            c.require(c.sha(job["query"])==job["query_sha256"],"Original chunk query changed")
            query = c.read_tsv(job["query"])
            c.require(len(query)==job["n"],"Original chunk length mismatch")
            c.align(c.read_tsv(attempt/"predictions.tsv"),query,job["component"])
            identity = c.read_json(attempt/"identity.json")
            if check_model_files:
                for path,digest in identity["files"].items():
                    if path not in checked_models:
                        checked_models[path] = c.sha(path)
                    c.require(checked_models[path]==digest,"Installed model changed; old predictions not reusable")
            ready.append({"job":job,"attempt":str(attempt),"stamp":stamp,
                          "marker_sha256":c.sha(marker),"marker":str(marker)})
        c.require(len(ready)>=spec["minimum_committed_chunks"],"Fewer verified old chunks than reported; inspect before recovery")
    return jobs,ready


def pending_gpu_required(plan):
    old_jobs,ready = inventory(plan)
    if not old_jobs:
        return True
    done = {entry["job"]["key"] for entry in ready}
    root = Path(plan["work"])
    if (root/"prepared.json").exists():
        c.prepared_checks(root)
        current = c.read_json(root/"jobs.json")
        done.update(job["key"] for job in current if c.completed_job(root,job) is not None)
    return any(job["component"] in ("tsAMP_I","HMD_AMP","camps_bert")
               and job["key"] not in done for job in old_jobs)


def adopt(plan, new_jobs):
    _,ready = inventory(plan,check_model_files=True)
    if not ready:
        return
    work = Path(plan["work"])
    new_index = {job["key"]:job for job in new_jobs}
    imported = 0
    for item in ready:
        old_job = item["job"]
        c.require(old_job["key"] in new_index,"Old chunk not in new query plan")
        job = new_index[old_job["key"]]
        c.require(all(job[k]==old_job[k] for k in ("key","component","query_sha256","n")),
                  "Cannot reuse changed queries/component/chunk size")
        if c.completed_job(work,job) is not None:
            continue
        source = Path(item["attempt"])
        attempt = work/"03_jobs"/job["key"]/("import_v1_1_"+uuid.uuid4().hex[:12])
        attempt.mkdir(parents=True)
        for name,digest in item["stamp"]["files"].items():
            src = (source/name).resolve(); src.relative_to(source.resolve())
            dst = (attempt/name).resolve(); dst.relative_to(attempt.resolve())
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(str(src),str(dst))
            c.require(c.sha(dst)==digest,"Imported file changed during copy")
        c.dump(attempt/"reuse_provenance.json",{
            "source_marker":item["marker"],"source_marker_sha256":item["marker_sha256"],
            "source_version":plan["recovery"]["version"],"source_inputs":item["stamp"]["inputs"],
            "native_probabilities_and_decisions_changed":False,
            "validation_policy":"v1.1 committed under stricter 2e-7 bound; valid under v1.2 5e-7 bound"})
        c.align(c.read_tsv(attempt/"predictions.tsv"),c.read_tsv(job["query"]),job["component"])
        c.commit_job(work,job,attempt)
        imported += 1
        print("REUSE VERIFIED",job["key"],flush=True)
    c.dump(work/"recovery_complete.json",{
        "version":c.VERSION,"source_work":plan["recovery"]["work"],
        "source_committed_chunks":len(ready),"newly_imported_chunks":imported,
        "original_results_modified":False,"failed_uncommitted_attempts_imported":False})
    print("RECOVERY COMPLETE:",len(ready),"old verified chunks available; newly copied",imported,flush=True)
