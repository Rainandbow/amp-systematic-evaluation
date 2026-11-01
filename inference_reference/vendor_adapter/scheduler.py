"""Four authorized GPUs plus three CPU lanes, committing isolated chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
import core as c


@contextmanager
def lock(work):
    import fcntl
    path = Path(work) / ".scheduler.lock"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def resources(plan, gpu_required=True):
    c.require(plan["version"] == c.VERSION and plan["gpu_indices"] == [4,5,6,7],"Unapproved GPU plan")
    affinity = sorted(os.sched_getaffinity(0))
    c.require(len(affinity)>=plan["cpu_budget"]==16,"Need approved 16-CPU affinity; nothing changed")
    available = int(re.search(r"^MemAvailable:\s+(\d+)\s+kB",Path("/proc/meminfo").read_text(),re.M)[1]) / 1024**2
    c.require(available>=plan["min_host_available_gib"],"Insufficient available host memory")
    if not gpu_required:
        for tool in plan["tools"].values():
            c.require(Path(tool["executable"]).is_file(),"Missing existing executable: "+tool["executable"])
        c.require(Path(plan["audit_python"]).is_file(),"Missing audit Python")
        return {"devices":[],"host_available_gib":available,"cpu_ids":affinity[:16],
                "gpu_required":False,"inherited_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
    data = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name,memory.free",
                                    "--format=csv,noheader,nounits"],text=True)
    devices = {int(row[0].strip()):{"index":int(row[0]),"uuid":row[1].strip(),"name":row[2].strip(),"free_mib":int(row[3])}
               for row in csv.reader(io.StringIO(data)) if len(row)==4}
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        selected = set()
        for token in inherited.split(","):
            token = token.strip()
            if token.isdigit() and int(token) in devices:
                selected.add(devices[int(token)]["uuid"])
            else:
                matching = [d["uuid"] for d in devices.values() if token and d["uuid"].startswith(token)]
                c.require(len(matching)==1,"Unrecognized/disabled inherited GPU mask; do not widen allocation")
                selected.update(matching)
        c.require({devices[i]["uuid"] for i in plan["gpu_indices"]}<=selected,"Requested GPUs exceed inherited allocation")
    processes = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader,nounits"],text=True)
    occupied = {r[0].strip() for r in csv.reader(io.StringIO(processes)) if len(r)==2}
    for i in plan["gpu_indices"]:
        d = devices[i]
        c.require(d["free_mib"]>=plan["min_gpu_free_mib"] and d["uuid"] not in occupied,
                  "GPU {} is occupied/not free enough; no process killed".format(i))
    for tool in plan["tools"].values():
        c.require(Path(tool["executable"]).is_file(),"Missing existing executable: "+tool["executable"])
    c.require(Path(plan["audit_python"]).is_file(),"Missing audit Python")
    return {"devices":[devices[i] for i in plan["gpu_indices"]],"host_available_gib":available,
            "cpu_ids":affinity[:16],"inherited_cuda_visible_devices":inherited}


def lanes(snapshot):
    gpu = {d["index"]:d["uuid"] for d in snapshot["devices"]}
    result = [
        {"name":"cpu_light","components":["Macrel","ampir"],"gpu":"","cpus":snapshot["cpu_ids"][0:2]},
        {"name":"cpu_amplify","components":["AMPlify"],"gpu":"","cpus":snapshot["cpu_ids"][2:4]},
        {"name":"cpu_camps_keras","components":["camps_keras"],"gpu":"","cpus":snapshot["cpu_ids"][4:6]},
    ]
    if not gpu:
        return result
    return result + [
        {"name":"gpu4_tsamp","components":["tsAMP_I"],"fallback":["HMD_AMP","camps_bert"],"gpu":gpu[4],"cpus":snapshot["cpu_ids"][6:8]},
        {"name":"gpu5_hmd","components":["HMD_AMP"],"fallback":["tsAMP_I","camps_bert"],"gpu":gpu[5],"cpus":snapshot["cpu_ids"][8:10]},
        {"name":"gpu6_camps_bert","components":["camps_bert"],"fallback":["tsAMP_I","HMD_AMP","camps_bert"],"gpu":gpu[6],"cpus":snapshot["cpu_ids"][10:12],"bert_mod":0},
        {"name":"gpu7_camps_bert","components":["camps_bert"],"fallback":["HMD_AMP","tsAMP_I","camps_bert"],"gpu":gpu[7],"cpus":snapshot["cpu_ids"][12:14],"bert_mod":1},
    ]


def jobs_for_lane(all_jobs,lane):
    return [j for j in all_jobs if j["component"] in lane["components"]
            and ("bert_mod" not in lane or int(j["key"].rsplit("__",1)[1])%2==lane["bert_mod"])]


def claim_job(all_jobs,lane,done,claimed):
    """Call under scheduler guard: primary queue, then authorized idle-GPU help."""
    candidates = jobs_for_lane(all_jobs,lane)
    for component in lane.get("fallback",[]):
        candidates += [j for j in all_jobs if j["component"]==component]
    for job in candidates:
        if job["key"] not in done and job["key"] not in claimed:
            claimed.add(job["key"])
            return job
    return None


def completed_experiment(work):
    marker = Path(work) / "experiment_complete.json"
    if not marker.exists():
        return False
    obj = c.read_json(marker)
    c.require(obj["version"]==c.VERSION and obj["complete"] and obj["completed_cells"]==36,"Invalid experiment completion")
    c.require(obj["prepared_sha256"]==c.sha(Path(work) / "prepared.json"),"Completion belongs to a different preparation")
    c.verify_files(work,obj["files"])
    for dataset in c.DATASETS:
        for tool in c.TOOLS:
            path = Path(work) / "04_predictions" / dataset / tool
            c.verify_files(path,c.read_json(path / "complete.json")["files"])
    return True


def run(plan_path, confirm):
    c.require(confirm,"Pass --confirm-resources only for your authorized GPU4-7/16CPU allocation")
    plan = c.read_json(plan_path)
    c.package_checks()
    cohorts = c.load_cohorts(plan)
    c.smoke_checks(plan)
    work = Path(plan["work"])
    with lock(work):
        c.prepare(plan,cohorts)
        if completed_experiment(work):
            print("SKIP VERIFIED COMPLETE 36/36 EXPERIMENT",flush=True)
            return
        jobs = c.read_json(work / "jobs.json")
        import legacy_reuse
        legacy_reuse.adopt(plan,jobs)
        done = {j["key"] for j in jobs if c.completed_job(work,j) is not None}
        gpu_required = any(j["component"] in ("tsAMP_I","HMD_AMP","camps_bert")
                           and j["key"] not in done for j in jobs)
        snapshot = resources(plan,gpu_required=gpu_required)
        c.dump(work / "resource_snapshot.json",snapshot)
        print("RESOURCE MODE:","GPU4-7 + CPU" if gpu_required else "CPU ONLY; existing GPU results reused",flush=True)
        claimed = set()
        stop = threading.Event(); guard = threading.Lock()
        state = {"version":c.VERSION,"status":"running","phase":"released-tool inference",
                 "expected_cells":36,"completed_cells":0,"total_chunks":len(jobs),"committed_chunks":len(done),
                 "active":{},"failed":{},"started_unix":time.time()}
        def update():
            state["committed_chunks"] = len(done)
            state["progress"] = {component:{"wanted":sum(j["n"] for j in jobs if j["component"]==component),
                "committed":sum(j["n"] for j in jobs if j["component"]==component and j["key"] in done)} for component in c.COMPONENTS}
            c.dump(work / "status.json",state)
        update()
        def lane_run(lane):
            while not stop.is_set():
                with guard:
                    job = claim_job(jobs,lane,done,claimed)
                if job is None:
                    return
                attempt = work / "03_jobs" / job["key"] / ("attempt_"+uuid.uuid4().hex[:12])
                attempt.mkdir(parents=True)
                component = job["component"]
                exe = plan["audit_python"] if component=="ampir" else plan["tools"][component]["executable"]
                cmd = [exe,"-B","-u",str(c.HERE / "worker.py"),"--plan",str(plan_path),"--key",job["key"],
                       "--attempt",str(attempt),"--cpus",",".join(map(str,lane["cpus"]))]
                if lane["gpu"]:
                    cmd += ["--gpu",lane["gpu"]]
                with guard:
                    state["active"][lane["name"]] = {"job":job["key"],"n":job["n"],"log":str(attempt / "console.log")}
                    update()
                    print("START LANE",lane["name"],job["key"],job["n"],flush=True)
                try:
                    env = dict(os.environ,CUDA_VISIBLE_DEVICES=lane["gpu"] or "-1",PYTHONDONTWRITEBYTECODE="1",
                               OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2")
                    with (attempt / "console.log").open("x",encoding="utf-8") as log:
                        rc = subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                    c.require(rc==0,"Worker failed exit={}; log={}".format(rc,attempt / "console.log"))
                    c.align(c.read_tsv(attempt / "predictions.tsv"),c.read_tsv(job["query"]),component)
                    c.commit_job(work,job,attempt)
                    with guard:
                        done.add(job["key"]); state["active"].pop(lane["name"],None); update()
                        print("COMMITTED",job["key"],"chunks",len(done),"/",len(jobs),flush=True)
                except BaseException as error:
                    with guard:
                        state["failed"][job["key"]] = str(error); state["active"].pop(lane["name"],None); update()
                    stop.set()
                    raise
        try:
            with ThreadPoolExecutor(max_workers=7) as pool:
                futures = [pool.submit(lane_run,lane) for lane in lanes(snapshot)]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop.set()  # Finish active chunks; never launch new work after interruption.
                    raise
            c.require(len(done)==len(jobs),"Not every prediction chunk completed")
            state["phase"]="same-cohort reports"; update()
            c.load_cohorts(plan)  # Recheck frozen source after inference.
            c.package_checks()
            import reports
            reports.summarize(plan,cohorts,jobs)
            c.require(completed_experiment(work),"Final 36-cell integrity check failed")
            state.update(status="complete",phase="complete",completed_cells=36,ended_unix=time.time()); update()
        except BaseException as error:
            state.update(status="failed_or_interrupted",error=repr(error),ended_unix=time.time()); update()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["check","run","status"])
    parser.add_argument("--plan",type=Path,default=c.HERE / "plan.server.json")
    parser.add_argument("--confirm-resources",action="store_true")
    args = parser.parse_args()
    plan = c.read_json(args.plan)
    if args.action == "status":
        path = Path(plan["work"]) / "status.json"
        print(json.dumps(c.read_json(path) if path.exists() else {"status":"not_started"},indent=2,ensure_ascii=False))
    elif args.action == "check":
        print("VERIFY FROZEN SIX-SOURCE DATA / c_AMPs SMOKE / RESOURCE ALLOCATION",flush=True)
        c.package_checks(); cohorts = c.load_cohorts(plan); c.smoke_checks(plan)
        import legacy_reuse
        gpu_required = legacy_reuse.pending_gpu_required(plan)
        snap = resources(plan,gpu_required=gpu_required)
        _,coverage = c.make_unions(cohorts)
        print(json.dumps({"resources":snap,"coverage":coverage},indent=2,ensure_ascii=False))
        print("READ-ONLY CHECK PASSED; no workers started, no cache/data/model/environment changed",flush=True)
    else:
        run(args.plan.resolve(),args.confirm_resources)


if __name__ == "__main__":
    main()
