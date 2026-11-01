import ast
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock
import core as c
import reports
import scheduler


def rows(dataset_index=0):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    result = []
    for i in range(4):
        seq = alphabet[dataset_index]*10 + alphabet[i+8] + alphabet[dataset_index+3]*3
        uid = "s_"+hashlib.sha256(seq.encode()).hexdigest()
        result.append(dict(sequence_id=uid,sequence=seq,group_id="g_"+uid,split="test",label=str(i%2)))
    return result


def plan(tmp):
    return {"version":c.VERSION,"source_work":str(Path(tmp)/"source"),"work":str(Path(tmp)/"work"),
            "source_manifest_sha256":"synthetic", "camps_smoke":str(Path(tmp)/"smoke"),
            "chunk_sizes":{x:3 for x in c.COMPONENTS}}


def synthetic_prepare(tmp):
    p = plan(tmp)
    cohorts = {dataset:rows(i) for i,dataset in enumerate(c.DATASETS)}
    with mock.patch.object(c,"package_checks",return_value={"synthetic":"no_real_weights"}), \
         mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
        c.prepare(p,cohorts)
    return p,cohorts,c.read_json(Path(p["work"])/"jobs.json")


class CoreTests(unittest.TestCase):
    def test_six_by_six_names(self):
        self.assertEqual(len(c.DATASETS)*len(c.TOOLS),36)
        self.assertEqual(len(set(c.DATASETS)),6)
        self.assertEqual(sum(t=="c_AMPs" for t in c.TOOLS),1)

    def test_eligibility_boundaries(self):
        for tool,accepted,rejected in [("c_AMPs",126,127),("AMPlify",200,201),("tsAMP_I",1022,1023),
                                       ("HMD_AMP",1022,1023),("ampir",10,9),("Macrel",10,9)]:
            self.assertTrue(c.support(tool,{"sequence":"A"*accepted}))
            self.assertFalse(c.support(tool,{"sequence":"A"*rejected}))

    def test_common_domain(self):
        for n,expected in [(1,False),(9,False),(10,True),(126,True),(127,False)]:
            self.assertEqual(all(c.support(t,{"sequence":"A"*n}) for t in c.TOOLS),expected)

    def test_union_deduplication(self):
        cohorts = {name:rows() for name in c.DATASETS}
        unions,coverage = c.make_unions(cohorts)
        self.assertEqual(len(coverage),36)
        self.assertTrue(all(len(q)==4 for q in unions.values()))

    def test_union_label_conflict(self):
        cohorts = {name:rows() for name in c.DATASETS}
        cohorts[c.DATASETS[-1]][0]["label"]="1"
        with self.assertRaises(RuntimeError): c.make_unions(cohorts)

    def test_alignment_order_and_metadata(self):
        expected=rows(); result=[dict(r,probability=.7,prediction=1) for r in expected]
        self.assertEqual(c.align(list(reversed(result)),expected,"Macrel"),result)
        result[0]["sequence"]="A"*10
        with self.assertRaises(RuntimeError): c.align(result,expected,"Macrel")

    def test_missing_nan_duplicate(self):
        expected=rows()
        for variant in ("missing","nan","duplicate"):
            saved=[dict(r,probability=.2,prediction=0) for r in expected]
            if variant=="missing": saved.pop()
            if variant=="nan": saved[0]["probability"]=float("nan")
            if variant=="duplicate": saved[-1]=saved[0]
            with self.assertRaises(RuntimeError): c.align(saved,expected,"Macrel")

    def test_keras_components_validation(self):
        saved=[dict(r,lstm=.2,attention=.8) for r in rows()]
        c.align(saved,rows(),"camps_keras")
        saved[0]["attention"]=1.1
        with self.assertRaises(RuntimeError): c.align(saved,rows(),"camps_keras")

    def test_manifest_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): c.verify_files(tmp,{"../outside":"bad"})

    def test_prepare_resume_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,cohorts,jobs=synthetic_prepare(tmp)
            before=c.sha(Path(p["work"])/"prepared.json")
            with mock.patch.object(c,"package_checks",return_value={"synthetic":"no_real_weights"}), \
                 mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
                c.prepare(p,cohorts)
                self.assertEqual(before,c.sha(Path(p["work"])/"prepared.json"))
                p["chunk_sizes"]["camps_bert"]=10
                with self.assertRaises(RuntimeError): c.prepare(p,cohorts)

    def test_incomplete_and_checksum_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,_,jobs=synthetic_prepare(tmp); job=jobs[0]
            self.assertIsNone(c.completed_job(p["work"],job))
            attempt=Path(p["work"])/"03_jobs"/job["key"]/"attempt_test"
            attempt.mkdir(parents=True)
            c.write_tsv(attempt/"predictions.tsv",[dict(r,probability=.2,prediction=0) for r in c.read_tsv(job["query"])],c.NATIVE_FIELDS)
            c.dump(attempt/"identity.json",{"synthetic":True})
            c.commit_job(p["work"],job,attempt)
            self.assertEqual(c.completed_job(p["work"],job),attempt.resolve())
            with self.assertRaises(RuntimeError): c.commit_job(p["work"],job,attempt)
            (attempt/"predictions.tsv").write_text("tampered")
            with self.assertRaises(RuntimeError): c.completed_job(p["work"],job)

    def test_frozen_source_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=plan(tmp); root=Path(p["source_work"]); files={}; expected={}
            for i,dataset in enumerate(c.DATASETS):
                test=rows(i)
                all_rows=[dict(r,length=len(r["sequence"])) for r in test]
                directory=root/"04_datasets"/dataset
                for split in ("all","train","validation","test"):
                    path=directory/(split+".tsv")
                    chosen=all_rows if split in ("all","test") else []
                    c.write_tsv(path,chosen,c.FIELDS+["length"])
                    files[path.relative_to(root).as_posix()]=c.sha(path)
                expected[dataset]=(4,2,2)
            c.dump(root/"dataset_complete.json",{"version":"external_plm_holdout_v2.2","files":files})
            p["source_manifest_sha256"]=c.sha(root/"dataset_complete.json")
            loaded=c.load_cohorts(p,expected)
            self.assertEqual(len(loaded),6)
            (root/"04_datasets/AMPlify/test.tsv").write_text("tamper")
            with self.assertRaises(RuntimeError): c.load_cohorts(p,expected)

    def test_python36_worker(self):
        for name in ("core.py","worker.py","amplify_output.py"):
            ast.parse((c.HERE/name).read_text(),feature_version=(3,6))


class MetricsTests(unittest.TestCase):
    def test_confusion_and_ap(self):
        data=[dict(label=str(y),prediction=str(d),score=p) for y,d,p in [(1,1,.9),(1,0,.4),(0,1,.7),(0,0,.1)]]
        result=reports.metrics(data,"Macrel")
        self.assertEqual((result["tp"],result["tn"],result["fp"],result["fn"]),(1,1,1,1))
        self.assertEqual(result["mcc"],0)
        self.assertEqual(result["accuracy"],.5)
        self.assertAlmostEqual(result["auprc"],5/6)

    def test_camps_native_no_invented_probability(self):
        data=[dict(label=str(y),prediction=str(y),score="",extension_min_score=p) for y,p in [(1,.9),(0,.1)]]
        native=reports.metrics(data,"c_AMPs")
        extended=reports.metrics(data,"c_AMPs",True)
        self.assertIsNone(native["auprc"])
        self.assertIsNone(native["auroc"])
        self.assertEqual(extended["auprc"],1)
        self.assertEqual(native["mcc"],1)

    def test_empty_and_single_class(self):
        empty=reports.metrics([],"Macrel")
        self.assertIsNone(empty["accuracy"])
        single=reports.metrics([dict(label="0",prediction="0",score=.1)],"Macrel")
        self.assertIsNone(single["auroc"])
        self.assertEqual(single["accuracy"],1)

    def test_end_to_end_synthetic_36_cells_108_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,cohorts,jobs=synthetic_prepare(tmp)
            for job in jobs:
                attempt=Path(p["work"])/"03_jobs"/job["key"]/"attempt_test"
                attempt.mkdir(parents=True)
                component=job["component"]
                data=[]
                for r in c.read_tsv(job["query"]):
                    prob=.8 if int(r["label"]) else .2
                    if component=="camps_keras": out=dict(r,lstm=prob,attention=prob)
                    elif component=="camps_bert": out=dict(r,probability=prob)
                    else: out=dict(r,probability=prob,prediction=int(r["label"]))
                    data.append(out)
                c.write_tsv(attempt/"predictions.tsv",data,list(data[0]))
                c.dump(attempt/"identity.json",{"synthetic":True,"component":component})
                c.commit_job(p["work"],job,attempt)
            with mock.patch.object(reports,"source_matches",return_value=([{"synthetic":True}],{"synthetic":True})):
                reports.summarize(p,cohorts,jobs)
            root=Path(p["work"])
            output=c.read_tsv(root/"05_reports/final_results.tsv")
            self.assertEqual(len(output),108)
            self.assertTrue(all(float(r["mcc"])==1 for r in output))
            self.assertEqual(len(c.read_tsv(root/"05_reports/camps_min_score_extension.tsv")),18)
            self.assertTrue(scheduler.completed_experiment(root))
            matrix=c.read_tsv(root/"05_reports/common_six_tools__mcc_matrix.tsv")
            self.assertEqual(len(matrix),6)
            self.assertEqual(list(matrix[0]),["tool"]+c.DATASETS)


class EntryPointTests(unittest.TestCase):
    def status_cli(self, tmp):
        p = {"work":str(Path(tmp)/"work")}
        path = Path(tmp)/"plan.json"
        c.dump(path,p)
        result = subprocess.run([sys.executable,"-B",str(c.HERE/"scheduler.py"),
                                 "status","--plan",str(path)],
                                capture_output=True,text=True,check=True)
        return p,json.loads(result.stdout)

    def test_real_status_cli_not_started_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,status = self.status_cli(tmp)
            self.assertEqual(status,{"status":"not_started"})
            self.assertFalse(Path(p["work"]).exists())

    def test_real_status_cli_with_existing_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = {"status":"running","completed_cells":0,"active":{"lane":"test"}}
            c.dump(Path(tmp)/"work/status.json",expected)
            _,status = self.status_cli(tmp)
            self.assertEqual(status,expected)

    def test_check_dispatch_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"plan.json"; c.dump(path,{"synthetic":True})
            output=io.StringIO()
            with mock.patch.object(sys,"argv",["scheduler.py","check","--plan",str(path)]), \
                 mock.patch.object(c,"package_checks"), \
                 mock.patch.object(c,"load_cohorts",return_value={}), \
                 mock.patch.object(c,"smoke_checks"), \
                 mock.patch.object(scheduler,"resources",return_value={"mocked_devices":True}), \
                 mock.patch.object(c,"make_unions",return_value=({},[])), \
                 redirect_stdout(output):
                scheduler.main()
            self.assertIn('"mocked_devices": true',output.getvalue())
            self.assertIn("READ-ONLY CHECK PASSED",output.getvalue())

    def test_run_dispatch_requires_explicit_resource_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"plan.json"; c.dump(path,{"synthetic":True})
            for confirmed in (False,True):
                argv=["scheduler.py","run","--plan",str(path)]
                if confirmed: argv.append("--confirm-resources")
                with mock.patch.object(sys,"argv",argv),mock.patch.object(scheduler,"run") as run:
                    scheduler.main()
                    run.assert_called_once_with(path.resolve(),confirmed)


class SchedulingTests(unittest.TestCase):
    def snapshot(self):
        return {"cpu_ids":list(range(16)),"devices":[{"index":i,"uuid":"GPU-device-"+str(i)} for i in (4,5,6,7)]}

    def test_disjoint_bert_queues_and_all_jobs_once(self):
        jobs=[{"component":name,"key":name+"__{:06d}".format(i)} for name in c.COMPONENTS for i in range(7)]
        selected=[j["key"] for lane in scheduler.lanes(self.snapshot()) for j in scheduler.jobs_for_lane(jobs,lane)]
        self.assertEqual(sorted(selected),sorted(j["key"] for j in jobs))
        self.assertEqual(len(selected),len(set(selected)))

    def test_cpu_and_gpu_lane_scope(self):
        lanes=scheduler.lanes(self.snapshot())
        cpus=[x for lane in lanes for x in lane["cpus"]]
        self.assertEqual(len(cpus),14)
        self.assertEqual(len(set(cpus)),14)
        self.assertEqual({q["gpu"] for q in lanes if q["gpu"]},{"GPU-device-"+str(i) for i in (4,5,6,7)})

    def test_cpu_only_lanes_do_not_access_gpu(self):
        snapshot=self.snapshot(); snapshot["devices"]=[]
        lanes=scheduler.lanes(snapshot)
        self.assertEqual(len(lanes),3)
        self.assertTrue(all(not lane["gpu"] for lane in lanes))
        self.assertEqual(sum("AMPlify" in lane["components"] for lane in lanes),1)

    def test_cpu_only_resources_do_not_query_nvidia(self):
        p=c.read_json(c.HERE/"plan.server.json")
        with mock.patch.object(os,"sched_getaffinity",return_value=set(range(16)),create=True), \
             mock.patch.object(Path,"read_text",return_value="MemAvailable: 1000000000 kB\n"), \
             mock.patch.object(Path,"is_file",return_value=True), \
             mock.patch.object(scheduler.subprocess,"check_output",side_effect=AssertionError("No GPU query allowed")):
            snapshot=scheduler.resources(p,gpu_required=False)
            self.assertFalse(snapshot["gpu_required"])
            self.assertEqual(snapshot["devices"],[])

    def test_idle_gpu_help_no_double_claim(self):
        jobs=[{"component":name,"key":name+"__{:06d}".format(i)} for name in c.COMPONENTS for i in range(7)]
        done=set(); claimed=set(); selected=[]
        lanes=scheduler.lanes(self.snapshot())
        while True:
            progress=False
            for lane in lanes:
                job=scheduler.claim_job(jobs,lane,done,claimed)
                if job:
                    progress=True; selected.append(job["key"])
            if not progress: break
        self.assertEqual(sorted(selected),sorted(j["key"] for j in jobs))
        self.assertEqual(len(selected),len(set(selected)))
        bert_lane=lanes[-1]
        only_ts=[{"component":"tsAMP_I","key":"tsAMP_I__000000"}]
        self.assertEqual(scheduler.claim_job(only_ts,bert_lane,set(),set()),only_ts[0])

    def resource_test(self,mask,occupied="",affinity=16,free=32000):
        p=c.read_json(c.HERE/"plan.server.json")
        data="\n".join("{}, GPU-device-{}, fake, {}".format(i,i,free) for i in range(8))
        environment={} if mask is None else {"CUDA_VISIBLE_DEVICES":mask}
        with mock.patch.dict(os.environ,environment,clear=True), \
             mock.patch.object(os,"sched_getaffinity",return_value=set(range(affinity)),create=True), \
             mock.patch.object(Path,"read_text",return_value="MemAvailable: 1000000000 kB\n"), \
             mock.patch.object(Path,"is_file",return_value=True), \
             mock.patch.object(subprocess_module(),"check_output",side_effect=[data,occupied]):
            return scheduler.resources(p)

    def test_allowed_gpu_masks(self):
        for mask in (None,"4,5,6,7","GPU-device-4,GPU-device-5,GPU-device-6,GPU-device-7"):
            self.assertEqual(len(self.resource_test(mask)["devices"]),4)

    def test_no_allocation_widening(self):
        for mask in ("4,5","","-1","0,1,2,3"):
            with self.assertRaises(RuntimeError): self.resource_test(mask)

    def test_busy_memory_and_cpu_fail_closed(self):
        for kwargs in ({"occupied":"GPU-device-4, 123"},{"free":1000},{"affinity":8}):
            with self.assertRaises(RuntimeError): self.resource_test(None,**kwargs)


def subprocess_module():
    return scheduler.subprocess


if __name__=="__main__": unittest.main(verbosity=2)
