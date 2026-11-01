from contextlib import nullcontext
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import core as c
import amplify_output as a
import legacy_reuse as reuse
from test_protocol import synthetic_prepare


class OutputRepairTests(unittest.TestCase):
    def native(self, root, score=.99850494, subs=None, prediction="AMP", sequence="ACDEFGHIKL", uid="one"):
        import pandas as pd
        subs = subs or [.9993049,.9961886,.9983837,.99875754,.99989104]
        row = dict(Sequence_ID=uid,Sequence=sequence,Probability_score=score,Prediction=prediction)
        row.update({"Sub_model_{}_probability_score".format(i+1):p for i,p in enumerate(subs)})
        path = Path(root)/"native.tsv"
        c.write_tsv(path,[row],list(row))
        expected = pd.DataFrame([dict(seq_uid="one",sequence="ACDEFGHIKL",label=1)])
        return path,expected

    def test_reported_row_preserves_native_score_and_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,expected = self.native(tmp)
            before=c.sha(path)
            result,audit = a.amplify_frame(path,expected)
            self.assertEqual(result.probability.iloc[0],.99850494)
            self.assertEqual(result.prediction.iloc[0],1)
            self.assertAlmostEqual(audit["max_absolute_delta"],2.16e-7,places=14)
            self.assertEqual(audit["above_old_bound"],1)
            self.assertEqual(c.sha(path),before)
            self.assertFalse(audit["scores_replaced_or_recomputed"])

    def test_large_discrepancy_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,expected=self.native(tmp,score=.99850)
            with self.assertRaisesRegex(ValueError,"mean mismatch"):
                a.amplify_frame(path,expected)

    def test_invalid_nan_and_out_of_range_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (float("nan"),float("inf"),-1,1.01):
                path,expected=self.native(tmp,score=bad)
                with self.assertRaises(ValueError): a.amplify_frame(path,expected)

    def test_id_sequence_and_missing_coverage_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for changes in ({"uid":"wrong"},{"sequence":"AAAA"}):
                path,expected=self.native(tmp,**changes)
                with self.assertRaises(ValueError): a.amplify_frame(path,expected)
            path,expected=self.native(tmp)
            c.write_tsv(path,[],list(c.read_tsv(path)[0]))
            with self.assertRaises(ValueError): a.amplify_frame(path,expected)

    def test_contradictory_class_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,expected=self.native(tmp,prediction="non-AMP")
            with self.assertRaisesRegex(ValueError,"classes contradict"):
                a.amplify_frame(path,expected)

    def test_float32_serialization_stress(self):
        import numpy as np
        rng=np.random.default_rng(9)
        inputs=rng.random((5,100000),dtype=np.float32)*np.float32(.2)+np.float32(.8)
        scores=np.round(inputs.mean(axis=0),8).astype(float)
        means=np.round(inputs,8).astype(float).mean(axis=0)
        self.assertLess(float(np.max(np.abs(scores-means))),a.MEAN_ATOL)


def legacy_fixture(tmp):
    plan,cohorts,jobs=synthetic_prepare(tmp)
    old=Path(tmp)/"old"; package=Path(tmp)/"old_package"
    package.mkdir(); (package/"old.py").write_text("# synthetic pinned old release\n")
    digest=c.sha(package/"old.py")
    (package/"SHA256SUMS.txt").write_text(digest+"  old.py\n")
    old_plan=dict(plan,version="published_amp_six_by_six_v1.1",work=str(old))
    plan["recovery"]={"version":"published_amp_six_by_six_v1.1","work":str(old),"package":str(package),
        "package_manifest_sha256":c.sha(package/"SHA256SUMS.txt"),"minimum_committed_chunks":2,"expected_total_chunks":len(jobs)}
    prepared=c.read_json(Path(plan["work"])/"prepared.json")
    for name in prepared["files"]:
        dest=old/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(str(Path(plan["work"])/name),str(dest))
    old_jobs=[dict(job,query=str(old/Path(job["query"]).relative_to(plan["work"]))) for job in jobs]
    c.dump(old/"jobs.json",old_jobs)
    protocol=c.read_json(old/"protocol.json"); protocol["version"]="published_amp_six_by_six_v1.1"
    c.dump(old/"protocol.json",protocol)
    prepared["inputs"].update(version="published_amp_six_by_six_v1.1",plan=old_plan,code={"old.py":digest})
    prepared["files"]={name:c.sha(old/name) for name in prepared["files"]}
    c.dump(old/"prepared.json",prepared)
    c.dump(old/"status.json",dict(status="failed_or_interrupted",active={}))
    model=Path(tmp)/"model.bin"; model.write_bytes(b"synthetic model only")
    for job in old_jobs[:2]:
        attempt=old/"03_jobs"/job["key"]/"attempt_old"; attempt.mkdir(parents=True)
        rows=c.read_tsv(job["query"])
        c.write_tsv(attempt/"predictions.tsv",[dict(r,probability=.8,prediction=1) for r in rows],c.NATIVE_FIELDS)
        c.dump(attempt/"identity.json",{"files":{str(model):c.sha(model)}})
        files={name:c.sha(attempt/name) for name in ("predictions.tsv","identity.json")}
        stamp={"inputs":{"version":"published_amp_six_by_six_v1.1","job":job,
            "protocol_sha256":c.sha(old/"protocol.json"),"prepared_sha256":c.sha(old/"prepared.json")},
            "attempt":"attempt_old","files":files}
        c.dump(attempt.parent/"complete.json",stamp)
    return plan,jobs,old_jobs,model


class RecoveryTests(unittest.TestCase):
    def guards(self):
        return mock.patch.object(reuse,"old_guard",side_effect=lambda root:nullcontext())

    def test_adopt_and_resume_without_old_writes(self):
        with tempfile.TemporaryDirectory() as tmp, self.guards(),mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
            plan,jobs,old_jobs,_=legacy_fixture(tmp)
            old=Path(plan["recovery"]["work"])
            before={p.relative_to(old).as_posix():c.sha(p) for p in old.rglob("*") if p.is_file()}
            reuse.adopt(plan,jobs)
            self.assertTrue(all(c.completed_job(plan["work"],j) for j in jobs[:2]))
            self.assertIsNone(c.completed_job(plan["work"],jobs[2]))
            reuse.adopt(plan,jobs)
            self.assertEqual(c.read_json(Path(plan["work"])/"recovery_complete.json")["newly_imported_chunks"],0)
            after={p.relative_to(old).as_posix():c.sha(p) for p in old.rglob("*") if p.is_file()}
            self.assertEqual(before,after)

    def test_corrupt_committed_predictions_fail(self):
        with tempfile.TemporaryDirectory() as tmp,self.guards(),mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
            plan,_,old_jobs,_=legacy_fixture(tmp)
            path=Path(plan["recovery"]["work"])/"03_jobs"/old_jobs[0]["key"]/"attempt_old/predictions.tsv"
            path.write_text("corrupt")
            with self.assertRaises(RuntimeError): reuse.inventory(plan)

    def test_uncommitted_native_tsv_not_imported(self):
        with tempfile.TemporaryDirectory() as tmp,self.guards(),mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
            plan,jobs,old_jobs,_=legacy_fixture(tmp)
            path=Path(plan["recovery"]["work"])/"03_jobs"/old_jobs[2]["key"]/"attempt_failed/native.tsv"
            path.parent.mkdir(parents=True); path.write_text("not committed")
            _,ready=reuse.inventory(plan)
            self.assertEqual(len(ready),2)
            reuse.adopt(plan,jobs)
            self.assertIsNone(c.completed_job(plan["work"],jobs[2]))

    def test_model_change_fails_reuse(self):
        with tempfile.TemporaryDirectory() as tmp,self.guards(),mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
            plan,_,_,model=legacy_fixture(tmp)
            model.write_bytes(b"changed")
            with self.assertRaises(RuntimeError): reuse.inventory(plan,check_model_files=True)

    def test_original_running_state_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as tmp,self.guards(),mock.patch.object(c,"smoke_checks",return_value="synthetic_smoke"):
            plan,_,_,_=legacy_fixture(tmp)
            c.dump(Path(plan["recovery"]["work"])/"status.json",dict(status="running",active={}))
            with self.assertRaises(RuntimeError): reuse.inventory(plan)


if __name__=="__main__": unittest.main(verbosity=2)
