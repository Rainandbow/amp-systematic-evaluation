"""One isolated prediction chunk, launched by the scheduler. Python >=3.6."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

import core as c


def limits(cpus, gpu):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu or "-1"
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if hasattr(os, "sched_setaffinity"):
        requested = {int(x) for x in cpus.split(",")}
        c.require(requested <= os.sched_getaffinity(0), "Worker CPU allocation exceeds inherited affinity")
        os.sched_setaffinity(0, requested)


def gpu_empty(gpu):
    data = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], universal_newlines=True)
    pids = [x.strip() for x in data.splitlines() if x.strip() and "No running processes" not in x]
    c.require(not [p for p in pids if p != str(os.getpid())], "GPU occupied by another process; nothing killed")


def camp_modules():
    sys.path.insert(0, str(c.HERE / "vendor_camps"))
    import common
    import author_loader
    return common, author_loader


def parity(values, reference, name, tolerance=2e-5):
    import numpy as np
    expected = np.asarray([r["probability"] for r in reference["rows"]], dtype=float)
    actual = np.asarray(values, dtype=float).reshape(-1)
    c.require(actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=tolerance),
              "Recovered CPU-smoke parity failed: " + name)
    return float(np.max(np.abs(actual-expected)))


def camp_identity(common, versions):
    files = {path: digest for path, digest in common.WEIGHTS.values()}
    for name in common.WEIGHTS:
        common.checked_weight(name)
    return {"commit": "cf7658bc5d504ba6d996fa7b152270e38275dc46", "files": files, "versions": versions}


def camps_keras(rows, attempt, plan):
    common, _ = camp_modules()
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import backend as K
    from keras.models import load_model
    c.require(tf.__version__ == "1.12.0" and keras.__version__ == "2.2.4", "Wrong c_AMPs legacy runtime")
    K.set_session(tf.Session(config=tf.ConfigProto(device_count={"GPU":0},
                                                  intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)))
    raw = (common.HERE / "vendor/Attention.py").read_text(encoding="utf-8")
    repaired = "\n".join(raw.splitlines()[:1]) + "\n" + textwrap.dedent("\n".join(raw.splitlines()[1:]))
    namespace = {"__name__": "camps_pinned_attention"}
    exec(compile(repaired, "author Attention.py [indentation repair only]", "exec"), namespace)
    identity = camp_identity(common, {"tensorflow":tf.__version__, "keras":keras.__version__, "numpy":np.__version__})
    predictions, details = {}, {}
    controls = np.asarray([common.encode_keras(seq) for _,seq in common.SEQUENCES], dtype="int32")
    x = np.asarray([common.encode_keras(r["sequence"]) for r in rows], dtype="int32")
    for name in ("lstm", "attention"):
        model = load_model(common.checked_weight(name), compile=False,
                           custom_objects={"Attention_layer":namespace["Attention_layer"]})
        c.require(tuple(model.input_shape) == (None,300) and tuple(model.output_shape) == (None,1), "Unexpected Keras architecture shape")
        control_p = model.predict(controls, batch_size=8, verbose=0).reshape(-1)
        details[name+"_cpu_smoke_max_abs"] = parity(control_p, c.read_json(Path(plan["camps_smoke"]) / (name+".json")), name)
        predictions[name] = model.predict(x, batch_size=32, verbose=1).reshape(-1)
    output = []
    for i,r in enumerate(rows):
        # Author Keras prediction scripts emit %.8f before result.pl thresholds.
        out = dict(r, lstm=float("%.8f" % predictions["lstm"][i]),
                   attention=float("%.8f" % predictions["attention"][i]))
        output.append(out)
    return output, identity, details


def camps_bert(rows, attempt, plan):
    common, loader = camp_modules()
    import numpy as np
    import torch
    c.require(torch.cuda.is_available(), "No authorized visible CUDA device")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    modeling, tokenization, head, features = loader.author_modules()
    state = torch.load(common.checked_weight("bert"), map_location="cpu", weights_only=False)
    expected = {"bert_model":"bert-base-uncased", "max_seq_length":128, "num_mlp_layers":0,
                "num_mlp_hiddens":500, "fp16":False}
    c.require(all(state["params"].get(k) == v for k,v in expected.items()), "BERT parameters changed")
    c.require(state["class_name"] == "BertClassifier" and state["model_type"] == "classifier"
              and state["num_labels"] == 2 and not state["input_text_pairs"], "Wrong BERT class/type")
    c.require({int(k):int(v) for k,v in state["id2label"].items()} == {0:0,1:1}, "Wrong BERT label order")
    config = modeling.BertConfig.from_json_file(str(common.HERE / "vendor/bert_config.json"))
    model = head.BertPlusMLP(config, model_type="classifier", num_labels=2,
                            num_mlp_layers=0, num_mlp_hiddens=500)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.float().eval().to("cuda")
    tok = tokenization.BertTokenizer(str(common.HERE / "vendor/vocab.txt"), do_lower_case=True)
    c.require(tok.tokenize(" ".join(common.AA)) == list(common.AA.lower()), "AA tokenizer coverage changed")
    def predict(sequences):
        fs = []
        for seq in sequences:
            f = features.convert_text_to_features(" ".join(seq), None, 128, tok)
            c.require(len(seq) <= 126 and sum(f.input_mask) == len(seq)+2, "No BERT truncation allowed")
            fs.append(f)
        args = [torch.tensor([getattr(f,k) for f in fs], dtype=torch.long, device="cuda")
                for k in ("input_ids","segment_ids","input_mask")]
        with torch.no_grad():
            values = torch.softmax(model(*args),dim=-1).cpu().numpy()
        c.require(values.shape == (len(sequences),2) and np.isfinite(values).all(), "Invalid BERT output")
        return values[:,1]
    controls = predict([seq for _,seq in common.SEQUENCES])
    difference = parity(controls, c.read_json(Path(plan["camps_smoke"]) / "bert.json"), "BERT GPU/CPU", 3e-5)
    output = []
    for offset in range(0,len(rows),8):
        subset = rows[offset:offset+8]
        p = predict([r["sequence"] for r in subset])
        output.extend(dict(r, probability=float(prob)) for r,prob in zip(subset,p))
        if offset == 0 or len(output) % 256 == 0 or len(output) == len(rows):
            print("c_AMPs BERT {}/{}".format(len(output),len(rows)),flush=True)
    identity = camp_identity(common, {"torch":torch.__version__, "numpy":np.__version__})
    return output, identity, {"gpu_cpu_smoke_max_abs":difference, "batch_size":8,
                              "precision":"FP32_TF32_off", "strict_state_dict":True}


def native(rows, attempt, plan, component):
    sys.path.insert(0,str(c.HERE / "vendor_native"))
    import published
    published.limits(os.environ.get("CUDA_VISIBLE_DEVICES") if component in ("tsAMP_I","HMD_AMP") else "")
    identity = published.identify(component,plan)
    if component == "Macrel":
        p,d,details = published.macrel(rows,attempt)
    elif component == "AMPlify":
        # Same official CLI, weights, preprocessing and predictions. Only output
        # consistency validation accounts for FP32 / separately rounded TSV scores.
        import pandas as pd
        import amplify_output
        adapter = published.s.vendor("04_predict_amplify_cpu.py")
        query = attempt / "query.fasta"
        published.s.write_fasta(query,rows)
        details = adapter.execute_native(Path(identity["source"]),query,attempt)
        outputs = list(attempt.glob("AMPlify_balanced_results_*.tsv"))
        c.require(len(outputs)==1,"Missing/ambiguous native AMPlify output")
        expected = pd.DataFrame([{"seq_uid":r["sequence_id"],"sequence":r["sequence"],
                                  "label":int(r["label"])} for r in rows])
        result,audit = amplify_output.amplify_frame(outputs[0],expected)
        p,d = result.probability.to_numpy(float),result.prediction.to_numpy(int)
        details["output_consistency_validation"] = audit
        c.dump(attempt / "amplify_output_validation.json",audit)
    elif component == "tsAMP_I":
        p,d,details = published.tsamp(rows,attempt,identity,plan)
    else:
        p,d,details = published.hmd(rows,attempt,identity,plan)
    c.require(len(p) == len(d) == len(rows),"Native coverage mismatch")
    output = [dict(r,probability=float(prob),prediction=int(decision)) for r,prob,decision in zip(rows,p,d)]
    return output,identity,details


def ampir(rows, attempt, plan, job):
    identity_path = attempt / "identity.json"
    script = c.HERE / "vendor_native/predict_ampir.R"
    exe = plan["tools"]["ampir"]["executable"]
    subprocess.check_call([exe,str(script),"check",job["query"],str(attempt / "unused"),str(identity_path)])
    subprocess.check_call([exe,str(script),"full",job["query"],str(attempt / "native"),str(identity_path)])
    output = c.read_tsv(attempt / "native/predictions.tsv")
    identity = c.read_json(identity_path)
    details = c.read_json(attempt / "native/run_manifest.json")
    return output,identity,details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--key",required=True)
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--cpus",required=True)
    parser.add_argument("--gpu",default="")
    args = parser.parse_args()
    limits(args.cpus,args.gpu)
    c.package_checks()
    plan = c.read_json(args.plan)
    jobs = c.read_json(Path(plan["work"]) / "jobs.json")
    job = next(j for j in jobs if j["key"] == args.key)
    attempt = Path(args.attempt).resolve()
    attempt.relative_to((Path(plan["work"]) / "03_jobs" / job["key"]).resolve())
    c.require(c.sha(job["query"]) == job["query_sha256"],"Query checksum changed")
    rows = c.read_tsv(job["query"])
    c.require(len(rows) == job["n"],"Query length mismatch")
    if args.gpu:
        gpu_empty(args.gpu)
    component = job["component"]
    print("START",component,job["key"],len(rows),"gpu",args.gpu or "CPU",flush=True)
    started = time.monotonic()
    if component == "camps_keras":
        output,identity,details = camps_keras(rows,attempt,plan)
    elif component == "camps_bert":
        output,identity,details = camps_bert(rows,attempt,plan)
    elif component == "ampir":
        output,identity,details = ampir(rows,attempt,plan,job)
    else:
        output,identity,details = native(rows,attempt,plan,component)
    output = c.align(output,rows,component)
    for path,digest in identity["files"].items():
        c.require(c.sha(path) == digest,"Model/runtime file changed during inference")
    c.package_checks()
    c.write_tsv(attempt / "predictions.tsv",output,list(output[0]))
    c.dump(attempt / "identity.json",identity)
    c.dump(attempt / "run_manifest.json",{"component":component,"n":len(rows),"seconds":time.monotonic()-started,
                                          "details":details,"inference_only":True,"gpu":args.gpu or None})
    print("COMPLETE",component,job["key"],flush=True)


if __name__ == "__main__":
    main()
