#!/usr/bin/env python3
"""Run the UNMODIFIED installed AMPlify 2.0.1 balanced CLI, CPU <=2 threads.

Python 3.6 compatible. Official five-submodel ensemble and input padding remain
unchanged. Only TensorFlow session/thread/device configuration and progress
printing differ. A ten-query smoke run is separate from real benchmark outputs.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


SOURCE_SHA = {
    "AMPlify.py": "68fdcf72745cd911d2d50624b619ccf322c08992c31d663c10539c1bd75ea0f6",
    "layers.py": "cac0b5a8ddbdc5d69c4dad2d62fdf296bf48c0823456144406fc45fde0aaff2f",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def verify_preparation(bench):
    for line in (bench / "checksums_sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = (bench / name).resolve()
        if bench.resolve() not in path.parents or sha(path) != digest:
            raise ValueError("Frozen benchmark file changed: " + name)


def parse_fasta(path):
    records, identifier, parts = [], None, []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(parts)))
                identifier, parts = line[1:], []
                if not identifier or len(identifier.split()) != 1:
                    raise ValueError("Query FASTA must have seq_uid-only headers")
            elif identifier is None:
                raise ValueError("Sequence before FASTA header")
            else:
                parts.append(line)
    if identifier is not None:
        records.append((identifier, "".join(parts)))
    return records


def amplify_frame(path, expected):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path, sep="\t", dtype={"Sequence_ID": str, "Sequence": str})
    sub_columns = ["Sub_model_{}_probability_score".format(i) for i in range(1, 6)]
    required = {"Sequence_ID", "Sequence", "Probability_score", "Prediction"}.union(sub_columns)
    if not required.issubset(frame.columns):
        raise ValueError("Need native AMPlify -sub on TSV columns; got " + str(list(frame.columns)))
    if frame.Sequence_ID.isna().any() or not frame.Sequence_ID.is_unique:
        raise ValueError("Missing or duplicate AMPlify sample IDs")
    if expected.seq_uid.isna().any() or not expected.seq_uid.is_unique:
        raise ValueError("Invalid expected IDs")
    if len(frame) != len(expected) or set(frame.Sequence_ID) != set(expected.seq_uid.astype(str)):
        raise ValueError("AMPlify coverage mismatch; never fill missing/invalid predictions with zero")
    aligned = frame.set_index("Sequence_ID").loc[expected.seq_uid.astype(str)].reset_index()
    if not np.array_equal(aligned.Sequence.to_numpy(), expected.sequence.to_numpy()):
        raise ValueError("AMPlify returned sequences disagree with original ID-matched canonical inputs")
    scores = aligned.Probability_score.to_numpy(dtype=float)
    sub_scores = aligned[sub_columns].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(sub_scores).all() or (
        (scores < 0) | (scores > 1)).any() or ((sub_scores < 0) | (sub_scores > 1)).any():
        raise ValueError("AMPlify has missing/invalid probability scores")
    if not np.allclose(scores, sub_scores.mean(axis=1), atol=2e-7, rtol=0):
        raise ValueError("Released ensemble score disagrees with its five native submodel scores")
    decisions = aligned.Prediction.map({"AMP": 1, "non-AMP": 0})
    if decisions.isna().any():
        raise ValueError("Invalid native AMPlify classes")
    if ((decisions.to_numpy() != (scores > 0.5).astype(int)) & (np.abs(scores - 0.5) > 1.1e-7)).any():
        raise ValueError("AMPlify classes contradict its native >0.5 rule beyond float32/TSV rounding")
    result = expected[["seq_uid", "sequence", "label"]].copy().reset_index(drop=True)
    result["probability"] = scores
    result["prediction"] = decisions.to_numpy(dtype=int)
    for index, column in enumerate(sub_columns):
        result[column] = sub_scores[:, index]
    return result


def configure_limits():
    # Set BEFORE importing NumPy, TensorFlow or Keras, including BLAS pools.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    cpu_affinity = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        cpu_affinity = sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0, set(cpu_affinity))
    return cpu_affinity


def execute_native(source, query, output):
    cpu_affinity = configure_limits()
    import tensorflow as tf
    import keras
    from keras import backend as K
    from keras.models import Model
    if tf.__version__ != "1.12.0" or keras.__version__ != "2.2.4":
        raise ValueError("Require official legacy runtime TensorFlow 1.12.0 / Keras 2.2.4")
    config = tf.ConfigProto(device_count={"GPU": 0}, intra_op_parallelism_threads=2,
                            inter_op_parallelism_threads=1)
    session = tf.Session(config=config)
    K.set_session(session)
    original_predict, original_argv = Model.predict, sys.argv
    prediction_times = []
    sys.path.insert(0, str(source.parent))

    def progress_predict(self, *arguments, **keywords):
        print("\nOfficial submodel {}/5: predicting {} queries".format(len(prediction_times) + 1, len(arguments[0])), flush=True)
        start = time.monotonic()
        keywords["verbose"] = 1  # Display only; keep default batch size 32.
        result = original_predict(self, *arguments, **keywords)
        elapsed = time.monotonic() - start
        prediction_times.append(elapsed)
        print("Submodel done: {:.1f}s".format(elapsed), flush=True)
        return result

    Model.predict = progress_predict
    sys.argv = [str(source), "-m", "balanced", "-s", str(query), "-od", str(output),
                "-of", "tsv", "-sub", "on", "-att", "off"]
    native = None
    try:
        spec = importlib.util.spec_from_file_location("native_amplify_2_0_1", source)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        native.main()  # Original loading, preprocessing, ensemble and native output.
    finally:
        Model.predict = original_predict
        sys.argv = original_argv
        sys.path.pop(0)
        # Keras prediction callables may be retained in model reference cycles.
        # Release them while their TensorFlow session is still open. This is
        # post-prediction cleanup only; do not suppress inference exceptions.
        native = None
        gc.collect()
        K.clear_session()
        gc.collect()
        session.close()
    if len(prediction_times) != 5:
        raise ValueError("Expected exactly five official submodel predictions")
    return {"tensorflow": tf.__version__, "keras": keras.__version__, "cpu_affinity": cpu_affinity,
            "submodel_prediction_seconds": prediction_times, "prediction_seconds": sum(prediction_times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.home() / "jinzy/ai-peptide")
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--cohort", choices=("smoke10", "validation_full", "test_full"), required=True)
    args = parser.parse_args()
    configure_limits()
    bench = args.benchmark_root or args.project / "runs/amp_binary_broad_v3_1_0/published_amp_benchmark/v1"
    verify_preparation(bench)
    source = Path(sys.prefix) / "share/amplify/src/AMPlify.py"
    native_sources = {source, source.parent / "layers.py"}
    for path in native_sources:
        if not path.is_file() or sha(path) != SOURCE_SHA[path.name]:
            raise ValueError("Installed source does not match official v2.0.1; do not patch it: " + str(path))
    packages = list((Path(sys.prefix) / "conda-meta").glob("amplify-2.0.1-*.json"))
    if len(packages) != 1:
        raise ValueError("Need the independent Bioconda AMPlify 2.0.1 environment")
    with packages[0].open(encoding="utf-8") as handle:
        package = json.load(handle)
    if package.get("version") != "2.0.1":
        raise ValueError("Unexpected AMPlify package version")
    weights = [source.parent.parent / "models/balanced/AMPlify_balanced_model_weights_{}.h5".format(i) for i in range(1, 6)]
    weight_hashes = {}
    for path in weights:
        if not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError("Missing official balanced weights: " + str(path))
        weight_hashes[path.name] = sha(path)
    import numpy as np
    import pandas as pd
    if np.__version__.split(".")[:2] != ["1", "16"]:
        raise ValueError("Use the declared NumPy 1.16 legacy environment")
    name = "validation_full" if args.cohort == "smoke10" else args.cohort
    metadata = bench / "02_queries/{}_ge10.tsv".format(name)
    expected = pd.read_csv(metadata, sep="\t", dtype={"seq_uid": str}).reset_index(drop=True)
    original_query = bench / "02_queries/{}_ge10.fasta".format(name)
    if parse_fasta(original_query) != list(zip(expected.seq_uid, expected.sequence)):
        raise ValueError("Frozen FASTA/TSV order or sequence mismatch")
    if not expected.sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{10,65}").all():
        raise ValueError("Use only the existing common canonical 10..65-aa cohort")
    tools = bench / "04_published_tools/AMPlify_balanced_v2_0_1"
    output = tools / args.cohort
    if output.exists():
        raise FileExistsError("Output exists; preserve it and inspect before rerunning: " + str(output))
    output.mkdir(parents=True)
    if args.cohort == "smoke10":
        expected = expected.head(10).copy()
        query = output / "query_smoke10.fasta"
        with query.open("w", encoding="utf-8") as handle:
            for record in expected.itertuples(index=False):
                handle.write(">{}\n{}\n".format(record.seq_uid, record.sequence))
    else:
        query = original_query
    query_digest = sha(query)
    print("Official AMPlify 2.0.1 BALANCED ensemble; CPU intra-op=2 / inter-op=1; no retraining", flush=True)
    print("Cohort: {} | queries: {} | output: {}".format(args.cohort, len(expected), output), flush=True)
    start = time.monotonic()
    runtime = execute_native(source, query, output)
    native_outputs = list(output.glob("AMPlify_balanced_results_*.tsv"))
    if len(native_outputs) != 1:
        raise ValueError("Expected exactly one complete native AMPlify TSV")
    native_output = native_outputs[0]
    normalized = amplify_frame(native_output, expected)
    for path in weights:
        if sha(path) != weight_hashes[path.name]:
            raise ValueError("An official weight file changed during prediction")
    if sha(query) != query_digest:
        raise ValueError("Query changed during prediction")
    normalized.to_csv(output / "normalized_predictions.tsv", sep="\t", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "tool": "AMPlify", "version": "2.0.1",
        "model_variant": "balanced", "model_variant_selection": "Predeclared paper balanced model; NOT selected by Test",
        "ensemble": "Released arithmetic mean of the five balanced submodels, not five independent benchmark seeds",
        "cohort": args.cohort, "n_samples": len(expected), "native_output_name": native_output.name,
        "native_output_sha256": sha(native_output), "normalized_output_sha256": sha(output / "normalized_predictions.tsv"),
        "query_sha256": query_digest, "metadata_sha256": sha(metadata), "official_cli_options": "-m balanced -of tsv -sub on -att off",
        "weights_sha256": weight_hashes, "source_sha256": {path.name: sha(path) for path in native_sources},
        "wrapper_sha256": sha(Path(__file__)), "conda_package": {key: package.get(key) for key in ("version", "build", "sha256", "channel")},
        "runtime": runtime, "total_elapsed_seconds": time.monotonic() - start,
        "runtime_policy": "CPU only; Linux affinity limited to at most two currently allowed CPUs; TF intra-op 2 / inter-op 1; BLAS threads 2. Original architecture, padding 200, batch size 32, weights and averaging unchanged; progress display enabled.",
        "output_precision": "Native CLI rounds scores to 8 decimals (float32); preserve native AMP/non-AMP decisions",
        "training_overlap_status": "NOT audited; current benchmark is an exploratory previously viewed internal holdout",
    }
    write_json(output / "run_manifest.json", manifest)
    print("\nAMPLIFY COMPLETE: {} queries, exact ID/sequence coverage, five-submodel ensemble verified".format(len(normalized)), flush=True)
    print("Prediction time: {:.1f}s | total {:.1f}s".format(runtime["prediction_seconds"], manifest["total_elapsed_seconds"]), flush=True)
    print("Manifest: " + str(output / "run_manifest.json"), flush=True)
    if args.cohort == "smoke10":
        print("SMOKE ONLY: do not use these ten predictions as a benchmark. Review runtime before the full cohorts.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError, ImportError) as error:
        print("STOP: " + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
