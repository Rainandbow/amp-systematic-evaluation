#!/usr/bin/env python3
"""Pinned HMD-AMP binary release preparation, classifier audit and GPU smoke.

Never trains, never changes existing benchmark results, never downloads a
backbone implicitly. Full-cohort inference is deliberately gated on a later
step, after the server smoke report has been inspected.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
import uuid
import zipfile

COMMIT = "158879d43b612b2000ee0ef541dc203cf495356e"
BASE = "https://raw.githubusercontent.com/ml4bio/HMD-AMP/" + COMMIT + "/"
DRIVE_ID = "1Z4IeD0rUfBtN4OwSh7S-2fJUCbk07qiA"
ZIP_URL = "https://drive.usercontent.google.com/download?id=" + DRIVE_ID + "&export=download&confirm=t"
ZIP_NAME = "AMP_prediction_model.zip"
ZIP_BYTES = 39780727
ZIP_SHA = "c95c527d29f3b6aabeed8408bfd8c908cf321e0904d99208d8a0cc2721259290"
BACKBONE = "esm2_t33_650M_UR50D"
FILES = {
    "README.md": "9ef7799cb330e28e13e057fd4061479fa4e8662880fdf8ad67cb74270247b578",
    "prediction.py": "1f124b724dd706fab8797ed0e9c088ba5705ae4aa560117b85a3d7aba27e8a77",
    "src/Net.py": "e93bc681c174535f3ab2220c5320cd762714f559c5b568f4f37a7d61547a3341",
    "src/utils.py": "5da3ecb777605b88e4aa9a6f9a9db041c83a64e52b9959fbaf2d1eb71df85ba0",
}
WEIGHTS = {
    "AMP/ft_parts.pth": "3a64d00b94a44a423146709e3d9fbb89c1200d02e23cb55374b02f7e80956993",
    "AMP/clsmodel/param.pkl": "67a48272cac4dd640ace6138681fa071637145d47059fa50d3f4aac8a96f33a5",
    "AMP/clsmodel/binner.pkl": "a7ebcbd7bb0e57f0ae819a70c29e8ad0fa392dd6026c08d28abe6a16c31fc51c",
    "AMP/clsmodel/estimator/0-0-erf.est": "db7d44842aefec23c7db822a3f88ddf9ecaaeb2f3efc5a5d57011a600ea3cd0e",
    "AMP/clsmodel/estimator/0-0-rf.est": "ab72bc06bc9bcebef68ac038a66be6c191201e530070a7d8803a9342efbda665",
    "AMP/clsmodel/estimator/0-1-erf.est": "8433f7f91a0a6f000ccd194a8612af0aef4eaa681c1aaddaf25c12cf2ae1ef19",
    "AMP/clsmodel/estimator/0-1-rf.est": "2f7acb839d663ae01cdea8473686f48362ff713481e38c84dc0442d2373abebe",
    "AMP/clsmodel/estimator/0-2-erf.est": "3b0b2ea4fc6010f0d4b95513e388e37688d8288066c6bc41e472f7bb26e137b1",
    "AMP/clsmodel/estimator/0-2-rf.est": "5bafd3d913a52e528490b6fa7c54545eed6573a2145c9469397baebdb3928f50",
    "AMP/clsmodel/estimator/0-3-erf.est": "c1c4d5b8248bb57467d808ddb09e26d2c419e93c37ac42bcd136d9871adb8af9",
    "AMP/clsmodel/estimator/0-3-rf.est": "c466147a60e3ff456d4c0d464483d8a5997034935c4160610c0fe73668e5d84f",
    "AMP/clsmodel/estimator/1-0-erf.est": "836ec11bd23f73fb9635b7d9500b4bda014f2dd942b9a68eff30e2df8340dcd8",
    "AMP/clsmodel/estimator/1-0-rf.est": "e2b1761c194a367f4d3db90b07601e15c8bdab55f9e2129dbb5a5d3ed6549e23",
    "AMP/clsmodel/estimator/1-1-erf.est": "2eef5e8a333141bfe61fe3f48d67f460fcedeb39e2bbd2ba3bfdd6a68754ece3",
    "AMP/clsmodel/estimator/1-1-rf.est": "3d269d5efc8d8a43817bc14bd4dcbe7e143e6e63d9cb03a78f4142d0239a4cac",
    "AMP/clsmodel/estimator/1-2-erf.est": "9906a6c19388187cbf1d09e77d407cce1d407d2b54845d7b0b4c177a103c1b71",
    "AMP/clsmodel/estimator/1-2-rf.est": "e21647b6af0a5b04912970ffcff8ecda944a8d8efb207c5608e4f8717627d0a9",
    "AMP/clsmodel/estimator/1-3-erf.est": "2bfb09faa5c32a3f5e049d835ccb22495e5a023107888308668310b2735fb357",
    "AMP/clsmodel/estimator/1-3-rf.est": "4fe6603eddaf68d7cd6554c3b81c1212b038801fa8e4b46e0b3448bdeebe00a3",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def download(url, destination, expected):
    if destination.exists():
        if sha(destination) != expected:
            raise ValueError("Existing file differs; preserved: " + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging paths preserve failed attempts; never silently overwrite.
    part = destination.with_name(destination.name + ".partial_" + uuid.uuid4().hex[:12])
    print("Download: " + url, flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "HMD-AMP-release-audit/1"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("xb") as output:
        while True:
            block = response.read(64 * 1024)
            if not block:
                break
            output.write(block)
    if sha(part) != expected:
        raise ValueError("Unexpected content/hash, possibly Drive login or quota page; preserved: " + str(part))
    if destination.exists():
        raise FileExistsError("Concurrent download; staging preserved: " + str(part))
    part.rename(destination)


def verify_zip(path):
    if path.stat().st_size != ZIP_BYTES or sha(path) != ZIP_SHA:
        raise ValueError("Wrong/incomplete official binary archive; preserved: " + str(path))
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != len(WEIGHTS) or {i.filename for i in members} != set(WEIGHTS):
            raise ValueError("Archive member set differs from the audited AMP binary release")
        for item in archive.infolist():
            if (item.filename.startswith(("/", "\\")) or "\\" in item.filename
                    or ".." in Path(item.filename).parts or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe archive member")
        for item in members:
            if hashlib.sha256(archive.read(item)).hexdigest() != WEIGHTS[item.filename]:
                raise ValueError("Wrong archive member: " + item.filename)


def verify_resources(resources):
    for relative, expected in FILES.items():
        path = resources / "official" / relative
        if sha(path) != expected:
            raise ValueError("Official source changed: " + str(path))
    for relative, expected in WEIGHTS.items():
        path = resources / "released" / relative
        if sha(path) != expected:
            raise ValueError("Official weight changed: " + str(path))


def prepare(args):
    resources = args.resources
    archive = resources / ZIP_NAME
    if args.archive is not None:
        verify_zip(args.archive)
        if archive.exists() and sha(archive) != ZIP_SHA:
            raise ValueError("Existing archive differs; preserved")
        if not archive.exists():
            import shutil
            resources.mkdir(parents=True, exist_ok=True)
            with args.archive.open("rb") as source, archive.open("xb") as target:
                shutil.copyfileobj(source, target)
    else:
        download(ZIP_URL, archive, ZIP_SHA)
    verify_zip(archive)
    for relative, expected in FILES.items():
        download(BASE + relative, resources / "official" / relative, expected)
    with zipfile.ZipFile(archive) as package:
        for relative, expected in WEIGHTS.items():
            target = resources / "released" / relative
            if target.exists():
                if sha(target) != expected:
                    raise ValueError("Existing extracted weight differs; preserved")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(package.read(relative))
    verify_resources(resources)
    manifest = {
        "tool": "HMD-AMP", "task": "AMP/non-AMP binary classification",
        "paper_doi": "10.1038/s41551-026-01630-w", "repository_commit": COMMIT,
        "weights_source": "https://drive.google.com/file/d/" + DRIVE_ID + "/view",
        "archive_sha256": ZIP_SHA, "archive_bytes": ZIP_BYTES,
        "hash_provenance": "Measured from the author-linked archive on 2026-10-03; not author-published checksums",
        "source_files_sha256": FILES, "weight_files_sha256": WEIGHTS,
        "backbone": BACKBONE, "backbone_downloaded_by_prepare": False,
        "original_code_loads": "Base ESM2-650M plus predictor/outlinear state dictionaries; the supplied part checkpoint is not a replacement full encoder",
        "not_a_completed_prediction": True,
    }
    manifest_path = resources / "release_manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Existing release manifest differs; preserved")
    else:
        write_json(manifest_path, manifest)
    print("HMD PREPARE COMPLETE: pinned official source + 19 binary weight files verified.", flush=True)
    print("No training, no predictions, no ESM2 backbone download.", flush=True)
    print("Manifest: " + str(manifest_path), flush=True)


def runtime():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    from importlib.metadata import version
    versions = {name: version(name) for name in (
        "torch", "fair-esm", "deep-forest", "scikit-learn", "numpy", "pandas", "scipy", "joblib", "biopython")}
    for name, expected in {"deep-forest": "0.1.7", "scikit-learn": "1.3.0", "fair-esm": "2.0.0"}.items():
        if versions[name] != expected:
            raise ValueError("Use isolated " + name + "==" + expected)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return versions


def load_classifier(resources):
    import joblib
    import numpy as np
    import torch
    from deepforest import CascadeForestClassifier
    verify_resources(resources)  # Before any trusted-author pickle is loaded.
    amp = resources / "released/AMP"
    parts = torch.load(amp / "ft_parts.pth", map_location="cpu", weights_only=True)
    if set(parts) != {"predictor", "outlinear"}:
        raise ValueError("Unexpected part-checkpoint keys; do not guess another architecture")
    expected = {"0.weight": (640, 1280), "0.bias": (640,), "3.weight": (320, 640), "3.bias": (320,)}
    if set(parts["predictor"]) != set(expected):
        raise ValueError("Projection state keys differ")
    for key, shape in expected.items():
        tensor = parts["predictor"][key]
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError("Projection shape or numerical error")
    if set(parts["outlinear"]) != {"weight", "bias"}:
        raise ValueError("Unexpected neural output layer")
    for key, shape in {"weight": (2, 320), "bias": (2,)}.items():
        if tuple(parts["outlinear"][key].shape) != shape or not torch.isfinite(parts["outlinear"][key]).all():
            raise ValueError("Output layer shape or numerical error")
    # Joblib deserialization is only for the exact author-linked hash-verified files.
    parameters = joblib.load(amp / "clsmodel/param.pkl")
    if (parameters["n_features"] != 320 or parameters["n_outputs"] != 2
            or parameters["n_layers"] != 2 or parameters["partial_mode"]
            or parameters["use_predictor"] or not parameters["labels_are_encoded"]):
        raise ValueError("Unexpected released deep forest structure")
    classifier = CascadeForestClassifier(n_jobs=2)
    classifier.load(str(amp / "clsmodel"))
    if not np.array_equal(classifier.label_encoder_.classes_, [0, 1]):
        raise ValueError("AMP positive-class probability column cannot be inferred safely")
    # Concurrency only: no tree weights, thresholds or ensemble members changed.
    for layer in classifier.layers_.values():
        for wrapper in layer.estimators_.values():
            for obj in (wrapper, getattr(wrapper, "estimator_", None)):
                if obj is not None and hasattr(obj, "n_jobs"):
                    obj.n_jobs = 2
    return classifier


def checked_scores(classifier, features):
    import numpy as np
    if features.ndim != 2 or features.shape[1] != 320 or not np.isfinite(features).all():
        raise ValueError("Need finite, released 320-dimensional projected features")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        probabilities = classifier.predict_proba(features)
        labels = classifier.predict(features)
    if (probabilities.shape != (len(features), 2) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)
            or not np.array_equal(labels, np.argmax(probabilities, axis=1))):
        raise ValueError("Invalid released probabilities / native-label mapping")
    return probabilities[:, 1], labels


def audit(args):
    versions = runtime()
    classifier = load_classifier(args.resources)
    import numpy as np
    checked_scores(classifier, np.zeros((2, 320), dtype=np.float32))
    report = {
        "tool": "HMD-AMP", "repository_commit": COMMIT, "versions": versions,
        "archive_sha256": ZIP_SHA, "weight_files_sha256": WEIGHTS,
        "classes": [0, 1], "positive_probability_column": 1,
        "n_features": 320, "n_layers": 2,
        "synthetic_feature_sanity_passed": True,
        "not_a_sequence_prediction_or_accuracy_test": True,
        "original_decision": "classifier.predict; argmax across [non-AMP, AMP], ties go to non-AMP",
    }
    destination = args.resources / "classifier_audit.json"
    if destination.exists():
        if read_json(destination) != report:
            raise ValueError("Prior classifier audit differs; preserved")
    else:
        write_json(destination, report)
    print("HMD CLASSIFIER AUDIT PASSED: AMP binary release, projection 1280->640->320, forest [0,1].")
    print("Synthetic check only; no real sequences evaluated and no accuracy estimated.")
    print("Audit: " + str(destination))


def frozen_hashes(bench):
    root = bench.resolve()
    checks = root / "checksums_sha256.txt"
    lines = checks.read_text(encoding="utf-8").splitlines()
    result = {str(checks): sha(checks)}
    for line in lines:
        if not line.strip():
            continue
        if not re.match(r"^[0-9a-f]{64}  ", line):
            raise ValueError("Invalid benchmark checksum line")
        path = (root / line[66:]).resolve()
        if root not in path.parents or sha(path) != line[:64]:
            raise ValueError("Frozen benchmark changed: " + str(path))
        result[str(path)] = line[:64]
    for relative in ("02_queries/validation_full_ge10.tsv", "02_queries/validation_full_ge10.fasta"):
        if str((root / relative).resolve()) not in result:
            raise ValueError("Frozen inventory does not cover validation queries")
    return result


@contextmanager
def local_backbone_factory(esm, encoder, alphabet):
    original = esm.pretrained.esm2_t33_650M_UR50D
    esm.pretrained.esm2_t33_650M_UR50D = lambda: (encoder, alphabet)
    try:
        yield
    finally:
        esm.pretrained.esm2_t33_650M_UR50D = original


def smoke(args):
    if args.backbone is None or not args.backbone.is_file():
        raise ValueError("Provide --backbone /absolute/path/esm2_t33_650M_UR50D.pt; no implicit download")
    if os.name == "nt":
        raise ValueError("Sequence smoke is Linux-server only; prepare may run on Windows E:")
    versions = runtime()
    import numpy as np
    import pandas as pd
    import torch
    import esm
    from Bio import SeqIO
    if not torch.cuda.is_available():
        raise ValueError("Official extractor is CUDA-only; do not silently substitute CPU or another model")
    classifier = load_classifier(args.resources)
    bench = args.project / "runs/amp_binary_broad_v3_1_0/published_amp_benchmark/v1"
    frozen = frozen_hashes(bench)
    directory = bench / ("04_published_tools/HMD_AMP_released_" + COMMIT[:8] + "/smoke10")
    if directory.exists():
        raise FileExistsError("Smoke directory exists; inspect it, do not overwrite: " + str(directory))
    metadata = bench / "02_queries/validation_full_ge10.tsv"
    fasta = bench / "02_queries/validation_full_ge10.fasta"
    frame = pd.read_csv(metadata, sep="\t", dtype={"seq_uid": str, "sequence": str})
    if (len(frame) != 2442 or frame.seq_uid.isna().any() or not frame.seq_uid.is_unique
            or not frame.sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{10,65}").all()
            or not frame.label.isin([0, 1]).all()):
        raise ValueError("Unexpected common validation cohort; no cropping/filtering/padding of sequences")
    records = [(r.id, str(r.seq)) for r in SeqIO.parse(fasta, "fasta")]
    if records != list(zip(frame.seq_uid, frame.sequence)):
        raise ValueError("Frozen TSV/FASTA ID or sequence mismatch")
    frame = frame.head(10).copy().reset_index(drop=True)
    source = args.resources / "official"
    base_hash = sha(args.backbone)
    print("Loading local ESM2-650M; FP32; no training or backbone network access", flush=True)
    # Official fair-esm format, not a renamed Hugging Face checkpoint.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.backbone, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model", "cfg"}.issubset(checkpoint):
        raise ValueError("Need an original fair-esm ESM2 checkpoint, not a Hugging Face state dict")
    encoder, alphabet = esm.pretrained.load_model_and_alphabet_core(BACKBONE, checkpoint)
    del checkpoint
    if getattr(encoder, "num_layers", None) != 33 or getattr(encoder, "embed_dim", None) != 1280:
        raise ValueError("Wrong ESM2 backbone variant")
    encoder.eval()
    directory.mkdir(parents=True)
    query = directory / "queries.fasta"
    with query.open("x", encoding="utf-8") as handle:
        for row in frame.itertuples():
            handle.write(">" + row.seq_uid + "\n" + row.sequence + "\n")
    # Execute pinned author feature-extraction code unchanged; the sole override
    # replaces its network-backed pretrained factory with the verified local one.
    sys.path.insert(0, str(source))
    for name in ("src", "src.Net", "src.utils"):
        if name in sys.modules:
            raise ValueError("Conflicting src module already loaded")
    spec = importlib.util.spec_from_file_location("hmd_author_utils", source / "src/utils.py")
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    start = time.monotonic()
    with local_backbone_factory(esm, encoder, alphabet):
        features, _, ids = author.amp_feature_extraction(
            str(args.resources / "released/AMP/ft_parts.pth"), "cuda", str(query))
    if list(ids) != frame.seq_uid.tolist() or features.shape != (10, 320):
        raise ValueError("Official extractor ID / shape coverage mismatch")
    probabilities, predictions = checked_scores(classifier, features)
    # Verify deterministic eval extraction using a second pass through the same
    # official implementation, not a claim of paper/reference-score reproduction.
    with local_backbone_factory(esm, encoder, alphabet):
        repeated, _, repeated_ids = author.amp_feature_extraction(
            str(args.resources / "released/AMP/ft_parts.pth"), "cuda", str(query))
    if list(repeated_ids) != list(ids) or not np.allclose(features, repeated, atol=2e-5, rtol=1e-5):
        raise ValueError("Eval extraction repeatability failed")
    if sha(args.backbone) != base_hash:
        raise ValueError("Backbone changed during prediction")
    verify_resources(args.resources)
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError("Frozen inputs changed during prediction")
    result = frame[["seq_uid", "sequence", "label"]].copy()
    result["probability"] = probabilities
    result["prediction"] = predictions
    normalized = directory / "normalized_predictions.tsv"
    result.to_csv(normalized, sep="\t", index=False, float_format="%.17g")
    write_json(directory / "run_manifest.json", {
        "tool": "HMD-AMP", "task": "AMP/non-AMP", "cohort": "smoke10",
        "n_samples": 10, "repository_commit": COMMIT, "archive_sha256": ZIP_SHA,
        "weight_files_sha256": WEIGHTS, "source_files_sha256": FILES,
        "versions": versions, "gpu": torch.cuda.get_device_name(0),
        "backbone_path": str(args.backbone.resolve()), "backbone_sha256": base_hash,
        "backbone_hash_provenance": "Locally measured; architecture/state validated; no author checksum available",
        "wrapper_sha256": sha(Path(__file__)), "input_sha256": frozen,
        "normalized_output_sha256": sha(normalized), "query_sha256": sha(query),
        "decision_rule": "Official CascadeForestClassifier.predict; probability = predict_proba column for class 1",
        "probability_not_neural_softmax": True,
        "official_feature_extractor_unmodified": True,
        "local_backbone_factory_override": True,
        "precision": "FP32, TF32 disabled", "repeatability_passed": True,
        "prediction_seconds": time.monotonic() - start,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Smoke only, no accuracy benchmark", "No reference/paper-score parity claimed",
                        "Released-tool training overlap with benchmark has not been audited"],
    })
    print(result.to_string(index=False))
    print("HMD SMOKE COMPLETE: 10 frozen validation queries, native labels/probabilities and repeatability verified.")
    print("Manifest: " + str(directory / "run_manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit", "smoke"))
    parser.add_argument("--project", type=Path, default=Path("/home/omics/jinzy/ai-peptide"))
    parser.add_argument("--resources", type=Path,
                        default=Path("/home/omics/jinzy/software/models/HMD-AMP_released_" + COMMIT[:8]))
    parser.add_argument("--archive", type=Path, help="Existing author binary zip; avoids a second download")
    parser.add_argument("--backbone", type=Path, help="Original local ESM2-650M fair-esm .pt, required only for smoke")
    args = parser.parse_args()
    if os.name == "nt" and args.resources.resolve().drive.upper() != "E:":
        raise ValueError("Windows resource directory must be on E:; no C: downloads authorized")
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "audit":
        audit(args)
    else:
        smoke(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit("STOP: " + str(error))
