#!/usr/bin/env python3
"""Audit the released tsAMP-I checkpoint and smoke-test ten frozen queries.

This is an inference adapter, not the authors' original end-to-end CLI.
No training and no evaluation on test labels. Full benchmarking is deliberately
not implemented until the reference-feature parity gate passes on the server.
"""
import argparse
import ast
import csv
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

COMMIT = "82c722090b10878edb0b9f45033e1806a659f2b7"
BASE = "https://raw.githubusercontent.com/YangLab-BUPT/tsAMP/" + COMMIT + "/"
# Hashes independently checked against the pinned author repository.
RESOURCES = {
    "model.py": ("model/tsAMP-I/model.py", "5ec035ea6c5178fa85e97dbafd90e846dd2639c6d60a190b537e78c26d60e3db"),
    "predict.py": ("model/tsAMP-I/predict.py", "bb58f297daa0f93d7d5bf4fe15c04ce0d368970ba9c79c1813e7107dd3ce10f1"),
    "charge.py": ("model/tsAMP-I/charge.py", "37ebe799bac09ec4bf3939bf195603a40fb4dfa43c217cf1703ea80a8318f441"),
    "tsAMP-I.pt": ("model/tsAMP-I/tsAMP-I.pt", "c0b85ecc7f0b8c2f9100cb829e09ad33dc204e470e4355877f945265d360f968"),
    "DBAASPN_10400.pt": ("data/AMPesm1v/DBAASPN_10400.pt", "fa26287cd8d355bb5f958f8b935209c03a686756e1d3ac540c90ca27842249d9"),
    "DBAASPN_10401.pt": ("data/AMPesm1v/DBAASPN_10401.pt", "295802f604840e0b5fa5c286470241daafaef7bf8d4cfd0ebb005de05ba2b5cf"),
    "L01A000044.pt": ("data/tsAMP-I/testAMP/L01A000044%7CBOMBININ-LIKE.pt", "c9aad5ca7f112e4b59c03b9a4bc4a0a9c3382723a0804a6494366c210f73582e"),
    "train_amp.fasta": ("data/tsAMP-I/train_amp.fasta", "6e9c0fc921b91af06e3d27411233377ab7095cb966e83c66f2f1e5b5a2f9646a"),
    "testLAMP2.xlsx": ("data/tsAMP-I/testLAMP2.xlsx", "9a8950f7eecbe8c8ff22197488dee01a7fa2274a0d43e489dff1ee6477b3e005"),
}
BACKBONE = "esm1v_t33_650M_UR90S_1"
BACKBONE_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/" + BACKBONE + ".pt"
# Size checked using the original host's HTTP Content-Range, 2026-10-03.
BACKBONE_BYTES = 7828635339


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def fetch(url, path, expected=None):
    if path.exists():
        if expected is None or sha(path) == expected:
            return
        raise ValueError("Existing resource has the wrong hash; preserved: " + str(path))
    part = path.with_name(path.name + ".partial")
    if part.exists():
        if expected is not None and sha(part) == expected:
            part.rename(path)
            print("Recovered complete, hash-verified download: " + str(path), flush=True)
            return
        preserve_partial(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tsAMP-release-audit/1"})
    for attempt in range(1, 4):
        print("Downloading (attempt %d/3): %s" % (attempt, url), flush=True)
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("xb") as handle:
                total, last = 0, time.monotonic()
                declared_size = response.headers.get("Content-Length")
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    total += len(block)
                    if time.monotonic() - last >= 20:
                        print("Downloaded %.1f MiB" % (total / 1024**2), flush=True)
                        last = time.monotonic()
                if declared_size is not None and total != int(declared_size):
                    raise OSError("Truncated HTTP response: %d / %s bytes" % (total, declared_size))
            break
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            if part.exists():
                preserve_partial(part)
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if isinstance(error, OSError) and error.errno in (13, 28, 30):
                raise  # Permission / disk-full / read-only errors are not network failures.
            if attempt == 3:
                raise OSError("Download failed after three attempts; existing resources preserved: " + str(error)) from error
            print("Network interrupted; retrying current file only: " + str(error), flush=True)
            time.sleep(5 * attempt)
    if expected is not None and sha(part) != expected:
        raise ValueError("Download hash mismatch; partial file preserved: " + str(part))
    part.rename(path)


def preserve_partial(part):
    """Preserve failed downloads without deleting or overwriting anything."""
    destination = part.with_name(part.name + ".failed_" + time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12])
    part.rename(destination)
    print("Incomplete download preserved: " + str(destination), flush=True)


def replace_json(path, value):
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    write_json(temporary, value)
    os.replace(temporary, path)


def fetch_backbone(path, attempts=6):
    """Resume only with a saved HTTP validator and exact Content-Range checks.

    Old failed fragments without HTTP validators remain untouched. Converted
    Hugging Face checkpoints must not be renamed to the fair-esm .pt filename.
    """
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected backbone path: " + str(path))
        print("Existing backbone reused; tensor loading/parity will check it: " + str(path), flush=True)
        return
    if not 1 <= attempts <= 30:
        raise ValueError("Download attempts must be between 1 and 30")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".resumable.partial")
    metadata = path.with_name(path.name + ".resumable.json")
    if part.is_symlink() or metadata.is_symlink():
        raise ValueError("Unexpected resumable-file symlink; stop without changes")

    def archive_pair():
        if part.exists():
            preserve_partial(part)
        if metadata.exists():
            preserve_partial(metadata)

    for attempt in range(1, attempts + 1):
        saved = None
        if metadata.exists():
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if saved.get("url") != BACKBONE_URL:
                raise ValueError("Resumable metadata URL changed; files preserved")
            if saved.get("total_bytes") != BACKBONE_BYTES:
                raise ValueError("Unexpected resumable model size; files preserved")
        offset = part.stat().st_size if part.exists() else 0
        if offset and saved is None:
            raise ValueError("Fragment has no HTTP validator; preserved, cannot safely append")
        if saved is not None and offset > saved["total_bytes"]:
            raise ValueError("Fragment is larger than the declared model; preserved")
        if saved is not None and offset == saved["total_bytes"]:
            finish_backbone(part, path, saved)
            return
        headers = {"User-Agent": "tsAMP-release-audit/2", "Accept-Encoding": "identity"}
        if offset:
            if not saved.get("validator"):
                raise ValueError("Server did not provide a validator; cannot safely resume this fragment")
            headers["Range"] = "bytes=%d-" % offset
            headers["If-Range"] = saved["validator"]
        print("Backbone attempt %d/%d; resume at %.2f MiB" % (attempt, attempts, offset / 1024**2), flush=True)
        request = urllib.request.Request(BACKBONE_URL, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Encoded HTTP body cannot be appended safely")
                if offset and status == 200:
                    # If-Range changed, or this server ignored Range. Never append
                    # an entire file to a fragment: preserve and start afresh.
                    print("Server ignored Range or the resource changed; preserving fragment", flush=True)
                    archive_pair()
                    continue
                if offset:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if status != 206 or match is None:
                        raise ValueError("Invalid HTTP range response; fragment preserved")
                    begin, end, total = map(int, match.groups())
                    if begin != offset or total != saved["total_bytes"] or end != total - 1:
                        raise ValueError("HTTP range offset/size mismatch; fragment preserved")
                    if response.headers.get(saved["validator_header"]) != saved["validator"]:
                        raise ValueError("HTTP resource validator changed; fragment preserved")
                else:
                    declared = response.headers.get("Content-Length", "")
                    if status != 200 or not declared.isdigit():
                        raise ValueError("Initial response must have HTTP 200 and a known file size")
                    total = int(declared)
                    if total != BACKBONE_BYTES:
                        raise ValueError("Unexpected original ESM-1v size: %d; expected %d" % (total, BACKBONE_BYTES))
                    etag = response.headers.get("ETag")
                    validator_header = "ETag" if etag and not etag.startswith("W/") else "Last-Modified"
                    saved = {"url": BACKBONE_URL, "total_bytes": total,
                             "validator_header": validator_header,
                             "validator": response.headers.get(validator_header),
                             "content_encoding": "identity"}
                    replace_json(metadata, saved)
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != total - offset:
                    raise ValueError("HTTP remaining-length mismatch; fragment preserved")
                mode = "ab" if offset else "xb"
                if not offset and part.exists():
                    # A previous connection might have created a zero-byte file.
                    preserve_partial(part)
                received, last = offset, time.monotonic()
                with part.open(mode) as handle:
                    while True:
                        block = response.read(16 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        received += len(block)
                        if received > total:
                            raise ValueError("Response exceeded the declared model size; preserved")
                        if time.monotonic() - last >= 20:
                            handle.flush()
                            print("Backbone %.2f / %.2f MiB (%.2f%%)" % (
                                received / 1024**2, total / 1024**2, 100 * received / total), flush=True)
                            last = time.monotonic()
                if received != total:
                    raise OSError("Interrupted HTTP body; keeping fragment for the next attempt")
            finish_backbone(part, path, saved)
            return
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if isinstance(error, OSError) and error.errno in (13, 28, 30):
                raise
            print("Network interrupted; fragment retained for resume: " + str(error), flush=True)
            if attempt < attempts:
                time.sleep(min(5 * attempt, 20))
    raise OSError("Backbone download incomplete. Rerun download-backbone to resume; no fragments deleted.")


def finish_backbone(part, path, metadata):
    if part.stat().st_size != metadata["total_bytes"]:
        raise ValueError("Model size mismatch; partial file preserved")
    digest = sha(part)
    # This is a locally measured checksum, not a publisher-provided SHA256.
    replace_json(path.with_name(path.name + ".download_manifest.json"),
                 {**metadata, "sha256": digest, "publisher_sha256_available": False})
    part.rename(path)
    print("BACKBONE DOWNLOAD COMPLETE: %d bytes; SHA256 %s" % (metadata["total_bytes"], digest), flush=True)


def verify_resources(root):
    for name, (_, expected) in RESOURCES.items():
        if sha(root / name) != expected:
            raise ValueError("Official resource changed: " + name)


def fasta_records(path):
    records, name, pieces = [], None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(pieces)))
            name, pieces = line[1:].split()[0], []
        elif line.strip():
            if name is None:
                raise ValueError("Malformed FASTA")
            pieces.append(line.strip())
    if name is not None:
        records.append((name, "".join(pieces)))
    if len({name for name, _ in records}) != len(records):
        raise ValueError("Duplicate FASTA IDs")
    return records


def property_functions(path):
    # Execute only the pinned authors' two formula functions and their constants.
    # Never execute their hard-coded file-writing entry point.
    source = ast.parse(Path(path).read_text(encoding="utf-8"))
    constants = {"hydrophobicity_values", "hydrophobic_aa", "hydrophilic_aa"}
    functions = {"calculate_hydrophobicity", "calculate_amphipathicity"}
    selected = []
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    if len(selected) != 5:
        raise ValueError("Unexpected official property definitions")
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def verify_frozen(bench):
    root = bench.resolve()
    for line in (root / "checksums_sha256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not re.match(r"^[0-9a-f]{64}  ", line):
            raise ValueError("Invalid frozen checksum line")
        target = (root / line[66:]).resolve()
        if root not in target.parents or sha(target) != line[:64]:
            raise ValueError("Frozen benchmark changed: " + str(target))


def smoke(args, root):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import numpy as np
    import pandas as pd
    import torch
    import esm
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from importlib.metadata import version
    if version("fair-esm") != "2.0.0" or version("biopython") != "1.81":
        raise ValueError("Use isolated fair-esm 2.0.0 / Biopython 1.81 environment")
    if not hasattr(esm.pretrained, "load_model_and_alphabet_core"):
        raise ValueError("Wrong esm package: do not use the ESMC runtime")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    bench = args.project / "runs/amp_binary_broad_v3_1_0/published_amp_benchmark/v1"
    verify_frozen(bench)
    output = bench / "04_published_tools/tsAMP_I_released_82c72209/smoke10"
    if output.exists():
        if args.archive_empty_smoke and output.is_dir() and not output.is_symlink() and not any(output.iterdir()):
            archived = output.with_name(output.name + "_empty_failed_" + time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12])
            output.rename(archived)
            print("Empty failed smoke directory preserved: " + str(archived), flush=True)
        else:
            raise ValueError("Output exists; preserve and inspect: " + str(output))
    metadata = bench / "02_queries/validation_full_ge10.tsv"
    fasta = bench / "02_queries/validation_full_ge10.fasta"
    hashes_before = {str(p): sha(p) for p in (metadata, fasta, Path(__file__))}
    frame = pd.read_csv(metadata, sep="\t", dtype={"seq_uid": str, "sequence": str})
    if len(frame) != 2442 or frame.seq_uid.isna().any() or not frame.seq_uid.is_unique:
        raise ValueError("Unexpected common validation cohort")
    if not frame.sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{10,65}").all():
        raise ValueError("Invalid frozen sequences; no trimming is allowed")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Invalid frozen labels")
    if fasta_records(fasta) != list(zip(frame.seq_uid, frame.sequence)):
        raise ValueError("Frozen FASTA/TSV ID or sequence mismatch")
    frame = frame.head(10).copy()
    checkpoint = args.resources / (BACKBONE + ".pt")
    fetch_backbone(checkpoint, args.download_attempts)
    output.mkdir(parents=True)
    backbone_hash = sha(checkpoint)
    print("Loading released ESM-1v backbone in FP32; no AMP training", flush=True)
    # Explicit safe allow-list for the old official ESM argparse metadata.
    with torch.serialization.safe_globals([argparse.Namespace]):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder, alphabet = esm.pretrained.load_model_and_alphabet_core(BACKBONE, data)
    del data
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; check selected GPU / environment")
    device = torch.device(args.device)
    print("Device: " + (torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"), flush=True)
    encoder = encoder.float().eval().to(device)
    converter = alphabet.get_batch_converter()

    def embedding(identifier, sequence):
        _, _, tokens = converter([(identifier, sequence)])
        with torch.inference_mode():
            result = encoder(tokens.to(device), repr_layers=[33], return_contacts=False)
        # Official fair-esm residue mean: exclude BOS, EOS, and padding.
        return result["representations"][33][0, 1:len(sequence) + 1].mean(0).cpu()

    references = dict(fasta_records(root / "train_amp.fasta"))
    references["L01A000044"] = "GIGAAILSAGKSIIKGLANGLAEHF"
    audits = []
    for identifier in ("DBAASPN_10400", "DBAASPN_10401", "L01A000044"):
        native = torch.load(root / (identifier + ".pt"), map_location="cpu", weights_only=True)
        if set(native["mean_representations"]) != {33}:
            raise ValueError("Official reference is not a single layer-33 mean")
        expected = native["mean_representations"][33].float()
        actual = embedding(identifier, references[identifier])
        passed = actual.shape == expected.shape == (1280,) and torch.allclose(
            actual, expected, atol=2e-4, rtol=1e-4
        )
        audits.append({"id": identifier, "max_abs_difference": (actual - expected).abs().max().item(),
                       "passed": bool(passed), "atol": 2e-4, "rtol": 1e-4})
        print("Reference feature parity: " + str(audits[-1]), flush=True)
    write_json(output / "reference_feature_audit.json", audits)
    if not all(audit["passed"] for audit in audits):
        raise ValueError("Reference features differ. STOP: do not benchmark or select settings using test scores.")
    formulas = property_functions(root / "charge.py")

    def properties(sequence):
        analysis = ProteinAnalysis(sequence)
        return [formulas["calculate_hydrophobicity"](sequence), round(analysis.charge_at_pH(7.4)),
                analysis.isoelectric_point(), formulas["calculate_amphipathicity"](sequence)]

    sheet = pd.read_excel(root / "testLAMP2.xlsx")
    reference = sheet[sheet.iloc[:, 0].astype(str).str.split("|").str[0] == "L01A000044"]
    if len(reference) != 1 or reference.iloc[0, 1] != references["L01A000044"]:
        raise ValueError("Official reference sequence / property row mismatch")
    if not np.allclose(properties(references["L01A000044"]), reference.iloc[0, 2:6].to_numpy(float),
                       atol=1e-7, rtol=1e-7):
        raise ValueError("Physicochemical features do not match official reference")
    spec = importlib.util.spec_from_file_location("tsamp_official_model", root / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = module.MLP(1284, 256, 1)
    state = torch.load(root / "tsAMP-I.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(state, strict=True)
    head = head.float().eval().to(device)
    probabilities = []
    start = time.monotonic()
    for record in frame.itertuples():
        vector = torch.cat((embedding(record.seq_uid, record.sequence),
                            torch.tensor(properties(record.sequence), dtype=torch.float32)))
        if vector.shape != (1284,) or not torch.isfinite(vector).all():
            raise ValueError("Invalid feature vector")
        with torch.inference_mode():
            probabilities.append(torch.sigmoid(head(vector.to(device)[None])).item())
        print("Smoke prediction %d/10" % len(probabilities), flush=True)
    if not np.isfinite(probabilities).all() or not all(0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid probabilities; never replace missing predictions with zero")
    verify_resources(root)
    if sha(checkpoint) != backbone_hash or any(sha(Path(p)) != h for p, h in hashes_before.items()):
        raise ValueError("Input, wrapper, or backbone changed during prediction")
    predictions = frame[["seq_uid", "sequence", "label"]].copy()
    predictions["probability"] = probabilities
    predictions["prediction"] = (predictions.probability > 0.5).astype(int)
    destination = output / "normalized_predictions.tsv"
    predictions.to_csv(destination, sep="\t", index=False, float_format="%.17g")
    write_json(output / "run_manifest.json", {
        "tool": "tsAMP-I", "repository_commit": COMMIT, "mode": "smoke10_only",
        "adapter_not_original_cli": True, "n_samples": 10, "backbone": BACKBONE,
        "backbone_sha256": backbone_hash, "resources_sha256": {n: h for n, (_, h) in RESOURCES.items()},
        "input_sha256": hashes_before, "normalized_output_sha256": sha(destination),
        "reference_feature_audit_sha256": sha(output / "reference_feature_audit.json"),
        "property_order": ["Normalized Hydrophobicity", "Net Charge", "Isoelectric Point", "Amphipathicity"],
        "reference_property_parity_passed": True, "precision": "float32, TF32 disabled",
        "decision_rule": "sigmoid > 0.5, from released training code; validation calibration not performed",
        "prediction_seconds": time.monotonic() - start,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "fair-esm", "biopython", "numpy", "pandas")},
        "limitation": "Adapter smoke test only; not paper metric reproduction or a training-overlap audit",
    })
    print(predictions.to_string(index=False), flush=True)
    print("TSAMP SMOKE COMPLETE: 10 queries; reference features / properties verified; no retraining", flush=True)
    print("Manifest: " + str(output / "run_manifest.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "download-backbone", "smoke"))
    parser.add_argument("--project", type=Path, default=Path("/home/omics/jinzy/ai-peptide"))
    parser.add_argument("--resources", type=Path, default=Path("/home/omics/jinzy/software/models/tsAMP-I_released_82c72209"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--download-attempts", type=int, default=6)
    parser.add_argument("--archive-empty-smoke", action="store_true",
                        help="Preserve/rename only an empty failed smoke directory; never archive populated results")
    args = parser.parse_args()
    root = args.resources / "official"
    if args.mode == "prepare":
        root.mkdir(parents=True, exist_ok=True)
        for name, (relative, expected) in RESOURCES.items():
            fetch(BASE + relative, root / name, expected)
        verify_resources(root)
        print("PREPARE COMPLETE: pinned official source, classifier and reference files verified.")
        print("Backbone is not downloaded in prepare mode.")
    elif args.mode == "download-backbone":
        # Standalone mode also works on the user's Windows E: drive, without
        # Torch or the server-side reference dataset. It only downloads bytes.
        fetch_backbone(args.resources / (BACKBONE + ".pt"), args.download_attempts)
    else:
        verify_resources(root)
        smoke(args, root)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit("STOP: " + str(error))
