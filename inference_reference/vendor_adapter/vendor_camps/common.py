"""CPU-only recovery diagnostics; Python 3.6 compatible. GPL-3.0-or-later."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AA = "ACDEFGHIKLMNPQRSTVWY"
SEQUENCES = [
    ("diag_01", "GIGKFLHSAKKFGKAFVGEIMNS"),
    ("diag_02", "GLFDIVKKVVGALGSL"),
    ("diag_03", "MKWVTFISLLFLFSSAYS"),
    ("diag_04", "ACDEFGHIKLMNPQRSTVWY"),
    ("diag_05", "AAAAAAAAAAAAAAAAAAAA"),
    ("diag_06", "EEEEEEEEEEEEEEEEEEEE"),
    ("diag_07", "KKLLKKLLKKLLKKLLKKLL"),
    ("diag_08", "GGGGSGGGGSGGGGSGGGGS"),
]
WEIGHTS = {
    "lstm": ("/home/omics/jinzy/software/envs/c_AMPs/lstm.h5",
             "b5c8fbcca9859f5982ff72b255be12ab752ef5151ed4b65402e58fbf8363dc3a"),
    "attention": ("/home/omics/jinzy/software/envs/c_AMPs/att.h5",
                  "3c8279f0bc8cf56f23a02c0e22ce6f4cc62577c33b2c661d4a3917a3f1bed808"),
    "bert": ("/home/omics/jinzy/software/models/c_AMPs_recovery/bert.bin",
             "83d53e596c4c59a3997738fef19fd48722a70591be34c25269e185407497bcda"),
}


def digest(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_weight(name):
    path, expected = WEIGHTS[name]
    if digest(path) != expected:
        raise RuntimeError("Weight SHA256 mismatch: " + name)
    return path


def encode_keras(seq):
    if not seq or any(c not in AA for c in seq) or len(seq) > 299:
        raise ValueError("Invalid / overlength Keras sequence; no truncation allowed")
    # Author format.pl: initial sentinel 0 followed by AA codes; left pad to 300.
    return [0] * (300 - len(seq)) + [AA.index(c) + 1 for c in seq]


def hard_and(probs):
    return int(all(float(p) > 0.5 for p in probs))


def write_json(path, value):
    with open(str(path), "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def check_probs(values):
    if len(values) != len(SEQUENCES):
        raise RuntimeError("Prediction coverage mismatch")
    values = [float(p) for p in values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise RuntimeError("Non-finite or out-of-range probabilities")
    if max(values) - min(values) < 1e-7:
        raise RuntimeError("Suspicious constant output across diverse diagnostic sequences")
    return values


def save_predictions(out, model, values, details):
    values = check_probs(values)
    rows = [{"sequence_id": uid, "sequence": seq, "probability": p}
            for (uid, seq), p in zip(SEQUENCES, values)]
    write_json(Path(out) / (model + ".json"), {"rows": rows, "details": details})


def prepare(out):
    out = Path(out)
    for name in WEIGHTS:
        checked_weight(name)
        print("WEIGHT VERIFIED:", name, flush=True)
    write_json(out / "protocol.json", {
        "version": "camps_recovery_smoke_v1.0", "device": "cpu",
        "purpose": "runtime and encoding diagnostics ONLY; not benchmark or accuracy evaluation",
        "source_commit": "cf7658bc5d504ba6d996fa7b152270e38275dc46",
        "n": len(SEQUENCES), "diagnostic_sequences": SEQUENCES,
        "weights": WEIGHTS, "hard_decision": "all three probabilities strictly > 0.5",
        "continuous_ensemble_score": "not defined in this smoke package",
        "inference_math": "author vendored BERT and Keras Attention; no training",
        "changes": ["offline local asset loader", "Attention.py indentation repair only",
                    "compile=False for Keras", "strict BERT state_dict loading",
                    "no singleton np.squeeze; explicit aligned sequence IDs"],
    })


def merge(out):
    out = Path(out)
    tables = {}
    for name in WEIGHTS:
        with open(str(out / (name + ".json")), encoding="utf-8") as handle:
            obj = json.load(handle)
        if [(r["sequence_id"], r["sequence"]) for r in obj["rows"]] != SEQUENCES:
            raise RuntimeError("Sequence ID / order mismatch: " + name)
        tables[name] = check_probs([r["probability"] for r in obj["rows"]])
    fields = ["sequence_id", "sequence", "lstm", "attention", "bert", "AMP_hard_AND"]
    with open(str(out / "smoke_predictions.tsv"), "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for i, (uid, seq) in enumerate(SEQUENCES):
            row = {"sequence_id": uid, "sequence": seq}
            row.update({name: tables[name][i] for name in tables})
            row["AMP_hard_AND"] = hard_and([row[k] for k in WEIGHTS])
            writer.writerow(row)
    for name in WEIGHTS:
        checked_weight(name)
    files = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "launcher.log"}
    write_json(out / "SMOKE_COMPLETE.json", {
        "version": "camps_recovery_smoke_v1.0", "smoke_complete": True,
        "benchmark_complete": False, "official_legacy_runtime_parity_proven": False,
        "scope": "three author architectures loaded and ran on 8 diagnostic sequences",
        "files": files,
    })
    print("C_AMPS CPU SMOKE COMPLETE (NOT SIX-DATASET BENCHMARK)", flush=True)
    print("OUTPUT:", out, flush=True)


if __name__ == "__main__":
    import sys
    {"prepare": prepare, "merge": merge}[sys.argv[1]](sys.argv[2])
