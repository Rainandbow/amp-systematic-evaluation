#!/usr/bin/env python3
"""Freeze the already-trained ESMC pipeline and compare released Macrel weights.

prepare: select by mean Validation Full AP only, preserve all 5 seeds, export
label-free query FASTA and labelled metadata for common 10..65-aa cohorts.
compare-macrel: strict full-coverage score comparison; never fit thresholds.
audit-macrel: verify an existing native output without changing any files.
"""
import argparse
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


def load(path, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checksums(directory, runner):
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "checksums_sha256.txt")
    (directory / "checksums_sha256.txt").write_text(
        "".join(f"{runner.sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in paths), encoding="utf-8")


def verify_preparation(bench, runner):
    for line in (bench / "checksums_sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = (bench / name).resolve()
        if bench.resolve() not in path.parents or runner.sha(path) != digest:
            raise ValueError(f"Prepared benchmark file changed: {name}")


def cohort_counts(frame):
    return {"n": len(frame), "positive": int((frame.label == 1).sum()), "negative": int((frame.label == 0).sum())}


def prepare(args, run, bench, runner, helper):
    if bench.exists() and any(bench.iterdir()):
        raise FileExistsError(f"Benchmark directory is nonempty: {bench}. No existing results were overwritten.")
    comparison = run / "model_comparison/esmc_four_architectures_multiseed"
    registry_path = comparison / "all_20_runs.tsv"
    registry = pd.read_csv(registry_path, sep="\t")
    pairs = list(zip(registry.kind, registry.seed))
    if len(registry) != 20 or len(set(pairs)) != 20 or set(pairs) != {(kind, seed) for kind in runner.METHODS for seed in runner.SEEDS}:
        raise ValueError("Need all four ESMC pipelines at all five fixed seeds")
    means = registry.groupby("kind").validation_full_auprc.mean().sort_values(ascending=False, kind="stable")
    selected_kind = str(means.index[0])
    # Current requested residue-model release. Refuse silently switching architectures.
    if selected_kind != "mean":
        raise ValueError(f"Validation winner is {selected_kind}, not residue mean; inspect the selection before preparing this release.")
    frames = runner.expected_frames(run)
    selected_rows = registry.loc[registry.kind.isin([selected_kind, "linear"])].copy()
    audited_rows, assets = [], []
    reference_config = None
    model_source = args.project / "scripts/04_classification/03_train_esmc6b_attention_pool.py"
    helper_source = args.project / "scripts/04_classification/02_train_esmc6b_mlp.py"
    for record in selected_rows.itertuples(index=False):
        directory = Path(record.workdir)
        current = runner.audit_result(directory, record.kind, int(record.seed), frames, helper)
        for name in ("selected_threshold", *(f"{s}_{m}" for s in runner.SETS for m in runner.MEASURES)):
            if not np.isclose(float(getattr(record, name)), current[name], atol=2e-6, rtol=0):
                raise ValueError(f"Stale training summary: {directory} / {name}")
        audited_rows.append(current)
        if record.kind == selected_kind:
            recorded_hashes = {}
            for line in (directory / "06_reproducibility/checksums_sha256.txt").read_text(encoding="utf-8").splitlines():
                digest, relative_name = line.split("  ", 1)
                recorded_hashes[relative_name] = digest
            for relative in ("02_classifier/attention_head.pt", "02_classifier/training_config.json", "04_reports/metrics.json"):
                path = directory / relative
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(path)
                if recorded_hashes.get(relative) != runner.sha(path):
                    raise ValueError(f"Trained artifact does not match its archived checksum: {path}")
                release_name = "classifier.pt" if path.name == "attention_head.pt" else path.name
                assets.append((path, Path(f"01_our_model/seed{int(record.seed)}") / release_name))
            config = runner.read_json(directory / "02_classifier/training_config.json")
            if (config.get("pooling") != "mean" or config.get("input_dimension") != 2560
                    or config.get("dataset") != runner.DATASET or int(config.get("seed", -1)) != int(record.seed)
                    or config.get("projection_dimension") != 128 or config.get("classifier_dimensions") != [128, 64, 1]):
                raise ValueError(f"Unexpected residue mean model: {directory}")
            for path in (model_source, helper_source):
                archived = directory / "06_reproducibility" / path.name
                if not archived.is_file() or runner.sha(archived) != runner.sha(path):
                    raise ValueError(f"Model source differs from archived training source: {path}")
            for key, path in (("input_sha256", run / "01_splits/full_clustered_v2.tsv"),
                              ("matched_test_sha256", run / "01_splits/test_matched.tsv")):
                if config.get(key) != runner.sha(path):
                    raise ValueError(f"Training input changed: {directory} / {key}")
            invariant_keys = ("pooling", "input_dimension", "projection_dimension", "classifier_dimensions",
                              "dropout", "epochs", "patience", "batch_size", "learning_rate", "weight_decay",
                              "normalization", "masking", "positive_weight", "residue_file_sha256")
            invariant = {key: config[key] for key in invariant_keys}
            if reference_config is not None and invariant != reference_config:
                raise ValueError("Residue mean pipeline settings differ between seeds")
            reference_config = invariant
    cohorts = {}
    for name, frame in frames.items():
        sequence = frame.sequence.astype(str)
        if not sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{5,65}").all():
            raise ValueError(f"Noncanonical or out-of-range sequence in {name}")
        core = frame.loc[sequence.str.len().between(10, 65)].copy()
        if core.empty or core.label.nunique() != 2:
            raise ValueError(f"Common cohort requires both classes: {name}")
        cohorts[name] = core
    bench.mkdir(parents=True, exist_ok=True)
    for source, relative in assets:
        target = bench / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if runner.sha(source) != runner.sha(target):
            raise ValueError(f"Copy verification failed: {source}")
    for source in (model_source, helper_source):
        shutil.copy2(source, bench / "01_our_model" / source.name)
    # These are reference/single-seed artifacts, not a new five-seed probability ensemble.
    model_manifest = {
        "model": "ESMC-6B + Residue Mean Pool + MLP", "dataset": runner.DATASET,
        "selected_by": "Highest mean Validation Full average precision across four pipelines, five training seeds",
        "validation_full_auprc_means": {str(k): float(v) for k, v in means.items()},
        "reference_seed": 20260927, "reference_seed_rule": "The original predeclared training seed; NOT best Test seed",
        "classifier_checkpoint_per_seed": "classifier.pt (copied from the original residue-mean attention_head.pt filename)",
        "seeds": list(runner.SEEDS), "encoder": "Frozen ESMC-6B, final-layer residue embeddings",
        "architecture": "Per-residue LayerNorm2560 -> Linear128 -> GELU -> mean pool -> Linear64 -> GELU -> Linear1",
        "special_tokens_and_padding": "Excluded before pooling", "dropout_at_training": 0.3,
        "encoder_reference_only": str(Path.home() / "jinzy/software/models/ESMC-6B"),
        "output": "AMP score; not an experimentally calibrated probability",
        "claim": "Already-trained checkpoint archive, not a new training run or evidence of superiority",
    }
    runner.write_json(bench / "01_our_model/model_manifest.json", model_manifest)
    query_dir = bench / "02_queries"
    query_dir.mkdir()
    counts = []
    for name, core in cohorts.items():
        core.to_csv(query_dir / f"{name}_ge10.tsv", sep="\t", index=False)
        with (query_dir / f"{name}_ge10.fasta").open("w", encoding="utf-8") as handle:
            for record in core.itertuples(index=False):
                handle.write(f">{record.seq_uid}\n{record.sequence}\n")
        excluded = frames[name].loc[~frames[name].seq_uid.isin(core.seq_uid)]
        excluded.to_csv(query_dir / f"{name}_short5to9.tsv", sep="\t", index=False)
        counts.append({"cohort": name, **cohort_counts(core), "excluded_short": len(excluded)})
    pd.DataFrame(counts).to_csv(query_dir / "cohort_counts.tsv", sep="\t", index=False)
    prediction_dir = bench / "03_our_predictions"
    prediction_dir.mkdir()
    for record in selected_rows.itertuples(index=False):
        directory = Path(record.workdir)
        for name, core in cohorts.items():
            full = helper.align_predictions(directory / f"03_predictions/{name}_predictions.tsv", frames[name])
            subset = full.set_index("seq_uid").loc[core.seq_uid].reset_index()
            subset.to_csv(prediction_dir / f"{record.kind}_seed{int(record.seed)}_{name}_ge10.tsv", sep="\t", index=False)
    pd.DataFrame(audited_rows).to_csv(bench / "frozen_run_registry.tsv", sep="\t", index=False)
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": runner.DATASET,
        "primary_method": runner.METHODS[selected_kind], "reference_seed": 20260927,
        "cohorts": counts, "common_length_range": [10, 65],
        "threshold_policy": "Keep each own model's previously Validation-Full-selected threshold; official tools use released defaults",
        "threshold_fitting_on_test": False, "tool_weights_policy": "Use released weights unchanged; no retraining on this dataset",
        "label_definition": "Broad AMP positive; UniProt negatives are presumed negatives, not universal experimental inactivity",
        "evaluation_status": "Exploratory deployment comparison on an already-inspected internal holdout, NOT fresh external validation",
        "published_tool_training_overlap": "NOT YET AUDITED; no homology-independent generalization claim allowed",
        "future_external_benchmark_requirement": "Independent data, harmonized labels, audit overlap/homology against own AND tool training sets",
        "query_headers": "Only seq_uid; no labels supplied to tools", "no_sequence_padding_or_mutation": True,
        "input_sha256": {str(registry_path): runner.sha(registry_path)},
        "planned_tools": {
            "Macrel": "https://github.com/BigDataBiology/macrel",
            "AMPlify": "https://github.com/BirolLab/AMPlify",
            "AMP_Scanner_v2": "https://github.com/dan-veltri/amp-scanner-v2",
        },
    }
    runner.write_json(bench / "benchmark_protocol.json", protocol)
    shutil.copy2(Path(__file__), bench / Path(__file__).name)
    checksums(bench, runner)
    print("\nPREPARATION COMPLETE: already-trained models frozen; no new training or ensembling.")
    print(pd.DataFrame(counts).to_string(index=False))
    print(f"\nBenchmark directory: {bench}")


def macrel_normalize(sequence):
    """Official Macrel normalize_seq: remove ONE initial M and terminal *."""
    if not sequence:
        raise ValueError("Empty input sequence")
    sequence = sequence[1:] if sequence.startswith("M") else sequence
    sequence = sequence[:-1] if sequence.endswith("*") else sequence
    if not sequence:
        raise ValueError("Empty sequence after official Macrel normalization")
    return sequence


def macrel_frame(path, expected):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if first_line != "# Prediction from macrel v1.6.1":
        raise ValueError(f"Expected the pinned Macrel v1.6.1 release, got: {first_line}")
    frame = pd.read_csv(path, sep="\t", comment="#", compression="infer", dtype={"Access": str, "Sequence": str})
    if not {"Access", "Sequence", "AMP_probability"}.issubset(frame.columns):
        raise ValueError(f"Unsupported Macrel columns: {list(frame.columns)}")
    if expected.seq_uid.isna().any() or not expected.seq_uid.is_unique:
        raise ValueError("Expected sample IDs must be nonempty and unique")
    expected_ids = expected.seq_uid.astype(str)
    if frame.Access.isna().any() or not frame.Access.is_unique:
        raise ValueError("Macrel Access IDs are missing or duplicated; cannot safely align records")
    if len(frame) != len(expected) or set(frame.Access) != set(expected_ids):
        missing = len(set(expected_ids) - set(frame.Access))
        extra = len(set(frame.Access) - set(expected_ids))
        raise ValueError(f"Macrel ID coverage mismatch: missing={missing}, extra={extra}. Need complete --keep-negatives output; never fill missing scores with zero.")
    # Peptide-mode Access retains the FASTA ID. Processed sequences may collide,
    # e.g. MACDE... and ACDE..., so never deduplicate or join by sequence alone.
    aligned = frame.set_index("Access").loc[expected_ids].reset_index()
    processed = expected.sequence.astype(str).map(macrel_normalize).to_numpy()
    if aligned.Sequence.isna().any() or not np.array_equal(aligned.Sequence.to_numpy(), processed):
        raise ValueError("Macrel processed sequences differ from official normalization of their ID-matched inputs")
    probability = aligned.AMP_probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid Macrel AMP scores")
    if "is_AMP" not in aligned.columns:
        raise ValueError("Need Macrel 1.6.1 --keep-negatives output containing is_AMP; rounded scores alone cannot recover native decisions at 0.500.")
    boolean = aligned.is_AMP.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    if boolean.isna().any():
        raise ValueError("Invalid Macrel is_AMP field")
    result = expected[["seq_uid", "sequence", "label"]].copy()
    result["tool_processed_sequence"] = processed
    result["leading_M_removed"] = expected.sequence.astype(str).str.startswith("M").to_numpy()
    result["probability"] = probability
    result["prediction"] = boolean.to_numpy(dtype=int)
    contradiction = (result.prediction.to_numpy() != (probability > 0.5).astype(int)) & (np.abs(probability - 0.5) > 0.000501)
    if contradiction.any():
        raise ValueError("Macrel labels disagree with its native >0.5 rule beyond output rounding")
    return result


def audit_macrel(args, bench, runner):
    if args.cohort not in ("validation_full", "test_full"):
        raise ValueError("audit-macrel requires --cohort validation_full or test_full")
    verify_preparation(bench, runner)
    tools = args.macrel_dir or bench / "04_published_tools/Macrel"
    path = tools / args.cohort / "benchmark.prediction.gz"
    expected = pd.read_csv(bench / f"02_queries/{args.cohort}_ge10.tsv", sep="\t", dtype={"seq_uid": str})
    aligned = macrel_frame(path, expected)
    print(f"AUDIT PASSED: {args.cohort}; input={len(expected)}, predictions={len(aligned)}, missing IDs=0, extra IDs=0", flush=True)
    print(f"Official leading-M removals: {int(aligned.leading_M_removed.sum())}; all processed sequences verified", flush=True)
    print(f"Native prediction SHA256: {runner.sha(path)}; no existing files changed", flush=True)


def predict_macrel(args, bench, runner):
    """Invoke the installed official CLI; only cap runtime threads, not model weights."""
    if args.cohort not in ("validation_full", "test_full"):
        raise ValueError("predict-macrel requires --cohort validation_full or test_full")
    verify_preparation(bench, runner)
    from macrel.macrel_version import __version__
    if __version__ != "1.6.1":
        raise ValueError(f"Expected Macrel 1.6.1, found {__version__}")
    tools = args.macrel_dir or bench / "04_published_tools/Macrel"
    query = bench / f"02_queries/{args.cohort}_ge10.fasta"
    expected = pd.read_csv(bench / f"02_queries/{args.cohort}_ge10.tsv", sep="\t", dtype={"seq_uid": str})
    output = tools / args.cohort
    if output.exists():
        raise FileExistsError(f"Tool output directory already exists: {output}. Do not use --force; first inspect whether it is complete.")
    temporary = tools / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    from macrel.main import main as macrel_main
    original_session = ort.InferenceSession

    def limited_session(model, sess_options=None, **kwargs):
        options = sess_options or ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return original_session(model, sess_options=options, **kwargs)

    original_argv = sys.argv
    ort.InferenceSession = limited_session
    sys.argv = ["macrel", "peptides", "--fasta", str(query), "--output", str(output),
                "--keep-negatives", "--threads", "2", "--tag", "benchmark", "--tmpdir", str(temporary)]
    print("Running official Macrel 1.6.1; ONNX intra-op=2, inter-op=1; keep negatives; no weight changes", flush=True)
    try:
        macrel_main()
    finally:
        sys.argv = original_argv
        ort.InferenceSession = original_session
    table = output / "benchmark.prediction.gz"
    aligned = macrel_frame(table, expected)
    import macrel.AMP_predict as native_predict
    runner.write_json(output / "run_manifest.json", {
        "version": __version__, "cohort": args.cohort, "query_sha256": runner.sha(query),
        "native_predict_source_sha256": runner.sha(Path(native_predict.__file__)),
        "prediction_sha256": runner.sha(table), "n_samples": len(expected),
        "runtime_limits": "ONNX intra-op 2, inter-op 1, sequential; shell OMP/MKL/OPENBLAS limits 2",
        "official_cli_options": "peptides --keep-negatives --threads 2 --tag benchmark",
        "weights": "Official installed release, unmodified", "training_overlap_status": "not audited",
        "alignment": "Unique original FASTA Access IDs; processed sequences verified against official normalization",
        "leading_M_removed": int(aligned.leading_M_removed.sum()),
    })
    print(f"MACREL PREDICTION COMPLETE: {table}", flush=True)


def compare_macrel(args, bench, runner, helper):
    verify_preparation(bench, runner)
    out = bench / "05_comparisons/Macrel"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Comparison already exists: {out}; no overwrite attempted")
    registry = pd.read_csv(bench / "frozen_run_registry.tsv", sep="\t")
    tools = args.macrel_dir or bench / "04_published_tools/Macrel"
    raw_rows, frames = [], {}
    input_hashes = {}
    # Predict validation & Test Full only once; derive Test Matched from Full scores.
    for name in ("validation_full", "test_full"):
        expected = pd.read_csv(bench / f"02_queries/{name}_ge10.tsv", sep="\t", dtype={"seq_uid": str})
        path = tools / name / "benchmark.prediction.gz"
        frames[name] = macrel_frame(path, expected)
        input_hashes[str(path)] = runner.sha(path)
    expected_matched = pd.read_csv(bench / "02_queries/test_matched_ge10.tsv", sep="\t", dtype={"seq_uid": str})
    frames["test_matched"] = frames["test_full"].set_index("seq_uid").loc[expected_matched.seq_uid].reset_index()
    for name in runner.SETS:
        expected = pd.read_csv(bench / f"02_queries/{name}_ge10.tsv", sep="\t", dtype={"seq_uid": str})
        tool = frames[name]
        values = helper.evaluate(tool.label.to_numpy(), tool.probability.to_numpy(), 0.5, tool.prediction.to_numpy())
        raw_rows.append({"method": "Macrel released v1.6.1", "kind": "macrel", "seed": "released", "evaluation_set": name,
                         "threshold_rule": "Native score >0.5 (saved is_AMP)", **values})
        for record in registry.itertuples(index=False):
            path = bench / f"03_our_predictions/{record.kind}_seed{int(record.seed)}_{name}_ge10.tsv"
            own = helper.align_predictions(path, expected)
            values = helper.evaluate(own.label.to_numpy(), own.probability.to_numpy(), record.selected_threshold, own.prediction.to_numpy())
            raw_rows.append({"method": record.method, "kind": record.kind, "seed": int(record.seed), "evaluation_set": name,
                             "threshold_rule": "Previously Validation-Full-selected per seed", **values})
    raw = pd.DataFrame(raw_rows)
    if len(raw) != 33:
        raise ValueError("Expect (5 Mean + 5 Linear + 1 released Macrel) x 3 evaluation sets")
    rows = []
    for (kind, method, name), group in raw.groupby(["kind", "method", "evaluation_set"], sort=False):
        record = {"kind": kind, "method": method, "evaluation_set": name, "n_runs": len(group),
                  "n": int(group.n.iloc[0]), "positive": int(group.positive.iloc[0]), "negative": int(group.negative.iloc[0])}
        for metric in runner.MEASURES:
            record[metric + "_mean"] = float(group[metric].mean())
            record[metric + "_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else None
        rows.append(record)
    summary = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=False)
    raw.to_csv(out / "all_runs_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out / "comparison_summary.tsv", sep="\t", index=False)
    for name, frame in frames.items():
        frame.to_csv(out / f"macrel_{name}_predictions.tsv", sep="\t", index=False)
    runner.write_json(out / "comparison_protocol.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": input_hashes,
        "coverage": "Complete shared original-input 10..65-aa cohorts aligned by sample ID; Test Matched is a subset of Test Full predictions",
        "macrel_preprocessing": "Official native pipeline removes ONE initial M and terminal *; original canonical query files remain unchanged. Native processed lengths may be 9..65 aa. This is a released-pipeline comparison, not identical processed-residue input.",
        "alignment": "Access must equal original unique seq_uid; every processed sequence verified; normalization collisions retained as distinct IDs",
        "leading_M_removed_per_cohort": {name: int(frame.leading_M_removed.sum()) for name, frame in frames.items()},
        "comparison_script_sha256": runner.sha(Path(__file__)),
        "macrel_normalize_source": "https://github.com/BigDataBiology/macrel/blob/main/macrel/macrel_features.py",
        "native_thresholds": "Own: saved validation-selected thresholds; Macrel: released >0.5 and saved native decisions",
        "macrel_scores": "Official CLI rounds AMP scores to 3 decimals; native decisions retained; AP/ROC/Brier use these rounded scores",
        "n_runs": "Own methods: 5 separate trained heads, not ensembled; Macrel: one released predictor, seed SD unavailable",
        "limitations": "Broad labels may differ from published-tool labels; presumed negatives; training overlap NOT audited; previously viewed internal Test, not new external validation",
    })
    checksums(out, runner)
    print("\nCOMPARISON COMPLETE (exploratory; training overlap not yet audited)")
    columns = ["method", "evaluation_set", "n", "n_runs", "auprc_mean", "auroc_mean", "mcc_mean", "accuracy_mean", "sensitivity_mean", "specificity_mean"]
    print(summary[columns].to_string(index=False))
    print(f"\nReports: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "predict-macrel", "audit-macrel", "compare-macrel"))
    parser.add_argument("--project", type=Path, default=Path.home() / "jinzy/ai-peptide")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--macrel-dir", type=Path)
    parser.add_argument("--cohort", choices=("validation_full", "test_full"))
    args = parser.parse_args()
    run = args.run_root or args.project / "runs/amp_binary_broad_v3_1_0"
    bench = args.benchmark_root or run / "published_amp_benchmark/v1"
    # Never let --benchmark-root overlap training or dataset assets.
    actual = bench.resolve()
    for protected in (run / "models", run / "01_splits", run / "model_comparison"):
        protected = protected.resolve()
        if actual == protected or protected in actual.parents or actual in protected.parents:
            raise ValueError("Benchmark output overlaps existing model/dataset/comparison assets")
    runner = load(args.project / "scripts/04_classification/04_run_esmc6b_architecture_multiseed.py", "published_benchmark_runner")
    helper = load(args.project / "scripts/04_classification/02_train_esmc6b_mlp.py", "published_benchmark_helper")
    if args.mode == "prepare":
        prepare(args, run, bench, runner, helper)
    elif args.mode == "predict-macrel":
        predict_macrel(args, bench, runner)
    elif args.mode == "audit-macrel":
        audit_macrel(args, bench, runner)
    else:
        compare_macrel(args, bench, runner, helper)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"STOP: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
