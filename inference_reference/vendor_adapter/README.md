# Published AMP tools: six tools × six frozen datasets, v1.2

Repair: the native AMPlify TSV mean validation uses a declared absolute 5e-7
FP32/output-serialization tolerance instead of 2e-7. Original probabilities,
five-submodel scores, native class decisions, model weights, official source and
all test sequences remain unchanged. This is not training or threshold tuning.
The reported failing row had a 2.16e-7 difference, with both scores >0.9985.
Larger inconsistencies still stop the run. A per-chunk validation audit is saved.

Recovery verifies the pinned v1.1 code manifest, frozen plan/query/protocol,
completion markers, artifact hashes, installed weights and ID-aligned predictions.
Only committed chunks are copied to the new v1.2 result tree with source receipts;
the old result tree is never changed. A bare native TSV in a failed attempt is
NOT imported. The reported 128/133 committed chunks should be reused, leaving
5 AMPlify CPU chunks (9,628 queries) to rerun, not all 36 experiments.
When all GPU components are already committed, no GPU is allocated or required;
the remaining AMPlify predictions run on CPU. Old restrictive CUDA masks are not
widened. CPU allocation, host RAM, existing environments and integrity are checked.

## Scope

Inference with released Macrel1.6.1, AMPlify2.0.1 balanced, ampir1.1.0 mature,
tsAMP-I, HMD-AMP and complete c_AMPs (Attention+LSTM+BERT). No fitting, no new
splits, no representation extraction and no old output overwrites. Six sources
are the already-completed external_plm_holdout_v2.2 cohorts:
AMPlify, tsAMP_I, c_AMPs_benchmark_resplit, HMD_AMP, AmPEP_raw_source and
ampir_raw_mature90. The last is NOT the recovered ampir official v1.1 split.

New output: /home/omics/jinzy/ai-peptide/runs/published_amp_six_by_six_v1_2_20261010
Original source: /home/omics/jinzy/ai-peptide/runs/external_plm_holdout_v2_2_20261007
Frozen source manifest SHA256:
b8499c6988678a74cd8045bc2c0f9cd8adc8b5227977d0a4bf5b1a1153b76d34
If that fingerprint differs, stop and inspect; don't change it just to bypass verification.

## Launch

Upload the ZIP into /home/omics/jinzy/ai-peptide/tools, verify it, extract without
overwriting an existing checkout, then:

    bash launch_server.sh --check-only
    bash launch_server.sh --confirm-resources

If GPU components are missing, this uses your previously authorized GPU4–7 and16CPU allocation only. GPU4 runs
tsAMP, GPU5 HMD, GPU6+7 two disjoint BERT queues. Three CPU lanes run Macrel then
ampir, AMPlify, and c_AMPs LSTM/Attention respectively. After a GPU's primary queue
is exhausted, it can help the remaining tsAMP/HMD/BERT queues; shared claims prevent
duplicate inference or cache writes. No GPU outside4–7 is added. Each lane has2 CPU affinity
IDs,14 total worker cores within the16-core inherited scope. No sudo, killing,
package installation, pretrained downloading or environment upgrading occurs.
Busy GPUs, restrictive inherited CUDA allocation or insufficient CPU/RAM cause
a stop. UUID masks are used after resolving physical indices. Resources are
checked again immediately before each GPU worker starts.

Existing tested adapters from amp_amplify_ml_tools_v1 are vendored unchanged;
their old mains/old cohort selection are not run. Existing environment/model
versions and identity hashes are checked during every chunk. c_AMPs uses the
exact vendor source of the passed camps_recovery_smoke_v1 package. The saved
CPU smoke is verified at the recorded server path. Every Keras chunk compares
diagnostic predictions to CPU smoke; every BERT GPU chunk checks FP32, strict
tensor loading, proper positive column, no UNK/truncation, and GPU/CPU smoke parity.
Keras scores are formatted to8 decimals as in the author's prediction scripts.
The reference is NOT proof of exact equivalence to the original2021 runtime.

Queries are deduplicated per-tool across datasets, scored once in small committed
chunks, then mapped back to each dataset's original fixed TEST. Cross-source
duplicate sequences must have consistent labels/groups. This saves redundant
inference but does not combine training sets or change per-dataset membership.

## Resume and monitor

    bash launch_server.sh --status
    tail -n40 -F scheduler_console.log
    tmux capture-pane -p -t amp_tools_six_by_six_v1_2 -S -40

status.json lists active chunk log paths and per-component committed query counts.
A crashed/uncommitted chunk remains in an attempt folder. Run the same scheduler
again after the prior run ends to verify and reuse committed chunks. An idle tmux
shell still exists after completion/failure, so session existence is not completion.
Do not create simultaneous launchers. If the old session is idle, attach and run
`bash run_server.sh` there; no need to delete the session. Output locks prevent races.

Definitive completion requires experiment_complete.json complete=true and36 cells
with verified prediction files, and status.json status=complete. Runtime errors do
not count as negative predictions. Unsupported samples are listed as exclusions.

## Reports

05_reports/final_results.tsv has3 panels ×36 cells =108 rows:

1. tool_supported: each tool's prespecified adapter support; unequal denominators,
   not a fair head-to-head ranking.
2. common_six_tools: exactly the same supported TEST IDs for all six tools in each
   dataset; by current adapter constraints this is10..126 aa.
3. common_short_10_49: the same IDs further restricted to10..49 aa, a domain
   sensitivity panel, not an assertion that all training domains are identical.

Each panel gets separate model-row × dataset-column MCC/F1/recall/AP/AUROC/accuracy
matrices. Confusion counts, positive/negative denominators and exclusions accompany
all results. Native labels are retained; ampir uses the previously declared0.5
benchmark reference, not an author-universal cutoff. Macrel's native leading-M
removal and native rounded scores are retained by its verified adapter.

c_AMPs official inference is hard AND (all3 scores strictly>0.5). There is no
author-defined continuous ensemble probability. Thus its primary AP/AUROC fields
are blank, NOT zero. Submodel scores are retained per sequence. A separate table,
camps_min_score_extension.tsv, reports min(LSTM,Attention,BERT) as an explicitly
named recovery ranking extension. It has the same >0.5 hard boundary but is NOT
the author's original ensemble probability or a calibrated probability.

training_source_match_summary.tsv reports exact matches to supplied source scopes;
Macrel rawAmPEP and ampir rawmature90 are not confirmed final release training
membership. HMD/tsAMP source scopes can also be incomplete. c_AMPs original TRAIN
membership is unknown. No lack of matches proves no training/homology overlap.

## Scientific limitations

This is a systematic exploratory frozen-source evaluation, not an independent
released-tool leaderboard. Some cells are resplits of training pools. Six datasets
share sequences;36 cells are not independent experiments. Presumed-negative source
and M-start bias remain. No threshold is chosen using TEST. Long peptides in mature
models are extrapolation; c_AMPs computational acceptance up to126 aa does not
establish biological deployment validity. The paper needs independent, preferably
homology-controlled evaluation plus uncertainty analysis before superiority claims.

Local validation covers synthetic protocol, partition, alignment, resume, metrics
and GPU-allocation resolution logic. No real server36-cell inference was run locally.

License: GPL-3.0-or-later for recovery/orchestration code. Vendor c_AMPs license and
upstream Apache notices are included. Author sources are packaged unchanged.
