# Systematic evaluation of released antimicrobial peptide predictors

## Release v1.3.0

This version integrates the six-main-figure evaluation, paired sequence baselines and family diagnostics, inference provenance, linked reliability analysis, and the 2,000-draw joint auxiliary-selector refitting update. It publishes existing calculations; no new AMP inference or wet experiment is claimed. Scientific labels, budgets, protocols, estimators and saved result values remain fixed. Historical/reference letters are replaced by verified hashes and lengths for saved-feature statistics.

## Start here

Use Python 3.12 in an isolated environment. The original main-figure module retains its own requirements. For the added reliability/refit modules install requirements-reliability.txt. The recorded analysis environment is not a portable AMP-model inference environment.

```bash
python -m pip install -r requirements-reliability.txt
python release.py check
python release.py selftest
python release.py main-points --output ../check_main_points
python release.py main-figures --output ../check_main_figures
python release.py reliability-figures --output ../check_reliability_figures
python release.py refit-figure --output ../check_updated_S29
```

Output directories must not already exist. Figure commands use saved estimates and intervals. To repeat numerical calculations without AMP inference:

```bash
python release.py reliability-statistics --output ../recompute_reliability
python release.py refit-smoke --output ../check_refit_smoke
python release.py refit-statistics --workers 3 --output ../recompute_refit
```

The smoke test has only 20 draws and does not validate formal results. The formal refit uses all 2,000 draws per fold. Reliability statistics uses the original 20 partitions and 10,000-draw conditional intervals. The original six-main-figure commands for baseline, transfer and independent uncertainty are documented in six_main_figures_v1_1_20261101/README.md.

## Figure and table map

| Manuscript object | Authoritative source / entry |
|---|---|
| Main Figures 1–6 | six_main_figures_v1_1_20261101; release.py main-points / main-figures |
| S28 high-confidence diagnostics | reliability; A_high_confidence_error_map from reliability-figures |
| Updated S29 and S16 | refit/data and refit/results/refit_formal; refit-figure / refit-statistics |
| S30 and S17 | reliability experimental retention tables; C_AMPlify_experimental_selection_utility |
| Earlier conditional-only distance intervals | reliability/results, supporting comparison only; not the updated manuscript S29 |
| Inference adapters / model identities | inference_reference/README.md and MODEL_ENVIRONMENT_REGISTRY.tsv |

Exact exported filenames are recorded in docs/RELEASE_FIGURE_REGISTRY.tsv. All source/model cases, omitted intervals and unavailable states remain represented. Repeated partitions are sensitivity, not independent external replications. Distance inventories are source-qualified, not complete proof of fitted-training independence.

## Data access and licences

Saved-feature statistics is self-contained, including the six development selection/distance feature files, hashed sequence index and nearest-reference features. Original letters are required for distance reconstruction, similarity grouping and new AMP inference. Read docs/RELIABILITY_DATA_ACCESS.md and docs/SOURCES.md for exact source routes, row identities and fingerprint checks. Raw historical sequence letters and complete third-party reference inventories are not redistributed in this release. Source article PDFs/workbooks and trained model weights are not bundled.

Project-authored code uses Apache-2.0. Reused components retain their licences, including the GPL c_AMPs vendor subtree. Specified processed statistical outputs and compilations use CC BY 4.0 with source attribution; third-party source rights remain in force. See LICENSE, NOTICE and docs/LICENSING.md. The version-specific DOI must be taken from the successfully verified Zenodo record; no DOI is preassigned here. Earlier version history is retained in the release history files and Git tags.
