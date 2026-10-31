# Systematic evaluation of released AMP predictors

Analysis and figure-generation code for six released antimicrobial peptide predictors: Macrel, AMPlify, ampir, tsAMP-I, HMD-AMP and c_AMPs.

This repository is being prepared for the manuscript. No public release or Zenodo DOI has been assigned yet.

## What can be run now

The included frozen statistical tables support reconstruction of main Figures 4 and 5 and the five baseline/family/validation supplementary figures (S20–S24 in the synchronized manuscript). They preserve 96 method–cohort–task points, 1,440 paired model–baseline contrasts, 1,800 selected-rule records and 5,400 phase-level evaluations.

The repository also includes the saved-output baseline/family and group-separated validation implementations, and historical reconstruction code retained as a reference. Full numerical recomputation requires the frozen sequence-level inputs described in `docs/DATA_ACCESS.md`. Original model weights, publication PDFs and source workbooks are not bundled.

## Install and check

Use Python 3.11 or later in an isolated environment:

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py test
python run.py figures
```

`check` verifies file hashes and formal-table alignment. `test` runs eleven synthetic numerical checks; it does not use peptide evidence. `figures` reconstructs seven figures from saved statistical results without model inference, threshold selection or bootstrap resampling. Each figure run creates a new directory under `figures/output_*`.

## Recompute the added analyses

After obtaining the exact frozen analysis inputs:

```bash
python tools/import_local_inputs.py --source /path/to/existing/baseline_honest_package.zip
python run.py analysis-check
python run.py baseline
python run.py transfer
```

The importer checks every input SHA256 before writing. `baseline` performs the formal 10,000-resample baseline/family analysis; `transfer` performs twenty group partitions per held-out source. These commands require CPU time but no GPU, retraining or new model inference. Figure regeneration and statistical recomputation are separate operations.

## Interpretation

- The original 156-sequence panel and the added 151-candidate primary panel have different membership and activity rules.
- The 151 candidates comprise 100 + 20 + 24 + 7 records and 121 primary positives plus 30 threshold negatives.
- c_AMPs uses an explicitly identified supplementary minimum-branch score for ranking; it is ineligible for score-extremeness retention policies.
- Bootstrap intervals are exploratory, pointwise and unadjusted for multiplicity. Repeated development partitions are sensitivity results, not confidence intervals or independent replications.
- The three retention policies have 458, 312 and 201 feasible instances out of 600 each. Paired comparisons use the same 201 instances and do not match realized coverage.
- Historical reference code is preserved separately. Its full-development transfer analysis and earlier figure numbering should not be substituted for the current group-separated analysis.

See `README_zh.md`, `docs/LICENSING.md`, `docs/REPRODUCIBILITY.md` and `docs/RELEASE_STEPS_zh.md` for scope, input access and release preparation.
