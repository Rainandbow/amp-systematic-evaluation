# Systematic evaluation of released AMP predictors

Analysis, processed inputs and figure-generation code for six released AMP
predictors: Macrel, AMPlify, ampir, tsAMP-I, HMD-AMP and c_AMPs.

**Software: Apache-2.0. Processed data: CC-BY-4.0.** See `LICENSE`, `NOTICE`,
`LICENSES/CC-BY-4.0.txt` and `docs/LICENSING.md` for the respective scopes.

This working tree adds the eight statistical input files to the earlier
code-only v1.0.0 release. The existing v1.0.0 archive is available at
https://doi.org/10.5281/zenodo.23834121; it does not include these inputs.
Use the exact updated commit or its new archived release when citing the
input-complete distribution. A new version DOI is assigned only after archival.

## Install and verify

Use Python 3.11 or later in an isolated environment:

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py analysis-check
python run.py test
```

All eight required files are in `analysis/inputs/`; no external author archive
is needed for these analyses. Exact public and original frozen fingerprints,
field definitions and access routes are in `analysis/input_manifest.json` and
`docs/DATA_ACCESS.md`. Two historical tables use sequence SHA256 identities
instead of historical sequence letters. All labels, memberships, groups,
score values, native decisions and row orders are preserved.

## Recompute the added analyses

```bash
python run.py baseline
python run.py transfer
```

`baseline` runs the formal 10,000-resample sequence-baseline/family analysis.
`transfer` runs 20 development-group partitions per held-out source. They use
saved predictions, require CPU computation and create new result directories.
The numerical algorithms, protocol and seeds are unchanged. This distribution
does not rerun model inference or rebuild historical sequence clusters.

## Rebuild the seven included figures

```bash
python run.py figures
```

The frozen formal tables reconstruct main Figures 4 and 5 and supplementary
Figures S20–S24 in the synchronized manuscript. The renderer uses those saved
results, separately from numerical recomputation. Its 96 method–cohort–task
points, 1,440 paired model–baseline contrasts, 1,800 selected rules and 5,400
phase-level evaluations are retained. Each render creates `figures/output_*`.
Earlier historical reconstruction scripts remain in `historical_reference/`;
their original 498-file input inventory is not bundled in full.

## Interpretation

- The additional primary panel has 151 candidates (100 + 20 + 24 + 7), with
  121 primary activity positives and 30 above-threshold records.
- The original 156-tested panel is a separate dataset with different membership
  and activity rules. The eight-input addition does not replace its full archive.
- c_AMPs uses the supplementary minimum-branch score for ranking and is
  ineligible for score-extremeness retention/transfer rules.
- Bootstrap intervals are exploratory, pointwise and unadjusted. Repeated
  group partitions describe split sensitivity, not confidence intervals.
- Three selection policies yield 458, 312 and 201 feasible rules out of 600
  planned instances each. Their paired comparison uses the same 201 instances;
  realized coverage is not matched.

See `README_zh.md`, `docs/SOURCES.md`, `docs/REPRODUCIBILITY.md` and
`docs/RELEASE_STEPS_zh.md`. Software creators follow the confirmed Zenodo record:
Zhiyu Jin and Feifei Xiao. Third-party authors retain their own credits.
