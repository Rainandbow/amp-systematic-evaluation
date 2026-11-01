# Access to the processed statistical inputs

The **eight processed input files are bundled in `analysis/inputs/`** in this
updated repository. A clone or full source ZIP of this version includes them;
no private author directory or request to an author is needed for the added
baseline/family and group-separated validation analyses.

Repository: https://github.com/Rainandbow/amp-systematic-evaluation

## Get and verify

```bash
git clone https://github.com/Rainandbow/amp-systematic-evaluation.git
cd amp-systematic-evaluation
python -m pip install -r requirements.txt
python run.py check
python run.py analysis-check
python run.py test
```

Alternatively download the complete source ZIP from GitHub and extract it
before running those commands. For citation, use the exact commit or archived
release used for your calculations. The earlier v1.0.0 archive at
https://doi.org/10.5281/zenodo.23834121 does **not** contain these input files.
A future v1.0.1 DOI must be copied from the actual archived release; none is
preassigned or implied here.

## Files and denominators

| File | Rows | Purpose |
|---|---:|---|
| candidates.tsv | 169 | Additional-panel inputs: 153 generated peptides and 16 templates; family, source and sequence/evidence links |
| members151.tsv | 151 | Primary active-design candidates: cohorts 100 + 20 + 24 + 7, excluding templates and two designed negative controls |
| scores151.tsv | 906 | Six saved model outputs per primary candidate |
| task_labels.tsv | 1,146 | Cohort/strain/threshold labels and evidence locations; not 1,146 independent peptides |
| historical_members.tsv.gz | 29,104 | Six historical common-support memberships, source labels, group IDs and native decisions |
| historical_scores.tsv.gz | 41,956 | Full saved historical score/export records; the 29,104 members select the common-support analysis |
| original_policies.tsv | 36 | Earlier full-development rules: 30 eligible and six score-ineligible model/source combinations |
| original_budget_points.tsv | 2,688 | Frozen budget/subset/task outputs used for reproduction checks |

`candidates.tsv` retains the original 169-input archive's role/selection flags;
they are provenance, not the definitive 151-member roster. Use `members151.tsv`
for primary membership and `task_labels.tsv` with the fixed protocol for the
activity task. Earlier archive labels/flags are not silently rewritten.

`analysis/input_manifest.json` gives every distributed input's SHA256.
`INPUT_TRANSFORMATIONS.json` links each public fingerprint to its frozen
original. Six inputs are byte-identical; only the two historical tables replace
sequence letters with `sequence_sha256`. Row order and every other cell value
are preserved. The original fingerprints are retained in
`ORIGINAL_INPUT_FINGERPRINTS.json`.

## Recompute and rebuild

```bash
python run.py baseline
python run.py transfer
python run.py figures
```

`baseline` runs 10,000 block-bootstrap resamples. `transfer` runs 20 development
group partitions per held-out source. Fresh numerical results are written to
new `analysis/results/FORMAL_*` directories. `figures` independently rebuilds
seven figures from the included, frozen formal statistical tables; it does not
silently replace them with a newly calculated directory.

The frozen protocol and analysis seeds are unchanged. Historical sequence
identities, leakage checks and grouping memberships remain available in the
hash-based inputs. Raw historical letters are needed to recompute new sequence
features, recluster or rerun the predictors; those operations require the
original sources and model environments. This package supports the specified
saved-output statistical analyses, not fresh inference or a complete independent
reconstruction of every historical figure in the manuscript.

## Rights and evidence

Study-generated processed inputs/compilations use CC BY 4.0, with the source
attribution in `SOURCES.md`; software uses Apache-2.0. Experimental letters and
evidence extracts retain attribution to the CC BY 4.0 ApexGO and HydrAMP articles.
Source-reported units, unresolved values and chemistry are not silently
reinterpreted. Model weights, publication PDFs and original workbooks are not
bundled. See `LICENSING.md` and `INPUT_DATA_DICTIONARY.tsv`.
