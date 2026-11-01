# Systematic evaluation of released AMP predictors

Analysis, processed inputs and figure-generation code for Macrel, AMPlify, ampir, tsAMP-I, HMD-AMP and c_AMPs.

## Current manuscript reconstruction prepared for v1.1.0

The complete current six-main-figure workflow is in [`six_main_figures_v1_1_20261101/`](six_main_figures_v1_1_20261101/). From that directory, use Python 3.12 and an isolated environment:

```bash
cd six_main_figures_v1_1_20261101
python -m pip install -r requirements.txt
python run.py check
python run.py points
python run.py figures
```

`points` recalculates saved-output point estimates and validates denominators. `figures` rebuilds all six current main figures from the frozen formal points and archived intervals. `uncertainty` reruns 10,000 grouped resamples independently for historical MCC/length strata and the ApexGO inferential panels of Figure6; its new fixed-seed outputs remain separate from the plotted historical intervals. `baseline` and `transfer` rerun the original formal baseline/family and group-separated selection protocols. No command above retrains or reruns an AMP predictor.

The panel-by-panel mapping is in [`MAIN_FIGURE_REGISTRY.tsv`](six_main_figures_v1_1_20261101/docs/MAIN_FIGURE_REGISTRY.tsv). Revision diagnostic packages under `revision_diagnostics/` contain absolute validation outcomes, all 48 paired shortest-first hit differences, and family/block composition and deletion checks. The archived scientific result tables were preserved. The third-step publication package replaces historical raw letters with the same public SHA256-based statistical input; 29,104 rows were checked cell-by-cell and all 14 influence-analysis tables rebuilt byte-for-byte.

The current formal baseline/family and group-separated analyses also reproduced 14 formal statistical tables byte-for-byte. These two 14-table checks cover different analysis sets and do not imply that every historical bootstrap stream was recreated exactly. Bootstrap intervals are exploratory, pointwise and unadjusted. Fig5 partition spread is sensitivity to reused group assignments, not a confidence interval or independent external replications.

## Earlier public version

The original repository-root commands remain available for the v1.0.1 baseline/family and group-separated analyses and the earlier seven-figure subset. Their eight processed inputs are in `analysis/inputs/`. Running the original root `python run.py figures` does not build the current six-figure set; enter the supplement directory above for those figures.

The earlier version is archived at https://doi.org/10.5281/zenodo.23836777. Cite the actual version used. A DOI for the v1.1.0 release must be taken from the successful Zenodo archive record; no new DOI is assumed here.

## Data, sources and licences

Project software uses Apache-2.0, retaining notices for reused components. The specified study-generated processed statistical data and compilation use CC BY4.0, with source attribution and applicable third-party rights retained. See `LICENSE`, `NOTICE`, `LICENSES/CC-BY-4.0.txt`, `docs/LICENSING.md`, `docs/SOURCES.md` and the corresponding documentation under the supplement's `formal_analysis/` directory.

Historical inputs preserve sequence SHA256 identities, labels, lengths, groups, predictions and scores, sufficient for fixed-output statistics. Original historical peptide letters are omitted. Original experimental evidence retains reported source identifiers, sequence/activity annotations and source locations. Source workbooks, publications and third-party model weights are accessed through the original sources and are not redistributed as original files here. New model inference and sequence grouping require those original inputs and environments.

Original and added experimental panels retain separate memberships: the original panel has 156 tested peptides and 84 primary activity positives; the added primary panel has151 candidates (100+20+24+7),121 activity positives and30 above-threshold records. c_AMPs uses the supplementary minimum-branch score only for ranking and remains ineligible for score-extremeness retention/transfer.

Software creators follow the existing confirmed record: Zhiyu Jin and Feifei Xiao. Third-party authors retain their own credits. The manuscript author list is separate from the software record metadata.
