# Six current main figures and numerical reconstruction

This reconstruction supplement is prepared for release v1.1.0 and complements archived v1.0.1 (DOI 10.5281/zenodo.23836777). The v1.0.1 DOI identifies the earlier version and does not cover these added files. Cite the actual v1.1.0 record after successful archival. Its purpose is to make all six CURRENT manuscript figures reproducible from fixed predictions and explicit memberships.

## Run from the extracted directory

Create an isolated Python 3.12 environment and install requirements.txt. The verified statistical runtime used NumPy 2.3.5. Then:

```
python run.py check
python run.py points
python run.py figures
```

points recalculates historical confusion matrices/MCC (36 overall and 144 planned length cells), the 28 exact-membership patterns, all 96 baseline/model/cohort/task points and 48 named paired point differences, original positive coverage/detection, ApexGO native decisions/AUROC/budget precision, and Fig5 feasibility plus 201 paired changes. It checks the recalculated points against the archived figure inputs. It does not rerun bootstrap.

figures rebuilds CURRENT Figures1-6 as PDF/SVG/600dpi PNG/TIFF, runs alignment, text-boundary and collision audits, and keeps source tables. It reads the archived formal CIs. Figure4b is the paired shortest-first contrast, not the superseded absolute-recovery panel. Figure5 is the group-separated 201-pair analysis, not the old full-development transfer plot. Output directories must be new; use --output another_directory when repeating a figure render. For a single figure: python run.py figures --figure 4 --output another_directory.

## Recalculate statistical uncertainty and policy selection

```
python run.py uncertainty --output independent_check
python run.py baseline
python run.py transfer
```

uncertainty independently reruns 10,000 block resamples for Fig2/3 and the ApexGO inferential panels of Fig6 using base seed 2026110101. This is an additional verification stream with a new fixed seed, not a claim to recreate the original Monte Carlo draws byte-for-byte. It preserves undefined/degenerate interval states and never replaces the archived plotted intervals. Figure6a is descriptive and has no bootstrap CI.

baseline calls the original public implementation, using its original protocol, seed, memberships, controls and 10,000 shared draws. It regenerates the Fig4 absolute, family and full paired-contrast statistical tables. transfer calls the original 20-partition, six-fold group-separated threshold-selection implementation underlying Fig5. These two commands create their own timestamped formal_analysis/analysis/results/ output directories. Their constraints, data splits and pointwise intervals retain the original exploratory status. Neither command trains or scores a model.

## Scope and access

docs/MAIN_FIGURE_REGISTRY.tsv maps every figure to members, endpoints, source tables, commands and limitations. MANIFEST_SHA256.json pins this package. Original source copies are retained under reference/, with adaptation origins in SOURCE_PROVENANCE.json. The integrated renderers load only the functions actually used for the current figures and change data locations and previously corrected display terminology.

Eight public statistical input tables, formal baseline/family and group-separated outputs, source access documentation and their original manifests are under formal_analysis/. Historical inputs use sequence SHA256 identities, labels, predictions, lengths and frozen group IDs, sufficient for saved-output statistical recalculation and exact-identity intersections. They cannot reconstruct original peptide sequences or rerun grouping/inference. Original experimental evidence retains raw reported assay states and source cell locations. Source workbooks and third-party model weights are not bundled. Original data URLs, attribution and source-specific rights remain in formal_analysis/docs/.

Software authored for this reconstruction follows the existing Apache-2.0 project licence; third-party code notices remain applicable. Do not infer that the original experimental source material has been relicensed by this package. Publication or repository upload requires retaining source attribution and applicable source terms. The archived version cited in the manuscript must contain this supplement; verify its record and downloadable files after publishing v1.1.0.
