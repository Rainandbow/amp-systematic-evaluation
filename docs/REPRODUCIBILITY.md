# Reconstruction scope and provenance

1. `analysis/` keeps the baseline/family and honest-transfer numerical implementations byte-for-byte from the checked author package. The analytic protocol is unchanged. Its package manifest was regenerated for the code-only distribution; sequence-input hashes remain unchanged.
2. `figures/` keeps the seven-figure renderer unchanged. Two unused sequence-level tables were removed from the plotting inputs and its input manifest was regenerated. All retained statistics are byte-identical to the formal results. Plot generation does not recreate missing numerical analyses.
3. `historical_reference/` retains earlier numerical and plotting source files. Its original 498-file manifest is an expected reference inventory, not a claim that all those source inputs are included in this code upload. Full historical reconstruction requires restoration of the original archive structure and manifest-verified files. Earlier scripts and figure numbering are retained as provenance.
4. The repository entry point verifies the complete upload manifest, checks formal table alignment and runs synthetic algorithm tests. It does not replace the separate source-evidence audit or verify training-set independence.
5. The release is not yet assigned v1.0.0, a Zenodo DOI, a blanket licence or software creators. These should be finalized with the public data-access record before the manuscript describes a completed reproducibility deposit.

`SOURCE_PROVENANCE.json` maps retained original files to source packages and hashes. `MANIFEST_SHA256.json` verifies this prepared upload. Validation receipts are supplied next to the upload directory, not mixed with formal study results.
