## Current release v1.2.0

This release adds the study inference adapters, exact model-identity records, recorded runtime versions and the 169-input panel queries. Start with inference_reference/README.md and MODEL_ENVIRONMENT_REGISTRY.tsv. Statistical results and saved figure intervals remain unchanged from v1.1.1 (DOI 10.5281/zenodo.23856452). The v1.1.1 DOI identifies that earlier snapshot and does not include the newly restored inference files. Cite the v1.2.0 version DOI only after its Zenodo record is verified. Until then, the precise source snapshot is the GitHub v1.2.0 release. Older release instructions below are historical.

# Reproducibility scope

The public inputs support numerical recomputation of the added sequence
baselines, within-family ranking and group-separated selection analyses from
saved model outputs. The mathematical implementations, protocol and seeds are
unchanged. The historical loader alone accepts a hash-based public sequence
representation; it also retains compatibility with original raw-letter inputs.

The eight public files have independent SHA256 fingerprints. Six are byte-for-
byte copies of the frozen inputs. The two historical tables replace only the
sequence column with sequence_sha256, checking it against the existing
sequence_id for every row. All other cell values and row order are identical.
See INPUT_TRANSFORMATIONS.json and ORIGINAL_INPUT_FINGERPRINTS.json.

Top-level check validates the whole distribution and frozen formal tables;
analysis-check validates all eight inputs and their alignment; test exercises
eleven synthetic numerical cases plus input alignment. Baseline and transfer
perform fresh calculations. Figures rebuild seven assets from frozen formal
results and do not implicitly use the fresh numerical output directory.

The package does not include model weights, inference environments, original
PDFs/workbooks or historical raw sequence letters. Its historical_reference
directory retains older code and an original 498-file inventory as provenance;
that full inventory is not bundled. Therefore this is not a claim that every
manuscript figure or every original evidence extraction can be independently
reconstructed from this repository alone.

Original implementation provenance remains in SOURCE_PROVENANCE.json. The
changed input loader and release documentation are identified in
RELEASE_CHANGES.json. Reused QA scripts retain their exact upstream contents,
Apache-2.0 licence and attribution.
