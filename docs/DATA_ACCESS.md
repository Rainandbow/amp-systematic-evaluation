# Frozen input access

The eight sequence-level input files required by `analysis/run.py` are listed with exact SHA256 values in `analysis/input_manifest.json`. They remain in the existing author analysis package. This upload preparation does not assert a new licence for third-party sequence data.

The current repository provides aggregate statistical outputs sufficient for the seven included figure reconstructions. Sequence strings and candidate-level activity inputs are excluded from this upload directory. A persistent public input-data record has not yet been assigned. Full independent recomputation requires that data-access step to be completed.

For local verification, import the exact existing baseline/honest ZIP or extracted directory with `tools/import_local_inputs.py`. The importer checks all hashes before writing and refuses to replace different files. Keep the imported inputs out of browser uploads; .gitignore is used by Git but should not be treated as a browser upload filter.

Original experimental evidence comes from the APEX, AMPGen, ApexGO and HydrAMP publications. Historical evaluation sources are AMPlify, tsAMP-I, c_AMPs, HMD-AMP, AmPEP and ampir. Complete evidence locations, memberships and provenance remain in the author archive and the manuscript supplementary tables. Public repository URLs and DOIs should be added here when the data deposit is completed.
