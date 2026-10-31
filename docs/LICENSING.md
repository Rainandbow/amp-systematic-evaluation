# Code and data licensing

Original study-specific software and its documentation in this distribution are
licensed under **Apache-2.0**; the full text is in the root `LICENSE`.
The unchanged, attributed nature-figure QA scripts also use Apache-2.0.
Their original provenance and licensing notices are retained in `NOTICE` and
`docs/THIRD_PARTY_SOURCES.json`.

## Processed data

The study-generated data compilation, saved prediction outputs, membership and
group mappings, activity-rule annotations and aggregate results under
`analysis/inputs/` and `figures/inputs/` are provided under **CC-BY-4.0**.
The full data licence is in `LICENSES/CC-BY-4.0.txt`.
Please credit Jin and Xiao's AMP evaluation and the relevant original sources
listed in `docs/SOURCES.md`. This grant covers the authors' contributions and
compilation; it does not assert ownership of the original publications or
supersede any applicable source rights.

Experimental sequence/evidence extracts are attributed to the ApexGO and
HydrAMP articles, whose publisher versions state CC BY 4.0. Processing included
membership selection, identifier normalization, evidence linking and endpoint
adjudication. Raw historical sequence letters are omitted from public historical
inputs; their exact SHA256 identifiers are retained for statistical recomputation.

## Third-party scope

Model packages/weights and original PDFs/workbooks are not included. Installed
dependencies (including NumPy, Matplotlib and PyMuPDF) retain their own licences;
they are installed separately and are not relicensed by this repository.
Do not interpret the root Apache-2.0 licence as a grant for source datasets,
third-party software, model architectures, weights or third-party article assets.

## Version scope

This licensing arrangement applies to this updated distribution. The existing
v1.0.0 Zenodo archive (10.5281/zenodo.23834121) is an earlier, immutable code-only
snapshot, with its recorded CC BY 4.0 metadata. This update does not replace its
files or withdraw permissions already granted for that earlier snapshot.
For a new archived release, use Apache-2.0 as the software metadata licence and
describe the CC BY 4.0 processed-data exception explicitly.
