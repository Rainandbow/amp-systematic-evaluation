# Byte-preserving archive repair v1.2.1

The v1.2.0 Zenodo download (10.5281/zenodo.23861993) changed CRLF to LF in 19 files during UTF-8 publication transport. All 19 contents were identical after newline normalization, but the byte-level SHA256 checks failed. This patch restores the intended bytes through base64 Git blobs and updates release metadata. Predictions, activity labels, model weights, selection thresholds and scientific results are unchanged. No model inference or bootstrap run is repeated.

Use the v1.2.1 release for byte-verified reproducibility. The v1.2.0 DOI remains a historical record, not proof of a successful checksum audit. The exact version DOI for this patch must be read from its actual Zenodo record after publication.
