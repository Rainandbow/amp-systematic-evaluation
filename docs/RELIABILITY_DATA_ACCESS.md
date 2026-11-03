# Reliability inputs and source access

The public reliability module is self-contained for saved-feature statistical recalculation. It includes the historical labels, scores, verified lengths, frozen global groups, source memberships, selection masks and nearest-reference distances. Historical/reference amino-acid letters are replaced by SHA256 identities, consistent with earlier releases. Experimental candidate letters retain ApexGO/HydrAMP attribution and their existing licence scope.

## Files required for the published statistical calculations

- reliability/inputs/historical_members.tsv.gz and historical_scores.tsv.gz: 29,104 common members and 41,956 full saved rows. Source labels, native predictions, scores, lengths, groups and order are retained.
- reliability/inputs/references.tsv.gz: reference SHA256, length and canonical status. reference_membership.tsv.gz records original inventory roles and source record locations. inventory.tsv summarizes inventories. These do not assert complete fitted-training membership.
- reliability/cache/historical_distance_index.json: hashed sequence identity, verified length and global group in frozen index order.
- reliability/cache/development_0.npz through development_5.npz: all 20 selection masks, development distance features and nearest-neighbour indices. They are statistical inputs, not optional acceleration caches. Keep them.
- reliability/results/inventory_nearest_reference.tsv.gz: frozen nearest-reference features. Gzip is lossless.
- Experimental members, scores, task labels and AMPlify submodel summaries are in reliability/inputs/.
- refit/results/refit_formal/ retains the original 2,000-draw joint, conditional and refit-only arrays. Fresh refitting writes a different directory.

## Recomputing distances from original letters

Distance features cannot be derived from hashes. Obtain original source sequences and training-source inventories using docs/SOURCES.md, the upstream model publications/repositories, and the record locations in reference_membership.tsv.gz. This release does not distribute raw historical letters or complete third-party reference inventories. Their exact frozen input fingerprints are in RELIABILITY_ORIGINAL_INPUT_FINGERPRINTS.json; public row hashes/lengths provide sequence-level verification. Inventory correspondence to released weights remains source-qualified.

Run release.py distance-check --source-dir PATH against a directory containing the original four files historical_members.tsv.gz, historical_scores.tsv.gz, references.tsv.gz and reference_membership.tsv.gz. It validates exact expected fingerprints without copying them into the public distribution. Run release.py distance --source-dir PATH --output NEW_DIRECTORY to hydrate a separate working copy and rebuild distances with the original algorithm. The full distance step requires roughly 10 GB free space and may be slow. No model weights or GPU are needed for distance/statistical recalculation. There is no automated promise that current upstream files equal the historical snapshot; report mismatches rather than bypassing checks. Without the original letters, use the complete frozen-feature statistical path.

## Licences

Project-authored code follows Apache-2.0. Processed statistical outputs/compilation follow CC BY 4.0 with source attribution. This grant does not relicense original sequences, inventories, article supplements or third-party software. See LICENSE, NOTICE, docs/LICENSING.md and docs/SOURCES.md. The existing c_AMPs vendor subtree retains its GPL terms. No model weights, source article PDFs, source workbooks or installed dependencies are bundled.
