# Released-model inference provenance

Published with v1.2.0. The seven runtime components represent six released AMP models, with separate c_AMPs Keras and BERT components. The recovered adapter, identity JSON, server plans and original statistical-environment record preserve their original bytes, verified against the historical input manifest. They document executed implementation identities; this publication does not claim a fresh inference run.

## Records and entry points

MODEL_ENVIRONMENT_REGISTRY.tsv links components to identity fingerprints, recorded library versions and original runtime paths. expected_identities/*.json records exact model/source hashes and the runtime versions actually recorded. server_plan.json records original per-component executables and resource paths. They are provenance, not portable installation locations. The root tested_environment.json describes local statistical reconstruction only, not the seven inference runtimes. Unrecorded dependency versions must not be inferred from it.

code/infer_panel.py is the original 169-query panel runner. inputs/new_panel_queries.tsv restores its sequence-identity-checked input. Classification and ranking rules remain in vendor_adapter/ and the frozen statistical inputs. To review package identity without third-party dependencies, run from the repository root:

    python inference_reference/verify_publication.py

For actual inference, obtain the original models and external language-model backbones from the sources in docs/SOURCES.md, reconstruct the separate environments using upstream instructions and the recorded identity versions, and configure a new copy of server_plan.json to point to those resources. Keep the archived plans unchanged. Use code/infer_panel.py check before run; inference requires the allocated resources, original third-party dependencies and matching weights. The older inference_reference/run_all.sh refers to the original larger analysis package; it is retained as provenance, not the current six-main-figure entry point.

The current statistical and figure workflow remains six_main_figures_v1_1_20261101/run.py. No historical prediction, activity label, threshold, bootstrap interval or figure estimate is changed in this release.

## Licences

Project-authored additions follow the root project licence. Reused components retain their own licences. In particular, vendor_adapter/vendor_camps/vendor/LICENSE contains the original GPL terms; that subtree must not be represented as Apache-2.0-only. Preserve source headers, licence text and provenance when redistributing. Third-party trained weights are not bundled.
