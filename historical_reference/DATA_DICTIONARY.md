# Data dictionary and provenance map

Frozen data are read-only inputs. Every selected file is identified in input_manifest.json by its package-relative path and SHA256. Source-reported locations and older manifests retain their original paths as provenance; the three portable reconstruction scripts resolve their actual inputs relative to this package.

| Directory | Role |
|---|---|
| frozen/budget153 | Original real-server 153-generated-candidate scores, settings and budget outputs; six saved model outputs per candidate |
| frozen/membership_analysis | Audited primary 151-candidate analysis, excluding two designed negative controls, and membership sensitivity |
| frozen/task_threshold | Fixed-member activity threshold/strain sensitivity; labels, MIC evidence, metrics, paired resampling, censoring bounds |
| frozen/source_decomposition | Exact frozen cross-source rules, per-fold/source class counts, retained weights and 707 reproduction checks |
| frozen/evidence_adjudication | 327 study records, 3,017 MIC rows, source-cell/PDF checks, chemistry/strain conflicts and separate original/new rosters |
| frozen/model_scope | Model task, executable input contract, native decisions/scores, inventory roles and verified implementation identities |
| frozen/reliability | Saved score precision/export trace and released AMPlify submodel disagreement diagnostics |
| frozen/original_evidence | Unchanged original APEX/AMPGen evidence workbook and audit tables; researcher-review fields preserved |
| frozen/v8_main, frozen/v8_supp | Previous formal figure source tables and settings; only explicitly revised endpoints are replaced |
| frozen/experimental_inputs | Additional-panel source inputs and historical members/scores used by the portable analyses |
| frozen/original_transfer_figure | Immutable original formal Figure 5 asset, retained as v9 Figure S17 |

Core record keys are stable study record IDs and canonical sequence strings. Record identity, exact-string identity, near-sequence grouping and chemical identity are distinct. Duplicate strings across publications do not erase their separate study records or prove identical terminal chemistry.

Common metric fields include cohort, subset, task, threshold, unit, model, n, positive, threshold_negative, unresolved, blocks, budget, budget_slots, positive_hits, precision_at_budget, positive_recovery and enrichment. Budget hits may be fractional expectations under label-blind boundary ties. No unannotated or unresolved activity is automatically treated as a negative.

Threshold-dependent labels are P (meets the specified activity rule), N (above that rule under eligible assays) or U (unresolved). Censored bounds keep their comparator and original units. An observation >64 μM is negative at 64 μM but unresolved at 128 μM. Follow-up assays are linked to their initial sequence records and do not create additional candidates.

CI fields ci_low/ci_high and status retain pointwise exploratory percentile uncertainty. Missing/NaN limits may indicate too few groups, insufficient valid replicates, degeneracy, single-class outcomes or an unavailable policy. A missing interval is not evidence of zero uncertainty. The observed five-source range in Figure 5 is not a CI. Formal selector-refit intervals remain in Figure S17.

Cross-source rules use five development sources and one held source in each of six folds. With n_s eligible, k_s retained and e_s retained errors:

C = (1/5) sum_s(k_s/n_s)

R = sum_s(e_s/n_s) / sum_s(k_s/n_s)

alpha_s = (k_s/n_s) / sum_j(k_j/n_j).

Thus R = sum_s alpha_s (e_s/k_s) for non-empty retained sources. Source class counts, retained fractions and contributions remain explicit; mixture feasibility does not assert per-source or held-source error control.

Historical negative labels, database-unannotated background and source-condition MIC threshold negatives are reported separately. All analyses are retrospective. No new biological activity measurements or model training are introduced.
