# Public Data Dictionary

`data/` in the public export is reserved for compact, reviewed inputs. Generated
payloads, caches, logs, checkpoints, local model weights, judge-output forests,
raw human-label assets, and training corpora belong outside GitHub.

| Path | Role | Public status |
|---|---|---|
| `README.md` | Human-readable data overview | Included |
| `DATA_DICTIONARY.md` | This inventory | Included |
| `eval_queries_v2.json` | Canonical 90-query evaluation manifest | Included |
| `all_90_queries.json` | Compact convenience index of the canonical query set | Included |
| `b2_subset.json` | B2 subset manifest | Included |
| `protocol_a_stratified_v2.json` | Protocol A stratification manifest | Included |
| `variance_stratified.json` | Variance experiment stratification manifest | Included |
| `public_judge_criteria.json` | Small API judge smoke/example criteria | Included |
| `benchmarks/` | Upstream benchmark cache directories | Excluded |
| `analysis/` | Tidy parquet/csv analysis frames the paper's statistics read | Included |
| `human_calibration_pack/` | Human-rater pilot packet | Excluded |
| `human_labels/` | Large local human-label assets with separate licenses | Excluded |
| `dr_judge_training/` | Large local DR-Judge training split | Excluded |
| `*_cache` directories | Live API/search caches | Excluded; runtime writes go under `artifacts/caches/` |
| `oracle_corpus_t1.json` and `e5_oracle_dose/` | Oracle/frozen corpus inputs needing separate review | Excluded |

The public export is governed by `PUBLIC_MANIFEST.json` at the repository root.

## Privacy Review

`eval_queries_v2.json` may contain `metadata.public_redaction` on rows that were
anonymized during release preparation. Public releases must not include named
private individuals, private planning-board details, secrets, or local paths in
query text or rubrics.
