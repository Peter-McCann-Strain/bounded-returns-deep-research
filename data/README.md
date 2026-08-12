# Public Evaluation Data

The public release includes compact, reviewed inputs plus the tidy analysis
frames the paper's statistics read. Generated reports, live-search caches,
human-label packets, model-training data, and upstream benchmark cache
directories are deliberately excluded from the public export.

## Included Files

- `eval_queries_v2.json`: canonical 90-query evaluation manifest used by the paper.
- `all_90_queries.json`: compact convenience index for the same query set.
- `b2_subset.json`: B2 subset manifest.
- `protocol_a_stratified_v2.json`: Protocol A stratification manifest.
- `variance_stratified.json`: variance experiment stratification manifest.
- `public_judge_criteria.json`: small public criteria file for API judge smoke tests and examples.
- `DATA_DICTIONARY.md`: data inventory for the public export.
- `analysis/`: the tidy parquet frames the paper's statistics read, plus their
  own data dictionary and build manifest. Also mirrored as a Hugging Face dataset.

## Query Manifest Schema

`eval_queries_v2.json` contains query records with `id`, `query`, `source`,
`domain`, `difficulty`, optional benchmark metadata, and a `rubric` object.
The original benchmark cache directories are not included in GitHub; the
canonical selected query manifest is the reproducible public input. One
DRACO-derived row is explicitly marked with `metadata.public_redaction` because
it was anonymized before public release.

## Redistribution Notes

The included query manifests are compact research inputs selected for the paper
supplement. `eval_queries_v2.json` is mixed-license by row: DRACO and ResearchQA
rows are MIT, DeepSearchQA rows are Apache-2.0, LitQA2/LAB-Bench rows are
CC-BY-SA-4.0, and custom rows are Apache-2.0. See `DATA_LICENSES.md` for the
source table and upstream links. Before any future expansion of `data/`, review
upstream license and privacy status and update both files with per-file terms.
