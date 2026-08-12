# Paper A Public Reproducibility Map

This file maps public commands to the paper artifacts they validate. It is intentionally explicit about what can and cannot be reproduced from a public GitHub checkout.

The repository ships the full research codebase, including all 18 pattern implementations. What limits reproduction is *data*, not code: the raw generated reports and judge verdict trees are not redistributable. See `REPRODUCIBILITY.md` for the three tiers.

## Public Commands

| Command | Paid APIs | Purpose | Comparable to paper metrics? |
|---|---:|---|---|
| `deep-research reproduce paper-a --mode smoke` | No | Verifies that reference summaries and public inputs are present. | No; integrity check only. |
| `deep-research reproduce paper-a --mode reference` | No | Prints the frozen headline reference: 90 queries, 13 patterns, primary `mean_3judge` ordering, and comparison policy. | Yes; this is the compact reference view. |
| `deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv` | No | Validates the shipped compact pattern-by-judge metrics CSV against the frozen reference JSON. | Yes; pattern-level audit only. |
| `deep-research doctor --verify-api` | Yes | Makes tiny live calls to verify current OpenAI hosted-search and Anthropic model entitlement. | No; access check only. |
| `deep-research cost paper-a --full --judge` | No | Estimates call counts and configurable budget guardrails for a full live API rerun. | No; planning only. |
| `deep-research reproduce paper-a --mode api-best-effort --execute --limit N` | Yes | Generates current OpenAI hosted-search reports for public queries without model downloads. | No; this is a live API demo, not the frozen 13-pattern matrix. |
| `deep-research reproduce paper-a --mode api-best-effort --execute --full --judge` | Yes | Generates 90 live reports and scores successful reports with OpenAI plus Anthropic API judges. | Partially; useful for qualitative drift checks, but not a historical equality claim. |
| `deep-research compare paper-a --run-summary RUN.json` | No | Compares candidate pattern-level `mean_3judge` metrics with the frozen public reference. | Yes only when `RUN.json` contains pattern-level metrics. API-demo summaries fail with a clear not-comparable status. |

## Frozen Reference

- Reference file: `repro/reference/paper_a_headline_numbers.json`
- Compact metrics CSV: `repro/reference/paper_a_pattern_metrics.csv`
- Query count: 90
- Pattern count: 13
- Primary metric: `mean_3judge`
- Top public reference pattern: `base_p1`
- Lowest public reference pattern: `base_p12`

### Why the reference lists 13 arms and the paper's table shows 11

`base_p11` and `base_p12` were not scored by Claude Opus. Their `mean_opus` is
null, so their `mean_3judge` is a **two-judge** mean (GPT-5.2 plus corrected
Sonnet) even though the column name is shared across the whole table. The
paper's headline table reports the 11 arms with complete three-judge coverage,
so those two appear in this ranking but not in that table. This is a coverage
difference, not a disagreement: all 11 rows of the paper's table match the
canonical store exactly on both mean and cell count.

When comparing a rerun against this reference, filter to rows where `mean_opus`
is non-null before treating `mean_3judge` as a three-judge average.

## Public Data

- `data/eval_queries_v2.json`: compact public query set.
- `data/public_judge_criteria.json`: small standalone judge-smoke criteria; integrated `--judge` uses each query's full bundled rubric.
- `data/README.md` and `data/DATA_DICTIONARY.md`: sources, licenses, and field definitions.

## Excluded From Public GitHub

Paper drafts and LaTeX sources, private notes and agent working files, generated report forests, raw judge verdict trees, caches, model weights, checkpoints, upstream benchmark corpora, and submission bundles are excluded. The public export command enforces this allowlist and writes `PUBLIC_EXPORT_REPORT.json` with file hashes and provenance.

The dominant exclusion is `results/**`: raw generated reports and judge verdict trees. It is withheld because it contains third-party web content gathered by the agents under mixed redistribution terms, not because of size alone.

## Source Scope

The public source tree ships the full codebase: all 18 pattern implementations (P0–P17, API and local/GPU), the retrieval and judging tools, the evaluation and benchmark harnesses, the experiment runners and training scripts, the paper's statistical analysis, the tidy analysis frames those analyses read, the public tests, and the final PDF.

Local/GPU patterns (P9–P17) are shipped as runnable code. They require a CUDA-matched torch build and model weights fetched from Hugging Face at run time (`python scripts/download_models.py`); weights are never vendored here. Analysis scripts that read past the tidy frames into `results/**` or into third-party benchmark annotations ship but cannot execute from a clean clone, and fail with an explicit missing-path error rather than emitting numbers from empty data.

## Comparable Candidate Schema

A comparable run summary should contain either `primary_ordering`, `pattern_metrics`, or `metrics_by_pattern`. Each pattern row must include a pattern name and numeric `mean_3judge` or equivalent `score` field. Example:

```json
{
  "primary_ordering": [
    {"pattern": "base_p1", "mean_3judge": 0.67},
    {"pattern": "base_p4", "mean_3judge": 0.64}
  ]
}
```
