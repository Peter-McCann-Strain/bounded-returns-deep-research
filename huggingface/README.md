---
license: apache-2.0
pretty_name: "Bounded Returns to Orchestration — Deep Research Evaluation Frames"
language:
  - en
tags:
  - deep-research
  - retrieval-augmented-generation
  - agents
  - orchestration
  - llm-as-judge
  - evaluation
  - reproducibility
size_categories:
  - 100K<n<1M
configs:
  - config_name: queries
    data_files: "df_queries.parquet"
  - config_name: runs
    data_files: "df_runs.parquet"
  - config_name: overall_scores
    data_files: "df_overall_scores.parquet"
  - config_name: scores
    data_files: "df_scores.parquet"
  - config_name: verdicts
    data_files: "df_verdicts-*.parquet"
  - config_name: citations
    data_files: "df_citations.parquet"
  - config_name: citations_protocol_a
    data_files: "df_citations_protocol_a.parquet"
  - config_name: c0_verdicts
    data_files: "df_c0_verdicts.parquet"
  - config_name: c0_per_report
    data_files: "df_c0_per_report.parquet"
  - config_name: e14_oracle_verdicts
    data_files: "df_e14_oracle_verdicts.parquet"
  - config_name: e14_oracle_per_report
    data_files: "df_e14_oracle_per_report.parquet"
---

# Bounded Returns to Orchestration — Deep Research Evaluation Frames

Judge-level evaluation data from a controlled comparison of automated
deep-research architectures over a 90-query manifest. These are the tidy frames
the paper's statistics are computed from: the reported numbers are recomputable
from the tables here using the analysis scripts in the code repository.

The paper compares **eleven architectures** on a single shared tool layer: a
single-pass baseline, eight orchestration pipelines (P0–P8, GPT-4o), and two
local 7B agents (P9, P10), with model and tools held fixed so that
*orchestration* is the variable. Those eleven are the rows of the paper's
headline table.

The code repository additionally implements seven later probes and controls
(P11–P17), for 18 pattern implementations in total. Two of them appear in these
frames: the reference ranking covers 13 arms, but `base_p11` and `base_p12` were
never scored by Claude Opus, so their scores rest on two judges and they are
reported as supporting analyses rather than leaderboard entries.

The `pattern` column carries **73 distinct labels** in `verdicts`, `scores`,
`overall_scores` and `runs` (the other configs carry 9–12), because the frames
cover far more than the headline arms:

| Group | Count | Example |
|---|---:|---|
| Leaderboard arms | 13 | `base_p0` … `base_p12` |
| Variance replicates | 32 | `base_p0_v1` … `base_p0_v11` |
| Ablations | 8 | `ablation_p4_no_triangulation` |
| Oracle-retrieval arms | 9 | `oracle_t1_p0` (the `e14_oracle_*` frames carry 11) |
| Tavily search arm | 6 | `protocol_a_tavily_p0` |
| Tool-layer disentanglement | 2 | `disentangle_matched_p1` |
| Other probes | 3 | `base_p1_7b`, `base_p11_16turn` |

**Do not filter on `base_p*` for a leaderboard view** — that prefix matches 48
labels, including every variance replicate. Select the 13 arms explicitly
(`base_p0` … `base_p12`), or filter `pattern_family == "base"` and drop the
`_v<n>`, `_7b` and `_16turn` suffixes.

Judgments come from a four-judge panel — `gpt52`, `claude_opus`,
`claude_sonnet`, and `claude_code` — so most frames carry a `judge` column.
Group by it rather than pooling across judges.

## What's here

| Config | Rows | Description |
|---|---:|---|
| `queries` | 90 | The evaluation manifest: query text, source benchmark, stratum |
| `runs` | 6,570 | One row per (pattern × query) execution, with cost and telemetry |
| `overall_scores` | 6,509 | Per-report aggregate score, per judge |
| `scores` | 58,552 | Per-report × per-rubric-dimension scores |
| `verdicts` | 248,536 | Per-criterion judge verdicts (binary SATISFIED + chain-of-thought), sharded across 6 parquet files |
| `citations` | 22,903 | Extracted citations with URL-resolution status |
| `citations_protocol_a` | 6,241 | Citation subset under the stratified Protocol A sample |
| `c0_verdicts` / `c0_per_report` | 3,096 / 269 | Claim-level factual verification |
| `e14_oracle_verdicts` / `e14_oracle_per_report` | 6,365 / 327 | Oracle-retrieval entailment arm |

Scoring uses a 9-dimension rubric (information recall, factual accuracy,
coverage, analytical depth, citation quality, logical coherence, organization,
instruction following, attribution quality), scored as binary per-criterion
verdicts with chain-of-thought and aggregated to weighted dimension scores.

**Read `DATA_DICTIONARY.md` (in this dataset, alongside the frames) before
using them.** It documents every column and, importantly, a set
of known upstream data issues that are recorded rather than silently repaired —
including a corrupted `overall_score` for one judge where a `_recomputed`
column must be used instead. Analyses that ignore that section will produce
wrong numbers.

## Quick start

```python
import pandas as pd

scores = pd.read_parquet("hf://datasets/PeterStrain77/bounded-returns-deep-research/df_overall_scores.parquet")
print(scores.groupby(["judge", "pattern"])["overall_score"].mean().unstack(0))
```

Or with `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("PeterStrain77/bounded-returns-deep-research", "overall_scores", split="train")
```

## Reproducing the paper's numbers

The frames alone are not the analysis. The code repository ships the full
pipeline implementations, the evaluation harness, and ~115 statistics scripts
that turn these tables into the paper's reported values:

**If you clone the code repository you already have these frames** — they ship
in `data/analysis/`, including an unsharded `df_verdicts.parquet`. Nothing below
is needed; just run the analysis.

The steps below are for using the Hub copy on its own. The analysis scripts read
from `data/analysis/` and expect `df_verdicts.parquet` as a **single file**, so
the shards must be reassembled:

```bash
git clone https://github.com/Peter-McCann-Strain/bounded-returns-deep-research
cd bounded-returns-deep-research
pip install -e ".[paper]" huggingface_hub   # hub client is only needed to fetch

# 1. Pull these frames into data/analysis/ (where the scripts look for them).
python - <<'EOF'
from huggingface_hub import snapshot_download
import glob, shutil, pathlib, pandas as pd

src = snapshot_download("PeterStrain77/bounded-returns-deep-research",
                        repo_type="dataset")
dst = pathlib.Path("data/analysis"); dst.mkdir(parents=True, exist_ok=True)
for f in glob.glob(f"{src}/*.parquet"):          # parquet only — do not overwrite
    shutil.copy(f, dst)                          # the repo's own docs with Hub copies

# 2. Reassemble the sharded verdict frame into the single file scripts expect.
shards = sorted(dst.glob("df_verdicts-*-of-*.parquet"))
pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True) \
  .to_parquet(dst / "df_verdicts.parquet", index=False)
print(f"reassembled {len(shards)} shards -> data/analysis/df_verdicts.parquet")
EOF

# 3. Recompute the paper's headline statistics.
python papers/paper_a_bounded_returns/analysis/build_numbers.py
```

`build_numbers.py` recomputes every headline statistic from the parquet files
and writes `canonical_numbers.json`, which is the paper's single source of truth.
Fourteen of the analysis scripts read `df_verdicts.parquet` directly, so if you
took the Hub-only route, skipping the reassembly step will make them fail with a
missing-file error.

Run the analysis scripts on a throwaway checkout: several read `results/**`,
which is not redistributed, and overwrite shipped files with empty results
rather than stopping. See the code repository's README for the full warning.

## Revision note (August 2026)

This is a corrected revision. Three changes matter to anyone who downloaded an earlier copy:

- **One benchmark query's real-world identity is now removed everywhere.** The query names a
  real small business and a named individual. Earlier redaction reached the query *prompt*
  only, so the identity survived in the citations the agents actually retrieved and in the
  judge text quoting them: business domains and URL slugs in `citations.cited_url`/`domain`/
  `cited_title`, the trading city, two community names, and 16 cells of `verdicts` reasoning.
  The rubric criteria in `eval_queries_v2.json` leaked it too, having been written against
  the unredacted prompt. All are now `[REDACTED]`.
- **A third party's email address was removed.** A scraped academic byline left a
  researcher's address in `citations.cited_title` (2 rows) and in one `verdicts.evidence`
  quote. The quote's analytic content is preserved -- the judge was citing it as an example
  of a malformed reference -- with the address replaced by `[email redacted]`.

  Both redactions are scoped to the affected `query_id`, so nothing else moved: row counts
  are unchanged (22,903 citations, 248,536 verdicts, 359,458 across all frames), every
  pattern mean in the paper still recomputes to the same value, and unrelated text that
  merely *contains* a matching substring -- Yellowstone's "Mesa Falls", "nevertheless",
  "cleverthai.com" -- is untouched.
- **The analysis scripts that consume these frames were corrected.** The cluster bootstrap
  resampled clusters without preserving draw multiplicity, which made every confidence
  interval and p-value derived from it too narrow, and 32 build scripts overwrote sibling
  keys in the results store instead of merging into it. Numbers recomputed from these frames
  with the current scripts will differ from a run of the older ones.

## Scope and limitations

- **Judge scores are model judgments, not ground truth.** The repository ships
  the human-calibration protocol and judge-vs-human agreement analysis; consult
  those before treating a score as an absolute quality measure.
- **`report_path` in `runs` points into a `results/` tree that is not
  redistributed.** The generated reports are large and carry mixed third-party
  content. Paths resolve only in a full working checkout.
- **Rows are not independent.** They are clustered by query and by pattern;
  naive pooled statistics will understate uncertainty. The analysis scripts use
  clustered and stratified estimators for this reason.
- **Mixed licensing on data-derived rows.** Query text and cited content derive
  from multiple upstream benchmarks under different terms. Apache-2.0 covers the
  code; see `DATA_LICENSES.md` and `NOTICE` in the code repository before
  redistributing data-derived material.

## Citation

```bibtex
@software{mccannstrain_deep_research,
  author = {McCann Strain, Peter},
  title  = {Deep Research: Bounded Returns to Orchestration},
  year   = {2026},
  doi    = {10.5281/zenodo.21118281},
  url    = {https://github.com/Peter-McCann-Strain/bounded-returns-deep-research}
}
```
