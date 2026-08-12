# Reproducibility

This repository ships the full research codebase: every architecture
implementation, the evaluation harness, the experiment runners, and the
statistics that produce the paper's numbers. What it does **not** ship is the
multi-gigabyte tree of generated reports and raw judge verdicts, or archived
model and search snapshots.

That distinction decides what you can reproduce, so it is worth being precise
about it up front.

## Three tiers of reproduction

**Tier 1 — Statistics, offline and free.** The tidy frames in `data/analysis/`
are shipped. Most scripts in `papers/paper_a_bounded_returns/analysis/` read
only those frames and re-derive the paper's reported values with no API keys, no
GPU, and no network. Measured with each script run in its own pristine checkout,
**55 of the 76 `build_*.py` scripts exit 0 and 21 fail**; a few of the 55 are
resampling-heavy and take several minutes. This is the tier that verifies the
paper's arithmetic and inference.

**Tier 2 — Re-running the systems, paid or GPU-bound.** All 18 pattern
implementations ship and can be executed. They call current hosted models or run
local weights, so their outputs will not match the paper's outputs token for
token — models, search indexes, and the live web have all moved. Use this tier
to study behaviour, not to check equality.

**Tier 3 — Bitwise reproduction, not available.** The original matrix was run
against model versions and a search index that no longer exist. No amount of
shipped code recovers that. The frozen reference summaries in `repro/reference/`
exist precisely so that Tier 1 and Tier 2 results can be compared against what
the paper reported.

## What the analysis scripts need

The paper's numbers were produced in two stages:

```
results/**  ──(scripts/build_analysis_dataframes.py)──>  data/analysis/*.parquet  ──(analysis scripts)──>  paper numbers
   raw generated reports                                     tidy frames                                   canonical_numbers.json
   and judge verdict trees                                    (SHIPPED)
   (NOT SHIPPED)
```

`results/**` is excluded because it contains raw third-party web content
gathered by the agents, under mixed and largely unclear redistribution terms.
Publishing it would mean republishing scraped pages.

The practical consequence:

| Script reads | Example | Runs from a clean clone |
|---|---|:--:|
| `data/analysis/*.parquet` only | `build_numbers.py` | ✅ |
| `results/**` raw verdicts or reports | `build_bestofn.py` | ❌ |
| `data/benchmarks/**` third-party annotations | `build_judge_vs_human.py` | ❌ |

Run these on a throwaway checkout. The failure modes are not uniform: some
scripts raise a clear missing-path error, but others crash downstream on empty
data, and — more importantly — several **exit 0 having overwritten shipped files
with empty results**. Five rewrite artifacts under `analysis/staging/` with nulls
and zeros, and `build_numbers.py` rewrites `canonical_numbers.json` in place,
dropping the ~715 values whose inputs are not shipped. Recover with
`git checkout -- papers/paper_a_bounded_returns/analysis`.

To regenerate the frames yourself you would need to run Tier 2 end to end and
then run `scripts/build_analysis_dataframes.py` over your own `results/` tree.
The frames you produce will describe your run, not the paper's.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -c constraints-public.txt -e ".[api,paper,dev]"
cp .env.example .env
```

Extras are separate so you install only what you need:

| Extra | Purpose |
|---|---|
| `.[paper]` | Statistics and figures — enough for Tier 1 |
| `.[api]` | Provider SDKs, retrieval, extraction — needed to run P0–P8 |
| `.[local]` | torch, transformers, peft, trl — needed for P9–P17 |
| `.[dev]` | pytest, ruff |

`constraints-public.txt` pins the tested versions of everything except the GPU
stack, which is deliberately unpinned: torch must match your CUDA build.

Verify the install:

```bash
deep-research doctor
pytest -q
```

## Tier 1: recompute the paper's numbers

```bash
python papers/paper_a_bounded_returns/analysis/build_numbers.py
```

This reads the shipped frames and writes `canonical_numbers.json` plus a
Markdown summary in the same directory. That JSON is the paper's single source
of truth for headline statistics — the paper is written against it, so drift
between the two is detectable.

It rewrites that file **in place**, and drops 715 values whose inputs are not
shipped — 601 from the oracle block, 67 IRR-robustness, 47 DR-Judge. Note the
script prints `[ok] oracle` while dropping those 601. Every value it *can*
recompute is reproduced exactly. Restore afterwards with
`git checkout -- papers/paper_a_bounded_returns/analysis` — the narrower
`canonical_numbers.json` path leaves the regenerated `.md` digest modified.

`papers/paper_a_bounded_returns/analysis/rebuild_all.sh` chains the full
analysis, but it runs under `set -e`, so it stops at the first failure rather
than skipping unrunnable steps. On a public checkout that is step 4 of 104
(`build_numbers_extended`, which writes into `papers/paper_a_bounded_returns/tables/`,
a LaTeX output directory the release excludes). It is intended for the full
working tree, not a clean clone.

Compare a run summary against the frozen reference:

```bash
deep-research reproduce paper-a --mode smoke        # verify shipped inputs
deep-research reproduce paper-a --mode reference    # frozen headline ordering
deep-research compare paper-a --run-summary repro/reference/paper_a_pattern_metrics.csv
```

See `repro/PAPER_A_REPRO_MAP.md` for the command-to-artifact map and the
comparability contract for each.

## Tier 2a: API patterns (P0–P8)

Set `OPENAI_API_KEY`, or the Azure block in `.env.example` with
`USE_AZURE_OPENAI=true`. Azure needs deployment names and, for hosted search, a
deployment entitled for the configured `web_search` tool on the
OpenAI-compatible v1 Responses API.

Always estimate before spending:

```bash
deep-research doctor --require-api --ensure-dirs
deep-research doctor --verify-api                   # paid entitlement probe
deep-research cost paper-a --limit 3 --judge
deep-research reproduce paper-a --mode api-best-effort --limit 3 --max-cost-usd 5
```

Then execute:

```bash
deep-research reproduce paper-a --mode api-best-effort --execute --limit 3 --max-cost-usd 5
deep-research reproduce paper-a --mode api-best-effort --execute --full --judge   # full run
```

A full pass is 90 generation calls plus roughly 90 billed hosted-search tool
calls, which `deep-research cost` reports as a separate line; `--judge` adds up
to 90 OpenAI and 180 Anthropic judge calls. Outputs land under
`artifacts/reproduction/paper_a_api_best_effort/`. The cost ledger uses the
configurable per-call estimates in `.env.example` — it is a guardrail, not
provider billing truth.

Record the exact model identifiers you used. "gpt-4o" is not a reproducible
identifier over time.

## Tier 2b: local patterns (P9–P17)

Install a CUDA-matched torch, then `.[local]`, then fetch weights:

```bash
python scripts/download_models.py
python scripts/gpu_sanity_check.py
```

Reference environment: RTX 5080 (16 GB), CUDA 13.0, 4-bit quantization via
bitsandbytes, with `torch==2.10.0` and `transformers==5.2.0`. Models: Qwen2.5-7B-Instruct
(P9), GAIR/DeepResearcher-7b (P10), and the vintage/scale arms for P13–P17.
Weights are pulled from Hugging Face at run time and are never vendored here.

## Judging

The judge panel calls provider APIs directly; no local assistant session is
required. Preview without spending:

```bash
deep-research judge run \
  --query "Research question" \
  --report-file repro/examples/example_report.md \
  --criteria-file data/public_judge_criteria.json \
  --panel paper-a-api \
  --dry-run
```

Remove `--dry-run` to execute. The default `paper-a-api` panel needs both
OpenAI and Anthropic credentials. `--panel openai-only` is for cheap debugging
and must not be reported as the full panel.

Judge versions are part of the result, not an implementation detail: the study
treats GPT-5.2 as the anchor judge and labels other judge cohorts explicitly.
Never pool verdicts across judge versions silently — the frames carry a `judge`
column for exactly this reason.

Integrated reproduction with `--judge` uses each query's bundled rubric from
`data/eval_queries_v2.json`. `data/public_judge_criteria.json` is a small
standalone smoke file, not the study rubric.

## Publishing a public tree

```bash
deep-research export-public --out /tmp/deep-research-public
deep-research release-audit --root /tmp/deep-research-public
```

The export copies only what `PUBLIC_MANIFEST.json` allows and then audits the
result. The audit fails on secrets, filled `.env` files, absolute local paths,
private notes and agent working files, LaTeX sources, model weights, generated
bundles, and oversized files. It scans parquet column values rather than raw
compressed bytes, so cited URLs are not mistaken for local paths.

Rebuild the export before publishing if you ran tests inside a candidate tree;
bytecode caches are rejected by the audit.
