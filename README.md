# Deep Research: Bounded Returns to Orchestration

Does adding orchestration machinery to an LLM research agent actually make its
reports better? This repository is the full, runnable codebase behind a
controlled study of that question. The study compares **eleven deep-research
architectures** on a single shared tool layer — a single-pass baseline, eight
orchestration pipelines, and two local 7B agents — holding model and tools fixed
so that orchestration is the variable, scored by a multi-model judge panel over
a 90-query manifest.

The repository ships **18 pattern implementations** (P0–P17): the eleven the
paper compares, plus seven later probes and controls (P11–P17: ReAct, an
RL-trained arm, model-vintage and parameter-scale controls, and replication
extensions) that were added after the main comparison and are reported as
supporting analyses rather than leaderboard entries.

It ships the pipelines, the retrieval and judging tools, the evaluation harness,
the experiment runners, the statistics that produce every number in the paper,
and the tidy data frames those statistics read.

- **Paper:** `papers/paper_a_bounded_returns/main.pdf` — DOI [10.5281/zenodo.21118281](https://doi.org/10.5281/zenodo.21118281)
- **Data:** evaluation frames on Hugging Face — [PeterStrain77/bounded-returns-deep-research](https://huggingface.co/datasets/PeterStrain77/bounded-returns-deep-research)
- **License:** Apache-2.0 applies to code. Public data files are mixed-license by row/source — read `NOTICE` and `DATA_LICENSES.md` before redistributing data-derived material.

## The architectures

The eleven the paper compares:

| Family | Patterns | Backbone | Isolates |
|---|---|---|---|
| A — prompt-engineered pipelines | P0 baseline, P1 iterative RAG, P2 supervisor-parallel, P3 MERIDIAN, P4 STORM, P5 hierarchical width-depth, P6 reactive, P7 graph decomposition, P8 beam search | GPT-4o | architecture, with model and tools fixed |
| B — local agents | P9 Qwen2.5-7B baseline, P10 DeepResearcher-7b | local 7B | model scale (P9 vs P0) and RL training (P9 vs P10) |

Later probes and controls, shipped as runnable code and reported as supporting
analyses rather than leaderboard entries:

| Patterns | Backbone | Purpose |
|---|---|---|
| P11 ReAct, P12 RL-trained | local 7B | never scored by Claude Opus; reported on the GPT-5.2 + corrected-Sonnet pair |
| P13–P14 vintage models, P15–P16 replication extensions, P17 14B scale arm | local | model vintage and parameter count |

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -c constraints-public.txt -e ".[api,paper,dev]"
cp .env.example .env          # fill in keys only for what you intend to run
deep-research doctor
```

Three things you can do, in increasing order of cost:

```bash
# 1. Free: recompute the paper's headline statistics from the shipped frames.
python papers/paper_a_bounded_returns/analysis/build_numbers.py

# 2. Free: inspect the frozen reference and verify shipped inputs.
deep-research reproduce paper-a --mode smoke
deep-research reproduce paper-a --mode reference

# 3. Paid: generate and judge reports against current APIs.
deep-research cost paper-a --limit 3 --judge          # estimate first
deep-research reproduce paper-a --mode api-best-effort --execute --limit 3 --max-cost-usd 5
```

Installing extras separately keeps heavy dependencies optional:
`.[api]` runs the patterns, `.[paper]` recomputes statistics, `.[local]` adds
the GPU stack for P9–P17, `.[dev]` adds pytest and ruff.

## What reproduces, and what doesn't

This is the honest map. Reproduction splits three ways depending on which
inputs a step needs.

| Step | Ships | Runs offline | Notes |
|---|:--:|:--:|---|
| Recompute headline statistics from analysis frames | ✅ | ✅ | `build_numbers.py` and 42 sibling scripts that read only `data/analysis/*.parquet` |
| Pattern implementations, tools, evaluation harness | ✅ | — | Full source for all 18 pattern implementations; running them calls paid APIs or needs a GPU |
| Frozen paper reference and comparison | ✅ | ✅ | `repro/reference/`, compared via `deep-research compare` |
| Statistics that read raw judge/report trees | ✅ | ❌ | Code ships; `results/**` does not (see below) |
| Statistics that read third-party human annotations | ✅ | ❌ | Code ships; upstream benchmark annotations are not redistributable |
| Live API regeneration | ✅ | ❌ | Best-effort against current models; **not** bitwise-equal to the paper |
| Bitwise-identical rerun of the original matrix | ❌ | ❌ | Needs archived model and search snapshots that no longer exist |

**Why some analysis scripts cannot run from a clean clone.** The paper's numbers
were computed in two stages: raw generated reports and judge verdict trees under
`results/**` were reduced into the tidy frames in `data/analysis/`, and the
statistics were computed from those frames. This release ships the frames, not
the multi-gigabyte `results/**` tree — it contains raw third-party web content
with mixed redistribution terms. So the reduction step
(`scripts/build_analysis_dataframes.py`) and the scripts that reach past the
frames into raw verdicts cannot be re-executed here, while the statistics that
read the frames run offline and reproduce the reported values.

Measured with each script run in its own pristine checkout: **55 of the 76
`build_*.py` scripts exit 0 and 21 fail.** A handful of the 55 are
resampling-heavy and take several minutes.

> **Run the analysis scripts on a throwaway checkout.** Of the 55 that exit 0,
> 12 read `results/**` and degrade silently when it is absent — five of those
> rewrite shipped files under `analysis/staging/` with nulls and zeros, and
> `build_numbers.py` itself rewrites `canonical_numbers.json` in place, dropping
> 715 values whose inputs are not shipped (601 oracle, 67 IRR-robustness,
> 47 DR-Judge). Every value it *can* recompute is reproduced exactly.
> Of the 21 that fail, only some raise a clear missing-path error; others crash
> downstream on empty data, and three fail on `reports/`, `tables/`, or an
> overwrite guard rather than on `results/`.
>
> Recover with `git checkout -- papers/paper_a_bounded_returns/analysis`.

Live API reruns are labelled `not-comparable` by `deep-research compare` by
design: they do not re-run the paper's 13-arm reference matrix (the eleven
compared architectures plus the two Opus-unscored probes P11/P12), and models
and search indexes drift.

## Repository map

| Path | Contents |
|---|---|
| `deep_research/patterns/` | The 18 pattern implementations (P0–P17) |
| `deep_research/tools/` | Search backends, URL extraction, LLM callers, rate limiting |
| `deep_research/evaluation/` | Rubric v2, LLM judges, statistics, calibration, human-eval harness |
| `deep_research/benchmarks/` | Loaders for DRACO, LitQA2, FreshWiki, ResearchQA, DeepSearchQA, RACE, and ScholarQA |
| `deep_research/` (root modules) | CLI, settings, reproduction, comparison, public export, release audit |
| `scripts/` | Experiment runners, dataset builders, model downloads, training |
| `papers/paper_a_bounded_returns/analysis/` | 76 statistics builders plus 34 figure/table generators |
| `data/analysis/` | Tidy evaluation frames the statistics read |
| `data/` | Query manifest, rubric criteria, stratification splits, dictionaries |
| `repro/` | Frozen paper reference summaries and the command-to-artifact map |
| `docs/` | Evaluation protocol, human-evaluation protocol, pre-registrations |
| `tests/` | Test suite covering the harness, statistics, and release contract |
| `huggingface/` | Dataset card and publishing steps for the Hugging Face mirror |

## Running the patterns

Patterns P0–P8 call hosted APIs. Set `OPENAI_API_KEY` (or the Azure block in
`.env.example`) and use the cost guardrails — `deep-research cost` reports call
counts, and `--max-cost-usd` caps a run. A full 90-query pass is 90 generation
calls plus roughly 90 billed hosted-search tool calls; adding `--judge` adds up
to 90 OpenAI and 180 Anthropic judge calls.

Patterns P9–P17 run locally on a GPU. Install `.[local]` with a CUDA-matched
torch build, then fetch weights:

```bash
python scripts/download_models.py
```

The reference environment was a 16 GB RTX 5080 with 4-bit quantization; see
`REPRODUCIBILITY.md` for the exact stack.

## Publishing a release

The public tree is built from an explicit allowlist in `PUBLIC_MANIFEST.json`
rather than by hand, and audited before it leaves the machine:

```bash
deep-research export-public --out /tmp/deep-research-public
deep-research release-audit --root /tmp/deep-research-public
```

The audit rejects secrets, absolute local paths, private notes, agent working
files, LaTeX sources, model weights, generated forests, and oversized files, and
enforces the manifest so anything outside the allowlist fails the build. Rebuild
the export before publishing if you ran tests inside a candidate tree.

**Publish the export as a fresh `git init`, never as a branch or remote of the
working repository.** The manifest controls which *files* are copied; it has no
effect on git history. A working repository that has ever tracked a filled
`.env`, generated outputs, or private notes will publish all of them through
history even when the current checkout is clean.

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
