# Pre-registration — E8 VINTAGE / CAPACITY (frozen-scaffold local backbone curve)

**Status:** pre-registered before the paid 14B judging run.
**Surrogate authority:** `reports/RESEARCH_PLAN_2026H2.md` §E8 VINTAGE (priority 7.5).
**Date:** 2026-06-23.

## Hypothesis

Holding the research scaffold (P9 local-baseline architecture), tools, prompts,
queries, evidence limit, and judge constant, and varying ONLY the local
backbone, the frontier-vs-local quality gap (GPT-4o P0 anchor minus local) will
either **close**, **persist**, or **reverse** across:

1. **Vintage axis (release date):** Qwen2.5-7B (2024-09) → DeepSeek-R1-Distill-Qwen-7B (2025-01).
2. **Capacity axis (params at fixed vintage):** Qwen2.5-7B (7B) → Qwen2.5-14B (14B), both 2024-09.

Directional sub-hypotheses (registered, not assumed):
- H1: vintage slope of the overall gap (gap-change per year) is **negative** (closing).
- H2: per-dimension closure is **asymmetric** — reasoning dimensions (analytical_depth,
  logical_coherence) move more than citation_quality / factual_accuracy, which the
  prior P0-P10 work found universally pinned.
- H3: at fixed vintage, ~2x capacity (14B vs 7B) yields a **positive but bounded**
  capacity gain on the overall gap (`capacity_gain_overall_vs_p9 > 0`).

## Design

- **Scaffold:** FROZEN P9 (search_batch(10) + academic → dedup → URL extract thin docs →
  two-step SourceExtractor → evidence truncated to 6000 words → single report-gen call).
- **Backbones:**
  - base_p9 — Qwen2.5-7B-Instruct (transformers, 4-bit nf4), EXISTS.
  - base_p14_vintage_deepseek_qwen7b — DeepSeek-R1-Distill-Qwen-7B (transformers, 4-bit).
  - base_p17_scale_qwen25_14b — Qwen2.5-14B-Instruct via **llama.cpp GGUF Q4_K_M**
    (transformers/bnb OOMs the 14B on the 16 GB RTX 5080; capacity anchor, same vintage as P9).
- **Endpoints:** overall weighted score + each of the 9 rubric_v2 dimensions;
  gap = anchor(base_p0, same judge) − local; vintage two-point slope; capacity delta vs P9.
- **n:** 90 queries (`data/eval_queries_v2.json`, sorted by id) per arm, matching the 7B arms.
- **Decoding:** transformers 7B arms use do_sample temp≈0.01; the 14B GGUF arm decodes
  **strict-greedy** (temp=0, top_k=1, top_p=1.0, repeat_penalty=1.0, seed=42). This small
  decode mismatch is **declared, not hidden** (a scaffold-frozenness imperfection).

## Judge & independence

- **Primary judge:** GPT-5.2 (OpenAI) on the JUDGE Azure endpoint (never PTU), temp=JUDGE.temperature.
- **Independence:** all subject arms are Qwen-family / DeepSeek-distilled-Qwen; the judge is
  OpenAI — clean per `feedback_judge_independence.md` (NEVER a Qwen judge). Optional Claude subsample.

## Exclusions

- Query **82de3e92** is quarantined from the **Claude-Code** judge panel only (reproducible
  AUP false-positive; judge-specific, NOT dropped from the study). It IS generated and IS
  judged by GPT-5.2 here.

## Analysis & canonical contract

- Single canonical key **`e8_vintage`**, atomic tmp+replace, append-only, never clobbers siblings.
- Deterministic: sorted inputs, no randomness.
- **Self-guard:** if any arm is unjudged on disk the builder still succeeds and emits a
  partial (`partial_pending_arm2` and/or `+partial_pending_14b`); it never invents a point.
- **Axis discipline:** the 14B is recorded as a `capacity_point` (scale axis, x=0 years),
  NOT as a third point on the gap-vs-YEARS vintage curve.

## Cost

- Generation: $0 (local). Judging (separate, human-launched): ~90 × ~$0.08 ≈ $7–8 on
  GPT-5.2, within the $30–60 E8 envelope and the $300 GPT-5.2 ceiling.
