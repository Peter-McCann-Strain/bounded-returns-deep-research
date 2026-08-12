# Pre-registration — E4 CITE-CAUSAL: does citation density CAUSALLY move factual-accuracy verdicts, and is the effect Claude-specific?

**Registered:** 2026-06-14 (before any paid re-judge run). **Canonical key:** `e4_cite_causal`.
**Decision gate:** G3 (content-preservation gate, below). Commit this file before the paid run.

## Question
Citation *density* correlates with judged factual accuracy across the corpus, but content and
provenance are confounded with density. By holding the prose **byte-identical** and perturbing
only citation tokens, E4 asks: does citation density *causally* move factual-accuracy verdicts,
and is that causal effect **specific to the Claude judge family**?

## Hypotheses
- **H1 (Claude-specific density bias):** For the Claude family (Opus 4.1, Sonnet 4.5), increasing
  citation density (C1→C2→C0→C3) raises the `factual_accuracy` score even though the prose is
  fixed; i.e. the `density_index × judge_family[claude]` interaction coefficient is **> 0** and
  significant.
- **H0 / artefact framing:** Because content is fixed across conditions, ANY non-zero density
  slope is a **judge artefact**, not a quality signal. A null density slope for a family means
  that family's factual-accuracy channel is density-robust.
- **H2 (shuffle is null):** C4 (shuffle claim↔citation mapping; density unchanged) should NOT move
  `factual_accuracy` beyond noise — markers present but mis-mapped. A non-null `is_shuffle` effect
  would indicate the judge keys on marker *presence/count*, not correctness, sharpening H1.
- **Secondary:** the same density effect is expected (and larger) on `citation_quality` and
  `attribution_quality`; these are positive controls — the transform *should* move them.

## Conditions (content held fixed; only citation tokens change)
| cond | transform | density_index | is_shuffle |
|------|-----------|:---:|:---:|
| C0 | original (copy) | 0 | 0 |
| C1 | strip ALL inline markers + references list | −2 | 0 |
| C2 | halve density (drop every other distinct marker, seeded) | −1 | 0 |
| C3 | double density (duplicate each marker in place) | +1 | 0 |
| C4 | shuffle claim↔citation mapping (permute ids, seeded; density unchanged) | 0 | 1 |

## Endpoints
Primary: `factual_accuracy`. Secondary (positive controls): `citation_quality`,
`attribution_quality`. Model:
`dim_score ~ density_index * C(judge_family) + is_shuffle + C(pattern) + (1 | query_id)`
fit per dimension; per-family OLS density slopes reported alongside for interpretability. The
`density_index × judge_family[claude]` interaction is the **primary test of H1**.

## n and design
- **n = 100 reports × 5 conditions = 500 transformed reports.** Sample is stratified by
  architecture arm (base_p0…base_p10) and source family (draco / deepsearch_qa / research_qa /
  litqa2 / custom), seeded (`SEED=4`), recorded in
  `results/experiments_e4_cite/sample_manifest.json`.
- **Judges:** GPT-5.2 (PRIMARY, sole authoritative score) + gpt-4.1 + gpt-4o (OpenAI panel
  comparators) + DR-Judge-7B (GPU; built but **excluded from this phase**); Claude Opus 4.1 +
  Claude Sonnet 4.5 at full n via the Claude Code subscription harness.
- **Dispatch counts:** GPT-5.2 = 500 calls (authoritative); +500 gpt-4.1 +500 gpt-4o (PTU/free);
  Claude = 100×5×2 = 1000 agent dispatches ($0 cash, subscription session-window).

## Exclusion rules
- **base_p10 near-null reports** (≤3 inline markers; RL-agent quirk): C1–C4 are near no-ops there,
  so the density contrast is weak. These are FLAGGED `near_null_transform` in the manifest and
  **down-weighted / excluded** from the density-slope estimation (kept for C0 description only).
- **Query 82de3e92** is FLAGGED per `quarantine_82de3e92.md` (reproducible AUP false-positive on
  the Claude-Code judge panel). Judge-specific: flagged in the task index, reconciled downstream;
  not dropped from the study.
- Any report failing the G3 content-preservation gate is **aborted** (never judged).

## GATE G3 (content-preservation, pre-committed)
Every transformed report must have **byte-identical non-citation prose** to its original after
stripping citation markers from BOTH sides (and the references list for C1) and normalising
whitespace. `scripts/build_e4_transforms.py` asserts this and writes
`results/experiments_e4_cite/preservation_report.json`. **If any prose diff is non-empty, abort
that report (and the build) — a content change would invalidate the causal interpretation.**
*(Build status at registration: 500/500 reports PASS, 0 prose diffs.)* An optional GPT-4o yes/no
"is the non-citation prose identical?" check (transform/classifier TOOL only, never a judge) is
available via `--gpt4o-check`.

## CoT-mining (localisation; READ-ONLY)
Over the baseline ~248k-verdict corpus (`data/analysis/df_verdicts.parquet`, READ-ONLY): among
**unsatisfied** `factual_accuracy` verdicts, measure how often each judge's stated `reasoning` /
`evidence` invokes citations/sources. A Claude > OpenAI elevation localises the channel through
which density could leak into the factual verdict. (Baseline-corpus mining is reported regardless
of the E4 re-judge outcome.)

## Mitigation prompts (drafted here; A/B'd in the paid run)
Three rubric amendments to test whether they shrink the Claude density slope toward the OpenAI
slope (A/B on C0 vs C3):
- **M1 (decouple):** "When judging FACTUAL ACCURACY, evaluate only whether stated claims are
  internally consistent and correct. Do NOT consider the presence, count, or formatting of
  citation markers — those are scored separately."
- **M2 (density-blind):** "Citation markers such as [3] have been programmatically altered and are
  NOT reliable signals of accuracy. Judge factual claims on their substance alone."
- **M3 (count-invariance):** "Two reports with identical prose but different numbers of citation
  markers MUST receive the same factual_accuracy verdict."
The mitigation "wins" if it reduces the Claude density slope toward the OpenAI slope without
distorting the positive-control dimensions.

## Analysis reported regardless of outcome
The per-family density slopes, the Claude×density interaction (with CI), the C4-shuffle null test,
the CoT-mining family contrast, and the mitigation A/B all ship whichever way they land. A clean
**null** for OpenAI with a **positive Claude slope** is the headline; a null everywhere is itself
the finding that judged long-form factual accuracy is density-robust.

## Artefacts / provenance
- Transforms + sample + preservation: `scripts/build_e4_transforms.py` →
  `results/experiments_e4_cite/{C0..C4}/{pattern}/{qid}.md`, `sample_manifest.json`,
  `preservation_report.json`.
- GPT-5.2 + OpenAI-panel re-judge (corpus-safe, NEW dirs): `scripts/run_gpt52_judge_e4.py` →
  `results/judge_gpt52_e4/`, `results/judge_gpt41_e4/`, `results/judge_gpt4o_e4/`.
- Claude full-n re-judge (prep/parse, Read+Write-only agents): `scripts/run_e4_claude_cite.py` →
  `results/judge_claude_opus_e4/`, `results/judge_claude_sonnet_e4/`.
- Analysis: `reports/paper_world_class/analysis/build_e4_cite_causal.py` →
  `canonical_numbers.json['e4_cite_causal']`; wired into `rebuild_all.sh` ([5m]).
- **Corpus protection:** `results/judge_gpt52/`, `results/experiments/`, `data/analysis/*.parquet`,
  `reports/eval_v2/verdicts/` are strictly READ-ONLY; every runner refuses to write there.
