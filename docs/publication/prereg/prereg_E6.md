# Pre-registration — E6 STC-AUDIT: search-time contamination audit

**Registered:** 2026-06-11 (before first run). **Cost:** ~$0 (free PTU classifier pass).
**Canonical key:** `contamination`. **Frame:** robustness appendix, NOT a headline
(2606.05241 owns the framing).

## Question
Does orchestration intensity amplify benchmark-contamination exposure, and does the flat top
cluster survive decontamination?

## Binding amendment
Report per-snippet contamination **rates** via logistic regression
`P(snippet contaminated) ~ search_count + architecture` — NEVER raw totals (totals are
tautological in search count).

## Design
Apply the 2606.05241 taxonomy (metadata / question-context / explicit-answer leakage) to the
complete logged retrieval telemetry across all 11 architectures on the public-benchmark
queries. URL/domain regex + ONE GPT-4o classifier pass over logged snippets (free PTU). Then
re-run the four-gate statistics dropping contaminated queries via `rebuild_all.sh`.

## Primary endpoint
The architecture coefficient in the per-snippet contamination-rate regression (does more
orchestration raise the per-snippet leakage rate?), and the four headline effects recomputed
on the decontaminated query set (do they survive?).

## n and exclusions
All logged snippets across 11 architectures on public-benchmark (non-custom) queries.
EXCLUSIONS.md respected. Classifier prompt + decision threshold fixed before the run and
recorded in the build script.

## Analysis reported regardless of outcome
The rate regression + the survive/not-survive recomputation ship whichever way they land. If
the top cluster does NOT survive decontamination, that is a scope correction we publish
ourselves (and a G-style escalation). If it survives, it is the robustness appendix the plan
intends.

## Build addendum — coverage basis + frozen classifier (recorded at build, 2026-06-14)

**STEP-0 coverage decision (DUAL BASIS).** Full pre-citation snippet text (search.json
extractions) is on disk only for P0/P1/P9/P12. df_citations cited-URL/domain is the only
uniform 11/12-architecture logged-snippet signal (995 pattern×query cells), but it is the
CITED subset and under-counts retrieved-but-uncited contaminated snippets. Resolution:

- **PRIMARY basis = `citation`** (df_citations cited-URL/domain): the uniform
  11/12-architecture basis on which the rate regression
  `P(snippet contaminated) ~ search_count + C(architecture)` is fit. Reported as a
  CONSERVATIVE (under-counting) lower bound on the per-snippet leakage rate.
- **SENSITIVITY basis = `search`** (search.json full extractions, P0/P1/P9/P12 only): a
  higher-recall robustness check, never the headline coefficient.

The human confirms/overrides this authoritative basis before the paid classifier pass. The
build honours `--basis` everywhere.

**Frozen contamination signal.** Per-snippet contaminated = `regex_contaminated OR
classifier_contaminated` (union). The regex gate
(`scripts/contamination_regex_gate.py`) is deterministic and offline; its 2606.05241
taxonomy buckets (metadata_host / question_context / explicit_answer_leak), the
benchmark-host domain list, and the answer-surface patterns are FIXED in that file and
echoed by `--dry-run`.

**Frozen GPT-4o classifier (a transform TOOL, never a judge).** Model `gpt-4o` (PTU, free),
temperature `0.0`, decision threshold `0.5`. The system + user prompts are FIXED in
`scripts/run_contamination_classifier.py` (CLASSIFIER_SYSTEM / CLASSIFIER_TEMPLATE) and
echoed verbatim by `--dry-run`. They must not change between this build/verify pass and the
human-launched paid pass. GPT-5.2 remains the only authoritative judge and is untouched by E6.

**Known coverage limits flagged in the manifest.** trace.json query_ids are only partially
canonical UUIDs, so the `search_count` regressor is available only on the canonical-id
subset (non-canonical snippets contribute via architecture with `search_count` imputed to
the per-architecture median, flagged). P11 (react) has ZERO trace.json files. Checkpoint
coverage is a partial replicate sample (~22-31 distinct canonical query_ids per pattern), so
the uniform 85-query basis is df_citations.

**Canonical write is gated.** `build_contamination.py` writes a provisional side-car by
default and mutates `canonical_numbers.json['contamination']` ONLY under an explicit
`--finalize` run AND when a real (non-stub) classifier pass is present.
