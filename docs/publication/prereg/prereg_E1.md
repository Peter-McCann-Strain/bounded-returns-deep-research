# Pre-registration — E1 ROUTABILITY-90: is per-query winner rotation signal or noise?

**Registered:** 2026-06-11 (before first run). **Canonical key:** `routability`.
**Decision gate:** G1 (end of week 2).

## Positioning (binding amendments)
- Best-fixed baseline is **P1 (0.653), NOT P4** — all headroom computed over P1.
- Lead with measured winner-label **reliability** (≈0.27 observed vs 0.20 chance) and the
  **noise-corrected headroom BEFORE any router training**.
- Frame to publish either way; the rigorous null ("routing headroom on judged long-form
  research is mostly noise") is itself the answer the routing field skips. Benchmark feature
  recipe against DAAO 2509.11079.

## Question
Does a realizable router over the 11 architectures beat best-fixed P1 once winner labels are
corrected for run + judge noise, or is the opportunity mostly noise plus the source-family feature?

## Stages
- **Stage A (CPU, on disk):** per-query oracle-selection gain over P1 from `df_scores`
  (overall, per-dimension, per-source); test-retest reliability of the per-query winner label
  from the replicates; **shrinkage-correct** the oracle gain to a noise-adjusted headroom.
- **Stage B (CPU + <0.5 GPU-day):** routers under LOOCV AND leave-one-benchmark-out
  (logistic/GBM on query embeddings + entity density + causal-question score + length +
  source family); IRT difficulty × ability decomposition; Qwen2.5-7B prefill-activation probe.
- **Stage C (prospective, separate registration):** 30 fresh held-out queries; only if G1 passes.

## Primary endpoint
**Noise-corrected routing headroom over P1** (overall score), with CI. Winner-label
reliability reported alongside as the interpretive anchor.

## n and exclusions
11 architectures × 90 queries for the oracle gain; replicate corpus for reliability.
`overall_score_recomputed`; EXCLUSIONS.md; seeded sorted bootstraps.

## GATE G1 (decision rule, pre-committed)
If Stage A's noise-corrected headroom **< 0.02 over P1**, skip Stage C router training and
pivot Paper 1 to the null/methods framing (Stage A+B only). Record the decision in
`RESEARCH_PLAN_2026H2.md` and `PROGRAMME_EXECUTION_STATE.md`.

## Analysis reported regardless of outcome
The headroom estimate + reliability + router-CV curves ship whichever side of 0.02 they land.
Needs E2's noise floors for the winner labels (run after E2 core).

## GATE G1 DECISION (recorded 2026-06-11, as the prereg requires)
**G1 FIRES.** Stage A: winner-label split-half reliability 0.376 vs 0.251 chance (weak); the
rigorous real-independent-run (replicate-CV) noise-corrected headroom is ~0.003 over best-fixed
(the parametric 0.071 is optimistically biased and was not used). Stage B: NO feature router beats
best-fixed out-of-sample (LOOCV realized headroom: source -0.004, kNN -0.003, GBM -0.029, logreg
+0.002; best +0.0017 ≪ 0.02; leave-one-benchmark-out both negative). Deviation: best-fixed is
base_p4 ≈ base_p1 (tie); plan's "P1 (0.653)" is a different metric. **Decision: skip Stage C router
training; Paper 1 pivots to the rigorous-null/methods framing ("per-query architecture-routing
headroom on judged long-form research is mostly noise"). Recorded in PROGRAMME_EXECUTION_STATE.md
and RESEARCH_PLAN_2026H2.md (gate table).** Canonical: `routability` + `routability.stage_b`.
