# Pre-registration — E10 NOISE-RL: structured vs random reward noise at matched κ (Paper 4)

**Registered:** 2026-06-23 (before first GRPO run). **Cost:** $0 cash — local RTX 5080
QLoRA only; NO paid API, NO Opus during training. Only the SEPARATE post-hoc held-out
judging touches GPT-5.2 (JUDGE endpoint via `run_gpt52_judge_namespaced.py`, never PTU).
**Canonical key:** `e10_noise_rl` (written ONLY by the post-hoc analysis pass, never by
training). **Frame:** Paper 4 — *"Correlated Error, Not Noise Magnitude: Why
Judge-Reward RL Fails for 7B Deep Researchers"* (ICML 2027).
**Gated on:** E7 (`e7_selector_kappa`) landed + `drjudge_youden_j.e10_gate.gate_pass_overall == true`.

## Question

When a 7B deep-research policy is RL-trained (GRPO) against a *noisy* judge reward, does the
**structure** of the judge's error (criteria flip together — measured DR-Judge-7B
error-correlation) degrade the final held-out report quality more than the **same amount** of
i.i.d. random error at matched marginal flip rate? And can an analytic FPR/FNR correction
rescue it?

## Conditional framing (PRE-COMMITTED both ways — REQUIRED by Gate G2)

The inference-time preview E7 (`e7_selector_kappa.matched_kappa`) found
`structured_minus_random ≈ −0.003` across κ = 0.20/0.35/0.50 with
`gate_g2.gate_fires_b_approx_e == true` (max |structured − random| = 0.0036 < 0.005). i.e. at
the **selection** layer, structured and random judge noise are indistinguishable. The plan's
Gate G2 therefore instructs E10 to pre-register **both** outcomes so the result is publishable
either way. We register both:

- **Framing 1 — "Correlated Error, Not Magnitude"** (the headline IF B ≪ C in *training*):
  RL amplifies correlated error beyond what selection reveals, so B (structured) ends up worse
  than C (random) at matched marginal κ.
- **Framing 2 — "Magnitude + Rescue"** (the fallback IF B ≈ C, consistent with E7): structure
  is NOT the killer at the RL layer either; the publishable contribution becomes
  (i) the **dose-response of noise magnitude** (clean A vs noisy B/C) and (ii) the **rescue**
  result (noise-corrected D recovering toward A). E7's `b ≈ e` is reported as the
  selection-layer precedent that predicted this.

Both analyses ship regardless of which way B-vs-C lands. We do NOT hard-code B ≪ C as the only
registered outcome.

## Hypotheses

- **H1 (primary, B-vs-C):** At matched marginal flip rate (pooled
  `drjudge_error_structure.calibration.pooled_marginal_flip_rate = 0.2811`), structured
  copula-correlated reward noise (arm B) yields a *lower* held-out overall score than i.i.d.
  matched-marginal noise (arm C): **B < C**. Falsified (→ Framing 2) if the seeded
  query-bootstrap CI for (B − C) contains 0 / equivalence holds (|B − C| < 0.005, mirroring
  E7's equivalence threshold).
- **H2 (rescue):** Noise-corrected GRPO (arm D = arm-B noise + arXiv:2510.18924 Bernoulli
  FPR/FNR debiasing with empirical per-criterion rates) recovers toward clean: **D ≈ A**
  (|D − A| within the same equivalence band).

## Arms (plan-of-record; supersedes the readiness-script additive-Gaussian arms)

| arm | name | reward | seeds |
|-----|------|--------|-------|
| A | `A_clean` | clean DR-Judge-7B LoRA reward | 1 |
| B | `B_struct` | Gaussian-copula correlated criterion flips, ρ = `latent_copula_rho_tetrachoric` = 0.3472, per-dimension asymmetric FPR/FNR from `drjudge_error_structure.per_dimension` | **{1,2,3}** |
| C | `C_random` | i.i.d. criterion flips at the SAME pooled marginal rate (ρ = 0) | **{1,2,3}** |
| D | `D_corrected` | arm-B noise then analytic FPR/FNR debiasing (empirical per-criterion rates) | 1 |

Noise is injected on per-criterion verdicts and anchored exactly as `run_e7_selector.py`
L48-50: `reward = clean + (recompute(flipped) − recompute(true))`. The noise layer is pure,
seeded, CPU-deterministic (`deep_research/training/e10_reward_noise.py`); calibration is read
from a PINNED read-only canonical snapshot, hash-guarded by
`drjudge_youden_j.drjudge_fixture_recompute_match == true`.

**Stack deviation (recorded):** plan §E10 names Unsloth; Unsloth is NOT installed and the
proven 16 GB path is plain TRL 1.3.0 + PEFT 0.19.1 (`train_p12_rl_v2`). E10 uses TRL. Benign;
the paper's methods section must match the code, not the plan text.

## Endpoints

- **Primary:** GPT-5.2-judged **held-out** overall-score delta **B − C** (and **D − A**), with a
  **seeded query bootstrap CI** computed over the ≥3 seeds of B and C. The reproducibility unit
  is **cross-seed variance**, NOT bit-identity (GRPO GPU rollouts are not bit-reproducible; the
  noise layer is fully seeded/CPU-deterministic).
- **Secondary (anti-Goodhart):** the **judge-free objective-score** delta on the held-out
  answer-checkable slice (`deep_research/training/e10_objective_endpoint.py`). This is a NOISY
  PROXY (deterministic string/entity match over gold key-facts), logged EVERY eval step
  alongside the judge reward to detect Goodharting (judge reward rises while objective is
  flat/falls). It MONITORS divergence and is NEVER the training reward.

## n and split

- **n = 90** `eval_queries_v2` queries, stratified by difficulty (5 simple / 25 moderate /
  60 complex), partitioned by `scripts/e10_prereg_split.py` (seed 20260623) into a held-out
  EVAL split and a TRAIN (rollout-prompt) split, written to `data/e10_split.json`. The
  **answer-checkable** queries (any with a non-empty `reference_answer` OR `expected_elements`)
  are FORCED into the EVAL split so the judge-free metric is only ever computed on held-out
  items.
- **Answer-checkable slice = 34 distinct queries** (29 with `reference_answer` + 15 with
  `expected_elements`, overlap 10). This corrects the design's "44", which double-counted the
  10-query overlap. The committed `data/e10_split.json` `content_hash` is the prereg anchor:

  ```
  content_hash = db4ae2affea3ea6f0b84113059f013ec090e5c3b6cd4fb56b5f4e11cc5586a04
  ```
  (seed 20260623, eval_frac 0.40; reproduce with `python scripts/e10_prereg_split.py --dry-run`)

- **≥3 seeds on the load-bearing B and C**; single seed on A and D (cost control). Effective
  arm-runs (full): A(1) + B(3) + C(3) + D(1) = **8 adapter trainings**.

## Gates restated

- **Gate G2 (E7):** `drjudge_youden_j.e10_gate.gate_pass_overall == true` (overall signed
  Youden J = 0.5028, "rate" phase) is the GPU go/no-go. PASS as of registration. Dimensions at
  J ≤ ε (`j_zero_epsilon = 0.05`) cannot be RL-trained against and are panel-routed, not
  trained.
- **G1 (GPU block):** a contiguous ≥6-GPU-day RTX 5080 window must be owner-reserved
  (`e10_noise_rl_readiness.py --gpu-block-hours N`, exit 0) before launch. Hard cap 10 GPU-days.

## Exclusion rules (pre-committed)

- **Degenerate-collapse runs** are flagged and excluded from the primary contrast: a run whose
  reward has **zero variance for > 30% of logged steps** (reward-collapse / mode-collapse) is
  reported but dropped from B-vs-C / D-vs-A, with the count disclosed.
- `EXCLUSIONS.md` respected. The quarantined query `82de3e92` (judge-specific AUP
  false-positive) is excluded from any GPT-5.2 held-out scoring, not from training.

## Analysis reported regardless of outcome

Both Framing 1 and Framing 2 analyses ship. If B ≈ C (consistent with E7), we publish the
magnitude + rescue result and the selection-vs-RL contrast (E7 `b ≈ e` predicted it). If the
rescue fails (D ≉ A), that is itself a finding about the limits of analytic FPR/FNR correction
under correlated error. The post-hoc pass appends the `e10_noise_rl` canonical key with the
B−C and D−A bootstrap deltas, per-arm collapse diagnostics, and the objective-vs-judge
divergence trace.

## Build addendum (recorded at build, 2026-06-23)

- **Readiness reconcile:** `scripts/e10_noise_rl_readiness.py` Gate G2 was rewritten to read the
  ACTUAL landed keys (`drjudge_error_structure.calibration`, `drjudge_youden_j.e10_gate`) — the
  old `selector_e7._calibration.{sigma_gpt52_run_sd,sigma_gpt4o,kappa_targets}` keys never
  existed in canonical. Arm labels renamed clean/struct_copula/matched_random/noise_corrected.
  Launch shape now calls `scripts/train_e10_noise_rl.py --arm …`.
- **Known canonical name-mismatch (out of E10 scope, flagged):** `run_e7_selector.py` writes
  `cn['selector_e7']` but the landed key is `e7_selector_kappa`. This is an E7 merge bug; E10
  does not fix it.
