# Pre-registration — E13' Injection-ROC (frontier-judge perturbation detection)

- **Track:** Track C, C1_e13_detector (E13' PERTURB-TRUTH), feeding Paper A "Bounded
  Returns to Orchestration".
- **Registered:** 2026-06-22, BEFORE the GPT-5.2 perturbed-set judging run.
- **Canonical key landed:** `canonical_numbers.json['e13_detector_roc']`
  (sub-block `gpt52_injection_roc`).
- **Builder:** `papers/paper_a_bounded_returns/analysis/build_e13_detector_roc.py`.
- **Sibling (already specified):** the LOCAL detector panel (constructed-ground-truth
  DETECTORS, not judges) is pre-registered with its AUC>=0.60 drop floor in the
  detector-panel docstring and lands in the same key under `local_detector_panel`.
  This note covers ONLY the new **frontier GPT-5.2 injection-ROC** variant.

## Hypothesis

The authoritative judge (GPT-5.2) should be SENSITIVE to deliberately injected,
REFLECT-style defects: when a known defect is injected into a clean report, the
judge's score on the *dimension the defect targets* should drop, and that drop should
discriminate perturbed reports from their matched clean originals above chance.

- **H_inj (primary):** Per defect-targeted dimension, the matched-pair score-DROP
  separates perturbed (positive) from matched-clean-original (negative) items with
  detection ROC-AUC > 0.50, and we report whether it clears the pre-registered
  0.60 floor used for the local panel (same floor, for comparability).

## Frozen design (decided before seeing any perturbed verdict)

- **Items:** the 36 perturbed reports already built (108 gold-label rows, k=3
  defects/report, 24 rejects) under `reports/perturbation_set/`. One detection ITEM
  per (report_id, defect_type); 36 matched pairs total (each report carries one
  rotated defect_type by construction).
- **Matched-pair, single-dimension scoring.** For each pair we read GPT-5.2's
  per-dimension score on EXACTLY the defect-targeted dimension, fixed in advance:
  - `numeric_flip`        -> `factual_accuracy`
  - `deleted_evidence`    -> `factual_accuracy`
  - `contradiction`       -> `factual_accuracy`
  - `fabricated_citation` -> `citation_quality`
- **Detection score** = `score(clean original) - score(perturbed)` on that dimension
  (the score-DROP). **Gold label** = 1 for the perturbed item, 0 for the matched
  clean original (whose baseline drop is 0). ROC-AUC computed per dimension
  (`factual_accuracy`, `citation_quality`), pooling all defect_types that map to it.
- **Originals are NOT re-judged.** All 36 matched clean originals already have GPT-5.2
  verdicts in `results/judge_gpt52/<pattern>/<qid>.json` (verified present 2026-06-22).
  The cloud run judges ONLY the 36 perturbed reports, writing to the NEW, corpus-safe
  root `results/judge_gpt52_perturb/` (never `results/judge_gpt52`).
- **Judge:** GPT-5.2 on the cloud Azure JUDGE endpoint (`JUDGE_OPENAI_ENDPOINT`,
  via `run_gpt52_judge_namespaced.py`). NEVER the PTU. **NO Opus anywhere**
  (hard rule, 2026-06-22). The local 7B panel is the only other detector and is
  reported strictly as a DETECTOR with a floor, never as a judge.
- **Same rubric V2** as the main corpus; identical judge prompt/temperature/seed
  path. The perturbed report is judged with the SAME query/rubric as its original,
  so the only difference between the matched pair is the injected defect.

## Endpoints reported (and only these)

1. Per-dimension injection-ROC AUC on `factual_accuracy` and `citation_quality`,
   with n, n_pos/n_neg, mean positive score-drop, and the above-0.60-floor flag.
2. The local detector panel's per-family, per-defect_type detection AUC + the
   floor outcome (already produced; reduced into the same canonical key).

## Honest scope / what this does NOT claim

- This tests judge SENSITIVITY to injected defects on a small matched set (36 pairs,
  split across two dimensions). It anchors detectability; it does not certify the
  judge's absolute calibration.
- `factual_accuracy` and `citation_quality` aggregate GENERAL rubric criteria, so a
  drop reflects the dimension's aggregate sensitivity, not per-claim grading.
- If GPT-5.2 perturbed verdicts are absent at build time, the `gpt52_injection_roc`
  block is emitted as `{"status":"pending"}` and the local-panel block still lands;
  no number is fabricated.

## Decision rule (pre-committed)

- Report all AUCs regardless of outcome. We do NOT drop GPT-5.2 below the floor (it is
  the authoritative judge, reported for sensitivity, not gated); the floor flag is
  descriptive. The local panel families ARE dropped below 0.60 per their own prereg.
- Matched-pair direction is fixed (drop = orig - pert); we do not flip it post hoc.

## Determinism / idempotency

- Builder is a pure reduction of on-disk JSON; same inputs -> identical key.
- Both upstream runs are `--resume`/idempotent and clobber-safe: the GPU panel caches
  per-(family,item) JSONL and atomically rewrites `detector_results.json`; the GPT-5.2
  run writes only under the new `results/judge_gpt52_perturb/` root and refuses
  (hard guard) to resolve into any protected corpus path.
