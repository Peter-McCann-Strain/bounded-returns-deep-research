# Documented Exclusions

This file records every query/cell deliberately excluded from an analysis arm, with the
reason, so that gaps in the released verdicts are auditable and are not mistaken for
incomplete work.

## E9 — Frontier scale-curve (gpt-4o-mini tier) descoped: deployment unavailable

**What was excluded.** The pre-registered E9 frontier orchestration scale-curve (sweeping the
generation backbone across gpt-4o-mini → gpt-4.1 → gpt-4o with architecture held fixed) is
**not run**. The `e9_scale_curve` canonical key is intentionally absent.

**Reason.** The low-capability tier `gpt-4o-mini` is **not deployed** on the available Azure/PTU
resource (`404 DeploymentNotFound`; confirmed by the resource owner). The remaining available
tiers — gpt-4.1 and gpt-4o — are both frontier-class and too close in capability to span a
meaningful low→high curve, and gpt-5.2 cannot be a subject (it is the judge). A frontier scale
curve is therefore infeasible on this infrastructure; this is a hardware/deployment limit, not a
negative result. The E9 backbone-propagation harness (`scripts/run_e9_scale_curve.py`) is built
and verified but unrun.

**Coverage.** The scale/capacity axis is instead addressed by the **local** frozen-source
vintage/capacity curve (`frozen_vintage`: Qwen2.5-7B → Qwen3-8B → Qwen2.5-14B, length-controlled)
plus the gpt-4o base study as the frontier anchor. The capacity claim (14B > same-vintage 7B) is
supported there: on **raw** GPT-5.2 scores the capacity gap is p17−p9 = **+0.025, CI [0.008,
0.043], p=0.005** (per-query paired bootstrap on the raw overall means). Controlling for output
length does not erase it — the **length-adjusted** p17−p9 point is **+0.034** (point estimate only;
no bootstrap CI is computed on the length-adjusted means). All four arms are de-confounded
(2026-06-30): p9/p14/p13/p17 now decode under a single llama.cpp GGUF greedy backend, so the axis
isolates the generation model. Raw arm overall means (89 frozen queries, GPT-5.2): P9 (7B,
2024-09) = 0.278, P14 (DeepSeek-R1-Distill-7B, 2025-01) = 0.223, P13 (Qwen3-8B, 2025-04) = 0.356,
P17 (14B, 2024-09) = 0.304; length-adjusted means: P9 = 0.278, P14 = 0.273, P13 = 0.299, P17 =
0.311. All numbers from `canonical_numbers.json['frozen_vintage']` (arms + capacity_axis).

## E1 — Query `82de3e92` excluded from the Claude-Code judging panel (AUP false positive)

**Query.** `82de3e92-abe2-46ac-ad17-23417b9c4da7` (source: LitQA2; domain: scientific
literature; difficulty: complex). A benign biomedical literature-QA item asking what
grafting the ECL3 region of the adenosine A3 receptor (A3AR) onto A2AAR does to binding
efficacy for the antagonist CF101.

**What was excluded.** The Claude-Code-path panel judgments (`claude_opus`,
`claude_sonnet`) for this query in two re-judge packets:
- `20260608_base_cluster_opus`: patterns base_p1, p4, p5, p6, p7, p8 (6 cells)
- `20260608_oracle_3judge`: patterns oracle_t1_p0, p1, p4, p5, p6, p7, p8 (7 cells)

13 (packet × pattern) judging cells in total.

**Reason.** On the Claude Code judging path, Anthropic's automated Usage Policy (AUP)
classifier refuses to process this query's report, returning *"Claude Code is unable to
respond to this request, which appears to violate our Usage Policy"* before any rubric
evaluation can run. This is a false positive — the content is routine receptor
pharmacology (drug-compound binding language is the most likely trigger). The refusal is
**reproducible**, confirmed across two independent fresh-context attempts:
- main session, request `req_011CbuWLnxL8CQm7Ucet1irb`
- an isolated single-task subagent, request `req_011CbuY6E7EGDw8Pw67dm1B5`

It is therefore a property of the judging *mechanism* (the Claude Code path), not a
transient timeout.

**Scope is narrow and judge-specific — the query is NOT dropped from the study.**
- GPT-5.2 scores `82de3e92` in **every** arm (it is a different model family on the API
  path and is unaffected).
- In the main base study, the Claude panel scored `82de3e92` for base_p0–p10 via a
  separate (non-Claude-Code) mechanism; those verdicts remain valid.
- Only the two Claude-Code re-judge packets above lack Claude-panel verdicts for this one
  query.

**Effect on results.** Downstream per-cell means are computed with `dropna`, so the only
effect is that each affected cell's n is reduced by 1 (e.g. the oracle_t1 Claude-panel
cells rest on the other queries in that arm). No denominator is hard-coded to include it;
no analysis breaks. The exclusion is recorded machine-readably in
`reports/claude_code_judging/<packet>/quarantine.json`.

Recorded 2026-06-10.

## Data-hygiene note — dual Claude judge baselines for the oracle arm

The oracle-retrieval analysis on the Claude judges must pair each oracle report against a
**within-version** baseline. There are two Claude base re-scorings on disk:
`results/judge_claude_opus/` and `results/judge_claude_sonnet/` (the version that *also* scored
the oracle reports) versus `results/judge_claude_opus48/` and `results/judge_claude_sonnet48/`
(the within-version baseline). The two disagree by ~0.22 on citation_quality for a fixed report —
a pure judge-version artifact, not an oracle effect. `df_scores.parquet` stores the *former*
(`judge_claude_opus` base = 0.76 for base_p1 citation), so any oracle-minus-base delta computed on
the Claude judges directly from `df_scores` is contaminated. The canonical build
(`build_oracle_opus.py`) deliberately pairs `judge_claude_opus/oracle_t1_{p}` against
`judge_claude_opus48/base_{p}` so the delta carries no version artifact, which is why the
canonical Claude oracle result is citation-up / factual-flat on all three judges. Any future
oracle analysis on Claude judges MUST use the `*48` within-version base, not the `df_scores`
Claude base. (Audit RA5, 2026-06-10.)

## E2 — Report-generation failures in the base panel (the headline `n_queries` holes)

**What is missing.** Six (pattern, query) reports in the headline base panel were never
produced because the generation pipeline errored (`df_runs.status == "error"`,
`report_exists == False`). With no report, **no judge** can score them, so each absent
report removes one (pattern, query) cell from all three panel judges simultaneously.

| pattern | query_id | source / difficulty | status |
|---|---|---|---|
| `base_p3`  | `82508e50-497c-445a-b1dd-fd9d7e6dafda` | DRACO / Academic / complex | error |
| `base_p5`  | `82508e50-497c-445a-b1dd-fd9d7e6dafda` | DRACO / Academic / complex | error |
| `base_p6`  | `82508e50-497c-445a-b1dd-fd9d7e6dafda` | DRACO / Academic / complex | error |
| `base_p6`  | `0a652d00-5c22-4621-8ec4-dd92b1f1450b` | DRACO / Personalized-Assistant / complex | error |
| `base_p6`  | `dsqa_0683`                              | DeepSearchQA / moderate    | error |
| `base_p11` | `82508e50-497c-445a-b1dd-fd9d7e6dafda` | DRACO / Academic / complex | error |

**These are exactly the headline `n_queries` holes.** They reproduce the canonical
`single_judge_gpt52` denominators with no residual: P3 = 89 (1 hole), P5 = 89 (1 hole),
P6 = 87 (3 holes), P11 gpt52 = 89 (1 hole). All other headline patterns are at the full 90.
`82508e50` (a complex DRACO "Academic" query) is a recurrent failure across P3/P5/P6/P11 —
a property of that query's generation difficulty, not of any one architecture.

**Effect on results.** Per-cell means use `dropna`; the only effect is the per-pattern n
shown above. No denominator is hard-coded to 90. Recorded 2026-06-10 (audit CG-8).

## E3 — Claude-panel judging coverage holes (reports exist; only the Claude judges lack a verdict)

Reconciling expected cells (patterns × queries × judges) against `df_verdicts` for the
3-judge base panel leaves **51 missing (pattern, query, judge) cells** that are NOT covered
by E1 (82de3e92) or E2 (missing reports): the report exists and GPT-5.2 scored it, but the
Claude panel has no verdict. They fall into two groups, both legitimate but previously
undocumented.

### E3a — P11/P12 were Claude-judged on a deliberately reduced query subset (47 cells)

P11 (the GPT-4o verbatim ReAct controller) and P12 (the GRPO-trained 7B agent) are the two
late-added post-hoc patterns.
Their **Claude-panel** verdicts were produced *only* via the manual Claude-Code judging path,
in packet `reports/claude_code_judging/20260604_e2_p11_p12/` (promoted to
`results/judge_claude_sonnet/`). That packet covers **80 of 90** P11 queries and **52 of 90**
P12 queries by design — these were the subsets prepared for manual judging — and **Claude
Opus was never ingested for P11/P12 at all** (canonical `mean_opus = NaN` for both; see
`canonical_numbers.json`). The missing Claude-Sonnet cells are precisely the queries outside
those packet subsets:

- `base_p11`, `claude_sonnet`: **9 cells** (80 of 90 judged; the 90th gpt52 hole `82508e50`
  is the E2 report failure, leaving 9 Sonnet-only residual queries).
- `base_p12`, `claude_sonnet`: **38 cells** (52 of 90 judged).

In the paper's analyses P11/P12 are single-judge (GPT-5.2) probes by design: the partial
Claude-Sonnet re-score (80 and 52 cells) exists in the released data and agrees in
direction, but it is excluded from the paper's analyses for comparability, and P11/P12
never enter the 3-judge Opus-bearing comparisons. The reduced Sonnet subset is a property
of the manual judging budget for the two newest patterns, not data loss.

### E3b — Two base reports missing from *both* Claude judges (4 cells)

Two specific base reports exist and were scored by GPT-5.2 but have no Claude-panel verdict
from either Opus or Sonnet (no result file in `results/judge_claude_opus/` or
`results/judge_claude_sonnet/`):

| pattern | query_id | source / domain / difficulty | judges missing |
|---|---|---|---|
| `base_p1` | `8e99d8d2-f6b9-4800-83a9-6f56829898fe` | DRACO / Law / complex | claude_opus, claude_sonnet |
| `base_p3` | `b3c576e7-dfc6-403f-90e7-53c011884d5c` | DRACO / General-Knowledge / complex | claude_opus, claude_sonnet |

These are **not** AUP refusals (distinct from E1: the domains are Law and General Knowledge,
not biomedical pharmacology, and there is no quarantine record) and **not** missing reports
(E2: both reports exist and GPT-5.2 judged them). They are incidental Claude-API judging
gaps — the Claude panel run produced 89/88 of the 90/89 available reports for these two
patterns and these two single reports were never backfilled. They lower the Claude-panel n
for `base_p1` (89 of 90) and `base_p3` (88 of 89) by one each. Per-cell means use `dropna`;
no analysis is affected beyond the n reduction. Not unexplained — a known, bounded backfill
gap, recorded here for completeness.

**Reconciliation status (audit CG-8).** After E1–E3, every missing cell in every analysis
arm is accounted for: the 14 oracle 82de3e92 Claude cells (E1), the 6 base report failures
(E2), the 47 P11/P12 reduced-subset Claude cells (E3a), and the 4 incidental Claude backfill
cells (E3b). The single-judge variance/disentanglement arms, the gpt52+sonnet ablation arm,
and the gpt52+claude_code+sonnet protocol_a probe are partial **by design** (single- or
dual-judge arms; their non-panel judges are not expected). No truly unexplained gap remains.
See `scripts/reconcile_exclusions.py` for the machine check. Recorded 2026-06-10.
