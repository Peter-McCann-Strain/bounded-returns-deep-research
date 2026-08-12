# Pre-registration — E13′ source-4: judge-vs-human-gold registry

**Registered:** 2026-06-11 (before first run). **Experiment:** E13′ PERTURB-TRUTH, source 4
(manifest human gold from existing verdicts). **Analyst:** autonomous execution team.
**Cost:** $0, CPU-only, no new judging. **Canonical key:** `judge_vs_gold`.

## Motivation
Every benchmark in the manifest ships human-authored ground truth not yet systematically
exploited. The cheapest human-anchored validity number in the programme is judge-vs-gold
agreement computed from the existing 248,536 verdicts. This is the first real input to the
"no human anchor" defence (risk register) and a contributor to gate G4.

## Scope of THIS registration (mechanical-gold slice)
On-disk mechanical gold exists only where `reference_answer` is an objective verifiable
answer: **LitQA2 (10 queries) + DeepSearchQA (19 with answers) = the mechanical slice.**
Each carries a tagged answer-bearing criterion (`source:"benchmark"`, dimension
`factual_accuracy`, e.g. "The report identifies the correct answer: Ogfrl1"). The DRACO
(1,503 expert) and ResearchQA (115) criterion tiers have no per-report gold label on disk
and are explicitly OUT of scope here; they are deferred to the entailment-pass / raw-DRACO
extension and will get their own registration.

## Hypotheses
- **H1 (primary).** On the mechanical slice, the panel judges' `satisfied` verdict on the
  answer-bearing criterion agrees with the mechanical answer-presence gold at rate > chance,
  and the agreement rate differs by judge family.
- **H2 (artefact-relevant).** Per dimension, factual_accuracy verdicts track the mechanical
  gold more closely than citation_quality does (citations are not answer-determined).

## Primary endpoint
Per-judge **balanced agreement** (and Cohen's κ vs the mechanical gold) on the answer-bearing
criterion verdict, pooled over the mechanical slice; reported per judge, per family, per
dimension. Mechanical gold = robust presence check of `reference_answer` in the report text
(LitQA2: single-token; DeepSearchQA: all comma-separated gold tokens present), computed with
BOTH strict (all tokens) and lenient (any token) matchers; sensitivity to matcher reported.

## n and units
Units = (pattern × query × judge) answer-criterion verdicts over the mechanical slice.
Reports read from `df_runs.report_path`. Expected order: ~29 queries × ~11 base patterns ×
3-4 judges, minus missing reports (E2 holes) and the 82de3e92 quarantine.

## Exclusion rules
- Respect `EXCLUSIONS.md`: 82de3e92 absent from Claude-Code panels; missing reports (E2)
  drop their cells; sonnet `overall_score` corruption irrelevant here (we use criterion
  verdicts, not overall). Use `satisfied_is_known == True` only.
- Drop any query whose mechanical match is ambiguous under the two matchers (disagree),
  and REPORT the count dropped — never silently.

## Analysis reported regardless of outcome
The per-judge / per-family agreement table ships whatever it shows. If all families agree
with gold equally (no family signal), that is reported as the null and weakens the
"Claude-specific artefact" reading; if Claude families agree LESS with factual gold while
scoring citation-bearing criteria higher, that is logged as a G4 input favouring the
density-artefact headline. Mechanical-gold noise (matcher sensitivity) is stated as the
key limitation; this slice anchors, it does not certify.
