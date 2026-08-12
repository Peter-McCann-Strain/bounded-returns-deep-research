# Pre-registration — E3 N-EFF: effective judge count and family dependence

**Registered:** 2026-06-11 (before first run). **Cost:** $0 CPU (Phase 1). **Canonical key:**
`n_eff` (the existing `irr` key holds pairwise correlations only; this adds effective count).

## Positioning (binding amendment)
Primary frame = the **fully-crossed gpt52 × opus × sonnet cell** (criterion verdicts where
all three judges scored the same pattern × query × criterion). Drop any three-Claude-triangle
claim (opus × claude_code overlap is n=0). Treat claude_code pairs as a labelled SECONDARY
analysis. Comply with the Rao & Callison-Burch reporting checklist (2606.00093).

## Question
What is the empirical effective number of independent judges in the real multi-family panel,
and is the two-Claude citation correlation an identified shared confounder (CARE sense)?

## Data
`df_verdicts`, restricted to the (pattern × query × criterion_id) cells with verdicts from
all three panel judges (gpt52, claude_opus, claude_sonnet). Verified n of the fully-crossed
cell is reported in the build (plan cites ≈44,425 criterion verdicts; confirm from disk).

## Hypotheses / endpoints
- **Primary endpoint.** Empirical **N_eff** for the 3-judge panel, computed from the
  inter-judge agreement structure (within-family vs cross-family). Report N_eff with the
  method named (correlation-based effective count; plus CARE 2603.00039 and Ising 2601.22336
  aggregation if their public code installs cleanly — if not, report the correlation-based
  N_eff and log the dependency gap honestly).
- Within-family (opus–sonnet) vs cross-family (gpt52–Claude) agreement per dimension and
  per criterion.
- Whether the recovered shared factor loads on citation density, report length, hedging
  (regress the latent/residual on these covariates).

## n and exclusions
Fully-crossed criterion verdicts only; `satisfied_is_known == True`; EXCLUSIONS.md respected.
Sonnet criterion verdicts are clean (only its stored *overall* is corrupt). Determinism:
seeded generator on sorted inputs for any resampling-based CI on N_eff.

## Analysis reported regardless of outcome
The N_eff number and the within-vs-cross agreement table ship whatever they show. N_eff ≈ 3
(judges independent) refutes the correlated-panel concern and is reported as such; N_eff ≈ 2
(the two Claudes collapse to ~one vote) is the headline-(iii) support and is reported with
its CI. Phase-2 (add gpt-4o + gpt-4.1 as judges 5-6, the symmetric within-OpenAI cell) is a
separate later step and is not part of this $0 registration.
