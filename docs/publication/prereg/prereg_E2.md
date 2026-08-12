# Pre-registration — E2 RANDOMNESS-DR: variance decomposition

**Registered:** 2026-06-11 (before first run). **Cost:** $0 CPU (core). **Canonical key:**
`variance_decomposition` (extends, does not overwrite, the existing `variance_components`).

## Positioning (binding amendment)
Cite Wang 2512.21326 ("Measuring all the noises") and CyclicJudge 2603.01865 up front;
position as the **first full-pipeline replicate variance study for live-retrieval long-form
report agents** (Wang covers short-form; ICC paper 2512.06710 covers GAIA only).

## Question
How large are run, query, and judge noise components for rubric-judged long-form reports,
and how often would single-run leaderboards flip?

## Data
The replicate corpus on disk: 12×P0 + P1/P4/P10 replicates over the 30 variance queries
(`pattern_family == "variance"`), scored by the panel. CPU-only, no new runs in the core.

## Hypotheses / endpoints
- **Primary endpoint.** Three-way **run × query × judge** variance decomposition of the
  overall score (using `overall_score_recomputed`; sonnet stored overall excluded per the
  data dictionary). Report σ²_run, σ²_query, σ²_judge, σ²_resid and the ICCs. The existing
  `variance_components` (query/judge/resid only, n=3262) is the prior; the NEW key adds an
  explicit **run** facet and the items below.
- Criterion-level verdict **flip-rate** per dimension across replicates (binary `satisfied`).
- Citation / reference **overlap** across replicates (the long-form analogue short-form
  studies cannot measure).
- **MDE / power curves** as a function of n_queries and n_replicates.
- **Single-run leaderboard flip simulation**: resample one run per architecture, recompute
  the ranking, report the false-discovery rate of single-run rankings.

## n and exclusions
Variance-query replicate cells only; `satisfied_is_known` filter for flip-rates; 82de3e92
and missing reports excluded per EXCLUSIONS.md. Determinism: all bootstrap/permutation via a
dedicated seeded generator on SORTED inputs (set-hash drift rule).

## Analysis reported regardless of outcome
The decomposition + flip-rate + MDE table ships whatever the magnitudes are. A finding of
small run noise is as publishable as large (it bounds how much single-run leaderboards can be
trusted). The workshop 4-pager (Paper 3) ships on the core regardless; the 4-architecture
12-replicate extension runs only if reviews demand it (plan gate).
