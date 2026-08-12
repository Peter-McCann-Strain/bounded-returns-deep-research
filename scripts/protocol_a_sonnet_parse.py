#!/usr/bin/env python3
"""V7 reviewer Q2 — parse Sonnet agent responses + compute paired bootstrap.

Reads the response files at reports/phase15_tavily_sonnet/responses/{pattern}__{qid}.json,
parses them via claude_judge_eval.parse_and_save_result() (which produces the
canonical results/judge_claude_sonnet/{pattern}/{qid}.json schema), then
computes per-pattern Sonnet means + paired bootstrap CIs against the
existing Bing-side base_p{N} GPT-5.2 reports.

Outputs:
  results/judge_claude_sonnet/{pattern}/{qid}.json     # canonical per-report
  reports/phase15_tavily_sonnet/sonnet_summary.md      # per-pattern table
  reports/phase15_tavily_sonnet/sonnet_summary.csv     # raw bootstrap deltas
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.claude_judge_eval import parse_and_save_result

RESPONSES = ROOT / "reports" / "phase15_tavily_sonnet" / "responses"
TASKS = ROOT / "reports" / "phase15_tavily_sonnet" / "tasks"
OUT_DIR = ROOT / "reports" / "phase15_tavily_sonnet"


def main():
    response_files = sorted(RESPONSES.glob("*.json"))
    print(f"Parsing {len(response_files)} Sonnet agent responses …")

    parsed: list[dict] = []
    for rf in response_files:
        try:
            data = json.loads(rf.read_text())
            evals = data.get("evaluations") or data.get("evaluations_list") or data
            if not isinstance(evals, list):
                print(f"  skip {rf.name}: no evaluations list")
                continue
            # Find the matching task spec
            task_path = TASKS / rf.name
            if not task_path.exists():
                print(f"  skip {rf.name}: task spec missing")
                continue
            task = json.loads(task_path.read_text())
            res = parse_and_save_result(
                pattern_name=task["pattern"], query_id=task["query_id"],
                evaluations=evals,
                criteria_texts=task["criteria_texts"],
                criteria_dimensions=task["criteria_dimensions"],
                dimension_weights=task["dimension_weights"],
                judge_model="claude-sonnet-4.5",
            )
            parsed.append({
                "pattern": task["pattern"], "query_id": task["query_id"],
                "overall_score": res["overall_score"],
                "n_criteria": res["n_criteria"], "n_satisfied": res["n_satisfied"],
            })
        except Exception as e:
            print(f"  FAIL {rf.name}: {e}")

    df_sonnet = pd.DataFrame(parsed)
    print(f"Parsed {len(df_sonnet)} Sonnet judgments across {df_sonnet['pattern'].nunique()} patterns")

    # Compare to existing GPT-5.2 Bing means via paired bootstrap
    df_overall = pd.read_parquet(ROOT / "data" / "analysis" / "df_overall_scores.parquet")
    gpt = df_overall[df_overall["judge"] == "gpt52"][["pattern", "query_id", "overall_score"]]

    rows = []
    for tav_pat in sorted(df_sonnet["pattern"].unique()):
        bing_pat = "base_" + tav_pat.replace("protocol_a_tavily_", "")
        sonnet_sub = df_sonnet[df_sonnet["pattern"] == tav_pat][["query_id", "overall_score"]]
        bing_sub = gpt[gpt["pattern"] == bing_pat][["query_id", "overall_score"]]
        gpt_tav = gpt[gpt["pattern"] == tav_pat][["query_id", "overall_score"]]
        if sonnet_sub.empty or bing_sub.empty or gpt_tav.empty:
            continue

        # Pair Sonnet-Tavily ↔ GPT-Bing (cross-judge mismatch on purpose: V7 asks
        # whether Sonnet on Tavily reports also depresses vs GPT-5.2 on Bing baseline)
        merged = sonnet_sub.merge(bing_sub, on="query_id",
                                   suffixes=("_sonnet_tav", "_gpt_bing"))
        # Also pair Sonnet-Tavily ↔ GPT-Tavily (judge-only difference) and
        # Sonnet-Tavily ↔ GPT-Bing (combined judge × backend)
        merged_jb = merged
        merged_judge_only = sonnet_sub.merge(gpt_tav, on="query_id",
                                              suffixes=("_sonnet_tav", "_gpt_tav"))

        deltas_jb = (merged_jb["overall_score_sonnet_tav"]
                     - merged_jb["overall_score_gpt_bing"]).to_numpy()
        deltas_judge_only = (merged_judge_only["overall_score_sonnet_tav"]
                              - merged_judge_only["overall_score_gpt_tav"]).to_numpy()

        rng = np.random.default_rng(42)
        def boot(d):
            if len(d) < 2: return None, None, None
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        m_jb, lo_jb, hi_jb = boot(deltas_jb)
        m_jo, lo_jo, hi_jo = boot(deltas_judge_only)
        rows.append({
            "tavily_pattern": tav_pat,
            "n": len(deltas_jb),
            "sonnet_tav_mean": float(sonnet_sub["overall_score"].mean()),
            "gpt_bing_mean": float(bing_sub["overall_score"].mean()),
            "gpt_tav_mean": float(gpt_tav["overall_score"].mean()),
            "delta_sonnetTav_minus_gptBing": m_jb,
            "ci_jb": (lo_jb, hi_jb),
            "delta_sonnetTav_minus_gptTav": m_jo,
            "ci_jo": (lo_jo, hi_jo),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "sonnet_summary.csv", index=False)
    print()
    print(out_df.to_string(index=False))

    md = [
        f"# §6.2.1 Tavily Sonnet re-judge (V7 reviewer Q2)",
        f"\nSample: {len(df_sonnet)} stratified Tavily reports re-judged by Claude Sonnet 4.5 via Claude Code Agent dispatch.",
        f"\nDelta interpretations:",
        f"- `Δ(Sonnet-Tavily − GPT-5.2-Bing)`: combined judge × backend effect (the cross-experiment delta the original §6.2.1 measured with GPT-5.2 only)",
        f"- `Δ(Sonnet-Tavily − GPT-5.2-Tavily)`: pure cross-judge effect on the *same* Tavily reports, isolating whether Sonnet rates Tavily reports higher or lower than GPT-5.2",
        f"\n## Per-pattern (paired bootstrap N=2,000, seed=42)\n",
        f"| Pattern | n | Sonnet-Tav mean | GPT-Bing mean | GPT-Tav mean | Δ (Sonnet-Tav − GPT-Bing) | 95% CI | Δ (Sonnet-Tav − GPT-Tav) | 95% CI |",
        f"|---|---:|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for r in rows:
        md.append(f"| {r['tavily_pattern']} | {r['n']} | {r['sonnet_tav_mean']:.3f} | "
                  f"{r['gpt_bing_mean']:.3f} | {r['gpt_tav_mean']:.3f} | "
                  f"{r['delta_sonnetTav_minus_gptBing']:+.3f} | "
                  f"({r['ci_jb'][0]:+.3f}, {r['ci_jb'][1]:+.3f}) | "
                  f"{r['delta_sonnetTav_minus_gptTav']:+.3f} | "
                  f"({r['ci_jo'][0]:+.3f}, {r['ci_jo'][1]:+.3f}) |")

    if rows:
        all_jb = np.concatenate([
            (df_sonnet[df_sonnet["pattern"] == r["tavily_pattern"]]
                .merge(gpt[gpt["pattern"] == "base_" + r["tavily_pattern"].replace("protocol_a_tavily_", "")],
                       on="query_id", suffixes=("_sonnet", "_gpt")))
            .pipe(lambda d: (d["overall_score_sonnet"] - d["overall_score_gpt"]).to_numpy())
            for r in rows
        ])
        rng = np.random.default_rng(42)
        boots = np.array([all_jb[rng.integers(0, len(all_jb), len(all_jb))].mean() for _ in range(2000)])
        md.append(f"\n**Pooled (n={len(all_jb)}):** mean Δ(Sonnet-Tav − GPT-Bing) = {all_jb.mean():+.3f}, "
                  f"95% CI ({np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f})")
        md.append(f"\n## Reading\n")
        if (out_df['delta_sonnetTav_minus_gptBing'] < 0).all():
            md.append("**All per-pattern Sonnet-Tavily − GPT-Bing deltas are negative**, replicating "
                      "the universal-direction depression the original §6.2.1 GPT-5.2-only intervention reported. "
                      "The retrieval-bottleneck reframing (LLM judges reward citation density irrespective "
                      "of source) survives a non-GPT-family judge axis: Sonnet on Tavily reports also "
                      "scores lower than GPT-5.2 on Bing reports, with the same direction.")
        elif (out_df['delta_sonnetTav_minus_gptBing'] < 0).sum() >= len(out_df) - 1:
            md.append("**5 of 6 per-pattern Sonnet-Tavily − GPT-Bing deltas are negative**, mostly "
                      "replicating the §6.2.1 universal-direction finding under a non-GPT judge axis. "
                      "The single exception suggests judge-specific reweighting on at least one pattern, "
                      "but the directional reading is preserved.")
        else:
            md.append("**Pattern-level deltas are mixed under the Sonnet axis**, indicating the "
                      "§6.2.1 universal-direction depression may be partially judge-specific. "
                      "We report this as a Tier-2 finding requiring further full-panel verification.")
    (OUT_DIR / "sonnet_summary.md").write_text("\n".join(md))
    print(f"\nWrote {OUT_DIR / 'sonnet_summary.md'}")


if __name__ == "__main__":
    main()
