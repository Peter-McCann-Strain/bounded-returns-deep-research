#!/usr/bin/env python
"""Bing-vs-Tavily citation VOLUME (count), matched query_ids, mirroring the score-level
extended.bing_vs_tavily computation in build_numbers_extended.py. Lands the per-pattern
% fewer citations under Tavily that the prose in main.tex's oracle-retrieval section
("Tavily returns X-Y% fewer citations") had previously stated without a canonical key
behind it (caught by an adversarial review pass, 2026-07-28).
"""
import json
import os

import pandas as pd

from deep_research.paths import PROJECT_ROOT

def _merge(dst, src):
    """Overlay src onto dst, preserving dst keys src does not produce.
    Assigning the whole subtree destroyed sibling keys written by other scripts (75 leaves
    across the two bing_vs_tavily branches, silently, on every run). Overlay, never replace."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict): _merge(dst[k], v)
        else: dst[k] = v
    return dst


ROOT = str(PROJECT_ROOT)
A = f"{ROOT}/data/analysis"
ANA = f"{ROOT}/papers/paper_a_bounded_returns/analysis"

cit = pd.read_parquet(f"{A}/df_citations_protocol_a.parquet")
counts = cit.groupby(["pattern", "query_id"]).size().rename("n_cites").reset_index()
MIN_WORDS = 200   # stable from 200w upward
_runs = pd.read_parquet(f"{A}/df_runs.parquet")
_runs["pattern"] = _runs["pattern"].astype(str)
_ok = _runs[(_runs.report_exists) & (_runs.report_word_count >= MIN_WORDS)]

out = {}
for tp in sorted(p for p in counts.pattern.unique() if p.startswith("protocol_a_tavily_")):
    pid = tp.replace("protocol_a_tavily_", "")
    base_pat = f"base_{pid}"
    tav = counts[counts.pattern == tp].set_index("query_id")["n_cites"]
    bing = counts[counts.pattern == base_pat].set_index("query_id")["n_cites"]
    # Matching on citation-frame indices drops every zero-citation report, which
    # conditions on the outcome variable (P3 read 16% instead of 40%). Take the
    # universe from df_runs and zero-fill. MIN_WORDS excludes degenerate generation
    # collapses, which would otherwise be scored as "cited nothing".
    common = sorted(set(_ok[_ok.pattern == tp].query_id) & set(_ok[_ok.pattern == base_pat].query_id))
    if len(common) < 5:
        continue
    tav_c = tav.reindex(common).fillna(0.0)
    bing_c = bing.reindex(common).fillna(0.0)
    pct_fewer_per_query = (1 - tav_c / bing_c).clip(lower=None)
    out[pid.upper()] = {
        "n": int(len(common)),
        "mean_citations_bing": round(float(bing_c.mean()), 2),
        "mean_citations_tavily": round(float(tav_c.mean()), 2),
        "pct_fewer_on_means": round(float(1 - tav_c.mean() / bing_c.mean()), 4),
        "pct_fewer_per_query_median": round(float(pct_fewer_per_query.median()), 4),
        "pct_fewer_per_query_min": round(float(pct_fewer_per_query.min()), 4),
        "pct_fewer_per_query_max": round(float(pct_fewer_per_query.max()), 4),
    }

pooled_pct = [v["pct_fewer_on_means"] for v in out.values()]
result = {
    "per_pattern": out,
    "range_across_patterns_pct_fewer_on_means": [round(min(pooled_pct), 4), round(max(pooled_pct), 4)],
    "note": "Per-pattern % fewer citations under Tavily vs Bing on matched query_ids (same protocol-A "
            "subset and pattern-matching as extended.bing_vs_tavily). pct_fewer_on_means uses each "
            "pattern's mean citation count under each backend (matches how the score-level delta is "
            "reported); the per_query min/max/median fields are the distribution of the same ratio "
            "computed per query before averaging, shown for robustness.",
}

DRY_RUN = ("--dry-run" in __import__("sys").argv) or ("--write" not in __import__("sys").argv)
if DRY_RUN:
    print("[dry-run] computed bing_vs_tavily_citation_volume; NOT writing (pass --write to land).")
else:
    cn = json.load(open(f"{ANA}/canonical_numbers.json"))
    _merge(cn.setdefault("extended", {}).setdefault("bing_vs_tavily_citation_volume", {}), result)
    tmp = f"{ANA}/canonical_numbers.json.tmp"
    open(tmp, "w").write(json.dumps(cn, indent=1))
    os.replace(tmp, f"{ANA}/canonical_numbers.json")

print(json.dumps(result, indent=1))
