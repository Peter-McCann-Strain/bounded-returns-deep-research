#!/usr/bin/env python3
"""E4 (corrected): re-judge the 30-report calibration pack with the current
Claude Opus model via Claude Code Agent dispatch, using the original
`scripts/claude_judge_eval.py` rubric/prompt architecture.

This mirrors the architecture documented in `docs/claude_code_evaluation.md`:
  - Each agent receives the canonical 38-criterion V2 rubric prompt for ONE report
  - Returns a strict JSON evaluations list (criterion_index + verdict + evidence + reasoning)
  - Output is parsed via `claude_judge_eval.parse_and_save_result()` into the
    SAME schema as the existing 3-judge panel (results/judge_claude_opus/...)

Outputs go to `results/judge_claude_opus_v47/{pattern}/{query_id}.json` to
keep them separate from the original Opus 4.1 panel data while remaining shape-
compatible for direct comparison.

Usage:
    python scripts/run_e4_opus_calibration.py prep
    # then dispatch Claude Code agents using files in tasks/
    python scripts/run_e4_opus_calibration.py parse <anon_id> <agent_response_file>
    python scripts/run_e4_opus_calibration.py analyse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.claude_judge_eval import build_judge_prompt

PACK_DIR = Path("data/human_calibration_pack")
KEY_PATH = PACK_DIR / "ANONYMIZED_KEY.csv"

PHASE_DIR = Path("reports/phase10_claude_code_judge")
TASKS_DIR = PHASE_DIR / "tasks"
RESPONSES_DIR = PHASE_DIR / "responses"
JUDGE_OUT = Path("results/judge_claude_opus_v47")


def prep():
    """Build a task JSON per calibration-pack report containing the canonical prompt."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    key_df = pd.read_csv(KEY_PATH)
    print(f"Building criterion-level rubric prompts for {len(key_df)} calibration reports …")
    written = 0
    for _, row in key_df.iterrows():
        anon = row["anon_id"]
        pattern = row["pattern"]
        qid = row["query_id"]
        try:
            prompt_pack = build_judge_prompt(pattern, qid)
        except FileNotFoundError as e:
            print(f"  WARN: {anon} — {e}")
            continue
        # Stash everything an agent + parser needs
        task = {
            "anon_id": anon,
            "pattern": pattern,
            "query_id": qid,
            "n_criteria": prompt_pack["criteria_count"],
            "system_prompt": prompt_pack["system_prompt"],
            "user_prompt": prompt_pack["user_prompt"],
            "criteria_texts": prompt_pack["criteria_texts"],
            "criteria_dimensions": prompt_pack["criteria_dimensions"],
            "dimension_weights": prompt_pack["dimension_weights"],
        }
        (TASKS_DIR / f"{anon}.json").write_text(json.dumps(task, indent=2, default=str))
        written += 1
    print(f"  wrote {written} task files -> {TASKS_DIR}")
    print()
    print("NEXT: dispatch one Claude Code Agent per task. Each agent:")
    print("  1. Reads tasks/{anon_id}.json")
    print("  2. Generates JSON with shape: {\"evaluations\": [{criterion_index, verdict, evidence, reasoning}, ...]}")
    print(f"  3. Writes its raw JSON to {RESPONSES_DIR}/{{anon_id}}_response.json")
    print()
    print("Then run: python scripts/run_e4_opus_calibration.py parse-all")
    print(f"  This drives parse_and_save_result() into {JUDGE_OUT}/")


def parse_one(anon_id: str, response_path: Path | None = None):
    """Parse one agent response and save through claude_judge_eval helpers."""
    task_path = TASKS_DIR / f"{anon_id}.json"
    if not task_path.exists():
        print(f"ERROR: {task_path} missing"); return False
    if response_path is None:
        response_path = RESPONSES_DIR / f"{anon_id}_response.json"
    if not response_path.exists():
        print(f"ERROR: {response_path} missing"); return False

    task = json.loads(task_path.read_text())
    raw = json.loads(response_path.read_text())
    evaluations = raw.get("evaluations", [])
    if not evaluations:
        print(f"  {anon_id}: empty evaluations"); return False

    # Inline the parse logic so we never touch the existing Opus 4.1 panel data dir.
    crit_texts = task["criteria_texts"]
    crit_dims = task["criteria_dimensions"]
    dim_weights = task["dimension_weights"]

    dim_stats: dict[str, dict] = {dim: {"met": 0, "total": 0} for dim in set(crit_dims)}
    verdicts = []
    for ev in evaluations:
        idx = ev.get("criterion_index", -1)
        if idx < 0 or idx >= len(crit_texts):
            continue
        satisfied = str(ev.get("verdict", "")).upper() == "SATISFIED"
        dim = crit_dims[idx]
        dim_stats[dim]["total"] += 1
        if satisfied:
            dim_stats[dim]["met"] += 1
        verdicts.append({
            "criterion_index": idx,
            "criterion": crit_texts[idx],
            "dimension": dim,
            "satisfied": satisfied,
            "evidence": ev.get("evidence", ""),
            "reasoning": ev.get("reasoning", ""),
        })
    dimensions = {
        dim: {"score": round(s["met"] / s["total"], 4) if s["total"] else 0.0,
              "met": s["met"], "total": s["total"]}
        for dim, s in dim_stats.items()
    }
    overall = sum(dimensions.get(d, {}).get("score", 0) * w for d, w in dim_weights.items())

    result = {
        "query_id": task["query_id"],
        "pattern": task["pattern"],
        "judge_model": "claude-opus-4.7",
        "overall_score": round(overall, 4),
        "dimensions": dimensions,
        "verdicts": verdicts,
        "n_criteria": len(verdicts),
        "n_satisfied": sum(1 for v in verdicts if v["satisfied"]),
    }
    dst = JUDGE_OUT / task["pattern"] / f"{task['query_id']}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2, default=str))
    print(f"  {anon_id} -> {dst}  overall={result['overall_score']:.3f}")
    return True


def parse_all():
    key_df = pd.read_csv(KEY_PATH)
    n_ok = 0
    for anon in key_df["anon_id"]:
        if parse_one(anon):
            n_ok += 1
    print(f"\nParsed {n_ok}/{len(key_df)} responses")


def analyse():
    """Compare Opus-4.7 calibration verdicts to the existing Opus-4.6 + GPT-5.2 + Sonnet panel."""
    key_df = pd.read_csv(KEY_PATH)
    print(f"Comparing {len(key_df)} calibration-pack scores Opus-4.7 vs existing 3-judge panel …\n")

    rows = []
    for _, row in key_df.iterrows():
        pattern = row["pattern"]
        qid = row["query_id"]
        v47_path = JUDGE_OUT / pattern / f"{qid}.json"
        v46_path = Path("results/judge_claude_opus") / pattern / f"{qid}.json"
        gpt_path = Path("results/judge_gpt52") / pattern / f"{qid}.json"
        son_path = Path("results/judge_claude_sonnet") / pattern / f"{qid}.json"
        if not v47_path.exists():
            continue
        v47 = json.loads(v47_path.read_text())
        v46 = json.loads(v46_path.read_text()) if v46_path.exists() else None
        gpt = json.loads(gpt_path.read_text()) if gpt_path.exists() else None
        son = json.loads(son_path.read_text()) if son_path.exists() else None
        rows.append({
            "anon_id": row["anon_id"], "pattern": pattern, "query_id": qid,
            "opus47": v47["overall_score"],
            "opus46": v46.get("overall_score") if v46 else None,
            "gpt52": gpt.get("overall_score") if gpt else None,
            "sonnet": son.get("overall_score") if son else None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No Opus-4.7 results yet — run prep + agent dispatches + parse-all first.")
        return
    print(df.select_dtypes("number").describe())
    out = PHASE_DIR / "calibration_compare.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote: {out}")

    # Pearson and mean |Δ| of Opus-4.7 vs each other judge
    from scipy.stats import pearsonr
    for col in ["opus46", "gpt52", "sonnet"]:
        sub = df.dropna(subset=["opus47", col])
        if len(sub) < 5:
            continue
        r, p = pearsonr(sub["opus47"], sub[col])
        mad = (sub["opus47"] - sub[col]).abs().mean()
        print(f"  Opus-4.7 vs {col:7s}: r={r:+.3f} (n={len(sub)}, p={p:.3g}, mean|Δ|={mad:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prep", "parse-all", "analyse"])
    parser.add_argument("--anon", default=None)
    args = parser.parse_args()
    if args.mode == "prep":
        prep()
    elif args.mode == "parse-all":
        parse_all()
    else:
        analyse()
