#!/usr/bin/env python3
"""V7 reviewer Q2 — Tavily Sonnet re-judge prep.

Generates per-report judge-prompt task files for a stratified 24-report
subset of the §6.2.1 Tavily intervention (4 reports per pattern × 6 patterns).
Each task file contains the canonical 38-criterion V2 rubric prompt for ONE
`protocol_a_tavily_p{N}/{qid}.md` report; a Claude Code agent reads the
task file, evaluates the report, and writes JSON evaluations.

Usage:
    python scripts/protocol_a_sonnet_prep.py
    # then dispatch agents via Claude Code Agent tool (one per task file)
    python scripts/protocol_a_sonnet_parse.py  # after responses arrive
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.claude_judge_eval import build_judge_prompt, RESULTS_BASE

PATTERNS = ["protocol_a_tavily_p0", "protocol_a_tavily_p1", "protocol_a_tavily_p3",
            "protocol_a_tavily_p4", "protocol_a_tavily_p5", "protocol_a_tavily_p8"]
N_PER_PATTERN = 4
SEED = 42

OUT_TASKS = ROOT / "reports" / "phase15_tavily_sonnet" / "tasks"
OUT_RESPONSES = ROOT / "reports" / "phase15_tavily_sonnet" / "responses"


def main():
    OUT_TASKS.mkdir(parents=True, exist_ok=True)
    OUT_RESPONSES.mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(SEED)

    written: list[dict] = []
    for pat in PATTERNS:
        pat_dir = RESULTS_BASE / pat
        if not pat_dir.exists():
            print(f"WARN: {pat_dir} missing; skipping")
            continue
        # Pick reports with non-trivial size (>5 KB) — exclude failed runs
        all_reports = [(p.stem, p.stat().st_size) for p in pat_dir.glob("*.md")]
        all_reports = [(qid, sz) for qid, sz in all_reports if sz > 5000]
        all_reports.sort()
        if not all_reports:
            continue
        rng.shuffle(all_reports)
        sub = [qid for qid, _ in all_reports[:N_PER_PATTERN]]
        for qid in sub:
            try:
                info = build_judge_prompt(pat, qid)
            except Exception as e:
                print(f"  skip {pat}/{qid}: {e}")
                continue
            task_path = OUT_TASKS / f"{pat}__{qid}.json"
            task_path.write_text(json.dumps({
                "pattern": pat, "query_id": qid,
                "system_prompt": info["system_prompt"],
                "user_prompt": info["user_prompt"],
                "criteria_count": info["criteria_count"],
                "criteria_texts": info["criteria_texts"],
                "criteria_dimensions": info["criteria_dimensions"],
                "dimension_weights": info["dimension_weights"],
                "report_path": info["report_path"],
            }))
            written.append({"pattern": pat, "query_id": qid,
                            "n_criteria": info["criteria_count"]})

    summary = OUT_TASKS.parent / "tasks_summary.json"
    summary.write_text(json.dumps(written, indent=2))
    print(f"Wrote {len(written)} task files to {OUT_TASKS}")
    print(f"Pattern distribution:")
    from collections import Counter
    for pat, n in Counter(t["pattern"] for t in written).items():
        print(f"  {pat}: {n}")
    print(f"\nNext: dispatch Claude Code Sonnet agents — one per task file in {OUT_TASKS}")


if __name__ == "__main__":
    main()
