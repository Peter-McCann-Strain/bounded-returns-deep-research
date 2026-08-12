#!/usr/bin/env python3
"""Prepare manual Claude Code judging packets.

The packet format is intentionally file-based because these judgments are run
manually in Claude Code, not through the Anthropic API. Each task file contains
the full system/user prompt, rubric mapping, and required JSON schema for one
report. Responses should be saved under the sibling `responses/` directory with
the same filename stem.

Default use prepares the P11/P12 cross-family slice requested in the v9 upgrade
plan:

    python scripts/claude_code_judging_prep.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.claude_judge_eval import build_judge_prompt  # noqa: E402

DEFAULT_QUERY_IDS = ROOT / "data" / "variance_stratified.json"
REPORTS_BASE = ROOT / "results" / "experiments"
OUT_ROOT = ROOT / "reports" / "claude_code_judging"

JSON_SCHEMA_HINT = {
    "evaluations": [
        {
            "criterion_index": 0,
            "verdict": "SATISFIED",
            "evidence": "Short quote or location from the report.",
            "reasoning": "One sentence explaining the judgment.",
        }
    ]
}


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _load_queries() -> list[dict[str, Any]]:
    with (ROOT / "data" / "eval_queries_v2.json").open("r", encoding="utf-8") as f:
        return json.load(f)["queries"]


def _query_ids_from_file(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data["query_ids"] if isinstance(data, dict) else data
    return [str(qid) for qid in ids]


def _stratified_query_ids(n: int, seed: int) -> list[str]:
    queries = _load_queries()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        buckets[(str(q.get("source", "")), str(q.get("difficulty", "")))].append(q)

    rng = random.Random(seed)
    for vals in buckets.values():
        rng.shuffle(vals)

    selected: list[str] = []
    ordered_buckets = sorted(buckets)
    while len(selected) < n and ordered_buckets:
        progressed = False
        for bucket in ordered_buckets:
            vals = buckets[bucket]
            if vals:
                selected.append(vals.pop()["id"])
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    return selected


def _write_readme(study_dir: Path, tasks: list[dict[str, Any]], judges: list[str]) -> None:
    if judges:
        responses_layout = "\n".join(
            f"- `responses_{j}/`: Claude Code responses written by the {j} subagent (one file per task)."
            for j in judges
        )
        manual_section = (
            "Each task is judged by every entry in the `--judges` list above. Subagents read the task JSON and write\n"
            "the canonical response JSON into the matching `responses_<judge>/` directory. The parser is run once per\n"
            "judge with `--responses-dir <study>/responses_<judge> --judge-model claude-<judge>-... --promote-dir`."
        )
    else:
        responses_layout = "- `responses/`: save Claude Code responses here with the same filename as the task."
        manual_section = (
            "For each task file, paste the `system_prompt` and `user_prompt` into Claude Code and request JSON only.\n"
            "Save the raw response as `responses/<task filename>`. The parser accepts either a top-level JSON object\n"
            "with `evaluations` or a response that contains one JSON object inside surrounding text."
        )
    readme = [
        f"# Claude Code Judging Packet: {study_dir.name}",
        "",
        "This directory is for manual Claude Code judging. It is deliberately not an API workflow.",
        "",
        "## Layout",
        "",
        "- `tasks/*.json`: one complete judging prompt per report.",
        responses_layout,
        "- `parsed/`: created by `scripts/claude_code_judging_parse.py`.",
        "",
        "## Manual Execution",
        "",
        manual_section,
        "",
        "Required response schema:",
        "",
        "```json",
        json.dumps(JSON_SCHEMA_HINT, indent=2),
        "```",
        "",
        "## Parsing",
        "",
        "```bash",
        f"python scripts/claude_code_judging_parse.py --study-dir {study_dir}",
        "```",
        "",
        "Parsed judgments stay under this packet by default. Promote only after spot-checking:",
        "",
        "```bash",
        f"python scripts/claude_code_judging_parse.py --study-dir {study_dir} --promote",
        "```",
        "",
        "## Packet Summary",
        "",
        f"- Tasks: {len(tasks)}",
    ]
    for pattern, count in sorted(Counter(t["pattern"] for t in tasks).items()):
        readme.append(f"- {pattern}: {count}")
    (study_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="p11_p12_crossfamily",
                    help="Study label embedded in the output directory name.")
    ap.add_argument("--patterns", default="base_p11,base_p12",
                    help="Comma-separated experiment IDs under results/experiments.")
    ap.add_argument("--query-ids-file", type=Path, default=DEFAULT_QUERY_IDS,
                    help="JSON list or {'query_ids': [...]} file. Defaults to the 30-query variance slice.")
    ap.add_argument("--stratified-n", type=int, default=0,
                    help="If >0, ignore --query-ids-file and select this many stratified queries.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-label", default="claude-code-manual")
    ap.add_argument("--judges", default="",
                    help="Comma-separated judge IDs (e.g. 'sonnet,opus'). When set, creates parallel "
                         "responses_<judge>/ subdirs for cross-family panel coverage.")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--min-report-bytes", type=int, default=5000,
                    help="Skip tiny/failed reports below this size.")
    ap.add_argument("--date-stamp", default="",
                    help="Override YYYYMMDD prefix for deterministic reruns.")
    args = ap.parse_args()

    stamp = args.date_stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    study_dir = args.out_root / f"{stamp}_{_safe_stem(args.study)}"
    tasks_dir = study_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if judges:
        for judge in judges:
            (study_dir / f"responses_{judge}").mkdir(parents=True, exist_ok=True)
    else:
        (study_dir / "responses").mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    if args.stratified_n > 0:
        query_ids = _stratified_query_ids(args.stratified_n, args.seed)
    else:
        query_ids = _query_ids_from_file(args.query_ids_file)

    written: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for pattern in patterns:
        for qid in query_ids:
            report_path = REPORTS_BASE / pattern / f"{qid}.md"
            if not report_path.exists():
                skipped.append({"pattern": pattern, "query_id": qid, "reason": "missing_report"})
                continue
            if report_path.stat().st_size < args.min_report_bytes:
                skipped.append({"pattern": pattern, "query_id": qid, "reason": "tiny_report"})
                continue
            try:
                info = build_judge_prompt(pattern, qid)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"pattern": pattern, "query_id": qid, "reason": str(exc)[:200]})
                continue

            stem = f"{_safe_stem(pattern)}__{_safe_stem(qid)}__{_safe_stem(args.judge_label)}"
            if judges:
                response_files = {j: f"responses_{j}/{stem}.json" for j in judges}
            else:
                response_files = {"default": f"responses/{stem}.json"}
            task = {
                "study": args.study,
                "judge_label": args.judge_label,
                "judges": judges,
                "pattern": pattern,
                "query_id": qid,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "system_prompt": info["system_prompt"],
                "user_prompt": (
                    info["user_prompt"]
                    + "\n\n## Required JSON Output Schema\n"
                    + json.dumps(JSON_SCHEMA_HINT, indent=2)
                    + "\n\nReturn valid JSON only."
                ),
                "criteria_count": info["criteria_count"],
                "criteria_texts": info["criteria_texts"],
                "criteria_dimensions": info["criteria_dimensions"],
                "dimension_weights": info["dimension_weights"],
                "report_path": info["report_path"],
                "response_filename": f"{stem}.json",
                "response_files": response_files,
            }
            task_path = tasks_dir / f"{stem}.json"
            task_path.write_text(json.dumps(task, indent=2), encoding="utf-8")
            written.append({
                "task_file": str(task_path.relative_to(study_dir)),
                "response_files": response_files,
                "pattern": pattern,
                "query_id": qid,
                "criteria_count": info["criteria_count"],
            })

    (study_dir / "tasks_summary.json").write_text(
        json.dumps({"judges": judges, "tasks": written, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    _write_readme(study_dir, written, judges)

    print(f"Wrote {len(written)} task files to {tasks_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} reports; see {study_dir / 'tasks_summary.json'}")
    for pattern, count in sorted(Counter(t["pattern"] for t in written).items()):
        print(f"  {pattern}: {count}")
    if judges:
        for judge in judges:
            print(f"Responses directory ({judge}): {study_dir / f'responses_{judge}'}")
    else:
        print(f"Responses directory: {study_dir / 'responses'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
