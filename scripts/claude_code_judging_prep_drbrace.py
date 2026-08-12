#!/usr/bin/env python3
"""Prepare Claude-Code judging packets for the DeepResearch-Bench RACE corpus.

This is the DRB-RACE sibling of ``scripts/claude_code_judging_prep.py``. It reuses
the *exact same* judging method (system prompt + user prompt = query + report +
rubric_v2 criteria + JSON schema), but sources from the staged DRB corpus rather
than ``results/experiments``:

  * 400 reports staged at
    ``results/drbrace/_judge_stage/drbrace_<system>/drb1_<task>.md``
  * 100 DRB task prompts in ``results/drbrace/eval_queries_drbrace.json``
  * rubric_v2 with ``source_type="drbench"`` (matches the GPT-5.2 panel exactly,
    so overall scores are directly comparable: citation_quality 0.15,
    instruction_following 0.05).

The packet's ``pattern`` is ``drbrace_<system>`` and ``query_id`` is
``drb1_<task>``. After judging, the generic parser
(``scripts/claude_code_judging_parse.py``) is run once per judge with
``--promote-dir results/drbrace/judge_claude_opus48`` (or ``_sonnet48``). Because
``pattern`` already carries the ``drbrace_<system>`` prefix, the promoted file
lands at ``results/drbrace/judge_claude_<fam>48/drbrace_<system>/drb1_<task>.json``
— exactly the layout ``build_judge_vs_human.py:load_drb_verdicts`` reads with
``--judge claude_opus`` / ``claude_sonnet``.

This builder makes ZERO API calls. It only writes packets + shard lists under the
study dir.

    python scripts/claude_code_judging_prep_drbrace.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deep_research.evaluation.rubric_v2 import (  # noqa: E402
    Criterion,
    build_rubric_v2,
    rubric_to_judge_prompt,
)

# Identical to the GPT-5.2 DRB panel system prompt (run_gpt52_judge_namespaced.py)
# so the Claude families judge against the exact same instructions.
SYSTEM_PROMPT = """You are an expert research report evaluator using the DRACO evaluation methodology.
You assess whether research reports satisfy specific evaluation criteria.

You will be given:
1. The original research query
2. A research report to evaluate
3. A list of evaluation criteria

For EACH criterion, you must provide:
- VERDICT: "SATISFIED" or "NOT_SATISFIED"
- EVIDENCE: A brief quote or reference to specific content in the report
- REASONING: One sentence explaining your judgment

Rules:
- Only mark SATISFIED if the criterion is clearly and fully met
- Partial or vague coverage counts as NOT_SATISFIED
- For citation criteria, check that actual sources/references are provided
- For factual criteria, verify claims are consistent and reasonable
- Be strict but fair -- do not penalize for minor omissions if the substance is there

Respond with valid JSON only."""

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

# GPT-5.2 panel truncated reports above this word count; mirror it for parity.
MAX_REPORT_WORDS = 12000

DEFAULT_MANIFEST = ROOT / "results" / "drbrace" / "eval_queries_drbrace.json"
DEFAULT_STAGE = ROOT / "results" / "drbrace" / "_judge_stage"
OUT_ROOT = ROOT / "reports" / "claude_code_judging"

# Quarantined query (judge-specific AUP false-positive). DRB ids are drb1_<n>, so
# this should never collide, but we honour it defensively.
QUARANTINE_IDS = {"82de3e92"}


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data["queries"] if isinstance(data, dict) else data
    return {str(q["id"]): q for q in queries}


def _build_prompt_for(query: dict[str, Any], report_text: str) -> dict[str, Any]:
    """Replicate run_gpt52_judge_namespaced.build_rubric_for_query + user_msg."""
    coverage_criteria = None
    if query.get("expected_elements"):
        coverage_criteria = [
            Criterion(
                text=f"The report covers: {elem}",
                dimension="coverage",
                source="task_specific",
            )
            for elem in query["expected_elements"]
        ]

    source_type = query.get("source", "default")
    rubric = build_rubric_v2(
        query_id=query["id"],
        query_text=query["query"],
        coverage_criteria=coverage_criteria,
        source_type=source_type,
    )
    criteria_prompt = rubric_to_judge_prompt(rubric)

    words = report_text.split()
    truncated = len(words) > MAX_REPORT_WORDS
    if truncated:
        report_text = " ".join(words[:MAX_REPORT_WORDS]) + (
            "\n\n[... report truncated for evaluation ...]"
        )

    user_prompt = (
        f"## Research Query\n{query['query']}\n\n"
        f"## Report to Evaluate\n{report_text}\n\n"
        f"## Criteria to Evaluate\n{criteria_prompt}"
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt
        + "\n\n## Required JSON Output Schema\n"
        + json.dumps(JSON_SCHEMA_HINT, indent=2)
        + "\n\nReturn valid JSON only.",
        "criteria_count": rubric.total_criteria,
        "criteria_texts": [c.text for c in rubric.criteria],
        "criteria_dimensions": [c.dimension for c in rubric.criteria],
        "dimension_weights": rubric.dimension_weights,
        "source_type": source_type,
        "truncated": truncated,
    }


def _write_shards(study_dir: str, stems: list[str], judge: str, shard_size: int) -> list[str]:
    shards_dir = Path(study_dir) / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for i in range(0, len(stems), shard_size):
        chunk = stems[i : i + shard_size]
        idx = i // shard_size
        path = shards_dir / f"{judge}_{idx:03d}.txt"
        path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        written.append(str(path))
    return written


def _write_readme(study_dir: Path, judges: list[str], tasks: list[dict[str, Any]],
                  shard_size: int, judge_label: str) -> None:
    promote_map = {
        "opus": "results/drbrace/judge_claude_opus48",
        "sonnet": "results/drbrace/judge_claude_sonnet48",
    }
    lines = [
        f"# Claude-Code DRB-RACE Judging Packet: {study_dir.name}",
        "",
        "Cross-family judge-validity for Paper 2: Opus 4.8 + Sonnet 4.6 verdicts on the",
        "400 released DeepResearch-Bench RACE reports, to compare against the DRB human",
        "experts (and the existing GPT-5.2 panel in results/drbrace/judge_gpt52/).",
        "",
        "## Layout",
        "- `tasks/*.json`: one complete judging prompt per (system, task) report (400).",
    ]
    for j in judges:
        lines.append(f"- `responses_{j}/`: Claude-Code responses for the {j} judge (one file per task, same stem).")
    lines += [
        "- `shards/{opus,sonnet}_NNN.txt`: stem lists (~%d per shard) for sequential batch judging." % shard_size,
        "- `parsed_<judge>/`: created by the parser (pre-promotion spot-check).",
        "",
        "## Method parity",
        "Identical system prompt, rubric_v2 criteria, and `source_type='drbench'` weights",
        "(citation_quality 0.15, instruction_following 0.05) as the GPT-5.2 DRB panel, so",
        "overall scores are directly comparable. Reports >%d words are truncated identically." % MAX_REPORT_WORDS,
        "",
        "## Judging (sequential-batch subagent, one shard at a time, under the Opus ceiling)",
        "For each shard file, a judging subagent reads each listed task `tasks/<stem>.json`,",
        "applies the `system_prompt` + `user_prompt`, and writes JSON-only to",
        "`responses_<judge>/<stem>.json` (Read+Write only; no Bash, no API).",
        "",
        "## Parse + promote (NO API; run once per judge after responses land)",
        "```bash",
    ]
    for j in judges:
        promo = promote_map.get(j, f"results/drbrace/judge_claude_{j}48")
        model_id = "claude-opus-4.8" if j == "opus" else "claude-sonnet-4.6"
        lines += [
            f"# {j}: spot-check first (writes parsed_{j}/ only)",
            f"python scripts/claude_code_judging_parse.py \\",
            f"  --study-dir {study_dir} \\",
            f"  --responses-dir {study_dir}/responses_{j} \\",
            f"  --parsed-dir {study_dir}/parsed_{j} \\",
            f"  --judge-model {model_id}",
            f"# {j}: then promote into the DRB namespace",
            f"python scripts/claude_code_judging_parse.py \\",
            f"  --study-dir {study_dir} \\",
            f"  --responses-dir {study_dir}/responses_{j} \\",
            f"  --parsed-dir {study_dir}/parsed_{j} \\",
            f"  --judge-model {model_id} \\",
            f"  --promote --promote-dir {promo}",
            "",
        ]
    lines += [
        "```",
        "",
        "## Read into the human-correlation analysis",
        "```bash",
        "python papers/paper_a_bounded_returns/analysis/build_judge_vs_human.py \\",
        "  --run-drb-race --judge claude_opus \\",
        "  --verdicts-dir results/drbrace/judge_claude_opus48",
        "python papers/paper_a_bounded_returns/analysis/build_judge_vs_human.py \\",
        "  --run-drb-race --judge claude_sonnet \\",
        "  --verdicts-dir results/drbrace/judge_claude_sonnet48",
        "```",
        "",
        "## Packet summary",
        f"- judge_label: {judge_label}",
        f"- Tasks: {len(tasks)}",
    ]
    for pattern, count in sorted(Counter(t["pattern"] for t in tasks).items()):
        lines.append(f"- {pattern}: {count}")
    (study_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="drbrace")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--judges", default="opus,sonnet")
    ap.add_argument("--judge-label", default="claude-code-manual")
    ap.add_argument("--shard-size", type=int, default=6)
    ap.add_argument("--date-stamp", default="20260616",
                    help="YYYYMMDD prefix for the study dir (deterministic reruns).")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    study_dir = args.out_root / f"{args.date_stamp}_{_safe_stem(args.study)}"
    tasks_dir = study_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for j in judges:
        (study_dir / f"responses_{j}").mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(args.manifest)

    written: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    stems_per_system: dict[str, list[str]] = {}

    system_dirs = sorted(args.stage_dir.glob("drbrace_*"))
    for sysdir in system_dirs:
        if not sysdir.is_dir():
            continue
        pattern = sysdir.name  # drbrace_<system>
        for report_path in sorted(sysdir.glob("drb1_*.md")):
            qid = report_path.stem  # drb1_<task>
            if qid in QUARANTINE_IDS:
                skipped.append({"pattern": pattern, "query_id": qid, "reason": "quarantined"})
                continue
            query = manifest.get(qid)
            if query is None:
                skipped.append({"pattern": pattern, "query_id": qid, "reason": "missing_manifest_entry"})
                continue
            report_text = report_path.read_text(encoding="utf-8", errors="replace")
            if not report_text.strip():
                skipped.append({"pattern": pattern, "query_id": qid, "reason": "empty_report"})
                continue

            info = _build_prompt_for(query, report_text)
            stem = f"{_safe_stem(pattern)}__{_safe_stem(qid)}__{_safe_stem(args.judge_label)}"
            response_files = {j: f"responses_{j}/{stem}.json" for j in judges}
            task = {
                "study": args.study,
                "judge_label": args.judge_label,
                "judges": judges,
                "pattern": pattern,
                "query_id": qid,
                "source_type": info["source_type"],
                "truncated": info["truncated"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "system_prompt": info["system_prompt"],
                "user_prompt": info["user_prompt"],
                "criteria_count": info["criteria_count"],
                "criteria_texts": info["criteria_texts"],
                "criteria_dimensions": info["criteria_dimensions"],
                "dimension_weights": info["dimension_weights"],
                "report_path": str(report_path.relative_to(ROOT)),
                "response_filename": f"{stem}.json",
                "response_files": response_files,
            }
            (tasks_dir / f"{stem}.json").write_text(
                json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            written.append({
                "task_file": f"tasks/{stem}.json",
                "stem": stem,
                "pattern": pattern,
                "query_id": qid,
                "criteria_count": info["criteria_count"],
                "truncated": info["truncated"],
            })
            stems_per_system.setdefault(pattern, []).append(stem)

    # Shards: round-robin systems so each shard mixes systems (balanced drip),
    # deterministically ordered.
    ordered_stems = [w["stem"] for w in written]
    shard_files: dict[str, list[str]] = {}
    for j in judges:
        shard_files[j] = _write_shards(str(study_dir), ordered_stems, j, args.shard_size)

    (study_dir / "tasks_summary.json").write_text(
        json.dumps({
            "study": args.study,
            "judges": judges,
            "judge_label": args.judge_label,
            "source_type_used": "drbench",
            "n_tasks": len(written),
            "shard_size": args.shard_size,
            "shards": {j: [Path(p).name for p in shard_files[j]] for j in judges},
            "promote_dirs": {
                "opus": "results/drbrace/judge_claude_opus48",
                "sonnet": "results/drbrace/judge_claude_sonnet48",
            },
            "tasks": written,
            "skipped": skipped,
        }, indent=2),
        encoding="utf-8",
    )
    _write_readme(study_dir, judges, written, args.shard_size, args.judge_label)

    print(f"Study dir: {study_dir}")
    print(f"Wrote {len(written)} task packets to {tasks_dir}")
    for pattern, count in sorted(Counter(t["pattern"] for t in written).items()):
        print(f"  {pattern}: {count}")
    if skipped:
        print(f"Skipped {len(skipped)} (see tasks_summary.json)")
    for j in judges:
        print(f"  {j} shards: {len(shard_files[j])} files ({args.shard_size}/shard) -> {study_dir / 'shards'}")
    print("ZERO API calls made (packet builder only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
