#!/usr/bin/env python3
"""E4 CITE-CAUSAL — Step 5: Claude full-n re-judge of the citation-perturbed reports.

Clones the Claude-Code agent-dispatch pattern from ``scripts/run_e4_opus_calibration.py``
+ ``papers/paper_a_bounded_returns/judge_oracle_claude.workflow.js`` and the file-based packet
format of ``scripts/claude_code_judging_prep.py``.  The Anthropic models are judged via
the Claude Code subscription harness (NOT the API), so this script does NOT call any LLM:
it only (a) builds the canonical 38-criterion V2 prompt packets from the E4 transformed
reports, and (b) parses the agent responses into the existing 3-judge JSON schema.

Two Claude judges, full-n: Claude Opus 4.1 and Claude Sonnet 4.5.
  100 reports x 5 conditions x 2 judges = 1000 agent dispatches (session-window budget, $0 cash).

Agent constraint (per feedback_judge_agents.md): judge subagents CANNOT use Bash.  This
prep step pre-generates every prompt to a task file; agents use Read+Write ONLY (Read the
task JSON, Write the response JSON).  No Bash, no tool calls beyond Read/Write.

Quarantine (per quarantine_82de3e92.md): query 82de3e92 is FLAGGED (not dropped) in the
task index so the Claude panel can be reconciled against the 82de3e92 quarantine downstream.

READS  (READ-ONLY): results/experiments_e4_cite/{condition}/{pattern}/{query_id}.md
WRITES:             reports/phase_e4_claude/{opus,sonnet}/tasks/*.json  (prompt packets)
                    reports/phase_e4_claude/{opus,sonnet}/responses/*.json  (agent output)
                    results/judge_claude_opus_e4/{condition}/{pattern}/{query_id}.json   (parsed)
                    results/judge_claude_sonnet_e4/{condition}/{pattern}/{query_id}.json (parsed)

Usage:
    python scripts/run_e4_claude_cite.py prep                  # build all task packets
    python scripts/run_e4_claude_cite.py prep --judge opus --limit 6   # tiny smoke
    python scripts/run_e4_claude_cite.py parse-all --judge opus        # after agents run
    python scripts/run_e4_claude_cite.py dispatch-plan                 # print agent run plan
    python scripts/run_e4_claude_cite.py --dry-run            # ZERO writes outside scratch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))  # MUST: avoids ModuleNotFoundError when run as a script

from deep_research.evaluation.rubric_v2 import (
    build_rubric_v2, rubric_to_judge_prompt, Criterion,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
E4_REPORTS = _REPO_ROOT / "results" / "experiments_e4_cite"   # READ-ONLY transformed reports
SAMPLE_MANIFEST = E4_REPORTS / "sample_manifest.json"
EVAL_QUERIES = _REPO_ROOT / "data" / "eval_queries_v2.json"
CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
QUARANTINE = {"82de3e92"}

PHASE_DIR = _REPO_ROOT / "reports" / "phase_e4_claude"
JUDGE_OUT = {
    "opus":   _REPO_ROOT / "results" / "judge_claude_opus_e4",
    "sonnet": _REPO_ROOT / "results" / "judge_claude_sonnet_e4",
}
JUDGE_MODEL_ID = {"opus": "claude-opus-4.1", "sonnet": "claude-sonnet-4.5"}

PROTECTED_PATHS = [
    _REPO_ROOT / "results" / "judge_gpt52",
    _REPO_ROOT / "results" / "experiments",
    _REPO_ROOT / "data" / "analysis",
    _REPO_ROOT / "reports" / "eval_v2" / "verdicts",
    _REPO_ROOT / "results" / "judge_claude_opus",     # baseline Claude panels: READ-ONLY
    _REPO_ROOT / "results" / "judge_claude_sonnet",
]

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
"""

JSON_SCHEMA_HINT = {
    "evaluations": [
        {"criterion_index": 0, "verdict": "SATISFIED",
         "evidence": "Short quote or location from the report.",
         "reasoning": "One sentence explaining the judgment."}
    ]
}


def _is_rel(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent); return True
    except ValueError:
        return False


def _assert_safe(out: Path) -> None:
    out = out.resolve()
    for prot in PROTECTED_PATHS:
        p = prot.resolve()
        if out == p or _is_rel(out, p) or _is_rel(p, out):
            raise SystemExit(f"REFUSING: write target {out} collides with protected path {p}")


def load_queries() -> dict[str, dict]:
    return {q["id"]: q for q in json.loads(EVAL_QUERIES.read_text())["queries"]}


def build_packet(query: dict, report_text: str) -> dict:
    """Build the canonical 38-criterion V2 prompt packet for one transformed report."""
    coverage_criteria = None
    if query.get("expected_elements"):
        coverage_criteria = [
            Criterion(text=f"The report covers: {elem}", dimension="coverage",
                      source="task_specific")
            for elem in query["expected_elements"]
        ]
    rubric = build_rubric_v2(
        query_id=query["id"], query_text=query["query"],
        coverage_criteria=coverage_criteria, source_type=query.get("source", "default"))
    criteria_prompt = rubric_to_judge_prompt(rubric)
    user_prompt = (f"## Research Query\n{query['query']}\n\n"
                   f"## Report to Evaluate\n{report_text}\n\n"
                   f"## Criteria to Evaluate\n{criteria_prompt}\n\n"
                   f"## Required JSON Output Schema\n{json.dumps(JSON_SCHEMA_HINT, indent=2)}\n\n"
                   f"Return valid JSON only.")
    return {
        "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt,
        "criteria_count": rubric.total_criteria,
        "criteria_texts": [c.text for c in rubric.criteria],
        "criteria_dimensions": [c.dimension for c in rubric.criteria],
        "dimension_weights": rubric.dimension_weights,
    }


def iter_sample():
    """Yield (condition, pattern, query_id, report_path) for every transformed report."""
    if SAMPLE_MANIFEST.exists():
        man = json.loads(SAMPLE_MANIFEST.read_text())
        reports = [(r["pattern"], r["query_id"]) for r in man["reports"]]
    else:
        reports = []
        c0 = E4_REPORTS / "C0"
        if c0.exists():
            for pat in sorted(p for p in c0.iterdir() if p.is_dir()):
                for f in sorted(pat.glob("*.md")):
                    reports.append((pat.name, f.stem))
    for cond in CONDITIONS:
        for pattern, qid in reports:
            rp = E4_REPORTS / cond / pattern / f"{qid}.md"
            if rp.exists():
                yield cond, pattern, qid, rp


def stem_for(cond, pattern, qid):
    return f"{cond}__{pattern}__{qid}"


def prep(judges, limit, out_phase=None, out_judge_root=None):
    queries = load_queries()
    phase = out_phase or PHASE_DIR
    written = {j: 0 for j in judges}
    index = []
    n_seen = 0
    for cond, pattern, qid, rp in iter_sample():
        if limit and n_seen >= limit:
            break
        n_seen += 1
        query = queries.get(qid)
        if query is None:
            continue
        packet = build_packet(query, rp.read_text(errors="ignore"))
        stem = stem_for(cond, pattern, qid)
        for j in judges:
            tasks_dir = phase / j / "tasks"
            resp_dir = phase / j / "responses"
            _assert_safe(tasks_dir); _assert_safe(resp_dir)
            tasks_dir.mkdir(parents=True, exist_ok=True)
            resp_dir.mkdir(parents=True, exist_ok=True)
            task = {
                "judge": j, "judge_model": JUDGE_MODEL_ID[j],
                "condition": cond, "pattern": pattern, "query_id": qid,
                "quarantine": qid in QUARANTINE,
                "report_path": str(rp.relative_to(_REPO_ROOT)),
                "response_path": str((resp_dir / f"{stem}.json").relative_to(_REPO_ROOT)),
                **packet,
            }
            (tasks_dir / f"{stem}.json").write_text(json.dumps(task, indent=2))
            written[j] += 1
        index.append({"condition": cond, "pattern": pattern, "query_id": qid,
                      "stem": stem, "quarantine": qid in QUARANTINE})
    (phase).mkdir(parents=True, exist_ok=True)
    (phase / "task_index.json").write_text(json.dumps({
        "experiment": "E4 CITE-CAUSAL (Claude full-n)",
        "judges": judges, "conditions": CONDITIONS,
        "n_tasks_per_judge": len(index),
        "n_dispatches": len(index) * len(judges),
        "agent_constraint": "Read+Write only — NO Bash (feedback_judge_agents.md)",
        "quarantine_flagged": sorted({i["query_id"] for i in index if i["quarantine"]}),
        "tasks": index,
    }, indent=2))
    for j in judges:
        print(f"  {j}: wrote {written[j]} task packets -> {phase / j / 'tasks'}")
    print(f"  task_index -> {phase / 'task_index.json'}")
    print(f"  total dispatches (this prep): {sum(written.values())}")
    print("\n  Agents read tasks/<stem>.json (Read tool), write responses/<stem>.json (Write tool).")
    print("  NO Bash. Required response shape: {\"evaluations\": [{criterion_index, verdict, "
          "evidence, reasoning}, ...]}")


def parse_one(task: dict, raw: dict, judge_out_root: Path) -> dict | None:
    crit_texts = task["criteria_texts"]
    crit_dims = task["criteria_dimensions"]
    dim_weights = task["dimension_weights"]
    evaluations = raw.get("evaluations", [])
    if not evaluations:
        return None
    dim_stats = {dim: {"met": 0, "total": 0} for dim in set(crit_dims)}
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
        verdicts.append({"criterion_index": idx, "criterion": crit_texts[idx],
                         "dimension": dim, "satisfied": satisfied,
                         "evidence": ev.get("evidence", ""), "reasoning": ev.get("reasoning", "")})
    dimensions = {dim: {"score": round(s["met"] / s["total"], 4) if s["total"] else 0.0,
                        "met": s["met"], "total": s["total"]}
                  for dim, s in dim_stats.items()}
    overall = sum(dimensions.get(d, {}).get("score", 0) * w for d, w in dim_weights.items())
    result = {"query_id": task["query_id"], "pattern": task["pattern"],
              "condition": task["condition"], "judge_model": task["judge_model"],
              "overall_score": round(overall, 4), "dimensions": dimensions,
              "verdicts": verdicts, "n_criteria": len(verdicts),
              "n_satisfied": sum(1 for v in verdicts if v["satisfied"])}
    dst = judge_out_root / task["condition"] / task["pattern"] / f"{task['query_id']}.json"
    _assert_safe(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2))
    return result


def parse_all(judges):
    for j in judges:
        tasks_dir = PHASE_DIR / j / "tasks"
        resp_dir = PHASE_DIR / j / "responses"
        out_root = JUDGE_OUT[j]
        if not tasks_dir.exists():
            print(f"  {j}: no tasks dir ({tasks_dir}); run prep first."); continue
        n_ok = n_missing = 0
        for tf in sorted(tasks_dir.glob("*.json")):
            task = json.loads(tf.read_text())
            rf = resp_dir / tf.name
            if not rf.exists():
                n_missing += 1; continue
            try:
                raw = json.loads(rf.read_text())
            except Exception:
                n_missing += 1; continue
            if parse_one(task, raw, out_root):
                n_ok += 1
        print(f"  {j}: parsed {n_ok}, missing/empty {n_missing} -> {out_root}")


def dispatch_plan(judges):
    n_per = sum(1 for _ in iter_sample()) if E4_REPORTS.exists() else 0
    print("E4 Claude full-n dispatch plan")
    print(f"  reports x conditions per judge: {n_per}")
    print(f"  judges: {judges}")
    print(f"  total agent dispatches: {n_per * len(judges)}  (subscription session-window, $0 cash)")
    print("  agent contract: Read task JSON -> Write response JSON. NO Bash, NO other tools.")
    print(f"  parsed verdicts land in: {JUDGE_OUT['opus']} and {JUDGE_OUT['sonnet']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["prep", "parse-all", "dispatch-plan"],
                    default="dispatch-plan")
    ap.add_argument("--judge", choices=["opus", "sonnet", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="Cap reports (smoke sizing).")
    ap.add_argument("--dry-run", action="store_true",
                    help="ZERO writes outside a scratch dir; tiny prep into scratch.")
    args = ap.parse_args()
    judges = ["opus", "sonnet"] if args.judge == "both" else [args.judge]

    if args.dry_run:
        scratch = _REPO_ROOT / "reports" / "_scratch_e4_claude_dryrun"
        import shutil
        if scratch.exists():
            shutil.rmtree(scratch)
        print(f"[DRY RUN] prep tiny packets into scratch {scratch} (no protected writes)")
        prep(judges, limit=max(args.limit, 4), out_phase=scratch)
        # Verify packet shape round-trips through parse_one without touching judge dirs.
        sample_task = next(iter(scratch.glob(f"{judges[0]}/tasks/*.json")), None)
        if sample_task:
            t = json.loads(sample_task.read_text())
            fake = {"evaluations": [{"criterion_index": 0, "verdict": "SATISFIED",
                                     "evidence": "x", "reasoning": "y"}]}
            scratch_out = scratch / "_parsed_check"
            r = parse_one(t, fake, scratch_out)
            ok = bool(r and "overall_score" in r and t["criteria_count"] >= 1)
            print(f"[DRY RUN] packet criteria_count={t['criteria_count']}; "
                  f"parse round-trip: {'OK' if ok else 'FAIL'} "
                  f"(overall={r['overall_score'] if r else 'n/a'})")
        print("[DRY RUN] ZERO API calls. Nothing written outside the scratch dir.")
        return 0

    if args.mode == "prep":
        prep(judges, limit=args.limit)
    elif args.mode == "parse-all":
        parse_all(judges)
    else:
        dispatch_plan(judges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
