#!/usr/bin/env python3
"""Helper: build judge prompt for a single report so Claude Code agents can evaluate it.

Usage (from Python):
    python scripts/claude_judge_eval.py <pattern_name> <query_id>

Prints the full judge prompt (system + user) and saves the rubric mapping
so results can be parsed back.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_research.evaluation.rubric_v2 import (
    build_general_criteria,
    build_rubric_v2,
    rubric_to_judge_prompt,
    DIMENSION_WEIGHTS_V2,
    Criterion,
)

RESULTS_BASE = Path("results/experiments")
JUDGE_OUT_OPUS = Path("results/judge_claude_opus")
JUDGE_OUT_SONNET = Path("results/judge_claude_sonnet")

# Default output dir — override via judge_model parameter in parse_and_save_result()
JUDGE_OUT = JUDGE_OUT_OPUS

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


def load_query(query_id: str) -> dict:
    with open("data/eval_queries_v2.json") as f:
        data = json.load(f)
    for q in data["queries"]:
        if q["id"] == query_id:
            return q
    raise ValueError(f"Query {query_id} not found")


def build_judge_prompt(pattern_name: str, query_id: str) -> dict:
    """Build the complete judge prompt for one report.

    Returns dict with: system_prompt, user_prompt, query_id, pattern_name,
    criteria_count, dimension_weights, report_path
    """
    query = load_query(query_id)
    report_path = RESULTS_BASE / pattern_name / f"{query_id}.md"
    if not report_path.exists():
        raise FileNotFoundError(f"No report at {report_path}")

    report_text = report_path.read_text()

    # Build rubric with query-specific coverage criteria if available
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
        query_id=query_id,
        query_text=query["query"],
        coverage_criteria=coverage_criteria,
        source_type=source_type,
    )

    criteria_prompt = rubric_to_judge_prompt(rubric)

    user_prompt = f"""## Research Query
{query['query']}

## Report to Evaluate
{report_text}

## Criteria to Evaluate
{criteria_prompt}"""

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "query_id": query_id,
        "pattern_name": pattern_name,
        "criteria_count": rubric.total_criteria,
        "dimensions": rubric.get_dimensions(),
        "dimension_weights": rubric.dimension_weights,
        "criteria_texts": [c.text for c in rubric.criteria],
        "criteria_dimensions": [c.dimension for c in rubric.criteria],
        "report_path": str(report_path),
    }


def parse_and_save_result(
    pattern_name: str,
    query_id: str,
    evaluations: list[dict],
    criteria_texts: list[str],
    criteria_dimensions: list[str],
    dimension_weights: dict[str, float],
    judge_model: str = "claude-opus-4.6",
) -> dict:
    """Parse evaluation JSON and save structured results."""
    # Score per dimension
    dim_stats: dict[str, dict] = {}
    for dim in set(criteria_dimensions):
        dim_stats[dim] = {"met": 0, "total": 0}

    verdicts = []
    for ev in evaluations:
        idx = ev.get("criterion_index", 0)
        if idx >= len(criteria_texts):
            continue
        satisfied = ev.get("verdict", "").upper() == "SATISFIED"
        dim = criteria_dimensions[idx]
        dim_stats[dim]["total"] += 1
        if satisfied:
            dim_stats[dim]["met"] += 1
        verdicts.append({
            "criterion": criteria_texts[idx],
            "dimension": dim,
            "satisfied": satisfied,
            "evidence": ev.get("evidence", ""),
            "reasoning": ev.get("reasoning", ""),
        })

    # Compute dimension scores
    dimensions = {}
    for dim, stats in dim_stats.items():
        score = stats["met"] / stats["total"] if stats["total"] > 0 else 0.0
        dimensions[dim] = {
            "score": round(score, 4),
            "met": stats["met"],
            "total": stats["total"],
        }

    # Weighted overall score
    overall = 0.0
    for dim, score_info in dimensions.items():
        w = dimension_weights.get(dim, 0.0)
        overall += w * score_info["score"]

    result = {
        "query_id": query_id,
        "pattern": pattern_name,
        "judge_model": judge_model,
        "overall_score": round(overall, 4),
        "dimensions": dimensions,
        "verdicts": verdicts,
        "n_criteria": len(verdicts),
        "n_satisfied": sum(1 for v in verdicts if v["satisfied"]),
    }

    # Save — route to correct output dir based on judge model
    if "sonnet" in judge_model.lower():
        out_base = JUDGE_OUT_SONNET
    else:
        out_base = JUDGE_OUT_OPUS
    out_dir = out_base / pattern_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{query_id}.json"
    out_path.write_text(json.dumps(result, indent=2))

    return result


def list_unevaluated(judge_model: str = "opus") -> list[tuple[str, str]]:
    """List (pattern_name, query_id) pairs that have reports but no evaluation yet.

    Args:
        judge_model: "opus" or "sonnet" — determines which output dir to check.

    Returns:
        List of (pattern_name, query_id) tuples needing evaluation.
    """
    judge_dir = JUDGE_OUT_SONNET if "sonnet" in judge_model.lower() else JUDGE_OUT_OPUS
    pending = []
    for report_path in sorted(RESULTS_BASE.rglob("*.md")):
        pattern_name = report_path.parent.name
        query_id = report_path.stem
        result_path = judge_dir / pattern_name / f"{query_id}.json"
        if not result_path.exists():
            pending.append((pattern_name, query_id))
    return pending


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--pending":
        # List all reports needing evaluation
        pending = list_unevaluated("opus")
        print(f"Pending Opus evaluations: {len(pending)}")
        for pattern, qid in pending[:20]:
            print(f"  {pattern}/{qid}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        sys.exit(0)

    if len(sys.argv) == 3 and sys.argv[1] == "--pending":
        # List pending for specific judge
        pending = list_unevaluated(sys.argv[2])
        print(f"Pending {sys.argv[2]} evaluations: {len(pending)}")
        for pattern, qid in pending[:20]:
            print(f"  {pattern}/{qid}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        sys.exit(0)

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pattern_name> <query_id>")
        print(f"       {sys.argv[0]} --pending [opus|sonnet]")
        sys.exit(1)
    info = build_judge_prompt(sys.argv[1], sys.argv[2])
    print(f"Criteria: {info['criteria_count']}")
    print(f"Dimensions: {info['dimensions']}")
    print(f"Report: {info['report_path']}")
    print(f"\nSystem prompt length: {len(info['system_prompt'])} chars")
    print(f"User prompt length: {len(info['user_prompt'])} chars")
