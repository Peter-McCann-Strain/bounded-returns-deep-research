#!/usr/bin/env python3
"""Parse manual Claude Code judging responses.

By default this script writes canonical judge JSON under
`reports/claude_code_judging/<study>/parsed/` and a parse summary. It does not
modify canonical analysis inputs unless `--promote` is passed, in which case it
writes to `results/judge_claude_code/<pattern>/<query_id>.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PROMOTE_DIR = ROOT / "results" / "judge_claude_code"


def _extract_json_object(text: str) -> Any:
    """Return the first parseable JSON value from text."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no parseable JSON object found")


def _load_response(responses_dir: Path, task_name: str) -> tuple[Path | None, str | None]:
    stem = Path(task_name).stem
    candidates = [
        responses_dir / task_name,
        responses_dir / f"{stem}.json",
        responses_dir / f"{stem}.txt",
        responses_dir / f"{stem}.md",
    ]
    for path in candidates:
        if path.exists():
            return path, path.read_text(encoding="utf-8", errors="replace")
    return None, None


def _normalise_evaluations(obj: Any) -> tuple[list[dict[str, Any]], bool, str]:
    if isinstance(obj, list):
        return obj, False, ""
    if not isinstance(obj, dict):
        return [], False, "response JSON is not an object or list"
    refusal = bool(obj.get("refusal") or obj.get("refused"))
    refusal_reason = str(obj.get("refusal_reason") or obj.get("reason") or "")
    evals = obj.get("evaluations") or obj.get("evaluations_list") or []
    if not isinstance(evals, list):
        return [], refusal, "evaluations is not a list"
    return evals, refusal, refusal_reason


def _score_result(task: dict[str, Any], evaluations: list[dict[str, Any]], judge_model: str) -> dict[str, Any]:
    criteria_texts = task["criteria_texts"]
    criteria_dimensions = task["criteria_dimensions"]
    dimension_weights = task["dimension_weights"]

    dim_stats = {dim: {"met": 0, "total": 0} for dim in set(criteria_dimensions)}
    verdicts: list[dict[str, Any]] = []
    for ev in evaluations:
        if not isinstance(ev, dict):
            continue
        try:
            idx = int(ev.get("criterion_index", 0))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(criteria_texts):
            continue
        verdict = str(ev.get("verdict", "")).strip().upper().replace(" ", "_")
        satisfied = verdict == "SATISFIED"
        dim = criteria_dimensions[idx]
        dim_stats[dim]["total"] += 1
        if satisfied:
            dim_stats[dim]["met"] += 1
        verdicts.append({
            "criterion": criteria_texts[idx],
            "dimension": dim,
            "satisfied": satisfied,
            "evidence": str(ev.get("evidence", "") or ""),
            "reasoning": str(ev.get("reasoning", "") or ""),
            "criterion_index": idx,
        })

    dimensions = {}
    for dim, stats in dim_stats.items():
        total = stats["total"]
        score = stats["met"] / total if total else 0.0
        dimensions[dim] = {
            "score": round(score, 4),
            "met": stats["met"],
            "total": total,
        }

    overall = sum(
        dimensions.get(dim, {}).get("score", 0.0) * float(weight)
        for dim, weight in dimension_weights.items()
    )
    return {
        "query_id": task["query_id"],
        "pattern": task["pattern"],
        "judge_model": judge_model,
        "judge_source": "claude_code_manual",
        "overall_score": round(overall, 4),
        "dimensions": dimensions,
        "verdicts": verdicts,
        "n_criteria": len(verdicts),
        "n_satisfied": sum(1 for v in verdicts if v["satisfied"]),
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "task_file": task.get("_task_file", ""),
        "raw_response_file": task.get("_raw_response_file", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--responses-dir", type=Path, default=None)
    ap.add_argument("--parsed-dir", type=Path, default=None)
    ap.add_argument("--judge-model", default="claude-code-manual")
    ap.add_argument("--promote", action="store_true",
                    help="Also copy parsed successes into results/judge_claude_code/.")
    ap.add_argument("--promote-dir", type=Path, default=PROMOTE_DIR)
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow --promote to overwrite existing judge_claude_code JSON.")
    args = ap.parse_args()

    study_dir = args.study_dir
    tasks_dir = study_dir / "tasks"
    responses_dir = args.responses_dir or (study_dir / "responses")
    parsed_dir = args.parsed_dir or (study_dir / "parsed")
    parsed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_path in sorted(tasks_dir.glob("*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["_task_file"] = str(task_path)
        response_path, response_text = _load_response(responses_dir, task_path.name)
        base_row = {
            "task_file": str(task_path),
            "response_file": str(response_path) if response_path else "",
            "pattern": task.get("pattern", ""),
            "query_id": task.get("query_id", ""),
            "status": "",
            "refusal": False,
            "reason": "",
            "n_criteria": 0,
            "n_satisfied": 0,
            "overall_score": "",
            "promoted_path": "",
        }
        if response_text is None or response_path is None:
            base_row.update({"status": "missing_response", "reason": "no response file"})
            rows.append(base_row)
            continue

        try:
            obj = _extract_json_object(response_text)
            evaluations, refusal, reason = _normalise_evaluations(obj)
            base_row["refusal"] = refusal
            if refusal:
                base_row.update({"status": "refusal", "reason": reason})
                rows.append(base_row)
                continue
            if not evaluations:
                base_row.update({"status": "parse_error", "reason": reason or "empty evaluations"})
                rows.append(base_row)
                continue
            task["_raw_response_file"] = str(response_path)
            result = _score_result(task, evaluations, args.judge_model)
            out_dir = parsed_dir / task["pattern"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task['query_id']}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")

            base_row.update({
                "status": "parsed",
                "n_criteria": result["n_criteria"],
                "n_satisfied": result["n_satisfied"],
                "overall_score": result["overall_score"],
            })

            if args.promote:
                promote_dir = args.promote_dir / task["pattern"]
                promote_dir.mkdir(parents=True, exist_ok=True)
                promote_path = promote_dir / f"{task['query_id']}.json"
                if promote_path.exists() and not args.overwrite:
                    base_row.update({
                        "status": "promote_skipped_exists",
                        "reason": "existing promoted result; pass --overwrite to replace",
                    })
                else:
                    promote_path.write_text(
                        json.dumps(result, indent=2, ensure_ascii=True),
                        encoding="utf-8",
                    )
                    base_row["promoted_path"] = str(promote_path)
        except Exception as exc:  # noqa: BLE001
            base_row.update({"status": "parse_error", "reason": f"{type(exc).__name__}: {exc}"})
        rows.append(base_row)

    summary_json = parsed_dir / "parse_summary.json"
    summary_csv = parsed_dir / "parse_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Parsed packet: {study_dir}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"Summary: {summary_csv}")
    if args.promote:
        print(f"Promote dir: {args.promote_dir}")
    return 0 if counts.get("parse_error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
