"""Public reproduction helpers."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from deep_research.judge_runner import run_judge_file
from deep_research.settings import PublicSettings, ensure_runtime_dirs


REFERENCE_RESULTS_PATH = Path("repro/reference/paper_a_reference.json")
REFERENCE_HEADLINE_PATH = Path("repro/reference/paper_a_headline_numbers.json")
REFERENCE_PATTERN_METRICS_CSV_PATH = Path("repro/reference/paper_a_pattern_metrics.csv")
PUBLIC_QUERIES_PATH = Path("data/eval_queries_v2.json")
PUBLIC_CRITERIA_PATH = Path("data/public_judge_criteria.json")
SAFE_FILE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ReproductionReport:
    mode: str
    status: str
    message: str
    created_utc: str
    reference_path: str
    output_path: str | None = None
    details: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_reference_results(project_root: Path) -> dict[str, Any]:
    path = project_root / REFERENCE_RESULTS_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_public_queries(project_root: Path) -> list[dict[str, Any]]:
    payload = json.loads((project_root / PUBLIC_QUERIES_PATH).read_text())
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError("data/eval_queries_v2.json must contain a `queries` list")
    return [query for query in queries if isinstance(query, dict) and query.get("query")]


def _select_queries(
    queries: list[dict[str, Any]],
    *,
    full: bool,
    limit: int,
) -> list[dict[str, Any]]:
    if full:
        return queries
    return queries[: max(1, limit)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _safe_file_stem(value: str, *, max_length: int = 96) -> str:
    stem = SAFE_FILE_STEM_RE.sub("_", value).strip("._-")
    return stem[:max_length].strip("._-")


def _query_file_stem(query_record: dict[str, Any]) -> str:
    explicit_id = str(query_record.get("id") or "").strip()
    if explicit_id:
        safe_id = _safe_file_stem(explicit_id)
        if safe_id:
            return safe_id
    digest = hashlib.sha256(query_record["query"].encode("utf-8")).hexdigest()[:16]
    return f"query_{digest}"


def _criteria_from_query_record(query_record: dict[str, Any]) -> list[str]:
    rubric = query_record.get("rubric")
    if not isinstance(rubric, dict):
        return []
    criteria: list[str] = []
    for item in rubric.get("criteria", []):
        text = ""
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            value = item.get("text") or item.get("criterion") or item.get("description")
            if isinstance(value, str):
                text = value.strip()
        if text:
            criteria.append(text)
    return criteria


def _api_generation_unsupported(settings: PublicSettings) -> list[str]:
    if settings.openai.use_azure and settings.openai.azure_api_version != "v1":
        return [
            "Azure OpenAI hosted-search generation requires AZURE_OPENAI_API_VERSION=v1 "
            "and a deployment entitled for Responses hosted web_search."
        ]
    return []


def _judge_missing(settings: PublicSettings) -> list[str]:
    missing = list(settings.openai.missing_for_judging())
    if not settings.has_anthropic:
        missing.append("ANTHROPIC_API_KEY")
    return _dedupe(missing)


def _cost_component(name: str, provider: str, calls: int, usd_per_call: float) -> dict[str, Any]:
    estimated = calls * usd_per_call
    return {
        "name": name,
        "provider": provider,
        "calls": calls,
        "usd_per_call": round(usd_per_call, 6),
        "estimated_usd": round(estimated, 4),
    }


def _cost_estimate_for_query_count(
    settings: PublicSettings,
    *,
    query_count: int,
    judge: bool,
) -> dict[str, Any]:
    components = [
        _cost_component(
            "openai_generation_responses",
            "azure_openai" if settings.openai.use_azure else "openai",
            query_count,
            settings.cost.openai_generation_usd_per_call,
        ),
        _cost_component(
            "openai_web_search_tool",
            "azure_openai" if settings.openai.use_azure else "openai",
            query_count,
            settings.cost.openai_web_search_usd_per_call,
        ),
    ]
    judge_calls = {"openai": 0, "anthropic_opus": 0, "anthropic_sonnet": 0}
    if judge:
        judge_calls = {
            "openai": query_count,
            "anthropic_opus": query_count,
            "anthropic_sonnet": query_count,
        }
        components.extend(
            [
                _cost_component(
                    "openai_judge",
                    "openai",
                    query_count,
                    settings.cost.openai_judge_usd_per_call,
                ),
                _cost_component(
                    "anthropic_opus_judge",
                    "anthropic",
                    query_count,
                    settings.cost.anthropic_opus_judge_usd_per_call,
                ),
                _cost_component(
                    "anthropic_sonnet_judge",
                    "anthropic",
                    query_count,
                    settings.cost.anthropic_sonnet_judge_usd_per_call,
                ),
            ]
        )

    total = sum(component["estimated_usd"] for component in components)
    return {
        "estimate_version": 1,
        "query_count": query_count,
        "generation_calls": query_count,
        "web_search_tool_calls_estimated": query_count,
        "judge_requested": judge,
        "judge_calls": judge_calls,
        "components": components,
        "estimated_total_usd": round(total, 4),
        "basis": settings.cost.note,
        "overridable_env_vars": [
            "DR_COST_OPENAI_GENERATION_USD_PER_CALL",
            "DR_COST_OPENAI_WEB_SEARCH_USD_PER_CALL",
            "DR_COST_OPENAI_JUDGE_USD_PER_CALL",
            "DR_COST_ANTHROPIC_OPUS_JUDGE_USD_PER_CALL",
            "DR_COST_ANTHROPIC_SONNET_JUDGE_USD_PER_CALL",
        ],
    }


def estimate_api_reproduction_cost(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
) -> dict[str, Any]:
    """Estimate paid API calls and configurable dollar guardrail for a rerun."""
    query_count = len(
        _select_queries(_load_public_queries(settings.paths.project_root), full=full, limit=limit)
    )
    return _cost_estimate_for_query_count(settings, query_count=query_count, judge=judge)


def _cost_block_message(cost_estimate: dict[str, Any], max_cost_usd: float | None) -> str:
    if max_cost_usd is None:
        return ""
    if max_cost_usd < 0:
        raise ValueError("--max-cost-usd must be non-negative")
    estimated = float(cost_estimate["estimated_total_usd"])
    if estimated > max_cost_usd:
        return f"estimated cost ${estimated:.4f} exceeds --max-cost-usd ${max_cost_usd:.4f}"
    return ""


def _generation_prompt(query_record: dict[str, Any]) -> str:
    query = query_record["query"]
    rubric = query_record.get("rubric", {})
    criteria = rubric.get("criteria", []) if isinstance(rubric, dict) else []
    criteria_text = "\n".join(
        f"- {item.get('text', item)}" for item in criteria[:8] if isinstance(item, (dict, str))
    )
    return (
        "Write a concise, citation-rich research report for the paper supplement. "
        "Use current web evidence where the hosted search tool is available. "
        "Be explicit about uncertainty and limitations.\n\n"
        f"Research query:\n{query}\n\n"
        f"Public rubric hints:\n{criteria_text or 'No query-specific rubric hints supplied.'}\n\n"
        "Return Markdown with a title, short abstract, sections, inline citations, and references."
    )


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", "")
    if direct:
        return direct
    texts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    data: dict[str, Any] = {}
    for field in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        value = getattr(usage, field, None)
        if value is not None:
            data[field] = value
    return data


def _openai_client(settings: PublicSettings) -> tuple[Any, str, str]:
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI generation requires the `api` extra: `pip install -e .[api]`."
        ) from exc
    timeout = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=30.0)
    if settings.openai.use_azure:
        return (
            AsyncOpenAI(
                api_key=settings.openai.azure_api_key,
                base_url=settings.openai.azure_v1_base_url,
                timeout=timeout,
                max_retries=0,
            ),
            "azure_openai",
            settings.openai.generation_call_model,
        )
    return (
        AsyncOpenAI(api_key=settings.openai.api_key, timeout=timeout, max_retries=0),
        "openai",
        settings.openai.generation_call_model,
    )


async def _generate_report(
    settings: PublicSettings,
    query_record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    query_id = _query_file_stem(query_record)
    markdown_path = output_dir / f"{query_id}.md"
    json_path = output_dir / f"{query_id}.json"
    client, provider_mode, call_model = _openai_client(settings)
    tool_type = settings.search.openai_web_search_tool
    result = {
        "query_id": query_id,
        "source_query_id": str(query_record.get("id") or ""),
        "query": query_record["query"],
        "provider": provider_mode,
        "provider_mode": provider_mode,
        "configured_model": settings.openai.model,
        "call_model_or_deployment": call_model,
        "web_search_tool": tool_type,
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "status": "failed",
        "error_type": "",
        "error_message": "",
        "usage": {},
    }
    try:
        response = await client.responses.create(
            model=call_model,
            input=_generation_prompt(query_record),
            tools=[{"type": tool_type}],
            max_output_tokens=4096,
        )
        markdown = _response_text(response)
        if not markdown:
            raise ValueError("OpenAI response did not contain output text")
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
        )
    else:
        markdown_path.write_text(markdown + "\n")
        result.update({"status": "success", "usage": _usage_dict(response)})
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


async def _generate_reports(
    settings: PublicSettings,
    query_records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for query_record in query_records:
        results.append(await _generate_report(settings, query_record, output_dir))
    return results


def run_smoke_reproduction(settings: PublicSettings) -> ReproductionReport:
    """Run the no-network public smoke check."""
    reference = load_reference_results(settings.paths.project_root)
    headline_path = settings.paths.project_root / REFERENCE_HEADLINE_PATH
    headline = json.loads(headline_path.read_text()) if headline_path.exists() else {}
    required_keys = {"paper", "reproduction_contract", "reference_metrics"}
    required_headline_keys = {"query_count", "pattern_count", "primary_ordering"}
    missing = sorted(required_keys - set(reference))
    missing_headline = sorted(required_headline_keys - set(headline))
    status = "error" if missing or missing_headline else "success"
    if status == "success":
        message = "reference summaries are present"
    else:
        message = f"missing reference keys: {missing}; missing headline keys: {missing_headline}"
    return ReproductionReport(
        mode="smoke",
        status=status,
        message=message,
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
        details={
            "reference_keys": sorted(reference.keys()),
            "headline_reference_path": str(headline_path),
            "headline_reference_keys": sorted(headline.keys()),
        },
    )


def run_reference_summary(settings: PublicSettings) -> ReproductionReport:
    """Return the compact public paper-reference summary without network calls."""
    headline_path = settings.paths.project_root / REFERENCE_HEADLINE_PATH
    if not headline_path.exists():
        return ReproductionReport(
            mode="reference",
            status="error",
            message=f"missing headline reference: {headline_path}",
            created_utc=datetime.now(timezone.utc).isoformat(),
            reference_path=str(headline_path),
        )
    headline = json.loads(headline_path.read_text())
    ordering = headline.get("primary_ordering", [])
    top_patterns = ordering[:5] if isinstance(ordering, list) else []
    details = {
        "paper": headline.get("paper"),
        "query_count": headline.get("query_count"),
        "pattern_count": headline.get("pattern_count"),
        "primary_metric": headline.get("primary_metric"),
        "headline_ranges": headline.get("headline_ranges"),
        "top_patterns": top_patterns,
        "comparison_policy": headline.get("comparison_policy"),
        "pattern_metrics_csv_path": str(
            settings.paths.project_root / REFERENCE_PATTERN_METRICS_CSV_PATH
        ),
    }
    return ReproductionReport(
        mode="reference",
        status="success",
        message="compact paper reference summary is present",
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference_path=str(headline_path),
        details=details,
    )


def _numeric_metric(item: dict[str, Any]) -> float | None:
    for key in ("mean_3judge", "score", "mean_score", "overall_score"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_pattern_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = [payload]
    for key in ("details", "reference_metrics", "metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for container in containers:
        for key in ("primary_ordering", "pattern_metrics", "metrics_by_pattern"):
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, list):
                metrics = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    pattern = item.get("pattern") or item.get("pattern_name") or item.get("name")
                    score = _numeric_metric(item)
                    if pattern and score is not None:
                        metrics.append({"pattern": str(pattern), "mean_3judge": score})
                if metrics:
                    return metrics
            if isinstance(value, dict):
                metrics = []
                for pattern, item in value.items():
                    if isinstance(item, (int, float)):
                        score = float(item)
                    elif isinstance(item, dict):
                        score = _numeric_metric(item)
                    else:
                        score = None
                    if score is not None:
                        metrics.append({"pattern": str(pattern), "mean_3judge": score})
                if metrics:
                    return sorted(metrics, key=lambda row: row["mean_3judge"], reverse=True)
    return []


def _load_candidate_pattern_payload(run_summary_path: Path) -> dict[str, Any]:
    if run_summary_path.suffix.lower() == ".csv":
        rows: list[dict[str, Any]] = []
        with run_summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                pattern = row.get("pattern") or row.get("pattern_name") or row.get("name")
                if not pattern:
                    continue
                metric_value = row.get("mean_3judge") or row.get("score") or row.get("mean_score")
                try:
                    score = float(metric_value) if metric_value not in (None, "") else None
                except ValueError:
                    score = None
                if score is not None:
                    rows.append({"pattern": pattern, "mean_3judge": score})
        return {"primary_ordering": rows}
    return json.loads(run_summary_path.read_text())


def compare_paper_a_run(settings: PublicSettings, run_summary_path: Path) -> ReproductionReport:
    """Compare a candidate pattern-metric run with the frozen public reference."""
    headline_path = settings.paths.project_root / REFERENCE_HEADLINE_PATH
    headline = json.loads(headline_path.read_text())
    run_summary_path = run_summary_path.resolve()
    candidate = _load_candidate_pattern_payload(run_summary_path)

    reference_metrics = _extract_pattern_metrics(headline)
    candidate_metrics = _extract_pattern_metrics(candidate)
    if not candidate_metrics:
        details = candidate.get("details", {}) if isinstance(candidate.get("details"), dict) else {}
        return ReproductionReport(
            mode="compare",
            status="not-comparable",
            message=(
                "run summary has no pattern-level mean_3judge metrics; "
                "api-best-effort summaries are live demos, not the frozen 13-pattern paper matrix"
            ),
            created_utc=datetime.now(timezone.utc).isoformat(),
            reference_path=str(headline_path),
            output_path=str(run_summary_path),
            details={
                "run_mode": candidate.get("mode"),
                "run_status": candidate.get("status"),
                "query_count": details.get("query_count"),
                "successful_generations": details.get("successful_generations"),
                "judge_requested": details.get("judge_requested"),
                "required_candidate_schema": {
                    "primary_ordering": [{"pattern": "p1_iterative_rag", "mean_3judge": 0.0}]
                },
                "comparison_contract": headline.get("comparison_policy"),
            },
        )

    reference_by_pattern = {row["pattern"]: row["mean_3judge"] for row in reference_metrics}
    candidate_by_pattern = {row["pattern"]: row["mean_3judge"] for row in candidate_metrics}
    overlaps = [pattern for pattern in reference_by_pattern if pattern in candidate_by_pattern]
    if not overlaps:
        return ReproductionReport(
            mode="compare",
            status="not-comparable",
            message="candidate pattern names do not overlap the public reference",
            created_utc=datetime.now(timezone.utc).isoformat(),
            reference_path=str(headline_path),
            output_path=str(run_summary_path),
            details={
                "reference_patterns": sorted(reference_by_pattern),
                "candidate_patterns": sorted(candidate_by_pattern),
            },
        )

    deltas = [
        {
            "pattern": pattern,
            "reference_mean_3judge": round(reference_by_pattern[pattern], 4),
            "candidate_mean_3judge": round(candidate_by_pattern[pattern], 4),
            "delta": round(candidate_by_pattern[pattern] - reference_by_pattern[pattern], 4),
        }
        for pattern in overlaps
    ]
    reference_order = [row["pattern"] for row in reference_metrics if row["pattern"] in overlaps]
    candidate_order = [
        row["pattern"]
        for row in sorted(candidate_metrics, key=lambda item: item["mean_3judge"], reverse=True)
        if row["pattern"] in overlaps
    ]
    same_top = bool(
        reference_order and candidate_order and reference_order[0] == candidate_order[0]
    )

    return ReproductionReport(
        mode="compare",
        status="success" if len(overlaps) == len(reference_by_pattern) else "partial",
        message=f"compared {len(overlaps)}/{len(reference_by_pattern)} reference pattern metrics",
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference_path=str(headline_path),
        output_path=str(run_summary_path),
        details={
            "metric": headline.get("primary_metric", "mean_3judge"),
            "overlap_count": len(overlaps),
            "reference_pattern_count": len(reference_by_pattern),
            "candidate_pattern_count": len(candidate_by_pattern),
            "top_pattern_matches_reference": same_top,
            "reference_ordering_overlap": reference_order,
            "candidate_ordering_overlap": candidate_order,
            "deltas": deltas,
            "comparison_contract": headline.get("comparison_policy"),
        },
    )


def _usage_token_totals(items: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in items:
        usage = item.get("usage", {}) if isinstance(item.get("usage"), dict) else {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += total_tokens
    return totals


def _judge_token_totals(judge_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    for judge_result in judge_results:
        for result in judge_result.get("results", []):
            provider = str(result.get("provider") or "unknown")
            model = str(result.get("model") or result.get("configured_model") or "unknown")
            for buckets, key in ((by_provider, provider), (by_model, model)):
                bucket = buckets.setdefault(
                    key, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
                )
                bucket["calls"] += 1
                bucket["input_tokens"] += int(result.get("input_tokens") or 0)
                bucket["output_tokens"] += int(result.get("output_tokens") or 0)
    return {
        "by_provider": by_provider,
        "by_model": by_model,
        "provider_call_count": sum(v["calls"] for v in by_provider.values()),
    }


def _actual_usage_summary(
    settings: PublicSettings,
    generation_results: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_attempts = len(generation_results)
    generation_successes = sum(1 for item in generation_results if item.get("status") == "success")
    judge_totals = _judge_token_totals(judge_results)
    estimated_incurred_usd = (
        generation_attempts * settings.cost.openai_generation_usd_per_call
        + generation_successes * settings.cost.openai_web_search_usd_per_call
    )
    provider_calls = judge_totals["by_provider"]
    model_calls = judge_totals["by_model"]
    estimated_incurred_usd += (
        provider_calls.get("openai", {}).get("calls", 0)
        * settings.cost.openai_judge_usd_per_call
    )
    estimated_incurred_usd += (
        model_calls.get(settings.anthropic.opus_model, {}).get("calls", 0)
        * settings.cost.anthropic_opus_judge_usd_per_call
    )
    estimated_incurred_usd += (
        model_calls.get(settings.anthropic.sonnet_model, {}).get("calls", 0)
        * settings.cost.anthropic_sonnet_judge_usd_per_call
    )
    return {
        "generation_attempts": generation_attempts,
        "generation_successes": generation_successes,
        "generation_tokens": _usage_token_totals(generation_results),
        "judge_tokens": judge_totals,
        "estimated_incurred_usd": round(estimated_incurred_usd, 4),
        "basis": "Actual token counts where providers returned usage; dollar values use configured per-call estimates.",
    }


async def _verify_openai_generation_entitlement(settings: PublicSettings) -> dict[str, Any]:
    unsupported = _api_generation_unsupported(settings)
    check = {
        "name": "openai_generation_with_hosted_search",
        "provider": "azure_openai" if settings.openai.use_azure else "openai",
        "configured_model": settings.openai.model,
        "call_model_or_deployment": settings.openai.generation_call_model,
        "search_tool": settings.search.openai_web_search_tool,
        "status": "blocked" if unsupported else "pending",
    }
    if unsupported:
        return {**check, "message": "; ".join(unsupported)}
    missing = settings.openai.missing_for_generation()
    if missing:
        return {**check, "status": "blocked", "missing_configuration": missing}
    try:
        client, provider_mode, call_model = _openai_client(settings)
        response = await client.responses.create(
            model=call_model,
            input="Use web search if available, then reply with the word OK.",
            tools=[{"type": settings.search.openai_web_search_tool}],
            tool_choice="required",
            max_output_tokens=32,
        )
        return {
            **check,
            "provider_mode": provider_mode,
            "call_model_or_deployment": call_model,
            "status": "success",
            "usage": _usage_dict(response),
        }
    except Exception as exc:
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }


async def _verify_anthropic_entitlement(settings: PublicSettings, *, model: str, label: str) -> dict[str, Any]:
    check = {
        "name": label,
        "provider": "anthropic",
        "configured_model": model,
        "call_model_or_deployment": model,
        "status": "pending",
    }
    if not settings.has_anthropic:
        return {**check, "status": "blocked", "missing_configuration": ["ANTHROPIC_API_KEY"]}
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": "Anthropic verification requires the `api` extra: `pip install -e .[api]`.",
        }
    try:
        client = AsyncAnthropic(api_key=settings.anthropic.api_key, max_retries=0)
        response = await client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        usage = getattr(response, "usage", None)
        return {
            **check,
            "status": "success",
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
            },
        }
    except Exception as exc:
        return {
            **check,
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }


async def _verify_api_entitlements_async(settings: PublicSettings) -> list[dict[str, Any]]:
    return [
        await _verify_openai_generation_entitlement(settings),
        await _verify_anthropic_entitlement(
            settings, model=settings.anthropic.opus_model, label="anthropic_opus_judge"
        ),
        await _verify_anthropic_entitlement(
            settings, model=settings.anthropic.sonnet_model, label="anthropic_sonnet_judge"
        ),
    ]


def verify_api_entitlements(settings: PublicSettings) -> dict[str, Any]:
    """Make explicit live API entitlement probes for model/tool access.

    This is intentionally separate from dry-run planning because it may create
    small billable provider requests.
    """
    checks = asyncio.run(_verify_api_entitlements_async(settings))
    statuses = {check.get("status") for check in checks}
    if statuses == {"success"}:
        status = "success"
    elif "success" in statuses:
        status = "partial"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "blocked"
    return {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paid_probe": True,
        "checks": checks,
    }


def plan_api_reproduction(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
    max_cost_usd: float | None = None,
) -> ReproductionReport:
    """Prepare a best-effort API reproduction run without launching paid calls."""
    ensure_runtime_dirs(settings)
    selected = _select_queries(
        _load_public_queries(settings.paths.project_root), full=full, limit=limit
    )
    cost_estimate = _cost_estimate_for_query_count(
        settings,
        query_count=len(selected),
        judge=judge,
    )
    missing = list(settings.openai.missing_for_generation())
    unsupported = _api_generation_unsupported(settings)
    if judge:
        missing.extend(_judge_missing(settings))
    if not selected:
        missing.append("data/eval_queries_v2.json contains no public queries")
    missing = _dedupe(missing)
    cost_block = _cost_block_message(cost_estimate, max_cost_usd)

    out_dir = settings.paths.artifacts_dir / "reproduction"
    out_path = out_dir / "paper_a_api_reproduction_plan.json"
    details = {
        "full": full,
        "judge_requested": judge,
        "query_count": len(selected),
        "execute_command": (
            "deep-research reproduce paper-a --mode api-best-effort --execute "
            + ("--full" if full else f"--limit {limit}")
            + (" --judge" if judge else "")
            + (f" --max-cost-usd {max_cost_usd}" if max_cost_usd is not None else "")
        ),
        "judge_command_template": (
            "deep-research judge run --query-file <query.txt> --report-file <report.md> "
            "--criteria-file data/public_judge_criteria.json --panel paper-a-api"
        ),
        "openai_provider_mode": "azure_openai" if settings.openai.use_azure else "openai",
        "openai_model": settings.openai.model,
        "openai_generation_call_model": settings.openai.generation_call_model,
        "azure_api_version": settings.openai.azure_api_version if settings.openai.use_azure else "",
        "openai_judge_model": settings.openai.judge_model,
        "anthropic_opus_model": settings.anthropic.opus_model,
        "anthropic_sonnet_model": settings.anthropic.sonnet_model,
        "search_tool": settings.search.openai_web_search_tool,
        "cost_estimate": cost_estimate,
        "max_cost_usd": max_cost_usd,
        "cost_guardrail_ok": not cost_block,
        "unsupported_configuration": unsupported,
        "contract": "live API demo; not the frozen 13-pattern paper matrix",
        "note": "Best-effort rerun. Exact paper equality is not promised because live APIs drift.",
    }
    blocked_reasons = []
    if missing:
        blocked_reasons.append("missing API settings: " + ", ".join(missing))
    if unsupported:
        blocked_reasons.append("unsupported configuration: " + "; ".join(unsupported))
    if cost_block:
        blocked_reasons.append(cost_block)
    report = ReproductionReport(
        mode="api-best-effort",
        status="blocked" if blocked_reasons else "ready",
        message="; ".join(blocked_reasons) if blocked_reasons else "API settings present",
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
        output_path=str(out_path),
        details=details,
    )
    out_path.write_text(report.to_json() + "\n")
    return report


def run_api_reproduction(
    settings: PublicSettings,
    *,
    full: bool = False,
    limit: int = 3,
    judge: bool = False,
    max_cost_usd: float | None = None,
) -> ReproductionReport:
    """Execute a no-download, OpenAI-hosted-search reproduction subset or full run."""
    ensure_runtime_dirs(settings)
    selected = _select_queries(
        _load_public_queries(settings.paths.project_root), full=full, limit=limit
    )
    cost_estimate = _cost_estimate_for_query_count(
        settings,
        query_count=len(selected),
        judge=judge,
    )
    missing = list(settings.openai.missing_for_generation())
    unsupported = _api_generation_unsupported(settings)
    if judge:
        missing.extend(_judge_missing(settings))
    if not selected:
        missing.append("data/eval_queries_v2.json contains no public queries")
    missing = _dedupe(missing)
    cost_block = _cost_block_message(cost_estimate, max_cost_usd)
    if missing or unsupported or cost_block:
        reasons = []
        if missing:
            reasons.append("missing API settings: " + ", ".join(missing))
        if unsupported:
            reasons.append("unsupported configuration: " + "; ".join(unsupported))
        if cost_block:
            reasons.append(cost_block)
        return ReproductionReport(
            mode="api-best-effort",
            status="blocked",
            message="; ".join(reasons),
            created_utc=datetime.now(timezone.utc).isoformat(),
            reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
            details={
                "missing_configuration": missing,
                "unsupported_configuration": unsupported,
                "judge_requested": judge,
                "cost_estimate": cost_estimate,
                "max_cost_usd": max_cost_usd,
                "cost_guardrail_ok": not cost_block,
            },
        )

    output_dir = settings.paths.artifacts_dir / "reproduction" / "paper_a_api_best_effort"
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_results = asyncio.run(_generate_reports(settings, selected, output_dir))

    judge_results: list[dict[str, Any]] = []
    if judge:
        fallback_criteria_path = settings.paths.project_root / PUBLIC_CRITERIA_PATH
        for query_record, generation in zip(selected, generation_results):
            if generation.get("status") != "success":
                continue
            query_id = generation["query_id"]
            query_criteria = _criteria_from_query_record(query_record)
            criteria_source = "query_rubric"
            if query_criteria:
                criteria_path = output_dir / f"{query_id}.criteria.json"
                criteria_path.write_text(
                    json.dumps({"source": criteria_source, "criteria": query_criteria}, indent=2)
                    + "\n"
                )
            else:
                criteria_source = "public_smoke_fallback"
                criteria_path = fallback_criteria_path
            judge_path = output_dir / f"{query_id}.judge.json"
            judge_report = run_judge_file(
                settings,
                query=query_record["query"],
                report_file=Path(generation["markdown_path"]),
                criteria_file=criteria_path,
                panel="paper-a-api",
                output_path=judge_path,
                dry_run=False,
            )
            judge_payload = asdict(judge_report)
            judge_payload["criteria_source"] = criteria_source
            judge_payload["criteria_count_from_query"] = len(query_criteria)
            judge_results.append(judge_payload)

    summary_path = output_dir / "summary.json"
    success_count = sum(1 for result in generation_results if result.get("status") == "success")
    generation_count = len(generation_results)
    if judge:
        judge_success_count = sum(
            1 for result in judge_results if result.get("status") == "success"
        )
        expected_judge_count = success_count
        if success_count == generation_count and judge_success_count == expected_judge_count:
            status = "success"
        elif success_count or judge_success_count:
            status = "partial"
        else:
            status = "failed"
        message = (
            f"generated {success_count}/{generation_count} public API reports; "
            f"judged {judge_success_count}/{expected_judge_count} successful generations"
        )
    else:
        status = (
            "success"
            if success_count == generation_count and generation_count > 0
            else "partial"
            if success_count
            else "failed"
        )
        message = f"generated {success_count}/{generation_count} public API reports"
    details = {
        "full": full,
        "query_count": len(selected),
        "successful_generations": success_count,
        "failed_generations": generation_count - success_count,
        "judge_requested": judge,
        "successful_judges": sum(
            1 for result in judge_results if result.get("status") == "success"
        ),
        "failed_or_partial_judges": sum(
            1 for result in judge_results if result.get("status") != "success"
        ),
        "cost_estimate": cost_estimate,
        "actual_usage_summary": _actual_usage_summary(settings, generation_results, judge_results),
        "max_cost_usd": max_cost_usd,
        "cost_guardrail_ok": True,
        "cost_guardrail_strategy": (
            "stop before paid execution if the selected plan exceeds --max-cost-usd"
        ),
        "contract": "live API demo; not the frozen 13-pattern paper matrix",
        "generation_results": generation_results,
        "judge_results": judge_results,
    }
    report = ReproductionReport(
        mode="api-best-effort",
        status=status,
        message=message,
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference_path=str(settings.paths.project_root / REFERENCE_RESULTS_PATH),
        output_path=str(summary_path),
        details=details,
    )
    summary_path.write_text(report.to_json() + "\n")
    return report
