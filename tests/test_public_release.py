"""Tests for public-release scaffolding."""

from __future__ import annotations

import json
import re
import tomllib

import pytest
from pathlib import Path

from deep_research.api_judges import parse_json_response
from deep_research.judge_runner import load_criteria, run_judge_file
from deep_research.public_export import export_public_tree
from deep_research.release_audit import load_pii_scopes, audit_release_tree
from deep_research.reproduce import (
    _query_file_stem,
    compare_paper_a_run,
    estimate_api_reproduction_cost,
    plan_api_reproduction,
    run_api_reproduction,
    run_reference_summary,
    run_smoke_reproduction,
    verify_api_entitlements,
)
from deep_research.settings import load_public_settings


def test_settings_load_public_api_env(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "OPENAI_API_KEY": "test-openai",
            "ANTHROPIC_API_KEY": "test-anthropic",
            "OPENAI_JUDGE_MODEL": "judge-model",
        },
    )

    assert settings.has_openai is True
    assert settings.has_anthropic is True
    assert settings.openai.judge_model == "judge-model"
    assert settings.paths.project_root == tmp_path.resolve()


def test_release_audit_flags_private_files(tmp_path):
    (tmp_path / ".env").write_text("PRIVATE_CONFIG=1\n")
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "deep_research" / "__init__.py").write_text("")

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    messages = [finding.message for finding in result.findings]
    assert any("forbidden public-release path" in message for message in messages)


def test_release_audit_allows_basic_source_tree(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "deep_research" / "__init__.py").write_text("input_tokens = 42\n")
    (tmp_path / "README.md").write_text("# Public\n")

    result = audit_release_tree(tmp_path)

    assert result.ok is True


def test_release_audit_enforces_manifest_allowlist(tmp_path):
    (tmp_path / "README.md").write_text("# Public\n")
    (tmp_path / "private_notes.md").write_text("do not ship\n")
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "include_globs": ["README.md", "PUBLIC_MANIFEST.json"],
                "exclude_globs": [],
                "required_paths": ["README.md"],
            }
        )
    )

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any(finding.path == "private_notes.md" for finding in result.findings)


def test_release_audit_scans_binary_local_markers(tmp_path):
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "include_globs": ["PUBLIC_MANIFEST.json", "paper.pdf"],
                "exclude_globs": [],
                "required_paths": ["paper.pdf"],
            }
        )
    )
    (tmp_path / "paper.pdf").write_bytes(
        b"%PDF-1.7\n/Producer (" + b"/home/" + b"researcher/private/main.tex)"
    )

    result = audit_release_tree(tmp_path)

    assert result.ok is False
    assert any(finding.path == "paper.pdf" for finding in result.findings)


def test_release_audit_ignores_env_template_placeholders(tmp_path):
    openai_key_name = "OPENAI_" + "API_KEY"
    anthropic_key_name = "ANTHROPIC_" + "API_KEY"
    (tmp_path / ".env.example").write_text(f"{openai_key_name}=\n{anthropic_key_name}=<your-key>\n")

    result = audit_release_tree(tmp_path, enforce_manifest=False)

    assert result.ok is True


def _write_minimal_reference(root: Path) -> None:
    reference_dir = root / "repro" / "reference"
    reference_dir.mkdir(parents=True)
    (reference_dir / "paper_a_reference.json").write_text(
        json.dumps(
            {
                "paper": "paper-a",
                "reproduction_contract": "best effort",
                "reference_metrics": {},
            }
        )
    )
    (reference_dir / "paper_a_headline_numbers.json").write_text(
        json.dumps(
            {
                "query_count": 90,
                "pattern_count": 2,
                "primary_metric": "mean_3judge",
                "headline_ranges": {"best_pattern": "base_p1"},
                "primary_ordering": [
                    {"pattern": "base_p1", "mean_3judge": 0.67},
                    {"pattern": "base_p0", "mean_3judge": 0.49},
                ],
                "comparison_policy": "compare direction and broad score ranges",
            }
        )
    )


def test_smoke_reproduction_reads_reference(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_smoke_reproduction(settings)

    assert report.status == "success"
    assert report.mode == "smoke"


def test_reference_summary_reports_headline_ordering(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_reference_summary(settings)

    assert report.status == "success"
    assert report.mode == "reference"
    assert report.details["top_patterns"][0]["pattern"] == "base_p1"
    assert report.details["comparison_policy"]


def test_parse_json_response_validates_object():
    assert parse_json_response('{"evaluations": []}') == {"evaluations": []}


def test_parse_json_response_rejects_non_json():
    try:
        parse_json_response("not json")
    except ValueError as exc:
        assert "valid JSON" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_criteria_accepts_json_object_and_text(tmp_path):
    json_path = tmp_path / "criteria.json"
    json_path.write_text(json.dumps({"criteria": [{"criterion": "cite sources"}, "be concise"]}))
    text_path = tmp_path / "criteria.txt"
    text_path.write_text("# comment\n- cover tradeoffs\n* state limits\n")

    assert load_criteria(json_path) == ["cite sources", "be concise"]
    assert load_criteria(text_path) == ["cover tradeoffs", "state limits"]


def test_judge_run_dry_run_does_not_require_api_or_anthropic_package(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    report_path = tmp_path / "report.md"
    criteria_path = tmp_path / "criteria.json"
    report_path.write_text("The report cites sources and states limitations.")
    criteria_path.write_text(json.dumps(["cites sources", "states limitations"]))
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_judge_file(
        settings,
        query="What happened?",
        report_file=report_path,
        criteria_file=criteria_path,
        dry_run=True,
    )

    assert report.status == "dry-run"
    assert report.criteria_count == 2
    assert "ANTHROPIC_API_KEY" in report.missing_configuration


def test_public_export_copies_allowlist_and_audits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "deep_research").mkdir()
    (source / "deep_research" / "__init__.py").write_text("")
    (source / "README.md").write_text("# Public\n")
    (source / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(
            {
                "max_file_mb": 1,
                "required_paths": ["README.md", "PUBLIC_MANIFEST.json"],
                "include_globs": [
                    "README.md",
                    "PUBLIC_MANIFEST.json",
                    "PUBLIC_EXPORT_REPORT.json",
                    "deep_research/**/*.py",
                ],
                "exclude_globs": ["private/**"],
            }
        )
    )
    (source / "private").mkdir()
    (source / "private" / "notes.md").write_text("secret notes\n")
    output = tmp_path / "export"

    result = export_public_tree(source, output)

    assert result.ok is True
    assert (output / "README.md").exists()
    report = json.loads((output / "PUBLIC_EXPORT_REPORT.json").read_text())
    assert report["manifest_sha256"]
    assert report["file_sha256"]["README.md"]
    assert report["source_git"]["commit"]
    assert not (output / "private" / "notes.md").exists()


def test_settings_azure_requires_endpoint_and_deployment(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
        },
    )

    assert settings.has_openai is False
    assert settings.openai.azure_api_version == "v1"
    assert "AZURE_OPENAI_ENDPOINT" in settings.openai.missing_for_judging()
    assert "AZURE_OPENAI_DEPLOYMENT" in settings.openai.missing_for_generation()


def test_parse_json_response_accepts_fenced_json():
    assert parse_json_response('```json\n{"evaluations": []}\n```') == {"evaluations": []}


def test_judge_run_records_provider_failure(monkeypatch, tmp_path):
    import deep_research.judge_runner as judge_runner

    class FailingProvider:
        label = "test-provider"
        model = "test-model"
        provider_mode = "test"
        configured_model = "test-model"
        call_model_or_deployment = "test-model"

        async def evaluate(self, request):
            raise RuntimeError("planned failure")

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    report_path = tmp_path / "report.md"
    criteria_path = tmp_path / "criteria.json"
    out_path = tmp_path / "judge.json"
    report_path.write_text("Report text")
    criteria_path.write_text(json.dumps(["criterion"]))
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai"},
    )
    monkeypatch.setattr(
        judge_runner, "_build_providers", lambda settings, panel: [FailingProvider()]
    )

    report = run_judge_file(
        settings,
        query="Question?",
        report_file=report_path,
        criteria_file=criteria_path,
        panel="openai-only",
        output_path=out_path,
    )

    assert report.status == "failed"
    assert report.results[0]["status"] == "failed"
    assert report.results[0]["error_type"] == "RuntimeError"
    assert out_path.exists()


def test_plan_api_reproduction_uses_public_queries(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "query": "Question 1?"},
                    {"id": "q2", "query": "Question 2?"},
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai"},
    )

    report = plan_api_reproduction(settings, limit=1)

    assert report.status == "ready"
    assert report.details["query_count"] == 1
    assert report.details["judge_requested"] is False
    assert "--execute" in report.details["execute_command"]
    assert "--judge" not in report.details["execute_command"]


def test_plan_api_reproduction_blocks_for_judge_without_anthropic(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai"},
    )

    report = plan_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "blocked"
    assert report.details["judge_requested"] is True
    assert "--judge" in report.details["execute_command"]
    assert "ANTHROPIC_API_KEY" in report.message


def test_cost_estimate_includes_full_judge_components(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {"id": "q1", "query": "Question 1?"},
                    {"id": "q2", "query": "Question 2?"},
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "OPENAI_API_KEY": "test-openai",
            "ANTHROPIC_API_KEY": "test-anthropic",
            "DR_COST_OPENAI_GENERATION_USD_PER_CALL": "1",
            "DR_COST_OPENAI_WEB_SEARCH_USD_PER_CALL": "0.5",
            "DR_COST_OPENAI_JUDGE_USD_PER_CALL": "0.25",
            "DR_COST_ANTHROPIC_OPUS_JUDGE_USD_PER_CALL": "2",
            "DR_COST_ANTHROPIC_SONNET_JUDGE_USD_PER_CALL": "3",
        },
    )

    estimate = estimate_api_reproduction_cost(settings, full=True, judge=True)

    assert estimate["generation_calls"] == 2
    assert estimate["web_search_tool_calls_estimated"] == 2
    assert estimate["judge_calls"] == {
        "openai": 2,
        "anthropic_opus": 2,
        "anthropic_sonnet": 2,
    }
    assert {component["name"] for component in estimate["components"]} == {
        "openai_generation_responses",
        "openai_web_search_tool",
        "openai_judge",
        "anthropic_opus_judge",
        "anthropic_sonnet_judge",
    }
    assert estimate["estimated_total_usd"] == 13.5


def test_plan_api_reproduction_blocks_when_cost_exceeds_limit(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(project_root=tmp_path, env={"OPENAI_API_KEY": "test-openai"})

    report = plan_api_reproduction(settings, limit=1, max_cost_usd=0.01)

    assert report.status == "blocked"
    assert "estimated cost" in report.message
    assert report.details["cost_guardrail_ok"] is False


def test_run_api_reproduction_blocks_for_judge_without_anthropic(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(project_root=tmp_path, env={"OPENAI_API_KEY": "test-openai"})

    report = run_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "blocked"
    assert "ANTHROPIC_API_KEY" in report.message
    assert report.details["judge_requested"] is True


def test_compare_rejects_api_demo_summary_without_pattern_metrics(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    run_summary = tmp_path / "summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "mode": "api-best-effort",
                "status": "success",
                "details": {"query_count": 1, "successful_generations": 1},
            }
        )
    )
    settings = load_public_settings(project_root=tmp_path, env={})

    report = compare_paper_a_run(settings, run_summary)

    assert report.status == "not-comparable"
    assert "13-pattern" in report.message
    assert report.details["required_candidate_schema"]["primary_ordering"]


def test_compare_accepts_pattern_level_metrics(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _write_minimal_reference(tmp_path)
    run_summary = tmp_path / "pattern_summary.json"
    run_summary.write_text(
        json.dumps(
            {
                "primary_ordering": [
                    {"pattern": "base_p1", "mean_3judge": 0.7},
                    {"pattern": "base_p0", "mean_3judge": 0.4},
                ]
            }
        )
    )
    settings = load_public_settings(project_root=tmp_path, env={})

    report = compare_paper_a_run(settings, run_summary)

    assert report.status == "success"
    assert report.details["top_pattern_matches_reference"] is True
    assert report.details["overlap_count"] == 2


def test_documented_judge_dry_run_example_is_shipped():
    assert Path("repro/examples/example_report.md").exists()


def test_run_api_reproduction_blocks_without_generation_key(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(json.dumps({"queries": []}))
    settings = load_public_settings(project_root=tmp_path, env={})

    report = run_api_reproduction(settings, limit=1)

    assert report.status == "blocked"
    assert "OPENAI_API_KEY" in report.message


def test_run_api_reproduction_execute_judge_failure_is_partial(monkeypatch, tmp_path):
    import deep_research.reproduce as reproduce
    from deep_research.judge_runner import JudgeRunReport

    class FakeUsage:
        input_tokens = 10
        output_tokens = 20
        total_tokens = 30

    class FakeResponse:
        output_text = "# Generated report\n\nEvidence-backed summary."
        usage = FakeUsage()

    class FakeResponses:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    def fake_judge_file(*args, **kwargs):
        return JudgeRunReport(
            status="failed",
            panel="paper-a-api",
            created_utc="2026-07-31T00:00:00+00:00",
            query=kwargs["query"],
            report_file=str(kwargs["report_file"]),
            criteria_file=str(kwargs["criteria_file"]),
            output_path=str(kwargs["output_path"]),
            providers=[],
            criteria_count=1,
            missing_configuration=[],
            results=[{"status": "failed", "error_type": "RuntimeError"}],
        )

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    (data_dir / "public_judge_criteria.json").write_text(json.dumps(["criterion"]))
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )
    monkeypatch.setattr(reproduce, "run_judge_file", fake_judge_file)

    report = reproduce.run_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "partial"
    assert report.details["successful_generations"] == 1
    assert report.details["successful_judges"] == 0
    assert report.details["failed_or_partial_judges"] == 1


def test_run_api_reproduction_execute_success_with_fake_openai(monkeypatch, tmp_path):
    import deep_research.reproduce as reproduce

    calls = []

    class FakeUsage:
        input_tokens = 10
        output_tokens = 20
        total_tokens = 30

    class FakeResponse:
        output_text = "# Generated report\n\nEvidence-backed summary."
        usage = FakeUsage()

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "Question 1?",
                        "rubric": {"criteria": [{"text": "cite sources"}]},
                    }
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )

    report = reproduce.run_api_reproduction(settings, limit=1)

    assert report.status == "success"
    assert report.details["successful_generations"] == 1
    assert calls[0]["model"] == settings.openai.model
    assert calls[0]["tools"] == [{"type": "web_search"}]
    assert (tmp_path / "artifacts/reproduction/paper_a_api_best_effort/q1.md").exists()


def test_public_query_manifest_anonymizes_named_affiliate_case():
    text = Path("data/eval_queries_v2.json").read_text()

    # The literal strings used to live here, split across a concatenation so a naive grep
    # would not find them. That made this test file itself a copy of the identity it asserts
    # is absent -- and the file ships. Terms now come from the untracked denylist; the test
    # skips on a clean clone rather than republishing them.
    scopes = load_pii_scopes()
    if not scopes:
        pytest.skip("private denylist absent (expected on a clean clone)")
    for scope in scopes:
        for term in scope.get("terms", []) + scope.get("terms_word_boundary", []):
            assert term.lower() not in text.lower(), f"unredacted identity term in manifest"

    payload = json.loads(text)
    redacted = [
        row
        for row in payload["queries"]
        if row.get("metadata", {}).get("public_redaction")
    ]
    assert redacted
    assert "anonymized" in redacted[0]["metadata"]["public_redaction"].lower()


def test_notice_and_pattern_metrics_are_manifested():
    manifest = json.loads(Path("PUBLIC_MANIFEST.json").read_text())

    assert Path("NOTICE").exists()
    assert "NOTICE" in manifest["required_paths"]
    assert "repro/reference/paper_a_pattern_metrics.csv" in manifest["required_paths"]
    assert "Apache-2.0 applies to code" in Path("README.md").read_text()
    assert "mixed-license" in Path("NOTICE").read_text()


def test_protocol_docs_only_reference_modules_that_actually_ship():
    """Protocol docs may cite implementation modules, but only shipped ones.

    The release ships the full codebase, so referencing
    ``deep_research/evaluation/`` is now correct rather than an overclaim. The
    invariant that still matters is the general one: every module path a
    protocol doc names must exist in the tree, so the docs cannot promise code
    that was never published.
    """
    docs = {
        name: Path(name).read_text()
        for name in ("docs/evaluation_protocol.md", "docs/human_evaluation_protocol.md")
    }

    assert "Public Export Note" in docs["docs/evaluation_protocol.md"]

    # Any `deep_research/...py` (or package dir) referenced must be present.
    module_ref = re.compile(r"`(deep_research/[A-Za-z0-9_/]+(?:\.py)?)`")
    referenced = 0
    for name, text in docs.items():
        for match in module_ref.finditer(text):
            target = Path(match.group(1))
            referenced += 1
            assert target.exists(), f"{name} references missing module {target}"

    # The evaluation protocol should actually cite its implementation.
    assert referenced > 0

    # Data that is deliberately withheld must stay described as withheld.
    protocol = docs["docs/evaluation_protocol.md"]
    assert "does not ship" in protocol
    for withheld in ("results/", "judge verdict"):
        assert withheld in protocol


def test_public_dependency_profile_separates_api_paper_and_local_extras():
    """The release ships the full codebase, so extras must be explicit.

    Heavy work is split into extras rather than forced on every install:
    ``api`` runs the patterns, ``paper`` recomputes the statistics, and
    ``local`` covers the GPU patterns (P9-P17).
    """
    project = tomllib.loads(Path("pyproject.toml").read_text())
    core = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]

    # Core stays light: no provider SDK, no scientific stack, no GPU runtime.
    core_names = {dep.split(">")[0].split("=")[0].strip().lower() for dep in core}
    assert core_names == {"python-dotenv", "pydantic", "structlog", "httpx"}

    def has(extra: str, name: str) -> bool:
        return any(dep.lower().startswith(name) for dep in extras[extra])

    assert has("api", "openai") and has("api", "anthropic") and has("api", "tavily")
    assert has("paper", "pandas") and has("paper", "numpy") and has("paper", "scipy")
    assert has("local", "tor" + "ch") and has("local", "transformers")

    # GPU packages must never land in core or in the API/paper extras: torch has
    # to match the user's CUDA build and would otherwise be pulled in blindly.
    gpu = ("tor" + "ch", "transformers", "bitsandbytes", "accelerate")
    for extra in ("api", "paper", "dev"):
        for dep in extras[extra]:
            assert not dep.lower().startswith(gpu)
    for dep in core:
        assert not dep.lower().startswith(gpu)

    # Default profile runs patterns and recomputes statistics, but not GPU work.
    # Check the install directives themselves, ignoring explanatory comments.
    directives = [
        line.strip()
        for line in Path("requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert directives == ["-e .[api,paper]"]

    # torch stays unpinned in the constraints file for the same CUDA reason.
    constraints = Path("constraints-public.txt").read_text().lower()
    assert not any(
        line.strip().startswith("tor" + "ch==") for line in constraints.splitlines()
    )


def test_query_file_stem_is_safe_and_deterministic():
    assert _query_file_stem({"id": "../../bad id?", "query": "Question?"}) == "bad_id"
    first = _query_file_stem({"query": "Question needing hashed fallback?"})
    second = _query_file_stem({"query": "Question needing hashed fallback?"})

    assert first == second
    assert first.startswith("query_")
    assert "/" not in first


def test_compare_accepts_public_pattern_metrics_csv():
    settings = load_public_settings(project_root=Path.cwd(), env={})

    report = compare_paper_a_run(settings, Path("repro/reference/paper_a_pattern_metrics.csv"))

    assert report.status == "success"
    assert report.details["overlap_count"] == 13
    assert report.details["top_pattern_matches_reference"] is True


def test_azure_v1_base_url_normalizes_resource_endpoint(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
        },
    )

    assert settings.openai.azure_v1_base_url == "https://example.openai.azure.com/openai/v1/"


def test_plan_api_reproduction_supports_azure_hosted_search_config(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps({"queries": [{"id": "q1", "query": "Question 1?"}]})
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={
            "USE_AZURE_OPENAI": "true",
            "AZURE_OPENAI_API_KEY": "test-azure",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "deployment",
        },
    )

    report = plan_api_reproduction(settings, limit=1)

    assert report.status == "ready"
    assert report.details["unsupported_configuration"] == []
    assert report.details["openai_provider_mode"] == "azure_openai"
    assert report.details["openai_generation_call_model"] == "deployment"
    assert report.details["azure_api_version"] == "v1"


def test_verify_api_entitlements_reports_missing_configuration_without_network(tmp_path):
    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    settings = load_public_settings(project_root=tmp_path, env={})

    result = verify_api_entitlements(settings)

    assert result["status"] == "blocked"
    assert result["paid_probe"] is True
    assert all(check["status"] == "blocked" for check in result["checks"])


def test_run_api_reproduction_judge_uses_query_rubric(monkeypatch, tmp_path):
    import deep_research.reproduce as reproduce
    from deep_research.judge_runner import JudgeRunReport

    class FakeResponse:
        output_text = "# Generated report\n\nEvidence-backed summary."
        usage = None

    class FakeResponses:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    captured = {}

    def fake_judge_file(*args, **kwargs):
        criteria_payload = json.loads(Path(kwargs["criteria_file"]).read_text())
        captured["criteria"] = criteria_payload["criteria"]
        return JudgeRunReport(
            status="success",
            panel="paper-a-api",
            created_utc="2026-07-31T00:00:00+00:00",
            query=kwargs["query"],
            report_file=str(kwargs["report_file"]),
            criteria_file=str(kwargs["criteria_file"]),
            output_path=str(kwargs["output_path"]),
            providers=[],
            criteria_count=len(criteria_payload["criteria"]),
            missing_configuration=[],
            results=[],
        )

    (tmp_path / "deep_research").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "eval_queries_v2.json").write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "Question 1?",
                        "rubric": {
                            "criteria": [
                                {"text": "cite sources"},
                                {"text": "state limitations"},
                            ]
                        },
                    }
                ]
            }
        )
    )
    settings = load_public_settings(
        project_root=tmp_path,
        env={"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"},
    )
    monkeypatch.setattr(
        reproduce,
        "_openai_client",
        lambda settings: (FakeClient(), "openai", settings.openai.generation_call_model),
    )
    monkeypatch.setattr(reproduce, "run_judge_file", fake_judge_file)

    report = reproduce.run_api_reproduction(settings, limit=1, judge=True)

    assert report.status == "success"
    assert captured["criteria"] == ["cite sources", "state limitations"]
    assert report.details["judge_results"][0]["criteria_source"] == "query_rubric"
    assert report.details["actual_usage_summary"]["generation_attempts"] == 1


def test_env_example_can_be_copied_verbatim_without_breaking_config(monkeypatch):
    """`cp .env.example .env` is the documented setup step; it must not brick imports.

    Optional settings are marked in .env.example as a bare ``KEY=``. A present-but-empty
    value previously beat the coded default, so the typed getters raised on ``float("")``
    and 18 test modules failed to collect. CI never copies the template, so only this
    test guards the path a reader actually follows.
    """
    from deep_research import config

    empty_keys = [
        line.split("=", 1)[0].strip()
        for line in Path(".env.example").read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#") and line.split("=", 1)[1].strip() == ""
    ]
    assert empty_keys, ".env.example should mark optional settings with a bare KEY="

    for key in empty_keys:
        monkeypatch.setenv(key, "")
        assert config._env(key, "fallback") == "fallback", f"{key}: empty value must not win"
        assert config._env_float(key, 1.5) == 1.5, f"{key}: empty value must not reach float()"
        assert config._env_int(key, 7) == 7, f"{key}: empty value must not reach int()"
        assert config._env_bool(key, True) is True, f"{key}: empty value must not flip a bool"

    # Numeric settings the template leaves blank must still yield usable defaults.
    for key in ("MAX_COST_PER_RUN", "MAX_COST_EVAL_RUN", "DR_LOCAL_CONTENT_CAP"):
        monkeypatch.setenv(key, "")
        config._env_float(key, 10.0)
        config._env_int(key, 10)
