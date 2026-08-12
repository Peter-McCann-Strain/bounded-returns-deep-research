"""Tests for execution_pipeline.py and judge_pipeline.py."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.evaluation.execution_pipeline import (
    ExecutionPipeline,
    PipelineProgress,
    RunResult,
)
from deep_research.evaluation.judge_pipeline import JudgePipeline, JudgeProgress
from deep_research.evaluation.multi_judge import EnsembleResult
from deep_research.evaluation.rubric_v2 import RubricV2, Criterion, DIMENSION_WEIGHTS_V2


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass
class FakeQuery:
    """Minimal duck-typed query for testing."""

    id: str
    query: str
    rubric: RubricV2 = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.rubric is None:
            self.rubric = RubricV2(
                query_id=self.id,
                query_text=self.query,
                criteria=[
                    Criterion("Test criterion", "coverage", source="test"),
                ],
                dimension_weights=DIMENSION_WEIGHTS_V2.copy(),
            )


def make_fake_report():
    """Create a minimal mock ResearchReport."""
    from deep_research.types import ResearchReport, Section, Citation

    return ResearchReport(
        query="test query",
        title="Test Report",
        abstract="Test abstract",
        sections=[Section(title="Section 1", content="Content here.")],
        citations=[Citation(claim="claim", source_title="source", source_url="http://example.com")],
        total_cost_usd=0.05,
        total_tokens=1500,
    )


def make_ensemble_result(
    query_id: str = "q1",
    pattern_name: str = "p0_baseline",
    overall: float = 0.5,
    agreement: float = 0.8,
) -> EnsembleResult:
    """Create a minimal EnsembleResult for testing."""
    return EnsembleResult(
        query_id=query_id,
        pattern_name=pattern_name,
        individual_passes=[],
        ensemble_overall=overall,
        ensemble_dimensions={"coverage": 0.6, "factual_accuracy": 0.4},
        intra_judge_consistency={},
        inter_judge_agreement=agreement,
        krippendorffs_alpha=0.7,
        per_dimension_agreement={"coverage": 0.9},
        n_judges=2,
        n_passes_per_judge=3,
        total_evaluations=6,
    )


# ── RunResult tests ──────────────────────────────────────────────────────────


class TestRunResult:
    def test_construction(self):
        r = RunResult(pattern="p0_baseline", query_id="q1", status="success")
        assert r.pattern == "p0_baseline"
        assert r.query_id == "q1"
        assert r.status == "success"
        assert r.report_text == ""
        assert r.elapsed_seconds == 0.0

    def test_succeeded_true_on_success(self):
        r = RunResult(pattern="p0", query_id="q1", status="success")
        assert r.succeeded is True

    def test_succeeded_false_on_error(self):
        r = RunResult(pattern="p0", query_id="q1", status="error")
        assert r.succeeded is False

    def test_succeeded_false_on_content_filter(self):
        r = RunResult(pattern="p0", query_id="q1", status="content_filter")
        assert r.succeeded is False

    def test_succeeded_false_on_skipped(self):
        r = RunResult(pattern="p0", query_id="q1", status="skipped")
        assert r.succeeded is False

    def test_metadata_default_empty(self):
        r = RunResult(pattern="p0", query_id="q1", status="success")
        assert r.metadata == {}


# ── PipelineProgress tests ───────────────────────────────────────────────────


class TestPipelineProgress:
    def test_remaining(self):
        p = PipelineProgress(total_runs=10, completed=3)
        assert p.remaining == 7

    def test_remaining_zero_when_done(self):
        p = PipelineProgress(total_runs=5, completed=5)
        assert p.remaining == 0

    def test_elapsed_minutes_zero_without_start(self):
        p = PipelineProgress(total_runs=10)
        assert p.elapsed_minutes == 0

    def test_eta_minutes_zero_when_no_completed(self):
        p = PipelineProgress(total_runs=10, start_time=1000.0)
        assert p.eta_minutes == 0

    def test_summary_format(self):
        p = PipelineProgress(
            total_runs=10,
            completed=3,
            succeeded=2,
            failed=1,
            skipped=0,
            start_time=1.0,
        )
        s = p.summary()
        assert "3/10" in s
        assert "2 ok" in s
        assert "1 failed" in s
        assert "0 skipped" in s


# ── ExecutionPipeline checkpoint tests ───────────────────────────────────────


class TestExecutionPipelineCheckpoint:
    def test_is_completed_no_checkpoint(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        assert pipeline.is_completed("p0_baseline", "q1") is False

    def test_save_checkpoint_creates_files(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        result = RunResult(
            pattern="p0_baseline",
            query_id="q1",
            status="success",
            report_text="# Test Report\n\nContent here.",
        )
        pipeline.save_checkpoint(result)

        # Checkpoint JSON should exist
        cp = tmp_path / "cp" / "p0_baseline" / "q1.json"
        assert cp.exists()
        data = json.loads(cp.read_text())
        assert data["status"] == "success"
        assert data["pattern"] == "p0_baseline"

        # Report markdown should also exist
        rp = tmp_path / "results" / "reports" / "p0_baseline" / "q1.md"
        assert rp.exists()
        assert rp.read_text() == "# Test Report\n\nContent here."

    def test_save_checkpoint_no_report_for_failure(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        result = RunResult(
            pattern="p0_baseline",
            query_id="q1",
            status="error",
            error_message="Something went wrong",
        )
        pipeline.save_checkpoint(result)

        # Checkpoint JSON should exist
        cp = tmp_path / "cp" / "p0_baseline" / "q1.json"
        assert cp.exists()

        # Report markdown should NOT exist (failed run)
        rp = tmp_path / "results" / "reports" / "p0_baseline" / "q1.md"
        assert not rp.exists()

    def test_load_checkpoint_roundtrip(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        original = RunResult(
            pattern="p1_iterative_rag",
            query_id="q2",
            status="success",
            report_text="Report content",
            elapsed_seconds=42.5,
            total_tokens=5000,
            cost_usd=0.10,
            timestamp="2026-03-11T10:00:00",
            metadata={"n_sections": 3},
        )
        pipeline.save_checkpoint(original)

        loaded = pipeline.load_checkpoint("p1_iterative_rag", "q2")
        assert loaded is not None
        assert loaded.pattern == "p1_iterative_rag"
        assert loaded.query_id == "q2"
        assert loaded.status == "success"
        assert loaded.report_text == "Report content"
        assert loaded.elapsed_seconds == 42.5
        assert loaded.total_tokens == 5000
        assert loaded.cost_usd == 0.10
        assert loaded.metadata == {"n_sections": 3}

    def test_load_checkpoint_nonexistent(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        assert pipeline.load_checkpoint("p0_baseline", "nonexistent") is None

    def test_is_completed_after_save(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        result = RunResult(
            pattern="p0_baseline",
            query_id="q1",
            status="success",
            report_text="content",
        )
        pipeline.save_checkpoint(result)
        assert pipeline.is_completed("p0_baseline", "q1") is True

    def test_is_completed_false_for_failed(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        result = RunResult(
            pattern="p0_baseline",
            query_id="q1",
            status="error",
        )
        pipeline.save_checkpoint(result)
        assert pipeline.is_completed("p0_baseline", "q1") is False


# ── ExecutionPipeline.run_single tests ───────────────────────────────────────


class TestRunSingle:
    @pytest.mark.asyncio
    async def test_run_single_success(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        fake_report = make_fake_report()

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = fake_report

            result = await pipeline.run_single("p0_baseline", "test query", "q1")

        assert result.status == "success"
        assert result.succeeded is True
        assert result.pattern == "p0_baseline"
        assert result.query_id == "q1"
        assert result.total_tokens == 1500
        assert result.cost_usd == 0.05
        assert result.report_text != ""
        assert result.elapsed_seconds > 0
        assert result.metadata["n_sections"] == 1
        assert result.metadata["n_citations"] == 1

    @pytest.mark.asyncio
    async def test_run_single_error(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = RuntimeError("Connection failed")

            result = await pipeline.run_single("p0_baseline", "test query", "q1")

        assert result.status == "error"
        assert result.succeeded is False
        assert "RuntimeError" in result.error_message
        assert "Connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_run_single_content_filter(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception(
                "Error code: content_filter - The response was filtered"
            )

            result = await pipeline.run_single("p0_baseline", "test query", "q1")

        assert result.status == "content_filter"
        assert result.succeeded is False

    @pytest.mark.asyncio
    async def test_run_single_budget_exceeded(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Budget exceeded: $2.05 >= $2.00")

            result = await pipeline.run_single("p0_baseline", "test query", "q1")

        assert result.status == "budget_exceeded"
        assert result.succeeded is False

    @pytest.mark.asyncio
    async def test_run_single_content_management_filter(self, tmp_path):
        """Test that content_management errors are also classified as content_filter."""
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("content_management_policy violation")

            result = await pipeline.run_single("p0_baseline", "test query", "q1")

        assert result.status == "content_filter"


# ── ExecutionPipeline.run_all tests ──────────────────────────────────────────


class TestRunAll:
    @pytest.mark.asyncio
    async def test_run_all_with_resume_skips_completed(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
            max_concurrent=1,
        )

        # Pre-save a checkpoint for q1
        completed_result = RunResult(
            pattern="p0_baseline",
            query_id="q1",
            status="success",
            report_text="Already done",
        )
        pipeline.save_checkpoint(completed_result)

        queries = [FakeQuery(id="q1", query="query 1"), FakeQuery(id="q2", query="query 2")]
        fake_report = make_fake_report()

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = fake_report

            results = await pipeline.run_all(
                queries=queries,
                patterns=["p0_baseline"],
                resume=True,
            )

        # q1 should be loaded from checkpoint, q2 should be run
        assert mock_exec.call_count == 1  # Only q2 was run
        assert len(results) == 2
        statuses = {r.query_id: r.status for r in results}
        assert statuses["q1"] == "success"  # From checkpoint
        assert statuses["q2"] == "success"  # Freshly run

    @pytest.mark.asyncio
    async def test_run_all_tracks_progress(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
            max_concurrent=1,
            log_interval=1,  # Log every run for testing
        )

        queries = [
            FakeQuery(id="q1", query="query 1"),
            FakeQuery(id="q2", query="query 2"),
        ]
        fake_report = make_fake_report()

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = fake_report

            results = await pipeline.run_all(
                queries=queries,
                patterns=["p0_baseline"],
                resume=False,
            )

        assert len(results) == 2
        # All should be successful
        assert all(r.succeeded for r in results)

        # Summary file should be saved
        summary_path = tmp_path / "results" / "pipeline_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["total_runs"] == 2
        assert summary["by_status"]["success"] == 2

    @pytest.mark.asyncio
    async def test_run_all_no_resume_runs_everything(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
            max_concurrent=1,
        )

        # Pre-save a checkpoint
        completed_result = RunResult(
            pattern="p0_baseline",
            query_id="q1",
            status="success",
            report_text="Already done",
        )
        pipeline.save_checkpoint(completed_result)

        queries = [FakeQuery(id="q1", query="query 1")]
        fake_report = make_fake_report()

        with patch.object(pipeline, "_execute_pattern", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = fake_report

            results = await pipeline.run_all(
                queries=queries,
                patterns=["p0_baseline"],
                resume=False,
            )

        # Even though checkpoint exists, resume=False should re-run
        assert mock_exec.call_count == 1
        assert len(results) == 1


# ── ExecutionPipeline.get_all_results tests ──────────────────────────────────


class TestGetAllResults:
    def test_get_all_results_loads_saved_checkpoints(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        # Save some checkpoints
        for qid in ["q1", "q2", "q3"]:
            result = RunResult(
                pattern="p0_baseline",
                query_id=qid,
                status="success",
                report_text=f"Report {qid}",
            )
            pipeline.save_checkpoint(result)

        loaded = pipeline.get_all_results()
        assert len(loaded) == 3
        query_ids = {r.query_id for r in loaded}
        assert query_ids == {"q1", "q2", "q3"}

    def test_get_all_results_multiple_patterns(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        for pattern in ["p0_baseline", "p1_iterative_rag"]:
            result = RunResult(
                pattern=pattern,
                query_id="q1",
                status="success",
                report_text=f"Report {pattern}",
            )
            pipeline.save_checkpoint(result)

        loaded = pipeline.get_all_results()
        assert len(loaded) == 2
        patterns = {r.pattern for r in loaded}
        assert patterns == {"p0_baseline", "p1_iterative_rag"}

    def test_get_all_results_empty(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )
        loaded = pipeline.get_all_results()
        assert loaded == []

    def test_get_all_results_skips_corrupt_json(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        # Save a valid checkpoint
        result = RunResult(
            pattern="p0_baseline", query_id="q1", status="success"
        )
        pipeline.save_checkpoint(result)

        # Write a corrupt checkpoint
        corrupt_dir = tmp_path / "cp" / "p0_baseline"
        (corrupt_dir / "q_corrupt.json").write_text("not valid json{{{")

        loaded = pipeline.get_all_results()
        assert len(loaded) == 1
        assert loaded[0].query_id == "q1"


# ── JudgeProgress tests ─────────────────────────────────────────────────────


class TestJudgeProgress:
    def test_remaining(self):
        p = JudgeProgress(total=20, completed=5)
        assert p.remaining == 15

    def test_summary_format(self):
        p = JudgeProgress(total=10, completed=3, start_time=1.0)
        s = p.summary()
        assert "3/10" in s
        assert "elapsed" in s


# ── JudgePipeline tests ─────────────────────────────────────────────────────


class TestJudgePipeline:
    def test_is_scored_false_initially(self, tmp_path):
        pipeline = JudgePipeline(
            multi_judge=MagicMock(),
            reports_dir=tmp_path / "reports",
            output_dir=tmp_path / "output",
        )
        assert pipeline.is_scored("p0_baseline", "q1") is False

    def test_is_scored_true_after_verdict_saved(self, tmp_path):
        pipeline = JudgePipeline(
            multi_judge=MagicMock(),
            reports_dir=tmp_path / "reports",
            output_dir=tmp_path / "output",
        )
        # Manually create a verdict file
        vp = tmp_path / "output" / "verdicts" / "p0_baseline" / "q1.json"
        vp.parent.mkdir(parents=True, exist_ok=True)
        vp.write_text(json.dumps({"query_id": "q1", "pattern_name": "p0_baseline"}))

        assert pipeline.is_scored("p0_baseline", "q1") is True

    def test_load_report_reads_markdown(self, tmp_path):
        pipeline = JudgePipeline(
            multi_judge=MagicMock(),
            reports_dir=tmp_path / "reports_root",
            output_dir=tmp_path / "output",
        )

        # Create the report file in the expected location
        report_dir = tmp_path / "reports_root" / "reports" / "p0_baseline"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "q1.md").write_text("# Test Report\n\nContent here.")

        text = pipeline._load_report("p0_baseline", "q1")
        assert text == "# Test Report\n\nContent here."

    def test_load_report_returns_none_when_missing(self, tmp_path):
        pipeline = JudgePipeline(
            multi_judge=MagicMock(),
            reports_dir=tmp_path / "reports_root",
            output_dir=tmp_path / "output",
        )
        assert pipeline._load_report("p0_baseline", "nonexistent") is None

    @pytest.mark.asyncio
    async def test_score_all_with_mock_multi_judge(self, tmp_path):
        # Set up the reports directory with some reports
        reports_dir = tmp_path / "reports_root"
        for pattern in ["p0_baseline"]:
            report_dir = reports_dir / "reports" / pattern
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "q1.md").write_text("# Report\n\nContent for q1.")
            (report_dir / "q2.md").write_text("# Report\n\nContent for q2.")

        # Create mock multi-judge
        mock_judge = AsyncMock()
        mock_result = make_ensemble_result()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        pipeline = JudgePipeline(
            multi_judge=mock_judge,
            reports_dir=reports_dir,
            output_dir=tmp_path / "output",
            log_interval=1,
        )

        queries = [
            FakeQuery(id="q1", query="query 1"),
            FakeQuery(id="q2", query="query 2"),
        ]

        results = await pipeline.score_all(
            queries=queries,
            patterns=["p0_baseline"],
            resume=False,
        )

        assert len(results) == 2
        assert mock_judge.evaluate.call_count == 2

        # Verdict files should be saved
        v1 = tmp_path / "output" / "verdicts" / "p0_baseline" / "q1.json"
        v2 = tmp_path / "output" / "verdicts" / "p0_baseline" / "q2.json"
        assert v1.exists()
        assert v2.exists()

        # Summary should be saved
        summary = tmp_path / "output" / "judge_summary.json"
        assert summary.exists()
        summary_data = json.loads(summary.read_text())
        assert summary_data["total_scored"] == 2

    @pytest.mark.asyncio
    async def test_score_all_resume_skips_scored(self, tmp_path):
        # Set up reports
        reports_dir = tmp_path / "reports_root"
        report_dir = reports_dir / "reports" / "p0_baseline"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "q1.md").write_text("# Report q1")
        (report_dir / "q2.md").write_text("# Report q2")

        # Pre-save a verdict for q1
        output_dir = tmp_path / "output"
        vp = output_dir / "verdicts" / "p0_baseline" / "q1.json"
        vp.parent.mkdir(parents=True, exist_ok=True)
        vp.write_text(json.dumps({"query_id": "q1", "pattern_name": "p0_baseline"}))

        mock_judge = AsyncMock()
        mock_result = make_ensemble_result(query_id="q2")
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        pipeline = JudgePipeline(
            multi_judge=mock_judge,
            reports_dir=reports_dir,
            output_dir=output_dir,
        )

        queries = [
            FakeQuery(id="q1", query="query 1"),
            FakeQuery(id="q2", query="query 2"),
        ]

        results = await pipeline.score_all(
            queries=queries,
            patterns=["p0_baseline"],
            resume=True,
        )

        # Only q2 should have been scored (q1 was already scored)
        assert mock_judge.evaluate.call_count == 1
        assert len(results) == 1
        assert results[0].query_id == "q2"

    @pytest.mark.asyncio
    async def test_score_all_skips_missing_reports(self, tmp_path):
        """Reports that don't exist should be silently skipped."""
        reports_dir = tmp_path / "reports_root"
        # Only create report for q1, not q2
        report_dir = reports_dir / "reports" / "p0_baseline"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "q1.md").write_text("# Report q1")

        mock_judge = AsyncMock()
        mock_result = make_ensemble_result(query_id="q1")
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        pipeline = JudgePipeline(
            multi_judge=mock_judge,
            reports_dir=reports_dir,
            output_dir=tmp_path / "output",
        )

        queries = [
            FakeQuery(id="q1", query="query 1"),
            FakeQuery(id="q2", query="query 2"),  # No report for this
        ]

        results = await pipeline.score_all(
            queries=queries,
            patterns=["p0_baseline"],
            resume=False,
        )

        # Only q1 should be scored (q2 has no report)
        assert mock_judge.evaluate.call_count == 1
        assert len(results) == 1


# ── ExecutionPipeline PATTERN_NAMES constant ─────────────────────────────────


class TestPatternNames:
    def test_has_registered_patterns(self):
        assert len(ExecutionPipeline.PATTERN_NAMES) == 16

    def test_pattern_names_ordered(self):
        assert ExecutionPipeline.PATTERN_NAMES[0] == "p0_baseline"
        assert ExecutionPipeline.PATTERN_NAMES[5] == "p5_hierarchical_wd"
        assert ExecutionPipeline.PATTERN_NAMES[10] == "p10_deep_researcher"
        assert ExecutionPipeline.PATTERN_NAMES[-1] == "p17_scale_qwen25_14b"

    def test_all_pattern_names_start_with_p(self):
        for name in ExecutionPipeline.PATTERN_NAMES:
            assert name.startswith("p")


# ── _save_summary tests ─────────────────────────────────────────────────────


class TestSaveSummary:
    def test_save_summary_creates_json(self, tmp_path):
        pipeline = ExecutionPipeline(
            checkpoint_dir=tmp_path / "cp",
            results_dir=tmp_path / "results",
        )

        results = [
            RunResult(pattern="p0_baseline", query_id="q1", status="success"),
            RunResult(pattern="p0_baseline", query_id="q2", status="error"),
            RunResult(pattern="p1_iterative_rag", query_id="q1", status="success"),
        ]
        pipeline._save_summary(results)

        summary_path = tmp_path / "results" / "pipeline_summary.json"
        assert summary_path.exists()

        data = json.loads(summary_path.read_text())
        assert data["total_runs"] == 3
        assert data["by_status"]["success"] == 2
        assert data["by_status"]["error"] == 1
        assert data["by_pattern"]["p0_baseline"]["success"] == 1
        assert data["by_pattern"]["p0_baseline"]["failed"] == 1
        assert data["by_pattern"]["p1_iterative_rag"]["success"] == 1
