"""Tests for pattern pipeline orchestration (P1-P5).

All LLM calls are mocked — these test the pipeline wiring, state management,
and data flow between stages.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.types import ResearchReport, SubQuery, Document, SourceType
from deep_research.tools.source_extractor import SourceExtraction, ExtractedSourceType


def _make_mock_extraction(doc_id="d1", score=8):
    return SourceExtraction(
        doc_id=doc_id,
        title=f"Source {doc_id}",
        url=f"https://example.com/{doc_id}",
        summary="Mock summary " * 20,
        relevance_score=score,
        source_type=ExtractedSourceType.RESEARCH_PAPER,
        key_findings=["finding 1", "finding 2"],
        confidence_notes="Mocked.",
    )


MOCK_REPORT_MD = (
    "# Test Report\n\n"
    "## Abstract\nAbstract text.\n\n"
    "## Introduction\nIntro [1].\n\n"
    "## Analysis\nAnalysis [2].\n\n"
    "## Conclusion\nConclusion.\n"
)


# ── P1: Iterative RAG ─────────────────────────────────────────────────────────

class TestP1IterativeRAG:
    @pytest.mark.asyncio
    async def test_pipeline_completes(self):
        """P1 pipeline runs to completion with mocked dependencies."""
        from deep_research.patterns.p1_iterative_rag import pipeline

        extractions = [_make_mock_extraction("d1"), _make_mock_extraction("d2")]

        with patch.object(pipeline, 'decompose_query', new_callable=AsyncMock) as mock_decompose, \
             patch.object(pipeline, 'Retriever') as mock_retriever_cls, \
             patch.object(pipeline, 'generate_report', new_callable=AsyncMock) as mock_gen, \
             patch.object(pipeline, 'reflect', new_callable=AsyncMock) as mock_reflect, \
             patch.object(pipeline, 'assemble_report') as mock_assemble, \
             patch.object(pipeline, 'LLMCaller'), \
             patch.object(pipeline, 'CostTracker') as mock_tracker_cls, \
             patch.object(pipeline, 'StateManager'):

            mock_decompose.return_value = [
                SubQuery(query="sub1"), SubQuery(query="sub2"),
            ]

            mock_retriever = MagicMock()
            mock_retriever.search_and_summarize = AsyncMock(return_value=extractions)
            mock_retriever_cls.return_value = mock_retriever

            mock_gen.return_value = MOCK_REPORT_MD
            mock_reflect.return_value = {
                "overall_score": 8,
                "should_continue": False,
            }

            mock_report = ResearchReport(query="test", pattern_name="p1_iterative_rag")
            mock_assemble.return_value = mock_report

            tracker = MagicMock()
            tracker.total_cost = 1.0
            tracker.total_tokens = 5000
            mock_tracker_cls.return_value = tracker

            report = await pipeline.run("test query", budget_usd=5.0)

        assert report.pattern_name == "p1_iterative_rag"
        mock_decompose.assert_called_once()
        mock_gen.assert_called_once()
        mock_reflect.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_gap_fill(self):
        """P1 triggers gap-fill when reflection score is low."""
        from deep_research.patterns.p1_iterative_rag import pipeline

        extractions = [_make_mock_extraction()]

        with patch.object(pipeline, 'decompose_query', new_callable=AsyncMock) as mock_decompose, \
             patch.object(pipeline, 'Retriever') as mock_retriever_cls, \
             patch.object(pipeline, 'generate_report', new_callable=AsyncMock) as mock_gen, \
             patch.object(pipeline, 'reflect', new_callable=AsyncMock) as mock_reflect, \
             patch.object(pipeline, 'assemble_report') as mock_assemble, \
             patch.object(pipeline, 'LLMCaller'), \
             patch.object(pipeline, 'CostTracker') as mock_tracker_cls, \
             patch.object(pipeline, 'StateManager'):

            mock_decompose.return_value = [SubQuery(query="sub1")]

            mock_retriever = MagicMock()
            mock_retriever.search_and_summarize = AsyncMock(return_value=extractions)
            mock_retriever_cls.return_value = mock_retriever

            mock_gen.return_value = MOCK_REPORT_MD

            # First reflect: continue, Second: stop
            mock_reflect.side_effect = [
                {"overall_score": 5, "should_continue": True,
                 "improvement_queries": ["fill gap 1"]},
                {"overall_score": 8, "should_continue": False},
            ]

            mock_assemble.return_value = ResearchReport(
                query="test", pattern_name="p1_iterative_rag",
            )
            tracker = MagicMock()
            tracker.total_cost = 1.0
            tracker.total_tokens = 5000
            mock_tracker_cls.return_value = tracker

            report = await pipeline.run("test", budget_usd=10.0)

        assert mock_gen.call_count == 2  # Generated twice
        assert mock_retriever.search_and_summarize.call_count == 2  # Initial + gap-fill


# ── P1 Report Assembler ───────────────────────────────────────────────────────

class TestP1ReportAssembler:
    def test_assembles_from_markdown(self, sample_extractions):
        from deep_research.patterns.p1_iterative_rag.report_assembler import assemble_report

        md = "# BERT vs GPT\n\n## Abstract\nComparison.\n\n## Arch\nContent [1]."
        report = assemble_report("query", md, sample_extractions, cost_usd=1.0, total_tokens=100)
        assert report.title == "BERT vs GPT"
        assert report.pattern_name == "p1_iterative_rag"
        assert len(report.citations) == 2


# ── P2: Supervisor + Workers ──────────────────────────────────────────────────

class TestP2Supervisor:
    @pytest.mark.asyncio
    async def test_pipeline_imports(self):
        """P2 pipeline module is importable."""
        from deep_research.patterns.p2_supervisor_parallel.pipeline import run
        assert callable(run)


# ── P3: MERIDIAN ─────────────────────────────────────────────────────────────

class TestP3Meridian:
    @pytest.mark.asyncio
    async def test_pipeline_imports(self):
        from deep_research.patterns.p3_meridian.pipeline import run
        assert callable(run)

    def test_rubric_dimensions(self):
        from deep_research.patterns.p3_meridian.rubric import DIMENSIONS
        assert len(DIMENSIONS) == 12
        names = [d.name for d in DIMENSIONS]
        assert "coverage" in names
        assert "accuracy" in names
        assert "depth" in names


# ── P4: Perspective STORM ────────────────────────────────────────────────────

class TestP4Storm:
    @pytest.mark.asyncio
    async def test_pipeline_imports(self):
        from deep_research.patterns.p4_perspective_storm.pipeline import run
        assert callable(run)


# ── P5: Hierarchical W&D ────────────────────────────────────────────────────

class TestP5HierarchicalWD:
    @pytest.mark.asyncio
    async def test_pipeline_imports(self):
        from deep_research.patterns.p5_hierarchical_wd.pipeline import run
        assert callable(run)

    def test_wd_schedule(self):
        from deep_research.patterns.p5_hierarchical_wd.wd_schedule import WDSchedule

        schedule = WDSchedule(w_0=4, alpha=0.5, w_min=1, max_steps=3)
        assert schedule.width_at(0) == 4
        assert schedule.width_at(1) == 2
        assert schedule.width_at(2) == 1

    def test_wd_schedule_floor(self):
        from deep_research.patterns.p5_hierarchical_wd.wd_schedule import WDSchedule

        schedule = WDSchedule(w_0=4, alpha=0.5, w_min=2, max_steps=5)
        # Even at step 10, should not go below w_min
        assert schedule.width_at(10) >= 2

    def test_wd_schedule_depth_increases(self):
        from deep_research.patterns.p5_hierarchical_wd.wd_schedule import WDSchedule

        schedule = WDSchedule(w_0=4, alpha=0.5, w_min=1, max_steps=3)
        # Depth should increase as width decreases
        d0 = schedule.depth_iterations_at(0)
        d2 = schedule.depth_iterations_at(2)
        assert d2 >= d0

    def test_wd_schedule_allocate(self):
        from deep_research.patterns.p5_hierarchical_wd.wd_schedule import WDSchedule

        schedule = WDSchedule(w_0=4, alpha=0.5, w_min=1, max_steps=3)
        alloc = schedule.allocate(0, remaining_budget=10.0)
        assert alloc.width_workers == 4
        assert alloc.width_budget > 0
        assert alloc.depth_budget > 0

    def test_wd_schedule_rebalance(self):
        from deep_research.patterns.p5_hierarchical_wd.wd_schedule import WDSchedule

        schedule = WDSchedule(w_0=4, alpha=0.5, w_min=1, max_steps=3)
        # Low coverage: should boost width
        alloc = schedule.rebalance(1, remaining_budget=5.0, coverage_score=0.3)
        assert alloc.width_workers >= schedule.width_at(1)
