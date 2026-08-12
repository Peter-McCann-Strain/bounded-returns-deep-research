"""Tests for P0 baseline pattern."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.patterns.p0_baseline.pipeline import run, _assemble_report
from deep_research.tools.source_extractor import SourceExtraction, ExtractedSourceType
from deep_research.types import ResearchReport


class TestAssembleReport:
    def test_parses_title(self):
        md = "# My Research Report\n\n## Abstract\nThis is the abstract.\n\n## Section 1\nContent."
        report = _assemble_report("test query", md, [], 1.0, 100)
        assert report.title == "My Research Report"

    def test_parses_abstract(self):
        md = "# Title\n\n## Abstract\nAbstract text here.\n\n## Intro\nIntro text."
        report = _assemble_report("q", md, [], 0, 0)
        assert report.abstract == "Abstract text here."

    def test_parses_sections(self):
        md = "# Title\n\n## Abstract\nAbs.\n\n## Intro\nIntro content.\n\n## Methods\nMethods content."
        report = _assemble_report("q", md, [], 0, 0)
        # Abstract is excluded from sections
        section_titles = [s.title for s in report.sections]
        assert "Intro" in section_titles
        assert "Methods" in section_titles
        assert "Abstract" not in section_titles

    def test_builds_citations(self, sample_extractions):
        md = "# Title\n\n## Body\nContent."
        report = _assemble_report("q", md, sample_extractions, 1.0, 500)
        assert len(report.citations) == 2
        assert report.citations[0].source_url == "https://arxiv.org/abs/1810.04805"

    def test_pattern_name(self):
        md = "# Title\n\n## Body\nContent."
        report = _assemble_report("q", md, [], 0, 0)
        assert report.pattern_name == "p0_baseline"

    def test_cost_and_tokens(self):
        md = "# Title\n\n## Body\nContent."
        report = _assemble_report("q", md, [], 5.0, 10000)
        assert report.total_cost_usd == 5.0
        assert report.total_tokens == 10000

    def test_title_fallback_to_query(self):
        md = "No title header here.\n\nJust some text."
        report = _assemble_report("my query", md, [], 0, 0)
        assert report.title == "my query"


class TestBaselineRun:
    @pytest.mark.asyncio
    async def test_run_end_to_end_mocked(self):
        """Full baseline run with all external calls mocked."""
        from deep_research.types import Document, SourceType

        mock_web_docs = [
            Document(id="w1", title="Web Result", content="Content " * 100,
                     url="https://web.com/1", source_type=SourceType.WEB),
        ]
        mock_academic_docs = [
            Document(id="a1", title="Paper", content="Abstract " * 50,
                     url="https://arxiv.org/1", source_type=SourceType.ACADEMIC),
        ]

        json_extraction = json.dumps({
            "summary": "Test summary " * 20,
            "relevance_score": 8,
            "source_type": "research_paper",
            "key_findings": ["finding 1"],
            "confidence_notes": "reliable",
        })

        report_md = (
            "# Test Research Report\n\n"
            "## Abstract\nThis is a test abstract.\n\n"
            "## Introduction\nIntroduction content [1].\n\n"
            "## Analysis\nAnalysis content [2].\n\n"
            "## Conclusion\nConclusion content.\n"
        )

        with patch('deep_research.patterns.p0_baseline.pipeline.get_web_searcher') as mock_web, \
             patch('deep_research.patterns.p0_baseline.pipeline.AcademicSearcher') as mock_acad_cls, \
             patch('deep_research.patterns.p0_baseline.pipeline.URLExtractor') as mock_url_cls, \
             patch('deep_research.patterns.p0_baseline.pipeline.LLMCaller') as mock_llm_cls, \
             patch('deep_research.patterns.p0_baseline.pipeline.StateManager'):

            # Web search mock
            mock_searcher = MagicMock()
            mock_searcher.search_batch = AsyncMock(return_value=mock_web_docs)
            mock_web.return_value = mock_searcher

            # Academic search mock
            mock_acad = MagicMock()
            mock_acad.search = AsyncMock(return_value=mock_academic_docs)
            mock_acad_cls.return_value = mock_acad

            # URL extractor mock
            mock_url = MagicMock()
            mock_url.extract_batch = AsyncMock(return_value=[])
            mock_url_cls.return_value = mock_url

            # LLM mock — returns analysis, then JSON, then final report
            mock_llm = MagicMock()
            mock_llm.cost_tracker = MagicMock()
            mock_llm.cost_tracker.total_cost = 0.50
            mock_llm.cost_tracker.total_tokens = 5000
            mock_llm.complete = AsyncMock(side_effect=[
                "Free-text analysis 1",  # Source extraction step 1
                json_extraction,          # Source extraction step 2
                "Free-text analysis 2",  # Source extraction step 1
                json_extraction,          # Source extraction step 2
                report_md,               # Final report generation
            ])
            mock_llm_cls.return_value = mock_llm

            report = await run("Compare BERT and GPT", budget_usd=5.0)

        assert isinstance(report, ResearchReport)
        assert report.pattern_name == "p0_baseline"
        assert len(report.sections) >= 2
        assert report.title == "Test Research Report"

    @pytest.mark.asyncio
    async def test_run_no_sources(self):
        """Baseline handles zero results gracefully."""
        with patch('deep_research.patterns.p0_baseline.pipeline.get_web_searcher') as mock_web, \
             patch('deep_research.patterns.p0_baseline.pipeline.AcademicSearcher') as mock_acad_cls, \
             patch('deep_research.patterns.p0_baseline.pipeline.URLExtractor') as mock_url_cls, \
             patch('deep_research.patterns.p0_baseline.pipeline.LLMCaller') as mock_llm_cls, \
             patch('deep_research.patterns.p0_baseline.pipeline.StateManager'):

            mock_searcher = MagicMock()
            mock_searcher.search_batch = AsyncMock(return_value=[])
            mock_web.return_value = mock_searcher

            mock_acad = MagicMock()
            mock_acad.search = AsyncMock(return_value=[])
            mock_acad_cls.return_value = mock_acad

            mock_url = MagicMock()
            mock_url.extract_batch = AsyncMock(return_value=[])
            mock_url_cls.return_value = mock_url

            mock_llm = MagicMock()
            mock_llm.cost_tracker = MagicMock()
            mock_llm.cost_tracker.total_cost = 0.0
            mock_llm.cost_tracker.total_tokens = 0
            mock_llm_cls.return_value = mock_llm

            report = await run("test", budget_usd=1.0)

        assert isinstance(report, ResearchReport)
        assert report.pattern_name == "p0_baseline"
