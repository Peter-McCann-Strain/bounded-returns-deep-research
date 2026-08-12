"""Tests for evaluation framework — metrics, comparator, test queries, runner."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from deep_research.evaluation.metrics import (
    EvalResult,
    score_coverage,
    count_citations,
    count_unique_sources,
    evaluate_report,
)
from deep_research.evaluation.test_queries import (
    TestQuery,
    get_query,
    get_all_queries,
    TEST_QUERIES,
)
from deep_research.evaluation.comparator import generate_comparison
from deep_research.types import ResearchReport, Section, Citation


# ── TestQuery tests ──────────────────────────────────────────────────────────

class TestTestQueries:
    def test_get_all_queries(self):
        queries = get_all_queries()
        assert len(queries) == 5

    def test_get_query_by_id(self):
        q = get_query("q1_bert_vs_gpt")
        assert q.difficulty == "simple"
        assert len(q.expected_elements) == 12

    def test_get_query_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown query ID"):
            get_query("nonexistent")

    def test_all_queries_have_elements(self):
        for q in TEST_QUERIES:
            assert len(q.expected_elements) >= 10
            assert q.difficulty in ("simple", "moderate", "complex")

    def test_query_ids_unique(self):
        ids = [q.id for q in TEST_QUERIES]
        assert len(ids) == len(set(ids))


# ── Metrics tests ────────────────────────────────────────────────────────────

class TestScoreCoverage:
    def test_full_coverage(self):
        report = ResearchReport(
            query="test",
            sections=[Section(
                title="Test",
                content="BERT uses bidirectional masked language modeling. "
                        "GPT uses autoregressive causal language modeling. "
                        "BERT encoder-only architecture. GPT decoder-only architecture."
            )],
        )
        tq = TestQuery(
            id="test",
            query="test",
            difficulty="simple",
            expected_elements=[
                "BERT uses bidirectional/masked language modeling",
                "GPT uses autoregressive/causal language modeling",
            ],
        )
        score, details = score_coverage(report, tq)
        assert score > 0.5

    def test_zero_coverage(self):
        report = ResearchReport(
            query="test",
            sections=[Section(title="Empty", content="Nothing relevant here.")],
        )
        tq = TestQuery(
            id="test", query="test", difficulty="simple",
            expected_elements=["quantum computing", "superconductor"],
        )
        score, details = score_coverage(report, tq)
        assert score == 0.0

    def test_partial_coverage(self):
        report = ResearchReport(
            query="test",
            sections=[Section(title="T", content="BERT uses masked language modeling.")],
        )
        tq = TestQuery(
            id="test", query="test", difficulty="simple",
            expected_elements=[
                "BERT uses masked language modeling",
                "GPT uses autoregressive generation",
            ],
        )
        score, details = score_coverage(report, tq)
        assert 0.0 < score < 1.0


class TestCountCitations:
    def test_count_inline_citations(self, sample_report):
        count = count_citations(sample_report)
        assert count >= 2  # [1] and [2]

    def test_no_citations(self):
        report = ResearchReport(
            query="test",
            sections=[Section(title="T", content="No citations here.")],
        )
        count = count_citations(report)
        assert count == 0

    def test_deduplicates_citations(self):
        report = ResearchReport(
            query="test",
            sections=[Section(title="T", content="See [1] and [1] and [2].")],
        )
        count = count_citations(report)
        assert count == 2  # [1] and [2] unique


class TestCountUniqueSources:
    def test_count_sources(self, sample_report):
        count = count_unique_sources(sample_report)
        assert count == 2

    def test_no_sources(self):
        report = ResearchReport(query="test")
        assert count_unique_sources(report) == 0


class TestEvaluateReport:
    def test_evaluate_report_basic(self, sample_report):
        tq = get_query("q1_bert_vs_gpt")
        result = evaluate_report(sample_report, tq)
        assert isinstance(result, EvalResult)
        assert result.pattern_name == "p1_iterative_rag"
        assert result.query_id == "q1_bert_vs_gpt"
        assert result.cost_usd == 1.50
        assert result.total_tokens == 50000
        assert result.latency_seconds == 120.0
        assert result.section_count == 3
        assert result.report_length_words > 0
        assert 0 <= result.overall_score <= 1.0

    def test_evaluate_report_overall_score_components(self, sample_report):
        tq = TestQuery(id="test", query="test", difficulty="simple",
                       expected_elements=["encoder", "decoder"])
        result = evaluate_report(sample_report, tq)
        # Overall should be a weighted sum of coverage, citations, etc.
        assert result.overall_score > 0

    def test_eval_result_to_dict(self):
        result = EvalResult(
            pattern_name="p1",
            query_id="q1",
            coverage_score=0.8,
            citation_count=10,
            cost_usd=1.5,
        )
        d = result.to_dict()
        assert d["pattern"] == "p1"
        assert d["coverage"] == "80.0%"
        assert d["cost"] == "$1.5000"


# ── Comparator tests ─────────────────────────────────────────────────────────

class TestComparator:
    def test_empty_results(self):
        text = generate_comparison([])
        assert "No results" in text

    def test_single_result(self):
        results = [
            EvalResult(
                pattern_name="p1",
                query_id="q1",
                coverage_score=0.8,
                citation_count=10,
                report_length_words=2000,
                cost_usd=1.5,
                latency_seconds=60,
                overall_score=0.75,
            ),
        ]
        text = generate_comparison(results)
        assert "p1" in text
        assert "Summary by Pattern" in text

    def test_multiple_patterns(self):
        results = [
            EvalResult(pattern_name="p0", query_id="q1", overall_score=0.5,
                       coverage_score=0.6),
            EvalResult(pattern_name="p1", query_id="q1", overall_score=0.7,
                       coverage_score=0.8),
        ]
        text = generate_comparison(results)
        assert "p0" in text
        assert "p1" in text

    def test_coverage_detail_section(self):
        results = [
            EvalResult(
                pattern_name="p1",
                query_id="q1",
                coverage_score=0.5,
                coverage_details={"element1": True, "element2": False},
            ),
        ]
        text = generate_comparison(results)
        assert "Coverage Detail" in text
        assert "+ element1" in text
        assert "- element2" in text


# ── Runner tests ─────────────────────────────────────────────────────────────

class TestRunner:
    @pytest.mark.asyncio
    async def test_run_single(self):
        from deep_research.evaluation.runner import run_single

        mock_report = ResearchReport(
            query="test", pattern_name="p0_baseline",
            total_cost_usd=0.5, total_tokens=1000,
        )
        mock_mod = MagicMock()
        mock_mod.run = AsyncMock(return_value=mock_report)

        with patch('deep_research.evaluation.runner.importlib') as mock_import:
            mock_import.import_module.return_value = mock_mod
            report = await run_single("p0_baseline", "test query", budget_usd=2.0)

        assert report.pattern_name == "p0_baseline"
        assert report.elapsed_seconds > 0

    @pytest.mark.asyncio
    async def test_run_pattern_suite_handles_errors(self):
        from deep_research.evaluation.runner import run_pattern_suite

        mock_mod = MagicMock()
        mock_mod.run = AsyncMock(side_effect=RuntimeError("LLM error"))

        with patch('deep_research.evaluation.runner.importlib') as mock_import:
            mock_import.import_module.return_value = mock_mod
            queries = [TestQuery(id="q1", query="test", difficulty="simple",
                                 expected_elements=["element"])]
            results = await run_pattern_suite("p0_baseline", queries)

        assert len(results) == 1
        assert results[0].overall_score == 0.0  # Error result
