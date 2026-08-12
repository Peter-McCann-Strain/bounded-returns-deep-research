"""Tests for component-level evaluation (search, extraction, synthesis)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.evaluation.component_eval import (
    SearchEvalResult,
    ExtractionEvalResult,
    SynthesisEvalResult,
    ComponentEvalResult,
    evaluate_search_component,
    evaluate_extraction_component,
    evaluate_synthesis_component,
    aggregate_component_results,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_llm(**overrides):
    """Create a mock LLMCaller that returns controlled JSON."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="mock")
    llm.complete_json = AsyncMock(return_value=overrides.get("default_json", {}))
    return llm


# ── 1. SearchEvalResult dataclass ────────────────────────────────────────────


class TestSearchEvalResult:
    def test_retrieval_quality_normal(self):
        r = SearchEvalResult(
            query_id="q1", pattern="p0",
            queries_issued=["q"],
            n_results_retrieved=10,
            n_unique_domains=5,
            n_academic_sources=3,
            diversity_score=0.5,
            relevance_scores=[0.8] * 10,
            mean_relevance=0.8,
            coverage_gaps=[],
        )
        # 0.5*0.8 + 0.3*0.5 + 0.2*min(1, 3/3) = 0.4 + 0.15 + 0.2 = 0.75
        assert abs(r.retrieval_quality - 0.75) < 1e-6

    def test_retrieval_quality_zero_results(self):
        r = SearchEvalResult(
            query_id="q1", pattern="p0",
            queries_issued=[],
            n_results_retrieved=0,
            n_unique_domains=0,
            n_academic_sources=0,
            diversity_score=0.0,
            relevance_scores=[],
            mean_relevance=0.0,
            coverage_gaps=["everything"],
        )
        assert r.retrieval_quality == 0.0

    def test_retrieval_quality_no_academic(self):
        r = SearchEvalResult(
            query_id="q1", pattern="p0",
            queries_issued=["q"],
            n_results_retrieved=5,
            n_unique_domains=5,
            n_academic_sources=0,
            diversity_score=1.0,
            relevance_scores=[1.0] * 5,
            mean_relevance=1.0,
            coverage_gaps=[],
        )
        # 0.5*1.0 + 0.3*1.0 + 0.2*0.0 = 0.8
        assert abs(r.retrieval_quality - 0.8) < 1e-6

    def test_retrieval_quality_excess_academic(self):
        """More than 3 academic sources caps at 1.0 for that component."""
        r = SearchEvalResult(
            query_id="q1", pattern="p0",
            queries_issued=["q"],
            n_results_retrieved=10,
            n_unique_domains=10,
            n_academic_sources=6,
            diversity_score=1.0,
            relevance_scores=[1.0] * 10,
            mean_relevance=1.0,
            coverage_gaps=[],
        )
        # 0.5*1.0 + 0.3*1.0 + 0.2*min(1, 6/3) = 0.5 + 0.3 + 0.2 = 1.0
        assert abs(r.retrieval_quality - 1.0) < 1e-6

    def test_metadata_default(self):
        r = SearchEvalResult(
            query_id="q1", pattern="p0",
            queries_issued=[], n_results_retrieved=0,
            n_unique_domains=0, n_academic_sources=0,
            diversity_score=0.0, relevance_scores=[],
            mean_relevance=0.0, coverage_gaps=[],
        )
        assert r.metadata == {}


# ── 2. ExtractionEvalResult dataclass ────────────────────────────────────────


class TestExtractionEvalResult:
    def test_extraction_quality_normal(self):
        r = ExtractionEvalResult(
            query_id="q1", pattern="p1",
            n_sources_processed=10, n_extractions_produced=8,
            extraction_rate=0.8,
            mean_faithfulness=0.9, mean_informativeness=0.7,
            hallucination_count=1,
        )
        # 0.4*0.9 + 0.3*0.7 + 0.3*0.8 = 0.36 + 0.21 + 0.24 = 0.81
        assert abs(r.extraction_quality - 0.81) < 1e-6

    def test_extraction_quality_zero_sources(self):
        r = ExtractionEvalResult(
            query_id="q1", pattern="p1",
            n_sources_processed=0, n_extractions_produced=0,
            extraction_rate=0.0,
            mean_faithfulness=0.0, mean_informativeness=0.0,
            hallucination_count=0,
        )
        assert r.extraction_quality == 0.0

    def test_extraction_quality_perfect(self):
        r = ExtractionEvalResult(
            query_id="q1", pattern="p1",
            n_sources_processed=5, n_extractions_produced=5,
            extraction_rate=1.0,
            mean_faithfulness=1.0, mean_informativeness=1.0,
            hallucination_count=0,
        )
        # 0.4 + 0.3 + 0.3 = 1.0
        assert abs(r.extraction_quality - 1.0) < 1e-6


# ── 3. SynthesisEvalResult dataclass ─────────────────────────────────────────


class TestSynthesisEvalResult:
    def test_synthesis_quality_normal(self):
        r = SynthesisEvalResult(
            query_id="q1", pattern="p2",
            n_sources_cited=5, n_sources_available=10,
            source_utilization=0.5,
            n_claims_total=20, n_claims_supported=15,
            support_rate=0.75,
            coherence_score=0.8,
        )
        # 0.4*0.75 + 0.3*0.5 + 0.3*0.8 = 0.3 + 0.15 + 0.24 = 0.69
        assert abs(r.synthesis_quality - 0.69) < 1e-6

    def test_synthesis_quality_zero(self):
        r = SynthesisEvalResult(
            query_id="q1", pattern="p2",
            n_sources_cited=0, n_sources_available=0,
            source_utilization=0.0,
            n_claims_total=0, n_claims_supported=0,
            support_rate=0.0, coherence_score=0.0,
        )
        assert r.synthesis_quality == 0.0

    def test_synthesis_quality_perfect(self):
        r = SynthesisEvalResult(
            query_id="q1", pattern="p2",
            n_sources_cited=10, n_sources_available=10,
            source_utilization=1.0,
            n_claims_total=20, n_claims_supported=20,
            support_rate=1.0, coherence_score=1.0,
        )
        assert abs(r.synthesis_quality - 1.0) < 1e-6


# ── 4. ComponentEvalResult.identify_bottleneck ───────────────────────────────


class TestIdentifyBottleneck:
    def test_search_is_bottleneck(self):
        r = ComponentEvalResult(
            query_id="q1", pattern="p0",
            search=SearchEvalResult(
                query_id="q1", pattern="p0",
                queries_issued=["q"], n_results_retrieved=1,
                n_unique_domains=1, n_academic_sources=0,
                diversity_score=0.1, relevance_scores=[0.2],
                mean_relevance=0.2, coverage_gaps=["topic"],
            ),
            extraction=ExtractionEvalResult(
                query_id="q1", pattern="p0",
                n_sources_processed=5, n_extractions_produced=5,
                extraction_rate=1.0,
                mean_faithfulness=0.9, mean_informativeness=0.9,
                hallucination_count=0,
            ),
            synthesis=SynthesisEvalResult(
                query_id="q1", pattern="p0",
                n_sources_cited=5, n_sources_available=5,
                source_utilization=1.0,
                n_claims_total=10, n_claims_supported=10,
                support_rate=1.0, coherence_score=0.9,
            ),
        )
        assert r.identify_bottleneck() == "search"
        assert r.bottleneck == "search"

    def test_extraction_is_bottleneck(self):
        r = ComponentEvalResult(
            query_id="q1", pattern="p0",
            search=SearchEvalResult(
                query_id="q1", pattern="p0",
                queries_issued=["q"], n_results_retrieved=10,
                n_unique_domains=10, n_academic_sources=5,
                diversity_score=1.0, relevance_scores=[0.9] * 10,
                mean_relevance=0.9, coverage_gaps=[],
            ),
            extraction=ExtractionEvalResult(
                query_id="q1", pattern="p0",
                n_sources_processed=10, n_extractions_produced=2,
                extraction_rate=0.2,
                mean_faithfulness=0.3, mean_informativeness=0.2,
                hallucination_count=5,
            ),
            synthesis=SynthesisEvalResult(
                query_id="q1", pattern="p0",
                n_sources_cited=5, n_sources_available=5,
                source_utilization=1.0,
                n_claims_total=10, n_claims_supported=10,
                support_rate=1.0, coherence_score=0.9,
            ),
        )
        assert r.identify_bottleneck() == "extraction"

    def test_synthesis_is_bottleneck(self):
        r = ComponentEvalResult(
            query_id="q1", pattern="p0",
            search=SearchEvalResult(
                query_id="q1", pattern="p0",
                queries_issued=["q"], n_results_retrieved=10,
                n_unique_domains=10, n_academic_sources=5,
                diversity_score=1.0, relevance_scores=[0.9] * 10,
                mean_relevance=0.9, coverage_gaps=[],
            ),
            extraction=ExtractionEvalResult(
                query_id="q1", pattern="p0",
                n_sources_processed=10, n_extractions_produced=10,
                extraction_rate=1.0,
                mean_faithfulness=0.9, mean_informativeness=0.9,
                hallucination_count=0,
            ),
            synthesis=SynthesisEvalResult(
                query_id="q1", pattern="p0",
                n_sources_cited=1, n_sources_available=10,
                source_utilization=0.1,
                n_claims_total=10, n_claims_supported=1,
                support_rate=0.1, coherence_score=0.2,
            ),
        )
        assert r.identify_bottleneck() == "synthesis"

    def test_no_components(self):
        r = ComponentEvalResult(query_id="q1", pattern="p0")
        assert r.identify_bottleneck() == "unknown"

    def test_only_search(self):
        r = ComponentEvalResult(
            query_id="q1", pattern="p0",
            search=SearchEvalResult(
                query_id="q1", pattern="p0",
                queries_issued=["q"], n_results_retrieved=5,
                n_unique_domains=3, n_academic_sources=1,
                diversity_score=0.6, relevance_scores=[0.7] * 5,
                mean_relevance=0.7, coverage_gaps=[],
            ),
        )
        assert r.identify_bottleneck() == "search"


# ── 5. aggregate_component_results ───────────────────────────────────────────


class TestAggregateComponentResults:
    def test_single_pattern_single_run(self):
        results = [
            ComponentEvalResult(
                query_id="q1", pattern="p0",
                search=SearchEvalResult(
                    query_id="q1", pattern="p0",
                    queries_issued=["q"], n_results_retrieved=10,
                    n_unique_domains=5, n_academic_sources=3,
                    diversity_score=0.5, relevance_scores=[0.8] * 10,
                    mean_relevance=0.8, coverage_gaps=[],
                ),
                extraction=ExtractionEvalResult(
                    query_id="q1", pattern="p0",
                    n_sources_processed=10, n_extractions_produced=8,
                    extraction_rate=0.8,
                    mean_faithfulness=0.9, mean_informativeness=0.7,
                    hallucination_count=1,
                ),
                synthesis=SynthesisEvalResult(
                    query_id="q1", pattern="p0",
                    n_sources_cited=5, n_sources_available=10,
                    source_utilization=0.5,
                    n_claims_total=20, n_claims_supported=15,
                    support_rate=0.75, coherence_score=0.8,
                ),
            ),
        ]
        agg = aggregate_component_results(results)
        assert "p0" in agg
        assert agg["p0"]["n_runs"] == 1
        assert abs(agg["p0"]["mean_search_quality"] - 0.75) < 1e-6
        assert abs(agg["p0"]["mean_extraction_quality"] - 0.81) < 1e-6
        assert abs(agg["p0"]["mean_synthesis_quality"] - 0.69) < 1e-6

    def test_multiple_patterns(self):
        r1 = ComponentEvalResult(
            query_id="q1", pattern="p0",
            search=SearchEvalResult(
                query_id="q1", pattern="p0",
                queries_issued=["q"], n_results_retrieved=10,
                n_unique_domains=10, n_academic_sources=3,
                diversity_score=1.0, relevance_scores=[1.0] * 10,
                mean_relevance=1.0, coverage_gaps=[],
            ),
        )
        r2 = ComponentEvalResult(
            query_id="q1", pattern="p1",
            search=SearchEvalResult(
                query_id="q1", pattern="p1",
                queries_issued=["q"], n_results_retrieved=5,
                n_unique_domains=2, n_academic_sources=0,
                diversity_score=0.4, relevance_scores=[0.5] * 5,
                mean_relevance=0.5, coverage_gaps=[],
            ),
        )
        agg = aggregate_component_results([r1, r2])
        assert "p0" in agg
        assert "p1" in agg
        assert agg["p0"]["n_runs"] == 1
        assert agg["p1"]["n_runs"] == 1

    def test_empty_results(self):
        agg = aggregate_component_results([])
        assert agg == {}

    def test_missing_components_in_aggregate(self):
        """Runs with only search should still aggregate."""
        results = [
            ComponentEvalResult(query_id="q1", pattern="p0"),
            ComponentEvalResult(query_id="q2", pattern="p0"),
        ]
        agg = aggregate_component_results(results)
        assert agg["p0"]["mean_search_quality"] == 0.0
        assert agg["p0"]["mean_extraction_quality"] == 0.0
        assert agg["p0"]["mean_synthesis_quality"] == 0.0
        assert agg["p0"]["n_runs"] == 2

    def test_bottleneck_counts(self):
        """Aggregate should count which component is bottleneck most often."""
        r1 = ComponentEvalResult(
            query_id="q1", pattern="p0",
            search=SearchEvalResult(
                query_id="q1", pattern="p0",
                queries_issued=["q"], n_results_retrieved=1,
                n_unique_domains=1, n_academic_sources=0,
                diversity_score=0.1, relevance_scores=[0.1],
                mean_relevance=0.1, coverage_gaps=[],
            ),
            extraction=ExtractionEvalResult(
                query_id="q1", pattern="p0",
                n_sources_processed=5, n_extractions_produced=5,
                extraction_rate=1.0,
                mean_faithfulness=0.9, mean_informativeness=0.9,
                hallucination_count=0,
            ),
        )
        r2 = ComponentEvalResult(
            query_id="q2", pattern="p0",
            search=SearchEvalResult(
                query_id="q2", pattern="p0",
                queries_issued=["q"], n_results_retrieved=1,
                n_unique_domains=1, n_academic_sources=0,
                diversity_score=0.1, relevance_scores=[0.1],
                mean_relevance=0.1, coverage_gaps=[],
            ),
            extraction=ExtractionEvalResult(
                query_id="q2", pattern="p0",
                n_sources_processed=5, n_extractions_produced=5,
                extraction_rate=1.0,
                mean_faithfulness=0.9, mean_informativeness=0.9,
                hallucination_count=0,
            ),
        )
        agg = aggregate_component_results([r1, r2])
        assert agg["p0"]["primary_bottleneck"] == "search"
        assert agg["p0"]["bottleneck_counts"]["search"] == 2


# ── 6. evaluate_search_component with mocked LLM ────────────────────────────


class TestEvaluateSearchComponent:
    @pytest.mark.asyncio
    async def test_empty_docs(self):
        llm = _make_mock_llm()
        result = await evaluate_search_component(
            query_text="What is AI?",
            query_id="q1",
            pattern="p0",
            search_queries=["AI overview"],
            retrieved_docs=[],
            llm=llm,
        )
        assert result.n_results_retrieved == 0
        assert result.mean_relevance == 0.0
        assert result.diversity_score == 0.0
        assert result.coverage_gaps == ["What is AI?"]

    @pytest.mark.asyncio
    async def test_with_docs(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(
            side_effect=[
                # Relevance scores for 2 docs
                {"relevance": 0.8, "reasoning": "relevant"},
                {"relevance": 0.6, "reasoning": "somewhat"},
                # Gap detection
                {"gaps": ["ethics"]},
            ]
        )
        docs = [
            {"url": "https://example.com/a", "title": "AI Intro", "content": "AI is...", "source_type": "web"},
            {"url": "https://arxiv.org/paper1", "title": "AI Paper", "content": "Deep...", "source_type": "arxiv"},
        ]
        result = await evaluate_search_component(
            query_text="What is AI?",
            query_id="q1",
            pattern="p0",
            search_queries=["AI overview"],
            retrieved_docs=docs,
            llm=llm,
        )
        assert result.n_results_retrieved == 2
        assert result.n_unique_domains == 2
        assert result.n_academic_sources == 1
        assert abs(result.mean_relevance - 0.7) < 1e-6
        assert result.coverage_gaps == ["ethics"]

    @pytest.mark.asyncio
    async def test_with_docs_same_domain(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(
            side_effect=[
                {"relevance": 0.5, "reasoning": "ok"},
                {"relevance": 0.5, "reasoning": "ok"},
                {"gaps": []},
            ]
        )
        docs = [
            {"url": "https://example.com/a", "title": "A", "content": "...", "source_type": "web"},
            {"url": "https://example.com/b", "title": "B", "content": "...", "source_type": "web"},
        ]
        result = await evaluate_search_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            search_queries=["q"],
            retrieved_docs=docs,
            llm=llm,
        )
        # Same domain, so diversity = 1/2 = 0.5
        assert result.n_unique_domains == 1
        assert abs(result.diversity_score - 0.5) < 1e-6

    @pytest.mark.asyncio
    async def test_llm_error_in_relevance(self):
        """LLM error should default to 0.5 relevance."""
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(
            side_effect=[
                RuntimeError("LLM down"),  # relevance error -> defaults to 0.5
                {"gaps": []},  # gap detection succeeds
            ]
        )
        docs = [
            {"url": "https://a.com", "title": "A", "content": "c", "source_type": "web"},
        ]
        result = await evaluate_search_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            search_queries=["q"],
            retrieved_docs=docs,
            llm=llm,
        )
        assert abs(result.mean_relevance - 0.5) < 1e-6


# ── 7. evaluate_extraction_component with mocked LLM ────────────────────────


class TestEvaluateExtractionComponent:
    @pytest.mark.asyncio
    async def test_empty_sources(self):
        llm = _make_mock_llm()
        result = await evaluate_extraction_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            sources=[],
            extractions=[],
            llm=llm,
        )
        assert result.n_sources_processed == 0
        assert result.extraction_quality == 0.0

    @pytest.mark.asyncio
    async def test_normal_extraction(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "faithfulness": 0.9,
            "informativeness": 0.8,
            "hallucinated_claims": 1,
        })
        sources = [{"content": "Source text about AI"}]
        extractions = [{"summary": "AI is important"}]
        result = await evaluate_extraction_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            sources=sources,
            extractions=extractions,
            llm=llm,
        )
        assert result.n_sources_processed == 1
        assert result.n_extractions_produced == 1
        assert abs(result.extraction_rate - 1.0) < 1e-6
        assert abs(result.mean_faithfulness - 0.9) < 1e-6
        assert abs(result.mean_informativeness - 0.8) < 1e-6
        assert result.hallucination_count == 1

    @pytest.mark.asyncio
    async def test_fewer_extractions_than_sources(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "faithfulness": 0.7,
            "informativeness": 0.6,
            "hallucinated_claims": 0,
        })
        sources = [{"content": "A"}, {"content": "B"}, {"content": "C"}]
        extractions = [{"summary": "A summary"}]
        result = await evaluate_extraction_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            sources=sources,
            extractions=extractions,
            llm=llm,
        )
        assert result.n_sources_processed == 3
        assert result.n_extractions_produced == 1
        assert abs(result.extraction_rate - 1 / 3) < 1e-6


# ── 8. evaluate_synthesis_component with mocked LLM ─────────────────────────


class TestEvaluateSynthesisComponent:
    @pytest.mark.asyncio
    async def test_empty_report(self):
        llm = _make_mock_llm()
        result = await evaluate_synthesis_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            report_text="",
            available_sources=5,
            llm=llm,
        )
        assert result.n_sources_cited == 0
        assert result.coherence_score == 0.0

    @pytest.mark.asyncio
    async def test_normal_report(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "n_claims_total": 10,
            "n_claims_supported": 7,
            "coherence_score": 0.85,
        })
        report = "Some text [1] about AI [2] and more [3]."
        result = await evaluate_synthesis_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            report_text=report,
            available_sources=5,
            llm=llm,
        )
        assert result.n_sources_cited == 3  # [1], [2], [3]
        assert abs(result.source_utilization - 0.6) < 1e-6  # 3/5
        assert result.n_claims_total == 10
        assert result.n_claims_supported == 7
        assert abs(result.support_rate - 0.7) < 1e-6
        assert abs(result.coherence_score - 0.85) < 1e-6

    @pytest.mark.asyncio
    async def test_more_citations_than_sources(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "n_claims_total": 5,
            "n_claims_supported": 5,
            "coherence_score": 0.9,
        })
        report = "Text [1][2][3][4][5][6][7]."
        result = await evaluate_synthesis_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            report_text=report,
            available_sources=3,
            llm=llm,
        )
        # Utilization capped at 1.0
        assert result.source_utilization == 1.0

    @pytest.mark.asyncio
    async def test_zero_available_sources(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "n_claims_total": 2,
            "n_claims_supported": 1,
            "coherence_score": 0.5,
        })
        report = "Text [1]."
        result = await evaluate_synthesis_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            report_text=report,
            available_sources=0,
            llm=llm,
        )
        assert result.source_utilization == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_in_synthesis(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(side_effect=RuntimeError("LLM down"))
        report = "Text [1]."
        result = await evaluate_synthesis_component(
            query_text="q",
            query_id="q1",
            pattern="p0",
            report_text=report,
            available_sources=5,
            llm=llm,
        )
        # Falls back to defaults
        assert result.n_claims_total == 0
        assert result.coherence_score == 0.5
