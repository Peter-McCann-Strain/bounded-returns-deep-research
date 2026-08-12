"""Tests for retrieval-generation separation evaluation."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.evaluation.retrieval_eval import (
    RetrievalGenerationReport,
    RetrievalMetrics,
    SynthesisMetrics,
    ThreeLevelCitationResult,
    compute_citation_density,
    compute_retrieval_metrics,
    compute_source_diversity,
    compute_synthesis_metrics,
    extract_domain,
    three_level_citation_accuracy,
)
from deep_research.types import Citation


# ── 1. compute_source_diversity ──────────────────────────────────────────────


class TestComputeSourceDiversity:
    def test_uniform_distribution_max_entropy(self):
        """N distinct domains => log2(N) entropy."""
        domains = ["a.com", "b.com", "c.com", "d.com"]
        entropy = compute_source_diversity(domains)
        assert abs(entropy - math.log2(4)) < 1e-6

    def test_single_domain_zero_entropy(self):
        """All same domain => 0 entropy."""
        domains = ["a.com", "a.com", "a.com"]
        entropy = compute_source_diversity(domains)
        assert entropy == 0.0

    def test_empty_list_zero(self):
        entropy = compute_source_diversity([])
        assert entropy == 0.0

    def test_single_item_zero(self):
        """One item => 0 entropy (no distribution to measure)."""
        entropy = compute_source_diversity(["a.com"])
        assert entropy == 0.0

    def test_two_domains_equal_split(self):
        """Two domains, 50/50 => log2(2) = 1.0."""
        domains = ["a.com", "b.com"]
        entropy = compute_source_diversity(domains)
        assert abs(entropy - 1.0) < 1e-6

    def test_skewed_distribution(self):
        """Skewed distribution => lower entropy than uniform."""
        domains = ["a.com", "a.com", "a.com", "b.com"]
        entropy = compute_source_diversity(domains)
        # Entropy should be between 0 and log2(2)=1.0
        assert 0.0 < entropy < 1.0

    def test_ten_unique_domains(self):
        domains = [f"domain{i}.com" for i in range(10)]
        entropy = compute_source_diversity(domains)
        assert abs(entropy - math.log2(10)) < 1e-6


# ── 2. extract_domain ────────────────────────────────────────────────────────


class TestExtractDomain:
    def test_basic_url(self):
        assert extract_domain("https://arxiv.org/abs/1234") == "arxiv.org"

    def test_with_www(self):
        """www. prefix is stripped."""
        assert extract_domain("https://www.example.com/page") == "example.com"

    def test_http_url(self):
        assert extract_domain("http://scholar.google.com/search") == "scholar.google.com"

    def test_no_scheme(self):
        """URL without scheme gets https:// prepended."""
        assert extract_domain("arxiv.org/abs/1234") == "arxiv.org"

    def test_with_port(self):
        assert extract_domain("https://localhost:8080/path") == "localhost"

    def test_empty_string(self):
        assert extract_domain("") == "unknown"

    def test_whitespace_only(self):
        assert extract_domain("   ") == "unknown"

    def test_subdomain(self):
        assert extract_domain("https://api.semanticscholar.org/v1") == "api.semanticscholar.org"

    def test_complex_path(self):
        domain = extract_domain("https://doi.org/10.1038/nature12373")
        assert domain == "doi.org"

    def test_none_like(self):
        """Empty-ish values return unknown."""
        assert extract_domain("") == "unknown"


# ── 3. compute_citation_density ──────────────────────────────────────────────


class TestComputeCitationDensity:
    def test_known_density(self):
        """100 words with 5 citation markers => 50 per 1000 words."""
        words = " ".join(["word"] * 95 + ["[1]", "[2]", "[3]", "[4]", "[5]"])
        density = compute_citation_density(words)
        # 5 markers / 100 tokens * 1000 = 50
        assert abs(density - 50.0) < 1e-6

    def test_no_citations(self):
        text = "This text has no citation markers at all."
        density = compute_citation_density(text)
        assert density == 0.0

    def test_empty_text(self):
        assert compute_citation_density("") == 0.0

    def test_whitespace_only(self):
        assert compute_citation_density("   ") == 0.0

    def test_repeated_citation(self):
        """Same citation marker multiple times counts each occurrence."""
        text = "Claim [1] and another claim [1] and a third [2]."
        density = compute_citation_density(text)
        # 3 markers in ~10 words => 300 per 1000
        words = len(text.split())
        expected = (3 / words) * 1000
        assert abs(density - expected) < 1e-6


# ── 4. compute_retrieval_metrics with mock sources ───────────────────────────


class TestComputeRetrievalMetrics:
    def test_basic_metrics(self):
        sources = [
            {"url": "https://arxiv.org/abs/1", "content": "A" * 1000},
            {"url": "https://arxiv.org/abs/2", "content": "B" * 500},
            {"url": "https://example.com/page", "content": "C" * 200},
        ]
        metrics = compute_retrieval_metrics(sources)
        assert metrics.total_sources_retrieved == 3
        assert metrics.unique_urls == 3
        assert metrics.urls_with_full_content == 3  # all > 50 chars
        assert metrics.academic_sources == 2  # arxiv
        assert metrics.web_sources == 1
        assert metrics.avg_content_length > 0
        assert metrics.source_diversity > 0  # two distinct domains

    def test_empty_sources(self):
        metrics = compute_retrieval_metrics([])
        assert metrics.total_sources_retrieved == 0
        assert metrics.unique_urls == 0
        assert metrics.source_diversity == 0.0
        assert metrics.avg_content_length == 0

    def test_duplicate_urls(self):
        sources = [
            {"url": "https://arxiv.org/abs/1", "content": "X" * 100},
            {"url": "https://arxiv.org/abs/1", "content": "Y" * 200},
        ]
        metrics = compute_retrieval_metrics(sources)
        assert metrics.total_sources_retrieved == 2
        assert metrics.unique_urls == 1

    def test_sources_with_no_content(self):
        sources = [
            {"url": "https://example.com/a", "content": ""},
            {"url": "https://example.com/b", "content": "short"},
        ]
        metrics = compute_retrieval_metrics(sources)
        assert metrics.urls_with_full_content == 0  # both < 50 chars

    def test_domain_distribution(self):
        sources = [
            {"url": "https://arxiv.org/1", "content": "A" * 100},
            {"url": "https://arxiv.org/2", "content": "B" * 100},
            {"url": "https://nature.com/3", "content": "C" * 100},
        ]
        metrics = compute_retrieval_metrics(sources)
        assert "arxiv.org" in metrics.domain_distribution
        assert metrics.domain_distribution["arxiv.org"] == 2
        assert metrics.domain_distribution["nature.com"] == 1

    def test_with_document_objects(self):
        """Test with objects that have url/content attributes."""
        doc1 = MagicMock()
        doc1.url = "https://arxiv.org/abs/123"
        doc1.content = "Paper content " * 50
        doc1.summary = ""

        doc2 = MagicMock()
        doc2.url = "https://blog.com/post"
        doc2.content = "Blog content " * 30
        doc2.summary = ""

        metrics = compute_retrieval_metrics([doc1, doc2])
        assert metrics.total_sources_retrieved == 2
        assert metrics.academic_sources == 1
        assert metrics.web_sources == 1

    def test_median_content_length_odd(self):
        sources = [
            {"url": "https://a.com/1", "content": "A" * 100},
            {"url": "https://b.com/2", "content": "B" * 300},
            {"url": "https://c.com/3", "content": "C" * 200},
        ]
        metrics = compute_retrieval_metrics(sources)
        # Sorted lengths: [100, 200, 300], median = 200
        assert metrics.median_content_length == 200

    def test_median_content_length_even(self):
        sources = [
            {"url": "https://a.com/1", "content": "A" * 100},
            {"url": "https://b.com/2", "content": "B" * 300},
        ]
        metrics = compute_retrieval_metrics(sources)
        # Sorted lengths: [100, 300], median = 200
        assert metrics.median_content_length == 200


# ── 5. compute_synthesis_metrics with known report text ──────────────────────


class TestComputeSynthesisMetrics:
    def test_basic_report(self):
        report = (
            "# Report Title\n"
            "## Abstract\n"
            "This is the abstract section with some words.\n"
            "## Methods\n"
            "The methods section describes [1] how the study was done [2].\n"
            "## Conclusion\n"
            "We conclude that the results are significant [1].\n"
        )
        citations = [
            Citation(source_url="https://a.com", source_title="A"),
            Citation(source_url="https://b.com", source_title="B"),
        ]
        metrics = compute_synthesis_metrics(
            report_text=report,
            citations=citations,
            claim_count=10,
            attributed_count=7,
        )
        assert metrics.total_sections > 0
        assert metrics.total_words > 0
        assert metrics.has_abstract is True
        assert metrics.has_conclusion is True
        assert metrics.unique_sources_cited == 2
        assert metrics.citation_density > 0
        assert abs(metrics.attribution_rate - 0.7) < 1e-6

    def test_empty_report(self):
        metrics = compute_synthesis_metrics("", claim_count=0, attributed_count=0)
        assert metrics.total_sections == 0
        assert metrics.total_words == 0
        assert metrics.citation_density == 0.0
        assert metrics.has_abstract is False
        assert metrics.has_conclusion is False

    def test_no_abstract_no_conclusion(self):
        report = "## Introduction\nSome intro text.\n## Discussion\nSome discussion.\n"
        metrics = compute_synthesis_metrics(report)
        assert metrics.has_abstract is False
        assert metrics.has_conclusion is False

    def test_attribution_rate_zero_claims(self):
        report = "## Section\nSome text here."
        metrics = compute_synthesis_metrics(report, claim_count=0, attributed_count=0)
        assert metrics.attribution_rate == 0.0

    def test_no_citations_list(self):
        report = "## Section\nSome text [1] here."
        metrics = compute_synthesis_metrics(report, citations=None)
        assert metrics.unique_sources_cited == 0

    def test_avg_section_length(self):
        report = (
            "## Section One\n"
            + " ".join(["word"] * 100) + "\n"
            + "## Section Two\n"
            + " ".join(["word"] * 200) + "\n"
        )
        metrics = compute_synthesis_metrics(report)
        assert metrics.total_sections == 2
        assert metrics.avg_section_length == 150


# ── 6. RetrievalMetrics fields populated correctly ───────────────────────────


class TestRetrievalMetricsFields:
    def test_all_fields_present(self):
        m = RetrievalMetrics(
            total_sources_retrieved=10,
            unique_urls=8,
            urls_with_full_content=7,
            academic_sources=4,
            web_sources=6,
            source_diversity=2.5,
            avg_content_length=3000,
            median_content_length=2500,
            domain_distribution={"arxiv.org": 4, "example.com": 6},
        )
        assert m.total_sources_retrieved == 10
        assert m.academic_sources + m.web_sources == 10
        assert "arxiv.org" in m.domain_distribution

    def test_default_domain_distribution(self):
        m = RetrievalMetrics(
            total_sources_retrieved=0,
            unique_urls=0,
            urls_with_full_content=0,
            academic_sources=0,
            web_sources=0,
            source_diversity=0.0,
            avg_content_length=0,
            median_content_length=0,
        )
        assert m.domain_distribution == {}


# ── 7. SynthesisMetrics attribution_rate computation ─────────────────────────


class TestSynthesisMetricsAttributionRate:
    def test_perfect_attribution(self):
        m = SynthesisMetrics(
            total_sections=3,
            total_words=1000,
            total_claims=10,
            attributed_claims=10,
            attribution_rate=1.0,
            unique_sources_cited=5,
            citation_density=15.0,
            has_abstract=True,
            has_conclusion=True,
            avg_section_length=333,
        )
        assert m.attribution_rate == 1.0

    def test_no_attribution(self):
        m = SynthesisMetrics(
            total_sections=3,
            total_words=1000,
            total_claims=10,
            attributed_claims=0,
            attribution_rate=0.0,
            unique_sources_cited=0,
            citation_density=0.0,
            has_abstract=False,
            has_conclusion=False,
            avg_section_length=333,
        )
        assert m.attribution_rate == 0.0

    def test_partial_attribution_via_function(self):
        report = "## Section\nText [1] and more text."
        citations = [Citation(source_url="https://a.com", source_title="A")]
        metrics = compute_synthesis_metrics(
            report, citations=citations, claim_count=4, attributed_count=2
        )
        assert abs(metrics.attribution_rate - 0.5) < 1e-6


# ── 8. Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_sources_retrieval(self):
        metrics = compute_retrieval_metrics([])
        assert metrics.total_sources_retrieved == 0
        assert metrics.source_diversity == 0.0

    def test_empty_report_synthesis(self):
        metrics = compute_synthesis_metrics("")
        assert metrics.total_words == 0
        assert metrics.total_sections == 0

    def test_whitespace_report_synthesis(self):
        metrics = compute_synthesis_metrics("  \n  \n  ")
        assert metrics.total_words == 0

    def test_sources_without_url_attribute(self):
        """Non-dict, non-object sources are skipped."""
        metrics = compute_retrieval_metrics(["not_a_source", 42])
        assert metrics.total_sources_retrieved == 2
        # These items don't have url/content, should still not crash
        assert metrics.unique_urls == 0

    def test_report_no_markdown_headers(self):
        """Report without markdown headers."""
        text = "This is a plain text report with no headers. It has [1] citations."
        metrics = compute_synthesis_metrics(text)
        # Should have one implicit section
        assert metrics.total_sections == 1
        assert metrics.total_words > 0


# ── ThreeLevelCitationResult construction ────────────────────────────────────


class TestThreeLevelCitationResult:
    def test_basic_construction(self):
        r = ThreeLevelCitationResult(
            doc_accuracy=0.9,
            sec_accuracy=0.7,
            sent_accuracy=0.5,
            n_citations_evaluated=10,
        )
        assert r.doc_accuracy == 0.9
        assert r.sec_accuracy == 0.7
        assert r.sent_accuracy == 0.5
        assert r.n_citations_evaluated == 10


# ── RetrievalGenerationReport construction ───────────────────────────────────


class TestRetrievalGenerationReport:
    def test_construction_without_three_level(self):
        r = RetrievalGenerationReport(
            pattern="p0",
            query_id="q1",
            retrieval=RetrievalMetrics(
                total_sources_retrieved=5,
                unique_urls=5,
                urls_with_full_content=4,
                academic_sources=3,
                web_sources=2,
                source_diversity=1.5,
                avg_content_length=2000,
                median_content_length=1800,
            ),
            synthesis=SynthesisMetrics(
                total_sections=4,
                total_words=3000,
                total_claims=20,
                attributed_claims=15,
                attribution_rate=0.75,
                unique_sources_cited=5,
                citation_density=10.0,
                has_abstract=True,
                has_conclusion=True,
                avg_section_length=750,
            ),
        )
        assert r.three_level is None
        assert r.pattern == "p0"


# ── three_level_citation_accuracy integration ────────────────────────────────


class TestThreeLevelIntegration:
    @pytest.mark.asyncio
    async def test_empty_citations(self):
        llm = MagicMock()
        llm.complete_json = AsyncMock(return_value={})
        result = await three_level_citation_accuracy(
            report_text="Some text.",
            citations=[],
            source_extractions=[],
            llm_caller=llm,
        )
        assert result.n_citations_evaluated == 0
        assert result.doc_accuracy == 0.0

    @pytest.mark.asyncio
    async def test_empty_report(self):
        llm = MagicMock()
        llm.complete_json = AsyncMock(return_value={})
        result = await three_level_citation_accuracy(
            report_text="",
            citations=[Citation(source_url="https://a.com", source_title="A")],
            source_extractions=[],
            llm_caller=llm,
        )
        assert result.n_citations_evaluated == 0

    @pytest.mark.asyncio
    async def test_all_levels_positive(self):
        """All three levels return positive results."""
        llm = MagicMock()
        llm.complete_json = AsyncMock(
            side_effect=[
                # Doc-Acc
                {"relevant": True, "reasoning": "Relevant to topic"},
                # Sec-Acc
                {"appropriate": True, "reasoning": "Good section placement"},
                # Sent-Acc
                {"supports": True, "reasoning": "Source confirms claim"},
            ]
        )

        citations = [
            Citation(
                source_url="https://arxiv.org/abs/1",
                source_title="Paper A",
                source_id="doc1",
            ),
        ]

        # Create mock source extraction
        source_ext = MagicMock()
        source_ext.url = "https://arxiv.org/abs/1"
        source_ext.doc_id = "doc1"
        source_ext.title = "Paper A"
        source_ext.summary = "This paper discusses methods for NLP."
        source_ext.content = "Full content of the paper."

        report = (
            "# NLP Methods\n"
            "## Introduction\n"
            "This paper uses transformers for NLP tasks [1].\n"
        )

        result = await three_level_citation_accuracy(
            report_text=report,
            citations=citations,
            source_extractions=[source_ext],
            llm_caller=llm,
        )

        assert result.n_citations_evaluated == 1
        assert result.doc_accuracy == 1.0
        assert result.sec_accuracy == 1.0
        assert result.sent_accuracy == 1.0
