"""Tests for evaluation-level agentic citation verifier (SAFE approach)."""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.evaluation.citation_verifier import (
    AgenticCitationVerifier,
    AtomicClaim,
    CitationVerificationResult,
    ClaimVerification,
    _CITATION_MARKER_PATTERN,
    _DOI_PATTERN,
)
from deep_research.types import Citation


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_llm():
    """Create a mock LLMCaller."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Mock response")
    llm.complete_json = AsyncMock(return_value={})
    return llm


def _make_mock_url_extractor():
    """Create a mock URLExtractor."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=None)
    return extractor


def _make_mock_web_searcher():
    """Create a mock web searcher."""
    searcher = MagicMock()
    searcher.search = AsyncMock(return_value=[])
    return searcher


def _make_verifier(**kwargs):
    """Create a verifier with mocks."""
    llm = kwargs.get("llm") or _make_mock_llm()
    url_ext = kwargs.get("url_extractor") or _make_mock_url_extractor()
    web = kwargs.get("web_searcher")
    return AgenticCitationVerifier(
        llm_caller=llm,
        url_extractor=url_ext,
        web_searcher=web,
        max_claims=kwargs.get("max_claims", 50),
        max_concurrent=kwargs.get("max_concurrent", 5),
    )


# ── 1. AtomicClaim construction ─────────────────────────────────────────────


class TestAtomicClaim:
    def test_basic_construction(self):
        claim = AtomicClaim(
            text="BERT uses bidirectional attention.",
            section="Architecture",
            has_citation=True,
            citation_markers=["[1]"],
            cited_urls=["https://arxiv.org/abs/1810.04805"],
        )
        assert claim.text == "BERT uses bidirectional attention."
        assert claim.section == "Architecture"
        assert claim.has_citation is True
        assert claim.citation_markers == ["[1]"]
        assert claim.cited_urls == ["https://arxiv.org/abs/1810.04805"]

    def test_defaults(self):
        claim = AtomicClaim(text="Fact.", section="Intro", has_citation=False)
        assert claim.citation_markers == []
        assert claim.cited_urls == []

    def test_multiple_citations(self):
        claim = AtomicClaim(
            text="Transformers use attention [1][3].",
            section="Methods",
            has_citation=True,
            citation_markers=["[1]", "[3]"],
        )
        assert len(claim.citation_markers) == 2


# ── 2. ClaimVerification construction ────────────────────────────────────────


class TestClaimVerification:
    def test_supported_verification(self):
        claim = AtomicClaim(text="Fact.", section="S", has_citation=True)
        v = ClaimVerification(
            claim=claim,
            verdict="supported",
            confidence=0.95,
            evidence="Source confirms this.",
            source_url="https://example.com",
            verification_method="citation_check",
        )
        assert v.verdict == "supported"
        assert v.confidence == 0.95
        assert v.verification_method == "citation_check"

    def test_not_supported_verification(self):
        claim = AtomicClaim(text="Wrong fact.", section="S", has_citation=True)
        v = ClaimVerification(
            claim=claim,
            verdict="not_supported",
            confidence=0.8,
            evidence="Source says otherwise.",
        )
        assert v.verdict == "not_supported"

    def test_source_unavailable(self):
        claim = AtomicClaim(text="Claim.", section="S", has_citation=True)
        v = ClaimVerification(
            claim=claim,
            verdict="source_unavailable",
            confidence=0.0,
            evidence="URL 404.",
        )
        assert v.verdict == "source_unavailable"


# ── 3. CitationVerificationResult metric computation ─────────────────────────


class TestCitationVerificationResult:
    def test_basic_fields(self):
        result = CitationVerificationResult(
            report_id="r1",
            pattern="p0",
            total_claims=10,
            claims_with_citations=7,
            claims_without_citations=3,
            supported=5,
            not_supported=2,
            unverifiable=2,
            source_unavailable=1,
            citation_precision=5 / 7,
            citation_recall=7 / 10,
            attribution_accuracy=5 / 7,
            source_availability=9 / 10,
        )
        assert result.total_claims == 10
        assert result.supported == 5
        assert abs(result.citation_precision - 5 / 7) < 1e-6
        assert abs(result.citation_recall - 0.7) < 1e-6

    def test_defaults(self):
        result = CitationVerificationResult(
            report_id="r1",
            pattern="p0",
            total_claims=0,
            claims_with_citations=0,
            claims_without_citations=0,
            supported=0,
            not_supported=0,
            unverifiable=0,
            source_unavailable=0,
            citation_precision=0.0,
            citation_recall=0.0,
            attribution_accuracy=0.0,
            source_availability=0.0,
        )
        assert result.dois_found == []
        assert result.doi_recall == 0.0
        assert result.verifications == []


# ── 4. _match_claims_to_citations links correctly ───────────────────────────


class TestMatchClaimsToCitations:
    def test_basic_matching(self):
        verifier = _make_verifier()
        claims = [
            AtomicClaim(
                text="BERT uses bidirectional attention [1].",
                section="Arch",
                has_citation=True,
                citation_markers=["[1]"],
            ),
            AtomicClaim(
                text="GPT uses causal attention [2].",
                section="Arch",
                has_citation=True,
                citation_markers=["[2]"],
            ),
        ]
        citations = [
            Citation(
                source_url="https://arxiv.org/abs/1810.04805",
                source_title="BERT",
            ),
            Citation(
                source_url="https://arxiv.org/abs/2005.14165",
                source_title="GPT",
            ),
        ]
        verifier._match_claims_to_citations(claims, citations)
        assert claims[0].cited_urls == ["https://arxiv.org/abs/1810.04805"]
        assert claims[1].cited_urls == ["https://arxiv.org/abs/2005.14165"]

    def test_no_matching_citation(self):
        verifier = _make_verifier()
        claims = [
            AtomicClaim(
                text="Unknown fact [99].",
                section="S",
                has_citation=True,
                citation_markers=["[99]"],
            ),
        ]
        citations = [
            Citation(source_url="https://example.com", source_title="Only source"),
        ]
        verifier._match_claims_to_citations(claims, citations)
        # [99] does not match citation index 1, so no URLs
        assert claims[0].cited_urls == []

    def test_multiple_markers_same_claim(self):
        verifier = _make_verifier()
        claims = [
            AtomicClaim(
                text="Both models use transformers [1][2].",
                section="S",
                has_citation=True,
                citation_markers=["[1]", "[2]"],
            ),
        ]
        citations = [
            Citation(source_url="https://a.com", source_title="A"),
            Citation(source_url="https://b.com", source_title="B"),
        ]
        verifier._match_claims_to_citations(claims, citations)
        assert "https://a.com" in claims[0].cited_urls
        assert "https://b.com" in claims[0].cited_urls

    def test_empty_citations_list(self):
        verifier = _make_verifier()
        claims = [
            AtomicClaim(
                text="Fact [1].",
                section="S",
                has_citation=True,
                citation_markers=["[1]"],
            ),
        ]
        verifier._match_claims_to_citations(claims, [])
        assert claims[0].cited_urls == []

    def test_no_url_in_citation(self):
        verifier = _make_verifier()
        claims = [
            AtomicClaim(
                text="Fact [1].",
                section="S",
                has_citation=True,
                citation_markers=["[1]"],
            ),
        ]
        citations = [
            Citation(source_url="", source_title="No URL Source"),
        ]
        verifier._match_claims_to_citations(claims, citations)
        assert claims[0].cited_urls == []


# ── 5. _extract_dois_from_report regex works ─────────────────────────────────


class TestExtractDois:
    def test_single_doi(self):
        verifier = _make_verifier()
        text = "See DOI 10.1038/nature12373 for details."
        dois = verifier._extract_dois_from_report(text)
        assert len(dois) == 1
        assert dois[0] == "10.1038/nature12373"

    def test_multiple_dois(self):
        verifier = _make_verifier()
        text = (
            "10.1145/3292500.3330885 and also 10.48550/arXiv.2305.14314 "
            "are referenced here."
        )
        dois = verifier._extract_dois_from_report(text)
        assert len(dois) == 2

    def test_no_dois(self):
        verifier = _make_verifier()
        text = "This report has no DOIs, just regular references."
        dois = verifier._extract_dois_from_report(text)
        assert dois == []

    def test_duplicate_dois_deduplicated(self):
        verifier = _make_verifier()
        text = "10.1038/nature12373 appears twice: 10.1038/nature12373."
        dois = verifier._extract_dois_from_report(text)
        assert len(dois) == 1

    def test_doi_with_trailing_punctuation(self):
        verifier = _make_verifier()
        text = "Reference: 10.1038/nature12373."
        dois = verifier._extract_dois_from_report(text)
        assert dois[0] == "10.1038/nature12373"

    def test_doi_in_url(self):
        verifier = _make_verifier()
        text = "https://doi.org/10.1145/3292500.3330885"
        dois = verifier._extract_dois_from_report(text)
        assert len(dois) == 1
        assert "10.1145/3292500.3330885" in dois[0]


# ── 6. _compute_metrics with various verification outcomes ───────────────────


class TestComputeMetrics:
    def test_all_supported(self):
        verifier = _make_verifier()
        verifications = [
            ClaimVerification(
                claim=AtomicClaim(text="F1.", section="S", has_citation=True),
                verdict="supported",
                confidence=0.9,
                evidence="ok",
            ),
            ClaimVerification(
                claim=AtomicClaim(text="F2.", section="S", has_citation=True),
                verdict="supported",
                confidence=0.8,
                evidence="ok",
            ),
        ]
        metrics = verifier._compute_metrics(verifications, None)
        assert metrics["supported"] == 2
        assert metrics["not_supported"] == 0
        assert metrics["citation_precision"] == 1.0
        assert metrics["attribution_accuracy"] == 1.0

    def test_mixed_verdicts(self):
        verifier = _make_verifier()
        verifications = [
            ClaimVerification(
                claim=AtomicClaim(text="F1.", section="S", has_citation=True),
                verdict="supported",
                confidence=0.9,
                evidence="ok",
            ),
            ClaimVerification(
                claim=AtomicClaim(text="F2.", section="S", has_citation=True),
                verdict="not_supported",
                confidence=0.7,
                evidence="nope",
            ),
            ClaimVerification(
                claim=AtomicClaim(text="F3.", section="S", has_citation=False),
                verdict="unverifiable",
                confidence=0.0,
                evidence="",
            ),
            ClaimVerification(
                claim=AtomicClaim(text="F4.", section="S", has_citation=True),
                verdict="source_unavailable",
                confidence=0.0,
                evidence="404",
            ),
        ]
        metrics = verifier._compute_metrics(verifications, None)
        assert metrics["supported"] == 1
        assert metrics["not_supported"] == 1
        assert metrics["unverifiable"] == 1
        assert metrics["source_unavailable"] == 1
        # citation_precision = 1/(1+1) = 0.5
        assert abs(metrics["citation_precision"] - 0.5) < 1e-6
        # citation_recall = 3 cited / 4 total = 0.75
        assert abs(metrics["citation_recall"] - 0.75) < 1e-6
        # source_availability = 3/4 = 0.75
        assert abs(metrics["source_availability"] - 0.75) < 1e-6

    def test_empty_verifications(self):
        verifier = _make_verifier()
        metrics = verifier._compute_metrics([], None)
        assert metrics["supported"] == 0
        assert metrics["citation_precision"] == 0.0
        assert metrics["doi_recall"] == 0.0

    def test_doi_recall_computation(self):
        verifier = _make_verifier()
        verifications = [
            ClaimVerification(
                claim=AtomicClaim(
                    text="Fact.",
                    section="S",
                    has_citation=True,
                    cited_urls=["https://doi.org/10.1038/nature12373"],
                ),
                verdict="supported",
                confidence=0.9,
                evidence="ok",
                source_url="https://doi.org/10.1038/nature12373",
            ),
        ]
        reference_dois = ["10.1038/nature12373", "10.1234/missing"]
        metrics = verifier._compute_metrics(verifications, reference_dois)
        # Found 1 of 2 reference DOIs
        assert abs(metrics["doi_recall"] - 0.5) < 1e-6

    def test_doi_recall_no_references(self):
        verifier = _make_verifier()
        verifications = [
            ClaimVerification(
                claim=AtomicClaim(text="F.", section="S", has_citation=True),
                verdict="supported",
                confidence=0.9,
                evidence="ok",
            ),
        ]
        metrics = verifier._compute_metrics(verifications, None)
        assert metrics["doi_recall"] == 0.0

    def test_doi_recall_empty_references(self):
        verifier = _make_verifier()
        verifications = [
            ClaimVerification(
                claim=AtomicClaim(text="F.", section="S", has_citation=True),
                verdict="supported",
                confidence=0.9,
                evidence="ok",
            ),
        ]
        metrics = verifier._compute_metrics(verifications, [])
        assert metrics["doi_recall"] == 0.0

    def test_attribution_accuracy_no_cited_claims(self):
        verifier = _make_verifier()
        verifications = [
            ClaimVerification(
                claim=AtomicClaim(text="F.", section="S", has_citation=False),
                verdict="unverifiable",
                confidence=0.0,
                evidence="",
            ),
        ]
        metrics = verifier._compute_metrics(verifications, None)
        # No cited claims => attribution_accuracy = 0.0
        assert metrics["attribution_accuracy"] == 0.0


# ── 7. verify_report integration with mocked LLM and URL extractor ──────────


class TestVerifyReportIntegration:
    @pytest.mark.asyncio
    async def test_basic_verification_flow(self):
        """Test full flow with mocked LLM returning claims and NLI results."""
        llm = _make_mock_llm()
        url_ext = _make_mock_url_extractor()

        # LLM returns claims on first call (extract_atomic_claims)
        llm.complete_json = AsyncMock(
            side_effect=[
                # First call: claim extraction
                {
                    "claims": [
                        {
                            "text": "BERT uses bidirectional attention [1].",
                            "section": "Architecture",
                            "citation_markers": ["[1]"],
                        },
                        {
                            "text": "GPT has 175 billion parameters [2].",
                            "section": "Scale",
                            "citation_markers": ["[2]"],
                        },
                    ]
                },
                # Second call: NLI check for claim 1
                {"verdict": "supported", "confidence": 0.95, "reasoning": "Confirmed"},
                # Third call: NLI check for claim 2
                {"verdict": "not_supported", "confidence": 0.8, "reasoning": "Wrong"},
            ]
        )

        # URL extractor returns content
        mock_doc = MagicMock()
        mock_doc.content = "BERT is a bidirectional encoder model using self-attention."
        url_ext.extract = AsyncMock(return_value=mock_doc)

        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            max_claims=50,
            max_concurrent=5,
        )

        citations = [
            Citation(
                source_url="https://arxiv.org/abs/1810.04805",
                source_title="BERT Paper",
            ),
            Citation(
                source_url="https://arxiv.org/abs/2005.14165",
                source_title="GPT-3 Paper",
            ),
        ]

        result = await verifier.verify_report(
            report_text="# Report\n## Architecture\nBERT uses bidirectional attention [1].\n## Scale\nGPT has 175 billion parameters [2].",
            report_id="test_r1",
            pattern="p0",
            citations=citations,
        )

        assert result.total_claims == 2
        assert result.claims_with_citations == 2
        assert result.claims_without_citations == 0
        assert result.supported == 1
        assert result.not_supported == 1
        assert len(result.verifications) == 2

    @pytest.mark.asyncio
    async def test_verify_report_empty_text(self):
        """Empty report should return zero-result."""
        verifier = _make_verifier()
        result = await verifier.verify_report(
            report_text="",
            report_id="empty",
            pattern="p0",
        )
        assert result.total_claims == 0
        assert result.citation_precision == 0.0

    @pytest.mark.asyncio
    async def test_verify_report_whitespace_only(self):
        """Whitespace-only report should return zero-result."""
        verifier = _make_verifier()
        result = await verifier.verify_report(
            report_text="   \n\n  ",
            report_id="ws",
            pattern="p0",
        )
        assert result.total_claims == 0


# ── 8. Edge case: report with no citations ──────────────────────────────────


class TestNoCitationsEdge:
    @pytest.mark.asyncio
    async def test_no_citations_uncited_claims(self):
        """When no citations list is provided, claims should be uncited."""
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(
            return_value={
                "claims": [
                    {
                        "text": "The sky is blue.",
                        "section": "Facts",
                        "citation_markers": [],
                    }
                ]
            }
        )
        url_ext = _make_mock_url_extractor()
        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            web_searcher=None,
            max_claims=50,
            max_concurrent=5,
        )

        result = await verifier.verify_report(
            report_text="The sky is blue.",
            report_id="no_cit",
            pattern="p0",
            citations=[],
        )
        assert result.total_claims == 1
        assert result.claims_with_citations == 0
        assert result.claims_without_citations == 1
        # Uncited + no web searcher => unverifiable
        assert result.unverifiable == 1


# ── 9. Edge case: report with no verifiable claims ──────────────────────────


class TestNoVerifiableClaims:
    @pytest.mark.asyncio
    async def test_no_claims_extracted(self):
        """When the LLM extracts no claims."""
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={"claims": []})
        url_ext = _make_mock_url_extractor()
        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            max_claims=50,
            max_concurrent=5,
        )

        result = await verifier.verify_report(
            report_text="This report is purely subjective opinion.",
            report_id="no_claims",
            pattern="p0",
        )
        assert result.total_claims == 0
        assert result.verifications == []

    @pytest.mark.asyncio
    async def test_llm_extraction_fails(self):
        """When the LLM raises an exception during claim extraction."""
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(side_effect=RuntimeError("LLM down"))
        url_ext = _make_mock_url_extractor()
        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            max_claims=50,
            max_concurrent=5,
        )

        result = await verifier.verify_report(
            report_text="Some report text.",
            report_id="error",
            pattern="p0",
        )
        assert result.total_claims == 0

    @pytest.mark.asyncio
    async def test_llm_returns_bad_format(self):
        """When the LLM returns non-list claims."""
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={"claims": "not a list"})
        url_ext = _make_mock_url_extractor()
        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            max_claims=50,
            max_concurrent=5,
        )

        result = await verifier.verify_report(
            report_text="Some text.",
            report_id="bad_fmt",
            pattern="p0",
        )
        assert result.total_claims == 0


# ── 10. DOI recall computation ──────────────────────────────────────────────


class TestDOIRecall:
    @pytest.mark.asyncio
    async def test_doi_recall_full_match(self):
        """All reference DOIs found in report."""
        llm = _make_mock_llm()
        # Return claims with DOI-based URLs
        llm.complete_json = AsyncMock(
            side_effect=[
                # Claim extraction
                {
                    "claims": [
                        {
                            "text": "Finding from paper [1].",
                            "section": "S",
                            "citation_markers": ["[1]"],
                        }
                    ]
                },
                # NLI check
                {"verdict": "supported", "confidence": 0.9, "reasoning": "ok"},
            ]
        )

        url_ext = _make_mock_url_extractor()
        mock_doc = MagicMock()
        mock_doc.content = "Source content about the finding."
        url_ext.extract = AsyncMock(return_value=mock_doc)

        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            max_claims=50,
            max_concurrent=5,
        )

        citations = [
            Citation(
                source_url="https://doi.org/10.1038/nature12373",
                source_title="Paper",
            ),
        ]

        result = await verifier.verify_report(
            report_text="Finding from paper [1]. DOI: 10.1038/nature12373",
            report_id="doi_test",
            pattern="p0",
            citations=citations,
            reference_dois=["10.1038/nature12373"],
        )

        assert result.doi_recall == 1.0
        assert "10.1038/nature12373" in result.dois_found

    @pytest.mark.asyncio
    async def test_doi_recall_partial_match(self):
        """Some but not all reference DOIs found."""
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(
            side_effect=[
                {"claims": [{"text": "Fact [1].", "section": "S", "citation_markers": ["[1]"]}]},
                {"verdict": "supported", "confidence": 0.9, "reasoning": "ok"},
            ]
        )

        url_ext = _make_mock_url_extractor()
        mock_doc = MagicMock()
        mock_doc.content = "Content."
        url_ext.extract = AsyncMock(return_value=mock_doc)

        verifier = AgenticCitationVerifier(
            llm_caller=llm,
            url_extractor=url_ext,
            max_claims=50,
            max_concurrent=5,
        )

        citations = [
            Citation(source_url="https://doi.org/10.1038/nature12373", source_title="P"),
        ]

        result = await verifier.verify_report(
            report_text="Fact [1].",
            report_id="doi_partial",
            pattern="p0",
            citations=citations,
            reference_dois=["10.1038/nature12373", "10.9999/missing"],
        )

        # One DOI found in source_url, one missing
        assert abs(result.doi_recall - 0.5) < 1e-6


# ── Regex pattern sanity checks ──────────────────────────────────────────────


class TestRegexPatterns:
    def test_citation_marker_pattern(self):
        text = "See [1] and [23] but not [abc]."
        matches = _CITATION_MARKER_PATTERN.findall(text)
        assert "1" in matches
        assert "23" in matches
        assert len(matches) == 2

    def test_doi_pattern_standard(self):
        text = "10.1038/nature12373"
        matches = _DOI_PATTERN.findall(text)
        assert len(matches) == 1

    def test_doi_pattern_arxiv(self):
        text = "10.48550/arXiv.2305.14314"
        matches = _DOI_PATTERN.findall(text)
        assert len(matches) == 1
