"""Tests for source extractor — two-step extraction, JSON parsing, formatting."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.tools.source_extractor import (
    SourceExtraction,
    SourceExtractor,
    ExtractedSourceType,
    _parse_extraction_json,
    format_extractions_as_evidence,
    format_summaries_as_evidence,
)
from deep_research.types import Document, SourceType


class TestSourceExtraction:
    def test_create_default(self):
        se = SourceExtraction()
        assert se.relevance_score == 0
        assert se.source_type == ExtractedSourceType.OTHER

    def test_to_evidence_dict_core_fields(self):
        se = SourceExtraction(
            doc_id="doc1",
            title="Test",
            url="https://example.com",
            summary="A summary",
            relevance_score=7,
            source_type=ExtractedSourceType.RESEARCH_PAPER,
            key_findings=["finding1"],
            confidence_notes="reliable",
        )
        d = se.to_evidence_dict()
        assert d["doc_id"] == "doc1"
        assert d["relevance_score"] == 7
        assert d["source_type"] == "research_paper"
        assert "methodology" not in d  # Optional, not set

    def test_to_evidence_dict_optional_fields(self):
        se = SourceExtraction(
            methodology="survey",
            data_points=["90% accuracy"],
            limitations="small sample",
            competing_perspectives=["view A"],
            practical_implications="use X",
            temporal_context="2024",
        )
        d = se.to_evidence_dict()
        assert d["methodology"] == "survey"
        assert d["data_points"] == ["90% accuracy"]
        assert d["limitations"] == "small sample"

    def test_relevance_score_bounds(self):
        se = SourceExtraction(relevance_score=10)
        assert se.relevance_score == 10

        se = SourceExtraction(relevance_score=0)
        assert se.relevance_score == 0


class TestParseExtractionJson:
    def test_valid_json(self):
        data = {
            "summary": "Test summary",
            "relevance_score": 8,
            "source_type": "research_paper",
            "key_findings": ["finding1"],
            "confidence_notes": "good",
        }
        result = _parse_extraction_json(json.dumps(data))
        assert result is not None
        assert result.relevance_score == 8
        assert result.source_type == ExtractedSourceType.RESEARCH_PAPER

    def test_markdown_code_fences(self):
        data = {
            "summary": "Test",
            "relevance_score": 5,
            "source_type": "blog_post",
            "key_findings": [],
            "confidence_notes": "",
        }
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_extraction_json(raw)
        assert result is not None
        assert result.source_type == ExtractedSourceType.BLOG_POST

    def test_json_embedded_in_text(self):
        data = {"summary": "Test", "relevance_score": 3, "source_type": "other",
                "key_findings": [], "confidence_notes": ""}
        raw = f"Here is the result:\n{json.dumps(data)}\nDone."
        result = _parse_extraction_json(raw)
        assert result is not None
        assert result.relevance_score == 3

    def test_invalid_source_type_falls_back(self):
        data = {"summary": "Test", "relevance_score": 5, "source_type": "invalid_type",
                "key_findings": [], "confidence_notes": ""}
        result = _parse_extraction_json(json.dumps(data))
        assert result is not None
        assert result.source_type == ExtractedSourceType.OTHER

    def test_relevance_score_clamped(self):
        data = {"summary": "Test", "relevance_score": 99, "source_type": "other",
                "key_findings": [], "confidence_notes": ""}
        result = _parse_extraction_json(json.dumps(data))
        assert result is not None
        assert result.relevance_score == 10

    def test_negative_relevance_clamped(self):
        data = {"summary": "Test", "relevance_score": -5, "source_type": "other",
                "key_findings": [], "confidence_notes": ""}
        result = _parse_extraction_json(json.dumps(data))
        assert result is not None
        assert result.relevance_score == 0

    def test_completely_invalid_json(self):
        result = _parse_extraction_json("not json at all")
        assert result is None

    def test_optional_fields_included(self):
        data = {
            "summary": "Test",
            "relevance_score": 7,
            "source_type": "research_paper",
            "key_findings": [],
            "confidence_notes": "",
            "methodology": "RCT",
            "data_points": ["95% CI"],
            "limitations": "Small n",
        }
        result = _parse_extraction_json(json.dumps(data))
        assert result.methodology == "RCT"
        assert result.data_points == ["95% CI"]


class TestSourceExtractorExtractOne:
    @pytest.mark.asyncio
    async def test_extract_one_success(self, mock_llm):
        """Test successful two-step extraction."""
        json_response = json.dumps({
            "summary": "BERT analysis",
            "relevance_score": 8,
            "source_type": "research_paper",
            "key_findings": ["bidirectional"],
            "confidence_notes": "reliable",
        })
        mock_llm.complete = AsyncMock(side_effect=[
            "Free-text analysis of BERT...",  # Step 1
            json_response,                     # Step 2
        ])

        extractor = SourceExtractor(llm=mock_llm)
        doc = Document(
            id="d1", title="BERT", url="https://example.com",
            content="BERT paper content here" * 10,
            source_type=SourceType.ACADEMIC,
        )

        result = await extractor.extract_one(doc, "Compare BERT and GPT")
        assert result is not None
        assert result.relevance_score == 8
        assert result.doc_id == "d1"
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_one_not_relevant(self, mock_llm):
        mock_llm.complete = AsyncMock(return_value="NOT RELEVANT")
        extractor = SourceExtractor(llm=mock_llm)
        doc = Document(id="d1", title="Cooking", content="Recipe for cake" * 10,
                       url="https://cook.com", source_type=SourceType.WEB)
        result = await extractor.extract_one(doc, "Compare BERT and GPT")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_one_empty_content(self, mock_llm):
        extractor = SourceExtractor(llm=mock_llm)
        doc = Document(id="d1", content="", url="https://empty.com",
                       source_type=SourceType.WEB)
        result = await extractor.extract_one(doc, "query")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_one_json_fallback(self, mock_llm):
        """When JSON parsing fails, should use free-text as summary."""
        mock_llm.complete = AsyncMock(side_effect=[
            "Good analysis text here",
            "not valid json {{{{",
        ])
        extractor = SourceExtractor(llm=mock_llm)
        doc = Document(id="d1", title="Test", content="Content" * 20,
                       url="https://test.com", source_type=SourceType.WEB)
        result = await extractor.extract_one(doc, "query")
        assert result is not None
        assert "Good analysis text here" in result.summary
        assert result.relevance_score == 5  # fallback default


class TestSourceExtractorExtractBatch:
    @pytest.mark.asyncio
    async def test_extract_batch(self, mock_llm, sample_documents):
        json_resp = json.dumps({
            "summary": "test", "relevance_score": 7,
            "source_type": "other", "key_findings": [],
            "confidence_notes": "",
        })
        mock_llm.complete = AsyncMock(side_effect=[
            "Analysis 1", json_resp,
            "Analysis 2", json_resp,
            "Analysis 3", json_resp,
        ])
        extractor = SourceExtractor(llm=mock_llm)
        results = await extractor.extract_batch(sample_documents, "test query")
        assert len(results) == 3
        # Should be sorted by relevance (all same score here)
        assert all(r.relevance_score == 7 for r in results)

    @pytest.mark.asyncio
    async def test_extract_batch_with_min_relevance(self, mock_llm, sample_documents):
        responses = []
        for score in [9, 3, 7]:
            responses.append("Analysis")
            responses.append(json.dumps({
                "summary": "test", "relevance_score": score,
                "source_type": "other", "key_findings": [],
                "confidence_notes": "",
            }))
        mock_llm.complete = AsyncMock(side_effect=responses)
        extractor = SourceExtractor(llm=mock_llm)
        results = await extractor.extract_batch(
            sample_documents, "query", min_relevance=5
        )
        assert len(results) == 2  # Only score 9 and 7
        assert results[0].relevance_score == 9  # Sorted descending


class TestFormatExtractionsAsEvidence:
    def test_basic_formatting(self, sample_extractions):
        text = format_extractions_as_evidence(sample_extractions)
        assert "[1]" in text
        assert "[2]" in text
        assert "BERT Paper" in text
        assert "Relevance: 9/10" in text

    def test_includes_key_findings(self, sample_extractions):
        text = format_extractions_as_evidence(sample_extractions)
        assert "bidirectional attention" in text

    def test_max_chars_truncation(self, sample_extractions):
        # With very large budget, all are included
        full_text = format_extractions_as_evidence(sample_extractions, max_chars=100_000)
        assert "[1]" in full_text
        assert "[2]" in full_text

        # With tight budget, fewer extractions included
        short_text = format_extractions_as_evidence(sample_extractions, max_chars=500)
        # First extraction should fit within 500 chars
        assert "[1]" in short_text

    def test_empty_list(self):
        text = format_extractions_as_evidence([])
        assert text == ""


class TestFormatSummariesAsEvidence:
    def test_legacy_format(self):
        summaries = [
            {"title": "Test Doc", "url": "https://test.com", "summary": "Test summary"},
        ]
        text = format_summaries_as_evidence(summaries)
        assert "Test Doc" in text
        assert "Test summary" in text
