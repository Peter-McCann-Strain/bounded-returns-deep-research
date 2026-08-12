"""Tests for shared Pydantic types."""

import pytest
from deep_research.types import (
    Document,
    Citation,
    SubQuery,
    Section,
    ResearchReport,
    LLMUsage,
    SourceType,
    Perspective,
    TopicCluster,
    WDAllocation,
)


class TestDocument:
    def test_create_default(self):
        doc = Document()
        assert doc.id == ""
        assert doc.source_type == SourceType.WEB

    def test_create_with_fields(self):
        doc = Document(
            id="abc123",
            title="Test",
            content="Hello world",
            url="https://example.com",
            source_type=SourceType.ACADEMIC,
        )
        assert doc.id == "abc123"
        assert doc.source_type == SourceType.ACADEMIC

    def test_short_id(self):
        doc = Document(id="abcdefghijklmnop")
        assert doc.short_id() == "abcdefghijkl"

    def test_short_id_fallback_url(self):
        doc = Document(url="https://example.com/very/long/path")
        assert len(doc.short_id()) <= 40

    def test_metadata_default(self):
        doc = Document()
        assert doc.metadata == {}


class TestCitation:
    def test_create(self):
        c = Citation(claim="test", source_url="https://x.com")
        assert c.relevance_score == 0.0

    def test_defaults(self):
        c = Citation()
        assert c.claim == ""
        assert c.source_id == ""


class TestSubQuery:
    def test_create(self):
        sq = SubQuery(query="test query")
        assert sq.priority == 1
        assert sq.intent == ""


class TestSection:
    def test_create(self):
        s = Section(title="Intro", content="Some content")
        assert s.citations == []


class TestResearchReport:
    def test_create_minimal(self):
        r = ResearchReport(query="test")
        assert r.title == ""
        assert r.sections == []
        assert r.total_cost_usd == 0.0

    def test_full_text_with_sections(self, sample_report):
        text = sample_report.full_text()
        assert "BERT vs GPT" in text
        assert "## Architecture" in text
        assert "## Training Objectives" in text

    def test_full_text_with_references(self, sample_report):
        text = sample_report.full_text()
        assert "References" in text
        assert "BERT Paper" in text

    def test_full_text_with_abstract(self, sample_report):
        text = sample_report.full_text()
        assert "Abstract" in text


class TestLLMUsage:
    def test_create(self):
        u = LLMUsage(model="gpt-4o")
        assert u.input_tokens == 0
        assert u.cost_usd == 0.0


class TestSourceType:
    def test_values(self):
        assert SourceType.WEB == "web"
        assert SourceType.ACADEMIC == "academic"
        assert SourceType.ARXIV == "arxiv"


class TestPerspective:
    def test_create(self):
        p = Perspective(name="NLP Expert", description="Focuses on NLP")
        assert p.focus_areas == []


class TestTopicCluster:
    def test_create(self):
        tc = TopicCluster(topic="Attention", summary="About attention")
        assert tc.importance == 0.0
        assert tc.source_ids == []


class TestWDAllocation:
    def test_create(self):
        a = WDAllocation(step=0, width_budget=0.6, depth_budget=0.4,
                         width_workers=4, depth_iterations=1)
        assert a.width_workers == 4
