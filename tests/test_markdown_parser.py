"""Comprehensive tests for the shared markdown parser utility.

Tests cover:
    - Title extraction (present, absent, multiple headings)
    - Abstract extraction (present, absent, case variations)
    - Section extraction (normal, with subsections, special chars, skip logic)
    - Citation building from SourceExtraction objects
    - Full integration: markdown + extractions -> ResearchReport
    - Edge cases: empty input, title-only, no sections, etc.
"""

from __future__ import annotations

import pytest

from deep_research.tools.source_extractor import ExtractedSourceType, SourceExtraction
from deep_research.types import Citation, ResearchReport, Section
from deep_research.utils.markdown_parser import (
    _extract_abstract,
    _extract_sections,
    _extract_title,
    build_citations_from_extractions,
    parse_markdown_report,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def well_formed_markdown() -> str:
    return (
        "# Advances in Transformer Architectures\n\n"
        "## Abstract\n"
        "This report surveys recent advances in transformer architectures, "
        "covering attention mechanisms, efficiency improvements, and "
        "emerging applications.\n\n"
        "## Introduction\n"
        "Transformers were introduced in 2017 [1] and have since "
        "revolutionised NLP and beyond.\n\n"
        "## Attention Mechanisms\n"
        "Multi-head attention allows the model to attend to different "
        "representation subspaces [2]. Recent work explores sparse "
        "attention variants.\n\n"
        "## Efficiency Improvements\n"
        "Several approaches reduce the quadratic cost of self-attention, "
        "including linear attention, kernelised methods, and local-global "
        "hybrid schemes [3].\n\n"
        "## References\n"
        "[1] Vaswani et al., 2017\n"
        "[2] Devlin et al., 2018\n"
        "[3] Kitaev et al., 2020\n"
    )


@pytest.fixture
def no_abstract_markdown() -> str:
    return (
        "# Survey of LLM Safety\n\n"
        "## Current Landscape\n"
        "Large language models present novel safety challenges.\n\n"
        "## Mitigation Strategies\n"
        "RLHF and constitutional AI are leading approaches [1].\n"
    )


@pytest.fixture
def no_title_markdown() -> str:
    return (
        "This report has no heading-1 title.\n\n"
        "## Abstract\n"
        "An abstract without a proper title.\n\n"
        "## Findings\n"
        "Some findings go here.\n"
    )


@pytest.fixture
def empty_markdown() -> str:
    return ""


@pytest.fixture
def single_section_markdown() -> str:
    return (
        "# Report Title\n\n"
        "## The Only Section\n"
        "This is the only body section in the report.\n"
    )


@pytest.fixture
def subsections_markdown() -> str:
    return (
        "# Deep Dive Report\n\n"
        "## Abstract\n"
        "Abstract text.\n\n"
        "## Methodology\n"
        "We used the following approach.\n\n"
        "### Data Collection\n"
        "We collected data from 10 sources.\n\n"
        "### Analysis Pipeline\n"
        "Data was processed through three stages.\n\n"
        "## Results\n"
        "Results are presented below.\n"
    )


@pytest.fixture
def special_chars_markdown() -> str:
    return (
        "# Report: Pros & Cons of C++ vs. Rust (2024)\n\n"
        "## Abstract\n"
        "An abstract.\n\n"
        '## Performance & Memory Safety\n'
        "C++ offers raw speed; Rust guarantees memory safety at compile time.\n\n"
        '## Cost-Benefit Analysis: $$$\n'
        "Evaluating trade-offs.\n"
    )


@pytest.fixture
def citation_markers_markdown() -> str:
    return (
        "# AI in Healthcare\n\n"
        "## Abstract\n"
        "AI transforms healthcare [1].\n\n"
        "## Diagnostics\n"
        "Deep learning improves radiology [1], pathology [2], and "
        "ophthalmology [3]. Combined approaches [1][2] yield best results.\n\n"
        "## Treatment Planning\n"
        "ML models assist in drug discovery [4] and clinical trials [5].\n"
    )


@pytest.fixture
def title_only_markdown() -> str:
    return "# My Research Title\n"


@pytest.fixture
def references_and_sources_markdown() -> str:
    """Markdown with both References and Sources sections to be skipped."""
    return (
        "# Report\n\n"
        "## Abstract\n"
        "An abstract.\n\n"
        "## Body Section\n"
        "Content here.\n\n"
        "## References\n"
        "[1] Source A\n\n"
        "## Sources\n"
        "Source B\n"
    )


@pytest.fixture
def sample_extractions() -> list[SourceExtraction]:
    return [
        SourceExtraction(
            doc_id="doc1",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            summary="Introduces the transformer architecture.",
            relevance_score=9,
            source_type=ExtractedSourceType.RESEARCH_PAPER,
            key_findings=["Self-attention replaces recurrence"],
            confidence_notes="Seminal paper.",
        ),
        SourceExtraction(
            doc_id="doc2",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            url="https://arxiv.org/abs/1810.04805",
            summary="BERT uses masked language modeling.",
            relevance_score=8,
            source_type=ExtractedSourceType.RESEARCH_PAPER,
            key_findings=["Bidirectional pre-training"],
            confidence_notes="Highly cited.",
        ),
        SourceExtraction(
            doc_id="doc3",
            title="Efficient Transformers: A Survey",
            url="https://arxiv.org/abs/2009.06732",
            summary="Surveys efficient transformer variants.",
            relevance_score=7,
            source_type=ExtractedSourceType.RESEARCH_PAPER,
            key_findings=["Linear attention", "Sparse attention"],
            confidence_notes="Comprehensive survey.",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _extract_title
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractTitle:
    def test_extracts_simple_title(self, well_formed_markdown: str) -> None:
        assert _extract_title(well_formed_markdown) == "Advances in Transformer Architectures"

    def test_returns_none_when_no_title(self, no_title_markdown: str) -> None:
        assert _extract_title(no_title_markdown) is None

    def test_returns_none_for_empty_markdown(self, empty_markdown: str) -> None:
        assert _extract_title(empty_markdown) is None

    def test_title_with_special_characters(self, special_chars_markdown: str) -> None:
        assert _extract_title(special_chars_markdown) == "Report: Pros & Cons of C++ vs. Rust (2024)"

    def test_title_only_document(self, title_only_markdown: str) -> None:
        assert _extract_title(title_only_markdown) == "My Research Title"

    def test_strips_whitespace(self) -> None:
        md = "#   Padded Title   \n\n## Section\nContent."
        assert _extract_title(md) == "Padded Title"

    def test_ignores_h2_headings(self) -> None:
        md = "## Not a title\nContent.\n\n## Another\nMore."
        assert _extract_title(md) is None

    def test_takes_first_h1_when_multiple(self) -> None:
        md = "# First Title\n\nSome text.\n\n# Second Title\n\nMore text."
        assert _extract_title(md) == "First Title"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _extract_abstract
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractAbstract:
    def test_extracts_abstract(self, well_formed_markdown: str) -> None:
        abstract = _extract_abstract(well_formed_markdown)
        assert abstract.startswith("This report surveys")
        assert "emerging applications" in abstract

    def test_returns_empty_when_no_abstract(self, no_abstract_markdown: str) -> None:
        assert _extract_abstract(no_abstract_markdown) == ""

    def test_returns_empty_for_empty_markdown(self, empty_markdown: str) -> None:
        assert _extract_abstract(empty_markdown) == ""

    def test_case_insensitive(self) -> None:
        md = "# T\n\n## ABSTRACT\nAbstract text.\n\n## Body\nContent."
        assert _extract_abstract(md) == "Abstract text."

    def test_abstract_before_end_of_file(self) -> None:
        """Abstract is the last section."""
        md = "# T\n\n## Abstract\nFinal abstract content."
        assert _extract_abstract(md) == "Final abstract content."

    def test_strips_whitespace(self) -> None:
        md = "# T\n\n## Abstract\n  \n  Padded abstract.  \n  \n\n## S\nC."
        assert _extract_abstract(md) == "Padded abstract."

    def test_multiline_abstract(self) -> None:
        md = (
            "# T\n\n"
            "## Abstract\n"
            "First paragraph of abstract.\n\n"
            "Second paragraph of abstract.\n\n"
            "## Body\nContent."
        )
        abstract = _extract_abstract(md)
        assert "First paragraph" in abstract
        assert "Second paragraph" in abstract


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _extract_sections
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractSections:
    def test_extracts_body_sections(self, well_formed_markdown: str) -> None:
        sections = _extract_sections(well_formed_markdown)
        titles = [t for t, _ in sections]
        assert "Introduction" in titles
        assert "Attention Mechanisms" in titles
        assert "Efficiency Improvements" in titles

    def test_skips_abstract_by_default(self, well_formed_markdown: str) -> None:
        sections = _extract_sections(well_formed_markdown)
        titles = [t for t, _ in sections]
        assert "Abstract" not in titles

    def test_skips_references_by_default(self, well_formed_markdown: str) -> None:
        sections = _extract_sections(well_formed_markdown)
        titles = [t for t, _ in sections]
        assert "References" not in titles

    def test_skips_sources_by_default(self, references_and_sources_markdown: str) -> None:
        sections = _extract_sections(references_and_sources_markdown)
        titles = [t for t, _ in sections]
        assert "Sources" not in titles
        assert "Body Section" in titles

    def test_empty_markdown_returns_empty(self, empty_markdown: str) -> None:
        assert _extract_sections(empty_markdown) == []

    def test_single_section(self, single_section_markdown: str) -> None:
        sections = _extract_sections(single_section_markdown)
        assert len(sections) == 1
        assert sections[0][0] == "The Only Section"
        assert "only body section" in sections[0][1]

    def test_sections_preserve_subsections(self, subsections_markdown: str) -> None:
        sections = _extract_sections(subsections_markdown)
        methodology = [s for s in sections if s[0] == "Methodology"]
        assert len(methodology) == 1
        content = methodology[0][1]
        assert "### Data Collection" in content
        assert "### Analysis Pipeline" in content

    def test_special_characters_in_titles(self, special_chars_markdown: str) -> None:
        sections = _extract_sections(special_chars_markdown)
        titles = [t for t, _ in sections]
        assert "Performance & Memory Safety" in titles
        assert "Cost-Benefit Analysis: $$$" in titles

    def test_custom_skip_set(self, well_formed_markdown: str) -> None:
        """Only skip 'abstract', keep references."""
        sections = _extract_sections(well_formed_markdown, skip={"abstract"})
        titles = [t for t, _ in sections]
        assert "Abstract" not in titles
        assert "References" in titles

    def test_empty_skip_set(self, well_formed_markdown: str) -> None:
        """Skip nothing -- all sections returned."""
        sections = _extract_sections(well_formed_markdown, skip=set())
        titles = [t for t, _ in sections]
        assert "Abstract" in titles
        assert "References" in titles


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: build_citations_from_extractions
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildCitations:
    def test_builds_citations(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions)
        assert len(citations) == 3

    def test_citation_numbering(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions)
        assert citations[0].claim == "[1]"
        assert citations[1].claim == "[2]"
        assert citations[2].claim == "[3]"

    def test_citation_source_id(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions)
        assert citations[0].source_id == "doc1"
        assert citations[1].source_id == "doc2"
        assert citations[2].source_id == "doc3"

    def test_citation_source_title(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions)
        assert citations[0].source_title == "Attention Is All You Need"

    def test_citation_source_url(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions)
        assert citations[0].source_url == "https://arxiv.org/abs/1706.03762"

    def test_relevance_score_normalised(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions)
        assert citations[0].relevance_score == pytest.approx(0.9)
        assert citations[1].relevance_score == pytest.approx(0.8)
        assert citations[2].relevance_score == pytest.approx(0.7)

    def test_empty_extractions(self) -> None:
        citations = build_citations_from_extractions([])
        assert citations == []

    def test_max_citations_cap(self, sample_extractions: list[SourceExtraction]) -> None:
        citations = build_citations_from_extractions(sample_extractions, max_citations=2)
        assert len(citations) == 2
        assert citations[0].claim == "[1]"
        assert citations[1].claim == "[2]"

    def test_max_citations_none_means_no_limit(
        self, sample_extractions: list[SourceExtraction]
    ) -> None:
        citations = build_citations_from_extractions(sample_extractions, max_citations=None)
        assert len(citations) == 3

    def test_max_citations_greater_than_available(
        self, sample_extractions: list[SourceExtraction]
    ) -> None:
        citations = build_citations_from_extractions(sample_extractions, max_citations=100)
        assert len(citations) == 3

    def test_zero_relevance_score(self) -> None:
        ext = SourceExtraction(
            doc_id="x", title="X", url="http://x.com",
            summary="X", relevance_score=0,
        )
        citations = build_citations_from_extractions([ext])
        assert citations[0].relevance_score == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: parse_markdown_report (full integration)
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseMarkdownReport:
    def test_well_formed_report(
        self,
        well_formed_markdown: str,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        report = parse_markdown_report(
            query="transformer survey",
            markdown=well_formed_markdown,
            extractions=sample_extractions,
            pattern_name="p0_baseline",
            cost_usd=1.50,
            total_tokens=5000,
        )
        assert isinstance(report, ResearchReport)
        assert report.title == "Advances in Transformer Architectures"
        assert report.abstract.startswith("This report surveys")
        assert len(report.sections) == 3  # Introduction, Attention, Efficiency
        assert len(report.citations) == 3
        assert report.pattern_name == "p0_baseline"
        assert report.total_cost_usd == 1.50
        assert report.total_tokens == 5000

    def test_no_abstract(
        self,
        no_abstract_markdown: str,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        report = parse_markdown_report(
            query="llm safety",
            markdown=no_abstract_markdown,
            extractions=sample_extractions,
            pattern_name="p1_iterative_rag",
            cost_usd=0.5,
            total_tokens=2000,
        )
        assert report.abstract == ""
        assert report.title == "Survey of LLM Safety"
        assert len(report.sections) == 2

    def test_no_title_falls_back_to_query(
        self,
        no_title_markdown: str,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        report = parse_markdown_report(
            query="my custom query",
            markdown=no_title_markdown,
            extractions=sample_extractions,
            pattern_name="p2_supervisor_parallel",
            cost_usd=0.0,
            total_tokens=0,
        )
        assert report.title == "my custom query"

    def test_empty_markdown(self) -> None:
        report = parse_markdown_report(
            query="empty query",
            markdown="",
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        assert report.title == "empty query"
        assert report.abstract == ""
        assert report.sections == []
        assert report.citations == []

    def test_single_section(
        self,
        single_section_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=single_section_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        assert report.title == "Report Title"
        assert len(report.sections) == 1
        assert report.sections[0].title == "The Only Section"

    def test_sections_with_subsections(
        self,
        subsections_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=subsections_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        # Methodology section should contain ### subsections in its content
        methodology = [s for s in report.sections if s.title == "Methodology"]
        assert len(methodology) == 1
        assert "### Data Collection" in methodology[0].content

    def test_inline_citation_markers_preserved(
        self,
        citation_markers_markdown: str,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=citation_markers_markdown,
            extractions=sample_extractions,
            pattern_name="p4_perspective_storm",
            cost_usd=0.0,
            total_tokens=0,
        )
        # Inline markers [1], [2] etc. should be preserved in section content
        diagnostics = [s for s in report.sections if s.title == "Diagnostics"]
        assert len(diagnostics) == 1
        assert "[1]" in diagnostics[0].content
        assert "[2]" in diagnostics[0].content

    def test_all_report_fields_populated(
        self,
        well_formed_markdown: str,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        metadata = {"custom_key": "custom_value", "n_sources": 3}
        report = parse_markdown_report(
            query="transformer survey",
            markdown=well_formed_markdown,
            extractions=sample_extractions,
            pattern_name="p4_perspective_storm",
            cost_usd=2.75,
            total_tokens=12000,
            elapsed_seconds=45.3,
            metadata=metadata,
        )
        assert report.query == "transformer survey"
        assert report.title == "Advances in Transformer Architectures"
        assert report.abstract != ""
        assert len(report.sections) > 0
        assert len(report.citations) == 3
        assert report.metadata == metadata
        assert report.pattern_name == "p4_perspective_storm"
        assert report.total_cost_usd == 2.75
        assert report.total_tokens == 12000
        assert report.elapsed_seconds == pytest.approx(45.3)
        assert report.created_at is not None

    def test_title_only_no_sections(
        self,
        title_only_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=title_only_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        assert report.title == "My Research Title"
        assert report.abstract == ""
        assert report.sections == []

    def test_special_characters_in_section_titles(
        self,
        special_chars_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=special_chars_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        titles = [s.title for s in report.sections]
        assert "Performance & Memory Safety" in titles
        assert "Cost-Benefit Analysis: $$$" in titles

    def test_elapsed_seconds_default(
        self,
        well_formed_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=well_formed_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        assert report.elapsed_seconds == 0.0

    def test_metadata_default_empty(
        self,
        well_formed_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=well_formed_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        assert report.metadata == {}

    def test_max_citations_kwarg(
        self,
        well_formed_markdown: str,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=well_formed_markdown,
            extractions=sample_extractions,
            pattern_name="p4_perspective_storm",
            cost_usd=0.0,
            total_tokens=0,
            max_citations=2,
        )
        assert len(report.citations) == 2

    def test_custom_skip_sections(
        self,
        well_formed_markdown: str,
    ) -> None:
        """Only skip abstract, keep references in body."""
        report = parse_markdown_report(
            query="q",
            markdown=well_formed_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
            skip_sections={"abstract"},
        )
        titles = [s.title for s in report.sections]
        assert "References" in titles

    def test_references_and_sources_skipped(
        self,
        references_and_sources_markdown: str,
    ) -> None:
        report = parse_markdown_report(
            query="q",
            markdown=references_and_sources_markdown,
            extractions=[],
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        titles = [s.title for s in report.sections]
        assert "References" not in titles
        assert "Sources" not in titles
        assert "Body Section" in titles

    def test_different_pattern_names(
        self,
        well_formed_markdown: str,
    ) -> None:
        for pname in [
            "p0_baseline",
            "p1_iterative_rag",
            "p2_supervisor_parallel",
            "p4_perspective_storm",
            "p5_hierarchical_wd",
        ]:
            report = parse_markdown_report(
                query="q",
                markdown=well_formed_markdown,
                extractions=[],
                pattern_name=pname,
                cost_usd=0.0,
                total_tokens=0,
            )
            assert report.pattern_name == pname

    def test_full_text_roundtrip(
        self,
        sample_extractions: list[SourceExtraction],
    ) -> None:
        """Verify that full_text() on the produced report generates sensible output."""
        md = (
            "# Roundtrip Test\n\n"
            "## Abstract\n"
            "Test abstract.\n\n"
            "## Body\n"
            "Body content [1].\n"
        )
        report = parse_markdown_report(
            query="q",
            markdown=md,
            extractions=sample_extractions,
            pattern_name="p0_baseline",
            cost_usd=0.0,
            total_tokens=0,
        )
        full = report.full_text()
        assert "# Roundtrip Test" in full
        assert "## Abstract" in full
        assert "Test abstract." in full
        assert "## Body" in full
        assert "## References" in full
