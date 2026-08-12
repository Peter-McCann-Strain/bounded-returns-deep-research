"""Root conftest for pytest — shared fixtures."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

_TEST_RUNTIME_ROOT = Path(
    os.environ.get("DR_TEST_RUNTIME_ROOT", tempfile.mkdtemp(prefix="deep-research-tests-"))
)
os.environ.setdefault("DR_ARTIFACTS_DIR", str(_TEST_RUNTIME_ROOT / "artifacts"))
os.environ.setdefault("DR_CHECKPOINTS_DIR", str(_TEST_RUNTIME_ROOT / "checkpoints"))
os.environ.setdefault("DR_REPORTS_DIR", str(_TEST_RUNTIME_ROOT / "reports"))
os.environ.setdefault("DR_RESULTS_DIR", str(_TEST_RUNTIME_ROOT / "results"))

# Prevent pytest from collecting functions in the source package whose names
# start with ``test_`` (e.g. rubric_converters.test_query_to_rubric_v2).
collect_ignore_glob = ["deep_research/*", "scripts/*", "venv/*"]


@pytest.fixture
def mock_llm():
    """Create a mock LLMCaller that returns configurable responses."""
    from deep_research.tools.cost_tracker import CostTracker

    tracker = CostTracker(budget_usd=100.0)
    llm = MagicMock()
    llm.cost_tracker = tracker

    # Default: return empty string for complete, empty dict for complete_json
    llm.complete = AsyncMock(return_value="Mock LLM response")
    llm.complete_json = AsyncMock(return_value={})
    llm.complete_messages = AsyncMock(return_value="Mock conversation response")

    return llm


@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing."""
    from deep_research.types import Document, SourceType

    return [
        Document(
            id="doc1",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            content="BERT uses masked language modeling and bidirectional attention. "
                    "The model achieves state-of-the-art results on 11 NLP tasks.",
            url="https://arxiv.org/abs/1810.04805",
            source_type=SourceType.ACADEMIC,
        ),
        Document(
            id="doc2",
            title="GPT-3: Language Models are Few-Shot Learners",
            content="GPT-3 is an autoregressive language model with 175 billion parameters. "
                    "It uses causal attention and achieves strong few-shot performance.",
            url="https://arxiv.org/abs/2005.14165",
            source_type=SourceType.ACADEMIC,
        ),
        Document(
            id="doc3",
            title="Transformer Architecture Overview",
            content="The transformer architecture uses self-attention mechanisms.",
            url="https://example.com/transformers",
            source_type=SourceType.WEB,
        ),
    ]


@pytest.fixture
def sample_extractions():
    """Create sample SourceExtraction objects for testing."""
    from deep_research.tools.source_extractor import SourceExtraction, ExtractedSourceType

    return [
        SourceExtraction(
            doc_id="doc1",
            title="BERT Paper",
            url="https://arxiv.org/abs/1810.04805",
            summary="BERT uses bidirectional masked language modeling for pre-training.",
            relevance_score=9,
            source_type=ExtractedSourceType.RESEARCH_PAPER,
            key_findings=[
                "BERT uses bidirectional attention",
                "Masked language modeling as training objective",
            ],
            confidence_notes="Foundational paper, highly reliable.",
        ),
        SourceExtraction(
            doc_id="doc2",
            title="GPT-3 Paper",
            url="https://arxiv.org/abs/2005.14165",
            summary="GPT uses autoregressive causal language modeling.",
            relevance_score=8,
            source_type=ExtractedSourceType.RESEARCH_PAPER,
            key_findings=[
                "GPT uses autoregressive generation",
                "Decoder-only transformer architecture",
            ],
            confidence_notes="Major paper by OpenAI.",
        ),
    ]


@pytest.fixture
def sample_report():
    """Create a sample ResearchReport for testing."""
    from deep_research.types import ResearchReport, Section, Citation

    return ResearchReport(
        query="Compare BERT and GPT",
        title="BERT vs GPT: A Comparative Analysis",
        abstract="This report compares BERT and GPT architectures.",
        sections=[
            Section(
                title="Architecture",
                content="BERT uses encoder-only architecture [1]. GPT uses decoder-only [2].",
            ),
            Section(
                title="Training Objectives",
                content="BERT uses masked language modeling [1]. GPT uses autoregressive modeling [2].",
            ),
            Section(
                title="Use Cases",
                content="BERT excels at classification tasks. GPT excels at generation tasks.",
            ),
        ],
        citations=[
            Citation(
                claim="[1]",
                source_id="doc1",
                source_title="BERT Paper",
                source_url="https://arxiv.org/abs/1810.04805",
                relevance_score=0.9,
            ),
            Citation(
                claim="[2]",
                source_id="doc2",
                source_title="GPT-3 Paper",
                source_url="https://arxiv.org/abs/2005.14165",
                relevance_score=0.8,
            ),
        ],
        pattern_name="p1_iterative_rag",
        total_cost_usd=1.50,
        total_tokens=50000,
        elapsed_seconds=120.0,
    )
