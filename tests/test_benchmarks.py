"""Tests for benchmark framework and dataset integrations."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.benchmarks.base import (
    BenchmarkLoadError,
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkSuite,
)
from deep_research.types import ResearchReport, Section, Citation


# ── BenchmarkQuery tests ────────────────────────────────────────────────────


class TestBenchmarkQuery:
    def test_create_minimal(self):
        q = BenchmarkQuery(id="q1", query="test query")
        assert q.domain == ""
        assert q.rubric == {}

    def test_create_full(self):
        q = BenchmarkQuery(
            id="q1",
            query="Compare X and Y",
            domain="cs",
            difficulty="complex",
            rubric={"coverage": ["point1", "point2"]},
            reference_answer="Reference text",
            expected_citations=["paper1"],
        )
        assert q.domain == "cs"
        assert len(q.rubric["coverage"]) == 2


# ── BenchmarkResult tests ───────────────────────────────────────────────────


class TestBenchmarkResult:
    def test_create_default(self):
        r = BenchmarkResult(
            benchmark_name="test",
            pattern_name="p0",
            query_id="q1",
        )
        assert r.overall_score == 0.0
        assert r.scores == {}

    def test_to_dict(self):
        r = BenchmarkResult(
            benchmark_name="DRACO",
            pattern_name="p1_iterative_rag",
            query_id="draco_0001",
            scores={"coverage": 0.85, "citations": 0.70},
            overall_score=0.775,
            cost_usd=2.5,
            total_tokens=50000,
            latency_seconds=120.5,
        )
        d = r.to_dict()
        assert d["benchmark"] == "DRACO"
        assert d["overall_score"] == 0.775
        assert d["cost_usd"] == 2.5
        assert d["scores"]["coverage"] == 0.85


# ── DRACO benchmark tests ───────────────────────────────────────────────────


class TestDRACOBenchmark:
    def test_name(self):
        from deep_research.benchmarks.draco import DRACOBenchmark

        bench = DRACOBenchmark()
        assert bench.name == "DRACO"

    @pytest.mark.asyncio
    async def test_score_report(self, sample_report):
        from deep_research.benchmarks.draco import DRACOBenchmark

        bench = DRACOBenchmark()

        query = BenchmarkQuery(
            id="test_1",
            query="Compare BERT and GPT",
            rubric={
                "factual_accuracy": [
                    {"description": "mentions encoder architecture", "weight": 5},
                    {"description": "mentions decoder architecture", "weight": 5},
                    {"description": "mentions transformer attention", "weight": 3},
                ],
            },
        )

        result = await bench.score(query, sample_report)
        assert result.benchmark_name == "DRACO"
        assert result.overall_score >= 0.0
        assert result.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_score_empty_rubric(self, sample_report):
        from deep_research.benchmarks.draco import DRACOBenchmark

        bench = DRACOBenchmark()

        query = BenchmarkQuery(id="test_2", query="test", rubric={})
        result = await bench.score(query, sample_report)
        assert result.overall_score == 0.0

    def test_check_criterion(self):
        from deep_research.benchmarks.draco import DRACOBenchmark

        bench = DRACOBenchmark()

        text = "BERT uses bidirectional masked language modeling."
        assert bench._check_criterion(text, "bidirectional language modeling")
        assert not bench._check_criterion(text, "quantum computing superconductor")


# ── ScholarQABench tests ────────────────────────────────────────────────────


class TestScholarQABenchmark:
    def test_name(self):
        from deep_research.benchmarks.scholar_qa import ScholarQABenchmark

        bench = ScholarQABenchmark()
        assert bench.name == "ScholarQABench"

    @pytest.mark.asyncio
    async def test_score_report(self, sample_report):
        from deep_research.benchmarks.scholar_qa import ScholarQABenchmark

        bench = ScholarQABenchmark()

        query = BenchmarkQuery(
            id="sqb_cs_0001",
            query="Compare BERT and GPT architectures",
            reference_answer="BERT uses bidirectional encoding. GPT uses autoregressive decoding.",
            expected_citations=["BERT Paper"],
        )

        result = await bench.score(query, sample_report)
        assert result.benchmark_name == "ScholarQABench"
        assert "coverage" in result.scores
        assert "citation_precision" in result.scores
        assert 0.0 <= result.overall_score <= 1.0

    def test_score_organization(self, sample_report):
        from deep_research.benchmarks.scholar_qa import ScholarQABenchmark

        bench = ScholarQABenchmark()
        score = bench._score_organization(sample_report)
        # Has sections, abstract, title, citations
        assert score > 0.5


# ── FreshWiki benchmark tests ───────────────────────────────────────────────


class TestFreshWikiBenchmark:
    def test_name(self):
        from deep_research.benchmarks.freshwiki import FreshWikiBenchmark

        bench = FreshWikiBenchmark()
        assert bench.name == "FreshWiki"

    @pytest.mark.asyncio
    async def test_score_report(self, sample_report):
        from deep_research.benchmarks.freshwiki import FreshWikiBenchmark

        bench = FreshWikiBenchmark()

        query = BenchmarkQuery(
            id="fw_0001",
            query="Write an article about BERT",
            reference_answer="BERT is a transformer model using masked language modeling.",
            rubric={"reference_headings": ["Architecture", "Training"]},
            metadata={"title": "BERT"},
        )

        result = await bench.score(query, sample_report)
        assert result.benchmark_name == "FreshWiki"
        assert "coverage" in result.scores
        assert "organization" in result.scores
        assert "verifiability" in result.scores

    def test_score_verifiability(self, sample_report):
        from deep_research.benchmarks.freshwiki import FreshWikiBenchmark

        bench = FreshWikiBenchmark()
        score = bench._score_verifiability(sample_report)
        # Has citations
        assert score > 0.0


# ── ResearchQA benchmark tests ─────────────────────────────────────────────


class TestResearchQABenchmark:
    def test_name(self):
        from deep_research.benchmarks.research_qa import ResearchQABenchmark

        bench = ResearchQABenchmark()
        assert bench.name == "ResearchQA"

    @pytest.mark.asyncio
    async def test_score_report(self, sample_report):
        from deep_research.benchmarks.research_qa import ResearchQABenchmark

        bench = ResearchQABenchmark()

        query = BenchmarkQuery(
            id="rqa_0001",
            query="Compare BERT and GPT architectures",
            rubric={
                "criteria": [
                    {
                        "question": "Does the response mention encoder architecture?",
                        "type": ["Architecture"],
                    },
                    {
                        "question": "Does the response describe masked language modeling?",
                        "type": ["Training"],
                    },
                    {
                        "question": "Does the response discuss autoregressive generation?",
                        "type": ["Training"],
                    },
                ],
                "types": ["Architecture", "Training"],
            },
        )

        result = await bench.score(query, sample_report)
        assert result.benchmark_name == "ResearchQA"
        assert "rubric_coverage" in result.scores
        assert "citation_quality" in result.scores
        assert 0.0 <= result.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_score_empty_rubric(self, sample_report):
        from deep_research.benchmarks.research_qa import ResearchQABenchmark

        bench = ResearchQABenchmark()

        query = BenchmarkQuery(id="rqa_empty", query="test", rubric={})
        result = await bench.score(query, sample_report)
        assert result.overall_score == 0.0

    def test_check_rubric_item(self):
        from deep_research.benchmarks.research_qa import ResearchQABenchmark

        bench = ResearchQABenchmark()

        text = "BERT uses bidirectional masked language modeling with transformer architecture."
        assert bench._check_rubric_item(
            text, "Does the response describe bidirectional transformer architecture?"
        )
        assert not bench._check_rubric_item(
            text, "Does the response discuss quantum computing algorithms?"
        )

    @pytest.mark.asyncio
    async def test_load_download_failure_is_explicit(self, monkeypatch, tmp_path):
        import deep_research.benchmarks.research_qa as research_qa

        async def fail_download(self):
            raise BenchmarkLoadError("planned failure")

        monkeypatch.setattr(research_qa, "_CACHE_DIR", tmp_path / "research_qa")
        monkeypatch.setattr(research_qa.ResearchQABenchmark, "_download", fail_download)

        with pytest.raises(BenchmarkLoadError, match="planned failure"):
            await research_qa.ResearchQABenchmark().load()


# ── DeepSearchQA benchmark tests ──────────────────────────────────────────


class TestDeepSearchQABenchmark:
    def test_name(self):
        from deep_research.benchmarks.deepsearch_qa import DeepSearchQABenchmark

        bench = DeepSearchQABenchmark()
        assert bench.name == "DeepSearchQA"

    @pytest.mark.asyncio
    async def test_score_report_with_answer(self, sample_report):
        from deep_research.benchmarks.deepsearch_qa import DeepSearchQABenchmark

        bench = DeepSearchQABenchmark()

        query = BenchmarkQuery(
            id="dsqa_0001",
            query="What architecture does BERT use?",
            rubric={
                "expected_answer": "encoder-only architecture",
                "answer_type": "Single Answer",
            },
        )

        result = await bench.score(query, sample_report)
        assert result.benchmark_name == "DeepSearchQA"
        assert "answer_found" in result.scores
        assert "evidence_quality" in result.scores
        # Answer "encoder-only architecture" IS in the sample report
        assert result.scores["answer_found"] > 0.0

    @pytest.mark.asyncio
    async def test_score_report_answer_not_found(self, sample_report):
        from deep_research.benchmarks.deepsearch_qa import DeepSearchQABenchmark

        bench = DeepSearchQABenchmark()

        query = BenchmarkQuery(
            id="dsqa_0002",
            query="What is the capital of Mars?",
            rubric={
                "expected_answer": "Olympus Mons City",
                "answer_type": "Single Answer",
            },
        )

        result = await bench.score(query, sample_report)
        assert result.scores["answer_found"] == 0.0

    def test_check_answer_direct(self):
        from deep_research.benchmarks.deepsearch_qa import DeepSearchQABenchmark

        bench = DeepSearchQABenchmark()

        text = "The answer is New Zealand, which saw crime decrease."
        assert bench._check_answer(text, "New Zealand", "Single Answer") == 1.0
        assert bench._check_answer(text, "Australia", "Single Answer") == 0.0

    def test_check_answer_list(self):
        from deep_research.benchmarks.deepsearch_qa import DeepSearchQABenchmark

        bench = DeepSearchQABenchmark()

        text = "The top countries are Australia, New Zealand, and Switzerland."
        score = bench._check_answer(text, "Australia, New Zealand, Canada", "List")
        assert score > 0.5  # Found 2 of 3


# ── LitQA2 benchmark tests ─────────────────────────────────────────────────


class TestLitQA2Benchmark:
    def test_name(self):
        from deep_research.benchmarks.litqa2 import LitQA2Benchmark

        bench = LitQA2Benchmark()
        assert bench.name == "LitQA2"

    @pytest.mark.asyncio
    async def test_score_correct_answer(self, sample_report):
        from deep_research.benchmarks.litqa2 import LitQA2Benchmark

        bench = LitQA2Benchmark()

        # "encoder-only architecture" appears in sample_report
        query = BenchmarkQuery(
            id="litqa2_0001",
            query="What type of architecture does BERT use?",
            reference_answer="encoder-only architecture",
            rubric={
                "ideal": "encoder-only architecture",
                "distractors": ["decoder-only architecture", "encoder-decoder architecture"],
                "options": [
                    "encoder-only architecture",
                    "decoder-only architecture",
                    "encoder-decoder architecture",
                    "Insufficient information",
                ],
            },
        )

        result = await bench.score(query, sample_report)
        assert result.benchmark_name == "LitQA2"
        assert result.scores["answer_correct"] == 1.0
        assert result.scores["answer_attempted"] == 1.0
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_score_wrong_answer(self, sample_report):
        from deep_research.benchmarks.litqa2 import LitQA2Benchmark

        bench = LitQA2Benchmark()

        # "quantum computing" does NOT appear in sample_report
        query = BenchmarkQuery(
            id="litqa2_0002",
            query="What method does BERT use?",
            reference_answer="quantum computing",
            rubric={
                "ideal": "quantum computing",
                "distractors": ["masked language modeling"],
                "options": [
                    "quantum computing",
                    "masked language modeling",
                    "Insufficient information",
                ],
            },
        )

        result = await bench.score(query, sample_report)
        # "masked language modeling" IS in report, but it's a distractor
        assert result.scores["answer_correct"] == 0.0
        assert result.scores["answer_attempted"] == 1.0

    @pytest.mark.asyncio
    async def test_score_no_answer(self, sample_report):
        from deep_research.benchmarks.litqa2 import LitQA2Benchmark

        bench = LitQA2Benchmark()

        query = BenchmarkQuery(
            id="litqa2_0003",
            query="What is the melting point of unobtanium?",
            reference_answer="3000 kelvin",
            rubric={
                "ideal": "3000 kelvin",
                "distractors": ["5000 kelvin", "1000 kelvin"],
                "options": [
                    "3000 kelvin",
                    "5000 kelvin",
                    "1000 kelvin",
                    "Insufficient information",
                ],
            },
        )

        result = await bench.score(query, sample_report)
        assert result.scores["answer_correct"] == 0.0
        assert result.scores["answer_attempted"] == 0.0


# ── BenchmarkSuite tests ────────────────────────────────────────────────────


class TestBenchmarkSuite:
    def test_generate_report_empty(self):
        suite = BenchmarkSuite(benchmarks=[], patterns=["p0"])
        report = suite.generate_report()
        assert "No benchmark results" in report

    def test_generate_report_with_results(self):
        suite = BenchmarkSuite(benchmarks=[], patterns=[])
        suite.results = [
            BenchmarkResult(
                benchmark_name="DRACO",
                pattern_name="p0",
                query_id="q1",
                overall_score=0.75,
                cost_usd=1.5,
                latency_seconds=60,
                scores={"coverage": 0.8, "citations": 0.7},
            ),
            BenchmarkResult(
                benchmark_name="DRACO",
                pattern_name="p1",
                query_id="q1",
                overall_score=0.85,
                cost_usd=3.0,
                latency_seconds=120,
                scores={"coverage": 0.9, "citations": 0.8},
            ),
        ]
        report = suite.generate_report()
        assert "DRACO" in report
        assert "p0" in report
        assert "p1" in report
        assert "Score Dimensions" in report

    @pytest.mark.asyncio
    async def test_run_pattern_mocked(self):
        mock_report = ResearchReport(
            query="test",
            pattern_name="p0_baseline",
            total_cost_usd=0.5,
            total_tokens=1000,
        )

        suite = BenchmarkSuite(benchmarks=[], patterns=["p0_baseline"])

        with patch.dict(
            "deep_research.benchmarks.base.PATTERN_MODULES",
            {"p0_baseline": "deep_research.patterns.p0_baseline.pipeline"},
        ):
            mock_mod = MagicMock()
            mock_mod.run = AsyncMock(return_value=mock_report)

            with patch("deep_research.benchmarks.base.importlib") as mock_import:
                mock_import.import_module.return_value = mock_mod
                report = await suite.run_pattern(
                    "p0_baseline", BenchmarkQuery(id="q1", query="test")
                )

        assert report.pattern_name == "p0_baseline"
