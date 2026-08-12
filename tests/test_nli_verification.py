"""Tests for NLI-based citation verification additions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.evaluation.citation_verifier import (
    NLIVerificationResult,
    nli_verify_claim,
    nli_verify_batch,
    compute_nli_metrics,
    CitationVerifier,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_llm(**overrides):
    """Create a mock LLMCaller."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value="mock")
    llm.complete_json = AsyncMock(return_value=overrides.get("default_json", {}))
    return llm


def _make_mock_url_extractor():
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=None)
    return extractor


# ── 1. NLIVerificationResult dataclass ───────────────────────────────────────


class TestNLIVerificationResult:
    def test_entailment(self):
        r = NLIVerificationResult(
            claim_text="BERT uses attention.",
            source_text="BERT is based on self-attention.",
            label="entailment",
            confidence=0.95,
            reasoning="Source confirms.",
        )
        assert r.is_supported is True
        assert r.is_contradicted is False

    def test_contradiction(self):
        r = NLIVerificationResult(
            claim_text="BERT uses RNNs.",
            source_text="BERT does not use RNNs.",
            label="contradiction",
            confidence=0.9,
            reasoning="Source contradicts.",
        )
        assert r.is_supported is False
        assert r.is_contradicted is True

    def test_neutral(self):
        r = NLIVerificationResult(
            claim_text="BERT is fast.",
            source_text="BERT is a language model.",
            label="neutral",
            confidence=0.5,
        )
        assert r.is_supported is False
        assert r.is_contradicted is False

    def test_default_reasoning(self):
        r = NLIVerificationResult(
            claim_text="X", source_text="Y",
            label="neutral", confidence=0.5,
        )
        assert r.reasoning == ""


# ── 2. nli_verify_claim function ─────────────────────────────────────────────


class TestNLIVerifyClaim:
    @pytest.mark.asyncio
    async def test_entailment_result(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "entailment",
            "confidence": 0.92,
            "reasoning": "Source clearly supports the claim.",
        })
        result = await nli_verify_claim(
            claim_text="BERT uses bidirectional attention.",
            source_text="BERT employs a bidirectional self-attention mechanism.",
            llm_caller=llm,
        )
        assert result.label == "entailment"
        assert result.confidence == 0.92
        assert result.is_supported is True

    @pytest.mark.asyncio
    async def test_contradiction_result(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "contradiction",
            "confidence": 0.85,
            "reasoning": "Source says the opposite.",
        })
        result = await nli_verify_claim(
            claim_text="GPT uses bidirectional attention.",
            source_text="GPT uses unidirectional (causal) attention only.",
            llm_caller=llm,
        )
        assert result.label == "contradiction"
        assert result.is_contradicted is True

    @pytest.mark.asyncio
    async def test_empty_claim(self):
        llm = _make_mock_llm()
        result = await nli_verify_claim(
            claim_text="",
            source_text="Some source text.",
            llm_caller=llm,
        )
        assert result.label == "neutral"
        assert result.confidence == 0.0
        # LLM should not be called
        llm.complete_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_source(self):
        llm = _make_mock_llm()
        result = await nli_verify_claim(
            claim_text="Some claim.",
            source_text="",
            llm_caller=llm,
        )
        assert result.label == "neutral"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_llm_error(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(side_effect=RuntimeError("API down"))
        result = await nli_verify_claim(
            claim_text="Claim.",
            source_text="Source.",
            llm_caller=llm,
        )
        assert result.label == "neutral"
        assert result.confidence == 0.0
        assert "LLM call failed" in result.reasoning

    @pytest.mark.asyncio
    async def test_invalid_label_normalised(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "maybe_entailment",
            "confidence": 0.6,
            "reasoning": "unclear",
        })
        result = await nli_verify_claim(
            claim_text="Claim.",
            source_text="Source.",
            llm_caller=llm,
        )
        assert result.label == "neutral"  # Invalid label normalised

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "entailment",
            "confidence": 1.5,  # out of range
            "reasoning": "very confident",
        })
        result = await nli_verify_claim(
            claim_text="Claim.",
            source_text="Source.",
            llm_caller=llm,
        )
        assert result.confidence == 1.0  # Clamped

    @pytest.mark.asyncio
    async def test_confidence_non_numeric(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "entailment",
            "confidence": "high",
            "reasoning": "ok",
        })
        result = await nli_verify_claim(
            claim_text="Claim.",
            source_text="Source.",
            llm_caller=llm,
        )
        assert result.confidence == 0.5  # Default on parse error


# ── 3. nli_verify_batch function ─────────────────────────────────────────────


class TestNLIVerifyBatch:
    @pytest.mark.asyncio
    async def test_empty_batch(self):
        llm = _make_mock_llm()
        results = await nli_verify_batch([], llm)
        assert results == []

    @pytest.mark.asyncio
    async def test_normal_batch(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "entailment",
            "confidence": 0.9,
            "reasoning": "supported",
        })
        pairs = [
            ("Claim 1", "Source 1"),
            ("Claim 2", "Source 2"),
            ("Claim 3", "Source 3"),
        ]
        results = await nli_verify_batch(pairs, llm)
        assert len(results) == 3
        assert all(r.label == "entailment" for r in results)

    @pytest.mark.asyncio
    async def test_batch_with_mixed_results(self):
        llm = _make_mock_llm()
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"label": "entailment", "confidence": 0.9, "reasoning": "ok"}
            elif call_count == 2:
                return {"label": "contradiction", "confidence": 0.8, "reasoning": "no"}
            else:
                return {"label": "neutral", "confidence": 0.5, "reasoning": "maybe"}

        llm.complete_json = AsyncMock(side_effect=_side_effect)
        pairs = [("C1", "S1"), ("C2", "S2"), ("C3", "S3")]
        results = await nli_verify_batch(pairs, llm, max_concurrent=1)
        labels = [r.label for r in results]
        assert "entailment" in labels
        assert "contradiction" in labels
        assert "neutral" in labels


# ── 4. compute_nli_metrics function ──────────────────────────────────────────


class TestComputeNLIMetrics:
    def test_empty_results(self):
        metrics = compute_nli_metrics([])
        assert metrics["n_total"] == 0
        assert metrics["entailment_rate"] == 0.0

    def test_all_entailment(self):
        results = [
            NLIVerificationResult("C1", "S1", "entailment", 0.9),
            NLIVerificationResult("C2", "S2", "entailment", 0.8),
        ]
        metrics = compute_nli_metrics(results)
        assert metrics["n_total"] == 2
        assert metrics["n_entailment"] == 2
        assert metrics["entailment_rate"] == 1.0
        assert metrics["contradiction_rate"] == 0.0

    def test_mixed_labels(self):
        results = [
            NLIVerificationResult("C1", "S1", "entailment", 0.9),
            NLIVerificationResult("C2", "S2", "neutral", 0.5),
            NLIVerificationResult("C3", "S3", "contradiction", 0.8),
            NLIVerificationResult("C4", "S4", "entailment", 0.7),
        ]
        metrics = compute_nli_metrics(results)
        assert metrics["n_total"] == 4
        assert metrics["n_entailment"] == 2
        assert metrics["n_neutral"] == 1
        assert metrics["n_contradiction"] == 1
        assert abs(metrics["entailment_rate"] - 0.5) < 1e-6
        assert abs(metrics["contradiction_rate"] - 0.25) < 1e-6
        assert abs(metrics["mean_confidence"] - 0.725) < 1e-6


# ── 5. CitationVerifier unified class ────────────────────────────────────────


class TestCitationVerifier:
    def test_init_nli_mode(self):
        llm = _make_mock_llm()
        verifier = CitationVerifier(
            llm_caller=llm,
            nli_mode=True,
            nli_model="gpt-4o-mini",
        )
        assert verifier.nli_mode is True
        assert verifier.nli_model == "gpt-4o-mini"
        # No url_extractor -> no agentic verifier
        assert verifier._agentic is None

    def test_init_with_url_extractor(self):
        llm = _make_mock_llm()
        ext = _make_mock_url_extractor()
        verifier = CitationVerifier(
            llm_caller=llm,
            url_extractor=ext,
        )
        assert verifier._agentic is not None

    @pytest.mark.asyncio
    async def test_verify_report_no_extractor_raises(self):
        llm = _make_mock_llm()
        verifier = CitationVerifier(llm_caller=llm)
        with pytest.raises(RuntimeError, match="url_extractor"):
            await verifier.verify_report(
                report_text="Report.",
                report_id="r1",
                pattern="p0",
            )

    @pytest.mark.asyncio
    async def test_nli_verify_single(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "entailment",
            "confidence": 0.95,
            "reasoning": "confirmed",
        })
        verifier = CitationVerifier(llm_caller=llm, nli_mode=True)
        result = await verifier.nli_verify_single("Claim.", "Source.")
        assert result.label == "entailment"
        assert result.is_supported is True

    @pytest.mark.asyncio
    async def test_nli_verify_claims_batch(self):
        llm = _make_mock_llm()
        llm.complete_json = AsyncMock(return_value={
            "label": "neutral",
            "confidence": 0.5,
            "reasoning": "unclear",
        })
        verifier = CitationVerifier(llm_caller=llm, nli_mode=True)
        results = await verifier.nli_verify_claims([
            ("C1", "S1"),
            ("C2", "S2"),
        ])
        assert len(results) == 2
        assert all(r.label == "neutral" for r in results)
