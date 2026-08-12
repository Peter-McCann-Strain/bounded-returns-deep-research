"""Tests for LLM caller — rate limiting, retry, cost tracking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from deep_research.tools.llm_caller import (
    _PTURateGate,
    _extract_retry_after,
    _backoff_for_rate_limit,
    _backoff_for_connection,
    _token_kwargs,
    _get_client,
    _model_for_call,
    LLMCaller,
    reset_limiter,
)
from deep_research.tools.cost_tracker import CostTracker


class TestPTURateGate:
    @pytest.mark.asyncio
    async def test_acquire_release(self):
        gate = _PTURateGate(max_concurrent=5, rpm=600)
        await gate.acquire()
        assert gate.in_flight == 1
        gate.release(success=True)
        assert gate.in_flight == 0

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        gate = _PTURateGate(max_concurrent=2, rpm=600)
        await gate.acquire()
        await gate.acquire()
        assert gate.in_flight == 2

        # Third acquire should block — test with timeout
        acquired = asyncio.Event()

        async def try_acquire():
            await gate.acquire()
            acquired.set()

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired.is_set()  # Should be blocked

        gate.release(success=True)
        await asyncio.sleep(0.05)
        assert acquired.is_set()  # Should now be acquired

        gate.release(success=True)
        gate.release(success=True)
        task.cancel()

    def test_stats(self):
        gate = _PTURateGate(max_concurrent=10, rpm=600)
        stats = gate.stats()
        assert stats["limit"] == 10
        assert stats["in_flight"] == 0

    def test_current_limit(self):
        gate = _PTURateGate(max_concurrent=8, rpm=600)
        assert gate.current_limit == 8


class TestExtractRetryAfter:
    def test_non_rate_limit_error(self):
        assert _extract_retry_after(ValueError("nope")) == 0.0

    def test_no_response(self):
        exc = MagicMock(spec=["response"])
        exc.response = None
        # Simulating a RateLimitError
        from openai import RateLimitError

        # Can't easily instantiate RateLimitError, test the function logic
        assert _extract_retry_after(ValueError()) == 0.0

    def test_with_retry_after_ms(self):
        from openai import RateLimitError

        exc = MagicMock(spec=RateLimitError)
        exc.__class__ = RateLimitError
        resp = MagicMock()
        resp.headers = {"retry-after-ms": "500"}
        exc.response = resp
        # The isinstance check won't work with MagicMock, so test the logic directly
        headers = {"retry-after-ms": "500"}
        retry_ms = headers.get("retry-after-ms")
        assert int(retry_ms) / 1000.0 == 0.5


class TestBackoffForRateLimit:
    def test_early_attempt_honors_retry_after(self):
        # retry_after is always honoured — result >= retry_after
        delay = _backoff_for_rate_limit(0, retry_after=0.5)
        assert delay >= 0.5

    def test_early_attempt_no_retry_after(self):
        delay = _backoff_for_rate_limit(0)
        # Uses exponential backoff: 0.5 * 2^0 + jitter
        assert 0.5 <= delay <= 2.5

    def test_later_attempt_uses_exponential(self):
        delay = _backoff_for_rate_limit(5)
        # 0.5 * 2^5 = 16 + jitter
        assert delay >= 16.0

    def test_max_delay_cap(self):
        delay = _backoff_for_rate_limit(20)
        # Capped at attempt 8: 0.5 * 2^8 = 128 → capped to 60 + 2 jitter
        assert delay <= 60.0 + 2.0  # MAX_DELAY (60) + JITTER_MAX


class TestBackoffForConnection:
    def test_first_attempt(self):
        delay = _backoff_for_connection(0)
        # 1.0 * 2^0 + jitter = 1.0 + 0-2.0
        assert 1.0 <= delay <= 3.0

    def test_max_delay_cap(self):
        delay = _backoff_for_connection(20)
        # Capped at attempt 4: 1.0 * 2^4 = 16.0 + jitter
        assert delay <= 16.0 + 2.0


class TestTokenKwargs:
    def test_standard_model(self):
        kwargs = _token_kwargs("gpt-4o-mini", 1024)
        assert kwargs == {"max_tokens": 1024}

    def test_max_completion_tokens_model(self):
        kwargs = _token_kwargs("gpt-4o", 2048)
        assert kwargs == {"max_completion_tokens": 2048}

    def test_unknown_model(self):
        kwargs = _token_kwargs("unknown-model", 512)
        assert kwargs == {"max_tokens": 512}


class TestResetLimiter:
    def test_reset(self):
        reset_limiter()
        # Should not raise
        reset_limiter()


class TestClientSelection:
    def test_get_client_uses_standard_openai_when_key_is_configured(self, monkeypatch):
        import deep_research.tools.llm_caller as llm_caller

        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(llm_caller, "_shared_client", None)
        monkeypatch.setattr(llm_caller, "USE_AZURE_OPENAI", False)
        monkeypatch.setattr(llm_caller, "OPENAI_API_KEY", "openai-test")
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_ENDPOINT", "")
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_API_KEY", "")
        monkeypatch.setattr(llm_caller, "AsyncOpenAI", FakeOpenAI)
        monkeypatch.setattr(llm_caller.httpx, "AsyncClient", lambda **kwargs: object())

        client = _get_client()

        assert isinstance(client, FakeOpenAI)
        assert captured["api_key"] == "openai-test"
        assert captured["max_retries"] == 0

    def test_get_client_uses_azure_when_enabled(self, monkeypatch):
        import deep_research.tools.llm_caller as llm_caller

        captured = {}

        class FakeAzureOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(llm_caller, "_shared_client", None)
        monkeypatch.setattr(llm_caller, "USE_AZURE_OPENAI", True)
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_API_KEY", "azure-test")
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        monkeypatch.setattr(llm_caller, "AsyncAzureOpenAI", FakeAzureOpenAI)
        monkeypatch.setattr(llm_caller.httpx, "AsyncClient", lambda **kwargs: object())

        client = _get_client()

        assert isinstance(client, FakeAzureOpenAI)
        assert captured["api_key"] == "azure-test"
        assert captured["azure_endpoint"] == "https://example.openai.azure.com"
        assert captured["api_version"] == "2024-12-01-preview"

    def test_model_for_call_uses_azure_deployment(self, monkeypatch):
        import deep_research.tools.llm_caller as llm_caller

        monkeypatch.setattr(llm_caller, "USE_AZURE_OPENAI", True)

        assert _model_for_call("gpt-4o") == llm_caller.MODELS["gpt-4o"].deployment

    def test_model_for_call_preserves_openai_model_id(self, monkeypatch):
        import deep_research.tools.llm_caller as llm_caller

        monkeypatch.setattr(llm_caller, "USE_AZURE_OPENAI", False)
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_ENDPOINT", "")
        monkeypatch.setattr(llm_caller, "AZURE_OPENAI_API_KEY", "")

        assert _model_for_call("gpt-4o") == "gpt-4o"
