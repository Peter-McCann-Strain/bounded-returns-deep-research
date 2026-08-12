"""Tests for web search (Tavily) — caching, dedup, Document creation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.tools.web_search import WebSearcher, _cache_key
from deep_research.types import Document, SourceType


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("test") == _cache_key("test")

    def test_different_queries(self):
        assert _cache_key("query1") != _cache_key("query2")

    def test_length(self):
        key = _cache_key("test query")
        assert len(key) == 16


class TestWebSearcher:
    @pytest.mark.asyncio
    async def test_search_returns_documents(self):
        searcher = WebSearcher(api_key="test")

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/test",
                    "content": "Short content",
                    "raw_content": "Full raw content of the page",
                    "score": 0.95,
                },
            ]
        })

        with patch.object(searcher, '_get_client', return_value=mock_client), \
             patch('deep_research.tools.web_search._CACHE_DIR', Path(tempfile.mkdtemp())):
            docs = await searcher.search("test query", use_cache=False)

        assert len(docs) == 1
        assert docs[0].title == "Test Result"
        assert docs[0].source_type == SourceType.WEB
        assert docs[0].content == "Full raw content of the page"
        assert docs[0].metadata["score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_prefers_raw_content(self):
        searcher = WebSearcher(api_key="test")
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={
            "results": [
                {
                    "title": "Test",
                    "url": "https://example.com",
                    "content": "Short snippet",
                    "raw_content": "Full detailed content",
                    "score": 0.5,
                },
            ]
        })

        with patch.object(searcher, '_get_client', return_value=mock_client), \
             patch('deep_research.tools.web_search._CACHE_DIR', Path(tempfile.mkdtemp())):
            docs = await searcher.search("test", use_cache=False)

        assert docs[0].content == "Full detailed content"

    @pytest.mark.asyncio
    async def test_search_batch_deduplicates(self):
        searcher = WebSearcher(api_key="test")

        mock_client = AsyncMock()
        # Both queries return same URL
        mock_client.search = AsyncMock(return_value={
            "results": [
                {"title": "Same", "url": "https://same.com", "content": "x", "score": 0.5},
            ]
        })

        with patch.object(searcher, '_get_client', return_value=mock_client), \
             patch('deep_research.tools.web_search._CACHE_DIR', Path(tempfile.mkdtemp())):
            docs = await searcher.search_batch(
                ["query1", "query2"], max_results_per=1
            )

        # Should deduplicate to 1
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_search_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            # Pre-populate cache
            ck = _cache_key("cached query")
            cache_data = [
                Document(id="c1", title="Cached", content="cached content",
                         url="https://cached.com", source_type=SourceType.WEB).model_dump(mode="json")
            ]
            (cache_dir / f"{ck}.json").write_text(json.dumps(cache_data, default=str))

            searcher = WebSearcher(api_key="test")
            with patch('deep_research.tools.web_search._CACHE_DIR', cache_dir):
                docs = await searcher.search("cached query", use_cache=True)

            assert len(docs) == 1
            assert docs[0].title == "Cached"
