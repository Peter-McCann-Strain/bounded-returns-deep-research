"""Tests for URL extractor — trafilatura + BS4 fallback."""

from unittest.mock import patch, MagicMock

import pytest

from deep_research.tools.url_extractor import URLExtractor
from deep_research.types import SourceType


class TestURLExtractor:
    @pytest.mark.asyncio
    async def test_extract_with_trafilatura(self):
        extractor = URLExtractor()
        # Content must be > 100 chars
        long_text = "This is extracted content. " * 10  # ~260 chars

        def mock_extract_sync(url, timeout):
            return long_text

        with patch.object(extractor, '_extract_sync', side_effect=mock_extract_sync):
            doc = await extractor.extract("https://example.com")

        assert doc is not None
        assert doc.source_type == SourceType.URL_EXTRACT
        assert doc.url == "https://example.com"
        assert len(doc.content) > 100

    @pytest.mark.asyncio
    async def test_extract_bs4_fallback(self):
        extractor = URLExtractor()

        html = "<html><body><p>" + "Content paragraph. " * 20 + "</p></body></html>"

        with patch('deep_research.tools.url_extractor.trafilatura') as mock_traf:
            mock_traf.fetch_url.return_value = html
            mock_traf.extract.return_value = None  # trafilatura fails

            doc = await extractor.extract("https://example.com")

        assert doc is not None
        assert "Content paragraph" in doc.content

    @pytest.mark.asyncio
    async def test_extract_returns_none_on_failure(self):
        extractor = URLExtractor()

        with patch('deep_research.tools.url_extractor.trafilatura') as mock_traf:
            mock_traf.fetch_url.return_value = None

            doc = await extractor.extract("https://example.com")

        assert doc is None

    @pytest.mark.asyncio
    async def test_extract_batch(self):
        extractor = URLExtractor()
        long_content = "Extracted text content. " * 10

        with patch('deep_research.tools.url_extractor.trafilatura') as mock_traf:
            mock_traf.fetch_url.return_value = "<html>test</html>"
            mock_traf.extract.return_value = long_content

            docs = await extractor.extract_batch(
                ["https://a.com", "https://b.com"],
                max_concurrent=2,
            )

        assert len(docs) == 2

    def test_extract_title(self):
        extractor = URLExtractor()
        title = extractor._extract_title("First Title Line\nSecond line\nThird")
        assert title == "First Title Line"

    def test_extract_title_skips_short_lines(self):
        extractor = URLExtractor()
        title = extractor._extract_title("Hi\nThis is the actual title\nMore")
        assert title == "This is the actual title"

    def test_extract_title_empty(self):
        extractor = URLExtractor()
        title = extractor._extract_title("")
        assert title == ""
