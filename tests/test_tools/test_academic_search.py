"""Tests for academic search (Semantic Scholar + arXiv)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.tools.academic_search import AcademicSearcher
from deep_research.types import SourceType


class TestAcademicSearcher:
    @pytest.mark.asyncio
    async def test_search_semantic_scholar(self):
        searcher = AcademicSearcher()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {
                    "paperId": "abc123",
                    "title": "BERT Paper",
                    "abstract": "BERT is a bidirectional transformer.",
                    "url": "https://semanticscholar.org/paper/abc123",
                    "year": 2019,
                    "authors": [{"name": "Devlin"}, {"name": "Chang"}],
                    "citationCount": 50000,
                    "externalIds": {"ArXiv": "1810.04805"},
                }
            ]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            docs = await searcher.search_semantic_scholar("BERT")

        assert len(docs) == 1
        assert docs[0].title == "BERT Paper"
        assert docs[0].source_type == SourceType.SEMANTIC_SCHOLAR
        assert docs[0].metadata["citations"] == 50000

    @pytest.mark.asyncio
    async def test_search_semantic_scholar_error(self):
        searcher = AcademicSearcher(use_cache=False)

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            docs = await searcher.search_semantic_scholar("test")

        assert docs == []

    @pytest.mark.asyncio
    async def test_search_deduplicates_by_title(self):
        searcher = AcademicSearcher()

        # Mock S2 and arXiv to return same-titled paper
        s2_doc_data = {
            "data": [{
                "paperId": "s2_id",
                "title": "Same Paper",
                "abstract": "Abstract",
                "url": "https://s2.org/1",
                "year": 2024,
                "authors": [],
                "citationCount": 10,
                "externalIds": {},
            }]
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=s2_doc_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # Mock arXiv to return paper with same title
        mock_arxiv_result = MagicMock()
        mock_arxiv_result.title = "Same Paper"
        mock_arxiv_result.summary = "Summary"
        mock_arxiv_result.entry_id = "https://arxiv.org/abs/1234"
        mock_arxiv_result.authors = []
        mock_arxiv_result.published = MagicMock(year=2024)
        mock_arxiv_result.categories = ["cs.CL"]

        with patch('aiohttp.ClientSession', return_value=mock_session), \
             patch('arxiv.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client.results = MagicMock(return_value=[mock_arxiv_result])
            mock_client_cls.return_value = mock_client

            docs = await searcher.search("Same Paper")

        # Should deduplicate to 1
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_search_handles_missing_abstract(self):
        searcher = AcademicSearcher()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [{
                "paperId": "no_abstract",
                "title": "No Abstract Paper",
                "abstract": None,
                "url": "https://s2.org/2",
                "year": 2024,
                "authors": [],
                "citationCount": 0,
                "externalIds": {},
            }]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            docs = await searcher.search_semantic_scholar("test")

        assert len(docs) == 1
        assert docs[0].content == ""
