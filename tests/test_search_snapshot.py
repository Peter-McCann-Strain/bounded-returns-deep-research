"""Tests for search result snapshot functionality."""

import json

import pytest

from deep_research.evaluation.search_snapshot import (
    SearchResultSnapshot,
    EvalRunSnapshot,
    SnapshotStore,
    capture_search_results,
)


# ── SearchResultSnapshot ─────────────────────────────────────────────────


class TestSearchResultSnapshot:
    def test_basic_construction(self):
        snap = SearchResultSnapshot(
            query="test query",
            search_queries=["q1", "q2"],
            results=[{"title": "A"}, {"title": "B"}],
        )
        assert snap.query == "test query"
        assert snap.n_results == 2
        assert snap.timestamp  # auto-populated

    def test_empty_results(self):
        snap = SearchResultSnapshot(query="q", search_queries=[], results=[])
        assert snap.n_results == 0

    def test_source_types(self):
        snap = SearchResultSnapshot(
            query="q",
            search_queries=["q"],
            results=[],
            source_types=["web", "arxiv"],
        )
        assert snap.source_types == ["web", "arxiv"]


# ── EvalRunSnapshot ──────────────────────────────────────────────────────


class TestEvalRunSnapshot:
    def test_basic_construction(self):
        snap = EvalRunSnapshot(run_id="r1", pattern="p0_baseline", query_id="q1")
        assert snap.run_id == "r1"
        assert snap.total_results == 0
        assert snap.timestamp

    def test_total_results(self):
        s1 = SearchResultSnapshot(query="q", search_queries=[], results=[{}, {}])
        s2 = SearchResultSnapshot(query="q", search_queries=[], results=[{}])
        snap = EvalRunSnapshot(
            run_id="r1", pattern="p0", query_id="q1", snapshots=[s1, s2]
        )
        assert snap.total_results == 3


# ── SnapshotStore ────────────────────────────────────────────────────────


class TestSnapshotStore:
    def test_save_and_load(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        s = SearchResultSnapshot(
            query="test", search_queries=["q1"], results=[{"title": "A"}]
        )
        snap = EvalRunSnapshot(
            run_id="r1", pattern="p0_baseline", query_id="q1", snapshots=[s]
        )
        store.save(snap)
        loaded = store.load("p0_baseline", "q1")
        assert loaded is not None
        assert loaded.run_id == "r1"
        assert loaded.pattern == "p0_baseline"
        assert loaded.total_results == 1

    def test_load_missing(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        assert store.load("nonexistent", "q1") is None

    def test_exists(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        snap = EvalRunSnapshot(run_id="r1", pattern="p0", query_id="q1")
        assert not store.exists("p0", "q1")
        store.save(snap)
        assert store.exists("p0", "q1")

    def test_list_snapshots_empty(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        assert store.list_snapshots() == []

    def test_list_snapshots_all(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        store.save(EvalRunSnapshot(run_id="r1", pattern="p0", query_id="q1"))
        store.save(EvalRunSnapshot(run_id="r2", pattern="p0", query_id="q2"))
        store.save(EvalRunSnapshot(run_id="r3", pattern="p1", query_id="q1"))
        result = store.list_snapshots()
        assert len(result) == 3

    def test_list_snapshots_filtered(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        store.save(EvalRunSnapshot(run_id="r1", pattern="p0", query_id="q1"))
        store.save(EvalRunSnapshot(run_id="r2", pattern="p1", query_id="q1"))
        result = store.list_snapshots(pattern="p0")
        assert len(result) == 1
        assert result[0] == ("p0", "q1")

    def test_snapshot_stats(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        store.save(EvalRunSnapshot(run_id="r1", pattern="p0", query_id="q1"))
        store.save(EvalRunSnapshot(run_id="r2", pattern="p1", query_id="q1"))
        stats = store.snapshot_stats()
        assert stats["total_snapshots"] == 2
        assert stats["by_pattern"]["p0"] == 1
        assert stats["by_pattern"]["p1"] == 1
        assert stats["total_size_bytes"] > 0

    def test_corrupted_file_returns_none(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        path = tmp_path / "snapshots" / "p0" / "q1.json"
        path.parent.mkdir(parents=True)
        path.write_text("not valid json{{{")
        assert store.load("p0", "q1") is None

    def test_overwrite_existing(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        snap1 = EvalRunSnapshot(
            run_id="r1", pattern="p0", query_id="q1", metadata={"v": 1}
        )
        snap2 = EvalRunSnapshot(
            run_id="r2", pattern="p0", query_id="q1", metadata={"v": 2}
        )
        store.save(snap1)
        store.save(snap2)
        loaded = store.load("p0", "q1")
        assert loaded.run_id == "r2"
        assert loaded.metadata["v"] == 2

    def test_save_returns_path(self, tmp_path):
        store = SnapshotStore(tmp_path / "snapshots")
        snap = EvalRunSnapshot(run_id="r1", pattern="p0", query_id="q1")
        path = store.save(snap)
        assert path.exists()
        assert path.suffix == ".json"


# ── capture_search_results ───────────────────────────────────────────────


class TestCaptureSearchResults:
    def test_basic_capture(self):
        from unittest.mock import MagicMock

        doc = MagicMock()
        doc.model_dump.return_value = {"title": "Test", "content": "text"}
        doc.source_type = "web"

        snap = capture_search_results("query", ["q1"], [doc])
        assert snap.query == "query"
        assert snap.n_results == 1
        assert snap.results[0]["title"] == "Test"
        assert "web" in snap.source_types

    def test_empty_documents(self):
        snap = capture_search_results("query", [], [])
        assert snap.n_results == 0
        assert snap.results == []

    def test_dict_fallback(self):
        class SimpleDoc:
            def __init__(self):
                self.title = "T"
                self.content = "C"
                self.source_type = "academic"

        snap = capture_search_results("q", ["q1"], [SimpleDoc()])
        assert snap.n_results == 1
        assert snap.source_types == ["academic"]

    def test_multiple_source_types(self):
        from unittest.mock import MagicMock

        doc1 = MagicMock()
        doc1.model_dump.return_value = {"t": "a"}
        doc1.source_type = "web"

        doc2 = MagicMock()
        doc2.model_dump.return_value = {"t": "b"}
        doc2.source_type = "arxiv"

        snap = capture_search_results("q", ["q1"], [doc1, doc2])
        assert snap.n_results == 2
        assert set(snap.source_types) == {"arxiv", "web"}
