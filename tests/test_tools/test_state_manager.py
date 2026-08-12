"""Tests for state manager."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from deep_research.tools.state_manager import StateManager


def test_save_load_roundtrip():
    sm = StateManager("test_pattern", run_id="test_run_001")
    data = {"queries": ["q1", "q2"], "count": 42, "nested": {"a": 1}}
    sm.save("stage1", data)

    loaded = sm.load("stage1")
    assert loaded == data


def test_has_checkpoint():
    sm = StateManager("test_pattern", run_id="test_run_002")
    assert not sm.has_checkpoint("missing")
    sm.save("exists", {"x": 1})
    assert sm.has_checkpoint("exists")


def test_list_stages():
    sm = StateManager("test_pattern", run_id="test_run_003")
    sm.save("stage_a", {})
    sm.save("stage_b", {})
    stages = sm.list_stages()
    assert "stage_a" in stages
    assert "stage_b" in stages


def test_load_missing():
    sm = StateManager("test_pattern", run_id="test_run_004")
    assert sm.load("nonexistent") is None
