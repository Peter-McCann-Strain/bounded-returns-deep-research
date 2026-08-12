"""Tests for cost tracker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from deep_research.tools.cost_tracker import CostTracker, BudgetExceeded


def test_record():
    ct = CostTracker(budget_usd=10.0)
    ct.record("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    assert ct.total_cost > 0
    assert ct.total_tokens == 1500


def test_budget_enforcement():
    ct = CostTracker(budget_usd=0.0001)
    ct.record("gpt-4o-mini", input_tokens=10000, output_tokens=5000)
    with pytest.raises(BudgetExceeded):
        ct.check_budget()


def test_summary_by_model():
    ct = CostTracker()
    ct.record("gpt-4o-mini", 100, 50)
    ct.record("gpt-4.1", 200, 100)
    ct.record("gpt-4o-mini", 50, 25)

    summary = ct.summary_by_model()
    assert "gpt-4o-mini" in summary
    assert summary["gpt-4o-mini"]["calls"] == 2
    assert "gpt-4.1" in summary
    assert summary["gpt-4.1"]["calls"] == 1


def test_summary_text():
    ct = CostTracker()
    ct.record("gpt-4o-mini", 100, 50)
    text = ct.summary_text()
    assert "Total:" in text
    assert "gpt-4o-mini" in text
