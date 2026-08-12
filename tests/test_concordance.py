"""Comprehensive tests for the concordance analysis module.

Tests cover:
- MethodRanking construction
- analyze_concordance with perfect agreement (W=1.0)
- analyze_concordance with complete disagreement (low W)
- analyze_concordance with known project rankings (3 methods, 6 patterns)
- identify_dimension_drivers
- generate_concordance_report produces valid markdown
- most_stable/most_volatile pattern identification
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_research.evaluation.concordance_analysis import (
    ConcordanceReport,
    MethodRanking,
    analyze_concordance,
    generate_concordance_report,
    identify_dimension_drivers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_agreement_rankings():
    """Three methods that all rank patterns identically."""
    patterns = ["P0", "P1", "P2", "P3", "P4", "P5"]
    # All methods agree: P4 > P3 > P5 > P0 > P1 > P2
    scores = {"P0": 0.458, "P1": 0.437, "P2": 0.407, "P3": 0.505, "P4": 0.557, "P5": 0.489}
    return [
        MethodRanking(method_name="method_a", pattern_names=["P4", "P3", "P5", "P0", "P1", "P2"], scores=dict(scores)),
        MethodRanking(method_name="method_b", pattern_names=["P4", "P3", "P5", "P0", "P1", "P2"], scores=dict(scores)),
        MethodRanking(method_name="method_c", pattern_names=["P4", "P3", "P5", "P0", "P1", "P2"], scores=dict(scores)),
    ]


@pytest.fixture
def disagreement_rankings():
    """Three methods with very different rankings."""
    return [
        MethodRanking(
            method_name="method_a",
            pattern_names=["P0", "P1", "P2", "P3", "P4", "P5"],
            scores={"P0": 0.9, "P1": 0.8, "P2": 0.7, "P3": 0.6, "P4": 0.5, "P5": 0.4},
        ),
        MethodRanking(
            method_name="method_b",
            pattern_names=["P5", "P4", "P3", "P2", "P1", "P0"],
            scores={"P0": 0.4, "P1": 0.5, "P2": 0.6, "P3": 0.7, "P4": 0.8, "P5": 0.9},
        ),
        MethodRanking(
            method_name="method_c",
            pattern_names=["P2", "P0", "P4", "P1", "P5", "P3"],
            scores={"P0": 0.8, "P1": 0.6, "P2": 0.9, "P3": 0.4, "P4": 0.7, "P5": 0.5},
        ),
    ]


@pytest.fixture
def project_rankings():
    """Rankings mimicking the actual project data: 3 methods, 6 patterns.

    Based on project memory:
    - LLM judge: P4(0.557) > P3(0.505) > P5(0.489) > P0(0.458) > P1(0.437) > P2(0.407)
    - Manual rubric (hypothetical): P3 > P4 > P0 > P5 > P1 > P2
    - Keyword matching (hypothetical): P0 > P1 > P4 > P3 > P2 > P5
    """
    return [
        MethodRanking(
            method_name="llm_judge",
            pattern_names=["P4", "P3", "P5", "P0", "P1", "P2"],
            scores={"P0": 0.458, "P1": 0.437, "P2": 0.407, "P3": 0.505, "P4": 0.557, "P5": 0.489},
        ),
        MethodRanking(
            method_name="manual_rubric",
            pattern_names=["P3", "P4", "P0", "P5", "P1", "P2"],
            scores={"P0": 0.50, "P1": 0.40, "P2": 0.35, "P3": 0.65, "P4": 0.60, "P5": 0.45},
        ),
        MethodRanking(
            method_name="keyword_matching",
            pattern_names=["P0", "P1", "P4", "P3", "P2", "P5"],
            scores={"P0": 0.70, "P1": 0.65, "P2": 0.45, "P3": 0.50, "P4": 0.55, "P5": 0.30},
        ),
    ]


# ---------------------------------------------------------------------------
# 1. MethodRanking construction
# ---------------------------------------------------------------------------

class TestMethodRanking:
    def test_construction(self):
        mr = MethodRanking(
            method_name="llm_judge",
            pattern_names=["P4", "P3", "P5"],
            scores={"P4": 0.557, "P3": 0.505, "P5": 0.489},
        )
        assert mr.method_name == "llm_judge"
        assert len(mr.pattern_names) == 3
        assert mr.scores["P4"] == 0.557

    def test_empty_scores(self):
        mr = MethodRanking(
            method_name="empty",
            pattern_names=[],
            scores={},
        )
        assert mr.method_name == "empty"
        assert len(mr.scores) == 0


# ---------------------------------------------------------------------------
# 2. Perfect agreement (W should be 1.0 or very close)
# ---------------------------------------------------------------------------

class TestPerfectAgreement:
    def test_kendalls_w_perfect(self, perfect_agreement_rankings):
        result = analyze_concordance(perfect_agreement_rankings)
        assert result.kendalls_w == pytest.approx(1.0, abs=0.01)

    def test_all_tau_perfect(self, perfect_agreement_rankings):
        result = analyze_concordance(perfect_agreement_rankings)
        for key, tau in result.pairwise_tau.items():
            assert tau == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Complete disagreement (low W)
# ---------------------------------------------------------------------------

class TestDisagreement:
    def test_kendalls_w_low(self, disagreement_rankings):
        result = analyze_concordance(disagreement_rankings)
        # With nearly reversed rankings, W should be low
        assert result.kendalls_w < 0.5

    def test_negative_tau_for_reversed(self, disagreement_rankings):
        result = analyze_concordance(disagreement_rankings)
        # method_a and method_b are exact reverses -> tau should be -1
        key = "method_a vs method_b"
        assert result.pairwise_tau[key] == pytest.approx(-1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Known project rankings (3 methods, 6 patterns)
# ---------------------------------------------------------------------------

class TestProjectRankings:
    def test_concordance_is_moderate(self, project_rankings):
        result = analyze_concordance(project_rankings)
        # With partial agreement, W should be moderate
        assert 0.0 < result.kendalls_w < 1.0

    def test_rank_changes_populated(self, project_rankings):
        result = analyze_concordance(project_rankings)
        assert "P0" in result.rank_changes
        assert "P4" in result.rank_changes
        assert "llm_judge" in result.rank_changes["P0"]
        assert "manual_rubric" in result.rank_changes["P0"]

    def test_summary_nonempty(self, project_rankings):
        result = analyze_concordance(project_rankings)
        assert len(result.summary) > 0

    def test_methods_preserved(self, project_rankings):
        result = analyze_concordance(project_rankings)
        assert len(result.methods) == 3


# ---------------------------------------------------------------------------
# 5. identify_dimension_drivers
# ---------------------------------------------------------------------------

class TestDimensionDrivers:
    def test_known_driver(self):
        """When one dimension shows large variance in differences, it should be identified."""
        method_dim_scores = {
            "method_a": {
                "P0": {"coverage": 0.8, "depth": 0.5, "citations": 0.3},
                "P1": {"coverage": 0.6, "depth": 0.7, "citations": 0.2},
                "P2": {"coverage": 0.7, "depth": 0.6, "citations": 0.4},
            },
            "method_b": {
                "P0": {"coverage": 0.3, "depth": 0.5, "citations": 0.3},
                "P1": {"coverage": 0.9, "depth": 0.7, "citations": 0.2},
                "P2": {"coverage": 0.2, "depth": 0.6, "citations": 0.4},
            },
        }
        drivers = identify_dimension_drivers(method_dim_scores)
        assert "method_a vs method_b" in drivers
        # Coverage has the most variable offset across patterns
        assert drivers["method_a vs method_b"] == "coverage"

    def test_single_method_returns_empty(self):
        drivers = identify_dimension_drivers({"method_a": {"P0": {"x": 1.0}}})
        assert drivers == {}

    def test_multiple_pairs(self):
        method_dim_scores = {
            "m1": {"P0": {"d1": 0.5, "d2": 0.5}, "P1": {"d1": 0.5, "d2": 0.5}},
            "m2": {"P0": {"d1": 0.5, "d2": 0.5}, "P1": {"d1": 0.5, "d2": 0.5}},
            "m3": {"P0": {"d1": 0.5, "d2": 0.5}, "P1": {"d1": 0.5, "d2": 0.5}},
        }
        drivers = identify_dimension_drivers(method_dim_scores)
        assert len(drivers) == 3  # 3 pairs from 3 methods


# ---------------------------------------------------------------------------
# 6. generate_concordance_report produces valid markdown
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_has_header(self, project_rankings):
        result = analyze_concordance(project_rankings)
        report = generate_concordance_report(result)
        assert "# Evaluation Method Concordance Analysis" in report

    def test_report_has_kendalls_w(self, project_rankings):
        result = analyze_concordance(project_rankings)
        report = generate_concordance_report(result)
        assert "Kendall's W" in report

    def test_report_has_table(self, project_rankings):
        result = analyze_concordance(project_rankings)
        report = generate_concordance_report(result)
        assert "|" in report
        assert "Pattern" in report

    def test_report_has_stability(self, project_rankings):
        result = analyze_concordance(project_rankings)
        report = generate_concordance_report(result)
        assert "Most stable pattern" in report
        assert "Most volatile pattern" in report

    def test_report_has_pairwise(self, project_rankings):
        result = analyze_concordance(project_rankings)
        report = generate_concordance_report(result)
        assert "Pairwise Method Correlations" in report
        assert "Kendall's tau" in report


# ---------------------------------------------------------------------------
# 7. most_stable / most_volatile pattern identification
# ---------------------------------------------------------------------------

class TestStabilityIdentification:
    def test_perfect_agreement_stability(self, perfect_agreement_rankings):
        result = analyze_concordance(perfect_agreement_rankings)
        # When all methods agree, all rank variances are 0
        # most_stable could be any pattern; the important thing is variance = 0
        for pattern, method_ranks in result.rank_changes.items():
            ranks = list(method_ranks.values())
            assert len(set(ranks)) == 1  # all ranks identical

    def test_volatile_pattern_has_higher_variance(self, project_rankings):
        result = analyze_concordance(project_rankings)

        # Compute variances for stable vs volatile
        stable_ranks = list(result.rank_changes[result.most_stable_pattern].values())
        volatile_ranks = list(result.rank_changes[result.most_volatile_pattern].values())

        stable_var = np.var(stable_ranks)
        volatile_var = np.var(volatile_ranks)

        assert volatile_var >= stable_var

    def test_disagreement_has_volatile_pattern(self, disagreement_rankings):
        result = analyze_concordance(disagreement_rankings)
        # With reversed rankings, some patterns must be volatile
        volatile_ranks = list(result.rank_changes[result.most_volatile_pattern].values())
        assert max(volatile_ranks) - min(volatile_ranks) > 0

    def test_error_on_single_method(self):
        with pytest.raises(ValueError, match="at least 2"):
            analyze_concordance([
                MethodRanking(
                    method_name="only_one",
                    pattern_names=["P0", "P1"],
                    scores={"P0": 0.5, "P1": 0.6},
                )
            ])
