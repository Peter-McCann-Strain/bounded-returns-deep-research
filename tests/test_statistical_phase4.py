"""Tests for Phase 4 statistical additions: variance_decomposition, pareto_frontier, failure_clustering.

Covers:
- variance_decomposition: system-dominant, query-dominant, constant, SS summation,
  metadata by difficulty, percentages sum to 100
- pareto_frontier: one dominates all, all Pareto optimal, mixed minimize/maximize,
  single system, two-objective known frontier
- failure_clustering: basic clustering, category correlations, empty input
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_research.evaluation.statistical_analysis import (
    pareto_frontier,
    variance_decomposition,
)
from deep_research.evaluation.error_analysis import (
    PatternErrorProfile,
    failure_clustering,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern_error_profile(
    pattern: str,
    category_distribution: dict[str, float],
    n_reports: int = 10,
) -> PatternErrorProfile:
    """Create a PatternErrorProfile with the given category distribution."""
    return PatternErrorProfile(
        pattern=pattern,
        n_reports=n_reports,
        avg_errors_per_report=sum(category_distribution.values()) * 5,
        category_distribution=category_distribution,
        severity_distribution={"critical": 0.3, "moderate": 0.5, "minor": 0.2},
        most_common_errors=sorted(
            category_distribution.items(), key=lambda x: -x[1]
        ),
        failure_modes=[f"{pattern}: dominant mode"],
    )


# ===========================================================================
# variance_decomposition tests
# ===========================================================================


class TestVarianceDecomposition:
    """Tests for variance_decomposition()."""

    def test_system_dominant_variance(self):
        """When systems have very different means and queries are similar,
        system variance should dominate."""
        rng = np.random.RandomState(42)
        n_sys, n_q = 5, 30
        # Systems have means spread far apart; per-query noise is tiny
        system_means = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mat = np.tile(system_means.reshape(-1, 1), (1, n_q))
        mat = mat + rng.normal(0, 0.01, (n_sys, n_q))

        names = [f"S{i}" for i in range(n_sys)]
        qids = [f"q{j}" for j in range(n_q)]

        result = variance_decomposition(mat, names, qids)

        assert result["system_pct"] > 90.0, (
            f"Expected system_pct > 90%, got {result['system_pct']:.1f}%"
        )
        assert result["ss_total"] > 0

    def test_query_dominant_variance(self):
        """When queries differ a lot but systems score similarly on each query,
        query variance should dominate."""
        rng = np.random.RandomState(123)
        n_sys, n_q = 5, 30
        # Queries have means spread far apart; systems score nearly the same on each
        query_means = np.linspace(0.1, 0.9, n_q)
        mat = np.tile(query_means.reshape(1, -1), (n_sys, 1))
        mat = mat + rng.normal(0, 0.01, (n_sys, n_q))

        names = [f"S{i}" for i in range(n_sys)]
        qids = [f"q{j}" for j in range(n_q)]

        result = variance_decomposition(mat, names, qids)

        assert result["query_pct"] > 90.0, (
            f"Expected query_pct > 90%, got {result['query_pct']:.1f}%"
        )

    def test_constant_data(self):
        """All same score means zero variance everywhere."""
        mat = np.full((3, 10), 0.42)
        names = ["A", "B", "C"]
        qids = [f"q{i}" for i in range(10)]

        result = variance_decomposition(mat, names, qids)

        assert result["ss_total"] == pytest.approx(0.0, abs=1e-10)
        assert result["ss_system"] == pytest.approx(0.0, abs=1e-10)
        assert result["ss_query"] == pytest.approx(0.0, abs=1e-10)
        assert result["ss_residual"] == pytest.approx(0.0, abs=1e-10)
        assert result["system_pct"] == pytest.approx(0.0, abs=1e-10)
        assert result["query_pct"] == pytest.approx(0.0, abs=1e-10)
        assert result["residual_pct"] == pytest.approx(0.0, abs=1e-10)
        assert result["grand_mean"] == pytest.approx(0.42)

    def test_components_sum(self):
        """SS_system + SS_query + SS_residual should equal SS_total."""
        rng = np.random.RandomState(99)
        mat = rng.uniform(0, 1, (4, 15))
        names = [f"S{i}" for i in range(4)]
        qids = [f"q{j}" for j in range(15)]

        result = variance_decomposition(mat, names, qids)

        reconstructed = result["ss_system"] + result["ss_query"] + result["ss_residual"]
        assert result["ss_total"] == pytest.approx(reconstructed, abs=1e-10)

    def test_with_metadata_by_difficulty(self):
        """Providing difficulty metadata should populate by_difficulty."""
        rng = np.random.RandomState(7)
        n_sys, n_q = 3, 12
        mat = rng.uniform(0, 1, (n_sys, n_q))
        names = ["A", "B", "C"]
        qids = [f"q{j}" for j in range(n_q)]

        # First 4 queries easy, next 4 medium, last 4 hard
        metadata = {}
        for j in range(n_q):
            if j < 4:
                metadata[qids[j]] = {"difficulty": "easy"}
            elif j < 8:
                metadata[qids[j]] = {"difficulty": "medium"}
            else:
                metadata[qids[j]] = {"difficulty": "hard"}

        result = variance_decomposition(mat, names, qids, query_metadata=metadata)

        assert result["by_difficulty"] is not None
        assert "easy" in result["by_difficulty"]
        assert "medium" in result["by_difficulty"]
        assert "hard" in result["by_difficulty"]
        for level_info in result["by_difficulty"].values():
            assert "n_queries" in level_info
            assert "mean_score" in level_info
            assert "ss_query_within" in level_info
            assert level_info["n_queries"] == 4

    def test_percentages_sum_to_100(self):
        """system_pct + query_pct + residual_pct should be approximately 100."""
        rng = np.random.RandomState(55)
        mat = rng.uniform(0, 1, (6, 20))
        names = [f"S{i}" for i in range(6)]
        qids = [f"q{j}" for j in range(20)]

        result = variance_decomposition(mat, names, qids)

        total_pct = result["system_pct"] + result["query_pct"] + result["residual_pct"]
        assert total_pct == pytest.approx(100.0, abs=1e-8)

    def test_mismatched_names_raises(self):
        """Wrong number of system names should raise ValueError."""
        mat = np.ones((3, 5))
        with pytest.raises(ValueError, match="system_names length"):
            variance_decomposition(mat, ["A", "B"], [f"q{i}" for i in range(5)])

    def test_mismatched_query_ids_raises(self):
        """Wrong number of query ids should raise ValueError."""
        mat = np.ones((3, 5))
        with pytest.raises(ValueError, match="query_ids length"):
            variance_decomposition(mat, ["A", "B", "C"], [f"q{i}" for i in range(3)])

    def test_by_domain_populated(self):
        """Providing domain metadata should populate by_domain."""
        rng = np.random.RandomState(88)
        mat = rng.uniform(0, 1, (2, 6))
        names = ["X", "Y"]
        qids = [f"q{j}" for j in range(6)]
        metadata = {
            "q0": {"domain": "science"},
            "q1": {"domain": "science"},
            "q2": {"domain": "history"},
            "q3": {"domain": "history"},
            "q4": {"domain": "math"},
            "q5": {"domain": "math"},
        }

        result = variance_decomposition(mat, names, qids, query_metadata=metadata)

        assert result["by_domain"] is not None
        assert set(result["by_domain"].keys()) == {"science", "history", "math"}

    def test_no_metadata_returns_none(self):
        """Without metadata, by_difficulty and by_domain should be None."""
        mat = np.ones((2, 3))
        result = variance_decomposition(mat, ["A", "B"], ["q0", "q1", "q2"])
        assert result["by_difficulty"] is None
        assert result["by_domain"] is None


# ===========================================================================
# pareto_frontier tests
# ===========================================================================


class TestParetoFrontier:
    """Tests for pareto_frontier()."""

    def test_one_dominates_all(self):
        """System A is best on everything -- it alone is Pareto-optimal."""
        metrics = {
            "A": {"score": 0.9, "cost_usd": 0.5, "tokens": 1000},
            "B": {"score": 0.5, "cost_usd": 1.0, "tokens": 2000},
            "C": {"score": 0.3, "cost_usd": 2.0, "tokens": 3000},
        }
        result = pareto_frontier(
            metrics,
            objectives=["score", "cost_usd", "tokens"],
            minimize=["cost_usd", "tokens"],
        )

        assert result["pareto_optimal"] == ["A"]
        assert "B" in result["dominated"]
        assert "C" in result["dominated"]
        # A should dominate both B and C
        assert "A" in result["dominated"]["B"]
        assert "A" in result["dominated"]["C"]

    def test_all_pareto_optimal(self):
        """No system dominates any other -- all are on the frontier."""
        metrics = {
            "A": {"score": 0.9, "cost_usd": 3.0},
            "B": {"score": 0.5, "cost_usd": 1.0},
            "C": {"score": 0.7, "cost_usd": 2.0},
        }
        result = pareto_frontier(
            metrics,
            objectives=["score", "cost_usd"],
            minimize=["cost_usd"],
        )

        assert sorted(result["pareto_optimal"]) == ["A", "B", "C"]
        assert result["dominated"] == {}

    def test_mixed_minimize_maximize(self):
        """Some metrics minimized, others maximized."""
        metrics = {
            "Fast_Bad": {"quality": 0.3, "speed_ms": 10, "cost_usd": 0.1},
            "Slow_Good": {"quality": 0.9, "speed_ms": 500, "cost_usd": 5.0},
            "Mid": {"quality": 0.6, "speed_ms": 100, "cost_usd": 1.0},
        }
        result = pareto_frontier(
            metrics,
            objectives=["quality", "speed_ms", "cost_usd"],
            minimize=["speed_ms", "cost_usd"],
        )

        # No system dominates all others on all objectives
        # Fast_Bad: best speed and cost, worst quality
        # Slow_Good: best quality, worst speed and cost
        # Mid: in between on all
        # Fast_Bad dominates Mid? quality: 0.3 < 0.6 (worse), so no.
        # All should be Pareto-optimal since each has a unique advantage
        assert sorted(result["pareto_optimal"]) == ["Fast_Bad", "Mid", "Slow_Good"]

    def test_single_system(self):
        """A single system is trivially Pareto-optimal."""
        metrics = {"OnlyOne": {"score": 0.5, "cost_usd": 1.0}}
        result = pareto_frontier(metrics, objectives=["score", "cost_usd"])

        assert result["pareto_optimal"] == ["OnlyOne"]
        assert result["dominated"] == {}
        assert result["efficiency_scores"]["OnlyOne"] == pytest.approx(1.0)

    def test_two_objectives(self):
        """Simple 2D case with a known frontier.

        A: (score=0.9, cost=0.5) -- best on both, dominates all
        B: (score=0.5, cost=0.5) -- same cost as A but worse score
        C: (score=0.9, cost=1.0) -- same score as A but worse cost
        """
        metrics = {
            "A": {"score": 0.9, "cost_usd": 0.5},
            "B": {"score": 0.5, "cost_usd": 0.5},
            "C": {"score": 0.9, "cost_usd": 1.0},
        }
        result = pareto_frontier(
            metrics,
            objectives=["score", "cost_usd"],
            minimize=["cost_usd"],
        )

        assert result["pareto_optimal"] == ["A"]
        assert "B" in result["dominated"]
        assert "C" in result["dominated"]

    def test_empty_input(self):
        """Empty system_metrics should return empty results."""
        result = pareto_frontier({})
        assert result["pareto_optimal"] == []
        assert result["dominated"] == {}
        assert result["efficiency_scores"] == {}

    def test_efficiency_scores_bounded(self):
        """All efficiency scores should be in [0, 1]."""
        metrics = {
            "A": {"x": 0.1, "y": 0.9},
            "B": {"x": 0.5, "y": 0.5},
            "C": {"x": 0.9, "y": 0.1},
        }
        result = pareto_frontier(metrics, objectives=["x", "y"], minimize=[])

        for name, eff in result["efficiency_scores"].items():
            assert 0.0 <= eff <= 1.0, f"Efficiency for {name} out of bounds: {eff}"

    def test_identical_systems(self):
        """Identical systems are all Pareto-optimal (no domination)."""
        metrics = {
            "A": {"score": 0.5, "cost_usd": 1.0},
            "B": {"score": 0.5, "cost_usd": 1.0},
        }
        result = pareto_frontier(
            metrics, objectives=["score", "cost_usd"], minimize=["cost_usd"]
        )

        assert sorted(result["pareto_optimal"]) == ["A", "B"]
        assert result["dominated"] == {}

    def test_default_minimize(self):
        """Without explicit minimize, default minimize list is used."""
        metrics = {
            "A": {"score": 0.9, "cost_usd": 0.1, "tokens": 100, "elapsed_seconds": 1},
            "B": {"score": 0.5, "cost_usd": 0.5, "tokens": 500, "elapsed_seconds": 5},
        }
        result = pareto_frontier(metrics)

        # A dominates B: higher score, lower cost, fewer tokens, less time
        assert result["pareto_optimal"] == ["A"]


# ===========================================================================
# failure_clustering tests
# ===========================================================================


class TestFailureClustering:
    """Tests for failure_clustering()."""

    def test_basic_clustering(self):
        """Patterns with similar category distributions should cluster together."""
        # Two "citation-heavy" patterns and one "hallucination-heavy" pattern
        profiles = [
            _make_pattern_error_profile(
                "P0", {"citation_fabrication": 0.6, "attribution_error": 0.3, "factual_error": 0.1}
            ),
            _make_pattern_error_profile(
                "P1", {"citation_fabrication": 0.5, "attribution_error": 0.4, "factual_error": 0.1}
            ),
            _make_pattern_error_profile(
                "P2", {"hallucination": 0.7, "topic_drift": 0.2, "factual_error": 0.1}
            ),
        ]

        result = failure_clustering(profiles, n_clusters=2)

        assert "clusters" in result
        assert len(result["clusters"]) == 2

        # P0 and P1 should be in the same cluster
        for cluster_id, members in result["clusters"].items():
            if "P0" in members:
                assert "P1" in members, "P0 and P1 should cluster together"
                assert "P2" not in members

    def test_category_correlations(self):
        """Co-occurrence rates should be computed for all category pairs."""
        profiles = [
            _make_pattern_error_profile(
                "P0", {"hallucination": 0.5, "factual_error": 0.5}
            ),
            _make_pattern_error_profile(
                "P1", {"hallucination": 0.3, "factual_error": 0.7}
            ),
            _make_pattern_error_profile(
                "P2", {"hallucination": 0.4, "citation_fabrication": 0.6}
            ),
        ]

        result = failure_clustering(profiles, n_clusters=3)

        correlations = result["category_correlations"]
        # hallucination + factual_error co-occur in P0, P1 (2 out of 3 patterns)
        key = ("factual_error", "hallucination")
        assert key in correlations
        assert correlations[key] == pytest.approx(2.0 / 3.0)

        # factual_error + citation_fabrication co-occur in 0 patterns
        key2 = ("citation_fabrication", "factual_error")
        assert key2 in correlations
        assert correlations[key2] == pytest.approx(0.0)

    def test_empty_input(self):
        """Empty input should return empty results, not raise."""
        result = failure_clustering([], n_clusters=3)

        assert result["clusters"] == {}
        assert result["cross_pattern_failures"] == []
        assert result["pattern_specific_failures"] == {}
        assert result["category_correlations"] == {}

    def test_cross_pattern_failures(self):
        """Categories present in all patterns should appear in cross_pattern_failures."""
        profiles = [
            _make_pattern_error_profile(
                "P0", {"hallucination": 0.3, "factual_error": 0.7}
            ),
            _make_pattern_error_profile(
                "P1", {"hallucination": 0.5, "factual_error": 0.3, "topic_drift": 0.2}
            ),
            _make_pattern_error_profile(
                "P2", {"hallucination": 0.2, "factual_error": 0.8}
            ),
        ]

        result = failure_clustering(profiles, n_clusters=3)

        # hallucination and factual_error are in all three
        assert "hallucination" in result["cross_pattern_failures"]
        assert "factual_error" in result["cross_pattern_failures"]
        # topic_drift is only in P1
        assert "topic_drift" not in result["cross_pattern_failures"]

    def test_pattern_specific_failures(self):
        """Categories unique to a single pattern should appear in pattern_specific_failures."""
        profiles = [
            _make_pattern_error_profile(
                "P0", {"hallucination": 0.5, "factual_error": 0.5}
            ),
            _make_pattern_error_profile(
                "P1", {"hallucination": 0.5, "source_quality": 0.5}
            ),
        ]

        result = failure_clustering(profiles, n_clusters=2)

        # factual_error is unique to P0
        assert "factual_error" in result["pattern_specific_failures"].get("P0", [])
        # source_quality is unique to P1
        assert "source_quality" in result["pattern_specific_failures"].get("P1", [])
        # hallucination is in both, so not pattern-specific
        for name, specifics in result["pattern_specific_failures"].items():
            assert "hallucination" not in specifics

    def test_single_profile(self):
        """A single profile should form one cluster."""
        profiles = [
            _make_pattern_error_profile("P0", {"hallucination": 1.0}),
        ]

        result = failure_clustering(profiles, n_clusters=3)

        assert len(result["clusters"]) == 1
        members = list(result["clusters"].values())[0]
        assert members == ["P0"]

    def test_n_clusters_capped(self):
        """n_clusters should be capped at the number of profiles."""
        profiles = [
            _make_pattern_error_profile("P0", {"hallucination": 0.5, "factual_error": 0.5}),
            _make_pattern_error_profile("P1", {"citation_fabrication": 1.0}),
        ]

        result = failure_clustering(profiles, n_clusters=10)

        # Should have at most 2 clusters (one per profile)
        assert len(result["clusters"]) == 2
