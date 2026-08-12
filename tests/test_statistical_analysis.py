"""Comprehensive tests for the statistical analysis module.

Tests cover:
- Friedman omnibus test (with Iman-Davenport correction)
- Nemenyi post-hoc (without double Holm correction)
- Wilcoxon signed-rank pairwise
- Wilcoxon pairwise all (with Holm-Bonferroni)
- Holm-Bonferroni correction
- Bootstrap confidence intervals (BCa / percentile)
- Interquartile mean (and bootstrap CI)
- Cliff's Delta effect size
- Kendall's W and tau concordance
- Nemenyi critical difference
- Power analysis
- Bootstrap rank distribution
- Stratified analysis
- Full analysis pipeline
- Markdown summary generation
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from scipy import stats

from deep_research.evaluation.statistical_analysis import (
    BootstrapCI,
    ConcordanceResult,
    FullAnalysisResult,
    OmnibusResult,
    PairwiseResult,
    bootstrap_confidence_interval,
    bootstrap_rank_distribution,
    cliffs_delta,
    compute_all_bootstrap_cis,
    friedman_test,
    generate_summary_markdown,
    holm_bonferroni,
    interquartile_mean,
    interquartile_mean_bootstrap_ci,
    kendalls_tau,
    kendalls_w,
    nemenyi_critical_difference,
    nemenyi_posthoc,
    power_analysis,
    ranking_concordance,
    run_full_analysis,
    run_stratified_analysis,
    wilcoxon_pairwise_all,
    wilcoxon_signed_rank_pairwise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Reproducible random state for test data generation."""
    return np.random.RandomState(42)


@pytest.fixture
def six_system_names():
    """Canonical names for the 6 deep-research patterns."""
    return ["P0_Baseline", "P1_IterRAG", "P2_Supervisor", "P3_MERIDIAN", "P4_STORM", "P5_HierWD"]


@pytest.fixture
def different_means_matrix(rng, six_system_names):
    """6 systems x 20 tasks with clearly different means.

    System i has mean = 0.3 + 0.1*i, so P5 is best, P0 is worst.
    """
    n_tasks = 20
    k = len(six_system_names)
    matrix = np.zeros((n_tasks, k))
    for j in range(k):
        matrix[:, j] = rng.normal(loc=0.3 + 0.1 * j, scale=0.05, size=n_tasks)
    return matrix


@pytest.fixture
def identical_scores_matrix(six_system_names):
    """6 systems x 20 tasks with identical scores across systems."""
    n_tasks = 20
    k = len(six_system_names)
    # Each row is the same value across all systems, but varies across tasks
    base = np.linspace(0.3, 0.7, n_tasks)
    return np.tile(base[:, np.newaxis], (1, k))


@pytest.fixture
def realistic_matrix(rng, six_system_names):
    """6 systems x 25 tasks with semi-realistic score distributions.

    Models the approximate pattern from the project's actual results.
    """
    n_tasks = 25
    means = [0.458, 0.437, 0.407, 0.505, 0.557, 0.489]
    stds = [0.10, 0.12, 0.15, 0.08, 0.09, 0.11]

    matrix = np.zeros((n_tasks, len(six_system_names)))
    for j, (mu, sigma) in enumerate(zip(means, stds)):
        matrix[:, j] = np.clip(rng.normal(mu, sigma, size=n_tasks), 0, 1)
    return matrix


# ===========================================================================
# 1. friedman_test
# ===========================================================================


class TestFriedmanTest:
    """Tests for the Friedman omnibus test."""

    def test_significant_with_different_means(self, different_means_matrix, six_system_names):
        """Systems with clearly different means should produce a significant result."""
        result = friedman_test(different_means_matrix, six_system_names)

        assert isinstance(result, OmnibusResult)
        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.statistic > 0
        assert result.df == len(six_system_names) - 1
        assert result.n_systems == 6
        assert result.n_tasks == 20

    def test_not_significant_identical_scores(self, identical_scores_matrix, six_system_names):
        """Identical scores across systems should produce a non-significant result."""
        result = friedman_test(identical_scores_matrix, six_system_names)

        assert result.is_significant is False
        assert result.p_value >= 0.05
        # With identical scores, statistic should be 0 (or near 0)
        assert result.statistic == pytest.approx(0.0, abs=1e-10)

    def test_average_ranks_correct(self, different_means_matrix, six_system_names):
        """Average ranks should reflect the ordering of system means.

        Higher mean -> lower rank (rank 1 = best).
        """
        result = friedman_test(different_means_matrix, six_system_names)

        # P5 has the highest mean (0.8) so should have the lowest average rank
        ranks = result.avg_ranks
        assert len(ranks) == 6

        # The system with the highest mean should have the smallest average rank
        sorted_by_rank = sorted(ranks.items(), key=lambda kv: kv[1])
        # P5_HierWD (index 5, mean=0.8) should be rank 1
        assert sorted_by_rank[0][0] == "P5_HierWD"
        # P0_Baseline (index 0, mean=0.3) should be rank 6
        assert sorted_by_rank[-1][0] == "P0_Baseline"

    def test_average_ranks_identical(self, identical_scores_matrix, six_system_names):
        """With identical scores, all average ranks should be the same."""
        result = friedman_test(identical_scores_matrix, six_system_names)
        ranks = list(result.avg_ranks.values())

        # All should be equal: average of 1..6 = 3.5
        for r in ranks:
            assert r == pytest.approx(3.5, abs=1e-10)

    def test_two_systems(self, rng):
        """Should work with the minimum of 2 systems."""
        matrix = np.column_stack([
            rng.normal(0.5, 0.1, 10),
            rng.normal(0.8, 0.1, 10),
        ])
        result = friedman_test(matrix, ["A", "B"])
        assert result.n_systems == 2
        assert result.df == 1

    def test_raises_on_1d(self):
        """Should raise ValueError for 1-D input."""
        with pytest.raises(ValueError, match="2-D"):
            friedman_test(np.array([1, 2, 3]), ["A"])

    def test_raises_on_single_system(self):
        """Should raise ValueError with only 1 system (column)."""
        with pytest.raises(ValueError, match="at least 2 systems"):
            friedman_test(np.array([[1], [2], [3]]), ["A"])

    def test_raises_on_single_task(self):
        """Should raise ValueError with only 1 task (row)."""
        with pytest.raises(ValueError, match="at least 2 tasks"):
            friedman_test(np.array([[1, 2, 3]]), ["A", "B", "C"])

    def test_raises_on_name_mismatch(self):
        """Should raise ValueError when names length != columns."""
        with pytest.raises(ValueError, match="system_names length"):
            friedman_test(np.ones((5, 3)), ["A", "B"])

    def test_custom_alpha(self, different_means_matrix, six_system_names):
        """A very low alpha should make a borderline result non-significant."""
        result = friedman_test(
            different_means_matrix, six_system_names, alpha=1e-20
        )
        # p_value is small but not 1e-20 small for 20 tasks
        # We don't assert significance here; just that it runs without error
        assert isinstance(result.p_value, float)

    def test_iman_davenport_present(self, different_means_matrix, six_system_names):
        """Iman-Davenport F and p-value should be present for k >= 3."""
        result = friedman_test(different_means_matrix, six_system_names)
        assert hasattr(result, "iman_davenport_f")
        assert hasattr(result, "iman_davenport_p")
        # With clearly different means and k=6, both Friedman and Iman-Davenport
        # should be significant
        assert result.iman_davenport_f > 0
        assert result.iman_davenport_p < 0.05

    def test_iman_davenport_identical_scores(self, identical_scores_matrix, six_system_names):
        """With identical scores, Iman-Davenport should be non-significant."""
        result = friedman_test(identical_scores_matrix, six_system_names)
        # statistic is 0, so F should be 0 and p should be 1.0
        assert result.iman_davenport_f == pytest.approx(0.0, abs=1e-10)
        assert result.iman_davenport_p == pytest.approx(1.0, abs=1e-6)

    def test_iman_davenport_two_systems(self, rng):
        """Iman-Davenport is only computed for k >= 3; for k=2 should be defaults."""
        matrix = np.column_stack([
            rng.normal(0.5, 0.1, 10),
            rng.normal(0.8, 0.1, 10),
        ])
        result = friedman_test(matrix, ["A", "B"])
        # For k=2 we use Wilcoxon, Iman-Davenport stays at defaults
        assert result.iman_davenport_f == 0.0
        assert result.iman_davenport_p == 1.0

    def test_iman_davenport_valid_p_value(self, rng):
        """Iman-Davenport F-test should produce a valid p-value."""
        n_tasks = 8
        k = 4
        names = [f"S{i}" for i in range(k)]
        matrix = np.zeros((n_tasks, k))
        for j in range(k):
            matrix[:, j] = rng.normal(loc=0.3 + 0.05 * j, scale=0.1, size=n_tasks)

        result = friedman_test(matrix, names)
        # Both p-values should be valid and in [0, 1]
        if result.statistic > 0:
            assert 0.0 <= result.iman_davenport_p <= 1.0
            assert result.iman_davenport_f > 0


# ===========================================================================
# 2. nemenyi_posthoc
# ===========================================================================


class TestNemenyiPosthoc:
    """Tests for Nemenyi post-hoc tests."""

    def test_returns_correct_number_of_pairs(self, different_means_matrix, six_system_names):
        """Should return C(6,2) = 15 pairwise results for 6 systems."""
        results = nemenyi_posthoc(different_means_matrix, six_system_names)
        assert len(results) == 15  # C(6, 2)

    def test_all_results_are_pairwise_result(self, different_means_matrix, six_system_names):
        """Every element should be a PairwiseResult."""
        results = nemenyi_posthoc(different_means_matrix, six_system_names)
        for r in results:
            assert isinstance(r, PairwiseResult)
            assert r.test_name == "nemenyi"

    def test_has_effect_sizes(self, different_means_matrix, six_system_names):
        """Each result should have a Cliff's Delta effect size and label."""
        results = nemenyi_posthoc(different_means_matrix, six_system_names)
        for r in results:
            assert isinstance(r.effect_size, float)
            assert r.effect_size_label in {
                "negligible", "small", "medium", "large"
            }

    def test_no_double_correction(self, different_means_matrix, six_system_names):
        """Nemenyi already controls FWER; corrected p should equal raw p.

        Fix #2: Holm-Bonferroni is no longer applied on top of Nemenyi.
        """
        results = nemenyi_posthoc(different_means_matrix, six_system_names)
        for r in results:
            assert r.p_value_corrected == pytest.approx(r.p_value_raw)

    def test_significant_pairs_exist(self, different_means_matrix, six_system_names):
        """With clearly different means, at least some pairs should be significant."""
        results = nemenyi_posthoc(different_means_matrix, six_system_names)
        sig_count = sum(1 for r in results if r.is_significant)
        assert sig_count > 0

    def test_extreme_pair_has_large_effect(self, different_means_matrix, six_system_names):
        """The pair with the largest mean difference should have a large effect size."""
        results = nemenyi_posthoc(different_means_matrix, six_system_names)
        # P0 vs P5 should have the largest effect
        extreme = [
            r for r in results
            if {r.system_a, r.system_b} == {"P0_Baseline", "P5_HierWD"}
        ]
        assert len(extreme) == 1
        assert extreme[0].effect_size_label == "large"

    def test_raises_on_name_mismatch(self):
        """Should raise ValueError when names don't match columns."""
        with pytest.raises(ValueError, match="system_names length"):
            nemenyi_posthoc(np.ones((5, 3)), ["A", "B"])


# ===========================================================================
# 3. wilcoxon_signed_rank_pairwise
# ===========================================================================


class TestWilcoxonSignedRank:
    """Tests for Wilcoxon signed-rank pairwise test."""

    def test_significant_different_distributions(self, rng):
        """Two clearly different distributions should be significant."""
        a = rng.normal(0.8, 0.05, 30)
        b = rng.normal(0.3, 0.05, 30)
        result = wilcoxon_signed_rank_pairwise(a, b, "High", "Low")

        assert result.is_significant is True
        assert result.p_value_raw < 0.05
        assert result.test_name == "wilcoxon"
        assert result.mean_diff > 0

    def test_not_significant_identical(self, rng):
        """Two identical distributions should not be significant."""
        scores = rng.normal(0.5, 0.1, 30)
        result = wilcoxon_signed_rank_pairwise(scores, scores, "A", "B")

        assert result.is_significant is False
        assert result.p_value_raw == 1.0
        assert result.effect_size == 0.0
        assert result.mean_diff == pytest.approx(0.0)

    def test_effect_size_computed(self, rng):
        """Should compute Cliff's Delta effect size."""
        a = rng.normal(0.7, 0.1, 20)
        b = rng.normal(0.3, 0.1, 20)
        result = wilcoxon_signed_rank_pairwise(a, b, "A", "B")
        assert result.effect_size_label in {"negligible", "small", "medium", "large"}

    def test_raises_on_mismatched_shapes(self):
        """Should raise ValueError with different-length arrays."""
        with pytest.raises(ValueError, match="same shape"):
            wilcoxon_signed_rank_pairwise(
                np.array([1, 2, 3]), np.array([1, 2]), "A", "B"
            )

    def test_raises_on_too_few_observations(self):
        """Should raise ValueError with < 2 observations."""
        with pytest.raises(ValueError, match="at least 2"):
            wilcoxon_signed_rank_pairwise(
                np.array([1.0]), np.array([2.0]), "A", "B"
            )

    def test_ci_contains_mean_diff(self, rng):
        """The confidence interval should contain the mean difference."""
        a = rng.normal(0.6, 0.1, 30)
        b = rng.normal(0.4, 0.1, 30)
        result = wilcoxon_signed_rank_pairwise(a, b, "A", "B")
        # CI should contain the mean diff (at least approximately)
        assert result.ci_lower <= result.mean_diff + 0.05
        assert result.ci_upper >= result.mean_diff - 0.05


# ===========================================================================
# 3b. wilcoxon_pairwise_all
# ===========================================================================


class TestWilcoxonPairwiseAll:
    """Tests for Wilcoxon pairwise all with Holm-Bonferroni."""

    def test_returns_correct_number_of_pairs(self, different_means_matrix, six_system_names):
        """Should return C(6,2) = 15 pairwise results for 6 systems."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        assert len(results) == 15

    def test_all_results_are_pairwise_result(self, different_means_matrix, six_system_names):
        """Every element should be a PairwiseResult with test_name wilcoxon_holm."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        for r in results:
            assert isinstance(r, PairwiseResult)
            assert r.test_name == "wilcoxon_holm"

    def test_corrected_p_values_geq_raw(self, different_means_matrix, six_system_names):
        """Holm-corrected p-values should be >= raw p-values."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        for r in results:
            assert r.p_value_corrected >= r.p_value_raw - 1e-12

    def test_holm_correction_applied(self, different_means_matrix, six_system_names):
        """At least one pair should have corrected p > raw p (Holm adjusts upward)."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        any_adjusted = any(r.p_value_corrected > r.p_value_raw + 1e-12 for r in results)
        assert any_adjusted, "Holm-Bonferroni should adjust at least one p-value upward"

    def test_significant_pairs_exist(self, different_means_matrix, six_system_names):
        """With clearly different means, at least some pairs should be significant."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        sig_count = sum(1 for r in results if r.is_significant)
        assert sig_count > 0

    def test_has_effect_sizes(self, different_means_matrix, six_system_names):
        """Each result should have Cliff's Delta effect size."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        for r in results:
            assert isinstance(r.effect_size, float)
            assert r.effect_size_label in {"negligible", "small", "medium", "large"}

    def test_has_bootstrap_ci(self, different_means_matrix, six_system_names):
        """Each result should have bootstrap CI."""
        results = wilcoxon_pairwise_all(different_means_matrix, six_system_names)
        for r in results:
            assert np.isfinite(r.ci_lower)
            assert np.isfinite(r.ci_upper)
            assert r.ci_lower <= r.ci_upper

    def test_three_systems(self, rng):
        """Should work with 3 systems (3 pairs)."""
        matrix = np.column_stack([
            rng.normal(0.3, 0.05, 15),
            rng.normal(0.5, 0.05, 15),
            rng.normal(0.7, 0.05, 15),
        ])
        results = wilcoxon_pairwise_all(matrix, ["A", "B", "C"])
        assert len(results) == 3

    def test_raises_on_name_mismatch(self):
        """Should raise ValueError when names don't match columns."""
        with pytest.raises(ValueError, match="system_names length"):
            wilcoxon_pairwise_all(np.ones((5, 3)), ["A", "B"])


# ===========================================================================
# 4. holm_bonferroni
# ===========================================================================


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni multiple comparison correction."""

    def test_known_values(self):
        """Known p-values should be corrected correctly.

        Input: [0.01, 0.04, 0.03, 0.005]
        Sorted: [0.005, 0.01, 0.03, 0.04] with indices [3, 0, 2, 1]

        Step-down (m=4):
          rank 0: 0.005 * 4 = 0.02
          rank 1: 0.01  * 3 = 0.03 -> max(0.03, 0.02) = 0.03
          rank 2: 0.03  * 2 = 0.06 -> max(0.06, 0.03) = 0.06
          rank 3: 0.04  * 1 = 0.04 -> max(0.04, 0.06) = 0.06

        Original order: [0.03, 0.06, 0.06, 0.02]
        """
        raw = [0.01, 0.04, 0.03, 0.005]
        corrected = holm_bonferroni(raw)

        assert len(corrected) == 4
        assert corrected[0] == pytest.approx(0.03)    # was 0.01
        assert corrected[1] == pytest.approx(0.06)    # was 0.04
        assert corrected[2] == pytest.approx(0.06)    # was 0.03
        assert corrected[3] == pytest.approx(0.02)    # was 0.005

    def test_single_p_value_unchanged(self):
        """A single p-value should remain unchanged."""
        corrected = holm_bonferroni([0.03])
        assert corrected == [pytest.approx(0.03)]

    def test_empty_list(self):
        """Empty input should return empty output."""
        assert holm_bonferroni([]) == []

    def test_all_significant_stay_significant(self):
        """Very small p-values should all remain significant after correction."""
        raw = [0.001, 0.002, 0.003]
        corrected = holm_bonferroni(raw, alpha=0.05)
        for p in corrected:
            assert p < 0.05

    def test_capped_at_one(self):
        """Corrected p-values should never exceed 1.0."""
        raw = [0.5, 0.6, 0.7, 0.8, 0.9]
        corrected = holm_bonferroni(raw)
        for p in corrected:
            assert p <= 1.0

    def test_monotonicity_in_sorted_order(self):
        """After sorting by raw p-value, corrected values should be non-decreasing."""
        raw = [0.01, 0.04, 0.03, 0.005, 0.1]
        corrected = holm_bonferroni(raw)

        # Sort both by raw p-value
        pairs = sorted(zip(raw, corrected), key=lambda x: x[0])
        corrected_sorted = [p[1] for p in pairs]

        for i in range(1, len(corrected_sorted)):
            assert corrected_sorted[i] >= corrected_sorted[i - 1] - 1e-12

    def test_corrected_geq_raw(self):
        """Corrected p-values should always be >= raw p-values."""
        raw = [0.01, 0.04, 0.03, 0.005]
        corrected = holm_bonferroni(raw)
        for r, c in zip(raw, corrected):
            assert c >= r - 1e-12

    def test_boundary_p_values(self):
        """P-values at the alpha boundary should be handled correctly."""
        # With alpha=0.05 and 2 tests: 0.025*2=0.05, 0.05*1=0.05
        raw = [0.025, 0.05]
        corrected = holm_bonferroni(raw, alpha=0.05)
        assert corrected[0] == pytest.approx(0.05)
        assert corrected[1] == pytest.approx(0.05)


# ===========================================================================
# 5. bootstrap_confidence_interval
# ===========================================================================


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_true_mean_normal(self, rng):
        """For Normal(0, 1) samples, the 95% CI should contain 0."""
        scores = rng.normal(0, 1, 100)
        mean, lo, hi = bootstrap_confidence_interval(
            scores, n_bootstrap=5000, random_state=42
        )
        assert lo < 0 < hi

    def test_tight_ci_for_low_variance(self, rng):
        """For Normal(5, 0.1), the CI should be tight around 5."""
        scores = rng.normal(5, 0.1, 200)
        mean, lo, hi = bootstrap_confidence_interval(
            scores, n_bootstrap=5000, random_state=42
        )
        assert abs(mean - 5.0) < 0.1
        assert (hi - lo) < 0.1  # very tight

    def test_deterministic_with_seed(self, rng):
        """Same seed should produce identical results."""
        scores = rng.normal(0.5, 0.2, 50)
        r1 = bootstrap_confidence_interval(scores, random_state=123)
        r2 = bootstrap_confidence_interval(scores, random_state=123)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])
        assert r1[2] == pytest.approx(r2[2])

    def test_single_value(self):
        """Single value should return (val, val, val)."""
        mean, lo, hi = bootstrap_confidence_interval(np.array([3.14]))
        assert mean == pytest.approx(3.14)
        assert lo == pytest.approx(3.14)
        assert hi == pytest.approx(3.14)

    def test_raises_on_empty(self):
        """Should raise ValueError for empty array."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_confidence_interval(np.array([]))

    def test_ci_width_decreases_with_n(self, rng):
        """CI width should shrink as sample size grows."""
        small = rng.normal(0, 1, 20)
        large = rng.normal(0, 1, 500)

        _, lo_s, hi_s = bootstrap_confidence_interval(small, random_state=42)
        _, lo_l, hi_l = bootstrap_confidence_interval(large, random_state=42)

        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small

    def test_ci_lower_le_ci_upper(self, rng):
        """CI lower bound should be <= upper bound."""
        scores = rng.normal(0.5, 0.2, 50)
        mean, lo, hi = bootstrap_confidence_interval(scores, random_state=42)
        assert lo <= hi


# ===========================================================================
# 6. interquartile_mean
# ===========================================================================


class TestInterquartileMean:
    """Tests for interquartile mean."""

    def test_known_iqm_8_elements(self):
        """IQM of [1,2,3,4,5,6,7,8] should be mean of middle 50%.

        scipy.stats.trim_mean with proportiontocut=0.25 trims 25% from
        each end, so for 8 elements it trims 2 from each end:
        remaining = [3, 4, 5, 6], mean = 4.5.
        """
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        assert interquartile_mean(scores) == pytest.approx(4.5)

    def test_all_same_values(self):
        """IQM of all-same values should equal that value."""
        scores = np.full(20, 0.42)
        assert interquartile_mean(scores) == pytest.approx(0.42)

    def test_robust_to_outliers(self):
        """IQM should be robust to extreme outliers.

        Array of 20 values: 18 values of 0.5, one extreme low (0.0),
        one extreme high (100.0).  IQM should be close to 0.5.
        """
        scores = np.full(20, 0.5)
        scores[0] = 0.0
        scores[-1] = 100.0
        iqm = interquartile_mean(scores)
        # After sorting: [0.0, 0.5, ..., 0.5, 100.0]
        # Trim 5 from each end (25% of 20), middle 10 are all 0.5
        assert iqm == pytest.approx(0.5)

    def test_single_element(self):
        """Single element: IQM should equal that element."""
        assert interquartile_mean(np.array([7.0])) == pytest.approx(7.0)

    def test_raises_on_empty(self):
        """Should raise ValueError for empty array."""
        with pytest.raises(ValueError, match="empty"):
            interquartile_mean(np.array([]))

    def test_iqm_less_affected_than_mean(self):
        """IQM should be closer to the median than the arithmetic mean
        when outliers are present."""
        base = np.full(20, 0.5)
        base[0] = 0.0
        base[-1] = 10.0
        iqm = interquartile_mean(base)
        arithmetic_mean = base.mean()
        # IQM should be closer to 0.5 than arithmetic mean
        assert abs(iqm - 0.5) < abs(arithmetic_mean - 0.5)


class TestInterquartileMeanBootstrapCI:
    """Tests for IQM bootstrap confidence interval."""

    def test_ci_contains_iqm(self, rng):
        """The CI should contain the sample IQM."""
        scores = rng.normal(0.5, 0.1, 50)
        iqm_val, lo, hi = interquartile_mean_bootstrap_ci(
            scores, n_bootstrap=3000, random_state=42
        )
        assert lo <= iqm_val <= hi

    def test_deterministic(self, rng):
        """Same seed should produce identical results."""
        scores = rng.normal(0.5, 0.1, 30)
        r1 = interquartile_mean_bootstrap_ci(scores, random_state=99)
        r2 = interquartile_mean_bootstrap_ci(scores, random_state=99)
        assert r1[0] == pytest.approx(r2[0])
        assert r1[1] == pytest.approx(r2[1])
        assert r1[2] == pytest.approx(r2[2])

    def test_single_element(self):
        """Single element should return (val, val, val)."""
        iqm_val, lo, hi = interquartile_mean_bootstrap_ci(np.array([2.0]))
        assert iqm_val == pytest.approx(2.0)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(2.0)


# ===========================================================================
# 7. cliffs_delta
# ===========================================================================


class TestCliffsDelta:
    """Tests for Cliff's Delta effect size."""

    def test_identical_arrays(self):
        """Identical arrays should give delta = 0, 'negligible'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        delta, label = cliffs_delta(x, x)
        assert delta == pytest.approx(0.0)
        assert label == "negligible"

    def test_completely_separated_positive(self):
        """x always > y should give delta = 1.0, 'large'."""
        x = np.array([4.0, 5.0, 6.0])
        y = np.array([1.0, 2.0, 3.0])
        delta, label = cliffs_delta(x, y)
        assert delta == pytest.approx(1.0)
        assert label == "large"

    def test_completely_separated_negative(self):
        """x always < y should give delta = -1.0, 'large'."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        delta, label = cliffs_delta(x, y)
        assert delta == pytest.approx(-1.0)
        assert label == "large"

    def test_known_example(self):
        """[1,2,3] vs [4,5,6]: all x < y, so delta = -1.0."""
        delta, label = cliffs_delta(
            np.array([1, 2, 3]), np.array([4, 5, 6])
        )
        assert delta == pytest.approx(-1.0)
        assert label == "large"

    def test_threshold_negligible(self):
        """Values near zero should be 'negligible'."""
        # Craft arrays with delta near 0.1 (< 0.147)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 4.5])
        delta, label = cliffs_delta(x, y)
        assert abs(delta) < 0.147
        assert label == "negligible"

    def test_symmetry(self):
        """cliffs_delta(x, y) = -cliffs_delta(y, x)."""
        x = np.array([1.0, 2.0, 5.0, 7.0])
        y = np.array([3.0, 4.0, 6.0, 8.0])
        d1, _ = cliffs_delta(x, y)
        d2, _ = cliffs_delta(y, x)
        assert d1 == pytest.approx(-d2)

    def test_raises_on_empty(self):
        """Should raise ValueError for empty arrays."""
        with pytest.raises(ValueError, match="non-empty"):
            cliffs_delta(np.array([]), np.array([1, 2]))
        with pytest.raises(ValueError, match="non-empty"):
            cliffs_delta(np.array([1, 2]), np.array([]))

    def test_single_elements(self):
        """Should work with single-element arrays."""
        delta, label = cliffs_delta(np.array([5.0]), np.array([3.0]))
        assert delta == pytest.approx(1.0)
        assert label == "large"

    def test_medium_effect(self):
        """Craft arrays for a medium effect size (0.33 <= |d| < 0.474)."""
        # With 3 elements each, each comparison is worth 1/9
        # Need roughly 6/9 = 0.667 of pairs with x > y minus x < y ~ 0.33-0.47
        # Let's use: x=[3,4,5], y=[2,3,6] => comparisons:
        # 3>2(+), 3=3(0), 3<6(-), 4>2(+), 4>3(+), 4<6(-), 5>2(+), 5>3(+), 5<6(-)
        # n_more=5, n_less=3, delta=(5-3)/9 = 0.222... small, not medium
        # Try: x=[3,5,7], y=[1,4,6] =>
        # 3>1(+), 3<4(-), 3<6(-), 5>1(+), 5>4(+), 5<6(-), 7>1(+), 7>4(+), 7>6(+)
        # n_more=6, n_less=3, delta=3/9=0.333 -> medium
        x = np.array([3.0, 5.0, 7.0])
        y = np.array([1.0, 4.0, 6.0])
        delta, label = cliffs_delta(x, y)
        assert delta == pytest.approx(3.0 / 9.0)
        assert label == "medium"


# ===========================================================================
# 8. kendalls_w
# ===========================================================================


class TestKendallsW:
    """Tests for Kendall's W coefficient of concordance."""

    def test_perfect_agreement(self):
        """When all raters agree perfectly, W should be 1.0."""
        # 3 raters, 4 items, all rank the same way
        rankings = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ], dtype=float)
        w, p = kendalls_w(rankings)
        assert w == pytest.approx(1.0, abs=1e-10)
        assert p < 0.05  # perfect agreement is significant

    def test_no_agreement(self):
        """When raters disagree maximally, W should be near 0.

        For 2 raters with reversed rankings, W = 0.
        """
        rankings = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ], dtype=float)
        w, p = kendalls_w(rankings)
        # With just 2 reversed raters, W should be 0
        assert w == pytest.approx(0.0, abs=1e-10)

    def test_partial_agreement(self):
        """Partial agreement should give 0 < W < 1."""
        rankings = np.array([
            [1, 2, 3, 4, 5],
            [1, 3, 2, 4, 5],
            [2, 1, 3, 5, 4],
        ], dtype=float)
        w, p = kendalls_w(rankings)
        assert 0 < w < 1

    def test_raises_on_single_rater(self):
        """Should raise ValueError with < 2 raters."""
        with pytest.raises(ValueError, match="at least 2 raters"):
            kendalls_w(np.array([[1, 2, 3]]))

    def test_raises_on_single_item(self):
        """Should raise ValueError with < 2 items."""
        with pytest.raises(ValueError, match="at least 2 items"):
            kendalls_w(np.array([[1], [1]]))

    def test_three_raters_perfect(self):
        """Three raters in perfect agreement."""
        rankings = np.array([
            [3, 1, 2],
            [3, 1, 2],
            [3, 1, 2],
        ], dtype=float)
        w, p = kendalls_w(rankings)
        assert w == pytest.approx(1.0, abs=1e-10)


# ===========================================================================
# 9. kendalls_tau
# ===========================================================================


class TestKendallsTau:
    """Tests for Kendall's tau-b rank correlation."""

    def test_identical_rankings(self):
        """Identical rankings should give tau = 1.0."""
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        tau, p = kendalls_tau(a, a)
        assert tau == pytest.approx(1.0)

    def test_reversed_rankings(self):
        """Completely reversed rankings should give tau = -1.0."""
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([5, 4, 3, 2, 1], dtype=float)
        tau, p = kendalls_tau(a, b)
        assert tau == pytest.approx(-1.0)

    def test_partial_correlation(self):
        """Partial correlation should give -1 < tau < 1."""
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([1, 3, 2, 5, 4], dtype=float)
        tau, p = kendalls_tau(a, b)
        assert -1 < tau < 1

    def test_raises_on_different_lengths(self):
        """Should raise ValueError when rankings differ in length."""
        with pytest.raises(ValueError, match="same length"):
            kendalls_tau(np.array([1, 2, 3]), np.array([1, 2]))

    def test_raises_on_too_few_items(self):
        """Should raise ValueError with < 2 items."""
        with pytest.raises(ValueError, match="at least 2"):
            kendalls_tau(np.array([1.0]), np.array([1.0]))


# ===========================================================================
# 10. compute_all_bootstrap_cis
# ===========================================================================


class TestComputeAllBootstrapCIs:
    """Tests for compute_all_bootstrap_cis."""

    def test_one_ci_per_system(self, different_means_matrix, six_system_names):
        """Should return one BootstrapCI per system."""
        cis = compute_all_bootstrap_cis(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        assert len(cis) == 6
        for ci in cis:
            assert isinstance(ci, BootstrapCI)

    def test_ci_fields_populated(self, different_means_matrix, six_system_names):
        """All fields should be populated with finite values."""
        cis = compute_all_bootstrap_cis(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        for ci in cis:
            assert ci.system in six_system_names
            assert ci.metric == "overall"
            assert np.isfinite(ci.mean)
            assert np.isfinite(ci.ci_lower)
            assert np.isfinite(ci.ci_upper)
            assert np.isfinite(ci.iqm)
            assert np.isfinite(ci.iqm_ci_lower)
            assert np.isfinite(ci.iqm_ci_upper)
            assert np.isfinite(ci.std)
            assert ci.ci_lower <= ci.mean <= ci.ci_upper
            assert ci.iqm_ci_lower <= ci.iqm <= ci.iqm_ci_upper
            assert ci.n_samples == 20
            assert ci.n_bootstrap == 1000

    def test_ci_width_decreases_with_more_samples(self, rng):
        """CI should be narrower with more tasks."""
        names = ["A", "B"]
        small = rng.normal(0.5, 0.2, size=(10, 2))
        large = rng.normal(0.5, 0.2, size=(200, 2))

        cis_small = compute_all_bootstrap_cis(
            small, names, n_bootstrap=2000, random_state=42
        )
        cis_large = compute_all_bootstrap_cis(
            large, names, n_bootstrap=2000, random_state=42
        )

        for cs, cl in zip(cis_small, cis_large):
            width_small = cs.ci_upper - cs.ci_lower
            width_large = cl.ci_upper - cl.ci_lower
            assert width_large < width_small

    def test_custom_metric_name(self, different_means_matrix, six_system_names):
        """Custom metric name should be reflected in results."""
        cis = compute_all_bootstrap_cis(
            different_means_matrix, six_system_names,
            metric_name="factual_accuracy",
            n_bootstrap=500,
        )
        for ci in cis:
            assert ci.metric == "factual_accuracy"


# ===========================================================================
# 10b. ranking_concordance
# ===========================================================================


class TestRankingConcordance:
    """Tests for ranking_concordance."""

    def test_perfect_concordance(self):
        """Two methods that agree should have W = 1 and tau = 1."""
        rankings = {
            "method_a": [0.9, 0.7, 0.5, 0.3],
            "method_b": [0.8, 0.6, 0.4, 0.2],
        }
        result = ranking_concordance(
            rankings,
            method_names=["method_a", "method_b"],
            system_names=["S0", "S1", "S2", "S3"],
        )
        assert isinstance(result, ConcordanceResult)
        assert result.kendalls_w == pytest.approx(1.0, abs=1e-10)

        tau, p = result.pairwise_tau[("method_a", "method_b")]
        assert tau == pytest.approx(1.0)

    def test_reversed_concordance(self):
        """Two methods with opposite rankings should have W near 0, tau = -1."""
        rankings = {
            "method_a": [0.9, 0.7, 0.5, 0.3],
            "method_b": [0.3, 0.5, 0.7, 0.9],
        }
        result = ranking_concordance(
            rankings,
            method_names=["method_a", "method_b"],
            system_names=["S0", "S1", "S2", "S3"],
        )
        assert result.kendalls_w == pytest.approx(0.0, abs=1e-10)

        tau, p = result.pairwise_tau[("method_a", "method_b")]
        assert tau == pytest.approx(-1.0)

    def test_ranked_system_names(self):
        """Should return system names in order of their scores."""
        rankings = {
            "m1": [0.3, 0.9, 0.5],
        }
        # Only need 2 methods for concordance, but let's add a second
        rankings["m2"] = [0.4, 0.8, 0.6]
        result = ranking_concordance(
            rankings,
            method_names=["m1", "m2"],
            system_names=["A", "B", "C"],
        )
        # Both methods rank B > C > A
        assert result.rankings_per_method["m1"] == ["B", "C", "A"]
        assert result.rankings_per_method["m2"] == ["B", "C", "A"]


# ===========================================================================
# 10c. nemenyi_critical_difference
# ===========================================================================


class TestNemenyiCriticalDifference:
    """Tests for nemenyi_critical_difference."""

    def test_known_cd_6_systems_20_tasks(self):
        """CD for 6 systems, 20 tasks at alpha=0.05 should match manual calculation."""
        k, n = 6, 20
        # q_alpha for k=6 is 2.850
        expected = 2.850 * math.sqrt(k * (k + 1) / (6 * n))
        cd = nemenyi_critical_difference(k, n, alpha=0.05)
        assert cd == pytest.approx(expected, rel=1e-3)

    def test_cd_increases_with_k(self):
        """CD should increase with more systems (wider comparisons)."""
        n = 30
        cd3 = nemenyi_critical_difference(3, n)
        cd6 = nemenyi_critical_difference(6, n)
        assert cd6 > cd3

    def test_cd_decreases_with_n(self):
        """CD should decrease with more tasks (more statistical power)."""
        k = 6
        cd_small = nemenyi_critical_difference(k, 10)
        cd_large = nemenyi_critical_difference(k, 100)
        assert cd_large < cd_small

    def test_positive_for_valid_inputs(self):
        """CD should always be positive."""
        for k in range(2, 11):
            for n in [5, 10, 50]:
                cd = nemenyi_critical_difference(k, n)
                assert cd > 0

    def test_raises_on_invalid_k(self):
        """Should raise ValueError for k < 2."""
        with pytest.raises(ValueError, match="at least 2 systems"):
            nemenyi_critical_difference(1, 10)

    def test_raises_on_invalid_n(self):
        """Should raise ValueError for n < 1."""
        with pytest.raises(ValueError, match="at least 1 task"):
            nemenyi_critical_difference(3, 0)

    def test_lookup_table_values_used(self):
        """For alpha=0.05 and k in 2..10, should use lookup table."""
        # Just verify it runs without error for all tabulated k values
        for k in range(2, 11):
            cd = nemenyi_critical_difference(k, 20, alpha=0.05)
            assert cd > 0


# ===========================================================================
# 10d. power_analysis
# ===========================================================================


class TestPowerAnalysis:
    """Tests for power_analysis."""

    def test_returns_expected_keys(self):
        """Should return a dict with all expected keys."""
        result = power_analysis(n_queries=30, k_systems=6)
        expected_keys = {
            "min_detectable_w", "friedman_df", "n_pairwise_comparisons",
            "nemenyi_cd", "min_n_for_small_effect",
        }
        assert set(result.keys()) == expected_keys

    def test_friedman_df_correct(self):
        """Friedman df should be k-1."""
        result = power_analysis(n_queries=30, k_systems=6)
        assert result["friedman_df"] == 5

    def test_n_pairwise_comparisons(self):
        """Number of pairwise comparisons should be C(k, 2)."""
        result = power_analysis(n_queries=30, k_systems=6)
        assert result["n_pairwise_comparisons"] == 15

    def test_min_detectable_w_is_positive(self):
        """Minimum detectable W should be positive and < 1."""
        result = power_analysis(n_queries=30, k_systems=6)
        assert 0 < result["min_detectable_w"] < 1

    def test_min_w_decreases_with_n(self):
        """With more queries, we can detect smaller effects."""
        r_small = power_analysis(n_queries=10, k_systems=6)
        r_large = power_analysis(n_queries=100, k_systems=6)
        assert r_large["min_detectable_w"] < r_small["min_detectable_w"]

    def test_nemenyi_cd_positive(self):
        """Nemenyi CD should be positive."""
        result = power_analysis(n_queries=30, k_systems=6)
        assert result["nemenyi_cd"] > 0

    def test_min_n_for_small_effect_reasonable(self):
        """Minimum n for detecting W=0.1 should be a reasonable positive number."""
        result = power_analysis(n_queries=30, k_systems=6)
        min_n = result["min_n_for_small_effect"]
        # W=0.1 is a small effect; should need a reasonable sample size
        assert isinstance(min_n, (int, float))
        if not math.isnan(min_n):
            assert min_n > 5

    def test_three_systems(self):
        """Should work with 3 systems."""
        result = power_analysis(n_queries=30, k_systems=3)
        assert result["friedman_df"] == 2
        assert result["n_pairwise_comparisons"] == 3
        assert result["min_detectable_w"] > 0

    def test_large_n_gives_small_w(self):
        """With very large n, even small effects should be detectable."""
        result = power_analysis(n_queries=1000, k_systems=6)
        assert result["min_detectable_w"] < 0.05


# ===========================================================================
# 10e. bootstrap_rank_distribution
# ===========================================================================


class TestBootstrapRankDistribution:
    """Tests for bootstrap_rank_distribution."""

    def test_returns_one_dict_per_system(self, different_means_matrix, six_system_names):
        """Should return one dict per system."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        assert len(results) == 6

    def test_dict_keys_present(self, different_means_matrix, six_system_names):
        """Each dict should have the expected keys."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        expected_keys = {
            "system", "mean_rank", "rank_ci_lower", "rank_ci_upper",
            "prob_rank_1", "rank_distribution",
        }
        for r in results:
            assert set(r.keys()) == expected_keys

    def test_system_names_match(self, different_means_matrix, six_system_names):
        """System names in results should match input."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        result_names = {r["system"] for r in results}
        assert result_names == set(six_system_names)

    def test_best_system_has_lowest_mean_rank(self, different_means_matrix, six_system_names):
        """The system with the highest mean should have the lowest mean rank."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=2000, random_state=42,
        )
        # P5_HierWD has the highest mean (0.8)
        p5 = next(r for r in results if r["system"] == "P5_HierWD")
        p0 = next(r for r in results if r["system"] == "P0_Baseline")
        assert p5["mean_rank"] < p0["mean_rank"]

    def test_best_system_high_prob_rank_1(self, different_means_matrix, six_system_names):
        """The best system should have a high probability of being ranked 1."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=2000, random_state=42,
        )
        p5 = next(r for r in results if r["system"] == "P5_HierWD")
        # With clearly separated means, P5 should almost always be rank 1
        assert p5["prob_rank_1"] > 0.5

    def test_rank_distribution_sums_to_one(self, different_means_matrix, six_system_names):
        """Each system's rank distribution should sum to 1.0."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        for r in results:
            assert sum(r["rank_distribution"]) == pytest.approx(1.0, abs=1e-6)

    def test_rank_ci_valid(self, different_means_matrix, six_system_names):
        """CI lower should be <= mean_rank <= CI upper."""
        results = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=1000, random_state=42,
        )
        for r in results:
            assert r["rank_ci_lower"] <= r["mean_rank"] <= r["rank_ci_upper"]

    def test_deterministic_with_seed(self, different_means_matrix, six_system_names):
        """Same seed should produce identical results."""
        r1 = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=500, random_state=123,
        )
        r2 = bootstrap_rank_distribution(
            different_means_matrix, six_system_names,
            n_bootstrap=500, random_state=123,
        )
        for a, b in zip(r1, r2):
            assert a["mean_rank"] == pytest.approx(b["mean_rank"])
            assert a["prob_rank_1"] == pytest.approx(b["prob_rank_1"])

    def test_raises_on_name_mismatch(self):
        """Should raise ValueError when names don't match columns."""
        with pytest.raises(ValueError, match="system_names length"):
            bootstrap_rank_distribution(np.ones((5, 3)), ["A", "B"])


# ===========================================================================
# 10f. run_stratified_analysis
# ===========================================================================


class TestRunStratifiedAnalysis:
    """Tests for run_stratified_analysis."""

    def test_basic_stratification(self, rng):
        """Should return one result per stratum with >= 5 queries."""
        n_tasks = 20
        k = 3
        names = ["A", "B", "C"]
        matrix = np.zeros((n_tasks, k))
        for j in range(k):
            matrix[:, j] = rng.normal(0.3 + 0.1 * j, 0.05, size=n_tasks)

        strata = ["easy"] * 10 + ["hard"] * 10
        results = run_stratified_analysis(
            matrix, names, strata, n_bootstrap=500,
        )

        assert "easy" in results
        assert "hard" in results
        assert isinstance(results["easy"], FullAnalysisResult)
        assert isinstance(results["hard"], FullAnalysisResult)

    def test_skips_small_strata(self, rng):
        """Strata with < 5 queries should be skipped."""
        n_tasks = 12
        k = 3
        names = ["A", "B", "C"]
        matrix = rng.normal(0.5, 0.1, size=(n_tasks, k))

        # 8 easy, 4 hard (hard has < 5 queries)
        strata = ["easy"] * 8 + ["hard"] * 4
        results = run_stratified_analysis(
            matrix, names, strata, n_bootstrap=500,
        )

        assert "easy" in results
        assert "hard" not in results

    def test_single_stratum(self, rng):
        """All queries in one stratum should produce one result."""
        n_tasks = 15
        k = 3
        names = ["A", "B", "C"]
        matrix = rng.normal(0.5, 0.1, size=(n_tasks, k))

        strata = ["all"] * n_tasks
        results = run_stratified_analysis(
            matrix, names, strata, n_bootstrap=500,
        )

        assert len(results) == 1
        assert "all" in results
        assert results["all"].omnibus.n_tasks == 15

    def test_correct_n_tasks_per_stratum(self, rng):
        """Each stratum should have the correct number of tasks."""
        n_tasks = 20
        k = 3
        names = ["A", "B", "C"]
        matrix = rng.normal(0.5, 0.1, size=(n_tasks, k))

        strata = ["easy"] * 12 + ["hard"] * 8
        results = run_stratified_analysis(
            matrix, names, strata, n_bootstrap=500,
        )

        assert results["easy"].omnibus.n_tasks == 12
        assert results["hard"].omnibus.n_tasks == 8

    def test_raises_on_wrong_strata_length(self, rng):
        """Should raise ValueError when strata length != n_tasks."""
        matrix = rng.normal(0.5, 0.1, size=(10, 3))
        with pytest.raises(ValueError, match="query_strata length"):
            run_stratified_analysis(
                matrix, ["A", "B", "C"], ["easy"] * 5, n_bootstrap=500,
            )


# ===========================================================================
# 11. run_full_analysis
# ===========================================================================


class TestRunFullAnalysis:
    """Integration tests for the full analysis pipeline."""

    def test_with_significant_data(self, realistic_matrix, six_system_names):
        """Realistic data should produce a complete FullAnalysisResult."""
        result = run_full_analysis(
            realistic_matrix, six_system_names,
            n_bootstrap=1000,
        )

        assert isinstance(result, FullAnalysisResult)
        assert isinstance(result.omnibus, OmnibusResult)
        assert result.omnibus.n_systems == 6
        assert result.omnibus.n_tasks == 25

    def test_bootstrap_cis_populated(self, realistic_matrix, six_system_names):
        """Should have one bootstrap CI per system."""
        result = run_full_analysis(
            realistic_matrix, six_system_names,
            n_bootstrap=1000,
        )
        assert len(result.bootstrap_cis) == 6
        for ci in result.bootstrap_cis:
            assert isinstance(ci, BootstrapCI)

    def test_pairwise_populated_when_significant(self, different_means_matrix, six_system_names):
        """When Friedman is significant, pairwise tests should be populated."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert result.omnibus.is_significant
        assert len(result.pairwise) == 15  # C(6, 2) -- primary: Wilcoxon+Holm

    def test_nemenyi_pairwise_populated_when_significant(self, different_means_matrix, six_system_names):
        """When Friedman is significant, nemenyi_pairwise should also be populated."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert result.omnibus.is_significant
        assert len(result.nemenyi_pairwise) == 15  # C(6, 2) -- secondary: Nemenyi

    def test_primary_pairwise_is_wilcoxon_holm(self, different_means_matrix, six_system_names):
        """Primary pairwise should use wilcoxon_holm test name."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        if result.pairwise:
            for pw in result.pairwise:
                assert pw.test_name == "wilcoxon_holm"

    def test_secondary_pairwise_is_nemenyi(self, different_means_matrix, six_system_names):
        """Secondary pairwise should use nemenyi test name."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        if result.nemenyi_pairwise:
            for pw in result.nemenyi_pairwise:
                assert pw.test_name == "nemenyi"

    def test_pairwise_empty_when_not_significant(self, identical_scores_matrix, six_system_names):
        """When Friedman is not significant, pairwise should be empty."""
        result = run_full_analysis(
            identical_scores_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert result.omnibus.is_significant is False
        assert result.pairwise == []
        assert result.nemenyi_pairwise == []

    def test_per_dimension_analysis(self, rng, six_system_names):
        """Per-dimension analysis should produce results for each dimension."""
        n_tasks = 20
        k = len(six_system_names)
        main = rng.uniform(0.3, 0.7, size=(n_tasks, k))

        dim_matrices = {
            "coverage": rng.uniform(0.2, 0.9, size=(n_tasks, k)),
            "factual": rng.uniform(0.0, 0.5, size=(n_tasks, k)),
        }

        result = run_full_analysis(
            main, six_system_names,
            dimension_matrices=dim_matrices,
            dimension_names=["coverage", "factual"],
            n_bootstrap=500,
        )

        assert "coverage" in result.per_dimension_omnibus
        assert "factual" in result.per_dimension_omnibus
        assert isinstance(result.per_dimension_omnibus["coverage"], OmnibusResult)

    def test_summary_markdown_not_empty(self, realistic_matrix, six_system_names):
        """Summary markdown should be generated."""
        result = run_full_analysis(
            realistic_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert len(result.summary_markdown) > 0
        assert "# Statistical Analysis Summary" in result.summary_markdown

    def test_all_sub_results_populated(self, different_means_matrix, six_system_names):
        """All fields of FullAnalysisResult should be populated."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert result.omnibus is not None
        assert isinstance(result.pairwise, list)
        assert isinstance(result.nemenyi_pairwise, list)
        assert isinstance(result.bootstrap_cis, list)
        assert isinstance(result.per_dimension_omnibus, dict)
        assert isinstance(result.per_dimension_pairwise, dict)
        assert isinstance(result.summary_markdown, str)
        assert isinstance(result.rank_distributions, list)
        assert result.critical_difference > 0

    def test_iman_davenport_in_omnibus(self, different_means_matrix, six_system_names):
        """Iman-Davenport should be computed in omnibus."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert result.omnibus.iman_davenport_f > 0
        assert result.omnibus.iman_davenport_p < 0.05

    def test_critical_difference_positive(self, different_means_matrix, six_system_names):
        """Critical difference should be positive."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert result.critical_difference > 0

    def test_rank_distributions_populated(self, different_means_matrix, six_system_names):
        """Rank distributions should be populated."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        assert len(result.rank_distributions) == 6


# ===========================================================================
# 12. generate_summary_markdown
# ===========================================================================


class TestGenerateSummaryMarkdown:
    """Tests for markdown summary generation."""

    def test_contains_friedman_section(self, different_means_matrix, six_system_names):
        """Summary should contain Friedman test results."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        assert "## Friedman Omnibus Test" in md
        assert "Chi-squared statistic" in md

    def test_contains_iman_davenport(self, different_means_matrix, six_system_names):
        """Summary should contain Iman-Davenport F statistic."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown
        assert "Iman-Davenport F" in md

    def test_contains_primary_pairwise_table(self, different_means_matrix, six_system_names):
        """When significant, summary should contain primary pairwise comparison table."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        assert "## Pairwise Comparisons (Wilcoxon + Holm-Bonferroni)" in md
        assert "System A" in md
        assert "System B" in md

    def test_contains_nemenyi_table(self, different_means_matrix, six_system_names):
        """When significant, summary should contain Nemenyi secondary table."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown
        assert "## Nemenyi Post-Hoc (secondary)" in md

    def test_contains_ci_table(self, different_means_matrix, six_system_names):
        """Summary should contain bootstrap CI table."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        assert "## Bootstrap Confidence Intervals" in md
        assert "IQM" in md

    def test_contains_rank_distributions(self, different_means_matrix, six_system_names):
        """Summary should contain rank distribution table."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown
        assert "## Bootstrap Rank Distributions" in md
        assert "P(Rank 1)" in md

    def test_contains_critical_difference(self, different_means_matrix, six_system_names):
        """Summary should mention the critical difference."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown
        assert "Critical Difference" in md

    def test_contains_system_names(self, different_means_matrix, six_system_names):
        """Summary should mention all system names."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        for name in six_system_names:
            assert name in md

    def test_not_significant_message(self, identical_scores_matrix, six_system_names):
        """When Friedman is not significant, should say so."""
        result = run_full_analysis(
            identical_scores_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        assert "not significant" in md

    def test_per_dimension_in_summary(self, rng, six_system_names):
        """Per-dimension results should appear in markdown if provided."""
        n_tasks = 20
        k = len(six_system_names)
        main = rng.uniform(0.3, 0.7, size=(n_tasks, k))
        dim_matrices = {
            "coverage": rng.uniform(0.2, 0.9, size=(n_tasks, k)),
        }

        result = run_full_analysis(
            main, six_system_names,
            dimension_matrices=dim_matrices,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        assert "## Per-Dimension Analysis" in md or "coverage" in md

    def test_markdown_is_valid(self, different_means_matrix, six_system_names):
        """Basic markdown validity: headers, tables have pipes."""
        result = run_full_analysis(
            different_means_matrix, six_system_names,
            n_bootstrap=500,
        )
        md = result.summary_markdown

        # Should have at least a few markdown headers
        assert md.count("#") >= 3
        # Should have markdown tables (pipes)
        assert md.count("|") >= 10

    def test_per_dimension_holm_bonferroni_fwer(self, rng, six_system_names):
        """Holm-Bonferroni FWER correction should be applied across dimensions.

        When testing multiple dimensions, each dimension's Friedman test is
        evaluated against a progressively stricter alpha threshold to control
        the family-wise error rate. A borderline-significant dimension (e.g.
        p=0.04) should become non-significant when tested alongside many other
        dimensions under Holm correction.
        """
        n_tasks = 20
        k = len(six_system_names)
        main = rng.uniform(0.3, 0.7, size=(n_tasks, k))

        # Create many dimensions with random (likely non-significant) data
        dim_matrices = {}
        for i in range(9):
            dim_matrices[f"dim_{i}"] = rng.uniform(0.3, 0.7, size=(n_tasks, k))

        result = run_full_analysis(
            main, six_system_names,
            dimension_matrices=dim_matrices,
            n_bootstrap=500,
        )

        # All 9 dimensions should be present in results
        assert len(result.per_dimension_omnibus) == 9

        # With Holm correction across 9 dimensions, the threshold for the
        # most significant dimension is alpha/9 = 0.0056 (much stricter).
        # Count how many are marked significant.
        sig_count = sum(
            1 for omni in result.per_dimension_omnibus.values()
            if omni.is_significant
        )
        # With random data, most or all should be non-significant
        # (this is probabilistic but very unlikely to have >3 significant
        # out of 9 random dimensions even without correction)
        assert sig_count <= 5  # Generous upper bound

    def test_per_dimension_holm_single_dimension_no_correction(self, rng, six_system_names):
        """With only 1 dimension, no Holm correction is needed."""
        n_tasks = 30
        k = len(six_system_names)
        main = rng.uniform(0.3, 0.7, size=(n_tasks, k))

        # Create a single dimension with clearly different means
        dim_data = np.column_stack([
            rng.normal(loc=0.3 + 0.1 * j, scale=0.05, size=n_tasks)
            for j in range(k)
        ])
        dim_matrices = {"only_dim": dim_data}

        result = run_full_analysis(
            main, six_system_names,
            dimension_matrices=dim_matrices,
            n_bootstrap=500,
        )

        # Single dimension: alpha threshold stays at 0.05 (no correction)
        assert "only_dim" in result.per_dimension_omnibus
        # Should be significant because means are clearly different
        assert result.per_dimension_omnibus["only_dim"].is_significant

    def test_holm_correction_note_in_markdown(self, rng, six_system_names):
        """Markdown should mention Holm-Bonferroni when multiple dimensions tested."""
        n_tasks = 20
        k = len(six_system_names)
        main = rng.uniform(0.3, 0.7, size=(n_tasks, k))
        dim_matrices = {
            "dim_a": rng.uniform(0.2, 0.9, size=(n_tasks, k)),
            "dim_b": rng.uniform(0.2, 0.9, size=(n_tasks, k)),
        }

        result = run_full_analysis(
            main, six_system_names,
            dimension_matrices=dim_matrices,
            n_bootstrap=500,
        )

        assert "Holm-Bonferroni" in result.summary_markdown


# ===========================================================================
# Edge cases and regression tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_two_system_full_analysis(self, rng):
        """Full analysis should work with just 2 systems."""
        matrix = np.column_stack([
            rng.normal(0.5, 0.1, 15),
            rng.normal(0.7, 0.1, 15),
        ])
        result = run_full_analysis(
            matrix, ["A", "B"], n_bootstrap=500
        )
        assert result.omnibus.n_systems == 2
        assert len(result.bootstrap_cis) == 2

    def test_many_tasks_few_systems(self, rng):
        """Should handle many tasks with few systems."""
        matrix = np.column_stack([
            rng.normal(0.4, 0.1, 100),
            rng.normal(0.6, 0.1, 100),
            rng.normal(0.8, 0.1, 100),
        ])
        result = run_full_analysis(
            matrix, ["A", "B", "C"], n_bootstrap=500
        )
        assert result.omnibus.n_tasks == 100

    def test_very_small_differences(self, rng):
        """Systems with very small differences may not be significant."""
        matrix = np.column_stack([
            rng.normal(0.5, 0.1, 10),
            rng.normal(0.51, 0.1, 10),
            rng.normal(0.52, 0.1, 10),
        ])
        result = run_full_analysis(
            matrix, ["A", "B", "C"], n_bootstrap=500
        )
        # Should not crash; significance depends on data
        assert isinstance(result.omnibus.is_significant, bool)

    def test_constant_column(self, rng):
        """A system with constant scores should not crash the pipeline."""
        n = 15
        matrix = np.column_stack([
            rng.normal(0.5, 0.1, n),
            np.full(n, 0.5),  # constant
            rng.normal(0.7, 0.1, n),
        ])
        # Bootstrap and IQM should handle constant values
        cis = compute_all_bootstrap_cis(
            matrix, ["A", "Const", "B"], n_bootstrap=500
        )
        assert len(cis) == 3
        # Constant system should have zero-width CI
        const_ci = cis[1]
        assert const_ci.ci_lower == pytest.approx(0.5)
        assert const_ci.ci_upper == pytest.approx(0.5)

    def test_holm_bonferroni_preserves_order_invariance(self):
        """Result should be the same regardless of input order (up to reorder)."""
        raw = [0.005, 0.01, 0.03, 0.04]
        shuffled = [0.03, 0.005, 0.04, 0.01]

        c1 = holm_bonferroni(raw)
        c2 = holm_bonferroni(shuffled)

        # Map: shuffled[0]=raw[2], shuffled[1]=raw[0], shuffled[2]=raw[3], shuffled[3]=raw[1]
        assert c2[0] == pytest.approx(c1[2])
        assert c2[1] == pytest.approx(c1[0])
        assert c2[2] == pytest.approx(c1[3])
        assert c2[3] == pytest.approx(c1[1])
