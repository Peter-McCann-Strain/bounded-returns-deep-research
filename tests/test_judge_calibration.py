"""Comprehensive tests for the judge calibration module.

Tests cover:
- detect_position_bias: known positive/negative bias, no bias, single position, empty
- detect_length_bias: positive correlation, no correlation, constant lengths, empty
- detect_severity: all passing (0.0), all failing (1.0), balanced (~0.5), empty
- detect_dimension_bias: uniform rates, one outlier dimension, single dimension
- calibrate_scores: basic correction, severity adjustment, clamping, empty
- run_calibration: integration with known data, empty data, recommendations
- CalibrationResult and CalibrationData dataclass construction
"""

from __future__ import annotations

import math

import pytest

from deep_research.evaluation.judge_calibration import (
    CalibrationData,
    CalibrationResult,
    calibrate_scores,
    detect_dimension_bias,
    detect_length_bias,
    detect_position_bias,
    detect_severity,
    run_calibration,
    _pearson_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verdicts(
    position_verdicts: dict[int, list[str]],
    dimension: str = "coverage",
) -> list[dict]:
    """Build a flat verdict list from position -> verdict mapping."""
    result = []
    for idx, vs in position_verdicts.items():
        for v in vs:
            result.append({
                "criterion_index": idx,
                "verdict": v,
                "dimension": dimension,
            })
    return result


# ---------------------------------------------------------------------------
# 1. _pearson_correlation (internal helper)
# ---------------------------------------------------------------------------

class TestPearsonCorrelation:
    def test_perfect_positive(self):
        r = _pearson_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative(self):
        r = _pearson_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_no_correlation(self):
        # Symmetric around mean in both axes, no linear trend
        r = _pearson_correlation([1.0, 2.0, 3.0], [5.0, 3.0, 5.0])
        assert abs(r) < 1.0  # just verify it's a valid result

    def test_zero_variance_x(self):
        r = _pearson_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        assert r == 0.0

    def test_zero_variance_y(self):
        r = _pearson_correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
        assert r == 0.0

    def test_empty(self):
        r = _pearson_correlation([], [])
        assert r == 0.0

    def test_mismatched_lengths(self):
        r = _pearson_correlation([1.0, 2.0], [1.0])
        assert r == 0.0


# ---------------------------------------------------------------------------
# 2. detect_position_bias
# ---------------------------------------------------------------------------

class TestDetectPositionBias:
    def test_positive_bias(self):
        """Later positions all pass, earlier all fail -> positive correlation."""
        verdicts = [
            ["NOT_SATISFIED"] * 10,  # position 0: 0% pass
            ["NOT_SATISFIED"] * 5 + ["SATISFIED"] * 5,  # position 1: 50%
            ["SATISFIED"] * 10,  # position 2: 100% pass
        ]
        r = detect_position_bias(verdicts)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_negative_bias(self):
        """Earlier positions all pass, later all fail -> negative correlation."""
        verdicts = [
            ["SATISFIED"] * 10,  # position 0: 100% pass
            ["NOT_SATISFIED"] * 5 + ["SATISFIED"] * 5,  # position 1: 50%
            ["NOT_SATISFIED"] * 10,  # position 2: 0% pass
        ]
        r = detect_position_bias(verdicts)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_no_bias_uniform(self):
        """All positions have the same pass rate -> zero correlation."""
        verdicts = [
            ["SATISFIED"] * 5 + ["NOT_SATISFIED"] * 5,
            ["SATISFIED"] * 5 + ["NOT_SATISFIED"] * 5,
            ["SATISFIED"] * 5 + ["NOT_SATISFIED"] * 5,
        ]
        r = detect_position_bias(verdicts)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_single_position_returns_zero(self):
        r = detect_position_bias([["SATISFIED", "NOT_SATISFIED"]])
        assert r == 0.0

    def test_empty_returns_zero(self):
        r = detect_position_bias([])
        assert r == 0.0

    def test_positions_with_empty_lists_skipped(self):
        """Positions with no verdicts are skipped."""
        verdicts = [
            [],  # position 0: no data
            ["NOT_SATISFIED"] * 10,  # position 1: 0%
            ["SATISFIED"] * 10,  # position 2: 100%
        ]
        # Only two effective positions with data -> should compute correlation
        r = detect_position_bias(verdicts)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_all_satisfied(self):
        """All verdicts are SATISFIED across positions -> zero variance in pass rate."""
        verdicts = [
            ["SATISFIED"] * 5,
            ["SATISFIED"] * 5,
            ["SATISFIED"] * 5,
        ]
        r = detect_position_bias(verdicts)
        assert r == 0.0  # zero variance in y


# ---------------------------------------------------------------------------
# 3. detect_length_bias
# ---------------------------------------------------------------------------

class TestDetectLengthBias:
    def test_positive_correlation(self):
        """Longer reports get higher scores."""
        word_counts = [100, 200, 300, 400, 500]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        r = detect_length_bias(word_counts, scores)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_negative_correlation(self):
        """Longer reports get lower scores."""
        word_counts = [100, 200, 300, 400, 500]
        scores = [0.5, 0.4, 0.3, 0.2, 0.1]
        r = detect_length_bias(word_counts, scores)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_no_correlation(self):
        """Scores don't depend on length."""
        word_counts = [100, 200, 300, 400, 500]
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        r = detect_length_bias(word_counts, scores)
        assert r == 0.0

    def test_constant_lengths(self):
        """All reports same length -> zero variance in x."""
        word_counts = [300, 300, 300, 300]
        scores = [0.1, 0.5, 0.8, 0.3]
        r = detect_length_bias(word_counts, scores)
        assert r == 0.0

    def test_empty_input(self):
        r = detect_length_bias([], [])
        assert r == 0.0

    def test_single_report(self):
        r = detect_length_bias([100], [0.5])
        assert r == 0.0

    def test_mismatched_lengths(self):
        r = detect_length_bias([100, 200], [0.5])
        assert r == 0.0


# ---------------------------------------------------------------------------
# 4. detect_severity
# ---------------------------------------------------------------------------

class TestDetectSeverity:
    def test_all_passing(self):
        """All criteria pass -> very lenient -> severity 0.0."""
        s = detect_severity([1.0, 1.0, 1.0, 1.0])
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_all_failing(self):
        """All criteria fail -> very strict -> severity 1.0."""
        s = detect_severity([0.0, 0.0, 0.0, 0.0])
        assert s == pytest.approx(1.0, abs=1e-9)

    def test_balanced(self):
        """50% pass rate -> balanced -> severity 0.5."""
        s = detect_severity([0.5, 0.5, 0.5, 0.5])
        assert s == pytest.approx(0.5, abs=1e-9)

    def test_mixed_rates(self):
        """Average pass rate 0.6 -> severity 0.4."""
        s = detect_severity([0.8, 0.4, 0.6, 0.6])
        assert s == pytest.approx(0.4, abs=1e-9)

    def test_empty_returns_balanced(self):
        s = detect_severity([])
        assert s == pytest.approx(0.5, abs=1e-9)

    def test_single_criterion(self):
        s = detect_severity([0.75])
        assert s == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. detect_dimension_bias
# ---------------------------------------------------------------------------

class TestDetectDimensionBias:
    def test_uniform_rates_zero_bias(self):
        """All dimensions have the same pass rate -> zero bias everywhere."""
        biases = detect_dimension_bias({
            "coverage": 0.5,
            "depth": 0.5,
            "citations": 0.5,
        })
        for dim, b in biases.items():
            assert b == pytest.approx(0.0, abs=1e-9)

    def test_one_outlier_dimension(self):
        """One dimension much easier than others."""
        biases = detect_dimension_bias({
            "coverage": 0.9,
            "depth": 0.3,
            "citations": 0.3,
        })
        mean_rate = (0.9 + 0.3 + 0.3) / 3  # 0.5
        assert biases["coverage"] == pytest.approx(0.9 - mean_rate, abs=1e-9)
        assert biases["depth"] == pytest.approx(0.3 - mean_rate, abs=1e-9)
        assert biases["citations"] == pytest.approx(0.3 - mean_rate, abs=1e-9)
        # coverage should be positive (easy), others negative (hard)
        assert biases["coverage"] > 0
        assert biases["depth"] < 0

    def test_single_dimension(self):
        """Single dimension -> deviation from mean is 0."""
        biases = detect_dimension_bias({"only_dim": 0.7})
        assert biases["only_dim"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns_empty(self):
        biases = detect_dimension_bias({})
        assert biases == {}

    def test_biases_sum_to_zero(self):
        """Deviations from mean must sum to approximately zero."""
        rates = {"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8}
        biases = detect_dimension_bias(rates)
        assert sum(biases.values()) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. calibrate_scores
# ---------------------------------------------------------------------------

class TestCalibrateScores:
    def test_basic_correction(self):
        """Subtracts dimension bias from raw scores."""
        raw = {"coverage": 0.8, "depth": 0.6}
        biases = {"coverage": 0.1, "depth": -0.1}
        calibrated = calibrate_scores(raw, biases)
        # coverage: 0.8 - 0.1 = 0.7
        # depth: 0.6 - (-0.1) = 0.7
        assert calibrated["coverage"] == pytest.approx(0.7, abs=1e-9)
        assert calibrated["depth"] == pytest.approx(0.7, abs=1e-9)

    def test_severity_adjustment_positive(self):
        """Positive severity adjustment (lenient judge) lowers scores."""
        raw = {"dim": 0.8}
        biases = {"dim": 0.0}
        calibrated = calibrate_scores(raw, biases, severity_adjustment=0.2)
        assert calibrated["dim"] == pytest.approx(0.6, abs=1e-9)

    def test_severity_adjustment_negative(self):
        """Negative severity adjustment (strict judge) raises scores."""
        raw = {"dim": 0.4}
        biases = {"dim": 0.0}
        calibrated = calibrate_scores(raw, biases, severity_adjustment=-0.3)
        assert calibrated["dim"] == pytest.approx(0.7, abs=1e-9)

    def test_clamping_upper(self):
        """Score above 1.0 after adjustment is clamped."""
        raw = {"dim": 0.9}
        biases = {"dim": -0.3}  # subtracting negative -> add 0.3 -> 1.2
        calibrated = calibrate_scores(raw, biases)
        assert calibrated["dim"] == pytest.approx(1.0, abs=1e-9)

    def test_clamping_lower(self):
        """Score below 0.0 after adjustment is clamped."""
        raw = {"dim": 0.1}
        biases = {"dim": 0.3}  # 0.1 - 0.3 = -0.2
        calibrated = calibrate_scores(raw, biases)
        assert calibrated["dim"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_input(self):
        calibrated = calibrate_scores({}, {})
        assert calibrated == {}

    def test_missing_bias_for_dimension(self):
        """Dimension not in biases dict -> treated as zero bias."""
        raw = {"unknown_dim": 0.5}
        biases = {"other_dim": 0.1}
        calibrated = calibrate_scores(raw, biases)
        assert calibrated["unknown_dim"] == pytest.approx(0.5, abs=1e-9)

    def test_combined_bias_and_severity(self):
        """Both dimension bias and severity adjustment applied."""
        raw = {"dim": 0.8}
        biases = {"dim": 0.1}
        calibrated = calibrate_scores(raw, biases, severity_adjustment=0.1)
        # 0.8 - 0.1 - 0.1 = 0.6
        assert calibrated["dim"] == pytest.approx(0.6, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. run_calibration (integration)
# ---------------------------------------------------------------------------

class TestRunCalibration:
    def test_integration_known_data(self):
        """Integration test with fully known data."""
        verdicts = []
        # 3 criteria, each evaluated 4 times
        # Criterion 0: all fail -> pass rate 0.0
        for _ in range(4):
            verdicts.append({"criterion_index": 0, "verdict": "NOT_SATISFIED", "dimension": "coverage"})
        # Criterion 1: half pass -> pass rate 0.5
        for _ in range(2):
            verdicts.append({"criterion_index": 1, "verdict": "SATISFIED", "dimension": "depth"})
        for _ in range(2):
            verdicts.append({"criterion_index": 1, "verdict": "NOT_SATISFIED", "dimension": "depth"})
        # Criterion 2: all pass -> pass rate 1.0
        for _ in range(4):
            verdicts.append({"criterion_index": 2, "verdict": "SATISFIED", "dimension": "citations"})

        data = CalibrationData(
            verdicts=verdicts,
            report_word_counts=[100, 200, 300, 400],
            overall_scores=[0.2, 0.4, 0.6, 0.8],
            judge_id="test_judge",
        )

        result = run_calibration(data)

        assert result.judge_id == "test_judge"
        assert result.n_samples == 4
        # Position bias: pass rates are [0.0, 0.5, 1.0] for positions [0, 1, 2]
        # This is a perfect positive correlation
        assert result.position_bias == pytest.approx(1.0, abs=1e-9)
        # Length bias: perfect positive correlation
        assert result.length_bias == pytest.approx(1.0, abs=1e-9)
        # Severity: mean pass rate = (0.0 + 0.5 + 1.0) / 3 = 0.5 -> severity 0.5
        assert result.severity_score == pytest.approx(0.5, abs=1e-9)
        # Dimension biases: coverage=0.0, depth=0.5, citations=1.0, mean=0.5
        assert result.dimension_biases["coverage"] == pytest.approx(-0.5, abs=1e-9)
        assert result.dimension_biases["depth"] == pytest.approx(0.0, abs=1e-9)
        assert result.dimension_biases["citations"] == pytest.approx(0.5, abs=1e-9)

    def test_empty_data(self):
        """Empty data produces defaults and no crash."""
        data = CalibrationData(
            verdicts=[],
            report_word_counts=[],
            overall_scores=[],
            judge_id="empty_judge",
        )
        result = run_calibration(data)

        assert result.judge_id == "empty_judge"
        assert result.n_samples == 0
        assert result.position_bias == 0.0
        assert result.length_bias == 0.0
        assert result.severity_score == 0.5  # default balanced
        assert result.dimension_biases == {}

    def test_recommendations_generated(self):
        """Recommendations list is populated for biased data."""
        # Strong position bias + strong length bias
        verdicts = []
        for _ in range(10):
            verdicts.append({"criterion_index": 0, "verdict": "NOT_SATISFIED", "dimension": "d1"})
        for _ in range(10):
            verdicts.append({"criterion_index": 1, "verdict": "SATISFIED", "dimension": "d2"})

        data = CalibrationData(
            verdicts=verdicts,
            report_word_counts=[100, 200, 300, 400, 500],
            overall_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            judge_id="biased_judge",
        )
        result = run_calibration(data)

        assert len(result.recommendations) > 0
        # Should mention position bias (r=1.0)
        rec_text = " ".join(result.recommendations)
        assert "position bias" in rec_text.lower()

    def test_recommendations_no_bias(self):
        """Well-calibrated judge gets 'no significant biases' recommendation."""
        verdicts = []
        # 3 criteria, all with 50% pass rate, same dimension
        for idx in range(3):
            for _ in range(5):
                verdicts.append({"criterion_index": idx, "verdict": "SATISFIED", "dimension": "dim"})
            for _ in range(5):
                verdicts.append({"criterion_index": idx, "verdict": "NOT_SATISFIED", "dimension": "dim"})

        data = CalibrationData(
            verdicts=verdicts,
            # Scores not correlated with word counts
            report_word_counts=[300, 100, 500, 200, 400],
            overall_scores=[0.5, 0.5, 0.5, 0.5, 0.5],
            judge_id="good_judge",
        )
        result = run_calibration(data)

        rec_text = " ".join(result.recommendations)
        assert "no significant biases" in rec_text.lower()

    def test_strict_judge_recommendation(self):
        """Very strict judge (all failing) triggers severity recommendation."""
        verdicts = []
        for idx in range(3):
            for _ in range(10):
                verdicts.append({"criterion_index": idx, "verdict": "NOT_SATISFIED", "dimension": "d"})

        data = CalibrationData(
            verdicts=verdicts,
            report_word_counts=[100, 200],
            overall_scores=[0.1, 0.1],
            judge_id="strict_judge",
        )
        result = run_calibration(data)
        assert result.severity_score == pytest.approx(1.0, abs=1e-9)
        rec_text = " ".join(result.recommendations)
        assert "strict" in rec_text.lower()

    def test_lenient_judge_recommendation(self):
        """Very lenient judge (all passing) triggers severity recommendation."""
        verdicts = []
        for idx in range(3):
            for _ in range(10):
                verdicts.append({"criterion_index": idx, "verdict": "SATISFIED", "dimension": "d"})

        data = CalibrationData(
            verdicts=verdicts,
            report_word_counts=[100, 200],
            overall_scores=[0.9, 0.9],
            judge_id="lenient_judge",
        )
        result = run_calibration(data)
        assert result.severity_score == pytest.approx(0.0, abs=1e-9)
        rec_text = " ".join(result.recommendations)
        assert "lenient" in rec_text.lower()


# ---------------------------------------------------------------------------
# 8. Dataclass construction
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_calibration_result_construction(self):
        cr = CalibrationResult(
            judge_id="j1",
            n_samples=42,
            position_bias=0.15,
            length_bias=-0.05,
            severity_score=0.55,
            dimension_biases={"coverage": 0.1, "depth": -0.1},
            recommendations=["All good."],
        )
        assert cr.judge_id == "j1"
        assert cr.n_samples == 42
        assert cr.position_bias == 0.15
        assert cr.length_bias == -0.05
        assert cr.severity_score == 0.55
        assert cr.dimension_biases["coverage"] == 0.1
        assert cr.recommendations == ["All good."]

    def test_calibration_data_construction(self):
        cd = CalibrationData(
            verdicts=[{"criterion_index": 0, "verdict": "SATISFIED", "dimension": "d"}],
            report_word_counts=[500],
            overall_scores=[0.7],
            judge_id="j2",
        )
        assert cd.judge_id == "j2"
        assert len(cd.verdicts) == 1
        assert cd.report_word_counts == [500]
        assert cd.overall_scores == [0.7]

    def test_calibration_data_default_judge_id(self):
        cd = CalibrationData(
            verdicts=[],
            report_word_counts=[],
            overall_scores=[],
        )
        assert cd.judge_id == "default"
