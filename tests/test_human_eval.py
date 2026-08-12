"""Tests for deep_research.evaluation.human_eval module."""

import pytest

from deep_research.evaluation.human_eval import (
    HumanVerdict,
    HumanEvalResult,
    JudgeHumanAgreement,
    compute_judge_human_agreement,
    generate_human_eval_report,
)


# ── HumanVerdict ─────────────────────────────────────────────────────────────


def test_human_verdict_construction():
    v = HumanVerdict(
        evaluator_id="eval_A",
        report_id="R-001",
        criterion="Claims are accurate",
        dimension="factual_accuracy",
        verdict="SATISFIED",
        confidence=0.9,
        comment="Verified via search",
        time_seconds=120,
    )
    assert v.evaluator_id == "eval_A"
    assert v.verdict == "SATISFIED"
    assert v.confidence == 0.9
    assert v.time_seconds == 120


def test_human_verdict_defaults():
    v = HumanVerdict(
        evaluator_id="eval_B",
        report_id="R-002",
        criterion="Coverage",
        dimension="coverage",
        verdict="NOT_SATISFIED",
        confidence=0.7,
    )
    assert v.comment == ""
    assert v.time_seconds == 0


# ── HumanEvalResult ─────────────────────────────────────────────────────────


def _make_verdicts(
    report_id: str,
    evaluators: list[str],
    dimension: str,
    results: list[str],
) -> list[HumanVerdict]:
    """Helper to create verdicts from a list of verdict strings."""
    verdicts = []
    for eval_id, verdict in zip(evaluators, results):
        verdicts.append(HumanVerdict(
            evaluator_id=eval_id,
            report_id=report_id,
            criterion=f"criterion_{dimension}",
            dimension=dimension,
            verdict=verdict,
            confidence=0.8,
        ))
    return verdicts


def test_n_evaluators():
    her = HumanEvalResult(
        report_id="R-001",
        pattern="p4_perspective_storm",
        query_id="q1",
        evaluators=["eval_A", "eval_B", "eval_C"],
        verdicts=[],
    )
    assert her.n_evaluators == 3


def test_avg_confidence():
    verdicts = [
        HumanVerdict("eval_A", "R-001", "c1", "d1", "SATISFIED", 0.8),
        HumanVerdict("eval_B", "R-001", "c1", "d1", "NOT_SATISFIED", 0.6),
    ]
    her = HumanEvalResult(
        report_id="R-001",
        pattern="p0_baseline",
        query_id="q1",
        evaluators=["eval_A", "eval_B"],
        verdicts=verdicts,
    )
    assert her.avg_confidence == pytest.approx(0.7)


def test_avg_confidence_empty():
    her = HumanEvalResult(
        report_id="R-001",
        pattern="p0_baseline",
        query_id="q1",
        evaluators=[],
        verdicts=[],
    )
    assert her.avg_confidence == 0.0


def test_dimension_scores_majority_vote():
    """3 evaluators: 2 SATISFIED + 1 NOT -> score = 2/3."""
    verdicts = _make_verdicts(
        "R-001", ["eval_A", "eval_B", "eval_C"],
        "factual_accuracy",
        ["SATISFIED", "SATISFIED", "NOT_SATISFIED"],
    )
    her = HumanEvalResult(
        report_id="R-001",
        pattern="p3_meridian",
        query_id="q2",
        evaluators=["eval_A", "eval_B", "eval_C"],
        verdicts=verdicts,
    )
    scores = her.dimension_scores()
    assert "factual_accuracy" in scores
    assert scores["factual_accuracy"] == pytest.approx(2 / 3)


def test_dimension_scores_multi_dimension():
    """Test with multiple dimensions."""
    verdicts = (
        _make_verdicts("R-001", ["A", "B"], "coverage", ["SATISFIED", "SATISFIED"])
        + _make_verdicts("R-001", ["A", "B"], "citation_quality", ["NOT_SATISFIED", "SATISFIED"])
    )
    her = HumanEvalResult(
        report_id="R-001",
        pattern="p0_baseline",
        query_id="q1",
        evaluators=["A", "B"],
        verdicts=verdicts,
    )
    scores = her.dimension_scores()
    assert scores["coverage"] == 1.0
    assert scores["citation_quality"] == 0.5


# ── JudgeHumanAgreement ─────────────────────────────────────────────────────


def test_judge_human_agreement_fields():
    jha = JudgeHumanAgreement(
        n_reports=10,
        overall_kappa=0.65,
        per_dimension_kappa={"coverage": 0.7, "factual_accuracy": 0.5},
        overall_correlation=0.8,
        dimension_correlation={"coverage": 0.85, "factual_accuracy": 0.6},
        agreement_rate=0.75,
        judge_bias=0.05,
    )
    assert jha.n_reports == 10
    assert jha.overall_kappa == 0.65
    assert jha.judge_bias == 0.05
    assert len(jha.per_dimension_kappa) == 2


# ── compute_judge_human_agreement ────────────────────────────────────────────


def test_compute_agreement_with_matched_data():
    """Test with matching judge and human results."""
    judge_scores = {
        "R-001": {"coverage": 0.8, "factual_accuracy": 0.3},
        "R-002": {"coverage": 0.6, "factual_accuracy": 0.5},
        "R-003": {"coverage": 0.9, "factual_accuracy": 0.2},
        "R-004": {"coverage": 0.7, "factual_accuracy": 0.4},
        "R-005": {"coverage": 0.5, "factual_accuracy": 0.6},
    }

    human_results = []
    for rid, (cov_verdicts, fa_verdicts) in {
        "R-001": (["SATISFIED", "SATISFIED"], ["NOT_SATISFIED", "NOT_SATISFIED"]),
        "R-002": (["SATISFIED", "NOT_SATISFIED"], ["SATISFIED", "NOT_SATISFIED"]),
        "R-003": (["SATISFIED", "SATISFIED"], ["NOT_SATISFIED", "NOT_SATISFIED"]),
        "R-004": (["SATISFIED", "NOT_SATISFIED"], ["NOT_SATISFIED", "SATISFIED"]),
        "R-005": (["NOT_SATISFIED", "SATISFIED"], ["SATISFIED", "SATISFIED"]),
    }.items():
        verdicts = (
            _make_verdicts(rid, ["A", "B"], "coverage", cov_verdicts)
            + _make_verdicts(rid, ["A", "B"], "factual_accuracy", fa_verdicts)
        )
        human_results.append(HumanEvalResult(
            report_id=rid,
            pattern="p0_baseline",
            query_id="q1",
            evaluators=["A", "B"],
            verdicts=verdicts,
        ))

    agreement = compute_judge_human_agreement(judge_scores, human_results)

    assert agreement.n_reports == 5
    assert isinstance(agreement.overall_kappa, float)
    assert isinstance(agreement.overall_correlation, float)
    assert isinstance(agreement.agreement_rate, float)
    assert 0.0 <= agreement.agreement_rate <= 1.0


def test_compute_agreement_no_matches():
    """When no report_ids match, return zero agreement."""
    judge_scores = {"R-999": {"coverage": 0.5}}
    human_results = [
        HumanEvalResult(
            report_id="R-001",
            pattern="p0_baseline",
            query_id="q1",
            evaluators=["A"],
            verdicts=[],
        )
    ]

    agreement = compute_judge_human_agreement(judge_scores, human_results)

    assert agreement.n_reports == 0
    assert agreement.overall_kappa == 0.0
    assert agreement.overall_correlation == 0.0
    assert agreement.agreement_rate == 0.0


# ── generate_human_eval_report ───────────────────────────────────────────────


def test_generate_report_contains_expected_sections():
    agreement = JudgeHumanAgreement(
        n_reports=5,
        overall_kappa=0.62,
        per_dimension_kappa={"coverage": 0.7, "factual_accuracy": 0.5},
        overall_correlation=0.78,
        dimension_correlation={"coverage": 0.85, "factual_accuracy": 0.6},
        agreement_rate=0.75,
        judge_bias=0.03,
    )

    report = generate_human_eval_report([], agreement)

    assert "# Human Evaluation Results" in report
    assert "## Summary" in report
    assert "Reports evaluated: 5" in report
    assert "0.620" in report  # kappa
    assert "0.780" in report  # correlation
    assert "75.0%" in report  # agreement rate
    assert "+0.030" in report  # bias
    assert "## Per-Dimension Agreement" in report
    assert "coverage" in report
    assert "factual_accuracy" in report


def test_generate_report_no_per_dimension():
    """When per_dimension_kappa is empty, skip that section."""
    agreement = JudgeHumanAgreement(
        n_reports=2,
        overall_kappa=0.5,
        per_dimension_kappa={},
        overall_correlation=0.6,
        dimension_correlation={},
        agreement_rate=0.7,
        judge_bias=-0.1,
    )

    report = generate_human_eval_report([], agreement)

    assert "# Human Evaluation Results" in report
    assert "## Summary" in report
    # Per-dimension section should not appear
    assert "## Per-Dimension Agreement" not in report
