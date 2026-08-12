"""Tests for the multi-judge ensemble evaluator."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from deep_research.evaluation.multi_judge import (
    CriterionVerdict,
    EnsembleResult,
    JudgeConfig,
    JudgePassResult,
    MultiJudge,
    _JUDGE_SYSTEM_PROMPT,
    cohens_kappa,
    fleiss_kappa,
    krippendorffs_alpha_binary,
)
from deep_research.evaluation.rubric_v2 import (
    Criterion,
    RubricV2,
    build_rubric_v2,
)


# ── Cohen's Kappa ────────────────────────────────────────────────────────────


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [True, True, False, False, True]
        b = [True, True, False, False, True]
        assert math.isclose(cohens_kappa(a, b), 1.0, abs_tol=1e-9)

    def test_perfect_disagreement(self):
        a = [True, True, False, False]
        b = [False, False, True, True]
        # Complete disagreement: kappa = -1.0
        assert math.isclose(cohens_kappa(a, b), -1.0, abs_tol=1e-9)

    def test_random_agreement_near_zero(self):
        """With mixed ratings where observed agreement ~ expected, kappa ~ 0."""
        # Constructed so p_o ~ p_e
        a = [True, False, True, False, True, False, True, False]
        b = [True, True, False, False, True, True, False, False]
        kappa = cohens_kappa(a, b)
        # Not exactly 0, but should be near 0
        assert -0.5 < kappa < 0.5

    def test_known_example(self):
        """Textbook example: 20 items, 2 raters.
        Rater A: 10 yes, 10 no. Rater B: 12 yes, 8 no.
        Agreement on 15 items (8 yes-yes, 7 no-no).
        """
        a = [True] * 10 + [False] * 10
        b = [True] * 8 + [False] * 2 + [True] * 4 + [False] * 6
        # p_o = 14/20 = 0.7
        # p_a1=10/20=0.5, p_b1=12/20=0.6
        # p_e = 0.5*0.6 + 0.5*0.4 = 0.5
        # kappa = (0.7 - 0.5)/(1 - 0.5) = 0.4
        kappa = cohens_kappa(a, b)
        assert math.isclose(kappa, 0.4, abs_tol=1e-9)

    def test_empty_lists(self):
        assert cohens_kappa([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([True, False], [True])

    def test_all_true(self):
        a = [True, True, True]
        b = [True, True, True]
        # Degenerate: p_e = 1.0, p_o = 1.0 -> kappa = 1.0
        assert math.isclose(cohens_kappa(a, b), 1.0, abs_tol=1e-9)

    def test_all_same_with_disagreement(self):
        # A always says True, B always says False
        a = [True, True, True]
        b = [False, False, False]
        kappa = cohens_kappa(a, b)
        # p_o = 0, p_e = 0 (since p_a1=1,p_b1=0 => p_e = 1*0+0*1 = 0)
        # Degenerate: (0-0)/(1-0) = 0
        assert math.isclose(kappa, 0.0, abs_tol=1e-9)


# ── Fleiss' Kappa ────────────────────────────────────────────────────────────


class TestFleissKappa:
    def test_perfect_agreement(self):
        # 3 raters, 4 items, all agree
        # Each row: [0, 3] (all say category 1) or [3, 0]
        matrix = np.array([
            [0, 3],  # All say yes
            [3, 0],  # All say no
            [0, 3],  # All say yes
            [3, 0],  # All say no
        ])
        assert math.isclose(fleiss_kappa(matrix), 1.0, abs_tol=1e-9)

    def test_known_textbook_example(self):
        """Fleiss' original paper example (simplified).
        10 items, 6 raters, 2 categories.
        """
        matrix = np.array([
            [0, 6],
            [0, 6],
            [0, 6],
            [6, 0],
            [6, 0],
            [3, 3],
            [1, 5],
            [5, 1],
            [2, 4],
            [4, 2],
        ])
        kappa = fleiss_kappa(matrix)
        # Should be moderate agreement
        assert -1.0 <= kappa <= 1.0
        # Compute manually: P_bar and P_e_bar
        n = 6
        p_j = matrix.sum(axis=0) / (10 * 6)  # [27/60, 33/60]
        P_i = ((matrix ** 2).sum(axis=1) - n) / (n * (n - 1))
        P_bar = P_i.mean()
        P_e_bar = (p_j ** 2).sum()
        expected_kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        assert math.isclose(kappa, expected_kappa, abs_tol=1e-9)

    def test_empty_matrix(self):
        matrix = np.array([]).reshape(0, 2)
        assert fleiss_kappa(matrix) == 0.0

    def test_single_rater(self):
        # Only 1 rater -> n_raters < 2 -> return 0
        matrix = np.array([[1, 0], [0, 1], [1, 0]])
        assert fleiss_kappa(matrix) == 0.0


# ── Krippendorff's Alpha ─────────────────────────────────────────────────────


class TestKrippendorffsAlpha:
    def test_perfect_agreement(self):
        # 3 raters, 5 items, all agree
        data = np.array([
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
        ], dtype=float)
        assert math.isclose(krippendorffs_alpha_binary(data), 1.0, abs_tol=1e-9)

    def test_perfect_disagreement(self):
        # 2 raters, always opposite
        data = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=float)
        alpha = krippendorffs_alpha_binary(data)
        assert alpha < 0  # Worse than chance

    def test_with_missing_data(self):
        data = np.array([
            [1, 0, np.nan, 1],
            [1, 0, 1, np.nan],
            [1, 0, 1, 1],
        ])
        alpha = krippendorffs_alpha_binary(data)
        assert -1.0 <= alpha <= 1.0

    def test_empty_data(self):
        data = np.array([]).reshape(2, 0)
        assert krippendorffs_alpha_binary(data) == 0.0

    def test_known_example(self):
        """Simple known case: 2 raters, 4 items, 1 disagreement."""
        data = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 1],  # disagrees on item 4
        ], dtype=float)
        alpha = krippendorffs_alpha_binary(data)
        # Should be positive (mostly agree) but < 1
        assert 0.0 < alpha < 1.0


# ── MultiJudge aggregate_ensemble ────────────────────────────────────────────


def _make_pass(
    judge_label: str,
    pass_number: int,
    verdicts_bools: list[bool],
    criteria_texts: list[str],
    dimensions: list[str],
) -> JudgePassResult:
    """Helper to create a JudgePassResult with given verdicts."""
    verdicts = [
        CriterionVerdict(
            criterion_text=ct,
            dimension=dim,
            satisfied=sat,
            reasoning="test",
        )
        for ct, dim, sat in zip(criteria_texts, dimensions, verdicts_bools)
    ]
    return JudgePassResult(
        judge_label=judge_label,
        pass_number=pass_number,
        verdicts=verdicts,
        dimension_scores={},
        overall_score=0.0,
        raw_response="",
    )


class TestAggregateEnsemble:
    def setup_method(self):
        self.judge_cfg = JudgeConfig(
            label="test", model="m", endpoint="https://e", api_key="k"
        )
        self.mj = MultiJudge(judges=[self.judge_cfg], passes_per_judge=3)

    def test_majority_vote_all_agree(self):
        criteria = ["c1", "c2", "c3"]
        dims = ["coverage", "coverage", "factual_accuracy"]
        rubric = RubricV2(
            query_id="q1",
            query_text="test",
            criteria=[
                Criterion("c1", "coverage"),
                Criterion("c2", "coverage"),
                Criterion("c3", "factual_accuracy"),
            ],
            dimension_weights={"coverage": 0.5, "factual_accuracy": 0.5},
        )

        passes = [
            _make_pass("j1", 0, [True, False, True], criteria, dims),
            _make_pass("j1", 1, [True, False, True], criteria, dims),
            _make_pass("j1", 2, [True, False, True], criteria, dims),
        ]

        overall, dim_scores = self.mj._aggregate_ensemble(passes, rubric)
        assert dim_scores["coverage"] == 0.5  # 1/2 met
        assert dim_scores["factual_accuracy"] == 1.0  # 1/1 met
        assert math.isclose(overall, 0.5 * 0.5 + 0.5 * 1.0, abs_tol=1e-9)

    def test_majority_vote_tie_breaking(self):
        """With 3 passes, 2 SATISFIED beats 1 NOT_SATISFIED."""
        criteria = ["c1"]
        dims = ["coverage"]
        rubric = RubricV2(
            query_id="q1",
            query_text="test",
            criteria=[Criterion("c1", "coverage")],
            dimension_weights={"coverage": 1.0},
        )

        passes = [
            _make_pass("j1", 0, [True], criteria, dims),
            _make_pass("j1", 1, [True], criteria, dims),
            _make_pass("j1", 2, [False], criteria, dims),
        ]

        overall, dim_scores = self.mj._aggregate_ensemble(passes, rubric)
        assert dim_scores["coverage"] == 1.0  # 2/3 voted satisfied -> MET

    def test_empty_passes(self):
        rubric = build_rubric_v2("q1", "test")
        overall, dims = self.mj._aggregate_ensemble([], rubric)
        assert overall == 0.0
        assert dims == {}


# ── Intra-judge consistency ──────────────────────────────────────────────────


class TestIntraJudgeConsistency:
    def setup_method(self):
        self.mj = MultiJudge(
            judges=[JudgeConfig("j", "m", "https://e", "k")],
            passes_per_judge=3,
        )

    def test_identical_passes_zero_flip_rate(self):
        criteria = ["c1", "c2", "c3"]
        dims = ["d1", "d1", "d2"]
        passes = [
            _make_pass("j1", i, [True, False, True], criteria, dims)
            for i in range(3)
        ]
        result = self.mj._compute_intra_judge_consistency(passes)
        assert result["j1"] == 0.0

    def test_all_flipping_passes(self):
        criteria = ["c1"]
        dims = ["d1"]
        passes = [
            _make_pass("j1", 0, [True], criteria, dims),
            _make_pass("j1", 1, [False], criteria, dims),
            _make_pass("j1", 2, [True], criteria, dims),
        ]
        result = self.mj._compute_intra_judge_consistency(passes)
        assert result["j1"] == 1.0  # c1 flipped

    def test_partial_flipping(self):
        criteria = ["c1", "c2"]
        dims = ["d1", "d1"]
        passes = [
            _make_pass("j1", 0, [True, False], criteria, dims),
            _make_pass("j1", 1, [True, True], criteria, dims),   # c2 flipped
            _make_pass("j1", 2, [True, False], criteria, dims),
        ]
        result = self.mj._compute_intra_judge_consistency(passes)
        assert math.isclose(result["j1"], 0.5, abs_tol=1e-9)  # 1/2 flipped

    def test_single_pass_zero(self):
        criteria = ["c1"]
        dims = ["d1"]
        passes = [_make_pass("j1", 0, [True], criteria, dims)]
        result = self.mj._compute_intra_judge_consistency(passes)
        assert result["j1"] == 0.0


# ── Inter-judge agreement ────────────────────────────────────────────────────


class TestInterJudgeAgreement:
    def setup_method(self):
        self.mj = MultiJudge(
            judges=[
                JudgeConfig("j1", "m", "https://e", "k"),
                JudgeConfig("j2", "m", "https://e", "k"),
            ],
            passes_per_judge=1,
        )

    def test_perfect_agreement_two_judges(self):
        criteria = ["c1", "c2", "c3"]
        dims = ["d1", "d1", "d1"]
        passes = [
            _make_pass("j1", 0, [True, False, True], criteria, dims),
            _make_pass("j2", 0, [True, False, True], criteria, dims),
        ]
        kappa = self.mj._compute_inter_judge_agreement(passes)
        assert math.isclose(kappa, 1.0, abs_tol=1e-9)

    def test_single_judge_trivial(self):
        mj_single = MultiJudge(
            judges=[JudgeConfig("j1", "m", "https://e", "k")],
            passes_per_judge=1,
        )
        criteria = ["c1"]
        dims = ["d1"]
        passes = [_make_pass("j1", 0, [True], criteria, dims)]
        kappa = mj_single._compute_inter_judge_agreement(passes)
        assert kappa == 1.0

    def test_disagreement(self):
        criteria = ["c1", "c2", "c3", "c4"]
        dims = ["d1"] * 4
        passes = [
            _make_pass("j1", 0, [True, True, False, False], criteria, dims),
            _make_pass("j2", 0, [False, False, True, True], criteria, dims),
        ]
        kappa = self.mj._compute_inter_judge_agreement(passes)
        assert kappa < 0  # Complete disagreement


# ── Data class construction ──────────────────────────────────────────────────


class TestDataClasses:
    def test_ensemble_result_construction(self):
        er = EnsembleResult(
            query_id="q1",
            pattern_name="p0",
            individual_passes=[],
            ensemble_overall=0.75,
            ensemble_dimensions={"coverage": 0.8},
            intra_judge_consistency={"j1": 0.1},
            inter_judge_agreement=0.85,
            krippendorffs_alpha=0.82,
            per_dimension_agreement={"coverage": 0.9},
            n_judges=2,
            n_passes_per_judge=3,
            total_evaluations=6,
        )
        assert er.query_id == "q1"
        assert er.ensemble_overall == 0.75
        assert er.n_judges == 2
        assert er.total_evaluations == 6

    def test_judge_config_construction(self):
        from deep_research.config import AZURE_OPENAI_API_VERSION, JUDGE
        jc = JudgeConfig(
            label="gpt52",
            model="gpt-5.2",
            endpoint="https://api.openai.azure.com",
            api_key="sk-test",
        )
        assert jc.label == "gpt52"
        assert jc.temperature == JUDGE.temperature
        assert jc.max_tokens == JUDGE.max_tokens
        assert jc.api_version == AZURE_OPENAI_API_VERSION

    def test_judge_config_custom_values(self):
        jc = JudgeConfig(
            label="custom",
            model="gpt-4o",
            endpoint="https://custom.azure.com",
            api_key="key",
            api_version="2024-12-01",
            temperature=0.3,
            max_tokens=4096,
        )
        assert jc.temperature == 0.3
        assert jc.max_tokens == 4096
        assert jc.api_version == "2024-12-01"

    def test_criterion_verdict_construction(self):
        cv = CriterionVerdict(
            criterion_text="Test criterion",
            dimension="coverage",
            satisfied=True,
            reasoning="Well covered",
            weight=2.0,
        )
        assert cv.satisfied is True
        assert cv.weight == 2.0

    def test_judge_pass_result_construction(self):
        jpr = JudgePassResult(
            judge_label="j1",
            pass_number=0,
            verdicts=[],
            dimension_scores={"coverage": 0.5},
            overall_score=0.5,
            raw_response="{}",
        )
        assert jpr.judge_label == "j1"
        assert jpr.overall_score == 0.5


# ── Integration test with mocked LLM calls ──────────────────────────────────


class TestMultiJudgeIntegration:
    @pytest.mark.asyncio
    async def test_evaluate_with_mocked_llm(self):
        """Full end-to-end test with mocked LLM responses."""
        import json

        judge1 = JudgeConfig("j1", "gpt-5.2", "https://e1", "k1")
        judge2 = JudgeConfig("j2", "gpt-4o", "https://e2", "k2")
        mj = MultiJudge(judges=[judge1, judge2], passes_per_judge=2)

        rubric = RubricV2(
            query_id="q1",
            query_text="Test query",
            criteria=[
                Criterion("Criterion A", "coverage"),
                Criterion("Criterion B", "factual_accuracy"),
            ],
            dimension_weights={"coverage": 0.5, "factual_accuracy": 0.5},
        )

        # Build a mock response: both criteria SATISFIED
        mock_response_satisfied = json.dumps({
            "evaluations": [
                {
                    "criterion_index": 0,
                    "verdict": "SATISFIED",
                    "evidence": "Found in report",
                    "reasoning": "Well covered",
                },
                {
                    "criterion_index": 1,
                    "verdict": "SATISFIED",
                    "evidence": "Accurate",
                    "reasoning": "Factually correct",
                },
            ]
        })

        # Mock the _call_with_retry method
        with patch.object(
            mj, "_call_with_retry", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_response_satisfied

            result = await mj.evaluate(
                query_id="q1",
                query_text="Test query",
                pattern_name="p0",
                report_text="This is a test report.",
                rubric=rubric,
            )

        assert isinstance(result, EnsembleResult)
        assert result.query_id == "q1"
        assert result.pattern_name == "p0"
        assert result.n_judges == 2
        assert result.n_passes_per_judge == 2
        assert result.total_evaluations == 4
        # All passes returned SATISFIED for both criteria
        assert result.ensemble_overall == 1.0
        assert result.ensemble_dimensions["coverage"] == 1.0
        assert result.ensemble_dimensions["factual_accuracy"] == 1.0
        # Perfect intra consistency (all passes identical)
        for label, flip_rate in result.intra_judge_consistency.items():
            assert flip_rate == 0.0
        # Perfect inter agreement
        assert math.isclose(result.inter_judge_agreement, 1.0, abs_tol=1e-9)
        # Perfect alpha
        assert math.isclose(result.krippendorffs_alpha, 1.0, abs_tol=1e-9)

    @pytest.mark.asyncio
    async def test_evaluate_with_mixed_verdicts(self):
        """Test with judges giving different verdicts."""
        import json

        judge1 = JudgeConfig("j1", "m1", "https://e1", "k1")
        judge2 = JudgeConfig("j2", "m2", "https://e2", "k2")
        mj = MultiJudge(judges=[judge1, judge2], passes_per_judge=1)

        rubric = RubricV2(
            query_id="q1",
            query_text="Test",
            criteria=[
                Criterion("C1", "dim1"),
                Criterion("C2", "dim1"),
            ],
            dimension_weights={"dim1": 1.0},
        )

        satisfied_response = json.dumps({
            "evaluations": [
                {"criterion_index": 0, "verdict": "SATISFIED", "reasoning": "ok"},
                {"criterion_index": 1, "verdict": "SATISFIED", "reasoning": "ok"},
            ]
        })
        mixed_response = json.dumps({
            "evaluations": [
                {"criterion_index": 0, "verdict": "SATISFIED", "reasoning": "ok"},
                {"criterion_index": 1, "verdict": "NOT_SATISFIED", "reasoning": "no"},
            ]
        })

        call_count = 0

        async def mock_call(client, judge, messages):
            nonlocal call_count
            call_count += 1
            if judge.label == "j1":
                return satisfied_response
            return mixed_response

        with patch.object(mj, "_call_with_retry", side_effect=mock_call):
            result = await mj.evaluate(
                query_id="q1",
                query_text="Test",
                pattern_name="p0",
                report_text="Report text",
                rubric=rubric,
            )

        assert result.total_evaluations == 2
        # C1: both SATISFIED -> MET. C2: j1 SATISFIED, j2 NOT -> MET (1>0.5*2=1, tie: 1 > 1.0? No, 1 > 1 is false)
        # Actually: votes for C2 = [True, False], len=2, satisfied_count=1, 1 > 2/2=1.0 is False -> NOT MET
        # So C1 MET, C2 NOT MET -> dim1 = 1/2 = 0.5
        assert result.ensemble_dimensions["dim1"] == 0.5


# ── Anchor examples in judge system prompt ──────────────────────────────────


class TestAnchorExamples:
    """Verify that the judge system prompt contains calibration examples."""

    def test_calibration_section_header(self):
        assert "## Calibration Examples" in _JUDGE_SYSTEM_PROMPT

    def test_satisfied_example_present(self):
        assert '"verdict": "SATISFIED"' in _JUDGE_SYSTEM_PROMPT

    def test_not_satisfied_example_present(self):
        assert '"verdict": "NOT_SATISFIED"' in _JUDGE_SYSTEM_PROMPT

    def test_factual_accuracy_example(self):
        assert "Vaswani et al." in _JUDGE_SYSTEM_PROMPT

    def test_citation_quality_example(self):
        assert "Research suggests that" in _JUDGE_SYSTEM_PROMPT

    def test_coverage_example(self):
        assert "No limitations, drawbacks, or counterarguments" in _JUDGE_SYSTEM_PROMPT

    def test_three_examples(self):
        # Each example has a "criterion_index" field
        count = _JUDGE_SYSTEM_PROMPT.count('"criterion_index"')
        assert count == 3

    def test_both_verdict_types_in_examples(self):
        """Ensure at least one SATISFIED and one NOT_SATISFIED example."""
        satisfied_count = _JUDGE_SYSTEM_PROMPT.count('"verdict": "SATISFIED"')
        not_satisfied_count = _JUDGE_SYSTEM_PROMPT.count('"verdict": "NOT_SATISFIED"')
        assert satisfied_count >= 1
        assert not_satisfied_count >= 1


# ── Criteria randomization via _run_single_pass ─────────────────────────────


class TestCriteriaRandomization:
    """Verify that _parse_verdicts with criterion_mapping correctly remaps."""

    def setup_method(self):
        self.mj = MultiJudge(
            judges=[JudgeConfig("j", "m", "https://e", "k")],
            passes_per_judge=1,
        )

    def test_parse_verdicts_with_mapping(self):
        """When mapping is [2, 0, 1], criterion_index 0 should resolve to rubric.criteria[2]."""
        import json

        rubric = RubricV2(
            query_id="q1",
            query_text="Test",
            criteria=[
                Criterion("Alpha", "dim_a"),
                Criterion("Beta", "dim_b"),
                Criterion("Gamma", "dim_c"),
            ],
            dimension_weights={"dim_a": 0.4, "dim_b": 0.3, "dim_c": 0.3},
        )

        # mapping: shuffled pos 0 -> original 2 ("Gamma")
        #          shuffled pos 1 -> original 0 ("Alpha")
        #          shuffled pos 2 -> original 1 ("Beta")
        mapping = [2, 0, 1]

        raw = json.dumps({
            "evaluations": [
                {"criterion_index": 0, "verdict": "SATISFIED", "reasoning": "ok"},
                {"criterion_index": 1, "verdict": "NOT_SATISFIED", "reasoning": "no"},
                {"criterion_index": 2, "verdict": "SATISFIED", "reasoning": "ok"},
            ]
        })

        verdicts = self.mj._parse_verdicts(raw, rubric, criterion_mapping=mapping)
        # criterion_index 0 -> mapping[0]=2 -> rubric.criteria[2] = "Gamma"
        assert verdicts[0].criterion_text == "Gamma"
        assert verdicts[0].satisfied is True
        # criterion_index 1 -> mapping[1]=0 -> rubric.criteria[0] = "Alpha"
        assert verdicts[1].criterion_text == "Alpha"
        assert verdicts[1].satisfied is False
        # criterion_index 2 -> mapping[2]=1 -> rubric.criteria[1] = "Beta"
        assert verdicts[2].criterion_text == "Beta"
        assert verdicts[2].satisfied is True

    def test_parse_verdicts_without_mapping_unchanged(self):
        """Without mapping, behaviour is the same as before."""
        import json

        rubric = RubricV2(
            query_id="q1",
            query_text="Test",
            criteria=[
                Criterion("Alpha", "dim_a"),
                Criterion("Beta", "dim_b"),
            ],
            dimension_weights={"dim_a": 0.5, "dim_b": 0.5},
        )

        raw = json.dumps({
            "evaluations": [
                {"criterion_index": 0, "verdict": "SATISFIED", "reasoning": "ok"},
                {"criterion_index": 1, "verdict": "NOT_SATISFIED", "reasoning": "no"},
            ]
        })

        verdicts = self.mj._parse_verdicts(raw, rubric, criterion_mapping=None)
        assert verdicts[0].criterion_text == "Alpha"
        assert verdicts[1].criterion_text == "Beta"

    def test_parse_verdicts_mapping_out_of_range_falls_back(self):
        """If criterion_index exceeds mapping length, falls back to criteria[0]."""
        import json

        rubric = RubricV2(
            query_id="q1",
            query_text="Test",
            criteria=[
                Criterion("Alpha", "dim_a"),
                Criterion("Beta", "dim_b"),
            ],
            dimension_weights={"dim_a": 0.5, "dim_b": 0.5},
        )

        mapping = [1, 0]

        raw = json.dumps({
            "evaluations": [
                {"criterion_index": 99, "verdict": "SATISFIED", "reasoning": "ok"},
            ]
        })

        verdicts = self.mj._parse_verdicts(raw, rubric, criterion_mapping=mapping)
        # idx=99 out of mapping range -> orig_idx=0 -> criteria[0] = "Alpha"
        assert verdicts[0].criterion_text == "Alpha"

    @pytest.mark.asyncio
    async def test_run_single_pass_uses_shuffled_criteria(self):
        """End-to-end: _run_single_pass produces correct verdicts with shuffled criteria."""
        import json

        judge = JudgeConfig("j1", "m1", "https://e1", "k1")
        mj = MultiJudge(judges=[judge], passes_per_judge=1)

        rubric = RubricV2(
            query_id="q1",
            query_text="Test query",
            criteria=[
                Criterion("First", "coverage"),
                Criterion("Second", "factual_accuracy"),
                Criterion("Third", "organization"),
            ],
            dimension_weights={
                "coverage": 0.4,
                "factual_accuracy": 0.3,
                "organization": 0.3,
            },
        )

        # The mock will return all SATISFIED regardless of order
        mock_response = json.dumps({
            "evaluations": [
                {"criterion_index": 0, "verdict": "SATISFIED", "reasoning": "a"},
                {"criterion_index": 1, "verdict": "SATISFIED", "reasoning": "b"},
                {"criterion_index": 2, "verdict": "SATISFIED", "reasoning": "c"},
            ]
        })

        with patch.object(
            mj, "_call_with_retry", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_response

            result = await mj._run_single_pass(
                judge, 0, "q1", "Test query", "Report text", rubric
            )

        # All 3 criteria should be present as SATISFIED
        assert len(result.verdicts) == 3
        verdict_texts = {v.criterion_text for v in result.verdicts}
        assert verdict_texts == {"First", "Second", "Third"}
        assert all(v.satisfied for v in result.verdicts)
