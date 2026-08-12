"""Tests for the V2 rubric system."""

import math

import pytest

from deep_research.evaluation.rubric_v2 import (
    ATTRIBUTION_CRITERIA,
    ANALYTICAL_DEPTH_CRITERIA,
    CITATION_CRITERIA,
    COVERAGE_CRITERIA,
    DIMENSION_WEIGHTS_V2,
    FACTUAL_ACCURACY_CRITERIA,
    INFORMATION_RECALL_CRITERIA,
    INSTRUCTION_FOLLOWING_CRITERIA,
    LOGICAL_COHERENCE_CRITERIA,
    ORGANIZATION_CRITERIA,
    Criterion,
    RubricV2,
    build_general_criteria,
    build_rubric_from_draco,
    build_rubric_from_test_query,
    build_rubric_v2,
    rubric_to_judge_prompt,
    rubric_to_judge_prompt_with_mapping,
)
from deep_research.evaluation.test_queries import TestQuery, get_query


# ── Criteria counts ──────────────────────────────────────────────────────────


class TestCriteriaCounts:
    """Verify the correct number of criteria per dimension."""

    def test_information_recall_has_4(self):
        assert len(INFORMATION_RECALL_CRITERIA) == 4

    def test_factual_accuracy_has_8(self):
        assert len(FACTUAL_ACCURACY_CRITERIA) == 8

    def test_instruction_following_has_4(self):
        assert len(INSTRUCTION_FOLLOWING_CRITERIA) == 4

    def test_citation_quality_has_4(self):
        assert len(CITATION_CRITERIA) == 4

    def test_coverage_has_5(self):
        assert len(COVERAGE_CRITERIA) == 5

    def test_analytical_depth_has_4(self):
        assert len(ANALYTICAL_DEPTH_CRITERIA) == 4

    def test_logical_coherence_has_3(self):
        assert len(LOGICAL_COHERENCE_CRITERIA) == 3

    def test_organization_has_4(self):
        assert len(ORGANIZATION_CRITERIA) == 4

    def test_attribution_has_2(self):
        assert len(ATTRIBUTION_CRITERIA) == 2

    def test_general_criteria_total(self):
        general = build_general_criteria()
        expected = (
            len(INFORMATION_RECALL_CRITERIA)
            + len(FACTUAL_ACCURACY_CRITERIA)
            + len(COVERAGE_CRITERIA)
            + len(ANALYTICAL_DEPTH_CRITERIA)
            + len(CITATION_CRITERIA)
            + len(LOGICAL_COHERENCE_CRITERIA)
            + len(ORGANIZATION_CRITERIA)
            + len(INSTRUCTION_FOLLOWING_CRITERIA)
            + len(ATTRIBUTION_CRITERIA)
        )
        assert len(general) == expected
        assert len(general) == 38

    def test_general_criteria_per_dimension(self):
        general = build_general_criteria()
        dims: dict[str, int] = {}
        for c in general:
            dims[c.dimension] = dims.get(c.dimension, 0) + 1
        assert dims["information_recall"] == 4
        assert dims["factual_accuracy"] == 8
        assert dims["coverage"] == 5
        assert dims["analytical_depth"] == 4
        assert dims["citation_quality"] == 4
        assert dims["logical_coherence"] == 3
        assert dims["organization"] == 4
        assert dims["instruction_following"] == 4
        assert dims["attribution_quality"] == 2


# ── Citation criteria do not require URL verification ────────────────────────


class TestCitationDesign:
    def test_no_url_verification_criteria(self):
        """Citation criteria should not require URL access or verification."""
        url_keywords = ["url", "verifiable", "link", "accessible", "fabricated"]
        for crit in CITATION_CRITERIA:
            lower = crit.text.lower()
            for kw in url_keywords:
                assert kw not in lower, (
                    f"Citation criterion should not require URL verification: "
                    f"'{crit.text}' contains '{kw}'"
                )


# ── All criteria validity ────────────────────────────────────────────────────


class TestCriteriaValidity:
    def test_all_criteria_have_non_empty_text(self):
        for c in build_general_criteria():
            assert c.text.strip(), f"Criterion has empty text: {c}"

    def test_all_criteria_have_valid_dimension(self):
        valid_dims = set(DIMENSION_WEIGHTS_V2.keys())
        for c in build_general_criteria():
            assert c.dimension in valid_dims, (
                f"Criterion dimension '{c.dimension}' not in weights: {c.text}"
            )

    def test_all_criteria_have_general_source(self):
        for c in build_general_criteria():
            assert c.source == "general"


# ── Dimension weights ────────────────────────────────────────────────────────


class TestDimensionWeights:
    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS_V2.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9), f"Weights sum to {total}"

    def test_all_weights_positive(self):
        for dim, w in DIMENSION_WEIGHTS_V2.items():
            assert w > 0, f"Weight for {dim} is not positive: {w}"

    def test_expected_dimensions_present(self):
        expected = {
            "information_recall",
            "factual_accuracy",
            "coverage",
            "analytical_depth",
            "citation_quality",
            "logical_coherence",
            "organization",
            "instruction_following",
            "attribution_quality",
        }
        assert set(DIMENSION_WEIGHTS_V2.keys()) == expected


# ── build_rubric_v2 ─────────────────────────────────────────────────────────


class TestBuildRubricV2:
    def test_basic_build_returns_general_criteria(self):
        rubric = build_rubric_v2("q1", "What is X?")
        assert rubric.query_id == "q1"
        assert rubric.query_text == "What is X?"
        assert rubric.total_criteria == 38

    def test_with_task_specific_adds_criteria(self):
        extra = [
            Criterion("Extra criterion 1", "coverage", source="task_specific"),
            Criterion("Extra criterion 2", "factual_accuracy", source="task_specific"),
        ]
        rubric = build_rubric_v2("q1", "What is X?", task_specific_criteria=extra)
        assert rubric.total_criteria == 40

    def test_with_custom_coverage_replaces_generic(self):
        custom_coverage = [
            Criterion("Covers topic A", "coverage", source="task_specific"),
            Criterion("Covers topic B", "coverage", source="task_specific"),
            Criterion("Covers topic C", "coverage", source="task_specific"),
        ]
        rubric = build_rubric_v2("q1", "What is X?", coverage_criteria=custom_coverage)
        coverage = rubric.get_criteria_by_dimension("coverage")
        assert len(coverage) == 3
        assert all(c.text.startswith("Covers topic") for c in coverage)
        # Total = 38 - 5 (generic coverage) + 3 (custom) = 36
        assert rubric.total_criteria == 36

    def test_custom_weights(self):
        custom_weights = {"dim_a": 0.5, "dim_b": 0.5}
        rubric = build_rubric_v2("q1", "X", dimension_weights=custom_weights)
        assert rubric.dimension_weights == custom_weights

    def test_default_weights_are_copy(self):
        rubric = build_rubric_v2("q1", "X")
        rubric.dimension_weights["factual_accuracy"] = 999
        # Original should be unmodified
        assert DIMENSION_WEIGHTS_V2["factual_accuracy"] == 0.20


# ── RubricV2 methods ─────────────────────────────────────────────────────────


class TestRubricV2Methods:
    def test_get_criteria_by_dimension(self):
        rubric = build_rubric_v2("q1", "X")
        fa = rubric.get_criteria_by_dimension("factual_accuracy")
        assert len(fa) == 8
        assert all(c.dimension == "factual_accuracy" for c in fa)

    def test_get_criteria_nonexistent_dimension(self):
        rubric = build_rubric_v2("q1", "X")
        result = rubric.get_criteria_by_dimension("nonexistent")
        assert result == []

    def test_get_dimensions(self):
        rubric = build_rubric_v2("q1", "X")
        dims = rubric.get_dimensions()
        assert "factual_accuracy" in dims
        assert "coverage" in dims
        assert "attribution_quality" in dims

    def test_total_criteria_property(self):
        rubric = build_rubric_v2("q1", "X")
        assert rubric.total_criteria == len(rubric.criteria)


# ── build_rubric_from_test_query ─────────────────────────────────────────────


class TestBuildFromTestQuery:
    def test_backward_compat_with_test_query(self):
        tq = get_query("q1_bert_vs_gpt")
        rubric = build_rubric_from_test_query(tq)
        assert rubric.query_id == "q1_bert_vs_gpt"
        assert rubric.query_text == tq.query

    def test_expected_elements_become_coverage(self):
        tq = TestQuery(
            id="test",
            query="Compare A and B",
            difficulty="simple",
            expected_elements=["Element 1", "Element 2", "Element 3"],
        )
        rubric = build_rubric_from_test_query(tq)
        coverage = rubric.get_criteria_by_dimension("coverage")
        assert len(coverage) == 3
        assert all("The report covers:" in c.text for c in coverage)

    def test_replaces_generic_coverage(self):
        tq = TestQuery(
            id="test",
            query="test",
            difficulty="simple",
            expected_elements=["A", "B"],
        )
        rubric = build_rubric_from_test_query(tq)
        coverage = rubric.get_criteria_by_dimension("coverage")
        # Should be exactly the 2 from test query, not the 5 generic ones
        assert len(coverage) == 2

    def test_other_dimensions_intact(self):
        tq = TestQuery(
            id="test",
            query="test",
            difficulty="simple",
            expected_elements=["A"],
        )
        rubric = build_rubric_from_test_query(tq)
        fa = rubric.get_criteria_by_dimension("factual_accuracy")
        assert len(fa) == 8
        inst = rubric.get_criteria_by_dimension("instruction_following")
        assert len(inst) == 4


# ── build_rubric_from_draco ──────────────────────────────────────────────────


class TestBuildFromDraco:
    def test_draco_criteria_preserved(self):
        sections = [{"id": "sec1", "title": "Architecture"}]
        criteria = [
            {
                "requirement": "Describes the transformer architecture",
                "weight": 5,
                "section": "sec1",
            },
            {
                "requirement": "Compares attention mechanisms",
                "weight": 3,
                "section": "sec1",
            },
        ]
        rubric = build_rubric_from_draco("d1", "Test query", sections, criteria)
        # Should have general criteria + 2 draco
        assert rubric.total_criteria == 38 + 2

    def test_draco_weights_preserved(self):
        sections = [{"id": "s1", "title": "S1"}]
        criteria = [
            {"requirement": "Criterion A", "weight": 10, "section": "s1"},
            {"requirement": "Criterion B", "weight": -5, "section": "s1"},
        ]
        rubric = build_rubric_from_draco("d1", "Q", sections, criteria)
        draco = [c for c in rubric.criteria if c.source == "draco"]
        assert len(draco) == 2
        weights = {c.text: c.weight for c in draco}
        assert weights["Criterion A"] == 10.0
        assert weights["Criterion B"] == -5.0

    def test_draco_source_tag(self):
        sections = []
        criteria = [{"requirement": "Test", "weight": 1}]
        rubric = build_rubric_from_draco("d1", "Q", sections, criteria)
        draco = [c for c in rubric.criteria if c.source == "draco"]
        assert len(draco) == 1

    def test_empty_draco(self):
        rubric = build_rubric_from_draco("d1", "Q", [], [])
        assert rubric.total_criteria == 38  # Just general criteria


# ── rubric_to_judge_prompt ───────────────────────────────────────────────────


class TestRubricToJudgePrompt:
    def test_produces_non_empty_string(self):
        rubric = build_rubric_v2("q1", "Test query")
        prompt = rubric_to_judge_prompt(rubric)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_all_criteria(self):
        rubric = build_rubric_v2("q1", "Test query")
        prompt = rubric_to_judge_prompt(rubric)
        # Check at least some criteria text appears
        for crit in rubric.criteria[:5]:
            assert crit.text in prompt

    def test_contains_dimension_headers(self):
        rubric = build_rubric_v2("q1", "Test query")
        prompt = rubric_to_judge_prompt(rubric)
        assert "Factual Accuracy" in prompt
        assert "Coverage" in prompt
        assert "Analytical Depth" in prompt

    def test_contains_json_format_instructions(self):
        rubric = build_rubric_v2("q1", "Test query")
        prompt = rubric_to_judge_prompt(rubric)
        assert "evaluations" in prompt
        assert "criterion_index" in prompt
        assert "SATISFIED" in prompt

    def test_contains_total_criteria_count(self):
        rubric = build_rubric_v2("q1", "Test query")
        prompt = rubric_to_judge_prompt(rubric)
        assert f"Total criteria: {rubric.total_criteria}" in prompt

    def test_contains_weight_percentages(self):
        rubric = build_rubric_v2("q1", "Test query")
        prompt = rubric_to_judge_prompt(rubric)
        assert "20%" in prompt  # information_recall / factual_accuracy
        assert "15%" in prompt  # analytical_depth
        assert "10%" in prompt  # coverage / citation / instruction
        assert "5%" in prompt   # logical_coherence / organization / attribution


# ── rubric_to_judge_prompt_with_mapping ─────────────────────────────────────


class TestRubricToJudgePromptWithMapping:
    def _make_rubric(self) -> RubricV2:
        return RubricV2(
            query_id="q1",
            query_text="Test query",
            criteria=[
                Criterion("Criterion A", "coverage"),
                Criterion("Criterion B", "factual_accuracy"),
                Criterion("Criterion C", "organization"),
                Criterion("Criterion D", "citation_quality"),
                Criterion("Criterion E", "analytical_depth"),
            ],
            dimension_weights={
                "coverage": 0.25,
                "factual_accuracy": 0.25,
                "organization": 0.15,
                "citation_quality": 0.15,
                "analytical_depth": 0.20,
            },
        )

    def test_different_seeds_produce_different_orderings(self):
        rubric = self._make_rubric()
        _, mapping_a = rubric_to_judge_prompt_with_mapping(rubric, seed=42)
        _, mapping_b = rubric_to_judge_prompt_with_mapping(rubric, seed=99)
        # With 5 criteria the probability of identical shuffle is 1/120.
        # Two different seeds should (almost certainly) give different orderings.
        assert mapping_a != mapping_b

    def test_same_seed_is_deterministic(self):
        rubric = self._make_rubric()
        prompt_a, mapping_a = rubric_to_judge_prompt_with_mapping(rubric, seed=42)
        prompt_b, mapping_b = rubric_to_judge_prompt_with_mapping(rubric, seed=42)
        assert prompt_a == prompt_b
        assert mapping_a == mapping_b

    def test_all_criteria_present_in_prompt(self):
        rubric = self._make_rubric()
        prompt, mapping = rubric_to_judge_prompt_with_mapping(rubric, seed=7)
        for crit in rubric.criteria:
            assert crit.text in prompt

    def test_mapping_is_valid_permutation(self):
        rubric = self._make_rubric()
        _, mapping = rubric_to_judge_prompt_with_mapping(rubric, seed=123)
        assert sorted(mapping) == list(range(len(rubric.criteria)))

    def test_mapping_length_matches_criteria(self):
        rubric = self._make_rubric()
        _, mapping = rubric_to_judge_prompt_with_mapping(rubric, seed=1)
        assert len(mapping) == len(rubric.criteria)

    def test_prompt_contains_orig_index_markers(self):
        rubric = self._make_rubric()
        prompt, _ = rubric_to_judge_prompt_with_mapping(rubric, seed=55)
        for i in range(len(rubric.criteria)):
            assert f"[orig={i}]" in prompt

    def test_prompt_contains_total_criteria(self):
        rubric = self._make_rubric()
        prompt, _ = rubric_to_judge_prompt_with_mapping(rubric, seed=10)
        assert f"Total criteria: {len(rubric.criteria)}" in prompt

    def test_prompt_contains_json_format(self):
        rubric = self._make_rubric()
        prompt, _ = rubric_to_judge_prompt_with_mapping(rubric, seed=10)
        assert "criterion_index" in prompt
        assert "evaluations" in prompt
