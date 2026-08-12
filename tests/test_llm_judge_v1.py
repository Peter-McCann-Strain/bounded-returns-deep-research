"""Tests for the V1 LLM judge module (backward compat with V2 alignment).

Verifies that llm_judge.py DIMENSION_WEIGHTS are now aligned with rubric_v2's
9-dimension system, and that GENERAL_CRITERIA cover all 9 dimensions.
"""

from deep_research.evaluation.llm_judge import (
    DIMENSION_WEIGHTS,
    GENERAL_CRITERIA,
    build_rubric,
)
from deep_research.evaluation.rubric_v2 import DIMENSION_WEIGHTS_V2


class TestDimensionWeightsV1V2Alignment:
    def test_v1_weights_match_v2(self):
        """V1 DIMENSION_WEIGHTS should be identical to V2."""
        assert DIMENSION_WEIGHTS == DIMENSION_WEIGHTS_V2

    def test_has_nine_dimensions(self):
        assert len(DIMENSION_WEIGHTS) == 9

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_includes_new_dimensions(self):
        assert "information_recall" in DIMENSION_WEIGHTS
        assert "logical_coherence" in DIMENSION_WEIGHTS
        assert "attribution_quality" in DIMENSION_WEIGHTS


class TestGeneralCriteriaV1:
    def test_general_criteria_cover_all_non_query_specific_dimensions(self):
        """GENERAL_CRITERIA should cover all V2 dimensions except 'coverage'.

        'coverage' criteria are query-specific (generated per expected element
        in build_rubric), so they intentionally do not appear in GENERAL_CRITERIA.
        """
        dims_covered = {dim for _, dim in GENERAL_CRITERIA}
        for dim in DIMENSION_WEIGHTS_V2:
            if dim == "coverage":
                continue  # coverage is query-specific, tested in TestBuildRubric
            assert dim in dims_covered, f"No GENERAL_CRITERIA for dimension '{dim}'"

    def test_has_information_recall_criteria(self):
        ir_criteria = [c for c, d in GENERAL_CRITERIA if d == "information_recall"]
        assert len(ir_criteria) >= 2

    def test_has_logical_coherence_criteria(self):
        lc_criteria = [c for c, d in GENERAL_CRITERIA if d == "logical_coherence"]
        assert len(lc_criteria) >= 2

    def test_has_attribution_quality_criteria(self):
        aq_criteria = [c for c, d in GENERAL_CRITERIA if d == "attribution_quality"]
        assert len(aq_criteria) >= 1


class TestBuildRubric:
    def test_build_rubric_returns_tuples(self):
        rubric = build_rubric("Test query", ["element A", "element B"])
        assert isinstance(rubric, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in rubric)

    def test_build_rubric_includes_coverage(self):
        rubric = build_rubric("Test query", ["element A"])
        coverage_criteria = [c for c, d in rubric if d == "coverage"]
        assert len(coverage_criteria) >= 1

    def test_build_rubric_includes_all_general_criteria(self):
        rubric = build_rubric("Test query", [])
        rubric_texts = {c for c, _ in rubric}
        for gen_text, _ in GENERAL_CRITERIA:
            assert gen_text in rubric_texts

    def test_build_rubric_includes_new_dimensions(self):
        rubric = build_rubric("Test query", ["element A"])
        dims_in_rubric = {d for _, d in rubric}
        assert "information_recall" in dims_in_rubric
        assert "logical_coherence" in dims_in_rubric
        assert "attribution_quality" in dims_in_rubric
