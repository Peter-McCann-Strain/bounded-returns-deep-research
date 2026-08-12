"""Tests for new benchmark converters (ResearchRubrics, DRB-II, DR.BENCH)
and their corresponding query registry loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research.evaluation.rubric_v2 import (
    Criterion,
    RubricV2,
    DIMENSION_WEIGHTS_BY_SOURCE,
    build_general_criteria,
)
from deep_research.evaluation.rubric_converters import (
    _infer_dimension,
    research_rubrics_to_rubric_v2,
    drb2_to_rubric_v2,
    drbench_to_rubric_v2,
    DRB2_CATEGORY_MAP,
)
from deep_research.evaluation.query_registry import (
    EvalQuery,
    QueryRegistry,
    classify_difficulty,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_research_rubrics_criteria() -> list[dict]:
    """Sample ResearchRubrics criteria list."""
    return [
        {
            "criterion_text": "Report covers the key technical innovations",
            "dimension_hint": "coverage",
            "weight": 2.0,
        },
        {
            "criterion_text": "All factual claims are accurate and verifiable",
            "dimension_hint": "factual_accuracy",
            "weight": 1.5,
        },
        {
            "criterion_text": "Citations are properly formatted",
            "dimension_hint": "citation_quality",
            "weight": 1.0,
        },
        {
            "criterion_text": "Analysis compares multiple approaches",
            "dimension_hint": "analytical_depth",
            "weight": 1.0,
        },
    ]


@pytest.fixture
def sample_drb2_items() -> list[dict]:
    """Sample DRB-II rubric items."""
    return [
        {
            "question": "Does the report recall the key finding about X?",
            "category": "information_recall",
            "weight": 1.0,
        },
        {
            "question": "Is the claim about Y factually accurate?",
            "category": "factual_accuracy",
            "weight": 1.0,
        },
        {
            "question": "Does the report cover all requested topics?",
            "category": "completeness",
            "weight": 2.0,
        },
        {
            "question": "Is the report organized with clear sections?",
            "category": "organization",
            "weight": 1.0,
        },
        {
            "question": "Does the report cite its sources?",
            "category": "citation",
            "weight": 1.0,
        },
    ]


@pytest.fixture
def sample_drbench_facts() -> list[str]:
    """Sample DR.BENCH reference facts."""
    return [
        "The Earth's core temperature exceeds 5000 degrees Celsius",
        "Iron constitutes approximately 85% of the Earth's core",
        "The inner core is solid while the outer core is liquid",
    ]


@pytest.fixture
def sample_drbench_axes() -> list[str]:
    """Sample DR.BENCH evaluation axes."""
    return [
        "Verify the factual accuracy of geological claims",
        "Assess the organization and structure of the report",
        "Check that sources are properly cited and attributed",
    ]


@pytest.fixture
def research_rubrics_json_file(tmp_path: Path) -> Path:
    """Write a sample ResearchRubrics JSON file."""
    data = [
        {
            "id": "rr-001",
            "prompt": "Analyze the impact of transformer models on NLP",
            "domain": "machine_learning",
            "criteria": [
                {
                    "criterion_text": "Discusses attention mechanism",
                    "dimension_hint": "coverage",
                    "weight": 1.0,
                },
                {
                    "criterion_text": "Compares with RNNs and LSTMs",
                    "dimension_hint": "analytical_depth",
                    "weight": 1.5,
                },
            ],
        },
        {
            "id": "rr-002",
            "prompt": "What is gradient descent?",
            "domain": "optimization",
            "criteria": [
                {
                    "criterion_text": "Defines gradient descent correctly",
                    "dimension_hint": "factual_accuracy",
                    "weight": 2.0,
                },
            ],
        },
    ]
    path = tmp_path / "research_rubrics_queries.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def drb2_json_file(tmp_path: Path) -> Path:
    """Write a sample DRB-II JSON file."""
    data = [
        {
            "task_id": "drb2-001",
            "query": "Compare and contrast supervised and unsupervised learning",
            "category": "machine_learning",
            "rubric": [
                {
                    "question": "Does the report explain supervised learning?",
                    "category": "completeness",
                    "weight": 1.0,
                },
                {
                    "question": "Does the report explain unsupervised learning?",
                    "category": "completeness",
                    "weight": 1.0,
                },
            ],
        },
        {
            "task_id": "drb2-002",
            "query": "Define backpropagation",
            "category": "optimization",
            "rubric": [
                {
                    "question": "Is the chain rule mentioned?",
                    "category": "information_recall",
                    "weight": 1.0,
                },
            ],
        },
    ]
    path = tmp_path / "drb2_queries.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def drbench_json_file(tmp_path: Path) -> Path:
    """Write a sample DR.BENCH JSON file."""
    data = [
        {
            "id": "drb-001",
            "query": "Explain how photosynthesis works in plants",
            "domain": "biology",
            "reference_facts": [
                "Photosynthesis converts CO2 and water into glucose",
                "Chlorophyll absorbs light energy",
            ],
            "evaluation_axes": [
                "Evaluate the accuracy of biological claims",
                "Assess the organization of the report",
            ],
        },
        {
            "id": "drb-002",
            "query": "What is the structure of DNA?",
            "domain": "biology",
            "reference_facts": [
                "DNA is a double helix",
            ],
            "evaluation_axes": [
                "Check source citations for correctness",
            ],
        },
    ]
    path = tmp_path / "drbench_queries.json"
    path.write_text(json.dumps(data))
    return path


# ════════════════════════════════════════════════════════════════════════════
#  Part 1: research_rubrics_to_rubric_v2 tests
# ════════════════════════════════════════════════════════════════════════════


class TestResearchRubricsConverter:
    """Tests for research_rubrics_to_rubric_v2."""

    def test_basic_conversion(self, sample_research_rubrics_criteria):
        rubric = research_rubrics_to_rubric_v2(
            "rr-test", "Test query", sample_research_rubrics_criteria
        )
        assert isinstance(rubric, RubricV2)
        assert rubric.query_id == "rr-test"
        assert rubric.query_text == "Test query"

    def test_criteria_count_includes_general(self, sample_research_rubrics_criteria):
        rubric = research_rubrics_to_rubric_v2(
            "rr-test", "Test query", sample_research_rubrics_criteria
        )
        general_count = len(build_general_criteria())
        # 4 task-specific + general criteria
        assert rubric.total_criteria == general_count + 4

    def test_dimension_mapping_direct(self, sample_research_rubrics_criteria):
        rubric = research_rubrics_to_rubric_v2(
            "rr-test", "Test query", sample_research_rubrics_criteria
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        dimensions = [c.dimension for c in task_criteria]
        assert "coverage" in dimensions
        assert "factual_accuracy" in dimensions
        assert "citation_quality" in dimensions
        assert "analytical_depth" in dimensions

    def test_weight_preservation(self, sample_research_rubrics_criteria):
        rubric = research_rubrics_to_rubric_v2(
            "rr-test", "Test query", sample_research_rubrics_criteria
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        weights = {c.text: c.weight for c in task_criteria}
        assert weights["Report covers the key technical innovations"] == 2.0
        assert weights["All factual claims are accurate and verifiable"] == 1.5

    def test_source_tag_is_benchmark(self, sample_research_rubrics_criteria):
        rubric = research_rubrics_to_rubric_v2(
            "rr-test", "Test query", sample_research_rubrics_criteria
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) == 4
        for c in task_criteria:
            assert c.source == "benchmark"

    def test_uses_research_rubrics_weights(self, sample_research_rubrics_criteria):
        rubric = research_rubrics_to_rubric_v2(
            "rr-test", "Test query", sample_research_rubrics_criteria
        )
        expected_weights = DIMENSION_WEIGHTS_BY_SOURCE["research_rubrics"]
        assert rubric.dimension_weights == expected_weights

    def test_empty_criteria_list(self):
        rubric = research_rubrics_to_rubric_v2("rr-empty", "Empty query", [])
        # Should still have general criteria
        assert rubric.total_criteria == len(build_general_criteria())

    def test_missing_criterion_text_skipped(self):
        criteria = [
            {"criterion_text": "", "dimension_hint": "coverage", "weight": 1.0},
            {"criterion_text": "Valid criterion", "dimension_hint": "coverage", "weight": 1.0},
        ]
        rubric = research_rubrics_to_rubric_v2("rr-skip", "Test", criteria)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) == 1
        assert task_criteria[0].text == "Valid criterion"

    def test_unknown_hint_uses_infer(self):
        criteria = [
            {
                "criterion_text": "The report analyzes depth of coverage",
                "dimension_hint": "unknown_dimension",
                "weight": 1.0,
            },
        ]
        rubric = research_rubrics_to_rubric_v2("rr-infer", "Test", criteria)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        # _infer_dimension should pick up "analyz" -> "analytical_depth"
        assert task_criteria[0].dimension == "analytical_depth"

    def test_missing_weight_defaults_to_one(self):
        criteria = [
            {"criterion_text": "No weight field", "dimension_hint": "coverage"},
        ]
        rubric = research_rubrics_to_rubric_v2("rr-def", "Test", criteria)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert task_criteria[0].weight == 1.0

    def test_single_criterion(self):
        criteria = [
            {"criterion_text": "Single item", "dimension_hint": "coverage", "weight": 3.0},
        ]
        rubric = research_rubrics_to_rubric_v2("rr-single", "Test", criteria)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) == 1


# ════════════════════════════════════════════════════════════════════════════
#  Part 2: drb2_to_rubric_v2 tests
# ════════════════════════════════════════════════════════════════════════════


class TestDrb2Converter:
    """Tests for drb2_to_rubric_v2."""

    def test_basic_conversion(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test query", sample_drb2_items)
        assert isinstance(rubric, RubricV2)
        assert rubric.query_id == "drb2-test"

    def test_criteria_count(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test query", sample_drb2_items)
        general_count = len(build_general_criteria())
        assert rubric.total_criteria == general_count + 5

    def test_category_mapping_information_recall(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        recall_criteria = [c for c in task_criteria if c.dimension == "information_recall"]
        assert len(recall_criteria) == 1

    def test_category_mapping_factual_accuracy(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        accuracy_criteria = [c for c in task_criteria if c.dimension == "factual_accuracy"]
        assert len(accuracy_criteria) == 1

    def test_category_mapping_completeness_to_coverage(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        coverage_criteria = [c for c in task_criteria if c.dimension == "coverage"]
        assert len(coverage_criteria) == 1

    def test_category_mapping_organization(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        org_criteria = [c for c in task_criteria if c.dimension == "organization"]
        assert len(org_criteria) == 1

    def test_category_mapping_citation(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        cite_criteria = [c for c in task_criteria if c.dimension == "citation_quality"]
        assert len(cite_criteria) == 1

    def test_weight_preservation(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        heavy = [c for c in task_criteria if c.weight == 2.0]
        assert len(heavy) == 1
        assert "cover all requested topics" in heavy[0].text

    def test_source_tag(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert all(c.source == "benchmark" for c in task_criteria)

    def test_uses_drb2_dimension_weights(self, sample_drb2_items):
        rubric = drb2_to_rubric_v2("drb2-test", "Test", sample_drb2_items)
        expected_weights = DIMENSION_WEIGHTS_BY_SOURCE["drb2"]
        assert rubric.dimension_weights == expected_weights

    def test_empty_rubric_items(self):
        rubric = drb2_to_rubric_v2("drb2-empty", "Test", [])
        assert rubric.total_criteria == len(build_general_criteria())

    def test_unknown_category_uses_infer(self):
        items = [
            {
                "question": "Does the report analyze the trade-offs?",
                "category": "unknown_cat",
                "weight": 1.0,
            },
        ]
        rubric = drb2_to_rubric_v2("drb2-unk", "Test", items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        # _infer_dimension should map "analyz" -> "analytical_depth"
        assert task_criteria[0].dimension == "analytical_depth"

    def test_missing_question_skipped(self):
        items = [
            {"question": "", "category": "information_recall", "weight": 1.0},
            {"question": "Valid question?", "category": "information_recall", "weight": 1.0},
        ]
        rubric = drb2_to_rubric_v2("drb2-skip", "Test", items)
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) == 1

    def test_drb2_category_map_completeness(self):
        """Verify all expected categories are in DRB2_CATEGORY_MAP."""
        expected_categories = [
            "information_recall",
            "factual_accuracy",
            "completeness",
            "organization",
            "citation",
        ]
        for cat in expected_categories:
            assert cat in DRB2_CATEGORY_MAP


# ════════════════════════════════════════════════════════════════════════════
#  Part 3: drbench_to_rubric_v2 tests
# ════════════════════════════════════════════════════════════════════════════


class TestDrbenchConverter:
    """Tests for drbench_to_rubric_v2."""

    def test_basic_conversion(self, sample_drbench_facts, sample_drbench_axes):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test query", sample_drbench_facts, sample_drbench_axes
        )
        assert isinstance(rubric, RubricV2)
        assert rubric.query_id == "drb-test"

    def test_criteria_count(self, sample_drbench_facts, sample_drbench_axes):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test query", sample_drbench_facts, sample_drbench_axes
        )
        general_count = len(build_general_criteria())
        # 3 facts + 3 axes
        assert rubric.total_criteria == general_count + 6

    def test_facts_become_factual_accuracy(self, sample_drbench_facts):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test query", sample_drbench_facts, []
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        for c in task_criteria:
            assert c.dimension == "factual_accuracy"
            assert c.text.startswith("The report includes the fact: ")

    def test_fact_text_preserved(self, sample_drbench_facts):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test query", sample_drbench_facts, []
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        texts = [c.text for c in task_criteria]
        assert any("5000 degrees" in t for t in texts)
        assert any("Iron constitutes" in t for t in texts)

    def test_axes_dimension_inference(self, sample_drbench_axes):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test query", [], sample_drbench_axes
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        dims = [c.dimension for c in task_criteria]
        # "accuracy of geological claims" -> factual_accuracy
        assert "factual_accuracy" in dims
        # "organization and structure" -> organization
        assert "organization" in dims
        # "sources are properly cited" -> citation_quality
        assert "citation_quality" in dims

    def test_source_tag(self, sample_drbench_facts, sample_drbench_axes):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test", sample_drbench_facts, sample_drbench_axes
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert all(c.source == "benchmark" for c in task_criteria)

    def test_uses_drbench_dimension_weights(self, sample_drbench_facts, sample_drbench_axes):
        rubric = drbench_to_rubric_v2(
            "drb-test", "Test", sample_drbench_facts, sample_drbench_axes
        )
        expected_weights = DIMENSION_WEIGHTS_BY_SOURCE["drbench"]
        assert rubric.dimension_weights == expected_weights

    def test_empty_facts_and_axes(self):
        rubric = drbench_to_rubric_v2("drb-empty", "Test", [], [])
        assert rubric.total_criteria == len(build_general_criteria())

    def test_empty_strings_skipped(self):
        rubric = drbench_to_rubric_v2(
            "drb-skip", "Test", ["", "Valid fact"], ["", "Check accuracy"]
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) == 2

    def test_single_fact(self):
        rubric = drbench_to_rubric_v2(
            "drb-single", "Test", ["Water boils at 100C at sea level"], []
        )
        task_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) == 1
        assert task_criteria[0].dimension == "factual_accuracy"


# ════════════════════════════════════════════════════════════════════════════
#  Part 4: Loader tests
# ════════════════════════════════════════════════════════════════════════════


class TestResearchRubricsLoader:
    """Tests for QueryRegistry.load_research_rubrics_queries."""

    def test_load_from_file(self, research_rubrics_json_file):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        assert len(loaded) == 2
        assert all(q.source == "research_rubrics" for q in loaded)

    def test_query_ids(self, research_rubrics_json_file):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        ids = {q.id for q in loaded}
        assert ids == {"rr-001", "rr-002"}

    def test_criteria_in_metadata(self, research_rubrics_json_file):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        for q in loaded:
            assert "criteria" in q.metadata

    def test_rubric_is_v2(self, research_rubrics_json_file):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        for q in loaded:
            assert isinstance(q.rubric, RubricV2)

    def test_missing_file_returns_empty(self, tmp_path):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=tmp_path / "nonexistent.json"
        )
        assert loaded == []

    def test_bad_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(data_path=bad_file)
        assert loaded == []

    def test_adds_to_registry(self, research_rubrics_json_file):
        registry = QueryRegistry()
        registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        assert len(registry.queries) == 2

    def test_difficulty_assigned(self, research_rubrics_json_file):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        for q in loaded:
            assert q.difficulty in ("simple", "moderate", "complex")


class TestDrb2Loader:
    """Tests for QueryRegistry.load_drb2_queries."""

    def test_load_from_file(self, drb2_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(data_path=drb2_json_file)
        assert len(loaded) == 2
        assert all(q.source == "drb2" for q in loaded)

    def test_task_ids(self, drb2_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(data_path=drb2_json_file)
        ids = {q.id for q in loaded}
        assert ids == {"drb2-001", "drb2-002"}

    def test_category_in_metadata(self, drb2_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(data_path=drb2_json_file)
        for q in loaded:
            assert "category" in q.metadata

    def test_missing_file_returns_empty(self, tmp_path):
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(
            data_path=tmp_path / "nonexistent.json"
        )
        assert loaded == []

    def test_bad_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{broken")
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(data_path=bad_file)
        assert loaded == []

    def test_rubric_has_correct_weights(self, drb2_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(data_path=drb2_json_file)
        expected = DIMENSION_WEIGHTS_BY_SOURCE["drb2"]
        for q in loaded:
            assert q.rubric.dimension_weights == expected


class TestDrbenchLoader:
    """Tests for QueryRegistry.load_drbench_queries."""

    def test_load_from_file(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        assert len(loaded) == 2
        assert all(q.source == "drbench" for q in loaded)

    def test_query_ids(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        ids = {q.id for q in loaded}
        assert ids == {"drb-001", "drb-002"}

    def test_metadata_has_reference_facts(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        for q in loaded:
            assert "reference_facts" in q.metadata
            assert "evaluation_axes" in q.metadata

    def test_missing_file_returns_empty(self, tmp_path):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(
            data_path=tmp_path / "nonexistent.json"
        )
        assert loaded == []

    def test_bad_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{{invalid}}")
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=bad_file)
        assert loaded == []

    def test_rubric_has_correct_weights(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        expected = DIMENSION_WEIGHTS_BY_SOURCE["drbench"]
        for q in loaded:
            assert q.rubric.dimension_weights == expected

    def test_domain_assigned(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        for q in loaded:
            assert q.domain == "biology"


# ════════════════════════════════════════════════════════════════════════════
#  Part 5: Integration tests (loader -> converter pipeline)
# ════════════════════════════════════════════════════════════════════════════


class TestIntegrationPipeline:
    """End-to-end tests: JSON file -> loader -> converter -> RubricV2."""

    def test_research_rubrics_pipeline_criteria_present(
        self, research_rubrics_json_file
    ):
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        q = loaded[0]
        task_criteria = [c for c in q.rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) > 0

    def test_drb2_pipeline_criteria_present(self, drb2_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drb2_queries(data_path=drb2_json_file)
        q = loaded[0]
        task_criteria = [c for c in q.rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) > 0

    def test_drbench_pipeline_criteria_present(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        q = loaded[0]
        task_criteria = [c for c in q.rubric.criteria if c.source == "benchmark"]
        assert len(task_criteria) > 0

    def test_serialization_roundtrip(self, research_rubrics_json_file):
        """EvalQuery serializes and deserializes correctly."""
        registry = QueryRegistry()
        loaded = registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        q = loaded[0]
        data = q.to_dict()
        restored = EvalQuery.from_dict(data)
        assert restored.id == q.id
        assert restored.query == q.query
        assert restored.source == q.source
        assert restored.rubric.total_criteria == q.rubric.total_criteria

    def test_drbench_facts_in_rubric(self, drbench_json_file):
        registry = QueryRegistry()
        loaded = registry.load_drbench_queries(data_path=drbench_json_file)
        q = [x for x in loaded if x.id == "drb-001"][0]
        task_criteria = [c for c in q.rubric.criteria if c.source == "benchmark"]
        fact_criteria = [c for c in task_criteria if c.text.startswith("The report includes the fact:")]
        assert len(fact_criteria) == 2

    def test_multiple_loaders_accumulate(
        self,
        research_rubrics_json_file,
        drb2_json_file,
        drbench_json_file,
    ):
        """Loading from multiple sources accumulates in the registry."""
        registry = QueryRegistry()
        registry.load_research_rubrics_queries(
            data_path=research_rubrics_json_file
        )
        registry.load_drb2_queries(data_path=drb2_json_file)
        registry.load_drbench_queries(data_path=drbench_json_file)
        assert len(registry.queries) == 6
        sources = {q.source for q in registry.queries}
        assert sources == {"research_rubrics", "drb2", "drbench"}
