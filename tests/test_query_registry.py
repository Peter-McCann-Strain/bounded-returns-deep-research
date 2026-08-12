"""Tests for query_registry and rubric_converters modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from deep_research.evaluation.rubric_v2 import Criterion, RubricV2, DIMENSION_WEIGHTS_V2
from deep_research.evaluation.rubric_converters import (
    draco_to_rubric_v2,
    research_qa_to_rubric_v2,
    deepsearch_qa_to_rubric_v2,
    litqa2_to_rubric_v2,
    DRACO_DIMENSION_MAP,
)
# Alias to avoid pytest collecting it as a test function
from deep_research.evaluation.rubric_converters import test_query_to_rubric_v2 as _convert_test_query
from deep_research.evaluation.query_registry import (
    EvalQuery,
    QueryRegistry,
    classify_difficulty,
    _stratified_sample,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_draco_rubric() -> dict:
    """DRACO rubric in cached format: {section_title: [criteria]}."""
    return {
        "Factual Accuracy": [
            {"id": "crit-1", "weight": 10, "description": "States that X is true"},
            {"id": "crit-2", "weight": 5, "description": "Mentions technique Y"},
            {"id": "crit-neg", "weight": -50, "description": "Incorrectly claims Z"},
        ],
        "Breadth and Depth of Analysis": [
            {"id": "crit-3", "weight": 8, "description": "Covers multiple perspectives"},
        ],
        "Citation Quality": [
            {"id": "crit-4", "weight": 3, "description": "Cites at least 3 sources"},
        ],
    }


@pytest.fixture
def sample_research_qa_items() -> list[dict]:
    """ResearchQA rubric items."""
    return [
        {"question": "Does the response discuss pyrolysis temperature?", "type": ["Other"], "citation_metadata": None},
        {"question": "Does the response cite relevant studies?", "type": ["Citation"], "citation_metadata": None},
        {"question": "Does the response compare method A vs B?", "type": ["Comparison"], "citation_metadata": None},
        {"question": "Does the response mention the impact on yield?", "type": ["Impact"], "citation_metadata": None},
    ]


@pytest.fixture
def sample_test_query():
    """A TestQuery-like object matching the interface of test_queries.TestQuery."""
    from deep_research.evaluation.test_queries import TEST_QUERIES
    return TEST_QUERIES[0]


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with minimal benchmark caches."""
    benchmarks = tmp_path / "benchmarks"

    # DRACO cache
    draco_dir = benchmarks / "draco"
    draco_dir.mkdir(parents=True)
    draco_queries = []
    for i in range(20):
        domain = ["Technology", "Medicine", "Finance", "Academic", "Law"][i % 5]
        draco_queries.append({
            "id": f"draco_{i:04d}",
            "query": f"Research question about {domain.lower()} topic {i}?",
            "domain": domain,
            "difficulty": "research",
            "rubric": {
                "Factual Accuracy": [
                    {"id": f"c{i}_1", "weight": 10, "description": f"States fact about topic {i}"},
                    {"id": f"c{i}_2", "weight": -20, "description": f"Avoids error about topic {i}"},
                ],
                "Breadth and Depth of Analysis": [
                    {"id": f"c{i}_3", "weight": 5, "description": f"Covers breadth of topic {i}"},
                ],
            },
            "reference_answer": "",
            "expected_citations": [],
            "metadata": {"source": "draco", "total_criteria": 3},
        })
    (draco_dir / "draco_queries.json").write_text(json.dumps(draco_queries))

    # DeepSearchQA cache
    dsqa_dir = benchmarks / "deepsearch_qa"
    dsqa_dir.mkdir(parents=True)
    dsqa_queries = []
    for i in range(10):
        domain = ["Science", "History", "Sports", "Arts", "Health"][i % 5]
        dsqa_queries.append({
            "id": f"dsqa_{i:04d}",
            "query": f"What is the answer to {domain.lower()} question {i}?",
            "domain": domain,
            "difficulty": "multi-step",
            "rubric": {"answer_type": "Single Answer", "expected_answer": f"Answer {i}"},
            "reference_answer": f"Answer {i}",
            "expected_citations": [],
            "metadata": {"problem_category": domain, "answer_type": "Single Answer"},
        })
    (dsqa_dir / "deepsearch_qa_queries.json").write_text(json.dumps(dsqa_queries))

    # ResearchQA cache
    rqa_dir = benchmarks / "research_qa"
    rqa_dir.mkdir(parents=True)
    rqa_queries = []
    for i in range(10):
        field = ["Geology", "Physics", "Chemistry", "Biology", "Economics"][i % 5]
        rqa_queries.append({
            "id": f"rqa_{i:04d}",
            "query": f"Research question in {field.lower()} number {i}?",
            "domain": field,
            "difficulty": "research",
            "rubric": {
                "criteria": [
                    {"question": f"Does the response address point {j}?", "type": ["Other"], "citation_metadata": None}
                    for j in range(3)
                ],
                "types": ["Other"],
            },
            "reference_answer": "",
            "expected_citations": [],
            "metadata": {"field": field, "general_domain": "Sciences", "subdomain": field, "date": "2025-01-01", "total_rubric_items": 3},
        })
    (rqa_dir / "research_qa_queries.json").write_text(json.dumps(rqa_queries))

    # LitQA2 cache
    litqa_dir = benchmarks / "litqa2"
    litqa_dir.mkdir(parents=True)
    litqa_queries = []
    for i in range(8):
        litqa_queries.append({
            "id": f"litqa2_{i:04d}",
            "query": f"What compound is most effective for condition {i}?",
            "domain": "scientific_literature",
            "difficulty": "expert",
            "rubric": {
                "ideal": f"compound_{i}",
                "distractors": [f"distractor_{i}_a", f"distractor_{i}_b"],
                "options": [f"distractor_{i}_a", f"distractor_{i}_b", f"compound_{i}", "Insufficient information"],
            },
            "reference_answer": f"compound_{i}",
            "expected_citations": [f"https://doi.org/10.1234/test_{i}"],
            "metadata": {"key_passage": "", "is_opensource": True, "tag": "test", "sources": [], "n_options": 4},
        })
    (litqa_dir / "litqa2_queries.json").write_text(json.dumps(litqa_queries))

    return tmp_path


# ── 1. QueryRegistry construction ───────────────────────────────────────────

class TestQueryRegistryConstruction:
    def test_default_construction(self):
        registry = QueryRegistry()
        assert registry._queries == []
        assert registry._data_dir == Path("data")

    def test_custom_data_dir(self, tmp_path: Path):
        registry = QueryRegistry(data_dir=tmp_path)
        assert registry._data_dir == tmp_path

    def test_queries_property_empty(self):
        registry = QueryRegistry()
        assert registry.queries == []


# ── 2. load_custom_queries ───────────────────────────────────────────────────

class TestLoadCustomQueries:
    def test_returns_5_queries(self):
        registry = QueryRegistry()
        loaded = registry.load_custom_queries()
        assert len(loaded) == 5

    def test_queries_have_v2_rubrics(self):
        registry = QueryRegistry()
        loaded = registry.load_custom_queries()
        for q in loaded:
            assert isinstance(q.rubric, RubricV2)
            assert q.rubric.total_criteria > 0
            assert q.rubric.dimension_weights == DIMENSION_WEIGHTS_V2

    def test_queries_have_correct_source(self):
        registry = QueryRegistry()
        loaded = registry.load_custom_queries()
        for q in loaded:
            assert q.source == "custom"

    def test_expected_elements_preserved(self):
        registry = QueryRegistry()
        loaded = registry.load_custom_queries()
        # q1_bert_vs_gpt has 12 expected elements
        q1 = [q for q in loaded if q.id == "q1_bert_vs_gpt"][0]
        assert len(q1.expected_elements) == 12


# ── 3. EvalQuery has all required fields ─────────────────────────────────────

class TestEvalQueryFields:
    def test_all_fields_present(self):
        rubric = RubricV2(
            query_id="test",
            query_text="test query",
            criteria=[Criterion("test criterion", "coverage")],
            dimension_weights=DIMENSION_WEIGHTS_V2.copy(),
        )
        eq = EvalQuery(
            id="test",
            query="test query",
            source="custom",
            domain="test_domain",
            difficulty="simple",
            rubric=rubric,
        )
        assert eq.id == "test"
        assert eq.query == "test query"
        assert eq.source == "custom"
        assert eq.domain == "test_domain"
        assert eq.difficulty == "simple"
        assert isinstance(eq.rubric, RubricV2)
        assert eq.expected_elements == []
        assert eq.reference_answer == ""
        assert eq.metadata == {}

    def test_to_dict_roundtrip(self):
        rubric = RubricV2(
            query_id="rt",
            query_text="roundtrip test",
            criteria=[
                Criterion("crit 1", "coverage", weight=2.0, source="draco"),
                Criterion("crit 2", "factual_accuracy"),
            ],
            dimension_weights=DIMENSION_WEIGHTS_V2.copy(),
        )
        original = EvalQuery(
            id="rt",
            query="roundtrip test",
            source="draco",
            domain="Technology",
            difficulty="complex",
            rubric=rubric,
            expected_elements=["elem1", "elem2"],
            reference_answer="the answer",
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = EvalQuery.from_dict(data)
        assert restored.id == original.id
        assert restored.query == original.query
        assert restored.source == original.source
        assert restored.domain == original.domain
        assert restored.difficulty == original.difficulty
        assert restored.reference_answer == original.reference_answer
        assert restored.expected_elements == original.expected_elements
        assert restored.metadata == original.metadata
        assert restored.rubric.total_criteria == original.rubric.total_criteria
        assert restored.rubric.query_id == original.rubric.query_id


# ── 4. get_by_source filtering ──────────────────────────────────────────────

class TestGetBySource:
    def test_filter_custom(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        registry.load_custom_queries()
        registry.load_draco_queries(max_queries=5)
        custom = registry.get_by_source("custom")
        draco = registry.get_by_source("draco")
        assert len(custom) == 5
        assert len(draco) == 5
        assert all(q.source == "custom" for q in custom)
        assert all(q.source == "draco" for q in draco)

    def test_empty_source_returns_empty(self):
        registry = QueryRegistry()
        registry.load_custom_queries()
        assert registry.get_by_source("nonexistent") == []


# ── 5. get_by_domain filtering ──────────────────────────────────────────────

class TestGetByDomain:
    def test_filter_by_domain(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        registry.load_draco_queries(max_queries=20)
        tech = registry.get_by_domain("Technology")
        assert len(tech) > 0
        assert all(q.domain == "Technology" for q in tech)

    def test_nonexistent_domain(self):
        registry = QueryRegistry()
        registry.load_custom_queries()
        assert registry.get_by_domain("Underwater Basket Weaving") == []


# ── 6. get_by_difficulty filtering ───────────────────────────────────────────

class TestGetByDifficulty:
    def test_custom_queries_have_difficulty(self):
        registry = QueryRegistry()
        registry.load_custom_queries()
        simple = registry.get_by_difficulty("simple")
        complex_ = registry.get_by_difficulty("complex")
        # q1 is simple, q3/q4/q5 are complex
        assert len(simple) >= 1
        assert len(complex_) >= 1

    def test_difficulty_is_valid(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        registry.load_all(custom=5, draco=10, deepsearch=5, research_qa=5, litqa2=5)
        valid = {"simple", "moderate", "complex"}
        for q in registry.queries:
            assert q.difficulty in valid, f"Invalid difficulty: {q.difficulty}"


# ── 7. save_manifest and from_manifest roundtrip ────────────────────────────

class TestManifestRoundtrip:
    def test_roundtrip(self, tmp_data_dir: Path, tmp_path: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        registry.load_all(custom=5, draco=10, deepsearch=5, research_qa=5, litqa2=5)
        original_count = len(registry.queries)
        assert original_count > 0

        manifest_path = tmp_path / "manifest.json"
        registry.save_manifest(manifest_path)
        assert manifest_path.exists()

        # Verify JSON structure
        data = json.loads(manifest_path.read_text())
        assert data["version"] == "2.0"
        assert data["total_queries"] == original_count
        assert len(data["queries"]) == original_count

        # Load back
        restored = QueryRegistry.from_manifest(manifest_path)
        assert len(restored.queries) == original_count

        # Verify individual query fidelity
        for orig, rest in zip(registry.queries, restored.queries):
            assert orig.id == rest.id
            assert orig.query == rest.query
            assert orig.source == rest.source
            assert orig.rubric.total_criteria == rest.rubric.total_criteria

    def test_manifest_includes_summary(self, tmp_data_dir: Path, tmp_path: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        registry.load_custom_queries()
        manifest_path = tmp_path / "summary_test.json"
        registry.save_manifest(manifest_path)
        data = json.loads(manifest_path.read_text())
        assert "summary" in data
        assert data["summary"]["total"] == 5


# ── 8. summary property ─────────────────────────────────────────────────────

class TestSummary:
    def test_summary_counts(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        registry.load_all(custom=5, draco=10, deepsearch=5, research_qa=5, litqa2=5)
        s = registry.summary
        assert s["total"] == len(registry.queries)
        assert sum(s["by_source"].values()) == s["total"]
        assert sum(s["by_difficulty"].values()) == s["total"]
        assert "custom" in s["by_source"]

    def test_summary_empty_registry(self):
        registry = QueryRegistry()
        s = registry.summary
        assert s["total"] == 0
        assert s["by_source"] == {}


# ── 9. draco_to_rubric_v2 preserves weights ─────────────────────────────────

class TestDracoConverter:
    def test_preserves_positive_weights(self, sample_draco_rubric: dict):
        rubric = draco_to_rubric_v2("draco_test", "Test query?", sample_draco_rubric)
        draco_criteria = [c for c in rubric.criteria if c.source == "draco"]
        weights = {c.text: c.weight for c in draco_criteria}
        # Should find the weight=10 and weight=5 criteria
        pos_weights = [w for w in weights.values() if w > 0]
        assert 10.0 in pos_weights
        assert 5.0 in pos_weights

    def test_preserves_negative_weights(self, sample_draco_rubric: dict):
        rubric = draco_to_rubric_v2("draco_test", "Test query?", sample_draco_rubric)
        draco_criteria = [c for c in rubric.criteria if c.source == "draco"]
        neg = [c for c in draco_criteria if c.weight < 0]
        assert len(neg) == 1
        assert neg[0].weight == -50.0
        assert neg[0].dimension == "factual_accuracy"

    def test_maps_sections_to_dimensions(self, sample_draco_rubric: dict):
        rubric = draco_to_rubric_v2("draco_test", "Test query?", sample_draco_rubric)
        draco_criteria = [c for c in rubric.criteria if c.source == "draco"]
        dimensions = {c.dimension for c in draco_criteria}
        assert "factual_accuracy" in dimensions
        assert "coverage" in dimensions  # Breadth and Depth -> coverage
        assert "citation_quality" in dimensions

    def test_includes_general_criteria(self, sample_draco_rubric: dict):
        rubric = draco_to_rubric_v2("draco_test", "Test query?", sample_draco_rubric)
        general = [c for c in rubric.criteria if c.source == "general"]
        assert len(general) > 0

    def test_empty_rubric(self):
        rubric = draco_to_rubric_v2("empty", "Empty query?", {})
        # Should still have general criteria
        assert rubric.total_criteria > 0
        general = [c for c in rubric.criteria if c.source == "general"]
        assert len(general) == rubric.total_criteria


# ── 10. test_query_to_rubric_v2 backward compatibility ──────────────────────

class TestTestQueryConverter:
    def test_backward_compat(self, sample_test_query):
        rubric = _convert_test_query(sample_test_query)
        assert isinstance(rubric, RubricV2)
        assert rubric.query_id == sample_test_query.id
        assert rubric.query_text == sample_test_query.query

        # Expected elements should appear as coverage criteria
        coverage = rubric.get_criteria_by_dimension("coverage")
        assert len(coverage) > 0
        coverage_texts = " ".join(c.text for c in coverage)
        for elem in sample_test_query.expected_elements[:3]:
            assert elem in coverage_texts

    def test_has_all_dimensions(self, sample_test_query):
        rubric = _convert_test_query(sample_test_query)
        dims = set(rubric.get_dimensions())
        assert "factual_accuracy" in dims
        assert "coverage" in dims
        assert "analytical_depth" in dims


# ── 11. load_all with zero counts for unavailable benchmarks ────────────────

class TestLoadAllEdgeCases:
    def test_zero_counts_all(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        queries = registry.load_all(custom=0, draco=0, deepsearch=0, research_qa=0, litqa2=0)
        assert queries == []

    def test_missing_cache_skipped(self, tmp_path: Path):
        """Registry with empty data dir should skip all benchmarks."""
        empty_dir = tmp_path / "empty_data"
        empty_dir.mkdir()
        registry = QueryRegistry(data_dir=empty_dir)
        # Custom queries don't need cache files
        queries = registry.load_all(custom=5, draco=10, deepsearch=5, research_qa=5, litqa2=5)
        assert len(queries) == 5  # Only custom queries loaded
        assert all(q.source == "custom" for q in queries)

    def test_only_custom(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        queries = registry.load_all(custom=5, draco=0, deepsearch=0, research_qa=0, litqa2=0)
        assert len(queries) == 5
        assert all(q.source == "custom" for q in queries)

    def test_repeated_load_all_resets(self, tmp_data_dir: Path):
        registry = QueryRegistry(data_dir=tmp_data_dir)
        q1 = registry.load_all(custom=5, draco=5, deepsearch=0, research_qa=0, litqa2=0)
        q2 = registry.load_all(custom=5, draco=5, deepsearch=0, research_qa=0, litqa2=0)
        assert len(q1) == len(q2)


# ── 12. Rubric converter edge cases ─────────────────────────────────────────

class TestConverterEdgeCases:
    def test_research_qa_empty_items(self):
        rubric = research_qa_to_rubric_v2("rqa_empty", "Empty?", [])
        # Should still have general criteria
        assert rubric.total_criteria > 0
        benchmark_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(benchmark_criteria) == 0

    def test_deepsearch_empty_answer(self):
        rubric = deepsearch_qa_to_rubric_v2("dsqa_empty", "Empty?", "", [])
        # No answer-specific criteria, but still has general + evidence criteria
        assert rubric.total_criteria > 0

    def test_deepsearch_with_extra_criteria(self):
        rubric = deepsearch_qa_to_rubric_v2(
            "dsqa_extra", "Extra?", "Answer X",
            ["The report provides a comprehensive list"]
        )
        benchmark_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        assert len(benchmark_criteria) > 3  # answer + evidence + extra

    def test_litqa2_no_distractors(self):
        rubric = litqa2_to_rubric_v2("litqa_nd", "Question?", "correct", [])
        benchmark_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        # Without distractors, no "distinguishes from alternatives" criterion
        assert all("alternatives" not in c.text for c in benchmark_criteria)

    def test_litqa2_with_distractors(self):
        rubric = litqa2_to_rubric_v2(
            "litqa_wd", "Question?", "correct", ["wrong1", "wrong2"]
        )
        benchmark_criteria = [c for c in rubric.criteria if c.source == "benchmark"]
        has_distractor_crit = any("alternatives" in c.text for c in benchmark_criteria)
        assert has_distractor_crit

    def test_draco_list_format(self):
        """DRACO rubric can also come as a list of section dicts."""
        sections = [
            {
                "name": "Factual Accuracy",
                "criteria": [
                    {"weight": 5, "description": "States fact A"},
                ],
            },
        ]
        rubric = draco_to_rubric_v2("list_fmt", "Query?", sections)
        draco_crit = [c for c in rubric.criteria if c.source == "draco"]
        assert len(draco_crit) == 1
        assert draco_crit[0].weight == 5.0


# ── 13. Difficulty classification ────────────────────────────────────────────

class TestDifficultyClassification:
    def test_simple_question(self):
        assert classify_difficulty("What is photosynthesis?") == "simple"

    def test_complex_multipart(self):
        assert classify_difficulty(
            "Compare and contrast the effects of X? Also discuss the implications of Y?"
        ) == "complex"

    def test_moderate_default(self):
        result = classify_difficulty(
            "Explain the mechanism of action of metformin in type 2 diabetes management"
        )
        assert result in ("simple", "moderate", "complex")

    def test_long_query_is_complex(self):
        q = " ".join(["word"] * 50) + "?"
        assert classify_difficulty(q) == "complex"


# ── 14. Stratified sampling ─────────────────────────────────────────────────

class TestStratifiedSample:
    def test_basic_stratification(self):
        items = [{"group": "A", "id": i} for i in range(10)]
        items += [{"group": "B", "id": i + 10} for i in range(10)]
        result = _stratified_sample(items, 6, key_fn=lambda x: x["group"])
        groups = [x["group"] for x in result]
        assert groups.count("A") == 3
        assert groups.count("B") == 3

    def test_max_larger_than_items(self):
        items = [{"id": i} for i in range(5)]
        result = _stratified_sample(items, 100, key_fn=lambda x: "all")
        assert len(result) == 5

    def test_zero_max(self):
        items = [{"id": i} for i in range(5)]
        result = _stratified_sample(items, 0, key_fn=lambda x: "all")
        assert result == []

    def test_empty_items(self):
        result = _stratified_sample([], 10, key_fn=lambda x: "all")
        assert result == []

    def test_uneven_groups(self):
        """Larger groups donate surplus to fill quota."""
        items = [{"group": "big", "id": i} for i in range(100)]
        items += [{"group": "small", "id": i + 100} for i in range(2)]
        result = _stratified_sample(items, 10, key_fn=lambda x: x["group"])
        assert len(result) == 10
        groups = [x["group"] for x in result]
        # Small group has only 2 items, so at most 2 from small
        assert groups.count("small") <= 5


# ── 15. DRACO dimension map coverage ────────────────────────────────────────

class TestDracoDimensionMap:
    def test_all_map_to_valid_dimensions(self):
        valid = set(DIMENSION_WEIGHTS_V2.keys())
        for key, dim in DRACO_DIMENSION_MAP.items():
            assert dim in valid, f"DRACO_DIMENSION_MAP['{key}'] = '{dim}' not in V2 dimensions"

    def test_actual_draco_sections_mapped(self):
        """The actual section titles from cached DRACO data should be mapped."""
        actual_titles = [
            "Factual Accuracy",
            "Breadth and Depth of Analysis",
            "Presentation Quality",
            "Citation Quality",
        ]
        for title in actual_titles:
            key = title.lower().strip()
            assert key in DRACO_DIMENSION_MAP, f"Missing mapping for '{title}'"


# ── 16. Full integration with real caches (if available) ────────────────────

class TestRealCacheIntegration:
    """Tests that use real benchmark caches. Skip if not available."""

    REAL_DATA_DIR = Path("data")

    @pytest.fixture(autouse=True)
    def _check_real_data(self):
        if not (self.REAL_DATA_DIR / "benchmarks" / "draco" / "draco_queries.json").exists():
            pytest.skip("Real benchmark caches not available")

    def test_load_draco_real(self):
        registry = QueryRegistry(data_dir=self.REAL_DATA_DIR)
        loaded = registry.load_draco_queries(max_queries=10)
        assert len(loaded) == 10
        for q in loaded:
            assert q.source == "draco"
            assert isinstance(q.rubric, RubricV2)
            assert q.rubric.total_criteria > 0

    def test_load_all_real(self):
        registry = QueryRegistry(data_dir=self.REAL_DATA_DIR)
        queries = registry.load_all(custom=5, draco=10, deepsearch=5, research_qa=5, litqa2=5)
        assert len(queries) >= 20  # At least custom + draco should work
        sources = {q.source for q in queries}
        assert "custom" in sources
        assert "draco" in sources
