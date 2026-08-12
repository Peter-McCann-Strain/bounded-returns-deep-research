"""Comprehensive tests for the ablation study framework.

Tests cover:
- AblationConfig construction and display_name
- ABLATION_REGISTRY coverage
- AblationResult construction
- AblationRunner checkpoint save/load roundtrip
- AblationRunner.is_completed logic
- compare_ablations with known score arrays
- compare_ablations with identical scores
- compare_ablations with clearly different scores
- generate_ablation_report markdown validation
- AblationComparison relative_change calculation
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from deep_research.ablation.framework import (
    ABLATION_REGISTRY,
    AblationComparison,
    AblationConfig,
    AblationResult,
    AblationRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Provide a temporary directory for checkpoints."""
    return tmp_path / "ablation_checkpoints"


@pytest.fixture
def runner(tmp_checkpoint_dir):
    """Provide an AblationRunner with a temp checkpoint dir."""
    return AblationRunner(checkpoint_dir=tmp_checkpoint_dir, budget_per_run=1.0)


@pytest.fixture
def sample_config():
    """A sample AblationConfig for testing."""
    return AblationConfig(
        id="test_ablation",
        base_pattern="p4_perspective_storm",
        description="Test ablation",
        component_removed="test_component",
        modification={"skip_test": True},
        expected_effect="No change expected",
    )


@pytest.fixture
def sample_result():
    """A sample successful AblationResult."""
    return AblationResult(
        ablation_id="test_ablation",
        query_id="q1",
        base_pattern="p4_perspective_storm",
        component_removed="test_component",
        status="success",
        report_text="# Test Report\n\nSome content.",
        elapsed_seconds=12.5,
        total_tokens=1500,
    )


# ---------------------------------------------------------------------------
# 1. AblationConfig construction and display_name
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_construction(self, sample_config):
        assert sample_config.id == "test_ablation"
        assert sample_config.base_pattern == "p4_perspective_storm"
        assert sample_config.component_removed == "test_component"
        assert sample_config.modification == {"skip_test": True}

    def test_display_name(self, sample_config):
        assert sample_config.display_name == "p4_perspective_storm - test_component"

    def test_display_name_format(self):
        config = AblationConfig(
            id="p3_no_quality_eval",
            base_pattern="p3_meridian",
            description="Skip quality eval",
            component_removed="quality_evaluator",
        )
        assert config.display_name == "p3_meridian - quality_evaluator"

    def test_default_fields(self):
        config = AblationConfig(
            id="x", base_pattern="p0", description="d", component_removed="c"
        )
        assert config.modification == {}
        assert config.expected_effect == ""


# ---------------------------------------------------------------------------
# 2. ABLATION_REGISTRY coverage
# ---------------------------------------------------------------------------

class TestAblationRegistry:
    def test_registry_not_empty(self):
        assert len(ABLATION_REGISTRY) > 0

    def test_has_p1_entry(self):
        p1_configs = [c for c in ABLATION_REGISTRY if c.base_pattern == "p1_iterative_rag"]
        assert len(p1_configs) >= 1

    def test_has_p2_entry(self):
        p2_configs = [c for c in ABLATION_REGISTRY if c.base_pattern == "p2_supervisor_parallel"]
        assert len(p2_configs) >= 1

    def test_has_p3_entry(self):
        p3_configs = [c for c in ABLATION_REGISTRY if c.base_pattern == "p3_meridian"]
        assert len(p3_configs) >= 1

    def test_has_p4_entry(self):
        p4_configs = [c for c in ABLATION_REGISTRY if c.base_pattern == "p4_perspective_storm"]
        assert len(p4_configs) >= 1

    def test_has_p5_entry(self):
        p5_configs = [c for c in ABLATION_REGISTRY if c.base_pattern == "p5_hierarchical_wd"]
        assert len(p5_configs) >= 1

    def test_unique_ids(self):
        ids = [c.id for c in ABLATION_REGISTRY]
        assert len(ids) == len(set(ids)), "Ablation IDs must be unique"

    def test_all_configs_have_required_fields(self):
        for config in ABLATION_REGISTRY:
            assert config.id, f"Config missing id"
            assert config.base_pattern, f"Config {config.id} missing base_pattern"
            assert config.description, f"Config {config.id} missing description"
            assert config.component_removed, f"Config {config.id} missing component_removed"


# ---------------------------------------------------------------------------
# 3. AblationResult construction
# ---------------------------------------------------------------------------

class TestAblationResult:
    def test_success_result(self, sample_result):
        assert sample_result.status == "success"
        assert sample_result.ablation_id == "test_ablation"
        assert sample_result.elapsed_seconds == 12.5
        assert sample_result.total_tokens == 1500
        assert sample_result.error_message == ""

    def test_error_result(self):
        result = AblationResult(
            ablation_id="test_err",
            query_id="q2",
            base_pattern="p1_iterative_rag",
            component_removed="reflection_loop",
            status="error",
            error_message="Budget exceeded",
        )
        assert result.status == "error"
        assert result.error_message == "Budget exceeded"
        assert result.report_text == ""

    def test_serializable(self, sample_result):
        d = asdict(sample_result)
        assert isinstance(d, dict)
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert "test_ablation" in json_str


# ---------------------------------------------------------------------------
# 4. AblationRunner checkpoint save/load roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_checkpoint_roundtrip(self, runner, sample_result):
        ablation_id = sample_result.ablation_id
        query_id = sample_result.query_id

        # Save
        cp_path = runner._checkpoint_path(ablation_id, query_id)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        cp_path.write_text(json.dumps(asdict(sample_result), indent=2))

        # Load
        data = json.loads(cp_path.read_text())
        assert data["ablation_id"] == ablation_id
        assert data["query_id"] == query_id
        assert data["status"] == "success"
        assert data["total_tokens"] == 1500

    def test_checkpoint_path_structure(self, runner):
        path = runner._checkpoint_path("p4_no_conv", "q_test_1")
        assert "p4_no_conv" in str(path)
        assert "q_test_1.json" in str(path)


# ---------------------------------------------------------------------------
# 5. AblationRunner.is_completed logic
# ---------------------------------------------------------------------------

class TestIsCompleted:
    def test_not_completed_when_missing(self, runner):
        assert not runner.is_completed("nonexistent", "q1")

    def test_completed_when_success(self, runner, sample_result):
        cp = runner._checkpoint_path(sample_result.ablation_id, sample_result.query_id)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(asdict(sample_result)))
        assert runner.is_completed(sample_result.ablation_id, sample_result.query_id)

    def test_not_completed_when_error(self, runner):
        error_result = AblationResult(
            ablation_id="err_test",
            query_id="q1",
            base_pattern="p1_iterative_rag",
            component_removed="reflection_loop",
            status="error",
            error_message="Failed",
        )
        cp = runner._checkpoint_path("err_test", "q1")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(asdict(error_result)))
        assert not runner.is_completed("err_test", "q1")

    def test_not_completed_when_corrupted(self, runner):
        cp = runner._checkpoint_path("corrupt", "q1")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("not valid json{{{")
        assert not runner.is_completed("corrupt", "q1")


# ---------------------------------------------------------------------------
# 6. compare_ablations with known score arrays
# ---------------------------------------------------------------------------

class TestCompareAblations:
    def test_known_scores(self, runner):
        """Test with known scores where base > ablated."""
        config = AblationConfig(
            id="p4_no_conversations",
            base_pattern="p4_perspective_storm",
            description="Test",
            component_removed="conversation_sim",
        )
        base_scores = {"p4_perspective_storm": [0.8, 0.7, 0.9, 0.6, 0.75, 0.85, 0.65]}
        ablation_scores = {"p4_no_conversations": [0.5, 0.4, 0.6, 0.3, 0.45, 0.55, 0.35]}

        comparisons = runner.compare_ablations(
            base_scores, ablation_scores, configs=[config]
        )

        assert len(comparisons) == 1
        c = comparisons[0]
        assert c.ablation_id == "p4_no_conversations"
        assert c.base_mean > c.ablated_mean
        assert c.score_delta > 0  # base > ablated means component helps
        assert c.relative_change > 0

    def test_missing_scores_skipped(self, runner):
        """Configs with no matching scores should be skipped."""
        base_scores = {"p4_perspective_storm": [0.5, 0.6]}
        ablation_scores = {}  # no ablation scores

        comparisons = runner.compare_ablations(
            base_scores, ablation_scores, configs=ABLATION_REGISTRY
        )
        assert len(comparisons) == 0


# ---------------------------------------------------------------------------
# 7. compare_ablations with identical scores (no significant difference)
# ---------------------------------------------------------------------------

class TestCompareIdentical:
    def test_identical_scores(self, runner):
        """Identical scores should yield no significant difference."""
        config = AblationConfig(
            id="p3_no_quality_eval",
            base_pattern="p3_meridian",
            description="Test",
            component_removed="quality_evaluator",
        )
        scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        base_scores = {"p3_meridian": scores}
        ablation_scores = {"p3_no_quality_eval": list(scores)}

        comparisons = runner.compare_ablations(
            base_scores, ablation_scores, configs=[config]
        )

        assert len(comparisons) == 1
        c = comparisons[0]
        assert abs(c.score_delta) < 1e-10
        assert not c.is_significant
        assert c.effect_label == "negligible"


# ---------------------------------------------------------------------------
# 8. compare_ablations with clearly different scores (significant)
# ---------------------------------------------------------------------------

class TestCompareDifferent:
    def test_clearly_different_scores(self, runner):
        """Very different scores should yield significant difference."""
        config = AblationConfig(
            id="p4_no_triangulation",
            base_pattern="p4_perspective_storm",
            description="Test",
            component_removed="triangulator",
        )
        # Large, consistent difference
        base_scores = {"p4_perspective_storm": [0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89]}
        ablation_scores = {"p4_no_triangulation": [0.3, 0.25, 0.28, 0.32, 0.27, 0.31, 0.26, 0.29]}

        comparisons = runner.compare_ablations(
            base_scores, ablation_scores, configs=[config]
        )

        assert len(comparisons) == 1
        c = comparisons[0]
        assert c.score_delta > 0.5
        assert c.is_significant
        assert c.effect_label == "large"
        assert c.p_value < 0.05


# ---------------------------------------------------------------------------
# 9. generate_ablation_report produces valid markdown with tables
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_has_header(self, runner):
        comparisons = [
            AblationComparison(
                ablation_id="p4_no_conversations",
                base_pattern="p4_perspective_storm",
                component_removed="conversation_sim",
                description="Skip conversations",
                expected_effect="Lower depth",
                base_scores=[0.8, 0.7],
                ablated_scores=[0.5, 0.4],
                base_mean=0.75,
                ablated_mean=0.45,
                score_delta=0.30,
                relative_change=40.0,
                is_significant=True,
                p_value=0.03,
                effect_size=0.8,
                effect_label="large",
            ),
        ]
        report = runner.generate_ablation_report(comparisons)

        assert "# Ablation Study Results" in report
        assert "p4_perspective_storm" in report
        assert "conversation_sim" in report

    def test_report_has_table(self, runner):
        comparisons = [
            AblationComparison(
                ablation_id="p3_no_quality_eval",
                base_pattern="p3_meridian",
                component_removed="quality_evaluator",
                description="Skip eval",
                expected_effect="Lower org",
                base_mean=0.6,
                ablated_mean=0.5,
                score_delta=0.1,
                relative_change=16.7,
                is_significant=False,
                p_value=0.15,
                effect_size=0.3,
                effect_label="small",
            ),
        ]
        report = runner.generate_ablation_report(comparisons)

        assert "|" in report  # Table separators
        assert "quality_evaluator" in report
        assert "Expected:" in report
        assert "Observed:" in report

    def test_report_groups_by_pattern(self, runner):
        comparisons = [
            AblationComparison(
                ablation_id="p4_no_conversations",
                base_pattern="p4_perspective_storm",
                component_removed="conversation_sim",
                description="d1", expected_effect="e1",
                base_mean=0.8, ablated_mean=0.5, score_delta=0.3,
                relative_change=37.5,
            ),
            AblationComparison(
                ablation_id="p3_no_quality_eval",
                base_pattern="p3_meridian",
                component_removed="quality_evaluator",
                description="d2", expected_effect="e2",
                base_mean=0.6, ablated_mean=0.55, score_delta=0.05,
                relative_change=8.3,
            ),
        ]
        report = runner.generate_ablation_report(comparisons)

        # Both patterns should have their own section
        assert "## p3_meridian" in report
        assert "## p4_perspective_storm" in report


# ---------------------------------------------------------------------------
# 10. AblationComparison relative_change calculation
# ---------------------------------------------------------------------------

class TestRelativeChange:
    def test_positive_relative_change(self):
        comp = AblationComparison(
            ablation_id="test",
            base_pattern="p4",
            component_removed="comp",
            description="d",
            expected_effect="e",
            base_mean=0.8,
            ablated_mean=0.6,
            score_delta=0.2,
            relative_change=25.0,  # (0.2 / 0.8) * 100
        )
        assert comp.relative_change == pytest.approx(25.0)

    def test_negative_relative_change(self):
        """When ablated > base, component is harmful."""
        comp = AblationComparison(
            ablation_id="test",
            base_pattern="p4",
            component_removed="comp",
            description="d",
            expected_effect="e",
            base_mean=0.5,
            ablated_mean=0.7,
            score_delta=-0.2,
            relative_change=-40.0,
        )
        assert comp.relative_change == pytest.approx(-40.0)

    def test_zero_base_mean_relative_change(self, runner):
        """When base mean is zero, relative change should be 0."""
        config = AblationConfig(
            id="zero_test",
            base_pattern="p0_baseline",
            description="Zero base",
            component_removed="x",
        )
        base_scores = {"p0_baseline": [0.0, 0.0, 0.0, 0.0, 0.0]}
        ablation_scores = {"zero_test": [0.1, 0.1, 0.1, 0.1, 0.1]}

        comparisons = runner.compare_ablations(
            base_scores, ablation_scores, configs=[config]
        )
        assert len(comparisons) == 1
        assert comparisons[0].relative_change == 0.0
