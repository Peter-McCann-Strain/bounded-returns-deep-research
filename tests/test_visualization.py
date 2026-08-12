"""Tests for deep_research.visualization.charts module."""

import numpy as np
import pytest
from pathlib import Path

from deep_research.visualization.charts import (
    dimension_heatmap,
    bootstrap_ci_plot,
    radar_chart,
    cost_quality_scatter,
    critical_difference_diagram,
    ablation_bar_chart,
    concordance_heatmap,
    performance_profile,
    generate_all_figures,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def dim_scores():
    """Sample dimension scores for 3 patterns x 3 dimensions."""
    return {
        'p0_baseline': {'factual_accuracy': 0.2, 'coverage': 0.77, 'analytical_depth': 0.9},
        'p3_meridian': {'factual_accuracy': 0.15, 'coverage': 0.65, 'analytical_depth': 1.0},
        'p4_perspective_storm': {'factual_accuracy': 0.13, 'coverage': 0.8, 'analytical_depth': 0.85},
    }


@pytest.fixture
def ci_data():
    """Sample CI data for 3 systems."""
    return [
        {'system': 'p0_baseline', 'mean': 0.458, 'ci_lower': 0.39, 'ci_upper': 0.52},
        {'system': 'p3_meridian', 'mean': 0.505, 'ci_lower': 0.44, 'ci_upper': 0.57},
        {'system': 'p4_perspective_storm', 'mean': 0.557, 'ci_lower': 0.49, 'ci_upper': 0.62},
    ]


@pytest.fixture
def cost_data():
    """Sample cost/quality data."""
    return [
        {'pattern': 'p0_baseline', 'quality': 0.458, 'tokens': 5000, 'latency_s': 30.0},
        {'pattern': 'p3_meridian', 'quality': 0.505, 'tokens': 15000, 'latency_s': 90.0},
        {'pattern': 'p4_perspective_storm', 'quality': 0.557, 'tokens': 20000, 'latency_s': 120.0},
    ]


@pytest.fixture
def ablation_data():
    """Sample ablation comparison data."""
    return [
        {'component': 'web_search', 'pattern': 'p3_meridian', 'base_mean': 0.505, 'ablated_mean': 0.35, 'significant': True},
        {'component': 'multi_perspective', 'pattern': 'p4_perspective_storm', 'base_mean': 0.557, 'ablated_mean': 0.42, 'significant': True},
        {'component': 'depth_control', 'pattern': 'p5_hierarchical_wd', 'base_mean': 0.489, 'ablated_mean': 0.47, 'significant': False},
    ]


@pytest.fixture
def concordance_data():
    """Sample concordance tau matrix."""
    return {
        'LLM Judge': {'LLM Judge': 1.0, 'Manual': 0.6, 'Keyword': 0.3},
        'Manual': {'LLM Judge': 0.6, 'Manual': 1.0, 'Keyword': 0.5},
        'Keyword': {'LLM Judge': 0.3, 'Manual': 0.5, 'Keyword': 1.0},
    }


# ── Individual chart tests ───────────────────────────────────────────────────


def test_dimension_heatmap_generates_file(tmp_path, dim_scores):
    out = tmp_path / "heatmap.png"
    dimension_heatmap(dim_scores, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_bootstrap_ci_plot_generates_file(tmp_path, ci_data):
    out = tmp_path / "ci.png"
    bootstrap_ci_plot(ci_data, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_radar_chart_generates_file(tmp_path, dim_scores):
    out = tmp_path / "radar.png"
    radar_chart(dim_scores, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_cost_quality_scatter_generates_file(tmp_path, cost_data):
    out = tmp_path / "scatter.png"
    cost_quality_scatter(cost_data, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_critical_difference_diagram_generates_file(tmp_path):
    avg_ranks = {'p0_baseline': 3.5, 'p3_meridian': 2.1, 'p4_perspective_storm': 1.5}
    out = tmp_path / "cd.png"
    critical_difference_diagram(avg_ranks, n_tasks=29, cd=1.8, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_ablation_bar_chart_generates_file(tmp_path, ablation_data):
    out = tmp_path / "ablation.png"
    ablation_bar_chart(ablation_data, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_concordance_heatmap_generates_file(tmp_path, concordance_data):
    out = tmp_path / "concordance.png"
    concordance_heatmap(concordance_data, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_performance_profile_generates_file(tmp_path):
    rng = np.random.RandomState(42)
    score_matrix = rng.rand(20, 3)
    names = ['p0_baseline', 'p3_meridian', 'p4_perspective_storm']
    out = tmp_path / "perf.png"
    performance_profile(score_matrix, names, out)
    assert out.exists()
    assert out.stat().st_size > 0


# ── generate_all_figures ─────────────────────────────────────────────────────


def test_generate_all_figures_returns_correct_paths(
    tmp_path, dim_scores, ci_data, cost_data, ablation_data, concordance_data,
):
    rng = np.random.RandomState(42)
    score_matrix = rng.rand(10, 3)
    system_names = ['p0_baseline', 'p3_meridian', 'p4_perspective_storm']
    avg_ranks = {'p0_baseline': 3.0, 'p3_meridian': 2.0, 'p4_perspective_storm': 1.0}

    out_dir = tmp_path / "figures"
    paths = generate_all_figures(
        results_dir=tmp_path,
        output_dir=out_dir,
        dimension_scores=dim_scores,
        ci_data=ci_data,
        cost_data=cost_data,
        ablation_data=ablation_data,
        concordance_data=concordance_data,
        score_matrix=score_matrix,
        system_names=system_names,
        avg_ranks=avg_ranks,
        n_tasks=10,
        cd=1.5,
    )

    # Should generate 8 figures: heatmap, radar, CI, cost, ablation,
    # concordance, performance profiles, CD diagram
    assert len(paths) == 8
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_generate_all_figures_partial_data(tmp_path, dim_scores):
    """Only dimension_scores provided -> should generate 2 figures."""
    out_dir = tmp_path / "partial"
    paths = generate_all_figures(
        results_dir=tmp_path,
        output_dir=out_dir,
        dimension_scores=dim_scores,
    )
    assert len(paths) == 2
    names = [p.name for p in paths]
    assert "dimension_heatmap.png" in names
    assert "radar_chart.png" in names


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_single_system(tmp_path):
    """Single system should still produce valid figures."""
    single_dim = {
        'p0_baseline': {'coverage': 0.5, 'analytical_depth': 0.8},
    }
    out = tmp_path / "single_heatmap.png"
    dimension_heatmap(single_dim, out)
    assert out.exists()
    assert out.stat().st_size > 0

    single_ci = [
        {'system': 'p0_baseline', 'mean': 0.5, 'ci_lower': 0.4, 'ci_upper': 0.6},
    ]
    out2 = tmp_path / "single_ci.png"
    bootstrap_ci_plot(single_ci, out2)
    assert out2.exists()
    assert out2.stat().st_size > 0


def test_single_dimension(tmp_path):
    """Single dimension should still produce valid figures."""
    single_dim_scores = {
        'p0_baseline': {'coverage': 0.7},
        'p3_meridian': {'coverage': 0.8},
    }
    out = tmp_path / "single_dim_heatmap.png"
    dimension_heatmap(single_dim_scores, out)
    assert out.exists()
    assert out.stat().st_size > 0

    out2 = tmp_path / "single_dim_radar.png"
    radar_chart(single_dim_scores, out2)
    assert out2.exists()
    assert out2.stat().st_size > 0


def test_cost_quality_scatter_latency_metric(tmp_path, cost_data):
    """Test scatter with latency_s instead of tokens."""
    out = tmp_path / "scatter_latency.png"
    cost_quality_scatter(cost_data, out, x_metric="latency_s")
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_dir_created_if_missing(tmp_path, dim_scores):
    """Output directory is created automatically."""
    out = tmp_path / "nested" / "dir" / "heatmap.png"
    dimension_heatmap(dim_scores, out)
    assert out.exists()
