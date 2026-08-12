"""Tests for central configuration."""

from deep_research.config import (
    MODELS,
    ModelSpec,
    DEFAULT_MODEL,
    SEARCH_MODEL,
    DATA_DIR,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    ensure_runtime_dirs,
)


class TestModels:
    def test_default_model_exists(self):
        assert DEFAULT_MODEL in MODELS

    def test_search_model_exists(self):
        assert SEARCH_MODEL in MODELS

    def test_model_spec_fields(self):
        spec = MODELS["gpt-4o"]
        assert isinstance(spec, ModelSpec)
        assert spec.deployment != ""
        assert spec.max_context > 0

    def test_ptu_model_zero_cost(self):
        spec = MODELS["gpt-4o"]
        assert spec.cost_per_1k_input == 0.0
        assert spec.cost_per_1k_output == 0.0

    def test_paid_model_has_cost(self):
        spec = MODELS["gpt-4o-mini"]
        assert spec.cost_per_1k_input > 0
        assert spec.cost_per_1k_output > 0

    def test_all_models_have_deployment(self):
        for name, spec in MODELS.items():
            assert spec.deployment, f"Model {name} missing deployment"


class TestPaths:
    def test_directories_are_configured_paths(self):
        assert DATA_DIR.name == "data"
        assert CHECKPOINTS_DIR.name == "checkpoints"
        assert REPORTS_DIR.name == "reports"

    def test_ensure_runtime_dirs_creates_paths(self):
        ensure_runtime_dirs()
        assert DATA_DIR.exists()
        assert CHECKPOINTS_DIR.exists()
        assert REPORTS_DIR.exists()


def test_config_import_does_not_mutate_environment(tmp_path):
    import os
    import subprocess
    import sys

    probe = "DR_PUBLIC_IMPORT_SIDE_EFFECT_PROBE"
    env = os.environ.copy()
    env.pop(probe, None)
    command = (
        "from pathlib import Path; "
        "Path('.env').write_text('DR_PUBLIC_IMPORT_SIDE_EFFECT_PROBE=mutated\\n'); "
        "import deep_research.config; "
        "import os; print(os.environ.get('DR_PUBLIC_IMPORT_SIDE_EFFECT_PROBE', ''))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""
