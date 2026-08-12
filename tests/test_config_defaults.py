"""Tests for configuration defaults related to evaluation pipeline."""

from deep_research.config import EVAL_PIPELINE, JUDGE


class TestEvalPipelineDefaults:
    def test_default_n_repeats_is_three(self):
        """Default n_repeats should be 3 for variance estimation."""
        assert EVAL_PIPELINE.default_n_repeats == 3

    def test_default_random_seed_is_42(self):
        """Default random seed should be 42 for reproducibility."""
        assert EVAL_PIPELINE.default_random_seed == 42

    def test_report_truncation_words(self):
        assert EVAL_PIPELINE.report_truncation_words == 12_000

    def test_passes_per_judge(self):
        assert EVAL_PIPELINE.passes_per_judge == 3

    def test_max_concurrent_runs(self):
        assert EVAL_PIPELINE.max_concurrent_runs == 2


class TestJudgeDefaults:
    def test_temperature(self):
        assert JUDGE.temperature == 0.1

    def test_max_tokens(self):
        assert JUDGE.max_tokens == 8192

    def test_max_concurrent(self):
        assert JUDGE.max_concurrent == 3
