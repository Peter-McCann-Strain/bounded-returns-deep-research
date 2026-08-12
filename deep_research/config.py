"""Central configuration — loads .env once, exposes typed settings.

All tuneable defaults live here.  Consumers import from this module rather
than hard-coding magic numbers.  Values can be overridden via environment
variables (prefixed where noted) or by editing the dataclass defaults below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import dotenv_values

# Read .env from the project root without mutating process environment.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV = {k: v or "" for k, v in dotenv_values(_PROJECT_ROOT / ".env").items()}


def _env(key: str, default: str) -> str:
    """Return the configured value for ``key``, falling back to ``default``.

    A key that is present but empty (``FOO=`` in .env, which is how the shipped
    .env.example marks optional settings) counts as unset. Without this, copying
    .env.example verbatim — exactly what the README instructs — made every typed
    getter raise on ``float("")``.
    """
    value = os.environ.get(key)
    if value is None:
        value = _DOTENV.get(key)
    if value is None or value == "":
        return default
    return value


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    value = _env(key, "" if default is False else "true")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = _env("OPENAI_API_KEY", "")
USE_AZURE_OPENAI: bool = _env_bool("USE_AZURE_OPENAI", False)

AZURE_OPENAI_API_KEY: str = _env("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = _env("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION: str = _env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ── Search OpenAI (for Responses API web search) ─────────────────────────────

SEARCH_OPENAI_API_KEY: str = _env("SEARCH_OPENAI_API_KEY", OPENAI_API_KEY or AZURE_OPENAI_API_KEY)
SEARCH_OPENAI_ENDPOINT: str = _env("SEARCH_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT)

# ── Judge endpoint (GPT-5.2 for LLM-as-judge evaluation) ────────────────────

JUDGE_OPENAI_API_KEY: str = _env("JUDGE_OPENAI_API_KEY", SEARCH_OPENAI_API_KEY)
JUDGE_OPENAI_ENDPOINT: str = _env("JUDGE_OPENAI_ENDPOINT", SEARCH_OPENAI_ENDPOINT)
JUDGE_MODEL: str = _env("JUDGE_MODEL", "gpt-5.2")

# ── Tavily ────────────────────────────────────────────────────────────────────

TAVILY_API_KEY: str = _env("TAVILY_API_KEY", "")

# ── Model registry ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    deployment: str
    cost_per_1k_input: float   # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    max_context: int = 128_000
    supports_json: bool = True
    use_max_completion_tokens: bool = False  # newer models require this param


# ── Default models (change here to switch everything) ────────────────────────

DEFAULT_MODEL: str = _env("DEFAULT_MODEL", "gpt-4o")
SEARCH_MODEL: str = _env("SEARCH_MODEL", "gpt-4o-mini")
SEARCH_BACKEND: str = _env("SEARCH_BACKEND", "bing")

MODELS: Dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec(
        # Deployment names are provisioning-specific. Set GPT4O_DEPLOYMENT to the
        # name used by your Azure OpenAI resource; the default matches the model.
        deployment=_env("GPT4O_DEPLOYMENT", "gpt-4o"),
        cost_per_1k_input=0.0,    # PTU — pre-paid, no per-token cost
        cost_per_1k_output=0.0,
        use_max_completion_tokens=True,
    ),
    "gpt-4o-mini": ModelSpec(
        deployment="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "gpt-4.1": ModelSpec(
        deployment="gpt-4.1",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.008,
    ),
    "gpt-5.2": ModelSpec(
        deployment="gpt-5.2",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.012,
        use_max_completion_tokens=True,
    ),
    # Local models — GPU inference, zero API cost
    "Qwen2.5-7B-Instruct": ModelSpec(
        deployment="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=32_768,
    ),
    "DeepResearcher-7b": ModelSpec(
        deployment="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=32_768,
    ),
    "DR-Judge-7B": ModelSpec(
        deployment="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=32_768,
    ),
}

# ── Budget ────────────────────────────────────────────────────────────────────

MAX_COST_PER_RUN: float = _env_float("MAX_COST_PER_RUN", 2.00)
MAX_COST_EVAL_RUN: float = _env_float("MAX_COST_EVAL_RUN", 10.00)


# ── LLM call defaults ────────────────────────────────────────────────────────
# Shared by llm_caller, llm_judge, multi_judge, citation_verifier, etc.


@dataclass(frozen=True)
class RetryConfig:
    """Retry and backoff settings for LLM API calls."""

    max_retries: int = 20              # More retries — rate limits are transient, not fatal
    base_delay: float = 0.5            # Initial backoff for rate-limit retries
    conn_base_delay: float = 1.0       # Initial backoff for connection errors
    max_delay: float = 60.0            # Higher cap — let sustained overload resolve
    jitter_max: float = 2.0            # Random jitter ceiling


@dataclass(frozen=True)
class TimeoutConfig:
    """httpx timeout settings (seconds)."""

    connect: float = 30.0
    read: float = 300.0                # 5 min for long generation under load
    write: float = 60.0
    pool: float = 30.0


@dataclass(frozen=True)
class ConnectionPoolConfig:
    """httpx connection pool settings."""

    max_connections: int = 50
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 15.0


@dataclass(frozen=True)
class RateLimitConfig:
    """Concurrency and rate limiting for PTU endpoint."""

    max_concurrent: int = 12           # Semaphore cap
    rpm: int = 200                     # Requests per minute (~80% PTU capacity)


# Singleton instances — import these rather than constructing your own.
RETRY = RetryConfig()
TIMEOUTS = TimeoutConfig()
POOL = ConnectionPoolConfig()
RATE_LIMIT = RateLimitConfig()


# ── Judge-specific defaults ──────────────────────────────────────────────────
# Shared by llm_judge.py and multi_judge.py.

@dataclass(frozen=True)
class JudgeDefaults:
    """Configuration defaults for LLM-as-judge evaluation."""

    max_concurrent: int = 3            # Conservative for standard (non-PTU) deployment
    max_tokens: int = 8192
    temperature: float = 0.1
    seed: int = 42                     # passed as `seed=` to the OpenAI API for reproducibility
    sdk_max_retries: int = 0           # We handle retries ourselves
    read_timeout: float = 600.0        # 10 min — judge prompts are large


JUDGE = JudgeDefaults()


# ── Evaluation pipeline defaults ─────────────────────────────────────────────


@dataclass(frozen=True)
class EvalPipelineDefaults:
    """Configuration for the V2 evaluation pipeline."""

    max_concurrent_runs: int = 2
    passes_per_judge: int = 3
    bootstrap_resamples: int = 10_000
    statistical_alpha: float = 0.05
    report_truncation_words: int = 12_000
    default_n_repeats: int = 3             # Repeated runs for variance estimation
    default_random_seed: int = 42          # Reproducible randomization


EVAL_PIPELINE = EvalPipelineDefaults()


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = _PROJECT_ROOT / _env("DR_DATA_DIR", "data")
ARTIFACTS_DIR = _PROJECT_ROOT / _env("DR_ARTIFACTS_DIR", "artifacts")
PAPERS_DIR = _PROJECT_ROOT / _env("DR_PAPERS_DIR", "papers")
DOCS_DIR = _PROJECT_ROOT / _env("DR_DOCS_DIR", "docs")

# Legacy compatibility roots. In the publication-prep layout these may be local
# symlinks into ARTIFACTS_DIR so older scripts continue to run unchanged.
CHECKPOINTS_DIR = _PROJECT_ROOT / _env("DR_CHECKPOINTS_DIR", "checkpoints")
REPORTS_DIR = _PROJECT_ROOT / _env("DR_REPORTS_DIR", "reports")
RESULTS_DIR = _PROJECT_ROOT / _env("DR_RESULTS_DIR", "results")

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments" / "canonical"
JUDGES_DIR = ARTIFACTS_DIR / "judges"
PHASE_REPORTS_DIR = ARTIFACTS_DIR / "phase_reports"
REPLICATION_DIR = ARTIFACTS_DIR / "replication"
CACHE_DIR = ARTIFACTS_DIR / "caches"
MODELS_DIR = ARTIFACTS_DIR / "models"

RUNTIME_DIRS = (
    DATA_DIR,
    ARTIFACTS_DIR,
    PAPERS_DIR,
    DOCS_DIR,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    EXPERIMENTS_DIR,
    JUDGES_DIR,
    PHASE_REPORTS_DIR,
    REPLICATION_DIR,
    CACHE_DIR,
)


def ensure_runtime_dirs() -> None:
    """Create runtime directories explicitly at command/run boundaries."""
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def get_environment_metadata() -> dict:
    """Capture environment metadata for reproducibility."""
    import platform
    import subprocess
    import sys
    from datetime import datetime, timezone

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "default_model": DEFAULT_MODEL,
        "judge_model": JUDGE_MODEL,
    }
    # Package versions
    for pkg_name in ["openai", "httpx", "numpy", "scipy", "structlog"]:
        try:
            mod = __import__(pkg_name)
            env[f"{pkg_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(Path(__file__).parent.parent), check=False,
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["git_commit"] = "unavailable"
    # Deployment info
    for name, spec in MODELS.items():
        env[f"deployment_{name}"] = spec.deployment
    return env
