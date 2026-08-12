"""Canonical project-root resolution.

Analysis and experiment scripts need to locate ``data/``, ``results/``, and the
paper directory without hardcoding a local absolute path. Resolution order:

1. ``DR_PROJECT_ROOT`` environment variable, if set.
2. The repository root inferred from this file's location, which is correct for
   an editable install (``pip install -e .``) and for running from a checkout.

Set ``DR_PROJECT_ROOT`` explicitly when running against a non-editable install
or when the data tree lives outside the checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "project_root", "data_dir", "analysis_dir", "results_dir"]


def project_root() -> Path:
    """Return the repository root as an absolute path."""
    override = os.environ.get("DR_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT: Path = project_root()


def data_dir() -> Path:
    """Return the ``data/`` directory."""
    return project_root() / "data"


def analysis_dir() -> Path:
    """Return the ``data/analysis/`` directory holding the tidy analysis frames."""
    return project_root() / "data" / "analysis"


def results_dir() -> Path:
    """Return the ``results/`` directory holding raw run and judge outputs."""
    return project_root() / "results"
