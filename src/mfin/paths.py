"""Project directory helpers (cache and results live at the project root)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the project root (the folder that holds pyproject.toml).

    Order: $MFIN_ROOT, then the parents of this file (editable install), then the
    current working directory and its parents.
    """
    env = os.environ.get("MFIN_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for p in [*here.parents, cwd, *cwd.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return cwd


ROOT = project_root()
CACHE_DIR = ROOT / "data_cache"
SNAPSHOT_DIR = CACHE_DIR / "snapshots"
RESULTS_DIR = ROOT / "results"
