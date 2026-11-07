"""프로젝트 폴더 경로 도우미 (캐시와 결과는 프로젝트 루트에 둔다)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """프로젝트 루트(pyproject.toml이 있는 폴더)를 반환한다.

    찾는 순서: $MFIN_ROOT, 이 파일의 상위 폴더들(편집 가능 설치), 현재 작업 폴더와 그 상위
    폴더들.
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
