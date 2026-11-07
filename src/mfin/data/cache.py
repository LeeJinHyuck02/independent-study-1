"""내려받은 시장 데이터의 parquet 캐시.

yfinance는 비공식 API라서 요청이 실패하거나 횟수 제한에 걸릴 수 있다. 모든 다운로드는
:func:`cached`를 거친다. 결과를 ``data_cache/``에 저장하고, 새로 받기가 실패하면 마지막
저장본을 쓴다.
"""

from __future__ import annotations

import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd

from mfin.paths import CACHE_DIR

T = TypeVar("T")


class DataUnavailableError(RuntimeError):
    """데이터를 받을 수 없고 캐시 저장본도 없을 때 발생한다."""


def cache_path(key: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("_")
    return CACHE_DIR / f"{safe}.parquet"


def retry(fn: Callable[[], T], tries: int = 3, wait: float = 2.0) -> T:
    """``fn``을 최대 ``tries``번 호출한다. 시도 사이의 대기 시간은 점점 늘어난다(선형)."""
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - 네트워크 라이브러리는 여러 종류의 예외를 낸다
            last = e
            if i < tries - 1:
                time.sleep(wait * (i + 1))
    assert last is not None
    raise last


def cached(
    key: str,
    fetch_fn: Callable[[], pd.DataFrame],
    max_age: timedelta | None = timedelta(days=1),
    refresh: bool = False,
) -> pd.DataFrame:
    """``key``에 해당하는 캐시 DataFrame을 반환하거나, 새로 받아 저장한 뒤 반환한다.

    매개변수
    ----------
    key      : 캐시 식별자 (파일 이름으로 바뀐다)
    fetch_fn : 데이터를 내려받는 인자 없는 함수
    max_age  : 이보다 오래된 저장본은 새로 받는다 (None = 만료 없음)
    refresh  : 캐시가 최신이어도 강제로 새로 받는다
    """
    path = cache_path(key)
    if path.exists() and not refresh:
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if max_age is None or age <= max_age:
            return pd.read_parquet(path)

    try:
        df = fetch_fn()
    except Exception as e:  # noqa: BLE001
        if path.exists():
            warnings.warn(f"download failed ({e!r}); using stale cache {path.name}", stacklevel=2)
            return pd.read_parquet(path)
        raise DataUnavailableError(f"could not download {key!r}: {e!r}") from e

    if df is None or df.empty:
        if path.exists():
            warnings.warn(f"download for {key!r} was empty; using stale cache", stacklevel=2)
            return pd.read_parquet(path)
        raise DataUnavailableError(f"download for {key!r} returned no rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
