"""Parquet cache for downloaded market data.

yfinance is an unofficial API: requests can fail or be rate limited. Every download
goes through :func:`cached`, which stores the result under ``data_cache/`` and falls
back to the last stored copy when a refresh fails.
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
    """Raised when data cannot be downloaded and no cached copy exists."""


def cache_path(key: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("_")
    return CACHE_DIR / f"{safe}.parquet"


def retry(fn: Callable[[], T], tries: int = 3, wait: float = 2.0) -> T:
    """Call ``fn`` up to ``tries`` times with a growing pause between attempts."""
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network libraries raise many types
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
    """Return the cached DataFrame for ``key`` or fetch and store a new one.

    Parameters
    ----------
    key      : cache identifier (turned into a file name)
    fetch_fn : zero-argument function that downloads the data
    max_age  : cached copies older than this are refreshed (None = never expire)
    refresh  : force a download even when the cache is fresh
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
