"""일별 가격과 무위험금리 로더 (yfinance + parquet 캐시)."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from mfin.data.cache import cached, retry

# 자산 유니버스로 쓰는 S&P 500 섹터 SPDR ETF (핵심 5종 + 확장 실험용 6종).
CORE_SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLU"]
ALL_SECTORS = CORE_SECTORS + ["XLB", "XLI", "XLP", "XLY", "XLRE", "XLC"]


def _download_close(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx
    close.index.name = "date"
    close.columns = [str(c) for c in close.columns]
    return close.dropna(how="all").astype(float)


def get_prices(
    tickers: str | Iterable[str],
    start: str = "2005-01-01",
    end: str | None = None,
    max_age: timedelta | None = timedelta(days=1),
    refresh: bool = False,
) -> pd.DataFrame:
    """수정 일별 종가. 종목마다 한 열 (인덱스: 날짜)."""
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    key = f"prices_{'-'.join(tickers)}_{start}_{end or 'latest'}"
    return cached(key, lambda: retry(lambda: _download_close(tickers, start, end)), max_age, refresh)


def irx_to_continuous(irx_percent: pd.Series | float) -> pd.Series | float:
    """^IRX 시세(13주 미국 단기국채 할인수익률, %)를 연속복리 금리로 바꾼다.

    국채 가격 P = 1 - d * 91/360 (할인 기준)이므로 r = -ln(P) / (91/365).
    """
    d = np.asarray(irx_percent, dtype=float) / 100.0
    r = -np.log1p(-d * 91.0 / 360.0) / (91.0 / 365.0)
    if isinstance(irx_percent, pd.Series):
        return pd.Series(r, index=irx_percent.index, name="r")
    return float(r)


def get_risk_free(start: str = "2005-01-01", end: str | None = None, **kw) -> pd.Series:
    """^IRX로 구한 일별 연속복리 무위험금리."""
    irx = get_prices(["^IRX"], start, end, **kw)["^IRX"].dropna()
    return irx_to_continuous(irx)


def latest_risk_free(default: float = 0.04) -> float:
    """가장 최근 무위험금리. 다운로드가 실패하면 ``default``."""
    try:
        start = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        r = get_risk_free(start=start)
        return float(r.iloc[-1])
    except Exception:  # noqa: BLE001
        return default
