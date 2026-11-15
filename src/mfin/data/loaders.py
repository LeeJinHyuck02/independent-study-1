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


def _download_field(tickers: list[str], start: str, end: str | None, field: str = "Close") -> pd.DataFrame:
    """yfinance에서 한 필드(Close, Volume 등)만 받아 종목마다 한 열로 정리한다."""
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
        out = df[field]
    else:
        out = df[[field]].rename(columns={field: tickers[0]})
    if isinstance(out, pd.Series):
        out = out.to_frame(tickers[0])
    out = out.reindex(columns=tickers)
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    out.index.name = "date"
    out.columns = [str(c) for c in out.columns]
    return out.dropna(how="all").astype(float)


def _tickers(tickers: str | Iterable[str]) -> list[str]:
    return [tickers] if isinstance(tickers, str) else list(tickers)


def get_prices(
    tickers: str | Iterable[str],
    start: str = "2005-01-01",
    end: str | None = None,
    max_age: timedelta | None = timedelta(days=1),
    refresh: bool = False,
) -> pd.DataFrame:
    """수정 일별 종가. 종목마다 한 열 (인덱스: 날짜)."""
    tickers = _tickers(tickers)
    key = f"prices_{'-'.join(tickers)}_{start}_{end or 'latest'}"
    return cached(key, lambda: retry(lambda: _download_field(tickers, start, end, "Close")), max_age, refresh)


def get_volume(
    tickers: str | Iterable[str],
    start: str = "2005-01-01",
    end: str | None = None,
    max_age: timedelta | None = timedelta(days=1),
    refresh: bool = False,
) -> pd.DataFrame:
    """일별 거래량(주). 시장충격 행렬 Lam의 평균 거래량(ADV) 추정에 쓴다."""
    tickers = _tickers(tickers)
    key = f"volume_{'-'.join(tickers)}_{start}_{end or 'latest'}"
    return cached(key, lambda: retry(lambda: _download_field(tickers, start, end, "Volume")), max_age, refresh)


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
