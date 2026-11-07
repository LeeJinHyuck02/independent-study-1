"""Daily price and risk-free rate loaders (yfinance + parquet cache)."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from mfin.data.cache import cached, retry

# S&P 500 sector SPDR ETFs used as the asset universe (5 core + 6 for the extension).
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
    """Adjusted daily closes, one column per ticker (index: date)."""
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    key = f"prices_{'-'.join(tickers)}_{start}_{end or 'latest'}"
    return cached(key, lambda: retry(lambda: _download_close(tickers, start, end)), max_age, refresh)


def irx_to_continuous(irx_percent: pd.Series | float) -> pd.Series | float:
    """Convert the ^IRX quote (13-week T-bill discount yield, %) to a continuous rate.

    Bill price P = 1 - d * 91/360 (discount basis), so r = -ln(P) / (91/365).
    """
    d = np.asarray(irx_percent, dtype=float) / 100.0
    r = -np.log1p(-d * 91.0 / 360.0) / (91.0 / 365.0)
    if isinstance(irx_percent, pd.Series):
        return pd.Series(r, index=irx_percent.index, name="r")
    return float(r)


def get_risk_free(start: str = "2005-01-01", end: str | None = None, **kw) -> pd.Series:
    """Daily continuously compounded risk-free rate from ^IRX."""
    irx = get_prices(["^IRX"], start, end, **kw)["^IRX"].dropna()
    return irx_to_continuous(irx)


def latest_risk_free(default: float = 0.04) -> float:
    """Most recent risk-free rate; ``default`` if the download fails."""
    try:
        start = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        r = get_risk_free(start=start)
        return float(r.iloc[-1])
    except Exception:  # noqa: BLE001
        return default
