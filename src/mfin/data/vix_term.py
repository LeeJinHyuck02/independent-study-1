"""VIX term structure (9 days ... 1 year) as a backtest proxy for implied volatility.

Historical SPX option chains are not freely available, so the backtest uses the Cboe
volatility indices, which are themselves computed from SPX option prices.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from mfin.data.loaders import get_prices

# ticker -> tenor in years
VIX_TENORS: dict[str, float] = {
    "^VIX9D": 9 / 365,
    "^VIX": 30 / 365,
    "^VIX3M": 0.25,
    "^VIX6M": 0.5,
    "^VIX1Y": 1.0,
}


def get_vix_term(
    start: str = "2005-01-01",
    end: str | None = None,
    max_age: timedelta | None = timedelta(days=1),
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily VIX term structure in decimals.

    Columns are tenors in years (float, ascending); index is the date. Indices that
    did not exist yet on a date (e.g. VIX1Y before 2007) are NaN.
    """
    raw = get_prices(list(VIX_TENORS), start, end, max_age=max_age, refresh=refresh)
    term = raw.reindex(columns=list(VIX_TENORS)) / 100.0
    term.columns = [VIX_TENORS[c] for c in term.columns]
    return term.sort_index(axis=1)


def latest_vix_term(**kw) -> pd.Series:
    """Most recent row of the VIX term structure (tenor -> vol), NaNs dropped."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    term = get_vix_term(start=start, **kw).dropna(how="all")
    row = term.iloc[-1].dropna()
    row.name = term.index[-1]
    return row
