"""VIX 기간구조(9일 ... 1년). 백테스트에서 내재변동성 대신 쓰는 대리변수.

과거 SPX 옵션체인은 무료로 구할 수 없으므로, 백테스트에서는 SPX 옵션가격으로 계산되는
Cboe 변동성 지수를 쓴다.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from mfin.data.loaders import get_prices

# 종목 코드 -> 만기(년)
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
    """일별 VIX 기간구조 (소수 단위).

    열은 만기(년, float, 오름차순), 인덱스는 날짜다. 그 날짜에 아직 없던 지수
    (예: 2007년 이전의 VIX1Y)는 NaN이다.
    """
    raw = get_prices(list(VIX_TENORS), start, end, max_age=max_age, refresh=refresh)
    term = raw.reindex(columns=list(VIX_TENORS)) / 100.0
    term.columns = [VIX_TENORS[c] for c in term.columns]
    return term.sort_index(axis=1)


def latest_vix_term(**kw) -> pd.Series:
    """VIX 기간구조의 가장 최근 행 (만기 -> 변동성). NaN은 뺀다."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    term = get_vix_term(start=start, **kw).dropna(how="all")
    row = term.iloc[-1].dropna()
    row.name = term.index[-1]
    return row
