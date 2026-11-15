"""백테스트 성과 지표: CAGR, 연변동성, Sharpe, Sortino, MDD, Calmar, 회전율, 비용 손실."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(nav: pd.Series) -> pd.Series:
    """고점 대비 하락률 (0 이하)."""
    return nav / nav.cummax() - 1.0


def cagr(nav: pd.Series, periods_per_year: int = 252) -> float:
    years = (len(nav) - 1) / periods_per_year
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)


def performance_table(
    nav_net: pd.Series,
    nav_gross: pd.Series,
    rf_daily: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """전략 하나의 성과 지표.

    Sharpe와 Sortino는 무위험수익률(일별) 대비 초과수익으로 계산한다. 회전율은 리밸런싱마다
    sum |거래액| / NAV(양방향)의 연간 합계, 비용 손실은 비용 전후 CAGR의 차이다.
    """
    ret = (nav_net / nav_net.shift(1) - 1.0).iloc[1:]
    rf = rf_daily.reindex(ret.index).fillna(0.0)
    ex = ret - rf
    years = len(ret) / periods_per_year
    sd = float(ex.std(ddof=1)) if len(ex) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2)))
    mdd = float(drawdown(nav_net).min())
    g_net = cagr(nav_net, periods_per_year)
    nan = float("nan")
    return {
        "cagr": g_net,
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(periods_per_year)) if len(ret) > 1 else 0.0,
        "sharpe": float(ex.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else nan,
        "sortino": float(ex.mean() * periods_per_year / (downside * np.sqrt(periods_per_year))) if downside > 0 else nan,
        "mdd": mdd,
        "calmar": g_net / abs(mdd) if mdd < 0 else nan,
        "ann_turnover": float(turnover.sum() / years),
        "cost_drag": cagr(nav_gross, periods_per_year) - g_net,
        "total_cost": float(costs.sum()),
        "final_nav": float(nav_net.iloc[-1]),
    }
