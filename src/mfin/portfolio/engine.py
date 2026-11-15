"""추천 엔진 (모듈 C): 시장 상태 -> 목표 비중 + 위험 리포트 + 체결 스케줄.

    python -m mfin.portfolio.engine --gamma 5 --aum 1e6          (오늘 기준 추천, 네트워크 필요)
    python -m mfin.portfolio.engine --gamma 5 --aum 1e6 --no-chain  (옵션체인 대신 VIX 사용)

solver="analytic"(기본)은 Merton 엔진의 구간 QP(정확한 해)를 쓴다. solver="neural"은 13주차
PI-DeepONet 경로를 연결할 자리다.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mfin.engine.merton import QPResult, merton_unconstrained, period_inputs, solve_mv_qp
from mfin.portfolio.estimators import EstimatorConfig, MarketState
from mfin.portfolio.execution import (
    ExecutionConfig,
    ExecutionSchedule,
    impact_matrix,
    order_from_weights,
    plan_execution,
    price_cov_daily,
)
from mfin.portfolio.risk import RiskReport, risk_report

CASH = "CASH"


@dataclass
class Recommendation:
    asof: pd.Timestamp
    weights: pd.Series  # 자산별 목표 비중 + CASH (합 = 1)
    unconstrained: pd.Series  # 무제약 Merton 비중 (레버리지·공매도 허용 시)
    qp: QPResult
    risk: RiskReport
    orders: pd.Series | None  # 주문 수량 (주, 양수 = 매도)
    schedule: ExecutionSchedule | None
    solver: str
    seconds: float


def _solve_analytic(state: MarketState, gamma: float, horizon: float) -> tuple[QPResult, np.ndarray]:
    m, S = period_inputs(state.coeffs, 0.0, horizon)
    return solve_mv_qp(m, S, gamma), merton_unconstrained(m, S, gamma)


def _solve_neural(state: MarketState, gamma: float, horizon: float):
    raise NotImplementedError("week 13: the PI-DeepONet solver (mfin.deeponet) is not implemented yet")


_SOLVERS: dict[str, Callable] = {"analytic": _solve_analytic, "neural": _solve_neural}


def recommend(
    state: MarketState,
    gamma: float,
    *,
    solver: str = "analytic",
    horizon: float = 1 / 12,
    current_weights=None,
    aum: float | None = None,
    exec_cfg: ExecutionConfig | None = None,
    put_hedge: bool = False,
) -> Recommendation:
    """목표 비중을 계산한다. aum을 주면 현재 비중(기본: 전액 현금)에서의 체결 스케줄도 만든다."""
    if solver not in _SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {sorted(_SOLVERS)}")
    t0 = time.perf_counter()
    qp, pi_u = _SOLVERS[solver](state, gamma, horizon)
    seconds = time.perf_counter() - t0
    idx = state.tickers
    weights = pd.Series(np.append(qp.pi, 1.0 - qp.pi.sum()), index=idx + [CASH], name="weight")
    report = risk_report(qp.pi, state.coeffs, horizon, state.beta, state.skew, put_hedge)

    orders, schedule = None, None
    if aum is not None:
        cfg = exec_cfg or ExecutionConfig()
        pi_cur = np.zeros(len(idx)) if current_weights is None else np.asarray(current_weights, dtype=float)
        q0 = order_from_weights(qp.pi, pi_cur, aum, state.prices)
        orders = pd.Series(q0, index=idx, name="shares")
        if state.adv_shares is not None:
            cov = state.coeffs.mean_cov(0.0, horizon)
            sig_d = np.sqrt(np.diag(cov) / cfg.periods_per_year)
            Lam = impact_matrix(state.prices, sig_d, state.adv_shares, cfg.impact_eta)
            Sp = price_cov_daily(state.prices, cov, cfg.periods_per_year)
            phi = cfg.phi if cfg.phi is not None else gamma / (2.0 * aum)
            schedule = plan_execution(q0, Lam, Sp, phi, cfg.horizon_days, cfg.n_buckets)
    return Recommendation(state.asof, weights, pd.Series(pi_u, index=idx, name="unconstrained"), qp, report,
                          orders, schedule, solver, seconds)


# ------------------------------------------------------------------------------- 실시간 상태


def live_state(
    tickers: list[str] | None = None,
    bench: str = "SPY",
    use_chain: bool = True,
    cfg: EstimatorConfig | None = None,
) -> MarketState:
    """오늘 기준 시장 상태 (네트워크 필요). 변동성: SPX 옵션체인 -> VIX 기간구조 -> EWMA 순으로 대체."""
    import warnings

    from mfin.data import CORE_SECTORS, get_prices, get_volume, latest_risk_free, latest_vix_term
    from mfin.portfolio.estimators import chain_knots, estimate_state, vix_knots

    cfg = cfg or EstimatorConfig()
    tickers = list(tickers or CORE_SECTORS)
    start = (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    prices = get_prices(tickers + [bench], start=start).dropna()
    volume = get_volume(tickers + [bench], start=start).reindex(prices.index)
    r = latest_risk_free()

    knots, skew, source = None, None, "ewma"
    if use_chain:
        try:
            from mfin.data.option_chain import fetch_chain, prepare_chain
            from mfin.engine import smile, term_structure

            otm, _ = prepare_chain(fetch_chain("^SPX"), r=r)
            ts = term_structure(smile(otm))
            knots = chain_knots(ts)
            skew = float(ts.iloc[(ts["T"] - 30 / 365).abs().argmin()]["skew"])
            source = "chain"
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"option chain unavailable ({e}); falling back to VIX", stacklevel=2)
    if knots is None:
        try:
            knots = vix_knots(latest_vix_term())
            source = "vix"
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"VIX term structure unavailable ({e}); using EWMA volatility", stacklevel=2)
    return estimate_state(prices, bench, r, knots, cfg, volume=volume, skew=skew, vol_source=source)


def _print(rec: Recommendation, state: MarketState) -> None:
    print(f"\n=== recommendation as of {rec.asof.date()}  (vol source: {state.vol_source}, solver: {rec.solver}) ===")
    tab = pd.DataFrame({"target": rec.weights, "unconstrained": rec.unconstrained.reindex(rec.weights.index)})
    print(tab.round(4).to_string())
    rk = rec.risk
    print(f"\nportfolio vol {rk.vol_ann:.2%}  excess return {rk.excess_ret_ann:.2%}  beta {rk.beta_p:.2f}")
    print(f"1-day VaR95 {rk.var95_1d:.2%}  CVaR95 {rk.cvar95_1d:.2%}")
    if rk.skew is not None:
        print(f"SPX skew sigma(0.9F)-sigma(1.1F): {rk.skew * 100:.2f} vol pt")
    if rk.put_hedge:
        h = rk.put_hedge
        print(f"protective put (K=0.9S, T={h['T']:.3f}y): cost {h['cost']:.3%} of NAV, delta {h['delta']:.3f}")
    if rec.schedule is not None:
        s = rec.schedule
        print(f"\nexecution over {s.times[-1]:g} day(s), {len(s.trades)} buckets, phi={s.phi:.3g}"
              " (shares per bucket, + = sell, - = buy; correlated assets may be cross-hedged)")
        print(pd.DataFrame(s.trades, columns=state.tickers).round(1).to_string())
        print(f"expected impact cost ${s.impact_cost:,.2f}, risk penalty ${s.risk_penalty:,.2f}")
    print(f"\nsolver time {rec.seconds * 1e3:.2f} ms", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--aum", type=float, default=1e6)
    ap.add_argument("--horizon", type=float, default=1 / 12, help="holding period in years")
    ap.add_argument("--no-chain", action="store_true", help="use the VIX term structure instead of SPX options")
    ap.add_argument("--put-hedge", action="store_true")
    args = ap.parse_args()
    state = live_state(use_chain=not args.no_chain)
    rec = recommend(state, args.gamma, horizon=args.horizon, aum=args.aum, put_hedge=args.put_hedge)
    _print(rec, state)


if __name__ == "__main__":
    main()
