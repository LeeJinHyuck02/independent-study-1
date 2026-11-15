"""모듈 C: 위험 리포트, 체결 스케줄, 추천 엔진."""

import numpy as np
import pandas as pd
import pytest

from mfin.engine.bs_analytic import bs_price
from mfin.engine.merton import MertonCoeffs
from mfin.portfolio.engine import CASH, recommend
from mfin.portfolio.estimators import estimate_state
from mfin.portfolio.execution import (
    impact_matrix,
    order_from_weights,
    plan_execution,
    price_cov_daily,
    twap_schedule,
)
from mfin.portfolio.risk import parametric_var_cvar, protective_put, risk_report

from test_estimators import synthetic_panel


def make_state(with_volume=True):
    panel = synthetic_panel()
    vol = None
    if with_volume:
        vol = pd.DataFrame(1e6, index=panel.index, columns=panel.columns)
    knots = (np.array([0.0, 30 / 365, 0.25, 1.0]), np.array([0.03, 0.035, 0.04]))
    return estimate_state(panel, "SPY", 0.03, knots, volume=vol, vol_source="vix")


# ----------------------------------------------------------------------------- 위험


def test_var_cvar_normal():
    var, cvar = parametric_var_cvar(0.0, 0.01)
    assert var == pytest.approx(0.016448536, abs=1e-9)
    assert cvar == pytest.approx(0.020627128, abs=1e-9)


def test_risk_contributions_sum_to_one():
    st = make_state()
    pi = np.array([0.2, 0.1, 0.15, 0.05, 0.1])
    rep = risk_report(pi, st.coeffs, 1 / 12, st.beta)
    assert rep.risk_contrib.sum() == pytest.approx(1.0, abs=1e-12)
    S = st.coeffs.mean_cov(0, 1 / 12)
    assert rep.vol_ann == pytest.approx(np.sqrt(pi @ S @ pi), rel=1e-14)
    assert rep.beta_p == pytest.approx(pi @ st.beta)


def test_protective_put_matches_black_scholes():
    h = protective_put(0.6, 0.03, 0.2, 0.25)
    assert h["cost"] == pytest.approx(0.6 * bs_price(1.0, 0.9, 0.25, 0.03, 0.2, kind="put"), rel=1e-14)
    assert h["delta"] < 0


# ------------------------------------------------------------------------------ 체결


def test_order_sign_convention():
    q0 = order_from_weights([0.2, 0.3], [0.5, 0.0], 1e6, [100.0, 50.0])
    assert q0 == pytest.approx([3000.0, -6000.0], abs=1e-12)  # XLK 3000주 매도, 둘째 자산 6000주 매수


def test_schedule_accounting():
    S = np.array([100.0, 50.0, 80.0])
    cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.0225]])
    Lam = impact_matrix(S, np.sqrt(np.diag(cov) / 252), [1e6, 2e6, 5e5])
    Sp = price_cov_daily(S, cov)
    q0 = np.array([3000.0, 6000.0, 1500.0])
    sch = plan_execution(q0, Lam, Sp, phi=5 / 2e6)
    assert sch.trades.sum(axis=0) == pytest.approx(q0, rel=1e-12)
    assert np.max(np.abs(sch.holdings[-1])) == 0.0
    assert np.all(np.diff(sch.holdings, axis=0) <= 1e-9)  # 상관이 없고 모두 매도이면 단조 감소
    assert sch.impact_cost + sch.risk_penalty == pytest.approx(sch.expected_cost, rel=1e-6)
    # 같은 비용 모형에서 TWAP은 최적 스케줄보다 비싸다
    tw = twap_schedule(q0, Lam, Sp, phi=5 / 2e6)
    assert tw.expected_cost > sch.expected_cost


def test_impact_matrix_requires_positive_adv():
    with pytest.raises(ValueError):
        impact_matrix([100.0], [0.01], [0.0])


# ------------------------------------------------------------------------------ 추천


def test_recommend_weights_and_schedule():
    st = make_state()
    rec = recommend(st, 5.0, aum=1e6)
    w = rec.weights
    assert list(w.index) == st.tickers + [CASH]
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert max(rec.qp.kkt.values()) <= 1e-10
    assert rec.schedule is not None and rec.schedule.phi == pytest.approx(5.0 / (2 * 1e6))
    # 전액 현금에서 시작하므로 모든 주문은 매수(음수) 또는 0
    assert (rec.orders <= 0).all()


def test_recommend_without_volume_skips_schedule():
    rec = recommend(make_state(with_volume=False), 5.0, aum=1e6)
    assert rec.orders is not None and rec.schedule is None


def test_recommend_solver_dispatch():
    st = make_state()
    with pytest.raises(NotImplementedError):
        recommend(st, 5.0, solver="neural")
    with pytest.raises(ValueError):
        recommend(st, 5.0, solver="foo")


def test_recommend_put_hedge_report():
    rec = recommend(make_state(), 5.0, put_hedge=True)
    assert rec.risk.put_hedge is not None and rec.risk.put_hedge["cost"] >= 0
