"""백테스트 루프와 성과 지표 (합성 데이터, 네트워크 없음)."""

import numpy as np
import pandas as pd
import pytest

from mfin.backtest.metrics import cagr, drawdown, performance_table
from mfin.backtest.rolling import (
    BacktestConfig,
    BacktestData,
    ProportionalCost,
    rebalance_dates,
    run_backtest,
    run_strategy,
)
from mfin.backtest.strategies import BenchmarkBuyHold, EqualWeight, build_strategy
from mfin.paths import CONFIG_DIR

from test_estimators import TICKERS, synthetic_panel

VIX_COLS = [9 / 365, 30 / 365, 0.25, 0.5, 1.0]


def synthetic_data(n_rows=600, rf=0.02):
    prices = synthetic_panel(n_rows)
    idx = prices.index
    vix = pd.DataFrame([[0.16, 0.17, 0.18, 0.19, 0.20]] * n_rows, index=idx, columns=VIX_COLS)
    vol = pd.DataFrame(1e6, index=idx, columns=prices.columns)
    return BacktestData(prices, pd.Series(rf, index=idx), vix, vol)


def make_cfg(**kw):
    base = dict(tickers=TICKERS, benchmark="SPY", start="2021-03-01", end="2021-12-31", cost_bps=5.0)
    base.update(kw)
    return BacktestConfig(**base)


# --------------------------------------------------------------------------------- 지표


def test_metrics_constant_return():
    g = 0.0004
    nav = pd.Series(100 * (1 + g) ** np.arange(505), index=pd.bdate_range("2021-01-01", periods=505))
    zero = pd.Series(0.0, index=nav.index)
    m = performance_table(nav, nav, zero, pd.Series([0.0]), pd.Series([0.0]))
    assert m["cagr"] == pytest.approx((1 + g) ** 252 - 1, rel=1e-12)
    assert m["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert m["mdd"] == 0.0 and np.isnan(m["calmar"])


def test_drawdown_and_calmar():
    nav = pd.Series([100.0, 120.0, 90.0, 108.0], index=pd.bdate_range("2021-01-01", periods=4))
    assert drawdown(nav).min() == pytest.approx(-0.25, abs=1e-15)
    zero = pd.Series(0.0, index=nav.index)
    m = performance_table(nav, nav, zero, pd.Series([0.0]), pd.Series([0.0]))
    assert m["calmar"] == pytest.approx(cagr(nav) / 0.25, rel=1e-12)


# ---------------------------------------------------------------------------- 시뮬레이션


def test_rebalance_dates_skip_holidays():
    idx = pd.bdate_range("2021-01-01", "2021-04-30").drop(pd.Timestamp("2021-02-01"))
    d = rebalance_dates(idx, "2021-01-01", "2021-04-30")
    assert list(d.strftime("%Y-%m-%d")) == ["2021-01-01", "2021-02-02", "2021-03-01", "2021-04-01"]


def test_buy_and_hold_nav_equals_price_ratio_minus_cost():
    data = synthetic_data(rf=0.0)
    cfg = make_cfg()
    res = run_strategy(BenchmarkBuyHold(6), data, cfg, ProportionalCost(5.0))
    p = data.prices["SPY"]
    p0 = p.loc[res.weights.index[0]]  # 첫 리밸런싱일 종가에 매수
    ratio = res.nav_net.iloc[-1] / cfg.initial_nav
    assert ratio == pytest.approx(p.loc[res.nav_net.index[-1]] / p0 * (1 - 5e-4), rel=1e-12)
    assert res.nav_net.iloc[0] == cfg.initial_nav  # 시계열은 첫 거래 전날 전액 현금에서 시작
    assert res.costs.iloc[1:].abs().max() == 0.0  # 첫 매수 뒤에는 거래가 없다


def test_equal_weight_matches_hand_loop_without_costs():
    data = synthetic_data(rf=0.0)
    cfg = make_cfg(start="2021-03-01", end="2021-04-30")
    res = run_strategy(EqualWeight(5), data, cfg, ProportionalCost(0.0))
    px = data.prices[TICKERS].loc["2021-03-01":"2021-04-30"]
    nav, h = cfg.initial_nav, None
    reb = set(rebalance_dates(px.index, cfg.start, cfg.end))
    out = []
    for i, d in enumerate(px.index):
        if i > 0:
            h = h * px.iloc[i].to_numpy() / px.iloc[i - 1].to_numpy()
            nav = h.sum()
        if d in reb:
            h = np.full(5, nav / 5)
        out.append(nav)
    assert res.nav_net.iloc[1:].to_numpy() == pytest.approx(np.array(out), rel=1e-12)


def test_cost_accounting():
    data = synthetic_data()
    cfg = make_cfg()
    res = run_strategy(EqualWeight(5), data, cfg, ProportionalCost(5.0))
    # 비용 = 5bp x 거래액 = 5bp x 회전율 x 거래 직전 NAV
    nav_pre = res.nav_net.reindex(res.costs.index) + res.costs
    assert res.costs.to_numpy() == pytest.approx(5e-4 * res.turnover.to_numpy() * nav_pre.to_numpy(), rel=1e-12)
    assert res.turnover.iloc[0] == pytest.approx(1.0, rel=1e-12)  # 전액 현금 -> 1/N 매수
    # 비용이 없는 장부와의 차이가 곧 비용 손실이다
    assert res.nav_gross.iloc[-1] > res.nav_net.iloc[-1]


def test_no_look_ahead_in_backtest_weights():
    data = synthetic_data()
    cfg = make_cfg()
    a = run_strategy(build_strategy("proposed", cfg), data, cfg, ProportionalCost(5.0))
    d = a.weights.index[3]
    i = data.prices.index.get_loc(d)
    shocked = synthetic_data()
    shocked.prices.iloc[i:] *= 3.0
    shocked.vix.iloc[i:] = 0.8
    b = run_strategy(build_strategy("proposed", cfg), shocked, cfg, ProportionalCost(5.0))
    assert np.array_equal(a.weights.loc[:d].to_numpy(), b.weights.loc[:d].to_numpy())


def test_all_strategies_run_and_respect_constraints():
    data = synthetic_data()
    cfg = make_cfg()
    results = run_backtest(cfg, data)
    assert set(results) == set(cfg.strategies)
    for res in results.values():
        w = res.weights.to_numpy()
        assert np.all(w >= -1e-12) and np.allclose(w.sum(axis=1), 1.0)
        assert np.all(np.isfinite(res.nav_net))
    diag = results["proposed"].diagnostics
    assert (diag["vol_source"] == "vix").all()
    assert (results["merton_hist"].diagnostics["vol_source"] == "ewma").all()
    assert diag["kkt_max"].max() <= 1e-10


def test_vix_gap_falls_back_to_ewma():
    data = synthetic_data()
    data.vix.loc[:, 30 / 365] = np.nan
    cfg = make_cfg(strategies=["proposed"])
    res = run_backtest(cfg, data)["proposed"]
    assert (res.diagnostics["vol_source"] == "ewma_fallback").all()


def test_config_from_toml():
    cfg = BacktestConfig.from_toml(CONFIG_DIR / "main.toml")
    assert cfg.tickers == ["XLK", "XLF", "XLV", "XLE", "XLU"]
    assert cfg.start == "2021-01-01" and cfg.end == "2025-12-31"
    assert cfg.gamma == 5.0 and cfg.horizon == pytest.approx(1 / 12)
    assert "proposed" in cfg.strategies
    assert cfg.out_dir.name == "main"
    assert cfg.exec_cfg.n_buckets == 13 and cfg.est_cfg.ewma_span == 63
