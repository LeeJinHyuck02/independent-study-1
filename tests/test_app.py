"""결과 확인 앱: 변환 함수, 그림 스펙, Streamlit 화면 end-to-end (합성 데이터, 네트워크 없음)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")

import mfin.portfolio.engine as eng  # noqa: E402
from mfin.app import charts  # noqa: E402
from mfin.app import data as appdata  # noqa: E402
from mfin.backtest.rolling import metrics_table, run_backtest, save_results  # noqa: E402
from mfin.experiments import _plot  # noqa: E402
from mfin.portfolio.engine import CASH, recommend  # noqa: E402

from test_backtest import make_cfg, synthetic_data  # noqa: E402
from test_portfolio import make_state  # noqa: E402

APP = Path(appdata.__file__).with_name("main.py")


@pytest.fixture(scope="module")
def state():
    return make_state()


@pytest.fixture(scope="module")
def bt_root(tmp_path_factory):
    """합성 백테스트를 실제 저장 함수로 저장한 결과 폴더 (root/syn/)."""
    root = tmp_path_factory.mktemp("backtest")
    d, cfg = synthetic_data(), make_cfg()
    results = run_backtest(cfg, d)
    save_results(results, metrics_table(results, d), root / "syn")
    return root


# ------------------------------------------------------------------------------ 변환 함수


def test_return_decomposition_sums_to_mu0(state):
    dec = appdata.return_decomposition(state, 1 / 12)
    total = dec["rf"] + dec["capm"] + dec["view"]
    assert total.to_numpy() == pytest.approx(state.coeffs.mu0, abs=1e-12)
    # 보유기간 평균은 view가 줄어든 만큼 mu0와 mu_inf 사이에 있다
    lo = np.minimum(dec["mu0"], dec["mu_inf"]) - 1e-12
    hi = np.maximum(dec["mu0"], dec["mu_inf"]) + 1e-12
    assert np.all((dec["horizon_mean"] >= lo) & (dec["horizon_mean"] <= hi))
    long = appdata.return_parts_long(dec)
    assert len(long) == 3 * len(state.tickers)


def test_forward_vol_frame_matches_knots(state):
    fv = appdata.forward_vol_frame(state, 1.0)
    assert fv["t"].to_numpy() == pytest.approx([0.0, 30 / 365, 0.25, 1.0], abs=1e-15)
    assert fv["vol"].to_numpy() == pytest.approx(np.sqrt([0.03, 0.035, 0.04, 0.04]), rel=1e-14)
    short = appdata.forward_vol_frame(state, 0.1)  # 0.25 이후 마디는 잘린다
    assert short["t"].iloc[-1] == 0.1 and len(short) == 3


def test_term_vol_frame_is_average_of_forward_variance(state):
    tv = appdata.term_vol_frame(state, 1.0)
    c = state.coeffs
    # 만기 T에서 sigma_bar^2 * T = 총분산 = 구간별 선도분산의 적분
    for T in (30 / 365, 0.25, 1.0):
        row = tv[np.isclose(tv["t"], T)]
        assert len(row) == 1
        assert row["vol"].iloc[0] ** 2 * T == pytest.approx(c.total_var(T), rel=1e-12)
    assert tv["vol"].iloc[-1] ** 2 == pytest.approx(c.mean_var(0.0, 1.0), rel=1e-12)
    assert tv["t"].is_monotonic_increasing and tv["t"].iloc[0] > 0


def test_weights_table_matches_recommendation(state):
    rec = recommend(state, 5.0)
    wt = appdata.weights_table(rec, state)
    assert list(wt["ticker"]) == state.tickers + [CASH]
    assert wt["target"].sum() == pytest.approx(1.0, abs=1e-12)
    assert wt["target"].to_numpy() == pytest.approx(rec.weights.to_numpy(), abs=0)
    assert wt["risk_contrib"].sum() == pytest.approx(1.0, abs=1e-12)


def test_explain_messages(state):
    rec = recommend(state, 5.0)
    assert not rec.qp.budget_active
    msgs = appdata.explain(rec, state, 5.0)
    text = "\n".join(msgs)
    assert "예산 제약(Σπ ≤ 1)은 걸리지 않았습니다" in text
    zeros = [t for t in state.tickers if rec.weights[t] <= 1e-6]
    assert zeros, "fixture should have at least one zero-weight asset"
    for t in zeros:
        assert any(m.startswith(f"{t}:") for m in msgs)
    neg = [t for t in zeros if rec.unconstrained[t] < 0]
    assert neg and any("공매도" in m for m in msgs if m.startswith(f"{neg[0]}:"))
    assert "내재 선도변동성" in text

    lev = recommend(state, 0.2)
    assert lev.qp.budget_active
    assert "예산 제약(Σπ ≤ 1)이 걸렸습니다" in "\n".join(appdata.explain(lev, state, 0.2))

    fb = dataclasses.replace(state, vol_source="ewma_fallback")
    fb_msgs = appdata.explain(recommend(fb, 5.0), fb, 5.0)
    assert "EWMA" in fb_msgs[0] and not any("내재 선도변동성" in m for m in fb_msgs)


def test_labels_cover_backtest_strategies():
    assert set(make_cfg().strategies) <= set(appdata.STRATEGY_KO)
    assert appdata.vol_source_label("ewma_fallback") == appdata.VOL_SOURCE_KO["ewma_fallback"]
    assert appdata.vol_source_label("ewma_x").startswith("EWMA")
    assert appdata.sector_label("XLK") == "XLK · 기술" and appdata.sector_label(CASH) == "현금"


def test_fetch_state_collects_fallback_warnings(monkeypatch, state):
    import warnings

    def fake(use_chain=True, **kw):
        warnings.warn("option chain unavailable (test); falling back to VIX", stacklevel=2)
        return state

    monkeypatch.setattr(eng, "live_state", fake)
    s, msgs = appdata.fetch_state(True)
    assert s is state and msgs == ["option chain unavailable (test); falling back to VIX"]


# --------------------------------------------------------------------------- 백테스트 로더


def test_backtest_roundtrip(bt_root):
    assert appdata.backtest_runs(bt_root) == ["syn"]
    assert appdata.backtest_runs(bt_root / "missing") == []
    view = appdata.load_backtest(bt_root / "syn")
    strategies = make_cfg().strategies
    assert set(view.strategies) == set(strategies)
    assert list(view.nav.columns) == view.strategies
    assert set(view.weights) == set(strategies)
    assert {"proposed", "merton_hist"} <= set(view.diagnostics)

    m = appdata.metrics_display(view)
    assert "CAGR" in m.columns and m.index.name == "전략"
    assert m.loc[appdata.strategy_label("proposed"), "CAGR"] == pytest.approx(100 * view.metrics.loc["proposed", "cagr"])

    nl = appdata.nav_long(view.nav, view.strategies)
    first = nl[nl["date"] == view.nav.index[0]]
    assert first["growth"].to_numpy() == pytest.approx(1.0, abs=1e-15)
    assert (appdata.drawdown_long(view.nav, view.strategies)["drawdown"] <= 0).all()

    wl = appdata.weights_long(view.weights["spy_buyhold"])
    assert set(wl["ticker"]) == {"SPY"}  # 전 기간 0인 열(섹터, 현금)은 빠진다
    wl = appdata.weights_long(view.weights["proposed"])
    assert wl.sort_values("order")["ticker"].iloc[-1] == CASH

    dg = appdata.diag_long(view, "cash")
    assert set(dg["strategy"]) == {"proposed", "merton_hist"}
    counts = appdata.vol_source_counts(view)
    assert counts.loc[appdata.strategy_label("proposed")].sum() == len(view.diagnostics["proposed"])

    with pytest.raises(FileNotFoundError):
        appdata.load_backtest(bt_root / "missing")


# ----------------------------------------------------------------------------------- 그림


def test_palettes_match_plot_module():
    assert list(charts.LIGHT.series) == _plot.SERIES
    assert charts.LIGHT.cash == _plot.CASH


def test_colors_follow_entity():
    universe = ["XLK", "XLF", "XLV", "XLE", "XLU", "SPY", CASH]
    s = charts.ticker_scale(["XLV", CASH], False, universe=universe)
    assert s.range == [charts.LIGHT.series[2], charts.LIGHT.cash]
    st_ = charts.strategy_scale(True, ["proposed"])
    assert st_.domain == [appdata.strategy_label("proposed")] and st_.range == [charts.DARK.series[4]]


@pytest.mark.parametrize("dark", [False, True])
def test_chart_specs_compile(state, bt_root, dark):
    rec = recommend(state, 5.0)
    wt = appdata.weights_table(rec, state)
    dec = appdata.return_decomposition(state, 1 / 12)
    cf, _ = appdata.corr_frame(state)
    view = appdata.load_backtest(bt_root / "syn")
    built = [
        charts.weights_bar(wt, dark),
        charts.risk_contrib_bar(wt, dark),
        charts.return_stack(appdata.return_parts_long(dec), dec, dark),
        charts.vol_term_chart(appdata.term_vol_frame(state), 0.15, 1 / 12, dark),
        charts.vol_term_chart(appdata.term_vol_frame(state), 0.15, 1 / 12, dark,
                              fwd=appdata.forward_vol_frame(state), hz_vol=0.18),
        charts.corr_heatmap(cf, state.tickers, dark),
        charts.constraint_bars(appdata.constraint_long(appdata.constraint_frame(rec, state)), dark),
        charts.nav_chart(appdata.nav_long(view.nav, view.strategies), True, dark),
        charts.drawdown_chart(appdata.drawdown_long(view.nav, view.strategies), dark),
        charts.weights_area(appdata.weights_long(view.weights["proposed"]), dark),
        charts.diag_chart(appdata.diag_long(view, "sigma_fwd_1m"), "vol", ".0%", dark),
    ]
    for ch in built:
        spec = ch.to_dict()
        assert "$schema" in spec


# ------------------------------------------------------------------------- 앱 end-to-end


def _app(monkeypatch, root, live_state):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(eng, "live_state", live_state)
    monkeypatch.setattr(appdata, "BACKTEST_ROOT", root)
    st.cache_data.clear()
    return AppTest.from_file(str(APP), default_timeout=60)


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _pct(s):
    return float(s.rstrip("%")) / 100.0


def test_app_end_to_end(monkeypatch, state, bt_root):
    at = _app(monkeypatch, bt_root, lambda use_chain=True, **kw: state)
    at.run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["오늘의 추천", "추천 근거", "백테스트 비교"]
    cash5 = _pct(_metric(at, "현금 비중"))
    assert cash5 == pytest.approx(float(recommend(state, 5.0).weights[CASH]), abs=6e-4)
    assert any("예산 제약" in i.value for i in at.info)
    assert len(at.dataframe) >= 5  # 추천 표, 근거 표들, 성과표

    at.slider(key="gamma").set_value(2.0).run()
    cash2 = _pct(_metric(at, "현금 비중"))
    at.slider(key="gamma").set_value(10.0).run()
    cash10 = _pct(_metric(at, "현금 비중"))
    assert cash2 < cash5 < cash10

    at.checkbox(key="put_hedge").check().run()
    assert not at.exception
    assert any(m.label == "풋 비용 (NAV 대비)" for m in at.metric)

    at.toggle(key="show_fwd").set_value(True).run()
    at.select_slider(key="horizon").set_value("1년").run()
    assert not at.exception, at.exception


def test_app_without_network_or_results(monkeypatch, tmp_path):
    def offline(use_chain=True, **kw):
        raise ConnectionError("offline")

    at = _app(monkeypatch, tmp_path, offline)
    at.run()
    assert not at.exception, at.exception
    assert any("시장 데이터를 불러오지 못했습니다" in e.value for e in at.error)
    assert any("저장된 백테스트 결과가 없습니다" in i.value for i in at.info)
