"""mfin 추천 결과 확인 앱 (Streamlit).

    python -m mfin.app                      (권장: 프로젝트 루트의 .streamlit/config.toml 적용)
    streamlit run src/mfin/app/main.py      (같은 앱, 루트에서 실행)

탭 세 개: 오늘의 추천(목표 비중·위험 지표), 추천 근거(기대수익 분해·변동성·상관·제약),
백테스트 비교(results/backtest/<run>/ 의 저장 결과). 앞의 두 탭은 네트워크로 오늘 시장 상태를
받고, 백테스트 탭은 저장된 CSV만 읽는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from mfin.app import charts, data
from mfin.paths import CONFIG_DIR
from mfin.portfolio.engine import CASH, recommend

st.set_page_config(page_title="mfin 포트폴리오 추천", layout="wide")

VOL_CHOICES = {"SPX 옵션체인": True, "VIX 기간구조": False}


# ----------------------------------------------------------------------------- 캐시·설정


@st.cache_data
def config_defaults() -> tuple[float, float, str]:
    """configs/main.toml의 gamma, 보유기간, 백테스트 결과 폴더 이름. 읽지 못하면 기본값."""
    try:
        from mfin.backtest.rolling import BacktestConfig

        cfg = BacktestConfig.from_toml(CONFIG_DIR / "main.toml")
        return float(cfg.gamma), float(cfg.horizon), cfg.out_dir.name
    except Exception:  # noqa: BLE001
        return 5.0, 1 / 12, "main"


@st.cache_data(ttl="1h", show_spinner="시장 데이터를 불러오는 중… (SPX 옵션체인은 30초 정도 걸립니다)")
def load_state(use_chain: bool):
    state, msgs = data.fetch_state(use_chain)
    return state, msgs, pd.Timestamp.now()


@st.cache_data(show_spinner=False)
def load_backtest(run: str, mtime: float) -> data.BacktestView:
    # mtime을 키에 넣어 백테스트를 다시 돌리면 새 결과를 읽는다
    return data.load_backtest(data.BACKTEST_ROOT / run)


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def show_chart(chart, width="stretch") -> None:
    st.altair_chart(chart, width=width)


dark = st.context.theme.type == "dark"
gamma0, horizon0, run0 = config_defaults()
horizon_labels = list(data.HORIZONS)
horizon_default = min(horizon_labels, key=lambda k: abs(data.HORIZONS[k] - horizon0))

# ---------------------------------------------------------------------------------- 사이드바

with st.sidebar:
    st.header("추천 설정")
    gamma = st.slider("위험회피도 γ", 1.0, 20.0, gamma0, 0.5, key="gamma",
                      help="클수록 위험을 싫어해 현금 비중이 커집니다. γ = 1은 로그 효용입니다.")
    horizon_label = st.select_slider("보유기간 (다음 리밸런싱까지)", options=horizon_labels, value=horizon_default,
                                     key="horizon")
    vol_choice = st.radio("변동성 출처", list(VOL_CHOICES), key="vol_source",
                          help="옵션체인을 받지 못하면 VIX 기간구조, 그것도 없으면 EWMA 실현변동성으로 대체합니다.")
    put_hedge = st.checkbox("보호적 풋 비용 계산 (K = 0.9 S)", key="put_hedge")
    if st.button("시장 데이터 새로고침", width="stretch"):
        load_state.clear()
        st.rerun()
    st.caption("위 설정은 '오늘의 추천'과 '추천 근거' 탭에 적용됩니다. "
               "새로고침은 앱의 메모리 캐시만 비웁니다. 디스크 캐시는 가격 1일, 옵션체인 1시간이 지나야 새로 받습니다.")

horizon = data.HORIZONS[horizon_label]

st.title("mfin 포트폴리오 추천")
st.caption("Black–Scholes 내재변동성 기간구조 + Merton HJB(롱온리 QP)로 S&P 500 섹터 ETF 5개와 현금의 목표 비중을 계산합니다.")

tab_rec, tab_why, tab_bt = st.tabs(["오늘의 추천", "추천 근거", "백테스트 비교"])

state = rec = None
state_error: Exception | None = None
msgs: list[str] = []
try:
    state, msgs, fetched = load_state(VOL_CHOICES[vol_choice])
    rec = recommend(state, gamma, horizon=horizon, put_hedge=put_hedge)
except Exception as e:  # noqa: BLE001
    state_error = e


def state_unavailable() -> None:
    st.error(f"시장 데이터를 불러오지 못했습니다: {state_error}")
    st.caption("네트워크 연결을 확인한 뒤 사이드바의 '시장 데이터 새로고침'을 누르세요. 백테스트 탭은 저장된 결과로 동작합니다.")


# ------------------------------------------------------------------------------- 오늘의 추천

with tab_rec:
    if rec is None:
        state_unavailable()
    else:
        cash = float(rec.weights[CASH])
        rk = rec.risk
        st.subheader(f"{state.asof:%Y-%m-%d} 종가 기준 추천")
        st.caption(f"변동성: {data.vol_source_label(state.vol_source)} · γ = {gamma:g} · 보유기간 {horizon_label} · "
                   f"무위험금리 {state.coeffs.r:.2%} · 데이터 수신 {fetched:%m-%d %H:%M}")
        for m in msgs:
            st.warning(m)
        st.info("\n".join(f"- {s}" for s in data.explain(rec, state, gamma)))

        c = st.columns(4)
        c[0].metric("주식 비중", pct(1.0 - cash), border=True)
        c[1].metric("현금 비중", pct(cash), border=True, help="1 − Σπ. 예산 제약이 걸리지 않으면 위험회피 때문에 남긴 비율입니다.")
        c[2].metric("기대 초과수익 (연)", pct(rk.excess_ret_ann, 2), border=True,
                    help="보유기간 평균 기대수익 m̄ 기준, 무위험금리 초과분")
        c[3].metric("포트폴리오 변동성 (연)", pct(rk.vol_ann, 2), border=True, help="√(πᵀ Σ̄ π), 내재 선도분산 기준")
        c = st.columns(4)
        sharpe = rk.excess_ret_ann / rk.vol_ann if rk.vol_ann > 0 else float("nan")
        c[0].metric("사전 Sharpe", f"{sharpe:.2f}", border=True, help="기대 초과수익 / 변동성. 롱온리 해는 γ와 무관하게 거의 같습니다.")
        c[1].metric("1일 VaR 95%", pct(rk.var95_1d, 2), border=True, help="정규분포 가정, NAV 대비 손실")
        c[2].metric("1일 CVaR 95%", pct(rk.cvar95_1d, 2), border=True, help="VaR를 넘는 손실의 평균")
        c[3].metric("포트폴리오 β", f"{rk.beta_p:.2f}", border=True, help="SPY 대비 베타 (현금 포함)")
        if rk.put_hedge:
            h = rk.put_hedge
            c = st.columns(4)
            c[0].metric("풋 비용 (NAV 대비)", pct(h["cost"], 3), border=True,
                        help="SPX 풋 K = 0.9S, 만기 = 보유기간, 상수 변동성 BS 가격 (skew 미반영이라 실제보다 쌈)")
            c[1].metric("풋 비용 (연환산)", pct(h["cost_ann"], 2), border=True)
            c[2].metric("풋 델타", f"{h['delta']:.3f}", border=True)
            if rk.skew is not None:
                c[3].metric("SPX skew σ(0.9F) − σ(1.1F)", f"{100 * rk.skew:.1f} vol pt", border=True)

        wt = data.weights_table(rec, state)
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**목표 비중**")
            show_chart(charts.weights_bar(wt, dark))
        with right:
            st.markdown("**위험기여도** (포트폴리오 분산 중 각 자산의 몫)")
            show_chart(charts.risk_contrib_bar(wt, dark))

        table = wt.drop(columns=["ticker"]).copy()
        for col in ["target", "unconstrained", "risk_contrib"]:
            table[col] = 100.0 * table[col]
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "label": st.column_config.TextColumn("자산"),
                "target": st.column_config.NumberColumn("목표 비중", format="%.2f%%"),
                "unconstrained": st.column_config.NumberColumn("무제약 비중", format="%.2f%%",
                                                               help="공매도·레버리지를 허용한 Merton 해"),
                "risk_contrib": st.column_config.NumberColumn("위험기여도", format="%.1f%%"),
                "price": st.column_config.NumberColumn("최근 종가", format="dollar"),
                "beta": st.column_config.NumberColumn("β", format="%.2f"),
            },
        )
        st.caption(f"solver: {rec.solver} (정확한 롱온리 QP) · 계산 {rec.seconds * 1e3:.2f} ms · "
                   "주문 수량과 체결 스케줄은 CLI(`python -m mfin.portfolio.engine --aum 1e6`)로 볼 수 있습니다.")

# ------------------------------------------------------------------------------- 추천 근거

with tab_why:
    if rec is None:
        state_unavailable()
    else:
        c = state.coeffs
        st.subheader("① 기대수익: CAPM 사전 + 모멘텀 view")
        dec = data.return_decomposition(state, horizon)
        half_life = np.log(2.0) / c.kappa if c.kappa > 0 else float("inf")
        st.caption(f"μ∞ = r + β·ERP는 장기 균형 기대수익, μ0 − μ∞ = IC·ω·z·√12는 12-1 모멘텀 신호(view)입니다. "
                   f"view는 반감기 {half_life:.1f}년으로 줄어들기 때문에, 검은 눈금(보유기간 {horizon_label} 평균 기대수익)은 "
                   "막대 끝(현재 μ0)보다 μ∞ 쪽에 있습니다.")
        show_chart(charts.return_stack(data.return_parts_long(dec), dec, dark))
        dtab = dec[["label", "rf", "capm", "view", "mu0", "horizon_mean"]].copy()
        for col in ["rf", "capm", "view", "mu0", "horizon_mean"]:
            dtab[col] = 100.0 * dtab[col]
        st.dataframe(dtab, hide_index=True, column_config={
            "label": st.column_config.TextColumn("자산"),
            "rf": st.column_config.NumberColumn("무위험 r", format="%.2f%%"),
            "capm": st.column_config.NumberColumn("CAPM β·ERP", format="%.2f%%"),
            "view": st.column_config.NumberColumn("모멘텀 view", format="%+.2f%%"),
            "mu0": st.column_config.NumberColumn("현재 μ0", format="%.2f%%"),
            "horizon_mean": st.column_config.NumberColumn(f"보유기간 평균", format="%.2f%%"),
        })

        st.subheader("② 변동성: 내재변동성 기간구조")
        ewma = float(state.diagnostics.get("bench_vol_ewma", np.nan))
        sig1m = float(state.diagnostics.get("sigma_fwd_1m", np.nan))
        hz_vol = float(np.sqrt(c.mean_var(0.0, horizon)))
        if sig1m >= ewma:
            vrp = ("지금은 내재가 실현보다 높습니다. 내재변동성에 들어 있는 분산위험프리미엄 때문이며, "
                   "공분산 Σ(t) = σ_fwd²(t)·C 전체가 그만큼 커져 주식 비중이 줄어듭니다.")
        else:
            vrp = ("지금은 내재가 실현보다 낮습니다. 최근의 큰 변동이 EWMA 실현변동성에 아직 남아 있고, "
                   "시장은 앞으로 변동성이 진정될 것으로 본다는 뜻입니다. 평소에는 분산위험프리미엄 때문에 내재가 더 높습니다.")
        st.caption(f"출처: {data.vol_source_label(state.vol_source)}. 곡선은 만기 T까지의 평균 내재변동성 "
                   "σ̄(0,T) = √(w(T)/T)이고, 엔진은 보유기간 끝의 값(굵은 점)을 씁니다. "
                   f"1개월 평균 {sig1m:.1%} vs SPY 실현변동성(EWMA) {ewma:.1%}. {vrp}")
        left, right = st.columns([3, 2])
        with left:
            t_max = max(1.0, horizon)
            show_fwd = st.toggle("구간별 선도변동성도 표시", key="show_fwd",
                                 help="이웃 만기 사이의 선도분산. 만기가 촘촘한 옵션체인에서는 크게 흔들리지만, "
                                      "보유기간 전체로 적분하면 잡음이 상쇄됩니다.")
            fwd = data.forward_vol_frame(state, t_max) if show_fwd else None
            show_chart(charts.vol_term_chart(data.term_vol_frame(state, t_max), ewma, horizon, dark, fwd=fwd, hz_vol=hz_vol))
        with right:
            vt = data.asset_vol_table(state, horizon)
            for col in ["implied", "realized"]:
                vt[col] = 100.0 * vt[col]
            st.dataframe(vt, hide_index=True, column_config={
                "ticker": None,
                "label": st.column_config.TextColumn("자산"),
                "implied": st.column_config.NumberColumn("내재 σ (보유기간)", format="%.1f%%",
                                                         help="σ_fwd × s_i의 보유기간 평균"),
                "realized": st.column_config.NumberColumn("실현 σ (EWMA)", format="%.1f%%"),
                "s": st.column_config.NumberColumn("배율 s_i", format="%.2f", help="자산 EWMA 변동성 / SPY EWMA 변동성"),
            })

        left, right = st.columns(2)
        with left:
            st.subheader("③ 상관: Ledoit–Wolf 수축")
            cf, delta = data.corr_frame(state)
            st.caption(f"수축강도 δ = {delta:.2f}: 표본 상관 {1 - delta:.0%}와 상수상관 타깃 {delta:.0%}를 섞었습니다. "
                       "섹터끼리 상관이 높으면 비슷한 자산 중 하나만 고르는 corner 해가 나오기 쉽습니다.")
            show_chart(charts.corr_heatmap(cf, state.tickers, dark), width="content")
        with right:
            st.subheader("④ 제약의 효과")
            cfr = data.constraint_frame(rec, state)
            st.caption("회색은 공매도·레버리지를 허용한 Merton 해 π = (γΣ̄)⁻¹m̄, 파란색은 π ≥ 0, Σπ ≤ 1 제약 아래의 정확한 해입니다.")
            show_chart(charts.constraint_bars(data.constraint_long(cfr), dark))
            with st.expander("해의 검증 (KKT 조건)"):
                st.markdown(f"- 예산 제약 Σπ ≤ 1: **{'활성' if rec.qp.budget_active else '비활성'}**, 승수 λ = {rec.qp.lam:.3g}\n"
                            "- η_i > 0인 자산은 π_i ≥ 0 제약이 걸려 0이 된 자산입니다.")
                st.dataframe(cfr[["label", "eta"]], hide_index=True, column_config={
                    "label": st.column_config.TextColumn("자산"),
                    "eta": st.column_config.NumberColumn("η (비음수 승수)", format="%.3e"),
                })
                kkt = pd.DataFrame({"조건": list(rec.qp.kkt), "최대 잔차": list(rec.qp.kkt.values())})
                st.dataframe(kkt, hide_index=True, column_config={"최대 잔차": st.column_config.NumberColumn(format="%.1e")})
        st.caption("solver: analytic (scipy NNLS + 예산 승수 탐색, KKT로 검증). 13주차에 PI-DeepONet(solver='neural')이 같은 자리에 들어갑니다.")

# ------------------------------------------------------------------------------ 백테스트

with tab_bt:
    runs = data.backtest_runs()
    if not runs:
        st.info("저장된 백테스트 결과가 없습니다. 프로젝트 루트에서 먼저 실행하세요.")
        st.code("python -m mfin.backtest.rolling --config configs/main.toml", language="bash")
    else:
        run = st.selectbox("실행 결과", runs, index=runs.index(run0) if run0 in runs else 0, key="bt_run",
                           help=f"{data.BACKTEST_ROOT} 아래 폴더")
        mtime = (data.BACKTEST_ROOT / run / "metrics.csv").stat().st_mtime
        view = load_backtest(run, mtime)
        start, end = view.period
        first_w = next(iter(view.weights.values()), None)
        n_reb = len(first_w) if first_w is not None else 0
        if first_w is not None and n_reb:
            start = first_w.index[0]  # NAV는 첫 거래 전날(전액 현금)부터 시작하므로 첫 리밸런싱일을 보여 준다
        st.caption(f"기간 {start:%Y-%m-%d} ~ {end:%Y-%m-%d} · 월초 리밸런싱 {n_reb}회 · 비용 차감 후 성과 · "
                   "Sharpe는 무위험금리(^IRX) 초과수익 기준")

        m = data.metrics_display(view)
        pct_cols = {"CAGR", "연변동성", "최대낙폭", "연회전율", "비용손실(연)"}
        cfg_cols = {col: st.column_config.NumberColumn(col, format="%.2f%%" if col in pct_cols else "%.3f")
                    for col in m.columns}
        cfg_cols["최종 NAV"] = st.column_config.NumberColumn("최종 NAV", format="dollar")
        st.dataframe(m, column_config=cfg_cols)

        picked = st.multiselect("비교할 전략", view.strategies, default=view.strategies, format_func=data.strategy_label,
                                key="bt_strategies")
        if picked:
            log = st.toggle("로그 축", value=True, key="bt_log")
            st.markdown("**누적 수익** (시작 = 1)")
            show_chart(charts.nav_chart(data.nav_long(view.nav, picked), log, dark))
            st.markdown("**낙폭** (직전 고점 대비)")
            show_chart(charts.drawdown_chart(data.drawdown_long(view.nav, picked), dark))

        st.subheader("전략별 비중 추이")
        options = [s for s in view.strategies if s in view.weights]
        default = options.index("proposed") if "proposed" in options else 0
        s = st.selectbox("전략", options, index=default, format_func=data.strategy_label, key="bt_weights")
        w = view.weights[s]
        show_chart(charts.weights_area(data.weights_long(w), dark, universe=list(w.columns)))

        if {"proposed", "merton_hist"} <= set(view.diagnostics):
            st.subheader("제안 모델 vs Merton(과거 변동성)")
            st.caption("두 전략은 기대수익 모형과 QP가 같고 변동성 입력만 다릅니다(VIX 내재 vs EWMA 실현). "
                       "VIX에 들어 있는 분산위험프리미엄 때문에 제안 모델이 위험을 더 크게 보고 현금을 더 듭니다. "
                       "그래서 낙폭은 작지만 Sharpe도 낮아졌습니다.")
            left, right = st.columns(2)
            with left:
                st.markdown("**현금 비중**")
                show_chart(charts.diag_chart(data.diag_long(view, "cash"), "현금 비중", ".0%", dark))
            with right:
                st.markdown("**1개월 선도변동성 σ_fwd**")
                show_chart(charts.diag_chart(data.diag_long(view, "sigma_fwd_1m"), "연율 변동성", ".0%", dark))
            counts = data.vol_source_counts(view)
            if not counts.empty:
                st.markdown("**변동성 출처별 리밸런싱 횟수** (대체 발생 확인)")
                st.dataframe(counts)
