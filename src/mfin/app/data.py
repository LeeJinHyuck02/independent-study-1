"""앱 화면에 쓰는 순수 변환 함수 (Streamlit 없이 테스트할 수 있다).

추천 결과(:class:`mfin.portfolio.engine.Recommendation`)와 시장 상태
(:class:`mfin.portfolio.estimators.MarketState`)를 표와 그림용 긴 형식 DataFrame으로 바꾸고,
저장된 백테스트 결과(results/backtest/<run>/)를 읽는다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import mfin.portfolio.engine as engine
from mfin.paths import RESULTS_DIR
from mfin.portfolio.engine import CASH, Recommendation
from mfin.portfolio.estimators import MarketState

# 테스트에서 monkeypatch로 바꿀 수 있게 모듈 속성으로 둔다
BACKTEST_ROOT: Path = RESULTS_DIR / "backtest"

SECTOR_KO = {
    "XLK": "기술",
    "XLF": "금융",
    "XLV": "헬스케어",
    "XLE": "에너지",
    "XLU": "유틸리티",
    "SPY": "S&P 500",
    CASH: "현금",
}

STRATEGY_KO = {
    "spy_buyhold": "SPY 매수보유",
    "equal_weight": "동일가중(1/N)",
    "markowitz": "롱온리 마코위츠",
    "merton_hist": "Merton(과거 변동성)",
    "proposed": "제안 모델(내재변동성)",
}

VOL_SOURCE_KO = {
    "chain": "SPX 옵션체인",
    "vix": "VIX 기간구조",
    "implied": "내재변동성",
    "ewma": "EWMA 실현변동성",
    "ewma_fallback": "EWMA 실현변동성(VIX 결측 대체)",
}

HORIZONS = {"1주": 1 / 52, "1개월": 1 / 12, "3개월": 0.25, "6개월": 0.5, "1년": 1.0}


def sector_label(ticker: str) -> str:
    name = SECTOR_KO.get(ticker)
    return f"{ticker} · {name}" if name and ticker != CASH else (name or ticker)


def strategy_label(name: str) -> str:
    return STRATEGY_KO.get(name, name)


def vol_source_label(source: str) -> str:
    if source in VOL_SOURCE_KO:
        return VOL_SOURCE_KO[source]
    return VOL_SOURCE_KO["ewma"] + "(대체)" if source.startswith("ewma") else source


# ------------------------------------------------------------------------------- 실시간 상태


def fetch_state(use_chain: bool = True) -> tuple[MarketState, list[str]]:
    """오늘 기준 시장 상태와 대체 경고 메시지 (네트워크 필요).

    ``engine.live_state``를 모듈 속성으로 찾아 부르므로 테스트에서 monkeypatch할 수 있다.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = engine.live_state(use_chain=use_chain)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return state, msgs


# ------------------------------------------------------------------------------ 추천 결과 표


def weights_table(rec: Recommendation, state: MarketState) -> pd.DataFrame:
    """자산별 목표 비중·무제약 비중·위험기여도·종가·베타. 마지막 행은 현금."""
    n = len(state.tickers)
    df = pd.DataFrame(
        {
            "ticker": state.tickers,
            "label": [sector_label(t) for t in state.tickers],
            "target": rec.weights.iloc[:n].to_numpy(),
            "unconstrained": rec.unconstrained.reindex(state.tickers).to_numpy(),
            "risk_contrib": rec.risk.risk_contrib,
            "price": state.prices,
            "beta": state.beta,
        }
    )
    cash = pd.DataFrame(
        {"ticker": [CASH], "label": [SECTOR_KO[CASH]], "target": [float(rec.weights[CASH])],
         "unconstrained": [1.0 - float(rec.unconstrained.sum())], "risk_contrib": [0.0],
         "price": [np.nan], "beta": [0.0]}
    )
    return pd.concat([df, cash], ignore_index=True)


def return_decomposition(state: MarketState, horizon: float) -> pd.DataFrame:
    """기대수익(연) 분해: 무위험 r + CAPM 프리미엄 beta*ERP + 모멘텀 view = mu0.

    ``horizon_mean``은 보유기간 평균 기대수익 r + m_bar (view가 반감기에 따라 줄어든 값)이다.
    """
    c = state.coeffs
    r = float(c.r)
    mu_inf, mu0 = np.asarray(c.mu_inf, dtype=float), np.asarray(c.mu0, dtype=float)
    return pd.DataFrame(
        {
            "ticker": state.tickers,
            "label": [sector_label(t) for t in state.tickers],
            "rf": r,
            "capm": mu_inf - r,
            "view": mu0 - mu_inf,
            "mu0": mu0,
            "mu_inf": mu_inf,
            "horizon_mean": r + c.mean_excess_mu(0.0, horizon),
        }
    )


RETURN_PARTS = {"rf": "무위험 r", "capm": "CAPM 프리미엄 β·ERP", "view": "모멘텀 view"}


def return_parts_long(decomp: pd.DataFrame) -> pd.DataFrame:
    """누적 막대용 긴 형식 (label, part, value)."""
    long = decomp.melt(id_vars=["ticker", "label"], value_vars=list(RETURN_PARTS), var_name="key", value_name="value")
    long["part"] = long["key"].map(RETURN_PARTS)
    long["order"] = long["key"].map({k: i for i, k in enumerate(RETURN_PARTS)})
    return long


def term_vol_frame(state: MarketState, t_max: float = 1.0, n_grid: int = 240) -> pd.DataFrame:
    """기간 평균 내재변동성 sigma_bar(0, T) = sqrt(w(T) / T)의 곡선 (t, days, vol).

    엔진이 보유기간 [0, T]에 쓰는 값이 바로 이것이다. 만기 마디에서는 원래 ATM 내재변동성과 같고,
    가까운 만기 사이의 선도분산 잡음(구간별 값)은 적분되면서 평탄해진다.
    """
    knots = np.asarray(state.coeffs.var_knots, dtype=float)
    t = np.union1d(np.linspace(t_max / n_grid, t_max, n_grid), knots[(knots > 0) & (knots <= t_max)])
    vol = np.sqrt(np.asarray(state.coeffs.total_var(t), dtype=float) / t)
    return pd.DataFrame({"t": t, "vol": vol, "days": t * 365.0})


def forward_vol_frame(state: MarketState, t_max: float = 1.0) -> pd.DataFrame:
    """구간상수 선도변동성 sigma_fwd(t)를 계단 그림용 점 (t, vol)으로. 마지막 마디 이후는 일정.

    옵션체인처럼 만기가 촘촘하면 이웃 만기 사이의 선도분산이 크게 흔들린다 (하한으로 잘린 구간 포함).
    """
    knots = np.asarray(state.coeffs.var_knots, dtype=float)
    vol = np.sqrt(np.asarray(state.coeffs.fwd_var, dtype=float))
    starts = knots[:-1]
    keep = starts < t_max
    t = np.append(starts[keep], t_max)
    v = np.append(vol[keep], vol[keep][-1])
    return pd.DataFrame({"t": t, "vol": v, "days": t * 365.0})


def asset_vol_table(state: MarketState, horizon: float) -> pd.DataFrame:
    """보유기간 평균 내재 변동성 sqrt(diag(Sigma_bar))와 EWMA 실현변동성, 배율 s_i."""
    implied = np.sqrt(np.diag(state.coeffs.mean_cov(0.0, horizon)))
    return pd.DataFrame(
        {
            "ticker": state.tickers,
            "label": [sector_label(t) for t in state.tickers],
            "implied": implied,
            "realized": state.sigma_ann,
            "s": state.coeffs.s,
        }
    )


def corr_frame(state: MarketState) -> tuple[pd.DataFrame, float]:
    """Ledoit–Wolf 상관행렬의 긴 형식 (row, col, rho)과 수축강도 delta."""
    R = np.asarray(state.coeffs.R, dtype=float)
    t = state.tickers
    rows = [{"row": t[i], "col": t[j], "rho": float(R[i, j])} for i in range(len(t)) for j in range(len(t))]
    return pd.DataFrame(rows), float(state.diagnostics.get("lw_delta", np.nan))


def constraint_frame(rec: Recommendation, state: MarketState) -> pd.DataFrame:
    """무제약 해 vs 제약 해와 비음수 승수 eta (eta > 0 이면 pi >= 0 제약이 걸린 자산)."""
    n = len(state.tickers)
    return pd.DataFrame(
        {
            "ticker": state.tickers,
            "label": [sector_label(t) for t in state.tickers],
            "unconstrained": rec.unconstrained.reindex(state.tickers).to_numpy(),
            "target": rec.weights.iloc[:n].to_numpy(),
            "eta": np.asarray(rec.qp.eta, dtype=float),
        }
    )


def constraint_long(cf: pd.DataFrame) -> pd.DataFrame:
    long = cf.melt(id_vars=["ticker", "label"], value_vars=["unconstrained", "target"], var_name="key", value_name="weight")
    long["kind"] = long["key"].map({"unconstrained": "무제약 Merton 해", "target": "롱온리 제약 해(추천)"})
    return long


def explain(rec: Recommendation, state: MarketState, gamma: float, tol: float = 1e-6) -> list[str]:
    """추천을 읽는 한국어 해석 문장."""
    out = []
    cash = float(rec.weights[CASH])
    if state.vol_source.startswith("ewma"):
        out.append("내재변동성 데이터를 받지 못해 SPY의 EWMA 실현변동성으로 대체했습니다. "
                   "비중이 평소보다 공격적으로 나올 수 있습니다.")
    if rec.qp.budget_active:
        out.append(f"예산 제약(Σπ ≤ 1)이 걸렸습니다. γ = {gamma:g}에서는 가진 돈을 전부 주식에 넣는 것이 최적이고, "
                   "레버리지가 허용되면 더 사고 싶은 상태입니다.")
    else:
        out.append(f"예산 제약(Σπ ≤ 1)은 걸리지 않았습니다. 현금 {cash:.0%}는 돈이 남아서가 아니라 "
                   f"γ = {gamma:g}의 위험회피 수준에서 최적인 비율입니다. γ를 낮추면 주식 비중이 늘어납니다.")

    n = len(state.tickers)
    target = rec.weights.iloc[:n]
    unc = rec.unconstrained.reindex(state.tickers)
    for t in state.tickers:
        if target[t] > tol:
            continue
        if unc[t] < 0:
            out.append(f"{sector_label(t)}: 무제약 해는 {unc[t]:+.0%}(공매도)지만 공매도를 허용하지 않으므로 0%입니다.")
        else:
            out.append(f"{sector_label(t)}: 무제약 해는 {unc[t]:+.0%}지만, 다른 자산의 공매도가 막히면서 "
                       "상관된 이 자산의 매력도 사라져 0%입니다.")

    sig_imp = float(state.diagnostics.get("sigma_fwd_1m", np.nan))
    sig_rv = float(state.diagnostics.get("bench_vol_ewma", np.nan))
    if np.isfinite(sig_imp) and np.isfinite(sig_rv) and not state.vol_source.startswith("ewma"):
        gap = 100 * (sig_imp - sig_rv)
        if gap > 0:
            out.append(f"1개월 내재 선도변동성 {sig_imp:.1%}가 SPY 실현변동성(EWMA) {sig_rv:.1%}보다 {gap:.1f}%p 높습니다. "
                       "내재변동성에는 분산위험프리미엄이 들어 있어 위험을 크게 보고, 그만큼 현금이 늘어납니다.")
        else:
            out.append(f"1개월 내재 선도변동성 {sig_imp:.1%}가 SPY 실현변동성(EWMA) {sig_rv:.1%}보다 {-gap:.1f}%p 낮습니다. "
                       "시장이 앞으로의 변동성을 최근보다 낮게 보고 있습니다.")

    rc = np.asarray(rec.risk.risk_contrib, dtype=float)
    if rc.size and np.isfinite(rc).all() and rc.max() > 0:
        i = int(np.argmax(rc))
        out.append(f"포트폴리오 위험의 {rc[i]:.0%}가 {sector_label(state.tickers[i])}에서 나옵니다.")
    return out


# --------------------------------------------------------------------------- 백테스트 결과


@dataclass
class BacktestView:
    run: str
    metrics: pd.DataFrame  # 인덱스 = 전략
    nav: pd.DataFrame  # 인덱스 = 날짜, 열 = 전략
    weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def strategies(self) -> list[str]:
        return list(self.metrics.index)

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.nav.index[0], self.nav.index[-1]


def backtest_runs(root: Path | None = None) -> list[str]:
    """root 아래에서 metrics.csv가 있는 실행 결과 폴더 이름."""
    root = Path(root) if root is not None else BACKTEST_ROOT
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "metrics.csv").is_file())


def load_backtest(run_dir: Path) -> BacktestView:
    """:func:`mfin.backtest.rolling.save_results`가 저장한 CSV를 읽는다."""
    run_dir = Path(run_dir)
    if not (run_dir / "metrics.csv").is_file():
        raise FileNotFoundError(f"no backtest results in {run_dir}")
    metrics = pd.read_csv(run_dir / "metrics.csv", index_col="strategy")
    nav = pd.read_csv(run_dir / "nav.csv", index_col="date", parse_dates=["date"])
    weights, diags = {}, {}
    for s in metrics.index:
        w = run_dir / f"weights_{s}.csv"
        if w.is_file():
            df = pd.read_csv(w, index_col=0, parse_dates=[0])
            df.index.name = "date"
            weights[s] = df
        d = run_dir / f"diagnostics_{s}.csv"
        if d.is_file():
            diags[s] = pd.read_csv(d, parse_dates=["date"])
    return BacktestView(run_dir.name, metrics, nav, weights, diags)


METRIC_COLUMNS = {
    "cagr": "CAGR",
    "ann_vol": "연변동성",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "mdd": "최대낙폭",
    "calmar": "Calmar",
    "ann_turnover": "연회전율",
    "cost_drag": "비용손실(연)",
    "final_nav": "최종 NAV",
}
PERCENT_METRICS = ["cagr", "ann_vol", "mdd", "ann_turnover", "cost_drag"]


def metrics_display(view: BacktestView) -> pd.DataFrame:
    """성과표: 비율 지표는 % 단위로 바꾸고 열 이름을 한국어로."""
    df = view.metrics[[c for c in METRIC_COLUMNS if c in view.metrics.columns]].copy()
    for c in PERCENT_METRICS:
        if c in df.columns:
            df[c] = 100.0 * df[c]
    df.index = [strategy_label(s) for s in df.index]
    df.index.name = "전략"
    return df.rename(columns=METRIC_COLUMNS)


def nav_long(nav: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """시작값을 1로 맞춘 누적수익 (date, strategy, label, growth)."""
    g = nav[strategies] / nav[strategies].iloc[0]
    long = g.reset_index().melt(id_vars="date", var_name="strategy", value_name="growth")
    long["label"] = long["strategy"].map(strategy_label)
    return long


def drawdown_long(nav: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    from mfin.backtest.metrics import drawdown

    dd = pd.DataFrame({s: drawdown(nav[s]) for s in strategies}, index=nav.index)
    dd.index.name = "date"
    long = dd.reset_index().melt(id_vars="date", var_name="strategy", value_name="drawdown")
    long["label"] = long["strategy"].map(strategy_label)
    return long


def weights_long(w: pd.DataFrame) -> pd.DataFrame:
    """리밸런싱 시점별 비중 (date, ticker, label, weight). 전 기간 0인 열은 뺀다. 현금은 맨 위."""
    w = w.loc[:, (w.abs() > 1e-9).any(axis=0)]
    cols = [c for c in w.columns if c != CASH] + ([CASH] if CASH in w.columns else [])
    long = w[cols].reset_index().melt(id_vars="date", var_name="ticker", value_name="weight")
    long["label"] = long["ticker"].map(sector_label)
    long["order"] = long["ticker"].map({c: i for i, c in enumerate(cols)})
    return long


def diag_long(view: BacktestView, field_name: str, strategies=("merton_hist", "proposed")) -> pd.DataFrame:
    """전략별 진단값 추이 (date, strategy, label, value)."""
    parts = []
    for s in strategies:
        d = view.diagnostics.get(s)
        if d is None or field_name not in d.columns:
            continue
        parts.append(pd.DataFrame({"date": d["date"], "strategy": s, "label": strategy_label(s), "value": d[field_name]}))
    if not parts:
        return pd.DataFrame(columns=["date", "strategy", "label", "value"])
    return pd.concat(parts, ignore_index=True)


def vol_source_counts(view: BacktestView) -> pd.DataFrame:
    """전략별 변동성 출처 횟수 (대체 횟수 확인용)."""
    rows = {}
    for s, d in view.diagnostics.items():
        if "vol_source" in d.columns:
            rows[strategy_label(s)] = d["vol_source"].map(vol_source_label).value_counts()
    return pd.DataFrame(rows).fillna(0).astype(int).T if rows else pd.DataFrame()
