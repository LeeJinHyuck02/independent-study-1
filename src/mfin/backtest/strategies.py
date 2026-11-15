"""백테스트 비교 전략. 모든 전략은 리밸런싱 시점에 t-1까지의 데이터만 받는다.

비중 벡터는 자산 순서 [유니버스..., 벤치마크]를 따르고, 현금은 1 - sum(비중)이다.

    spy_buyhold    벤치마크 매수 후 보유
    equal_weight   유니버스 1/N
    markowitz      표본 평균·공분산의 long-only 평균-분산 (같은 QP)
    merton_hist    Merton 엔진 + 과거(EWMA) 변동성
    proposed       Merton 엔진 + 내재(VIX 기간구조) 선도변동성  (= 추천 엔진 C)

merton_hist와 proposed는 기대수익 모형이 같고 변동성 입력만 다르다(ablation 쌍).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from mfin.engine.merton import solve_mv_qp
from mfin.portfolio.engine import recommend
from mfin.portfolio.estimators import EstimatorConfig, estimate_state, simple_returns, vix_knots
from mfin.portfolio.execution import ExecutionConfig


@dataclass
class RebalanceContext:
    date: pd.Timestamp  # 체결일 t (이 날 종가에 체결)
    prices: pd.DataFrame  # t-1까지의 가격 (열 = 유니버스 + 벤치마크)
    vix_row: pd.Series | None  # t-1의 VIX 기간구조
    r: float  # t-1의 무위험금리 (연속복리)
    volume: pd.DataFrame | None  # t-1까지의 거래량
    current_weights: np.ndarray  # 체결 직전 비중 (자산 순서)
    nav: float


class Strategy(Protocol):
    name: str
    diagnostics: list[dict]

    def weights(self, ctx: RebalanceContext) -> np.ndarray: ...


@dataclass
class BenchmarkBuyHold:
    n_assets: int
    name: str = "spy_buyhold"
    diagnostics: list[dict] = field(default_factory=list)

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        if ctx.current_weights.sum() > 0:
            return ctx.current_weights.copy()  # 이미 보유 중이면 거래하지 않는다
        w = np.zeros(self.n_assets)
        w[-1] = 1.0
        return w


@dataclass
class EqualWeight:
    n_universe: int
    name: str = "equal_weight"
    diagnostics: list[dict] = field(default_factory=list)

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        return np.append(np.full(self.n_universe, 1.0 / self.n_universe), 0.0)


@dataclass
class LongOnlyMarkowitz:
    gamma: float
    window: int = 252
    periods_per_year: int = 252
    name: str = "markowitz"
    diagnostics: list[dict] = field(default_factory=list)

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        P = ctx.prices.to_numpy(dtype=float)[-(self.window + 1):, :-1]
        R = simple_returns(P)
        m = R.mean(axis=0) * self.periods_per_year - ctx.r
        S = np.cov(R, rowvar=False) * self.periods_per_year
        res = solve_mv_qp(m, S, self.gamma)
        self.diagnostics.append({"date": ctx.date, "cash": 1 - res.pi.sum(), "kkt_max": max(res.kkt.values())})
        return np.append(res.pi, 0.0)


@dataclass
class MertonStrategy:
    name: str
    gamma: float
    bench: str
    horizon: float = 1 / 12
    vol_source: str = "vix"  # "vix" | "ewma"
    est_cfg: EstimatorConfig = field(default_factory=EstimatorConfig)
    solver: str = "analytic"
    exec_cfg: ExecutionConfig | None = None  # 주면 체결 스케줄의 충격비용을 진단값으로 기록
    diagnostics: list[dict] = field(default_factory=list)

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        knots, used = None, "ewma"
        if self.vol_source == "vix":
            try:
                knots = vix_knots(ctx.vix_row) if ctx.vix_row is not None else None
                used = "vix" if knots is not None else "ewma_fallback"
            except ValueError:
                used = "ewma_fallback"
        state = estimate_state(ctx.prices, self.bench, ctx.r, knots, self.est_cfg,
                               volume=ctx.volume, vol_source=used)
        aum = ctx.nav if self.exec_cfg is not None else None
        rec = recommend(state, self.gamma, solver=self.solver, horizon=self.horizon,
                        current_weights=ctx.current_weights[:-1], aum=aum, exec_cfg=self.exec_cfg)
        pi = rec.qp.pi
        sch = rec.schedule
        self.diagnostics.append(
            {
                "date": ctx.date,
                "vol_source": state.vol_source,
                "sigma_fwd_1m": state.diagnostics["sigma_fwd_1m"],
                "lw_delta": state.diagnostics["lw_delta"],
                "cash": 1 - pi.sum(),
                "budget_active": rec.qp.budget_active,
                "kkt_max": max(rec.qp.kkt.values()),
                "impact_cost": sch.impact_cost if sch is not None else np.nan,
                "risk_penalty": sch.risk_penalty if sch is not None else np.nan,
                "solver_ms": rec.seconds * 1e3,
            }
        )
        return np.append(pi, 0.0)


def build_strategy(name: str, cfg) -> Strategy:
    """설정(:class:`mfin.backtest.rolling.BacktestConfig`)에서 이름으로 전략을 만든다."""
    n = len(cfg.tickers)
    exec_cfg = cfg.exec_cfg if cfg.execution_enabled else None
    factories = {
        "spy_buyhold": lambda: BenchmarkBuyHold(n + 1),
        "equal_weight": lambda: EqualWeight(n),
        "markowitz": lambda: LongOnlyMarkowitz(cfg.gamma, cfg.window),
        "merton_hist": lambda: MertonStrategy("merton_hist", cfg.gamma, cfg.benchmark, cfg.horizon, "ewma",
                                              cfg.est_cfg, exec_cfg=exec_cfg),
        "proposed": lambda: MertonStrategy("proposed", cfg.gamma, cfg.benchmark, cfg.horizon, "vix",
                                           cfg.est_cfg, exec_cfg=exec_cfg),
    }
    if name not in factories:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(factories)}")
    return factories[name]()
