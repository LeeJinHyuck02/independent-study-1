"""Rolling-window 백테스트 (모듈 D 1차).

    python -m mfin.backtest.rolling --config configs/main.toml
    python -m mfin.backtest.rolling --config configs/main.toml --strategies equal_weight,proposed

결과: results/backtest/<설정 이름>/ 에 metrics.csv, nav.csv, weights_<전략>.csv,
diagnostics_<전략>.csv, nav.png, drawdown.png, weights_proposed.png.

시뮬레이션 규약 (미래참조 방지)
- 리밸런싱일 t(매월 첫 거래일)의 결정에는 t-1까지의 가격·VIX·금리만 쓴다(행 위치 기준 iloc[:t]).
- 체결은 t의 종가로 가정한다. 비용(거래액 x cost_bps)을 먼저 NAV에서 빼고 남은 NAV로 비중을 맞춘다.
- 그 사이 보유액은 매일 (1 + 수익률)로 불어나고(비중 표류), 현금은 전날 무위험금리로 붙는다.
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from mfin.backtest.metrics import drawdown, performance_table
from mfin.backtest.strategies import RebalanceContext, Strategy, build_strategy
from mfin.paths import CONFIG_DIR, ROOT
from mfin.portfolio.estimators import EstimatorConfig
from mfin.portfolio.execution import ExecutionConfig

CASH = "CASH"


@dataclass
class BacktestConfig:
    tickers: list[str]
    benchmark: str = "SPY"
    data_start: str = "2005-01-01"
    data_end: str = "2026-01-01"  # yfinance의 end는 그 날을 포함하지 않는다
    start: str = "2021-01-01"
    end: str = "2025-12-31"
    rebalance: str = "monthly"
    window: int = 252
    cost_bps: float = 5.0
    initial_nav: float = 1_000_000.0
    gamma: float = 5.0
    horizon: float = 1 / 12
    strategies: list[str] = field(default_factory=lambda: ["spy_buyhold", "equal_weight", "markowitz",
                                                          "merton_hist", "proposed"])
    est_cfg: EstimatorConfig = field(default_factory=EstimatorConfig)
    exec_cfg: ExecutionConfig = field(default_factory=ExecutionConfig)
    execution_enabled: bool = True
    out_dir: Path = ROOT / "results" / "backtest" / "main"

    @classmethod
    def from_toml(cls, path) -> "BacktestConfig":
        path = Path(path)
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        data, bt, model = raw.get("data", {}), raw.get("backtest", {}), raw.get("model", {})
        ex, out = raw.get("execution", {}), raw.get("output", {})
        est_keys = EstimatorConfig.__dataclass_fields__.keys()
        est = EstimatorConfig(**{k: v for k, v in model.items() if k in est_keys})
        exe = ExecutionConfig(**{k: v for k, v in ex.items() if k != "enabled"})
        out_dir = Path(out.get("dir", "results/backtest"))
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        kw = {
            "tickers": list(data["tickers"]),
            "benchmark": data.get("benchmark", "SPY"),
            "data_start": data.get("start", "2005-01-01"),
            "data_end": data.get("end", "2026-01-01"),
            "start": bt.get("start", "2021-01-01"),
            "end": bt.get("end", "2025-12-31"),
            "rebalance": bt.get("rebalance", "monthly"),
            "window": int(bt.get("window", est.window)),
            "cost_bps": float(bt.get("cost_bps", 5.0)),
            "initial_nav": float(bt.get("initial_nav", 1e6)),
            "gamma": float(model.get("gamma", 5.0)),
            "horizon": float(model.get("horizon", 1 / 12)),
            "est_cfg": est,
            "exec_cfg": exe,
            "execution_enabled": bool(ex.get("enabled", True)),
            "out_dir": out_dir / path.stem,
        }
        if "strategies" in bt:
            kw["strategies"] = list(bt["strategies"])
        return cls(**kw)

    @property
    def assets(self) -> list[str]:
        return self.tickers + [self.benchmark]


@dataclass
class BacktestData:
    prices: pd.DataFrame  # ETF 거래일 달력, 열 = 유니버스 + 벤치마크
    rf: pd.Series  # 연속복리 무위험금리 (연)
    vix: pd.DataFrame | None  # VIX 기간구조 (열 = 만기(년))
    volume: pd.DataFrame | None


@dataclass
class StrategyResult:
    nav_net: pd.Series
    nav_gross: pd.Series
    weights: pd.DataFrame  # 리밸런싱일별 목표 비중 (+ CASH)
    turnover: pd.Series  # 리밸런싱일별 sum |거래액| / NAV
    costs: pd.Series  # 리밸런싱일별 비용 ($)
    diagnostics: pd.DataFrame


class CostModel(Protocol):
    def __call__(self, trade_notional: np.ndarray, ctx: RebalanceContext) -> float: ...


@dataclass
class ProportionalCost:
    """비례비용 bps x sum |거래액|. (14주차: Almgren–Chriss 충격비용 모형을 같은 자리에 붙인다.)"""

    bps: float

    def __call__(self, trade_notional: np.ndarray, ctx: RebalanceContext) -> float:
        return self.bps * 1e-4 * float(np.abs(trade_notional).sum())


def load_data(cfg: BacktestConfig) -> BacktestData:
    """가격·금리·VIX·거래량을 ETF 달력에 맞춘다. 고정된 기간이므로 캐시를 만료시키지 않는다."""
    from mfin.data import get_prices, get_risk_free, get_vix_term, get_volume

    kw = {"start": cfg.data_start, "end": cfg.data_end, "max_age": None}
    prices = get_prices(cfg.assets, **kw).dropna(how="any")
    idx = prices.index
    rf = get_risk_free(**kw).reindex(idx).ffill(limit=5)
    vix = get_vix_term(**kw).reindex(idx).ffill(limit=5)
    volume = get_volume(cfg.assets, **kw).reindex(idx)
    return BacktestData(prices, rf, vix, volume)


def rebalance_dates(index: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    """[start, end] 안에서 매월 첫 거래일."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).first().to_numpy())


def run_strategy(strategy: Strategy, data: BacktestData, cfg: BacktestConfig, cost_model: CostModel) -> StrategyResult:
    """전략 하나를 시뮬레이션한다."""
    prices = data.prices
    dates = prices.index
    P = prices.to_numpy(dtype=float)
    rf = data.rf.reindex(dates).fillna(0.0).to_numpy()
    rebal = rebalance_dates(dates, cfg.start, cfg.end)
    if len(rebal) == 0:
        raise ValueError("no rebalance dates inside the backtest window")
    pos = {d: i for i, d in enumerate(dates)}
    reb_pos = set(pos[d] for d in rebal)
    i0 = pos[rebal[0]]
    i_end = int(np.searchsorted(dates, pd.Timestamp(cfg.end), side="right")) - 1
    need = max(cfg.window, cfg.est_cfg.mom_lookback) + 1
    if i0 < need:
        raise ValueError(f"first rebalance needs {need} prior rows, only {i0} available")

    n = P.shape[1]
    h = np.zeros(n)  # 보유액 ($)
    hg = np.zeros(n)  # 비용이 없는 장부
    cash = cashg = float(cfg.initial_nav)
    # 시계열은 첫 거래 전날 종가(전액 현금)에서 시작한다. 그래야 첫 매수 비용도 수익률에 들어간다
    nav_net, nav_gross = [cash], [cashg]
    w_rows, w_idx, turns, costs = [], [], [], []
    for i in range(i0, i_end + 1):
        growth = P[i] / P[i - 1]
        h *= growth
        hg *= growth
        carry = np.exp(rf[i - 1] * (dates[i] - dates[i - 1]).days / 365.0)
        cash *= carry
        cashg *= carry
        nav = h.sum() + cash
        navg = hg.sum() + cashg
        if i in reb_pos:
            ctx = RebalanceContext(
                date=dates[i],
                prices=prices.iloc[:i],
                vix_row=data.vix.iloc[i - 1] if data.vix is not None else None,
                r=float(rf[i - 1]),
                volume=data.volume.iloc[:i] if data.volume is not None else None,
                current_weights=h / nav,
                nav=nav,
            )
            w = np.asarray(strategy.weights(ctx), dtype=float)
            if w.shape != (n,) or np.any(w < -1e-12) or w.sum() > 1 + 1e-9:
                raise ValueError(f"{strategy.name}: invalid weights {w}")
            trade = w * nav - h
            cost = cost_model(trade, ctx)
            # 비용을 낸 뒤의 NAV로 목표 비중을 맞춘다 (현금이 음수가 되지 않게)
            nav -= cost
            h = w * nav
            cash = nav - h.sum()
            hg = w * navg
            cashg = navg - hg.sum()
            w_rows.append(np.append(w, 1 - w.sum()))
            w_idx.append(dates[i])
            turns.append(float(np.abs(trade).sum() / (nav + cost)))
            costs.append(cost)
        nav_net.append(nav)
        nav_gross.append(navg)

    span = dates[i0 - 1 : i_end + 1]
    return StrategyResult(
        nav_net=pd.Series(nav_net, index=span, name=strategy.name),
        nav_gross=pd.Series(nav_gross, index=span, name=strategy.name),
        weights=pd.DataFrame(w_rows, index=pd.DatetimeIndex(w_idx, name="date"), columns=cfg.assets + [CASH]),
        turnover=pd.Series(turns, index=w_idx),
        costs=pd.Series(costs, index=w_idx),
        diagnostics=pd.DataFrame(strategy.diagnostics),
    )


def run_backtest(cfg: BacktestConfig, data: BacktestData | None = None) -> dict[str, StrategyResult]:
    data = data or load_data(cfg)
    out = {}
    for name in cfg.strategies:
        print(f"=== {name} ===", flush=True)
        out[name] = run_strategy(build_strategy(name, cfg), data, cfg, ProportionalCost(cfg.cost_bps))
    return out


def metrics_table(results: dict[str, StrategyResult], data: BacktestData) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        idx = res.nav_net.index
        dt = pd.Series(np.r_[0.0, np.diff(idx.values).astype("timedelta64[D]").astype(float)], index=idx)
        rf_daily = np.expm1(data.rf.reindex(idx).fillna(0.0).shift(1).fillna(0.0) * dt / 365.0)
        rows[name] = performance_table(res.nav_net, res.nav_gross, rf_daily, res.turnover, res.costs)
    return pd.DataFrame(rows).T


def save_results(results: dict[str, StrategyResult], table: pd.DataFrame, out: Path) -> None:
    from mfin.experiments import _plot

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index_label="strategy")
    navs = pd.DataFrame({k: v.nav_net for k, v in results.items()})
    navs.to_csv(out / "nav.csv", index_label="date")
    for name, res in results.items():
        res.weights.to_csv(out / f"weights_{name}.csv")
        if not res.diagnostics.empty:
            res.diagnostics.to_csv(out / f"diagnostics_{name}.csv", index=False)

    _plot.setup()
    import matplotlib.pyplot as plt

    every = max(1, len(navs) // 20)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(navs.columns):
        ax.plot(navs.index, navs[name] / navs[name].iloc[0], color=_plot.SERIES[i % 5],
                marker=_plot.MARKERS[i % 5], markevery=every, label=name)
    ax.set_yscale("log")
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    if navs.max().max() / navs.iloc[0].min() > 4:
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Net asset value, monthly rebalancing")
    ax.legend()
    _plot.save(fig, out / "nav.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, name in enumerate(navs.columns):
        ax.plot(navs.index, 100 * drawdown(navs[name]), color=_plot.SERIES[i % 5],
                marker=_plot.MARKERS[i % 5], markevery=every, label=name, linewidth=1.5)
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Drawdown from running peak")
    ax.legend()
    _plot.save(fig, out / "drawdown.png")

    if "proposed" in results:
        w = results["proposed"].weights
        w = w.loc[:, (w > 1e-9).any(axis=0)]
        cols = [c for c in w.columns if c != CASH] + ([CASH] if CASH in w.columns else [])
        colors = [_plot.SERIES[i % 5] for i in range(len(cols) - 1)] + [_plot.CASH]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.stackplot(w.index, (100 * w[cols]).T.to_numpy(), labels=cols, colors=colors,
                     edgecolor=_plot.SURFACE, linewidth=0.8, step="post")
        ax.set_ylim(0, 100)
        ax.set_ylabel("weight (%)")
        ax.set_title("Proposed model: target weights at each rebalance")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        _plot.save(fig, out / "weights_proposed.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_DIR / "main.toml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strategies", type=str, default=None, help="comma-separated subset")
    args = ap.parse_args()
    cfg = BacktestConfig.from_toml(args.config)
    if args.strategies:
        cfg.strategies = [s.strip() for s in args.strategies.split(",")]
    out = args.out or cfg.out_dir
    data = load_data(cfg)
    results = run_backtest(cfg, data)
    table = metrics_table(results, data)
    save_results(results, table, out)
    pd.set_option("display.width", 160)
    fmt = table.copy()
    for c in ["cagr", "ann_vol", "mdd", "cost_drag"]:
        fmt[c] = (100 * fmt[c]).round(2)
    print("\n(percent: cagr, ann_vol, mdd, cost_drag)")
    print(fmt.drop(columns=["total_cost", "final_nav"]).round(3).to_string())
    for name, res in results.items():
        d = res.diagnostics
        if "vol_source" in d:
            print(f"{name}: vol source counts {d['vol_source'].value_counts().to_dict()}")
    print(f"\nsaved to {out}", flush=True)


if __name__ == "__main__":
    main()
