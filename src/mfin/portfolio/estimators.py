"""모듈 C의 추정 단계: 가격 이력에서 시장 상태 함수 u (Merton 계수)를 만든다.

모든 함수는 리밸런싱 시점 t의 전날(t-1)까지의 행만 받는 순수 함수다. 미래참조를 막는 책임은
호출하는 쪽(백테스트 루프)에 있고, 여기서는 받은 행을 모두 과거로 취급한다.

    R       252일 수익률의 Ledoit–Wolf(2004) 축소 상관행렬 (상수상관 타깃)
    s_i     63일 EWMA 실현변동성 비율 (자산 i / 벤치마크)
    mu_inf  CAPM 균형수익률 r + beta_i ERP (사전분포)
    mu0     mu_inf + IC omega_i z_i sqrt(12)   (Grinold: 12-1 모멘텀 z-점수를 월간 IC로 수익률화)
    kappa   ln 2 / 반감기   (모멘텀 신호가 균형수익률로 감쇠)
    sigma_fwd  VIX 기간구조(백테스트) 또는 SPX 옵션체인(실시간)의 구간상수 선도분산
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mfin.engine.merton import MertonCoeffs
from mfin.engine.vol_term import forward_variance_knots

VIX_30D = 30 / 365  # ^VIX의 만기 (년)


@dataclass
class EstimatorConfig:
    window: int = 252  # 공분산·베타 추정 창 (거래일)
    ewma_span: int = 63  # 변동성 배율 s_i의 EWMA span
    mom_lookback: int = 252  # 12-1 모멘텀: 252거래일 전부터
    mom_skip: int = 21  # 최근 21거래일(1개월)은 뺀다 (단기 반전 효과)
    erp: float = 0.05  # 주식 위험프리미엄 (연)
    ic: float = 0.05  # 모멘텀 신호의 월간 정보계수
    signal_halflife: float = 0.5  # 신호 반감기 (년)
    z_clip: float = 3.0
    adv_window: int = 63  # 평균 거래량(중앙값) 창
    periods_per_year: int = 252


@dataclass
class MarketState:
    asof: pd.Timestamp  # 추정에 쓴 마지막 데이터 날짜 (t-1)
    tickers: list[str]
    coeffs: MertonCoeffs
    prices: np.ndarray  # (n,) 마지막 종가
    sigma_ann: np.ndarray  # (n,) EWMA 실현변동성 (연)
    beta: np.ndarray  # (n,) 벤치마크 대비 베타
    adv_shares: np.ndarray | None  # (n,) 평균 일 거래량(주)
    skew: float | None  # SPX skew sigma(0.9F) - sigma(1.1F) (실시간 경로만)
    vol_source: str  # "vix" | "chain" | "ewma" ...
    diagnostics: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------- 추정량


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """단순수익률 P_t / P_{t-1} - 1 (첫 행 제외)."""
    prices = np.asarray(prices, dtype=float)
    return prices[1:] / prices[:-1] - 1.0


def ledoit_wolf_corr(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit–Wolf(2004) 상수상관 타깃 축소 상관행렬과 축소 강도 delta.

    표본 공분산 S와 타깃 F(F_ii = S_ii, F_ij = r_bar s_i s_j)를 delta F + (1-delta) S로 섞는다.
    대각이 그대로이므로 상관행렬도 (1-delta) R_hat + delta [(1-r_bar) I + r_bar 11^T]이다.
    """
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    if n < 2:
        return np.ones((1, 1)), 0.0
    X = X - X.mean(axis=0)
    S = X.T @ X / T
    sd = np.sqrt(np.diag(S))
    R_hat = S / np.outer(sd, sd)
    off = ~np.eye(n, dtype=bool)
    rbar = float(R_hat[off].mean())
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))

    Z = np.einsum("ti,tj->tij", X, X) - S  # x_t x_t^T - S
    pi_mat = np.mean(Z**2, axis=0)
    Y = X**2 - np.diag(S)  # x_ti^2 - S_ii
    theta = np.einsum("ti,tij->ij", Y, Z) / T  # theta[i, j] = mean((x_i^2 - S_ii)(x_i x_j - S_ij))
    ratio = sd[None, :] / sd[:, None]  # ratio[i, j] = s_j / s_i
    rho = np.trace(pi_mat) + 0.5 * rbar * np.sum((ratio * theta + ratio.T * theta.T)[off])
    gamma_hat = float(np.sum((F - S) ** 2))
    if gamma_hat <= 1e-300:
        delta = 0.0
    else:
        delta = float(np.clip((pi_mat.sum() - rho) / gamma_hat / T, 0.0, 1.0))
    target = (1 - rbar) * np.eye(n) + rbar * np.ones((n, n))
    R = (1 - delta) * R_hat + delta * target
    np.fill_diagonal(R, 1.0)
    return R, delta


def ewma_vol(returns: np.ndarray, span: int, periods_per_year: int = 252) -> np.ndarray:
    """유한 창 EWMA 변동성 (평균 0 가정, 연율). 가장 최근 수익률의 가중치가 가장 크다."""
    X = np.asarray(returns, dtype=float)
    a = 2.0 / (span + 1.0)
    w = (1.0 - a) ** np.arange(X.shape[0])[::-1]
    w = w / w.sum()
    return np.sqrt(periods_per_year * (w[:, None] * X**2).sum(axis=0))


def betas_and_resid_vol(asset_ret, bench_ret, periods_per_year: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """벤치마크에 대한 OLS 베타와 잔차 변동성 omega (연율)."""
    A = np.asarray(asset_ret, dtype=float)
    b = np.asarray(bench_ret, dtype=float)
    Ac = A - A.mean(axis=0)
    bc = b - b.mean()
    beta = (Ac * bc[:, None]).sum(axis=0) / (bc @ bc)
    resid = Ac - np.outer(bc, beta)
    omega = np.sqrt(periods_per_year * (resid**2).sum(axis=0) / (len(b) - 2))
    return beta, omega


def momentum_12_1(prices: np.ndarray, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """12-1 모멘텀 P[-1-skip] / P[-1-lookback] - 1 (마지막 행 = t-1)."""
    P = np.asarray(prices, dtype=float)
    if P.shape[0] < lookback + 1:
        raise ValueError(f"momentum needs at least {lookback + 1} rows, got {P.shape[0]}")
    return P[-1 - skip] / P[-1 - lookback] - 1.0


def cross_sectional_z(x, clip: float = 3.0) -> np.ndarray:
    """횡단면 z-점수 (모집단 표준편차, +-clip에서 자름). 분산이 0이면 0."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if not np.isfinite(sd) or sd < 1e-14:
        return np.zeros_like(x)
    return np.clip((x - x.mean()) / sd, -clip, clip)


def expected_return_model(r: float, beta, resid_vol, mom, cfg: EstimatorConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """(mu0, mu_inf, kappa). 균형수익률을 사전으로 하고 모멘텀 view를 Grinold 공식으로 더한다.

    월간 IC와 월간 잔차변동성 omega/sqrt(12)로 만든 월간 알파를 연율로 바꾸면 IC omega z sqrt(12).
    """
    beta = np.asarray(beta, dtype=float)
    mu_inf = r + beta * cfg.erp
    z = cross_sectional_z(mom, cfg.z_clip)
    mu0 = mu_inf + cfg.ic * np.asarray(resid_vol, dtype=float) * z * np.sqrt(12.0)
    kappa = np.log(2.0) / cfg.signal_halflife
    return mu0, mu_inf, float(kappa)


# ----------------------------------------------------------------------- 변동성 기간구조


def vix_knots(vix_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """VIX 기간구조 한 행(만기(년) -> 변동성)에서 선도분산 마디. ^VIX(30일)는 반드시 있어야 한다."""
    row = vix_row.dropna()
    taus = row.index.to_numpy(dtype=float)
    if not np.any(np.isclose(taus, VIX_30D)):
        raise ValueError("VIX row has no 30-day (^VIX) value")
    return forward_variance_knots(taus, row.to_numpy(dtype=float))


def chain_knots(ts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """옵션체인 ATM 기간구조(:func:`mfin.engine.vol_term.term_structure` 출력)에서 선도분산 마디."""
    ok = ts["atm_iv"].notna()
    return forward_variance_knots(ts.loc[ok, "T"].to_numpy(), ts.loc[ok, "atm_iv"].to_numpy())


# ------------------------------------------------------------------------------- 상태 추정


def estimate_state(
    prices: pd.DataFrame,
    bench: str,
    r: float,
    vol_knots: tuple[np.ndarray, np.ndarray] | None,
    cfg: EstimatorConfig | None = None,
    volume: pd.DataFrame | None = None,
    skew: float | None = None,
    vol_source: str = "implied",
) -> MarketState:
    """t-1까지의 가격으로 시장 상태를 추정한다.

    prices: 열 = 자산들 + 벤치마크, 마지막 행 = t-1. vol_knots가 None이면 벤치마크 EWMA
    변동성을 일정한 선도변동성으로 쓴다 (vol_source = "ewma").
    """
    cfg = cfg or EstimatorConfig()
    tickers = [c for c in prices.columns if c != bench]
    need = max(cfg.window, cfg.mom_lookback) + 1
    panel = prices[tickers + [bench]]
    if len(panel) < need:
        raise ValueError(f"need at least {need} price rows, got {len(panel)}")
    P = panel.to_numpy(dtype=float)[-need:]
    if not np.all(np.isfinite(P)):
        raise ValueError("price panel has missing values inside the estimation window")
    n = len(tickers)
    rets = simple_returns(P[-(cfg.window + 1):])

    R, delta = ledoit_wolf_corr(rets[:, :n])
    vol = ewma_vol(rets, cfg.ewma_span, cfg.periods_per_year)
    s = vol[:n] / vol[n]
    beta, omega = betas_and_resid_vol(rets[:, :n], rets[:, n], cfg.periods_per_year)
    mom = momentum_12_1(P[:, :n], cfg.mom_lookback, cfg.mom_skip)
    mu0, mu_inf, kappa = expected_return_model(r, beta, omega, mom, cfg)

    if vol_knots is None:
        knots, fwd_var = np.array([0.0, 1.0]), np.array([vol[n] ** 2])
        # 호출자가 붙인 대체 표시(예: "ewma_fallback")는 유지한다
        vol_source = vol_source if vol_source.startswith("ewma") else "ewma"
    else:
        knots, fwd_var = vol_knots
    coeffs = MertonCoeffs(r, s, R, mu0, mu_inf, kappa, knots, fwd_var)

    adv = None
    if volume is not None:
        v = volume[tickers].to_numpy(dtype=float)[-cfg.adv_window:]
        adv = np.nanmedian(v, axis=0)

    diag = {
        "lw_delta": delta,
        "bench_vol_ewma": float(vol[n]),
        "sigma_fwd_1m": float(np.sqrt(coeffs.mean_var(0.0, 1 / 12))),
    }
    return MarketState(
        asof=pd.Timestamp(panel.index[-1]),
        tickers=tickers,
        coeffs=coeffs,
        prices=P[-1, :n].copy(),
        sigma_ann=vol[:n],
        beta=beta,
        adv_shares=adv,
        skew=skew,
        vol_source=vol_source,
        diagnostics=diag,
    )
