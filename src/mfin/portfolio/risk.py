"""위험 리포트: 포트폴리오 변동성, 모수적 VaR/CVaR, 위험기여도, (선택) SPX 보호적 풋 비용.

모든 양은 추천 비중을 유지할 구간 [0, horizon]의 평균 계수(Merton 엔진과 같은 Sigma_bar, m_bar)로
계산한다. 1일 VaR/CVaR는 정규분포 가정(모수적)이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from mfin.engine.bs_analytic import bs_delta, bs_price
from mfin.engine.merton import MertonCoeffs


@dataclass
class RiskReport:
    vol_ann: float  # 포트폴리오 연변동성
    excess_ret_ann: float  # 기대 초과수익 (연)
    var95_1d: float  # 1일 95% VaR (NAV 대비 손실 비율, 양수 = 손실)
    cvar95_1d: float  # 1일 95% CVaR
    risk_contrib: np.ndarray  # 자산별 위험기여도 (합 = 1)
    beta_p: float  # 포트폴리오 베타
    skew: float | None  # SPX skew sigma(0.9F) - sigma(1.1F)
    put_hedge: dict | None  # 보호적 풋 비용·델타 (NAV 대비)


def parametric_var_cvar(mu_h: float, sigma_h: float, alpha: float = 0.95) -> tuple[float, float]:
    """정규분포 수익률 N(mu_h, sigma_h^2)의 VaR, CVaR (손실을 양수로)."""
    z = norm.ppf(alpha)
    var = -mu_h + z * sigma_h
    cvar = -mu_h + sigma_h * norm.pdf(z) / (1.0 - alpha)
    return float(var), float(cvar)


def protective_put(exposure: float, r: float, sigma: float, T: float, moneyness: float = 0.9) -> dict:
    """주식 노출 exposure(NAV 대비, 베타 조정)를 행사가 moneyness * S인 SPX 풋으로 헤지하는 비용.

    상수 변동성 블랙-숄즈 가격이므로 실제 OTM 풋 가격(skew로 더 비쌈)보다 낮게 나온다.
    """
    unit = float(bs_price(1.0, moneyness, T, r, sigma, kind="put"))
    delta = float(bs_delta(1.0, moneyness, T, r, sigma, kind="put"))
    return {
        "moneyness": moneyness,
        "T": T,
        "sigma": sigma,
        "cost": exposure * unit,
        "cost_ann": exposure * unit / T,
        "delta": exposure * delta,
    }


def risk_report(
    pi,
    coeffs: MertonCoeffs,
    horizon: float,
    beta,
    skew: float | None = None,
    put_hedge: bool = False,
    alpha: float = 0.95,
    periods_per_year: int = 252,
) -> RiskReport:
    """비중 pi(현금 제외)의 위험 리포트."""
    pi = np.asarray(pi, dtype=float)
    S = coeffs.mean_cov(0.0, horizon)
    m = coeffs.mean_excess_mu(0.0, horizon)
    var_p = float(pi @ S @ pi)
    vol = np.sqrt(var_p)
    ex = float(pi @ m)
    mu_1d = (coeffs.r + ex) / periods_per_year
    var95, cvar95 = parametric_var_cvar(mu_1d, vol / np.sqrt(periods_per_year), alpha)
    rc = pi * (S @ pi) / var_p if var_p > 0 else np.zeros_like(pi)
    beta_p = float(pi @ np.asarray(beta, dtype=float))
    hedge = None
    if put_hedge:
        sigma = float(np.sqrt(coeffs.mean_var(0.0, horizon)))
        hedge = protective_put(beta_p, coeffs.r, sigma, horizon)
    return RiskReport(vol, ex, var95, cvar95, rc, beta_p, skew, hedge)
