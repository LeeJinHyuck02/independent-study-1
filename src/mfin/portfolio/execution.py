"""체결 계획: 추천 비중의 변경분을 다자산 Almgren–Chriss 엔진(Riccati 닫힌 해)으로 나눠 체결한다.

단위: 수량은 주식 수, 금액은 달러, 시간은 거래일.

    q0      = (pi_cur - pi*) W / S          (양수 = 매도, 음수 = 매수.  dq = -nu dt와 같은 부호)
    Lam_ii  = eta sigma_i,d S_i / ADV_i      ($ day / share^2, 선형 일시 충격. eta = 0.142, Almgren et al. 2005)
    Sigma$  = diag(S) Sigma_ann diag(S) / 252  ($^2 / share^2 / day)
    phi     = gamma / (2 W)                  (부 W에서 CRRA를 CARA로 근사한 위험회피, $^{-1})

phi가 운용자산 W에 반비례하므로 W가 클수록 충격비용이 상대적으로 커져 천천히 체결한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfin.engine.riccati_ac import (
    ACProblem,
    eigen_modes,
    optimal_cost,
    rate_closed_form,
    schedule_cost,
    trajectory_closed_form,
)


@dataclass
class ExecutionConfig:
    horizon_days: float = 1.0  # 집행 기간 (거래일)
    n_buckets: int = 13  # 30분 구간 (정규장 6.5시간)
    impact_eta: float = 0.142
    phi: float | None = None  # None이면 gamma / (2 W)
    periods_per_year: int = 252


@dataclass
class ExecutionSchedule:
    times: np.ndarray  # (k+1,) 거래일 단위 시점
    holdings: np.ndarray  # (k+1, n) 남은 주문량 (주)
    trades: np.ndarray  # (k, n) 구간별 체결량 (양수 = 매도)
    q0: np.ndarray  # (n,) 전체 주문 (주)
    impact_cost: float  # 기대 충격비용 ($)
    risk_penalty: float  # phi * 집행 중 가격위험 분산 ($)
    expected_cost: float  # 최적 비용 q0^T A(0) q0 ($)
    kappa: np.ndarray  # 모드별 긴급도 (1/일)
    phi: float


def order_from_weights(pi_target, pi_cur, aum: float, prices) -> np.ndarray:
    """비중 변경을 주식 수 주문으로 바꾼다: (pi_cur - pi_target) W / S (양수 = 매도)."""
    pi_target = np.asarray(pi_target, dtype=float)
    pi_cur = np.asarray(pi_cur, dtype=float)
    return (pi_cur - pi_target) * float(aum) / np.asarray(prices, dtype=float)


def impact_matrix(prices, sigma_daily, adv_shares, eta: float = 0.142) -> np.ndarray:
    """대각 일시 충격 행렬 Lam_ii = eta sigma_i,d S_i / ADV_i."""
    prices = np.asarray(prices, dtype=float)
    adv = np.asarray(adv_shares, dtype=float)
    if np.any(~np.isfinite(adv)) or np.any(adv <= 0):
        raise ValueError("ADV must be positive and finite")
    return np.diag(eta * np.asarray(sigma_daily, dtype=float) * prices / adv)


def price_cov_daily(prices, cov_ann, periods_per_year: int = 252) -> np.ndarray:
    """수익률 연 공분산을 주당 가격의 일 공분산으로 바꾼다."""
    S = np.asarray(prices, dtype=float)
    return np.outer(S, S) * np.asarray(cov_ann, dtype=float) / periods_per_year


def plan_execution(q0, Lam, Sigma_price, phi: float, horizon: float = 1.0, n_buckets: int = 13) -> ExecutionSchedule:
    """Almgren–Chriss 최적 체결 스케줄 (전량 체결, 닫힌 해)."""
    q0 = np.asarray(q0, dtype=float)
    p = ACProblem(Lam, Sigma_price, phi, horizon)
    times = np.linspace(0.0, horizon, n_buckets + 1)
    holdings = trajectory_closed_form(p, q0, times)
    holdings[-1] = 0.0
    t = np.linspace(0.0, horizon, 4001)
    cost = schedule_cost(p, t, trajectory_closed_form(p, q0, t), rate_closed_form(p, q0, t))
    kappa, *_ = eigen_modes(p)
    return ExecutionSchedule(
        times=times,
        holdings=holdings,
        trades=-np.diff(holdings, axis=0),
        q0=q0,
        impact_cost=cost["impact"],
        risk_penalty=cost["risk"],
        expected_cost=optimal_cost(p, q0),
        kappa=kappa,
        phi=float(phi),
    )


def twap_schedule(q0, Lam, Sigma_price, phi: float, horizon: float = 1.0, n_buckets: int = 13) -> ExecutionSchedule:
    """비교용 TWAP (일정 속도) 스케줄과 같은 비용 모형에서의 비용."""
    q0 = np.asarray(q0, dtype=float)
    p = ACProblem(Lam, Sigma_price, phi, horizon)
    times = np.linspace(0.0, horizon, n_buckets + 1)
    holdings = np.outer(1.0 - times / horizon, q0)
    t = np.linspace(0.0, horizon, 4001)
    q = np.outer(1.0 - t / horizon, q0)
    nu = np.tile(q0 / horizon, (len(t), 1))
    cost = schedule_cost(p, t, q, nu)
    kappa, *_ = eigen_modes(p)
    return ExecutionSchedule(times, holdings, -np.diff(holdings, axis=0), q0, cost["impact"], cost["risk"],
                             cost["impact"] + cost["risk"], kappa, float(phi))
