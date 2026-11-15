"""Merton HJB 엔진 (모듈 B1): 시간가변 계수와 제약이 있는 CRRA 자산배분.

모형
----
위험자산 n개와 무위험자산(금리 r). 투자 비중 pi에 대한 로그부 x = log w의 HJB는

    V_t + sup_{pi in A} { (r + pi^T m(t)) V_x + 1/2 pi^T Sigma(t) pi (V_xx - V_x) } = 0,
    V(T, x) = e^{(1-gamma) x} / (1-gamma),    A = {pi >= 0, 1^T pi <= 1}

이고 계수는 다음과 같이 둔다.

    Sigma(t) = sigma_fwd^2(t) C,   C = diag(s) R diag(s)     (시간 의존성은 스칼라 하나)
    m(t) = mu(t) - r 1 = (mu_inf - r) + (mu0 - mu_inf) e^{-kappa t}

sigma_fwd^2는 모듈 A(:mod:`mfin.engine.vol_term`)의 구간상수 선도분산이다.

해
--
계수가 결정적이면 V = U(w) f(t)로 분리되고 최적 비중은 근시안적이다. 즉 각 시점에서

    Q*(t) = max_{pi in A} pi^T m(t) - gamma/2 pi^T Sigma(t) pi

의 최대점이 HJB의 정확한 해다(무제약이면 pi* = Sigma^{-1} m / gamma). 가치함수는

    V(t, x) = e^{(1-gamma) x} / (1-gamma) * exp((1-gamma) int_t^T (r + Q*(s)) ds).

구간 정책: 비중을 [t0, t0 + dt] 동안 고정하면 log 부가 정규분포를 따르므로
E[W^{1-gamma}] ∝ exp((1-gamma) dt [r + pi^T m_bar - gamma/2 pi^T Sigma_bar pi])이다.
따라서 구간 평균 m_bar, Sigma_bar로 푼 QP가 "구간 동안 고정된 비중" 중에서 정확한 최적이다.
월간 리밸런싱 백테스트는 이 구간 정책을 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import brentq, nnls

from mfin.engine.vol_term import DEFAULT_SENSORS, forward_variance_knots


def _phi1(x):
    """(1 - e^{-x}) / x. x -> 0에서 1로 이어 붙인다."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(small, 1.0 - 0.5 * x, -np.expm1(-x) / np.where(small, 1.0, x))
    return out[()] if out.ndim == 0 else out


def _exp_integral(rate: float, s0: float, s1: float) -> float:
    """int_{s0}^{s1} e^{-rate s} ds (rate -> 0이면 구간 길이)."""
    return float(np.exp(-rate * s0) * (s1 - s0) * _phi1(rate * (s1 - s0)))


# ---------------------------------------------------------------------------------------- 계수


@dataclass(frozen=True, eq=False)
class MertonCoeffs:
    """시장 상태 함수 u의 엔진용 표현 (모든 값은 연율, 소수)."""

    r: float
    s: np.ndarray  # (n,) 자산별 변동성 배율 (자산 변동성 / 지수 변동성)
    R: np.ndarray  # (n, n) 상관행렬
    mu0: np.ndarray  # (n,) 현재 기대수익률
    mu_inf: np.ndarray  # (n,) 장기 균형 기대수익률
    kappa: float  # 신호 감쇠율 (1/년)
    var_knots: np.ndarray  # (m+1,) [0, T_1, ..., T_m] (년)
    fwd_var: np.ndarray  # (m,) 구간 (knots[j], knots[j+1]]의 선도분산. 마지막 마디 이후는 일정

    def __post_init__(self):
        for name in ("s", "R", "mu0", "mu_inf", "var_knots", "fwd_var"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        object.__setattr__(self, "r", float(self.r))
        object.__setattr__(self, "kappa", float(self.kappa))
        n = self.s.shape[0]
        if self.R.shape != (n, n) or self.mu0.shape != (n,) or self.mu_inf.shape != (n,):
            raise ValueError("inconsistent shapes for s, R, mu0, mu_inf")
        if self.var_knots.ndim != 1 or self.var_knots[0] != 0.0 or len(self.var_knots) != len(self.fwd_var) + 1:
            raise ValueError("var_knots must be [0, T_1, ..., T_m] with len(fwd_var) + 1 entries")
        if np.any(np.diff(self.var_knots) <= 0) or np.any(self.fwd_var <= 0):
            raise ValueError("var_knots must increase and fwd_var must be positive")
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative")

    # 생성자 -------------------------------------------------------------------------------
    @classmethod
    def from_vol_term(cls, r, s, R, mu0, mu_inf, kappa, taus, sigmas, min_fwd_var: float = 1e-6) -> "MertonCoeffs":
        """ATM 변동성(또는 VIX) 기간구조 (taus, sigmas)로 만든다."""
        knots, fwd_var = forward_variance_knots(taus, sigmas, min_fwd_var)
        return cls(r, s, R, mu0, mu_inf, kappa, knots, fwd_var)

    @classmethod
    def flat(cls, r, s, R, mu0, mu_inf, kappa, sigma: float) -> "MertonCoeffs":
        """선도변동성이 sigma로 일정한 경우."""
        return cls(r, s, R, mu0, mu_inf, kappa, np.array([0.0, 1.0]), np.array([float(sigma) ** 2]))

    # 기본 양 ------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.s.shape[0])

    @property
    def C(self) -> np.ndarray:
        """Sigma(t) = sigma_fwd^2(t) C 인 상수 행렬 C = diag(s) R diag(s)."""
        return np.outer(self.s, self.s) * self.R

    def _cum_var(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.fwd_var * np.diff(self.var_knots))])

    def total_var(self, t):
        """총분산 w(t) = int_0^t sigma_fwd^2 ds (구간별 선형, 정확)."""
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError("t must be non-negative")
        k = self.var_knots
        idx = np.clip(np.searchsorted(k, t, side="right") - 1, 0, len(self.fwd_var) - 1)
        out = self._cum_var()[idx] + self.fwd_var[idx] * (t - k[idx])
        return out[()] if out.ndim == 0 else out

    def fwd_var_at(self, t):
        """순간 선도분산 sigma_fwd^2(t). 구간 j는 (knots[j], knots[j+1]] (vol_term과 같은 규약)."""
        t = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(self.var_knots, t, side="left") - 1, 0, len(self.fwd_var) - 1)
        out = self.fwd_var[idx]
        return out[()] if out.ndim == 0 else out

    def excess_mu(self, t) -> np.ndarray:
        """초과 기대수익 m(t) = mu(t) - r. 모양 (..., n)."""
        t = np.asarray(t, dtype=float)[..., None]
        return (self.mu_inf - self.r) + (self.mu0 - self.mu_inf) * np.exp(-self.kappa * t)

    def mean_excess_mu(self, t0: float, dt: float) -> np.ndarray:
        """구간 평균 (1/dt) int_{t0}^{t0+dt} m(s) ds (정확)."""
        if dt < 0:
            raise ValueError("dt must be non-negative")
        decay = np.exp(-self.kappa * t0) * _phi1(self.kappa * dt)
        return (self.mu_inf - self.r) + (self.mu0 - self.mu_inf) * decay

    def cov(self, t) -> np.ndarray:
        """순간 공분산 Sigma(t). 모양 (..., n, n)."""
        return np.asarray(self.fwd_var_at(t))[..., None, None] * self.C

    def mean_var(self, t0: float, dt: float) -> float:
        """구간 평균 선도분산 (w(t0 + dt) - w(t0)) / dt."""
        if dt <= 0:
            return float(self.fwd_var_at(np.nextafter(t0, np.inf)))
        return float((self.total_var(t0 + dt) - self.total_var(t0)) / dt)

    def mean_cov(self, t0: float, dt: float) -> np.ndarray:
        """구간 평균 공분산 (1/dt) int Sigma(s) ds = mean_var * C (정확)."""
        return self.mean_var(t0, dt) * self.C

    def branch_vector(self, gamma: float, sensors=DEFAULT_SENSORS) -> np.ndarray:
        """DeepONet branch 입력 (13주차 연구 모듈용).

        [sigma_fwd(센서), s, mu0, mu_inf, kappa, R 상삼각(대각 제외), r, gamma]
        """
        iu = np.triu_indices(self.n, k=1)
        return np.concatenate(
            [
                np.sqrt(self.fwd_var_at(np.asarray(sensors, dtype=float))),
                self.s,
                self.mu0,
                self.mu_inf,
                [self.kappa],
                self.R[iu],
                [self.r],
                [float(gamma)],
            ]
        )


# ------------------------------------------------------------------------------------ 이차계획


@dataclass
class QPResult:
    pi: np.ndarray  # 최적 비중 (n,). 현금 = budget - sum(pi)
    lam: float  # 예산 제약 1^T pi <= budget의 승수
    eta: np.ndarray  # pi >= 0 제약의 승수 (n,)
    objective: float  # m^T pi - gamma/2 pi^T Sigma pi
    budget_active: bool
    kkt: dict[str, float] = field(default_factory=dict)


def merton_unconstrained(m, Sigma, gamma: float) -> np.ndarray:
    """무제약 Merton 비중 Sigma^{-1} m / gamma."""
    m = np.asarray(m, dtype=float)
    return cho_solve(cho_factor(np.asarray(Sigma, dtype=float)), m) / float(gamma)


def kkt_residuals(pi, lam: float, m, Sigma, gamma: float, budget: float = 1.0) -> dict[str, float]:
    """KKT 조건 잔차. 모두 0이면 최적이다.

    g = m - gamma Sigma pi (목적함수 기울기), eta = lam 1 - g (pi >= 0의 승수).
    """
    pi = np.asarray(pi, dtype=float)
    m = np.asarray(m, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    g = m - gamma * Sigma @ pi
    eta = lam - g
    free = pi > 0
    slack = budget - pi.sum()
    return {
        "stationarity": float(np.max(np.abs(eta[free]), initial=0.0)),
        "primal": float(max(0.0, -pi.min(initial=0.0), -slack)),
        "dual": float(max(0.0, -eta.min(initial=0.0), -lam)),
        "complementarity": float(max(abs(lam * slack), np.max(np.abs(pi * eta), initial=0.0))),
    }


def _nnls_policy(Lt: np.ndarray, m: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """예산 승수 lam을 고정했을 때의 해: max (m - lam 1)^T pi - gamma/2 pi^T Sigma pi, pi >= 0.

    Sigma = L L^T이면 목적함수는 -1/2 ||sqrt(gamma) L^T pi - b||^2 + 상수,
    b = L^{-1}(m - lam 1) / sqrt(gamma). 이를 Lawson–Hanson NNLS(정확한 active-set)로 푼다.
    """
    sg = np.sqrt(gamma)
    b = solve_triangular(Lt.T, m - lam, lower=True) / sg
    pi, _ = nnls(sg * Lt, b)
    return pi


def _polish(pi: np.ndarray, m, Sigma, gamma, budget, budget_active) -> tuple[np.ndarray, float] | None:
    """자유 집합 F = {pi > 0}에서 KKT 등식계를 정확히 풀어 반올림 오차를 없앤다."""
    F = np.flatnonzero(pi > 0)
    n = len(pi)
    out = np.zeros(n)
    if len(F) == 0:
        return out, 0.0
    S_FF = gamma * Sigma[np.ix_(F, F)]
    if budget_active:
        k = len(F)
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = S_FF
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        rhs = np.concatenate([m[F], [budget]])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return None
        pi_F, lam = sol[:k], float(sol[k])
    else:
        try:
            pi_F = cho_solve(cho_factor(S_FF), m[F])
        except np.linalg.LinAlgError:
            return None
        lam = 0.0
    if np.any(pi_F <= 0) or lam < 0:
        return None
    out[F] = pi_F
    return out, lam


def solve_mv_qp(m, Sigma, gamma: float, budget: float = 1.0) -> QPResult:
    """max m^T pi - gamma/2 pi^T Sigma pi,  제약 pi >= 0, 1^T pi <= budget  (정확한 해법).

    1. 승수 lam = 0으로 NNLS를 풀고, 예산을 넘지 않으면 끝.
    2. 넘으면 1^T pi(lam)가 lam에 대해 연속이고 감소하므로 brentq로 1^T pi(lam) = budget을 푼다.
    3. 찾은 자유 집합에서 KKT 등식계로 다듬는다(실패하면 NNLS 해를 그대로 쓴다).
    """
    m = np.asarray(m, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    gamma = float(gamma)
    if gamma <= 0 or budget <= 0:
        raise ValueError("gamma and budget must be positive")
    try:
        L = cholesky(Sigma, lower=True)
    except np.linalg.LinAlgError:
        jitter = 1e-12 * np.trace(Sigma) / len(m)
        L = cholesky(Sigma + jitter * np.eye(len(m)), lower=True)
    Lt = L.T

    pi = _nnls_policy(Lt, m, gamma, 0.0)
    lam = 0.0
    active = pi.sum() > budget
    if active:
        hi = float(max(m.max(), 0.0)) + 1e-12
        lam = brentq(lambda l: _nnls_policy(Lt, m, gamma, l).sum() - budget, 0.0, hi, xtol=1e-15, maxiter=500)
        pi = _nnls_policy(Lt, m, gamma, lam)

    polished = _polish(pi, m, Sigma, gamma, budget, active)
    if polished is not None:
        cand, cand_lam = polished
        res = kkt_residuals(cand, cand_lam, m, Sigma, gamma, budget)
        if res["dual"] <= 1e-12 and res["primal"] <= 1e-12:
            pi, lam = cand, cand_lam

    eta = lam - (m - gamma * Sigma @ pi)
    obj = float(m @ pi - 0.5 * gamma * pi @ Sigma @ pi)
    return QPResult(pi, float(lam), eta, obj, bool(active), kkt_residuals(pi, lam, m, Sigma, gamma, budget))


# -------------------------------------------------------------------------------------- 정책


def period_inputs(coeffs: MertonCoeffs, t0: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """구간 [t0, t0 + dt]의 평균 초과수익과 평균 공분산 (m_bar, Sigma_bar)."""
    return coeffs.mean_excess_mu(t0, dt), coeffs.mean_cov(t0, dt)


def merton_policy(coeffs: MertonCoeffs, t: float, gamma: float, constrained: bool = True) -> np.ndarray:
    """시점 t의 순간 최적 비중 pi*(t) (HJB의 정확한 해)."""
    m, S = coeffs.excess_mu(t), coeffs.cov(t)
    return solve_mv_qp(m, S, gamma).pi if constrained else merton_unconstrained(m, S, gamma)


def merton_policy_period(
    coeffs: MertonCoeffs, t0: float, dt: float, gamma: float, constrained: bool = True
) -> np.ndarray:
    """[t0, t0 + dt] 동안 고정할 최적 비중 (구간 평균 계수의 QP, 정확)."""
    m, S = period_inputs(coeffs, t0, dt)
    return solve_mv_qp(m, S, gamma).pi if constrained else merton_unconstrained(m, S, gamma)


def _segments(coeffs: MertonCoeffs, t: float, T: float) -> list[tuple[float, float, float]]:
    """[t, T]를 선도분산이 일정한 구간으로 나눈다: (s0, s1, 분산)."""
    inner = coeffs.var_knots[(coeffs.var_knots > t) & (coeffs.var_knots < T)]
    pts = np.concatenate([[t], inner, [T]])
    return [(a, b, float(coeffs.fwd_var_at(0.5 * (a + b)))) for a, b in zip(pts[:-1], pts[1:]) if b > a]


def integrated_hamiltonian(
    coeffs: MertonCoeffs, t: float, T: float, gamma: float, constrained: bool = False, n_gauss: int = 8
) -> float:
    """int_t^T Q*(s) ds.

    무제약이면 Q* = m^T Sigma^{-1} m / (2 gamma)이고 구간마다 닫힌 형태로 적분한다
    (m = a + b e^{-kappa s}). 제약이 있으면 구간마다 Gauss–Legendre 적분으로 QP 값을 더한다.
    """
    if T < t:
        raise ValueError("T must be >= t")
    gamma = float(gamma)
    total = 0.0
    if not constrained:
        Cf = cho_factor(coeffs.C)
        a = coeffs.mu_inf - coeffs.r
        b = coeffs.mu0 - coeffs.mu_inf
        aa = a @ cho_solve(Cf, a)
        ab = a @ cho_solve(Cf, b)
        bb = b @ cho_solve(Cf, b)
        k = coeffs.kappa
        for s0, s1, v in _segments(coeffs, t, T):
            e1 = _exp_integral(k, s0, s1)
            e2 = _exp_integral(2 * k, s0, s1)
            total += (aa * (s1 - s0) + 2 * ab * e1 + bb * e2) / v
        return total / (2 * gamma)
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    C = coeffs.C
    for s0, s1, v in _segments(coeffs, t, T):
        half = 0.5 * (s1 - s0)
        for xi, wi in zip(nodes, weights):
            s = s0 + half * (xi + 1.0)
            total += wi * half * solve_mv_qp(coeffs.excess_mu(s), v * C, gamma).objective
    return total


def merton_value(
    coeffs: MertonCoeffs, t: float, x, T: float, gamma: float, constrained: bool = False, n_gauss: int = 8
):
    """로그부 좌표의 가치함수 V(t, x).

    gamma != 1:  e^{(1-gamma) x} / (1-gamma) * exp((1-gamma) int_t^T (r + Q*) ds)
    gamma == 1:  x + int_t^T (r + Q*) ds   (로그 효용)
    """
    x = np.asarray(x, dtype=float)
    growth = coeffs.r * (T - t) + integrated_hamiltonian(coeffs, t, T, gamma, constrained, n_gauss)
    if abs(gamma - 1.0) < 1e-12:
        out = x + growth
    else:
        g1 = 1.0 - gamma
        out = np.exp(g1 * x) / g1 * np.exp(g1 * growth)
    return out[()] if out.ndim == 0 else out
