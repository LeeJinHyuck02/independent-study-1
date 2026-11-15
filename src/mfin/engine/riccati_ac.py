"""다자산 Almgren–Chriss 엔진 (모듈 B2): 행렬 Riccati ODE와 닫힌 해.

문제
----
보유량 q in R^n (주식 수, 양수 = 매도할 물량)을 [0, T] 동안 dq = -nu dt로 줄인다. 비용은

    J = int_0^T ( nu^T Lam nu + phi q^T Sigma q ) dt + q_T^T A_T q_T

(Lam: 일시 충격 행렬, Sigma: 가격 공분산 (시간당), phi: 위험회피). HJB를 h = -q^T A(t) q로
풀면 A는 행렬 Riccati 방정식을 따른다.

    A' = A Lam^{-1} A - phi Sigma,   A(T) = A_T,    nu* = Lam^{-1} A(t) q,    최소 비용 = q0^T A(0) q0

닫힌 해
-------
M = phi Lam^{-1/2} Sigma Lam^{-1/2} = U diag(kappa_i^2) U^T 로 두고 A_T = alpha Lam (alpha = inf이면
전량 청산)이면 A = Lam^{1/2} U diag(a_i(tau)) U^T Lam^{1/2} (tau = T - t)이고 각 모드는
스칼라 Riccati da/dtau = kappa^2 - a^2, a(0) = alpha의 해다.

    a(tau) = (1 + zeta e^{-2 kappa tau}) / D(tau),
    D(tau) = 2 tau phi1(2 kappa tau) + 2 e^{-2 kappa tau} / (alpha + kappa),   zeta = (alpha - kappa)/(alpha + kappa)

(phi1(x) = (1 - e^{-x})/x). alpha = inf이면 a = kappa coth(kappa tau)이고, kappa -> 0이면 TWAP 해
1/(tau + 1/alpha)가 된다. 이 형태는 지수 증가항이 없어 overflow가 나지 않는다.

모드 좌표 y = U^T Lam^{1/2} q에서 최적 궤적은 y_i(t) = y_i(0) e^{-kappa_i t} D_i(T - t) / D_i(T)
(n = 1, alpha = inf이면 q(t) = q0 sinh(kappa (T - t)) / sinh(kappa T), kappa = sqrt(phi sigma^2 / k)).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson, solve_ivp


def _phi1(x):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(small, 1.0 - 0.5 * x, -np.expm1(-x) / np.where(small, 1.0, x))


def _sym_sqrt(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """대칭 양정치 행렬의 제곱근과 역제곱근."""
    w, V = np.linalg.eigh(S)
    if np.any(w <= 0):
        raise ValueError("Lam must be symmetric positive definite")
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


@dataclass(frozen=True, eq=False)
class ACProblem:
    Lam: np.ndarray  # (n, n) 일시 충격 행렬 (대칭 양정치)
    Sigma: np.ndarray  # (n, n) 단위시간당 가격 공분산
    phi: float  # 위험회피 계수
    T: float  # 집행 기간
    alpha: float | None = None  # A_T = alpha * Lam. None이면 전량 청산 (A_T = inf)

    def __post_init__(self):
        object.__setattr__(self, "Lam", np.atleast_2d(np.asarray(self.Lam, dtype=float)))
        object.__setattr__(self, "Sigma", np.atleast_2d(np.asarray(self.Sigma, dtype=float)))
        n = self.Lam.shape[0]
        if self.Lam.shape != (n, n) or self.Sigma.shape != (n, n):
            raise ValueError("Lam and Sigma must be square with the same size")
        if not (np.allclose(self.Lam, self.Lam.T) and np.allclose(self.Sigma, self.Sigma.T)):
            raise ValueError("Lam and Sigma must be symmetric")
        if self.T <= 0 or self.phi < 0:
            raise ValueError("T must be positive and phi non-negative")
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError("alpha must be positive (or None for full liquidation)")

    @property
    def n(self) -> int:
        return int(self.Lam.shape[0])

    @property
    def A_T(self) -> np.ndarray:
        return np.full((self.n, self.n), np.inf) if self.alpha is None else self.alpha * self.Lam


def eigen_modes(p: ACProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(kappa, U, Lam^{1/2}, Lam^{-1/2}). kappa_i^2는 phi Lam^{-1/2} Sigma Lam^{-1/2}의 고윳값."""
    Lh, Lih = _sym_sqrt(p.Lam)
    M = p.phi * Lih @ p.Sigma @ Lih
    w, U = np.linalg.eigh(0.5 * (M + M.T))
    return np.sqrt(np.clip(w, 0.0, None)), U, Lh, Lih


def _mode_parts(kappa: np.ndarray, tau: np.ndarray, alpha: float | None):
    """모드별 분자 N(tau) = 1 + zeta e^{-2 kappa tau}와 분모 D(tau). a = N / D, b = 1/a = D / N."""
    tau = np.asarray(tau, dtype=float)[..., None]
    e = np.exp(-2.0 * kappa * tau)
    if alpha is None:
        zeta, tail = 1.0, 0.0
    else:
        zeta = (alpha - kappa) / (alpha + kappa)
        tail = 2.0 * e / (alpha + kappa)
    num = 1.0 + zeta * e
    den = 2.0 * tau * _phi1(2.0 * kappa * tau) + tail
    return num, den


def _check_t(p: ACProblem, t) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if np.any(t < 0) or np.any(t > p.T * (1 + 1e-12)):
        raise ValueError("t must lie in [0, T]")
    return np.minimum(t, p.T)


def riccati_closed_form(p: ACProblem, t) -> np.ndarray:
    """A(t), 모양 (len(t), n, n). alpha = None이면 t = T에서 무한대이므로 t < T만 쓴다."""
    t = np.atleast_1d(_check_t(p, t))
    kappa, U, Lh, _ = eigen_modes(p)
    num, den = _mode_parts(kappa, p.T - t, p.alpha)
    with np.errstate(divide="ignore"):
        a = num / den
    G = Lh @ U
    return np.einsum("ij,tj,kj->tik", G, a, G)


def riccati_inverse_closed_form(p: ACProblem, t) -> np.ndarray:
    """B(t) = A(t)^{-1}. t = T에서도 유한하다 (alpha = None이면 B(T) = 0)."""
    t = np.atleast_1d(_check_t(p, t))
    kappa, U, _, Lih = eigen_modes(p)
    num, den = _mode_parts(kappa, p.T - t, p.alpha)
    G = Lih @ U
    return np.einsum("ij,tj,kj->tik", G, den / num, G)


@dataclass
class RiccatiSolution:
    t: np.ndarray  # 오름차순 시점
    A: np.ndarray  # (len(t), n, n). 전량 청산이면 t = T에서 inf
    B: np.ndarray | None  # 전량 청산일 때 적분한 A^{-1}
    nfev: int
    seconds: float


def solve_riccati(
    p: ACProblem, t_eval, A_T: np.ndarray | None = None, rtol: float = 1e-11, atol: float = 1e-13
) -> RiccatiSolution:
    """행렬 Riccati ODE를 tau = T - t 방향으로 수치 적분한다 (DOP853).

    - 전량 청산(alpha = None, A_T 미지정): A는 t = T에서 발산하므로 B = A^{-1}을 적분한다.
      dB/dtau = Lam^{-1} - phi B Sigma B,  B(tau = 0) = 0.
    - 그 외: dA/dtau = phi Sigma - A Lam^{-1} A,  A(tau = 0) = A_T (A_T를 주면 그 행렬, 아니면 alpha Lam).
    """
    t = np.sort(np.atleast_1d(_check_t(p, t_eval)))
    tau = p.T - t[::-1]
    n = p.n
    Lam_inv = np.linalg.inv(p.Lam)
    Lam_inv = 0.5 * (Lam_inv + Lam_inv.T)
    S = p.Sigma
    use_B = A_T is None and p.alpha is None

    if use_B:

        def rhs(_, y):
            Bm = y.reshape(n, n)
            d = Lam_inv - p.phi * Bm @ S @ Bm
            return (0.5 * (d + d.T)).ravel()

        y0 = np.zeros(n * n)
    else:
        A0 = p.alpha * p.Lam if A_T is None else np.asarray(A_T, dtype=float)

        def rhs(_, y):
            Am = y.reshape(n, n)
            d = p.phi * S - Am @ Lam_inv @ Am
            return (0.5 * (d + d.T)).ravel()

        y0 = A0.ravel()

    t0 = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, p.T), y0, method="DOP853", t_eval=tau, rtol=rtol, atol=atol)
    secs = time.perf_counter() - t0
    if not sol.success:
        raise RuntimeError(f"Riccati integration failed: {sol.message}")
    Y = sol.y.T.reshape(len(tau), n, n)[::-1]  # 다시 t 오름차순
    if use_B:
        A = np.full_like(Y, np.inf)
        ok = (p.T - t) > 0
        A[ok] = np.linalg.inv(Y[ok])
        return RiccatiSolution(t, A, Y, sol.nfev, secs)
    return RiccatiSolution(t, Y, None, sol.nfev, secs)


# ------------------------------------------------------------------------------------ 궤적


def _modal(p: ACProblem, q0):
    kappa, U, Lh, Lih = eigen_modes(p)
    q0 = np.asarray(q0, dtype=float)
    return kappa, U, Lh, Lih, U.T @ Lh @ q0


def trajectory_closed_form(p: ACProblem, q0, t) -> np.ndarray:
    """최적 보유량 q*(t), 모양 (len(t), n)."""
    t = np.atleast_1d(_check_t(p, t))
    kappa, U, _, Lih, y0 = _modal(p, q0)
    _, den_t = _mode_parts(kappa, p.T - t, p.alpha)
    _, den_0 = _mode_parts(kappa, np.array(p.T), p.alpha)
    y = y0 * np.exp(-kappa * t[:, None]) * den_t / den_0
    return y @ (Lih @ U).T


def rate_closed_form(p: ACProblem, q0, t) -> np.ndarray:
    """최적 매매 속도 nu*(t) = Lam^{-1} A(t) q*(t), 모양 (len(t), n). t = T에서도 유한하다."""
    t = np.atleast_1d(_check_t(p, t))
    kappa, U, _, Lih, y0 = _modal(p, q0)
    num_t, _ = _mode_parts(kappa, p.T - t, p.alpha)
    _, den_0 = _mode_parts(kappa, np.array(p.T), p.alpha)
    # a(tau) y(t) = (N/D)(tau) * y0 e^{-kappa t} D(tau)/D(T) = y0 e^{-kappa t} N(tau) / D(T)
    v = y0 * np.exp(-kappa * t[:, None]) * num_t / den_0
    return v @ (Lih @ U).T


def optimal_cost(p: ACProblem, q0) -> float:
    """최소 비용 q0^T A(0) q0."""
    q0 = np.asarray(q0, dtype=float)
    return float(q0 @ riccati_closed_form(p, [0.0])[0] @ q0)


def schedule_cost(p: ACProblem, t, q, nu) -> dict[str, float]:
    """주어진 궤적의 비용 분해 (Simpson 적분): 충격, 위험 벌점, 종단 벌점."""
    t = np.asarray(t, dtype=float)
    q = np.atleast_2d(np.asarray(q, dtype=float))
    nu = np.atleast_2d(np.asarray(nu, dtype=float))
    impact = float(simpson(np.einsum("ti,ij,tj->t", nu, p.Lam, nu), x=t))
    risk = float(p.phi * simpson(np.einsum("ti,ij,tj->t", q, p.Sigma, q), x=t))
    qT = q[-1]
    if p.alpha is None:
        terminal = 0.0 if np.allclose(qT, 0.0, atol=1e-12 * max(1.0, np.abs(q[0]).max())) else np.inf
    else:
        terminal = float(p.alpha * qT @ p.Lam @ qT)
    return {"impact": impact, "risk": risk, "terminal": terminal, "total": impact + risk + terminal}
