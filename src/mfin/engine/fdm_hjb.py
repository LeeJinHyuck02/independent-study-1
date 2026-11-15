"""Almgren–Chriss HJB의 격자 해법 (n = 1, 2). Riccati 엔진의 교차검증과 PINN 비교 기준.

tau = T - t, 비용함수 u = -h (u = q^T A q >= 0)로 쓰면 HJB는

    u_tau + sum_i H_i(d_i u) = phi q^T Sigma q,    H_i(p) = p^2 / (4 Lam_ii),    u(tau = 0, q) = q^T A_T q

인 1계 Hamilton–Jacobi 방정식이다(확산항 없음). Lam이 대각이면 해밀토니안이 차원별로 분리되므로
차원마다 Godunov 수치 해밀토니안을 쓴다(볼록 H에 대한 표준형).

    H_i^G = max( max(D^- u, 0)^2, min(D^+ u, 0)^2 ) / (4 Lam_ii)

이 스킴은 CFL 조건 dtau * sum_i |nu_i| / dq_i <= 1에서 단조이고, 따라서 점성해로 수렴한다
(Barles–Souganidis). 정확도는 1차이므로 Richardson 외삽을 함께 보고한다.

경계: 해가 q의 2차식이므로 경계 밖 가상 격자점을 2차 외삽 (3u_0 - 3u_1 + u_2)으로 채운다.
종단조건은 유한한 A_T가 필요하다(전량 청산 A_T = inf는 격자로 표현할 수 없다).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FDMConfig:
    n_points: int = 101  # 차원당 격자점 수 (홀수면 q = 0이 격자점)
    q_max: float | Sequence[float] = 1.5  # 영역 [-q_max, q_max]^n
    cfl: float = 0.9
    n_snapshots: int = 101  # 저장할 시점 수 (t = 0 ... T 등간격)


@dataclass
class FDMResult:
    grids: list[np.ndarray]
    t: np.ndarray  # (n_snap,) 오름차순
    h: np.ndarray  # (n_snap, N[, N]) 가치함수 h = -u (<= 0)
    lam_diag: np.ndarray
    n_steps: int
    seconds: float
    _rate: np.ndarray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.grids)

    def _blend(self, arr: np.ndarray, t: float) -> np.ndarray:
        """스냅샷 사이를 시간에 대해 선형 보간."""
        t = float(np.clip(t, self.t[0], self.t[-1]))
        j = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        w = (t - self.t[j]) / (self.t[j + 1] - self.t[j])
        return (1 - w) * arr[j] + w * arr[j + 1]

    def value(self, t: float, q) -> np.ndarray:
        """h(t, q). q 모양 (..., n). 공간은 3차 보간 (해가 2차식이라 보간 오차가 거의 없다)."""
        q = np.asarray(q, dtype=float)
        interp = RegularGridInterpolator(self.grids, self._blend(self.h, t), method="cubic",
                                         bounds_error=False, fill_value=None)
        out = interp(q.reshape(-1, self.n)).reshape(q.shape[:-1])
        return out[()] if out.ndim == 0 else out

    def _rates(self) -> np.ndarray:
        """스냅샷마다 nu = -1/2 Lam^{-1} grad h. 모양 (n_snap, n, N[, N])."""
        if self._rate is None:
            out = []
            for h in self.h:
                g = np.gradient(h, *self.grids)
                g = [g] if self.n == 1 else g
                out.append(np.stack([-0.5 * gi / lam for gi, lam in zip(g, self.lam_diag)]))
            self._rate = np.stack(out)
        return self._rate

    def rate(self, t: float, q) -> np.ndarray:
        """매매 속도 nu*(t, q), 모양 (..., n)."""
        q = np.asarray(q, dtype=float)
        field_t = self._blend(self._rates(), t)
        pts = q.reshape(-1, self.n)
        comps = [
            RegularGridInterpolator(self.grids, field_t[i], bounds_error=False, fill_value=None)(pts)
            for i in range(self.n)
        ]
        return np.stack(comps, axis=-1).reshape(q.shape)

    def trajectory(self, q0, n_steps: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """q' = -nu*(t, q)를 Heun 방법으로 적분한 최적 궤적 (t, q)."""
        t = np.linspace(self.t[0], self.t[-1], n_steps + 1)
        q = np.zeros((n_steps + 1, self.n))
        q[0] = np.asarray(q0, dtype=float)
        for k in range(n_steps):
            dt = t[k + 1] - t[k]
            k1 = -self.rate(t[k], q[k])
            k2 = -self.rate(t[k + 1], q[k] + dt * k1)
            q[k + 1] = q[k] + 0.5 * dt * (k1 + k2)
        return t, q


def _pad(u: np.ndarray, axis: int) -> np.ndarray:
    """axis 방향 양 끝에 2차 외삽 가상 격자점을 붙인다."""
    u = np.moveaxis(u, axis, 0)
    lo = 3 * u[0] - 3 * u[1] + u[2]
    hi = 3 * u[-1] - 3 * u[-2] + u[-3]
    return np.moveaxis(np.concatenate([lo[None], u, hi[None]], axis=0), 0, axis)


def _godunov(u: np.ndarray, axis: int, dq: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """차원 axis의 Godunov 해밀토니안과 국소 특성 속도 |nu|."""
    P = np.moveaxis(_pad(u, axis), axis, 0)
    dm = (P[1:-1] - P[:-2]) / dq
    dp = (P[2:] - P[1:-1]) / dq
    a = np.maximum(dm, 0.0)
    b = np.minimum(dp, 0.0)
    H = np.maximum(a * a, b * b) / (4 * lam)
    speed = np.maximum(a, -b) / (2 * lam)
    return np.moveaxis(H, 0, axis), np.moveaxis(speed, 0, axis)


def solve_ac_fdm(Lam, Sigma, phi: float, T: float, A_T, cfg: FDMConfig | None = None) -> FDMResult:
    """(t, q) 격자에서 Almgren–Chriss HJB를 푼다. n in {1, 2}, Lam은 대각이어야 한다."""
    cfg = cfg or FDMConfig()
    Lam = np.atleast_2d(np.asarray(Lam, dtype=float))
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    A_T = np.atleast_2d(np.asarray(A_T, dtype=float))
    n = Lam.shape[0]
    if n not in (1, 2):
        raise ValueError("FDM supports n = 1 or 2 only (grid size grows as N^n)")
    if not np.allclose(Lam, np.diag(np.diag(Lam))):
        raise ValueError("FDM requires a diagonal impact matrix Lam (separable Hamiltonian)")
    if not np.all(np.isfinite(A_T)):
        raise ValueError("FDM needs a finite terminal penalty A_T")
    lam = np.diag(Lam).copy()
    qmax = np.broadcast_to(np.asarray(cfg.q_max, dtype=float), (n,))
    grids = [np.linspace(-qm, qm, cfg.n_points) for qm in qmax]
    dq = np.array([g[1] - g[0] for g in grids])
    Q = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)  # (N[, N], n)
    source = phi * np.einsum("...i,ij,...j->...", Q, Sigma, Q)
    u = np.einsum("...i,ij,...j->...", Q, A_T, Q)

    taus = np.linspace(0.0, T, cfg.n_snapshots)
    snaps = [u.copy()]
    tau, steps = 0.0, 0
    t0 = time.perf_counter()
    for target in taus[1:]:
        while tau < target - 1e-15 * T:
            Hs, cfl_rate = 0.0, 0.0
            for i in range(n):
                H, sp = _godunov(u, i, dq[i], lam[i])
                Hs = Hs + H
                cfl_rate = cfl_rate + sp / dq[i]
            m = float(np.max(cfl_rate))
            dtau = target - tau if m == 0 else min(cfg.cfl / m, target - tau)
            u = u + dtau * (source - Hs)
            tau += dtau
            steps += 1
        tau = target
        snaps.append(u.copy())
    secs = time.perf_counter() - t0
    h = -np.stack(snaps)[::-1]  # t = T - tau 오름차순
    return FDMResult(grids, T - taus[::-1], h, lam, steps, secs)


def richardson(h_coarse: float, h_fine: float, order: int = 1) -> float:
    """격자 간격을 반으로 줄인 두 해로 만든 Richardson 외삽값."""
    f = 2.0**order
    return (f * h_fine - h_coarse) / (f - 1.0)
