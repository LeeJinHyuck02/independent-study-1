"""다자산 Almgren–Chriss HJB의 PINN (모듈 B2 연구 경로, 차원 확장 실험용).

서비스의 체결 스케줄은 Riccati 엔진(``mfin.engine.riccati_ac``)이 계산한다. 이 모듈은 같은
방정식을 PINN으로 풀어, 정답이 있는 문제에서 상태 차원 1+n이 커질 때 정확도·시간·메모리가
어떻게 변하는지 잰다(격자법은 격자점이 N^n으로 늘어 n >= 3부터 사실상 불가능).

문제 (tau = T - t)
------------------
    h_t - phi q^T Sigma q + 1/4 grad h^T Lam^{-1} grad h = 0,    h(T, q) = -q^T A_T q,  A_T = alpha Lam
    nu* = -1/2 Lam^{-1} grad h        (정답: h = -q^T A(t) q, nu* = Lam^{-1} A(t) q)

alpha가 유한해야 한다(전량 청산 A_T = inf는 신경망으로 표현할 수 없다). 1계 방정식이라 2계
도함수가 필요 없다.

기준해 + 보정 (모듈 A의 Bachelier 기준해와 같은 발상)
--------------------------------------------------
alpha = 10이면 A(t)가 만기 직전 tau ~ 1/alpha 구간에서 alpha Lam에서 정상상태로 급변한다
(n = 10에서 tr A(0) / tr A_T = 0.13). 종단조건을 (tau/T) 인자로만 강제하면 신경망이 이 경계층을
통째로 배워야 한다. 그래서 위험 벌점이 없는(phi = 0) 문제의 정확해

    h_ref(tau, q) = -q^T Lam q / (tau + 1/alpha)        (TWAP + 종단 벌점)

를 기준으로 두고 신경망은 위험 벌점이 만드는 보정만 배운다.

    gate = "ref"    : h = h_ref + S_ref(tau) (tau/T) N(z),   S_ref(tau) = E_box[q^T Lam q] / (tau + 1/alpha)
    gate = "linear" : h = -q^T A_T q - alpha E_box[q^T Lam q] (tau/T) N(z)      (비교용, 기준해 없음)

두 형태 모두 tau = 0에서 종단조건을 정확히 만족한다. N은 일반 tanh MLP이고, 내부 해가 q의
2차식이라는 구조는 넣지 않는다(보고서에 기준해 사용을 명시한다).

손실: mean((T R / S_ref(tau))^2). 잔차를 기준해 크기로 나누므로 차원 n과 시간 tau에 대해 스케일이
고르다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from mfin.engine.riccati_ac import ACProblem, riccati_closed_form, schedule_cost, trajectory_closed_form
from mfin.pinn.core import DTYPE, MLP, TrainConfig, set_seed, train

HFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

DEFAULT_SLICES = (0.0, 0.25, 0.5, 0.75, 0.9)


# --------------------------------------------------------------------------- 잔차 연산자


def hjb_residual(
    h_fn: HFn, t: torch.Tensor, q: torch.Tensor, lam_inv: torch.Tensor, sigma: torch.Tensor, phi: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """임의의 가치함수 h(t, q)에 대한 HJB 잔차와 grad_q h.

    t: (B, 1), q: (B, n), 둘 다 requires_grad. 도함수는 autograd 한 번으로 함께 구한다.
    """
    h = h_fn(t, q)
    h_t, h_q = torch.autograd.grad(h, (t, q), grad_outputs=torch.ones_like(h), create_graph=True)
    risk = phi * ((q @ sigma) * q).sum(dim=1, keepdim=True)
    ham = 0.25 * ((h_q @ lam_inv) * h_q).sum(dim=1, keepdim=True)
    return h_t - risk + ham, h_q


def policy_from_value(h_fn: HFn, t: torch.Tensor, q: torch.Tensor, lam_inv: torch.Tensor) -> torch.Tensor:
    """nu = -1/2 Lam^{-1} grad_q h (Lam은 대칭). q는 requires_grad여야 한다."""
    h = h_fn(t, q)
    (h_q,) = torch.autograd.grad(h.sum(), q)
    return -0.5 * h_q @ lam_inv


# ------------------------------------------------------------------------------- 모형


@dataclass
class ACPinnConfig:
    q_max: float = 1.0  # 정의역 [-q_max, q_max]^n
    n_pde: int = 4096  # 콜로케이션 점 수
    width: int = 64
    depth: int = 4
    gate: str = "ref"  # "ref" (기준해 + 보정) | "linear" (기준해 없음, 비교용)
    radial_frac: float = 0.5  # q 표본 중 원점 쪽 반경 표본 비율 (n이 크면 상자 균일 표본이 원점 근처를 비운다)
    terminal_frac: float = 0.5  # t 표본 중 만기 근처 집중 표본 비율 (tau = T u^2)
    seed: int = 0
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(adam_iters=5000, lbfgs_iters=1500, resample_every=500, log_every=1000)
    )


@dataclass
class ACPinnResult:
    train_seconds: float
    history: pd.DataFrame
    model: "ACPinn"


class ACPinn(nn.Module):
    """(t, q) in [0, T] x [-q_max, q_max]^n 위의 가치함수 h(t, q)."""

    def __init__(self, problem: ACProblem, cfg: ACPinnConfig | None = None):
        super().__init__()
        if problem.alpha is None:
            raise ValueError("ACPinn needs a finite terminal penalty (alpha), not full liquidation")
        self.cfg = cfg or ACPinnConfig()
        if self.cfg.gate not in ("ref", "linear"):
            raise ValueError("gate must be 'ref' or 'linear'")
        self.problem = problem
        self.n, self.T, self.alpha, self.phi = problem.n, float(problem.T), float(problem.alpha), float(problem.phi)
        set_seed(self.cfg.seed)
        self.net = MLP(1 + self.n, 1, self.cfg.width, self.cfg.depth)

        lam_inv = np.linalg.inv(problem.Lam)
        self.register_buffer("lam", torch.as_tensor(problem.Lam, dtype=DTYPE))
        self.register_buffer("lam_inv", torch.as_tensor(0.5 * (lam_inv + lam_inv.T), dtype=DTYPE))
        self.register_buffer("sigma", torch.as_tensor(problem.Sigma, dtype=DTYPE))
        # 상자 균일분포에서 E[q^T Lam q] = q_max^2 / 3 * tr(Lam) (좌표가 독립이라 비대각 항은 0)
        self.m_box = float(self.cfg.q_max**2 / 3.0 * np.trace(problem.Lam))

        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        self.resample()

    # ------------------------------------------------------------------ 모형
    def _qlq(self, q: torch.Tensor) -> torch.Tensor:
        return ((q @ self.lam) * q).sum(dim=1, keepdim=True)

    def scale(self, tau: torch.Tensor) -> torch.Tensor:
        """기준해 크기 S_ref(tau) = E_box[q^T Lam q] / (tau + 1/alpha)."""
        return self.m_box / (tau + 1.0 / self.alpha)

    def forward(self, t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        tau = self.T - t
        z = torch.cat([2.0 * t / self.T - 1.0, q / self.cfg.q_max], dim=1)
        corr = (tau / self.T) * self.net(z)
        if self.cfg.gate == "ref":
            return -self._qlq(q) / (tau + 1.0 / self.alpha) + self.scale(tau) * corr
        return -self.alpha * self._qlq(q) - self.alpha * self.m_box * corr

    # ------------------------------------------------------------- 점 샘플링
    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self._gen, dtype=DTYPE)

    def sample_q(self, m: int) -> torch.Tensor:
        """상자 균일 + 원점 쪽 반경 표본 q = q_max rho u (u: 단위구 균일, rho ~ U(0, 1))."""
        c = self.cfg
        n_rad = int(c.radial_frac * m)
        box = c.q_max * (2.0 * self._rand(m - n_rad, self.n) - 1.0)
        u = torch.randn(n_rad, self.n, generator=self._gen, dtype=DTYPE)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rad = c.q_max * self._rand(n_rad, 1) * u
        return torch.cat([box, rad])

    def sample_t(self, m: int) -> torch.Tensor:
        """균일 + 만기 근처 집중(tau = T u^2)."""
        n_term = int(self.cfg.terminal_frac * m)
        uni = self.T * self._rand(m - n_term, 1)
        term = self.T - self.T * self._rand(n_term, 1) ** 2
        return torch.cat([uni, term])

    def resample(self) -> None:
        m = self.cfg.n_pde
        perm = torch.randperm(m, generator=self._gen)
        self.t_pde = self.sample_t(m)[perm].requires_grad_(True)
        self.q_pde = self.sample_q(m).requires_grad_(True)

    # ----------------------------------------------------------------- 손실
    def losses(self) -> tuple[torch.Tensor, dict[str, float]]:
        t, q = self.t_pde, self.q_pde
        res, _ = hjb_residual(self, t, q, self.lam_inv, self.sigma, self.phi)
        w = self.T / self.scale(self.T - t.detach())
        loss = torch.mean((res * w) ** 2)
        return loss, {"pde": loss.item()}

    def fit(self, train_cfg: TrainConfig | None = None) -> ACPinnResult:
        t0 = time.perf_counter()
        hist = train(self.parameters(), self.losses, train_cfg or self.cfg.train, self.resample)
        return ACPinnResult(time.perf_counter() - t0, pd.DataFrame(hist), self)

    # ----------------------------------------------------------- numpy 평가
    def _inputs(self, t, q, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        q = np.atleast_2d(np.asarray(q, dtype=float))
        t = np.broadcast_to(np.asarray(t, dtype=float).reshape(-1, 1), (q.shape[0], 1))
        tt = torch.as_tensor(np.array(t, dtype=float), dtype=DTYPE)
        qq = torch.as_tensor(q, dtype=DTYPE).requires_grad_(requires_grad)
        return tt, qq

    def value(self, t, q) -> np.ndarray:
        """h(t, q), q 모양 (B, n), t는 스칼라 또는 (B,)."""
        tt, qq = self._inputs(t, q)
        with torch.no_grad():
            return self(tt, qq).numpy()[:, 0]

    def rate(self, t, q) -> np.ndarray:
        """nu*(t, q) = -1/2 Lam^{-1} grad h, 모양 (B, n)."""
        tt, qq = self._inputs(t, q, requires_grad=True)
        return policy_from_value(self, tt, qq, self.lam_inv).detach().numpy()

    def residual(self, t, q) -> np.ndarray:
        """스케일된 잔차 T R / S_ref(tau), 모양 (B,)."""
        tt, qq = self._inputs(t, q, requires_grad=True)
        tt.requires_grad_(True)
        res, _ = hjb_residual(self, tt, qq, self.lam_inv, self.sigma, self.phi)
        return (res * self.T / self.scale(self.T - tt)).detach().numpy()[:, 0]

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def memory_estimate(n_params: int, n_pde: int, n_in: int, width: int, depth: int, lbfgs_history: int = 50) -> float:
    """학습 메모리의 대략적 추정치 (바이트, 상수배 오차).

    파라미터와 Adam 상태(파라미터 x 4), L-BFGS 이력(2 x history x 파라미터), 콜로케이션 입력,
    순전파·1계 도함수 그래프의 활성값(층마다 점 x 폭 x 4). 격자법 메모리 8 N^n과 같은 기준으로
    비교하기 위한 값이다.
    """
    floats = 4 * n_params + 2 * lbfgs_history * n_params + n_pde * n_in + 4 * n_pde * width * depth
    return 8.0 * floats


# ------------------------------------------------------------------ Riccati 대비 평가


def exact_value_rate(p: ACProblem, t: float, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """정답 h = -q^T A(t) q, nu = Lam^{-1} A(t) q (닫힌 해)."""
    A = riccati_closed_form(p, [t])[0]
    q = np.atleast_2d(q)
    return -np.einsum("bi,ij,bj->b", q, A, q), q @ (np.linalg.solve(p.Lam, A)).T


def eval_points(p: ACProblem, q_max: float = 1.0, n_box: int = 2000, n_paths: int = 20, seed: int = 123,
                slices=DEFAULT_SLICES) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """평가용 q0(경로 출발점)와 시간 단면별 시험점(상자 균일 + 정확 궤적 위 점).

    정확 궤적이 정의역을 벗어나면 PINN 평가가 외삽이 되므로 확인한다.
    """
    rng = np.random.default_rng(seed)
    box = q_max * (2.0 * rng.random((n_box, p.n)) - 1.0)
    q0s = q_max * (2.0 * rng.random((n_paths, p.n)) - 1.0)
    paths = np.stack([trajectory_closed_form(p, q0, list(slices)) for q0 in q0s], axis=1)  # (n_slices, n_paths, n)
    if np.abs(paths).max() > q_max * (1 + 1e-9):
        raise ValueError("an optimal path leaves the domain [-q_max, q_max]^n; enlarge q_max")
    return q0s, {float(s): np.vstack([box, paths[k]]) for k, s in enumerate(slices)}


def evaluate(value_fn, rate_fn, p: ACProblem, points: dict[float, np.ndarray]) -> dict[str, float]:
    """value_fn(t, q) -> (B,), rate_fn(t, q) -> (B, n)를 닫힌 해와 비교한다 (상대 L2 오차).

    rel_err_h_t{t}: 단면별 h 오차, rel_err_h0: t = 0 (최소 집행비용), rel_err_nu: 모든 단면을 합친 nu 오차.
    """
    out: dict[str, float] = {}
    num_nu = den_nu = 0.0
    for t, q in points.items():
        h_ex, nu_ex = exact_value_rate(p, t, q)
        h = np.asarray(value_fn(t, q)).reshape(-1)
        nu = np.asarray(rate_fn(t, q)).reshape(nu_ex.shape)
        out[f"rel_err_h_t{t:g}"] = float(np.linalg.norm(h - h_ex) / np.linalg.norm(h_ex))
        num_nu += float(np.sum((nu - nu_ex) ** 2))
        den_nu += float(np.sum(nu_ex**2))
    out["rel_err_h0"] = out[f"rel_err_h_t{min(points):g}"]
    out["rel_err_nu"] = float(np.sqrt(num_nu / den_nu))
    return out


def integrate_feedback(rate_fn, q0s: np.ndarray, T: float, n_steps: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q' = -nu(t, q)를 Heun 방법으로 적분한다. 여러 출발점을 한 번에 다룬다.

    반환: t (n_steps+1,), q (n_steps+1, K, n), nu (n_steps+1, K, n).
    """
    t = np.linspace(0.0, T, n_steps + 1)
    q = np.zeros((n_steps + 1, *q0s.shape))
    nu = np.zeros_like(q)
    q[0] = q0s
    for k in range(n_steps):
        dt = t[k + 1] - t[k]
        nu[k] = rate_fn(t[k], q[k])
        pred = q[k] - dt * nu[k]
        q[k + 1] = q[k] - 0.5 * dt * (nu[k] + rate_fn(t[k + 1], pred))
    nu[-1] = rate_fn(t[-1], q[-1])
    return t, q, nu


def trajectory_metrics(rate_fn, p: ACProblem, q0s: np.ndarray, n_steps: int = 2000) -> dict[str, float]:
    """정책을 실제로 집행했을 때의 지표.

    traj_err   : max_t |q(t) - q*(t)| / |q0| 의 최댓값 (q*는 닫힌 해 궤적, 1차 지표)
    excess_cost: 비용 / (같은 적분기로 정답 피드백 정책을 돌린 비용) - 1 의 평균 (2차 지표: 정책 오차 e에 대해 O(e^2))
    integrator_bias: 정답 피드백 비용 / 최소 비용 q0^T A(0) q0 - 1 (적분기 오차 확인용)
    """
    A = riccati_closed_form(p, np.linspace(0.0, p.T, n_steps + 1))
    grid = np.linspace(0.0, p.T, n_steps + 1)
    M = np.linalg.solve(p.Lam[None], A)  # Lam^{-1} A(t)

    def exact_rate(t, q):
        k = int(round(t / p.T * n_steps))
        return q @ M[k].T

    t, q, nu = integrate_feedback(rate_fn, q0s, p.T, n_steps)
    _, q_ex, nu_ex = integrate_feedback(exact_rate, q0s, p.T, n_steps)
    q_star = np.stack([trajectory_closed_form(p, q0, grid) for q0 in q0s], axis=1)
    norms = np.linalg.norm(q0s, axis=1)
    traj_err = float(np.max(np.linalg.norm(q - q_star, axis=2).max(axis=0) / norms))

    excess, bias = [], []
    for j, q0 in enumerate(q0s):
        c = schedule_cost(p, t, q[:, j], nu[:, j])["total"]
        c_ex = schedule_cost(p, t, q_ex[:, j], nu_ex[:, j])["total"]
        excess.append(c / c_ex - 1.0)
        bias.append(c_ex / float(q0 @ A[0] @ q0) - 1.0)
    return {"traj_err": traj_err, "excess_cost": float(np.mean(excess)),
            "excess_cost_max": float(np.max(excess)), "integrator_bias": float(np.max(np.abs(bias)))}
