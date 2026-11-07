"""역방향 블랙-숄즈 PINN: 한 만기의 옵션가격에서 σ를 복원한다.

문제 (선도가격으로 정규화한 블랙-숄즈, ``mfin.engine.bs_analytic`` 참고)
----------------------------------------------------------------------------
미지수: 가격함수 v(tau, x)와 변동성 sigma > 0.

    PDE        v_tau = 1/2 sigma^2 x^2 v_xx         tau in (0, T], x in (x_lo, x_hi)
               x_lo, x_hi = 1 -/+ n_std * sigma_init * sqrt(T)   (확산 길이에 맞춘 정의역)
    초기조건   v(0, x) = (x - 1)^+
    경계조건   v(tau, x_lo) = (x_lo - 1)^+,  v(tau, x_hi) = x_hi - 1   (내재가치)
    데이터     v(T, x_k) = v_k   (만기 T 옵션의 시장가격을 정규화한 값)

특이점 제거 (기준해 빼기)
-------------------------
만기 손익 (x - 1)^+는 x = 1에서 꺾이는데, 매끄러운 tanh 신경망은 이 꺾임을 표현하지
못한다. 초기조건을 손실로만 맞추면 꺾임이 뭉개지고, 그만큼 시간가치가 더해져 σ가 작게
추정된다(합성 데이터에서 -22% 관찰). 그래서 해를 다음과 같이 쓴다.

    v(tau, x) = Phi(tau, x; sigma) + scale * s * N_theta(s, x),     s = tau / T

Phi는 같은 초기조건을 가진 *열방정식* Phi_tau = 1/2 sigma^2 Phi_xx의 정확해다
(Bachelier 가격).

    Phi = (x - 1) N(z) + a phi(z),   a = sigma sqrt(tau),   z = (x - 1) / a.

꺾임은 Phi가 담당하므로 v(0, x) = (x - 1)^+가 정확히 성립하고, 신경망은 블랙-숄즈
확산항의 x^2 인자 때문에 생기는 매끄러운 보정분만 배운다. Phi는 블랙-숄즈의 해가 아니다.
블랙-숄즈 PDE는 여전히 잔차 손실을 통해 신경망이 풀어야 한다.

두 가지 변형 (``InverseBSConfig.ref_sigma``):
- None (기본값): Phi가 학습 중인 sigma를 쓴다. 모든 만기에서 정확하다. 다만 ATM 근방에서는
  Phi가 블랙-숄즈 가격과 거의 같아서 sigma가 상당 부분 Phi 자체로 정해진다
  (``bachelier_only_sigma``와 비교).
- 숫자: Phi가 그 고정 변동성을 쓰므로 sigma는 PDE 잔차를 통해서만 들어온다.
  더 엄밀한 역문제지만 짧은 만기에서 어렵다.

손실 = w_pde L_pde + w_bc L_bc + w_data L_data. 각 항은 잔차를 ``scale``(데이터 중 가장 큰
시간가치)로 나눈 뒤 제곱평균한 값이다. PDE 잔차는 s 단위로 쓴다. PDE 항이 v가 블랙-숄즈
방정식을 만족하도록 강제하므로 순방향 해(가격함수)도 sigma와 함께 학습된다. 따라서 별도의
순방향 모형이 필요 없다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from scipy.special import ndtr
from torch import nn

from mfin.engine.bs_analytic import black_call
from mfin.pinn.core import DTYPE, MLP, TrainConfig, grad, set_seed, to_tensor, train

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


def bachelier_reference(tau: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """초기조건이 (x - 1)^+인 열방정식의 해 (정규 변동성 sigma의 Bachelier 가격)."""
    # clamp로 두 분기를 모두 유한하게 유지해야 torch.where의 기울기도 유한하다
    a = sigma * torch.sqrt(torch.clamp(tau, min=1e-14))
    z = (x - 1.0) / a
    val = (x - 1.0) * torch.special.ndtr(z) + a * _INV_SQRT_2PI * torch.exp(-0.5 * z * z)
    return torch.where(tau > 0, val, torch.relu(x - 1.0))


@dataclass
class InverseBSConfig:
    x_max: float = 3.0  # x 정의역의 상한
    n_std: float = 6.0  # 정의역 반폭 (sigma_init * sqrt(T)의 배수)
    n_pde: int = 2000
    n_bc: int = 100
    width: int = 64
    depth: int = 4
    w_pde: float = 1.0
    w_bc: float = 1.0
    # 데이터 가중치를 일부러 PDE 가중치보다 작게 둔다. 실제 시세에는 상수 sigma 모형으로는
    # 맞출 수 없는 smile이 있다. w_data = 10이면 신경망이 smile을 맞추려고 만기 근처에서
    # PDE를 어겼고, sigma가 최대 +3.8 vol pt 치우쳤다. 0.1이면 PINN이 제약 최소제곱과
    # 0.1 vol pt 이내로 일치한다. mfin.experiments.module_a_robustness 참고.
    w_data: float = 0.1
    sigma_init: float = 0.3
    s_min: float = 1e-3  # PDE 점에 쓰는 tau/T의 최솟값
    # 기준해 Phi: None이면 학습 중인 sigma를 쓰고, 숫자면 그 값을 고정 정규 변동성으로 쓴다.
    # 숫자일 때 sigma는 PDE 잔차를 통해서만 모형에 들어온다.
    ref_sigma: float | None = None
    focus_frac: float = 0.5  # PDE 점 중 ATM 근방에 집중시키는 비율
    seed: int = 0
    train: TrainConfig = field(default_factory=lambda: TrainConfig(adam_iters=2000, lbfgs_iters=1000))


@dataclass
class InverseBSResult:
    T: float
    sigma: float
    train_seconds: float
    history: pd.DataFrame
    model: "InverseBSPINN"


class InverseBSPINN(nn.Module):
    """만기 T 하나에 대한 PINN. 데이터 (x_k, v_k)는 tau = T 위에 놓인다."""

    def __init__(self, T: float, x_data, v_data, weights=None, cfg: InverseBSConfig | None = None):
        super().__init__()
        self.cfg = cfg or InverseBSConfig()
        self.T = float(T)
        set_seed(self.cfg.seed)
        self.net = MLP(2, 1, self.cfg.width, self.cfg.depth)
        self.rho = nn.Parameter(torch.tensor(_inv_softplus(self.cfg.sigma_init), dtype=DTYPE))

        self.x_data = to_tensor(x_data)
        self.v_data = to_tensor(v_data)
        w = np.ones(len(np.atleast_1d(x_data))) if weights is None else np.asarray(weights, dtype=float)
        self.w_data = to_tensor(w / w.mean())

        # 국소 정의역 [x_lo, x_hi] = 1 -/+ n_std * a sqrt(T). 해는 확산 길이 a sqrt(T) 규모로
        # 변하므로 정의역을 이 길이에 맞추면 신경망에게는 모든 만기가 비슷해 보인다
        # (고정 정의역 [0, 3]에서는 짧은 만기가 실패한다).
        a = max(self.cfg.sigma_init, self.cfg.ref_sigma or 0.0)
        half = self.cfg.n_std * a * np.sqrt(self.T)
        self.x_lo = float(max(0.0, 1.0 - half))
        self.x_hi = float(min(self.cfg.x_max, 1.0 + half))

        x = np.asarray(x_data, dtype=float)
        if x.min() < self.x_lo or x.max() > self.x_hi:
            raise ValueError(f"data moneyness outside the PDE domain [{self.x_lo:.3f}, {self.x_hi:.3f}]")
        time_value = np.asarray(v_data, dtype=float) - np.maximum(x - 1.0, 0.0)
        self.scale = float(max(time_value.max(), 1e-3))

        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        self.resample()

    # ------------------------------------------------------------------ 모형
    @property
    def sigma(self) -> torch.Tensor:
        return nn.functional.softplus(self.rho)

    def correction(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s = tau / self.T
        z = torch.cat([2.0 * s - 1.0, 2.0 * (x - self.x_lo) / (self.x_hi - self.x_lo) - 1.0], dim=1)
        # 학습형 기준해면 보정분이 O(s), 고정 기준해면 O(sqrt(s))
        gate = s if self.cfg.ref_sigma is None else torch.sqrt(s)
        return self.scale * gate * self.net(z)

    def reference_sigma(self) -> torch.Tensor:
        if self.cfg.ref_sigma is None:
            return self.sigma
        return torch.tensor(self.cfg.ref_sigma, dtype=DTYPE)

    def forward(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return bachelier_reference(tau, x, self.reference_sigma()) + self.correction(tau, x)

    def price(self, tau, x) -> np.ndarray:
        """학습된 정규화 가격 v(tau, x)를 numpy 배열로 반환한다 (입력은 브로드캐스트)."""
        tau_b, x_b = np.broadcast_arrays(np.asarray(tau, float), np.asarray(x, float))
        with torch.no_grad():
            v = self(to_tensor(tau_b), to_tensor(x_b))
        return v.numpy().reshape(tau_b.shape)

    # ------------------------------------------------------------- 점 샘플링
    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, 1, generator=self._gen, dtype=DTYPE)

    def _randn(self, n: int) -> torch.Tensor:
        return torch.randn(n, 1, generator=self._gen, dtype=DTYPE)

    def resample(self) -> None:
        c, T = self.cfg, self.T
        n_focus = int(c.focus_frac * c.n_pde)
        n_uni = c.n_pde - n_focus

        s_uni = c.s_min + (1 - c.s_min) * self._rand(n_uni)
        x_uni = self.x_lo + (self.x_hi - self.x_lo) * self._rand(n_uni)
        # ATM 근방에 집중시키되, 해가 가장 빨리 변하는 작은 tau 쪽을 더 촘촘하게 뽑는다
        s_foc = c.s_min + (1 - c.s_min) * self._rand(n_focus) ** 2
        width = 3.0 * c.sigma_init * torch.sqrt(T * s_foc) + 0.02
        x_foc = (1.0 + width * self._randn(n_focus)).clamp(self.x_lo, self.x_hi)

        self.tau_pde = (T * torch.cat([s_uni, s_foc])).requires_grad_(True)
        self.x_pde = torch.cat([x_uni, x_foc]).requires_grad_(True)
        self.tau_bc = T * self._rand(c.n_bc)

    # ----------------------------------------------------------------- 손실
    def losses(self) -> tuple[torch.Tensor, dict[str, float]]:
        c, T, sc = self.cfg, self.T, self.scale
        sigma = self.sigma

        tau, x = self.tau_pde, self.x_pde
        v = self(tau, x)
        v_tau = grad(v, tau)
        v_x = grad(v, x)
        v_xx = grad(v_x, x)
        res = T * (v_tau - 0.5 * sigma**2 * x**2 * v_xx)  # s = tau/T 단위의 잔차
        l_pde = torch.mean((res / sc) ** 2)

        # ATM에서 멀면 옵션 가치는 내재가치와 같다 (오차 ~ e^{-n_std^2/2})
        ones = torch.ones_like(self.tau_bc)
        v_lo = self(self.tau_bc, ones * self.x_lo)
        v_hi = self(self.tau_bc, ones * self.x_hi)
        l_bc = torch.mean(((v_lo - max(self.x_lo - 1.0, 0.0)) / sc) ** 2) + torch.mean(
            ((v_hi - (self.x_hi - 1.0)) / sc) ** 2
        )

        v_d = self(torch.full_like(self.x_data, T), self.x_data)
        l_data = torch.mean(self.w_data * ((v_d - self.v_data) / sc) ** 2)

        total = c.w_pde * l_pde + c.w_bc * l_bc + c.w_data * l_data
        parts = {"pde": l_pde.item(), "bc": l_bc.item(), "data": l_data.item(), "sigma": sigma.item()}
        return total, parts

    # ------------------------------------------------------------------ 학습
    def fit(self, train_cfg: TrainConfig | None = None) -> InverseBSResult:
        t0 = time.perf_counter()
        hist = train(self.parameters(), self.losses, train_cfg or self.cfg.train, self.resample)
        return InverseBSResult(
            T=self.T,
            sigma=float(self.sigma.detach()),
            train_seconds=time.perf_counter() - t0,
            history=pd.DataFrame(hist),
            model=self,
        )

    def residual(self, tau, x) -> np.ndarray:
        """주어진 점에서 스케일된 PDE 잔차 T (v_tau - 1/2 sigma^2 x^2 v_xx) / scale."""
        tau_b, x_b = np.broadcast_arrays(np.asarray(tau, float), np.asarray(x, float))
        t = to_tensor(tau_b, requires_grad=True)
        xx = to_tensor(x_b, requires_grad=True)
        v = self(t, xx)
        v_x = grad(v, xx)
        res = self.T * (grad(v, t) - 0.5 * self.sigma**2 * xx**2 * grad(v_x, xx)) / self.scale
        return res.detach().numpy().reshape(tau_b.shape)

    def warm_start(self, other: "InverseBSPINN") -> None:
        """이미 학습된 다른 모형에서 시작한다 (예: 같은 만기의 전날 학습 결과)."""
        self.load_state_dict(other.state_dict())


def bachelier_price_np(tau, x, sigma) -> np.ndarray:
    """:func:`bachelier_reference`의 numpy 버전 (tau = 0이면 만기 손익)."""
    tau, x = np.broadcast_arrays(np.asarray(tau, float), np.asarray(x, float))
    a = sigma * np.sqrt(np.maximum(tau, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - 1.0) / a
        val = (x - 1.0) * ndtr(z) + a * _INV_SQRT_2PI * np.exp(-0.5 * z * z)
    return np.where(a > 0, val, np.maximum(x - 1.0, 0.0))


def bachelier_only_sigma(T: float, x_data, v_data) -> float:
    """제거 실험: 기준해 Phi만으로 맞춘 sigma (신경망 보정분 = 0).

    sigma의 정확도 중 얼마가 신경망이 블랙-숄즈 PDE를 푼 덕분이 아니라 기준해 덕분인지
    보여 준다.
    """
    x = np.asarray(x_data, float)
    v = np.asarray(v_data, float)
    res = least_squares(
        lambda p: bachelier_price_np(T, x, p[0]) - v, x0=[0.2], bounds=([1e-4], [5.0]), xtol=1e-15, ftol=1e-15
    )
    return float(res.x[0])


def price_error_vs_black(model: InverseBSPINN, sigma_ref: float, n_tau: int = 101, n_x: int = 301) -> dict:
    """학습된 가격함수를 Black 해석식 가격과 비교한다.

    rel_l2      : [0, T] x [x_lo, x_hi] 균일 격자에서 ||v - v_black|| / ||v_black||
    max_abs_atm : tau = T, x in [0.8, 1.2] (데이터가 있는 구간)에서 max |v - v_black|
    """
    tau = np.linspace(0.0, model.T, n_tau)
    x = np.linspace(model.x_lo, model.x_hi, n_x)
    TT, XX = np.meshgrid(tau, x, indexing="ij")
    v_hat = model.price(TT, XX)
    v_true = black_call(XX, sigma_ref**2 * TT)
    rel_l2 = float(np.linalg.norm(v_hat - v_true) / np.linalg.norm(v_true))

    xa = np.linspace(max(0.8, model.x_lo), min(1.2, model.x_hi), 201)
    err_atm = np.abs(model.price(np.full_like(xa, model.T), xa) - black_call(xa, sigma_ref**2 * model.T))
    return {"rel_l2": rel_l2, "max_abs_atm": float(err_atm.max())}


def synthetic_quotes(T: float, sigma: float, k_band: float = 0.1, n: int = 21, noise: float = 0.0, seed: int = 0):
    """로그 머니니스 k in [-k_band, k_band]에서 정규화 콜 가격 v_k = c(x_k, sigma^2 T)를 만든다.

    ``noise``는 정규화 가격 단위의 표준편차로, 그만큼의 정규분포 잡음을 더한다.
    (x, v)를 반환한다.
    """
    k = np.linspace(-k_band, k_band, n)
    x = np.exp(-k)  # x = F/K = e^{-k}
    v = black_call(x, sigma**2 * T)
    if noise > 0:
        v = v + np.random.default_rng(seed).normal(0.0, noise, size=n)
    return x, v
