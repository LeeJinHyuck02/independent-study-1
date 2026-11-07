"""Inverse Black-Scholes PINN: recover sigma from option prices of one maturity.

Problem (forward-normalised Black-Scholes, see ``mfin.engine.bs_analytic``)
----------------------------------------------------------------------------
Unknowns: price function v(tau, x) and volatility sigma > 0.

    PDE        v_tau = 1/2 sigma^2 x^2 v_xx         tau in (0, T], x in (0, x_max)
    initial    v(0, x) = (x - 1)^+
    boundary   v(tau, 0) = 0,  v(tau, x_max) = x_max - 1
    data       v(T, x_k) = v_k   (market prices of maturity T, normalised)

Singularity subtraction
-----------------------
The payoff kink at x = 1 cannot be represented by a smooth tanh network; fitting it
softly smears the kink, which adds time value and biases sigma downward (observed:
-22% on synthetic data). We therefore write

    v(tau, x) = Phi(tau, x; sigma) + scale * s * N_theta(s, x),     s = tau / T

where Phi is the exact solution of the *heat equation* Phi_tau = 1/2 sigma^2 Phi_xx with
the same initial data (the Bachelier price):

    Phi = (x - 1) N(z) + a phi(z),   a = sigma sqrt(tau),   z = (x - 1) / a.

Phi carries the kink, so v(0, x) = (x - 1)^+ holds exactly, and the network only learns
the smooth correction caused by the x^2 factor in the Black-Scholes diffusion. Phi is
not the Black-Scholes solution: the network still has to solve the Black-Scholes PDE,
enforced through the residual loss.

Loss = w_pde L_pde + w_bc L_bc + w_data L_data, each a mean squared residual divided by
``scale`` (the largest option time value in the data). The PDE residual is written in
s units. Because the PDE term forces v to solve Black-Scholes, the forward solution is
learned together with sigma; no separate forward model is needed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from mfin.engine.bs_analytic import black_call
from mfin.pinn.core import DTYPE, MLP, TrainConfig, grad, set_seed, to_tensor, train

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


def bachelier_reference(tau: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Heat-equation solution with initial data (x - 1)^+ (Bachelier price, normal vol sigma)."""
    a = sigma * torch.sqrt(torch.clamp(tau, min=1e-14))
    z = (x - 1.0) / a
    return (x - 1.0) * torch.special.ndtr(z) + a * _INV_SQRT_2PI * torch.exp(-0.5 * z * z)


@dataclass
class InverseBSConfig:
    x_max: float = 3.0
    n_pde: int = 2000
    n_bc: int = 100
    width: int = 64
    depth: int = 4
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_data: float = 10.0
    sigma_init: float = 0.3
    s_min: float = 1e-3  # smallest tau/T used for PDE points
    focus_frac: float = 0.5  # share of PDE points concentrated near the money
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
    """PINN for one maturity T with data (x_k, v_k) at tau = T."""

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

        x = np.asarray(x_data, dtype=float)
        time_value = np.asarray(v_data, dtype=float) - np.maximum(x - 1.0, 0.0)
        self.scale = float(max(time_value.max(), 1e-3))

        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        self.resample()

    # ------------------------------------------------------------------ model
    @property
    def sigma(self) -> torch.Tensor:
        return nn.functional.softplus(self.rho)

    def correction(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s = tau / self.T
        z = torch.cat([2.0 * s - 1.0, 2.0 * x / self.cfg.x_max - 1.0], dim=1)
        return self.scale * s * self.net(z)

    def forward(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return bachelier_reference(tau, x, self.sigma) + self.correction(tau, x)

    def price(self, tau, x) -> np.ndarray:
        """Learned normalised price v(tau, x) as a numpy array (broadcast inputs)."""
        tau_b, x_b = np.broadcast_arrays(np.asarray(tau, float), np.asarray(x, float))
        with torch.no_grad():
            v = self(to_tensor(tau_b), to_tensor(x_b))
        return v.numpy().reshape(tau_b.shape)

    # ------------------------------------------------------------- sampling
    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, 1, generator=self._gen, dtype=DTYPE)

    def _randn(self, n: int) -> torch.Tensor:
        return torch.randn(n, 1, generator=self._gen, dtype=DTYPE)

    def resample(self) -> None:
        c, T = self.cfg, self.T
        n_focus = int(c.focus_frac * c.n_pde)
        n_uni = c.n_pde - n_focus

        s_uni = c.s_min + (1 - c.s_min) * self._rand(n_uni)
        x_uni = c.x_max * self._rand(n_uni)
        # near the money, denser at small tau where the solution changes fastest
        s_foc = c.s_min + (1 - c.s_min) * self._rand(n_focus) ** 2
        width = 3.0 * c.sigma_init * torch.sqrt(T * s_foc) + 0.02
        x_foc = (1.0 + width * self._randn(n_focus)).clamp(0.0, c.x_max)

        self.tau_pde = (T * torch.cat([s_uni, s_foc])).requires_grad_(True)
        self.x_pde = torch.cat([x_uni, x_foc]).requires_grad_(True)
        self.tau_bc = T * self._rand(c.n_bc)

    # ----------------------------------------------------------------- loss
    def losses(self) -> tuple[torch.Tensor, dict[str, float]]:
        c, T, sc = self.cfg, self.T, self.scale
        sigma = self.sigma

        tau, x = self.tau_pde, self.x_pde
        v = self(tau, x)
        v_tau = grad(v, tau)
        v_x = grad(v, x)
        v_xx = grad(v_x, x)
        res = T * (v_tau - 0.5 * sigma**2 * x**2 * v_xx)  # residual in s = tau/T units
        l_pde = torch.mean((res / sc) ** 2)

        zeros = torch.zeros_like(self.tau_bc)
        v_lo = self(self.tau_bc, zeros)
        v_hi = self(self.tau_bc, zeros + c.x_max)
        l_bc = torch.mean((v_lo / sc) ** 2) + torch.mean(((v_hi - (c.x_max - 1.0)) / sc) ** 2)

        v_d = self(torch.full_like(self.x_data, T), self.x_data)
        l_data = torch.mean(self.w_data * ((v_d - self.v_data) / sc) ** 2)

        total = c.w_pde * l_pde + c.w_bc * l_bc + c.w_data * l_data
        parts = {"pde": l_pde.item(), "bc": l_bc.item(), "data": l_data.item(), "sigma": sigma.item()}
        return total, parts

    # ------------------------------------------------------------------ fit
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

    def warm_start(self, other: "InverseBSPINN") -> None:
        """Start from another fitted model (e.g. yesterday's fit of the same maturity)."""
        self.load_state_dict(other.state_dict())


def bachelier_only_sigma(T: float, x_data, v_data) -> float:
    """Ablation: sigma fitted with the reference Phi alone (network correction = 0).

    Shows how much of the accuracy comes from the network solving the PDE rather than
    from the Bachelier reference.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x_data, float)
    v = np.asarray(v_data, float)
    from scipy.special import ndtr

    def phi(sig):
        a = sig * np.sqrt(T)
        z = (x - 1.0) / a
        return (x - 1.0) * ndtr(z) + a * _INV_SQRT_2PI * np.exp(-0.5 * z * z)

    res = least_squares(lambda p: phi(p[0]) - v, x0=[0.2], bounds=([1e-4], [5.0]), xtol=1e-15, ftol=1e-15)
    return float(res.x[0])


def price_error_vs_black(model: InverseBSPINN, sigma_ref: float, n_tau: int = 101, n_x: int = 301) -> dict:
    """Compare the learned price function with the closed-form Black price.

    rel_l2      : ||v - v_black|| / ||v_black|| on a uniform grid of [0, T] x [0, x_max]
    max_abs_atm : max |v - v_black| at tau = T for x in [0.8, 1.2] (where data live)
    """
    tau = np.linspace(0.0, model.T, n_tau)
    x = np.linspace(0.0, model.cfg.x_max, n_x)
    TT, XX = np.meshgrid(tau, x, indexing="ij")
    v_hat = model.price(TT, XX)
    v_true = black_call(XX, sigma_ref**2 * TT)
    rel_l2 = float(np.linalg.norm(v_hat - v_true) / np.linalg.norm(v_true))

    xa = np.linspace(0.8, 1.2, 201)
    err_atm = np.abs(model.price(np.full_like(xa, model.T), xa) - black_call(xa, sigma_ref**2 * model.T))
    return {"rel_l2": rel_l2, "max_abs_atm": float(err_atm.max())}


def synthetic_quotes(T: float, sigma: float, k_band: float = 0.1, n: int = 21, noise: float = 0.0, seed: int = 0):
    """Normalised call prices v_k = c(x_k, sigma^2 T) at log-moneyness k in [-k_band, k_band].

    ``noise`` adds Gaussian noise with that standard deviation in normalised price units.
    Returns (x, v).
    """
    k = np.linspace(-k_band, k_band, n)
    x = np.exp(-k)  # x = F/K = e^{-k}
    v = black_call(x, sigma**2 * T)
    if noise > 0:
        v = v + np.random.default_rng(seed).normal(0.0, noise, size=n)
    return x, v
