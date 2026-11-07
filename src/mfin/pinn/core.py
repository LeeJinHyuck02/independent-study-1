"""PINN building blocks shared by every physics-informed model in the project.

- MLP with tanh activations (smooth, so second derivatives exist everywhere)
- ``grad``: first derivative through autograd, keeping the graph for higher orders
- ``train``: Adam with exponential learning-rate decay, then L-BFGS refinement
All tensors are float64: second derivatives and L-BFGS are noticeably more accurate
than in float32, and the networks are small enough for the CPU.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64

LossFn = Callable[[], tuple[torch.Tensor, dict[str, float]]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(a, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=DTYPE).reshape(-1, 1)
    return t.requires_grad_(requires_grad)


class MLP(nn.Module):
    """Fully connected tanh network: in_dim -> [width] * depth -> out_dim."""

    def __init__(self, in_dim: int, out_dim: int = 1, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            lin = nn.Linear(d, width, dtype=DTYPE)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = width
        out = nn.Linear(d, out_dim, dtype=DTYPE)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx for a column of outputs y w.r.t. a column of inputs x (graph kept)."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


@dataclass
class TrainConfig:
    adam_iters: int = 5000
    lr: float = 1e-3
    lr_final: float = 1e-4
    lbfgs_iters: int = 1500
    resample_every: int = 500
    log_every: int = 250
    verbose: bool = True


def train(
    params: Iterable[torch.nn.Parameter],
    loss_fn: LossFn,
    cfg: TrainConfig,
    resample_fn: Callable[[], None] | None = None,
) -> list[dict]:
    """Minimise ``loss_fn`` with Adam then L-BFGS; returns the logged history.

    ``loss_fn`` returns (total loss tensor, dict of float diagnostics). Collocation
    points are resampled every ``resample_every`` Adam steps and kept fixed during
    L-BFGS (it needs a deterministic objective).
    """
    params = [p for p in params if p.requires_grad]
    history: list[dict] = []
    t0 = time.perf_counter()

    def log(phase: str, it: int, loss: torch.Tensor, parts: dict[str, float]) -> None:
        rec = {"phase": phase, "iter": it, "loss": float(loss.detach()), **parts, "seconds": time.perf_counter() - t0}
        history.append(rec)
        if cfg.verbose:
            extras = "  ".join(f"{k}={v:.3e}" for k, v in parts.items())
            print(f"[{phase:5s} {it:6d}] loss={float(loss.detach()):.3e}  {extras}", flush=True)

    if cfg.adam_iters > 0:
        opt = torch.optim.Adam(params, lr=cfg.lr)
        gamma = (cfg.lr_final / cfg.lr) ** (1.0 / cfg.adam_iters)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        for it in range(cfg.adam_iters):
            if resample_fn is not None and it > 0 and it % cfg.resample_every == 0:
                resample_fn()
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_fn()
            loss.backward()
            opt.step()
            sched.step()
            if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
                log("adam", it, loss, parts)

    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=cfg.lbfgs_iters,
            max_eval=int(cfg.lbfgs_iters * 1.25),
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-16,
            line_search_fn="strong_wolfe",
        )
        n_eval = [0]
        last: list = [None, None]

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss, parts = loss_fn()
            loss.backward()
            if n_eval[0] % cfg.log_every == 0:
                log("lbfgs", n_eval[0], loss, parts)
            n_eval[0] += 1
            last[0], last[1] = loss.detach(), parts
            return loss

        lbfgs.step(closure)
        if last[0] is not None:
            log("lbfgs", n_eval[0], last[0], last[1])
    return history
