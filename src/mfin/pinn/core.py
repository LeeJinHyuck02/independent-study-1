"""프로젝트의 모든 물리 기반 신경망(PINN)이 함께 쓰는 구성 요소.

- tanh 활성화 MLP (매끄러워서 2계 도함수가 어디서나 존재한다)
- ``grad``: autograd로 1계 도함수를 구하되, 고계 도함수를 위해 계산 그래프를 유지한다
- ``train``: 학습률을 지수적으로 줄이는 Adam 후 L-BFGS로 미세조정
모든 텐서는 float64다. 2계 도함수와 L-BFGS가 float32보다 눈에 띄게 정확하고, 신경망이
작아서 CPU로도 충분하다.
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
    """완전연결 tanh 신경망: in_dim -> [width] * depth -> out_dim."""

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
    """출력 열벡터 y를 입력 열벡터 x로 미분한 dy/dx (계산 그래프 유지)."""
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
    """Adam 후 L-BFGS로 ``loss_fn``을 최소화하고 기록된 학습 이력을 반환한다.

    ``loss_fn``은 (전체 손실 텐서, 진단용 float 값의 dict)를 반환한다. 콜로케이션 점은
    Adam ``resample_every`` 단계마다 새로 뽑고, L-BFGS 동안에는 고정한다(L-BFGS는 목적함수가
    결정적이어야 한다).
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
