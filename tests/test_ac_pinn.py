"""다자산 Almgren–Chriss PINN: 잔차 연산자, 종단조건, 정책 변환, (slow) 학습 정확도."""

import numpy as np
import pytest
import torch

from mfin.engine.riccati_ac import ACProblem, eigen_modes
from mfin.pinn.ac_multi import (
    ACPinn,
    ACPinnConfig,
    eval_points,
    evaluate,
    exact_value_rate,
    hjb_residual,
    integrate_feedback,
    policy_from_value,
    trajectory_metrics,
)
from mfin.pinn.core import DTYPE, TrainConfig


def problem(n, seed=0, alpha=10.0, phi=1.3):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 2))
    Sigma = 0.5 * F @ F.T + np.diag(rng.uniform(0.5, 1.5, n))
    Lam = np.diag(rng.uniform(0.5, 2.0, n))
    return ACProblem(Lam, Sigma, phi, 1.0, alpha)


def tensors(p):
    li = np.linalg.inv(p.Lam)
    return torch.as_tensor(li, dtype=DTYPE), torch.as_tensor(p.Sigma, dtype=DTYPE)


def sample(n, m=200, seed=1):
    rng = np.random.default_rng(seed)
    t = torch.as_tensor(rng.random((m, 1)), dtype=DTYPE).requires_grad_(True)
    q = torch.as_tensor(2 * rng.random((m, n)) - 1, dtype=DTYPE).requires_grad_(True)
    return t, q


def coth_solution(p, tau0=0.3):
    """시간 의존 정확해 A(t) = Lam^{1/2} U diag(kappa coth(kappa (T - t + tau0))) U^T Lam^{1/2}."""
    kappa, U, Lh, _ = eigen_modes(p)
    G = torch.as_tensor(Lh @ U, dtype=DTYPE)
    k = torch.as_tensor(kappa, dtype=DTYPE)

    def h_fn(t, q):
        a = k / torch.tanh(k * (p.T - t + tau0))  # (B, n)
        y = q @ G  # 모드 좌표 G^T q
        return -(a * y * y).sum(dim=1, keepdim=True)

    return h_fn


@pytest.mark.parametrize("n", [1, 3, 6])
def test_residual_vanishes_for_time_dependent_exact_solution(n):
    p = problem(n)
    li, S = tensors(p)
    t, q = sample(n)
    res, _ = hjb_residual(coth_solution(p), t, q, li, S, p.phi)
    assert res.abs().max().item() < 1e-10
    # 부호가 틀린 해(시간 반전)는 잔차가 0이 아니어야 한다 (h_t 부호 검사)
    h = coth_solution(p)
    res_bad, _ = hjb_residual(lambda t, q: h(p.T - t, q), t, q, li, S, p.phi)
    assert res_bad.abs().max().item() > 1e-2


def test_residual_vanishes_for_steady_state_and_policy_matches():
    p = problem(4)
    Lh = np.diag(np.sqrt(np.diag(p.Lam)))
    Lih = np.diag(1 / np.sqrt(np.diag(p.Lam)))
    w, V = np.linalg.eigh(Lih @ (p.phi * p.Sigma) @ Lih)
    A_inf = Lh @ (V * np.sqrt(w)) @ V.T @ Lh  # A Lam^{-1} A = phi Sigma
    At = torch.as_tensor(A_inf, dtype=DTYPE)

    def h_fn(t, q):
        return -((q @ At) * q).sum(dim=1, keepdim=True) + 0.0 * t

    li, S = tensors(p)
    t, q = sample(4)
    res, _ = hjb_residual(h_fn, t, q, li, S, p.phi)
    assert res.abs().max().item() < 1e-10
    nu = policy_from_value(h_fn, t, q, li).detach().numpy()
    assert np.allclose(nu, q.detach().numpy() @ np.linalg.solve(p.Lam, A_inf).T, atol=1e-12)


@pytest.mark.parametrize("gate", ["ref", "linear"])
def test_terminal_condition_exact_before_training(gate):
    p = problem(3)
    m = ACPinn(p, ACPinnConfig(gate=gate, n_pde=64))
    q = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    expected = -p.alpha * np.einsum("bi,ij,bj->b", q, p.Lam, q)
    assert np.allclose(m.value(p.T, q), expected, rtol=1e-13, atol=1e-13)
    # 종단에서의 정책도 정확하다: nu(T, q) = alpha q
    assert np.allclose(m.rate(p.T, q), p.alpha * q, atol=1e-11)


def test_reference_solution_solves_riskless_problem():
    # phi = 0이면 기준해만으로(보정 0) 정확해이고, 잔차도 0이다
    p = problem(2, phi=0.0)
    m = ACPinn(p, ACPinnConfig(n_pde=64))
    torch.nn.init.zeros_(m.net.net[-1].weight)
    torch.nn.init.zeros_(m.net.net[-1].bias)
    q = np.random.default_rng(1).uniform(-1, 1, (40, 2))
    for t in (0.0, 0.5, 0.95):
        h_ex, nu_ex = exact_value_rate(p, t, q)
        assert np.allclose(m.value(t, q), h_ex, rtol=1e-12)
        assert np.allclose(m.rate(t, q), nu_ex, rtol=1e-10)
    assert np.abs(m.residual(np.full(40, 0.3), q)).max() < 1e-10


def test_requires_finite_terminal_penalty():
    with pytest.raises(ValueError):
        ACPinn(problem(2, alpha=None))
    with pytest.raises(ValueError):
        ACPinn(problem(2), ACPinnConfig(gate="other"))


def test_sampling_stays_in_domain_and_covers_origin():
    m = ACPinn(problem(10), ACPinnConfig(n_pde=2000))
    t, q = m.t_pde.detach().numpy(), m.q_pde.detach().numpy()
    assert t.min() >= 0 and t.max() <= 1 and np.abs(q).max() <= 1
    assert np.mean(np.linalg.norm(q, axis=1) < 0.5) > 0.2  # 상자 균일만이면 10차원에서 거의 0
    assert np.mean(t > 0.9) > 0.15  # 만기 근처 집중 (균일이면 0.1, 기대값 0.21)


def test_evaluation_helpers_with_exact_policy():
    p = problem(3)
    q0s, pts = eval_points(p, n_box=100, n_paths=5)
    exact_h = lambda t, q: exact_value_rate(p, t, q)[0]  # noqa: E731
    exact_nu = lambda t, q: exact_value_rate(p, t, q)[1]  # noqa: E731
    err = evaluate(exact_h, exact_nu, p, pts)
    assert err["rel_err_h0"] < 1e-12 and err["rel_err_nu"] < 1e-12
    tm = trajectory_metrics(exact_nu, p, q0s, n_steps=400)
    assert tm["traj_err"] < 1e-3 and abs(tm["excess_cost"]) < 1e-10 and tm["integrator_bias"] < 1e-3
    t, q, nu = integrate_feedback(exact_nu, q0s, p.T, 400)
    assert q.shape == (401, 5, 3) and nu.shape == q.shape


@pytest.mark.slow
def test_pinn_matches_riccati_n2():
    p = problem(2, phi=1.0)
    m = ACPinn(p, ACPinnConfig(train=TrainConfig(adam_iters=3000, lbfgs_iters=1000, resample_every=500,
                                                 verbose=False)))
    m.fit()
    q0s, pts = eval_points(p)
    err = evaluate(m.value, m.rate, p, pts)
    assert err["rel_err_nu"] < 0.01 and err["rel_err_h0"] < 0.01
