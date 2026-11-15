"""다자산 Almgren–Chriss Riccati 엔진: 해석해, ODE, 비용 항등식."""

import numpy as np
import pytest

from mfin.engine.riccati_ac import (
    ACProblem,
    optimal_cost,
    rate_closed_form,
    riccati_closed_form,
    riccati_inverse_closed_form,
    schedule_cost,
    solve_riccati,
    trajectory_closed_form,
)


def random_problem(n, seed=0, phi=2.0, T=1.0, alpha=None):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, n))
    Sigma = F @ F.T / n + 0.2 * np.eye(n)
    Lam = np.diag(rng.uniform(0.5, 2.0, n))
    return ACProblem(Lam, Sigma, phi, T, alpha)


def test_riccati_vs_sinh():
    k, sigma2, phi, T = 0.7, 1.8, 2.5, 1.0
    p = ACProblem([[k]], [[sigma2]], phi, T)
    kappa = np.sqrt(phi * sigma2 / k)
    t = np.linspace(0, 0.99 * T, 50)
    A = riccati_closed_form(p, t)[:, 0, 0]
    exact = np.sqrt(k * phi * sigma2) / np.tanh(kappa * (T - t))
    assert np.max(np.abs(A / exact - 1)) < 1e-12
    num = solve_riccati(p, t).A[:, 0, 0]
    assert np.max(np.abs(num / exact - 1)) < 1e-8
    t = np.linspace(0, T, 101)
    q = trajectory_closed_form(p, [1.0], t)[:, 0]
    assert q == pytest.approx(np.sinh(kappa * (T - t)) / np.sinh(kappa * T), abs=1e-12)


def test_ode_matches_closed_form_full_liquidation():
    p = random_problem(3)
    t = np.linspace(0, p.T, 41)
    sol = solve_riccati(p, t)
    B = riccati_inverse_closed_form(p, t)
    assert np.max(np.abs(sol.B - B)) / np.max(np.abs(B)) < 1e-9
    A = riccati_closed_form(p, t[:-1])
    assert np.max(np.abs(sol.A[:-1] - A) / np.abs(A).max(axis=(1, 2), keepdims=True)) < 1e-9


def test_ode_matches_closed_form_finite_penalty():
    p = random_problem(3, seed=1, alpha=4.0)
    t = np.linspace(0, p.T, 41)
    sol = solve_riccati(p, t)
    A = riccati_closed_form(p, t)
    assert np.max(np.abs(sol.A - A)) / np.max(np.abs(A)) < 1e-9
    assert A[-1] == pytest.approx(4.0 * p.Lam, abs=1e-14)


def test_riccati_solution_symmetric_positive_definite():
    p = random_problem(4, seed=2)
    for A in riccati_closed_form(p, np.linspace(0, 0.95, 20)):
        assert np.max(np.abs(A - A.T)) < 1e-12
        assert np.linalg.eigvalsh(A).min() > 0


@pytest.mark.parametrize("alpha", [None, 3.0])
def test_optimal_cost_equals_trajectory_cost(alpha):
    p = random_problem(3, seed=3, alpha=alpha)
    q0 = np.array([1.0, -0.6, 0.4])
    t = np.linspace(0, p.T, 20001)
    q = trajectory_closed_form(p, q0, t)
    nu = rate_closed_form(p, q0, t)
    cost = schedule_cost(p, t, q, nu)
    assert cost["total"] == pytest.approx(optimal_cost(p, q0), rel=1e-8)


def test_rate_is_minus_derivative_of_trajectory():
    p = random_problem(3, seed=4, alpha=5.0)
    q0 = np.array([0.5, 1.0, -0.3])
    t = np.linspace(0.1, 0.9, 9)
    h = 1e-6
    dq = (trajectory_closed_form(p, q0, t + h) - trajectory_closed_form(p, q0, t - h)) / (2 * h)
    assert rate_closed_form(p, q0, t) == pytest.approx(-dq, rel=1e-7, abs=1e-9)


def test_perturbation_increases_cost_quadratically():
    p = random_problem(3, seed=5)
    q0 = np.array([1.0, 0.5, -0.5])
    t = np.linspace(0, p.T, 20001)
    q = trajectory_closed_form(p, q0, t)
    nu = rate_closed_form(p, q0, t)
    v = np.array([1.0, -2.0, 0.5])
    base = schedule_cost(p, t, q, nu)["total"]
    dJ = {}
    for eps in [1e-2, -1e-2, 1e-3, -1e-3]:
        # 끝점이 고정된 섭동 eps sin(pi t / T) v
        qp = q + eps * np.sin(np.pi * t / p.T)[:, None] * v
        nup = nu - eps * (np.pi / p.T) * np.cos(np.pi * t / p.T)[:, None] * v
        dJ[eps] = schedule_cost(p, t, qp, nup)["total"] - base
        assert dJ[eps] > 0
    assert dJ[1e-2] / dJ[1e-3] == pytest.approx(100.0, rel=1e-3)


def test_diagonal_problem_separates():
    Lam = np.diag([0.5, 2.0])
    Sigma = np.diag([1.0, 3.0])
    p = ACProblem(Lam, Sigma, 1.5, 2.0)
    t = np.array([0.0, 0.7, 1.5])
    A = riccati_closed_form(p, t)
    for i in range(2):
        k, s2 = Lam[i, i], Sigma[i, i]
        kap = np.sqrt(1.5 * s2 / k)
        assert A[:, i, i] == pytest.approx(np.sqrt(k * 1.5 * s2) / np.tanh(kap * (2.0 - t)), rel=1e-12)
    assert np.max(np.abs(A[:, 0, 1])) < 1e-12


def test_limits():
    # phi -> 0: TWAP 직선
    p = ACProblem([[1.0]], [[1.0]], 1e-12, 1.0)
    t = np.linspace(0, 1, 11)
    assert trajectory_closed_form(p, [1.0], t)[:, 0] == pytest.approx(1 - t, abs=1e-6)
    # kappa T = 1000: overflow 없이 지수 감쇠
    p = ACProblem([[1.0]], [[1e6]], 1.0, 1.0)
    q = trajectory_closed_form(p, [1.0], [0.01])[0, 0]
    assert np.isfinite(q) and q == pytest.approx(np.exp(-1000 * 0.01), rel=1e-10)
    # tau -> 0: A ~ Lam / tau
    p = random_problem(2, seed=6)
    t = p.T - 1e-12
    tau = p.T - t  # 부동소수점 상쇄를 포함한 실제 tau
    A = riccati_closed_form(p, [t])[0]
    assert A * tau == pytest.approx(p.Lam, rel=1e-6, abs=1e-9)


def test_invalid_problem_rejected():
    with pytest.raises(ValueError):
        ACProblem([[1.0, 0.2], [0.0, 1.0]], np.eye(2), 1.0, 1.0)
    with pytest.raises(ValueError):
        ACProblem(np.eye(2), np.eye(2), 1.0, -1.0)
    with pytest.raises(ValueError):
        riccati_closed_form(ACProblem(np.eye(2), np.eye(2), 1.0, 1.0), [1.5])
