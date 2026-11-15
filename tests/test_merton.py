"""Merton HJB 엔진: 제약 QP의 정확성, 구간 평균, 가치함수."""

import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize

from mfin.engine.merton import (
    MertonCoeffs,
    integrated_hamiltonian,
    kkt_residuals,
    merton_policy,
    merton_policy_period,
    merton_unconstrained,
    merton_value,
    period_inputs,
    solve_mv_qp,
)


def random_problem(rng, n):
    vol = rng.uniform(0.1, 0.4, n)
    F = rng.normal(size=(n, 2))
    corr = F @ F.T + np.diag(rng.uniform(0.3, 1.0, n))
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    Sigma = np.outer(vol, vol) * corr
    m = rng.uniform(-0.03, 0.12, n)
    return m, Sigma


def brute_force_qp(m, Sigma, gamma, budget=1.0):
    """모든 자유 집합과 예산 상태를 열거해 실현 가능한 KKT 후보 중 최댓값을 고른다."""
    n = len(m)
    best, best_obj = np.zeros(n), 0.0
    for k in range(1, n + 1):
        for F in itertools.combinations(range(n), k):
            F = list(F)
            S = gamma * Sigma[np.ix_(F, F)]
            cands = [np.linalg.solve(S, m[F])]
            K = np.block([[S, np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            cands.append(np.linalg.solve(K, np.concatenate([m[F], [budget]]))[:k])
            for c in cands:
                if np.all(c >= -1e-14) and c.sum() <= budget + 1e-12:
                    pi = np.zeros(n)
                    pi[F] = c
                    obj = m @ pi - 0.5 * gamma * pi @ Sigma @ pi
                    if obj > best_obj:
                        best, best_obj = pi, obj
    return best


def make_coeffs(kappa=1.4):
    s = np.array([1.2, 0.9, 1.0])
    R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    mu0 = np.array([0.10, 0.07, 0.05])
    mu_inf = np.array([0.07, 0.06, 0.06])
    knots = np.array([0.0, 0.25, 0.5, 1.0])
    fwd_var = np.array([0.04, 0.0225, 0.03])
    return MertonCoeffs(0.03, s, R, mu0, mu_inf, kappa, knots, fwd_var)


# ---------------------------------------------------------------------------------- QP


def test_unconstrained_solves_first_order_condition():
    rng = np.random.default_rng(0)
    m, Sigma = random_problem(rng, 5)
    pi = merton_unconstrained(m, Sigma, 3.0)
    assert np.allclose(3.0 * Sigma @ pi, m, atol=1e-12)


def test_interior_solution_matches_analytic():
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    m = np.array([0.03, 0.04])
    gamma = 5.0
    pi_u = merton_unconstrained(m, Sigma, gamma)
    assert np.all(pi_u > 0) and pi_u.sum() < 1
    res = solve_mv_qp(m, Sigma, gamma)
    assert res.pi == pytest.approx(pi_u, abs=1e-12)
    assert res.lam == 0.0 and not res.budget_active
    assert np.max(np.abs(res.eta)) < 1e-12


def test_budget_binding_hand_computed():
    # 대칭 문제: 무제약 해 (2.5, 2.5)가 예산을 넘으므로 (0.5, 0.5), lam = 0.1 - 0.04 * 0.5 = 0.08
    res = solve_mv_qp(np.array([0.1, 0.1]), 0.04 * np.eye(2), 1.0)
    assert res.pi == pytest.approx([0.5, 0.5], abs=1e-14)
    assert res.lam == pytest.approx(0.08, abs=1e-14)
    assert res.budget_active


def test_kkt_on_random_instances():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 12))
        m, Sigma = random_problem(rng, n)
        gamma = float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0]))
        res = solve_mv_qp(m, Sigma, gamma)
        assert max(res.kkt.values()) <= 1e-10, res.kkt
        assert res.pi.min() >= -1e-15
        assert res.pi.sum() <= 1 + 1e-12


def test_matches_brute_force_enumeration():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(2, 8))
        m, Sigma = random_problem(rng, n)
        gamma = float(rng.choice([1.0, 2.0, 5.0]))
        res = solve_mv_qp(m, Sigma, gamma)
        assert np.max(np.abs(res.pi - brute_force_qp(m, Sigma, gamma))) <= 1e-10


def test_limits():
    rng = np.random.default_rng(3)
    _, Sigma = random_problem(rng, 4)
    # 초과수익이 모두 음수면 전액 현금
    assert np.all(solve_mv_qp(-np.abs(rng.normal(size=4)), Sigma, 2.0).pi == 0.0)
    m = np.array([0.02, 0.08, 0.05, 0.01])
    # 위험회피가 0에 가까우면 초과수익이 가장 큰 자산에 전액
    assert solve_mv_qp(m, Sigma, 1e-4).pi == pytest.approx([0, 1, 0, 0], abs=1e-6)
    # 위험회피가 매우 크면 거의 현금
    assert np.linalg.norm(solve_mv_qp(m, Sigma, 1e6).pi) < 1e-5


def test_kkt_residuals_detect_suboptimal_point():
    m, Sigma = np.array([0.1, 0.1]), 0.04 * np.eye(2)
    res = kkt_residuals(np.array([0.3, 0.3]), 0.0, m, Sigma, 1.0)
    assert res["stationarity"] > 1e-3


# -------------------------------------------------------------------------- 계수와 구간 평균


def test_mean_cov_exact():
    c = MertonCoeffs(0.0, [1.0], [[1.0]], [0.1], [0.05], 1.0, [0.0, 0.5, 1.0], [0.01, 0.09])
    assert c.mean_cov(0.0, 1.0) == pytest.approx(0.05 * c.C, abs=1e-14)
    assert c.total_var(0.75) == pytest.approx(0.005 + 0.25 * 0.09, abs=1e-15)
    # 마지막 마디 이후는 마지막 선도분산으로 이어진다
    assert c.total_var(2.0) == pytest.approx(0.05 + 0.09, abs=1e-15)


def test_mean_excess_mu_matches_quadrature():
    c = make_coeffs()
    t0, dt = 0.2, 1 / 12
    num = np.array([quad(lambda s, i=i: c.excess_mu(s)[i], t0, t0 + dt, epsabs=1e-15)[0] for i in range(c.n)]) / dt
    assert c.mean_excess_mu(t0, dt) == pytest.approx(num, abs=1e-12)
    # kappa = 0이면 상수
    c0 = make_coeffs(kappa=0.0)
    assert c0.mean_excess_mu(0.3, 0.5) == pytest.approx(c0.mu0 - c0.r, abs=1e-15)


def test_mean_var_matches_quadrature_across_knots():
    c = make_coeffs()
    num = quad(lambda s: c.fwd_var_at(s), 0.1, 0.7, points=[0.25, 0.5], epsabs=1e-14)[0] / 0.6
    assert c.mean_var(0.1, 0.6) == pytest.approx(num, abs=1e-12)


def test_period_policy_maximizes_exact_crra_certainty_equivalent():
    # 비중을 [t0, t0+dt] 동안 고정하면 log 부 ~ N(M, S2):
    # 확실성등가 성장률 = M + (1-gamma) S2 / 2
    c = make_coeffs()
    t0, dt, gamma = 0.2, 0.5, 4.0
    pts = [k for k in c.var_knots if t0 < k < t0 + dt]

    def ce(pi):
        M = quad(lambda s: c.r + pi @ c.excess_mu(s) - 0.5 * pi @ c.cov(s) @ pi, t0, t0 + dt, points=pts)[0]
        S2 = quad(lambda s: pi @ c.cov(s) @ pi, t0, t0 + dt, points=pts)[0]
        return M + 0.5 * (1 - gamma) * S2

    pi0 = merton_policy_period(c, t0, dt, gamma)
    cons = [{"type": "ineq", "fun": lambda p: 1 - p.sum()}]
    opt = minimize(lambda p: -ce(p), pi0, bounds=[(0, None)] * c.n, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 200})
    assert -opt.fun - ce(pi0) < 1e-10
    assert np.max(np.abs(opt.x - pi0)) < 1e-6


def test_period_policy_equals_instantaneous_for_constant_coefficients():
    c = MertonCoeffs.flat(0.02, [1.0, 1.3], [[1.0, 0.5], [0.5, 1.0]], [0.06, 0.08], [0.0, 0.0], 0.0, 0.2)
    assert merton_policy_period(c, 0.0, 1 / 12, 5.0) == pytest.approx(merton_policy(c, 0.3, 5.0), abs=1e-14)


def test_period_inputs_shapes():
    c = make_coeffs()
    m, S = period_inputs(c, 0.0, 1 / 12)
    assert m.shape == (3,) and S.shape == (3, 3)
    assert c.branch_vector(5.0).shape == (10 + 3 + 3 + 3 + 1 + 3 + 1 + 1,)


def test_invalid_coefficients_rejected():
    with pytest.raises(ValueError):
        MertonCoeffs(0.0, [1.0], [[1.0]], [0.1], [0.05], 1.0, [0.1, 0.5], [0.01])
    with pytest.raises(ValueError):
        MertonCoeffs(0.0, [1.0], [[1.0]], [0.1], [0.05], 1.0, [0.0, 0.5], [-0.01])


# -------------------------------------------------------------------------------- 가치함수


def _hjb_residual(c, t, x, T, gamma, h=1e-4):
    V = lambda tt, xx: merton_value(c, tt, xx, T, gamma)  # noqa: E731
    Vt = (V(t + h, x) - V(t - h, x)) / (2 * h)
    Vx = (V(t, x + h) - V(t, x - h)) / (2 * h)
    Vxx = (V(t, x + h) - 2 * V(t, x) + V(t, x - h)) / h**2
    m, S = c.excess_mu(t), c.cov(t)
    pi = merton_unconstrained(m, S, gamma)
    H = (c.r + pi @ m) * Vx + 0.5 * pi @ S @ pi * (Vxx - Vx)
    return (Vt + H) / abs(Vt)


def test_value_terminal_condition():
    c = make_coeffs()
    assert merton_value(c, 1.5, 0.3, 1.5, 3.0) == pytest.approx(np.exp(-2 * 0.3) / -2, abs=1e-14)


@pytest.mark.parametrize("gamma", [0.5, 3.0, 1.0])
def test_value_satisfies_hjb(gamma):
    # 마디점(0.25, 0.5, 1.0)을 피해 t = 0.6에서 잰다
    c = make_coeffs()
    assert abs(_hjb_residual(c, 0.6, 0.3, 1.5, gamma)) < 1e-6


def test_constrained_value_equals_closed_form_when_inactive():
    # 위험회피가 커서 제약이 걸리지 않는 경우: Gauss–Legendre 적분 == 닫힌 형태
    base = make_coeffs()
    c = MertonCoeffs(base.r, base.s, base.R, [0.10, 0.08, 0.08], [0.07, 0.07, 0.07], base.kappa,
                     base.var_knots, base.fwd_var)
    gamma = 10.0
    for s in np.linspace(0, 2, 9):
        pi = merton_unconstrained(c.excess_mu(s), c.cov(s), gamma)
        assert np.all(pi > 0) and pi.sum() < 1
    a = integrated_hamiltonian(c, 0.1, 2.0, gamma, constrained=False)
    b = integrated_hamiltonian(c, 0.1, 2.0, gamma, constrained=True)
    assert b == pytest.approx(a, rel=1e-10)


def test_constraint_lowers_value_growth():
    c = make_coeffs()
    gamma = 1.0  # 무제약 해가 레버리지를 쓰는 경우
    assert integrated_hamiltonian(c, 0.0, 1.0, gamma, True) < integrated_hamiltonian(c, 0.0, 1.0, gamma, False)
