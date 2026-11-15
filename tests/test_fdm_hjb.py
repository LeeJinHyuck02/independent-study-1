"""Almgren–Chriss HJB 격자 해법을 Riccati 닫힌 해와 교차검증한다 (n = 1, 2)."""

import numpy as np
import pytest

from mfin.engine.fdm_hjb import FDMConfig, richardson, solve_ac_fdm
from mfin.engine.riccati_ac import ACProblem, rate_closed_form, riccati_closed_form, trajectory_closed_form

ALPHA = 10.0  # 종단 벌점 A_T = alpha Lam (alpha T = 10)
LAM2 = np.diag([1.0, 2.0])
SIG2 = np.array([[4.0, 1.2], [1.2, 1.0]])
Q0_2 = np.array([1.0, -0.6])


def reference(Lam, Sigma, q0):
    p = ACProblem(Lam, Sigma, 1.0, 1.0, ALPHA)
    h = -q0 @ riccati_closed_form(p, [0.0])[0] @ q0
    return p, h, rate_closed_form(p, q0, [0.0])[0]


def solve(Lam, Sigma, N):
    Lam = np.atleast_2d(Lam)
    return solve_ac_fdm(Lam, Sigma, 1.0, 1.0, ALPHA * Lam, FDMConfig(n_points=N))


def test_one_dimensional_first_order_convergence():
    _, h_ex, _ = reference([[1.0]], [[4.0]], np.array([1.0]))
    err = {N: abs(solve([[1.0]], [[4.0]], N).value(0.0, [1.0]) / h_ex - 1) for N in (101, 201, 401)}
    assert err[401] < 0.01
    assert 0.9 <= np.log2(err[201] / err[401]) <= 1.1


def test_two_dimensional_value_rate_and_richardson():
    p, h_ex, nu_ex = reference(LAM2, SIG2, Q0_2)
    r51, r101 = solve(LAM2, SIG2, 51), solve(LAM2, SIG2, 101)
    h51, h101 = r51.value(0.0, Q0_2), r101.value(0.0, Q0_2)
    assert abs(h101 / h_ex - 1) < 0.07
    assert np.linalg.norm(r101.rate(0.0, Q0_2) - nu_ex) / np.linalg.norm(nu_ex) < 0.03
    assert 0.9 <= np.log2(abs(h51 / h_ex - 1) / abs(h101 / h_ex - 1)) <= 1.1
    assert abs(richardson(h51, h101) / h_ex - 1) < 3e-3


def test_two_dimensional_trajectory():
    p, _, _ = reference(LAM2, SIG2, Q0_2)
    t, q = solve(LAM2, SIG2, 101).trajectory(Q0_2, n_steps=1000)
    q_ref = trajectory_closed_form(p, Q0_2, t)
    assert np.max(np.linalg.norm(q - q_ref, axis=1)) / np.linalg.norm(Q0_2) < 0.025


def test_invariants():
    r = solve(LAM2, SIG2, 51)
    assert r.h.max() <= 0.0
    # 비용은 q -> -q에 대해 대칭
    assert np.max(np.abs(r.h - r.h[:, ::-1, ::-1])) < 1e-12
    assert r.t[0] == 0.0 and r.t[-1] == 1.0
    # 종단 스냅샷은 격자에서 정확하고, 격자 밖 점은 3차 스플라인 보간 오차만 남는다
    Q = np.stack(np.meshgrid(*r.grids, indexing="ij"), axis=-1)
    assert r.h[-1] == pytest.approx(-ALPHA * np.einsum("...i,ij,...j->...", Q, LAM2, Q), abs=1e-12)
    assert r.value(1.0, Q0_2) == pytest.approx(-ALPHA * Q0_2 @ LAM2 @ Q0_2, rel=1e-5)


def test_richardson_formula():
    assert richardson(1.0, 1.5) == 2.0
    assert richardson(1.0, 1.75, order=2) == pytest.approx(2.0)


def test_unsupported_inputs_rejected():
    with pytest.raises(ValueError):
        solve_ac_fdm([[1.0, 0.1], [0.1, 1.0]], np.eye(2), 1.0, 1.0, np.eye(2))
    with pytest.raises(ValueError):
        solve_ac_fdm(np.eye(3), np.eye(3), 1.0, 1.0, np.eye(3))
    with pytest.raises(ValueError):
        solve_ac_fdm(np.eye(2), np.eye(2), 1.0, 1.0, np.full((2, 2), np.inf))
