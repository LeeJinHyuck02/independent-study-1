"""모듈 B 엔진 실험: Riccati 차원 확장, FDM 수렴, Merton 위험회피 비교.

    python -m mfin.experiments.module_b --riccati      # n = 1..10, ODE vs 닫힌 해 (오차, 시간)
    python -m mfin.experiments.module_b --fdm          # n = 1, 2 격자 수렴표 + Richardson
    python -m mfin.experiments.module_b --merton       # 실데이터 한 시점, gamma in {2, 5, 10} (캐시/네트워크)
    python -m mfin.experiments.module_b --riccati --fdm --merton

결과: results/module_b/ 에 CSV와 PNG. 12주차 PINN 차원 확장 실험이 같은 문제 설정을 쓴다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mfin.engine.fdm_hjb import FDMConfig, richardson, solve_ac_fdm
from mfin.engine.riccati_ac import (
    ACProblem,
    rate_closed_form,
    riccati_closed_form,
    solve_riccati,
    trajectory_closed_form,
)
from mfin.experiments import _plot
from mfin.paths import RESULTS_DIR

DIMS = [1, 2, 3, 5, 10]
GRID_PER_AXIS = 50  # 구현방안 3.4: FDM 격자 50^n


def ac_problem(n: int, seed: int = 0, alpha: float | None = None) -> ACProblem:
    """차원 확장 실험의 표준 문제 (대각 Lam, 요인 구조 Sigma, phi = 1, T = 1)."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 2))
    Sigma = 0.5 * F @ F.T + np.diag(rng.uniform(0.5, 1.5, n))
    Lam = np.diag(rng.uniform(0.5, 2.0, n))
    return ACProblem(Lam, Sigma, 1.0, 1.0, alpha)


# ------------------------------------------------------------------------------- Riccati


def run_riccati(out: Path) -> pd.DataFrame:
    rows = []
    t = np.linspace(0.0, 1.0, 201)
    for n in DIMS:
        p = ac_problem(n)
        t0 = time.perf_counter()
        A_cf = riccati_closed_form(p, t[:-1])
        cf_s = time.perf_counter() - t0
        sol = solve_riccati(p, t)
        A_ode = sol.A[:-1]
        err = np.max(np.abs(A_ode - A_cf) / np.abs(A_cf).max(axis=(1, 2), keepdims=True))
        rows.append(
            {
                "n": n,
                "state_dim": 1 + n,
                "rel_err_ode_vs_closed": err,
                "ode_seconds": sol.seconds,
                "ode_nfev": sol.nfev,
                "closed_form_seconds": cf_s,
                "fdm_grid_points_50": float(GRID_PER_AXIS) ** n,
                "fdm_memory_bytes_50": 8.0 * float(GRID_PER_AXIS) ** n,
            }
        )
        print(f"n={n:2d}  rel err {err:.2e}  ode {sol.seconds * 1e3:.1f} ms  closed {cf_s * 1e3:.2f} ms", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "riccati_scaling.csv", index=False)

    # n = 1 궤적: 위험회피가 커질수록 sinh 곡선이 앞쪽으로 휜다 (phi -> 0이면 TWAP)
    _plot.setup()
    import matplotlib.pyplot as plt

    tt = np.linspace(0, 1, 201)
    traj = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, kT in enumerate([1e-6, 1.0, 3.0, 8.0]):
        p = ACProblem([[1.0]], [[1.0]], kT**2, 1.0)
        q = trajectory_closed_form(p, [1.0], tt)[:, 0]
        traj[f"kappaT={kT:g}"] = q
        label = "TWAP (kappa T -> 0)" if kT < 1e-3 else f"kappa T = {kT:g}"
        ax.plot(tt, q, color=_plot.SERIES[i], marker=_plot.MARKERS[i], markevery=25, label=label)
    ax.set_xlabel("t / T")
    ax.set_ylabel("remaining order q(t) / q0")
    ax.set_title("Almgren-Chriss optimal liquidation (n = 1)")
    ax.legend()
    _plot.save(fig, out / "riccati_trajectories.png")
    pd.DataFrame(traj, index=pd.Index(tt, name="t")).to_csv(out / "riccati_trajectories.csv")
    return df


# ----------------------------------------------------------------------------------- FDM


def run_fdm(out: Path) -> pd.DataFrame:
    cases = {
        1: (np.array([[1.0]]), np.array([[4.0]]), np.array([1.0]), [101, 201, 401]),
        2: (np.diag([1.0, 2.0]), np.array([[4.0, 1.2], [1.2, 1.0]]), np.array([1.0, -0.6]), [51, 101, 201]),
    }
    alpha = 10.0
    rows = []
    for n, (Lam, Sigma, q0, Ns) in cases.items():
        p = ACProblem(Lam, Sigma, 1.0, 1.0, alpha)
        h_ex = -q0 @ riccati_closed_form(p, [0.0])[0] @ q0
        nu_ex = rate_closed_form(p, q0, [0.0])[0]
        prev = None
        for N in Ns:
            r = solve_ac_fdm(Lam, Sigma, 1.0, 1.0, alpha * Lam, FDMConfig(n_points=N))
            h = float(r.value(0.0, q0))
            nu = r.rate(0.0, q0)
            row = {
                "n": n,
                "N_per_axis": N,
                "dq": float(r.grids[0][1] - r.grids[0][0]),
                "grid_points": N**n,
                "time_steps": r.n_steps,
                "seconds": r.seconds,
                "rel_err_h": abs(h / h_ex - 1),
                "rel_err_nu": float(np.linalg.norm(nu - nu_ex) / np.linalg.norm(nu_ex)),
                "rel_err_richardson": abs(richardson(prev, h) / h_ex - 1) if prev is not None else np.nan,
            }
            rows.append(row)
            prev = h
            print(f"n={n} N={N:4d}  h err {row['rel_err_h']:.3%}  nu err {row['rel_err_nu']:.3%}  "
                  f"richardson {row['rel_err_richardson']:.3%}  {r.seconds:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fdm_convergence.csv", index=False)

    _plot.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (n, g) in enumerate(df.groupby("n")):
        ax.loglog(g["dq"], g["rel_err_h"], color=_plot.SERIES[2 * i], marker=_plot.MARKERS[2 * i],
                  label=f"n = {n}, raw FDM")
        rich = g.dropna(subset=["rel_err_richardson"])
        ax.loglog(rich["dq"], rich["rel_err_richardson"], color=_plot.SERIES[2 * i + 1],
                  marker=_plot.MARKERS[2 * i + 1], label=f"n = {n}, Richardson")
    ref = df[df.n == 1]
    ax.loglog(ref["dq"], ref["rel_err_h"].iloc[0] * ref["dq"] / ref["dq"].iloc[0], color=_plot.TEXT_2,
              linestyle="--", linewidth=1, label="first order")
    ax.set_xlabel("grid spacing dq")
    ax.set_ylabel("relative error of h(0, q0)")
    ax.set_title("Upwind FDM vs Riccati closed form")
    ax.legend()
    _plot.save(fig, out / "fdm_convergence.png")
    return df


# -------------------------------------------------------------------------------- Merton


def run_merton(out: Path, asof: str = "2025-12-31") -> pd.DataFrame:
    from mfin.backtest.rolling import BacktestConfig, load_data
    from mfin.engine.merton import merton_unconstrained, period_inputs, solve_mv_qp
    from mfin.paths import CONFIG_DIR
    from mfin.portfolio.estimators import estimate_state, vix_knots

    cfg = BacktestConfig.from_toml(CONFIG_DIR / "main.toml")
    data = load_data(cfg)
    i = int(np.searchsorted(data.prices.index, pd.Timestamp(asof), side="right"))
    state = estimate_state(data.prices.iloc[:i], cfg.benchmark, float(data.rf.iloc[i - 1]),
                           vix_knots(data.vix.iloc[i - 1]), cfg.est_cfg, vol_source="vix")
    m, S = period_inputs(state.coeffs, 0.0, cfg.horizon)
    rows = []
    for gamma in [2.0, 5.0, 10.0]:
        t0 = time.perf_counter()
        res = solve_mv_qp(m, S, gamma)
        secs = time.perf_counter() - t0
        pu = merton_unconstrained(m, S, gamma)
        for k, tk in enumerate(state.tickers):
            rows.append({"gamma": gamma, "asset": tk, "qp": res.pi[k], "unconstrained": pu[k]})
        rows.append({"gamma": gamma, "asset": "CASH", "qp": 1 - res.pi.sum(), "unconstrained": 1 - pu.sum()})
        print(f"gamma={gamma:g}  cash {1 - res.pi.sum():.1%}  budget active {res.budget_active}  "
              f"kkt {max(res.kkt.values()):.1e}  {secs * 1e3:.2f} ms", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "merton_gamma.csv", index=False)
    print(f"state as of {state.asof.date()}: sigma_fwd(1m) {state.diagnostics['sigma_fwd_1m']:.1%}", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--riccati", action="store_true")
    ap.add_argument("--fdm", action="store_true")
    ap.add_argument("--merton", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "module_b")
    args = ap.parse_args()
    if not (args.riccati or args.fdm or args.merton):
        ap.error("choose at least one of --riccati, --fdm, --merton")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.riccati:
        print("\n=== Riccati: ODE vs closed form ===", flush=True)
        run_riccati(args.out)
    if args.fdm:
        print("\n=== FDM convergence ===", flush=True)
        run_fdm(args.out)
    if args.merton:
        print("\n=== Merton: risk aversion ===", flush=True)
        run_merton(args.out)
    print(f"\nsaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
