"""모듈 B 실험: Riccati 차원 확장, FDM 수렴, Merton 위험회피 비교, PINN 차원 확장.

    python -m mfin.experiments.module_b --riccati      # n = 1..10, ODE vs 닫힌 해 (오차, 시간)
    python -m mfin.experiments.module_b --fdm          # n = 1, 2 격자 수렴표 + Richardson
    python -m mfin.experiments.module_b --merton       # 실데이터 한 시점, gamma in {2, 5, 10} (캐시/네트워크)
    python -m mfin.experiments.module_b --riccati --fdm --merton
    python -m mfin.experiments.module_b --pinn         # n = 1, 2, 3, 5, 10 PINN vs Riccati (CPU 약 1시간)
    python -m mfin.experiments.module_b --pinn --fast  # 반복 수를 줄인 빠른 확인
    python -m mfin.experiments.module_b --pinn --gate linear --seeds 1   # 기준해 없는 형태 (ablation)

결과: results/module_b/ 에 CSV와 PNG.
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


# ---------------------------------------------------------------------------------- PINN

PINN_ALPHA = 10.0  # 종단 벌점 A_T = alpha Lam (FDM과 같은 유한값)


def _pinn_row(method: str, n: int, err: dict, tm: dict, **extra) -> dict:
    row = {"method": method, "n": n, "state_dim": n + 1, **extra, **err, **tm}
    if "rel_err_nu" in err:
        row["pass_1pct"] = bool(err["rel_err_nu"] < 0.01 and err["rel_err_h0"] < 0.01)
    return row


def run_pinn(out: Path, dims=DIMS, seeds: int = 3, fast: bool = False, gate: str = "ref") -> pd.DataFrame:
    """PINN을 n = 1..10에서 학습해 Riccati 닫힌 해와 비교한다. n <= 2는 FDM도 같은 지표로 잰다.

    문제 seed는 0으로 고정하고(ac_problem), seeds는 신경망 초기화·샘플링 seed만 바꾼다.
    """
    from mfin.pinn.ac_multi import (
        ACPinn,
        ACPinnConfig,
        eval_points,
        evaluate,
        integrate_feedback,
        memory_estimate,
        trajectory_metrics,
    )
    from mfin.pinn.core import TrainConfig

    if fast:
        tcfg = TrainConfig(adam_iters=800, lbfgs_iters=300, resample_every=200, log_every=400)
        n_pde = 2048
    else:
        tcfg = TrainConfig(adam_iters=5000, lbfgs_iters=1500, resample_every=500, log_every=1000)
        n_pde = 4096
    suffix = "" if gate == "ref" else f"_{gate}"
    rows, hists, traj = [], [], None

    def save() -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.to_csv(out / f"pinn_scaling{suffix}.csv", index=False)
        if hists:
            pd.concat(hists, ignore_index=True).to_csv(out / f"pinn_history{suffix}.csv", index=False)
        return df

    for n in dims:
        p = ac_problem(n, seed=0, alpha=PINN_ALPHA)
        q0s, pts = eval_points(p)

        # 격자법: n <= 2만 가능. 같은 시험점·지표로 N = 101, 201과 Richardson 외삽을 잰다.
        if n <= 2 and gate == "ref":
            h0_prev = None
            h0_ex = -np.einsum("bi,ij,bj->b", pts[0.0], riccati_closed_form(p, [0.0])[0], pts[0.0])
            for N in (101, 201):
                r = solve_ac_fdm(p.Lam, p.Sigma, p.phi, p.T, PINN_ALPHA * p.Lam, FDMConfig(n_points=N, q_max=1.0))
                err = evaluate(r.value, r.rate, p, pts)
                tm = trajectory_metrics(r.rate, p, q0s)
                h0 = r.value(0.0, pts[0.0])
                if h0_prev is not None:
                    err["rel_err_h0_richardson"] = float(
                        np.linalg.norm(richardson(h0_prev, h0) - h0_ex) / np.linalg.norm(h0_ex))
                h0_prev = h0
                rows.append(_pinn_row(f"fdm_N{N}", n, err, tm, seconds=r.seconds, grid_points=float(N) ** n,
                                      memory_est_bytes=8.0 * float(N) ** n))
                print(f"n={n:2d} FDM N={N}  nu err {err['rel_err_nu']:.2%}  h(0) err {err['rel_err_h0']:.2%}  "
                      f"{r.seconds:.2f}s", flush=True)
        elif gate == "ref":
            rows.append(_pinn_row("fdm_N50_estimate", n, {}, {}, grid_points=float(GRID_PER_AXIS) ** n,
                                  memory_est_bytes=8.0 * float(GRID_PER_AXIS) ** n))

        for s in range(seeds):
            cfg = ACPinnConfig(n_pde=n_pde, gate=gate, seed=s, train=tcfg)
            m = ACPinn(p, cfg)
            print(f"\n--- PINN n={n} seed={s} gate={gate} ({m.n_params()} parameters) ---", flush=True)
            res = m.fit()
            err = evaluate(m.value, m.rate, p, pts)
            tm = trajectory_metrics(m.rate, p, q0s)
            mem = memory_estimate(m.n_params(), n_pde, n + 1, cfg.width, cfg.depth)
            rows.append(_pinn_row(f"pinn_{gate}", n, err, tm, seed=s, seconds=res.train_seconds,
                                  n_params=m.n_params(), n_colloc=n_pde, memory_est_bytes=mem,
                                  final_loss=float(res.history["loss"].iloc[-1])))
            hists.append(res.history.assign(n=n, seed=s, gate=gate))
            print(f"n={n:2d} PINN seed={s}  nu err {err['rel_err_nu']:.2%}  h(0) err {err['rel_err_h0']:.2%}  "
                  f"traj err {tm['traj_err']:.2%}  excess cost {tm['excess_cost']:.1e}  "
                  f"{res.train_seconds:.0f}s", flush=True)
            if n == 5 and s == 0:
                t, q, _ = integrate_feedback(m.rate, q0s[:1], p.T)
                traj = (t, q[:, 0], trajectory_closed_form(p, q0s[0], t))
            save()

    df = save()
    _plot_pinn(df, out, suffix, gate)
    if traj is not None:
        _plot_pinn_trajectory(*traj, out / f"pinn_trajectory{suffix}.png")
    return df


def _plot_pinn(df: pd.DataFrame, out: Path, suffix: str, gate: str) -> None:
    _plot.setup()
    import matplotlib.pyplot as plt

    pinn = df[df.method == f"pinn_{gate}"]
    med = pinn.groupby("n")[["rel_err_nu", "rel_err_h0", "seconds", "memory_est_bytes"]].median()
    fdm = df[df.method == "fdm_N201"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    for i, (col, label) in enumerate([("rel_err_nu", "PINN, rate nu"), ("rel_err_h0", "PINN, cost h(0)")]):
        ax.scatter(pinn["n"], pinn[col], color=_plot.SERIES[i], marker=_plot.MARKERS[i], alpha=0.35, s=18)
        ax.plot(med.index, med[col], color=_plot.SERIES[i], marker=_plot.MARKERS[i], label=label + " (median)")
    if len(fdm):
        ax.plot(fdm["n"], fdm["rel_err_nu"], color=_plot.SERIES[2], marker=_plot.MARKERS[2], linestyle="none",
                markersize=8, label="FDM N=201, rate nu")
        ax.plot(fdm["n"], fdm["rel_err_h0"], color=_plot.SERIES[3], marker=_plot.MARKERS[3], linestyle="none",
                markersize=8, label="FDM N=201, cost h(0)")
    ax.axhline(0.01, color=_plot.TEXT_2, linestyle="--", linewidth=1, label="1% target")
    ax.set_yscale("log")
    ax.set_xlabel("number of assets n (state dimension n + 1)")
    ax.set_ylabel("relative L2 error vs Riccati")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(med.index, med["seconds"], color=_plot.SERIES[0], marker=_plot.MARKERS[0], label="PINN training")
    if len(fdm):
        ax.plot(fdm["n"], fdm["seconds"], color=_plot.SERIES[2], marker=_plot.MARKERS[2], linestyle="none",
                markersize=8, label="FDM N=201")
    ax.set_yscale("log")
    ax.set_xlabel("number of assets n")
    ax.set_ylabel("seconds (CPU)")
    ax.set_title("Compute time")
    ax.legend(fontsize=8)

    ax = axes[2]
    ns = np.array(sorted(df["n"].unique()))
    ax.plot(ns, 8.0 * float(GRID_PER_AXIS) ** ns, color=_plot.SERIES[1], marker=_plot.MARKERS[1], linestyle="--",
            label="FDM, 50 points per axis")
    ax.plot(med.index, med["memory_est_bytes"], color=_plot.SERIES[0], marker=_plot.MARKERS[0],
            label="PINN (estimate)")
    ax.axhline(16e9, color=_plot.TEXT_2, linestyle=":", linewidth=1, label="16 GB laptop RAM")
    ax.set_yscale("log")
    ax.set_xlabel("number of assets n")
    ax.set_ylabel("memory (bytes)")
    ax.set_title("Memory")
    ax.legend(fontsize=8)
    for a in axes:
        a.set_xticks(ns)
    fig.suptitle("Multi-asset Almgren-Chriss HJB: PINN vs grid (exact Riccati reference)", fontsize=11)
    _plot.save(fig, out / f"pinn_scaling{suffix}.png")


def _plot_pinn_trajectory(t: np.ndarray, q_pinn: np.ndarray, q_exact: np.ndarray, path: Path) -> None:
    _plot.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6, 4))
    scale = np.abs(q_exact[0]).max()
    every = max(1, len(t) // 12)
    for i in range(q_pinn.shape[1]):
        c = _plot.SERIES[i % len(_plot.SERIES)]
        ax.plot(t, q_exact[:, i] / scale, color=c, linewidth=1.5)
        ax.plot(t, q_pinn[:, i] / scale, color=c, linestyle="none", marker=_plot.MARKERS[i % len(_plot.MARKERS)],
                markevery=every)
    ax.axhline(0, color=_plot.GRID, linewidth=1)
    ax.set_xlabel("t / T")
    ax.set_ylabel("remaining order q_i(t) / max |q0|")
    ax.set_title("Optimal execution, n = 5 (one color per asset)")
    ax.legend(handles=[Line2D([], [], color=_plot.TEXT_2, linewidth=1.5, label="Riccati (exact)"),
                       Line2D([], [], color=_plot.TEXT_2, marker="o", linestyle="none", label="PINN")], fontsize=8)
    _plot.save(fig, path)


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
    ap.add_argument("--pinn", action="store_true", help="PINN dimension-scaling experiment (slow)")
    ap.add_argument("--dims", default=",".join(map(str, DIMS)), help="comma-separated n for --pinn")
    ap.add_argument("--seeds", type=int, default=3, help="network seeds per n for --pinn")
    ap.add_argument("--fast", action="store_true", help="fewer training iterations for --pinn")
    ap.add_argument("--gate", choices=["ref", "linear"], default="ref",
                    help="PINN terminal form: reference solution + correction (ref) or linear gate only")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "module_b")
    args = ap.parse_args()
    if not (args.riccati or args.fdm or args.merton or args.pinn):
        ap.error("choose at least one of --riccati, --fdm, --merton, --pinn")
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
    if args.pinn:
        print("\n=== PINN dimension scaling ===", flush=True)
        dims = [int(d) for d in args.dims.split(",") if d.strip()]
        run_pinn(args.out, dims, args.seeds, args.fast, args.gate)
    print(f"\nsaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
