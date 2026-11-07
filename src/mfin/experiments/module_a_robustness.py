"""실데이터에서 역방향 PINN이 최소제곱과 어긋나는 이유는 무엇인가?

2026-11-06 SPX 체인에서 PINN의 sigma가 *같은* 시세에 맞춘 해석식 최소제곱 sigma보다 최대
3.6 vol pt 높았다. 깨끗한 합성 데이터에서는 둘이 일치했다. 실제 시세는 상수 sigma 모형과
두 가지 점에서 다르다. 잡음(오래된 체결가, 약 1-2.5 vol pt)과 smile(적합 구간 안에서 로그
머니니스 1단위당 -0.5 ~ -1.7의 skew)이다. 이 실험은 두 원인을 분리한다.

    python -m mfin.experiments.module_a_robustness --phase diagnose
    python -m mfin.experiments.module_a_robustness --phase weights

diagnose : 실데이터와 비슷한 세 만기에서 clean / noise / smile / noise+smile 시나리오를
           이전 기본 가중치(w_data = 10)로 돌린다. PINN - LSQ(vol pt), 데이터 적합 오차,
           만기 근처의 PDE 잔차(신경망이 블랙-숄즈에서 얼마나 벗어나는지)를 보고한다.
weights  : 같은 시나리오를 더 작은 데이터 가중치(물리 항 강화)로 돌려 해결책을 시험한다.
결과: results/module_a/robustness/<phase>.csv
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from mfin.paths import RESULTS_DIR

# 2026-11-06 체인에서 잰 실데이터 유사 설정 (T, 구간 안 시세 수, 구간 폭, skew 기울기)
MATURITIES = [
    {"T": 0.0274, "n": 31, "band": 0.020, "slope": -1.7},
    {"T": 0.2301, "n": 91, "band": 0.042, "slope": -0.5},
    {"T": 0.9396, "n": 45, "band": 0.078, "slope": -0.5},
]
SCENARIOS = {
    "clean": (False, False),
    "noise": (True, False),
    "smile": (False, True),
    "noise+smile": (True, True),
}
SIGMA0 = 0.19
NOISE_REL = 0.06  # 가격 잡음 / ATM 시간가치 (실데이터: 4-9%)


def make_quotes(T, n, band, slope, noise, smile, seed=0):
    from mfin.engine.bs_analytic import black_call

    k = np.linspace(-band, band, n)
    x = np.exp(-k)
    sig = SIGMA0 + (slope * k if smile else 0.0)
    v = black_call(x, sig**2 * T)
    if noise:
        tv_atm = float(black_call(1.0, SIGMA0**2 * T))
        v = v + np.random.default_rng(seed).normal(0.0, NOISE_REL * tv_atm, size=n)
    return x, v


def run_case(case: dict) -> dict:
    import torch

    torch.set_num_threads(case.get("threads", 2))
    from mfin.engine.bs_analytic import black_call
    from mfin.engine.implied_vol import fit_constant_vol
    from mfin.pinn.bs_inverse import InverseBSConfig, InverseBSPINN
    from mfin.pinn.core import TrainConfig

    T = case["T"]
    x, v = make_quotes(T, case["n"], case["band"], case["slope"], *SCENARIOS[case["scenario"]], seed=case["seed"])
    sig_lsq = fit_constant_vol(x, v, T, sigma0=SIGMA0)
    train = TrainConfig(adam_iters=2000, lbfgs_iters=1000, log_every=10**9, verbose=False)
    if case.get("fast"):
        train = TrainConfig(adam_iters=1000, lbfgs_iters=500, log_every=10**9, verbose=False)
    cfg = InverseBSConfig(w_pde=case["w_pde"], w_data=case["w_data"], train=train)
    m = InverseBSPINN(T, x, v, cfg=cfg)
    res = m.fit()

    tv = float((v - np.maximum(x - 1, 0)).max())
    fit_pinn = m.price(np.full_like(x, T), x)
    rmse_pinn = float(np.sqrt(np.mean((fit_pinn - v) ** 2)) / tv)
    rmse_lsq = float(np.sqrt(np.mean((black_call(x, sig_lsq**2 * T) - v) ** 2)) / tv)
    # 데이터 구간에서, 시간 구간의 마지막 10% 동안의 PDE 잔차
    tau = np.linspace(0.9 * T, T, 21)
    xs = np.linspace(x.min(), x.max(), 41)
    TT, XX = np.meshgrid(tau, xs, indexing="ij")
    res_near_T = float(np.sqrt(np.mean(m.residual(TT, XX) ** 2)))
    tau_mid = np.linspace(0.2 * T, 0.8 * T, 21)
    TT, XX = np.meshgrid(tau_mid, xs, indexing="ij")
    res_mid = float(np.sqrt(np.mean(m.residual(TT, XX) ** 2)))

    out = {
        **{k: case[k] for k in ("scenario", "T", "n", "band", "w_pde", "w_data", "seed")},
        "sigma_lsq": sig_lsq,
        "sigma_pinn": res.sigma,
        "pinn_minus_lsq_volpt": 100 * (res.sigma - sig_lsq),
        "data_rmse_rel_pinn": rmse_pinn,
        "data_rmse_rel_lsq": rmse_lsq,
        "pde_res_near_T": res_near_T,
        "pde_res_mid": res_mid,
        "train_seconds": res.train_seconds,
    }
    print(f"{case['scenario']:12s} T={T:.4f} w_data={case['w_data']:<5g} "
          f"PINN-LSQ={out['pinn_minus_lsq_volpt']:+.2f} volpt  resT={res_near_T:.2e}", flush=True)
    return out


def cases(phase: str, fast: bool) -> list[dict]:
    if phase == "diagnose":
        weights = [(1.0, 10.0)]
        scen = list(SCENARIOS)
    else:  # weights: 데이터보다 물리 항을 강화
        weights = [(1.0, 1.0), (1.0, 0.1)]
        scen = ["clean", "noise+smile"]
    return [
        {**m, "scenario": s, "w_pde": wp, "w_data": wd, "seed": 0, "fast": fast}
        for (wp, wd) in weights
        for s in scen
        for m in MATURITIES
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["diagnose", "weights"], required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "module_a" / "robustness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    todo = cases(args.phase, args.fast)
    threads = max(1, (os.cpu_count() or 8) // (2 * args.workers))
    for c in todo:
        c["threads"] = threads
    with Pool(args.workers) as pool:
        rows = pool.map(run_case, todo, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"{args.phase}.csv", index=False)
    cols = ["scenario", "T", "w_data", "sigma_lsq", "sigma_pinn", "pinn_minus_lsq_volpt",
            "data_rmse_rel_pinn", "data_rmse_rel_lsq", "pde_res_near_T", "pde_res_mid", "train_seconds"]
    print("\n" + df[cols].round(4).to_string(index=False))
    print(f"\nsaved to {args.out / (args.phase + '.csv')}")


if __name__ == "__main__":
    main()
