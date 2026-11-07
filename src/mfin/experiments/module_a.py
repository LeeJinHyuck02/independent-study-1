"""모듈 A 실험: 역방향 블랙-숄즈 PINN과 해석식 엔진의 비교.

    python -m mfin.experiments.module_a --synthetic [--fast]
    python -m mfin.experiments.module_a --live [--fast] [--n-pinn 5]

--synthetic : 참값 sigma = 0.2, 만기 0.1 / 0.5 / 1.0. 만기마다 다음을 비교한다.
              PINN(학습형 기준해), PINN(고정 기준해, sigma는 PDE로만 정해짐),
              Bachelier 기준해만 쓴 제거 실험, 해석식 최소제곱.
--live      : SPX 옵션체인 -> Brent smile / ATM 기간구조 / skew, 몇 개 만기에 역방향 PINN,
              선도변동성 곡선(Brent, PINN, VIX).
결과(CSV + PNG)는 results/module_a/<synthetic | live_YYYYMMDD>/에 저장한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mfin.engine.bs_analytic import black_call
from mfin.engine.implied_vol import fit_constant_vol, smile
from mfin.engine.vol_term import DEFAULT_SENSORS, forward_vol_curve, term_structure, vix_forward_vol_curve
from mfin.experiments import _plot
from mfin.paths import RESULTS_DIR
from mfin.pinn.bs_inverse import (
    InverseBSConfig,
    InverseBSPINN,
    bachelier_only_sigma,
    bachelier_price_np,
    price_error_vs_black,
    synthetic_quotes,
)
from mfin.pinn.core import TrainConfig

# 변형 -> (기준해 sigma, 데이터 가중치). 고정 기준해는 데이터 + PDE로만 sigma를 정하므로
# 큰 데이터 가중치를 유지한다 (0.1이면 sigma가 초기값 근처에 머문다).
VARIANTS = {"trainable_ref": (None, 0.1), "fixed_ref": (0.3, 10.0)}


def _train_cfg(fast: bool) -> TrainConfig:
    if fast:
        return TrainConfig(adam_iters=1000, lbfgs_iters=500, log_every=500)
    return TrainConfig(adam_iters=2000, lbfgs_iters=1000, log_every=250)


def _step_index(hist: pd.DataFrame) -> np.ndarray:
    """전체 최적화 단계 번호: Adam 반복 다음에 L-BFGS 평가 횟수를 이어 붙인다."""
    adam = hist["phase"] == "adam"
    n_adam = int(hist.loc[adam, "iter"].max()) + 1 if adam.any() else 0
    return np.where(adam, hist["iter"], n_adam + hist["iter"])


# --------------------------------------------------------------------------- 합성 데이터
def run_synthetic(out: Path, fast: bool, sigma: float = 0.2, maturities=(0.1, 0.5, 1.0)) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows, hists, models = [], [], {}
    for T in maturities:
        x, v = synthetic_quotes(T, sigma)
        sig_engine = fit_constant_vol(x, v, T)
        sig_bach = bachelier_only_sigma(T, x, v)
        for name, (ref, w_data) in VARIANTS.items():
            print(f"\n=== synthetic T={T} variant={name} ===", flush=True)
            cfg = InverseBSConfig(ref_sigma=ref, w_data=w_data, train=_train_cfg(fast))
            m = InverseBSPINN(T, x, v, cfg=cfg)
            res = m.fit()
            err = price_error_vs_black(m, sigma)
            # 같은 격자에서 기준해 Phi만 쓴 오차 (신경망이 보정해야 하는 양)
            tau = np.linspace(0, T, 101)
            xs = np.linspace(m.x_lo, m.x_hi, 301)
            TT, XX = np.meshgrid(tau, xs, indexing="ij")
            v_true = black_call(XX, sigma**2 * TT)
            ref_l2 = np.linalg.norm(bachelier_price_np(TT, XX, sigma) - v_true) / np.linalg.norm(v_true)
            rows.append(
                {
                    "T": T,
                    "variant": name,
                    "w_data": w_data,
                    "sigma_true": sigma,
                    "sigma_pinn": res.sigma,
                    "rel_err_pinn": abs(res.sigma - sigma) / sigma,
                    "sigma_engine_lsq": sig_engine,
                    "rel_err_engine": abs(sig_engine - sigma) / sigma,
                    "sigma_bachelier_only": sig_bach,
                    "rel_err_bachelier_only": abs(sig_bach - sigma) / sigma,
                    "price_rel_l2_pinn": err["rel_l2"],
                    "price_rel_l2_reference_only": ref_l2,
                    "price_max_abs_atm": err["max_abs_atm"],
                    "x_domain": f"[{m.x_lo:.2f}, {m.x_hi:.2f}]",
                    "train_seconds": res.train_seconds,
                }
            )
            h = res.history.copy()
            h["T"], h["variant"], h["step"] = T, name, _step_index(h)
            hists.append(h)
            models[(T, name)] = m

    table = pd.DataFrame(rows)
    table.to_csv(out / "synthetic_results.csv", index=False)
    history = pd.concat(hists, ignore_index=True)
    history.to_csv(out / "synthetic_history.csv", index=False)

    _plot.setup()
    import matplotlib.pyplot as plt

    # sigma 수렴: 변형마다 패널 하나, 만기마다 선 하나
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name in zip(axes, VARIANTS):
        for i, T in enumerate(maturities):
            h = history[(history["T"] == T) & (history["variant"] == name)]
            ax.plot(h["step"], h["sigma"], color=_plot.SERIES[i], marker=_plot.MARKERS[i], markevery=4, label=f"T = {T}")
        ax.axhline(sigma, color=_plot.TEXT_2, linewidth=1, linestyle="--")
        ax.set_title({"trainable_ref": "Trainable reference", "fixed_ref": "Fixed reference (sigma via PDE only)"}[name])
        ax.set_xlabel("optimisation step (Adam, then L-BFGS)")
    axes[0].set_ylabel("estimated sigma")
    axes[0].legend()
    _plot.save(fig, out / "sigma_convergence.png")

    # tau = T에서 점별 가격 오차 (학습형 기준해)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, T in enumerate(maturities):
        m = models[(T, "trainable_ref")]
        xs = np.linspace(max(0.7, m.x_lo), min(1.3, m.x_hi), 301)
        e = m.price(np.full_like(xs, T), xs) - black_call(xs, sigma**2 * T)
        ax.plot(xs, e, color=_plot.SERIES[i], marker=_plot.MARKERS[i], markevery=30, label=f"T = {T}")
    ax.axhline(0, color=_plot.TEXT_2, linewidth=1)
    ax.set_xlabel("moneyness x = F/K")
    ax.set_ylabel("PINN price - Black price")
    ax.set_title("Price error at maturity (trainable reference)")
    ax.legend()
    _plot.save(fig, out / "price_error.png")

    print("\n" + table[["T", "variant", "sigma_pinn", "rel_err_pinn", "rel_err_bachelier_only",
                        "price_rel_l2_pinn", "price_rel_l2_reference_only", "train_seconds"]].to_string(index=False))
    print(f"\nsaved to {out}")
    return table


# -------------------------------------------------------------------------------- 실데이터
def _pick(n_total: int, n: int) -> list[int]:
    if n_total <= n:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, n).round().astype(int).tolist()))


def run_live(out_root: Path, fast: bool, n_pinn: int = 5, ticker: str = "^SPX") -> pd.DataFrame:
    from mfin.data.loaders import latest_risk_free
    from mfin.data.option_chain import fetch_chain, prepare_chain
    from mfin.data.vix_term import latest_vix_term

    r = latest_risk_free()
    chain = fetch_chain(ticker, min_days=7, max_days=365, target_days=[7, 14, 30, 60, 90, 120, 180, 270, 365])
    quote_day = chain["quote_time"].iloc[0].strftime("%Y%m%d")
    out = out_root / f"live_{quote_day}"
    out.mkdir(parents=True, exist_ok=True)

    otm, fwd = prepare_chain(chain, r)
    sm = smile(otm)
    ts = term_structure(sm)
    fwd.to_csv(out / "forwards.csv")
    sm.to_csv(out / "smile.csv", index=False)
    ts.to_csv(out / "term_structure.csv", index=False)
    print(f"{chain['underlying'].iloc[0]} spot {chain['spot'].iloc[0]:.2f}, r = {r:.4f}, "
          f"{len(sm)} OTM quotes over {sm['expiry'].nunique()} expiries "
          f"(price source: {sm['price_source'].value_counts().to_dict()})")
    print(ts.to_string(index=False))
    stale_share = float((sm["price_source"] == "last").mean())
    if stale_share > 0.5:
        msg = (f"WARNING: {stale_share:.0%} of quotes are last-trade prices (no live bid/ask, e.g. weekend). "
               "Last trades happen at different times, so implied vols are noisy; rerun during US market hours.")
        print(msg)
        (out / "WARNING_stale_quotes.txt").write_text(msg + "\n", encoding="utf-8")

    valid = ts[np.isfinite(ts["atm_iv"])].reset_index(drop=True)
    rows = []
    for i in _pick(len(valid), n_pinn):
        row = valid.iloc[i]
        T, atm = float(row["T"]), float(row["atm_iv"])
        band = float(np.clip(0.5 * atm * np.sqrt(T), 0.02, 0.1))
        g = sm[(sm["expiry"] == row["expiry"]) & (sm["k"].abs() <= band)]
        if len(g) < 5:
            continue
        x, v = g["x"].to_numpy(), g["v"].to_numpy()
        sig_engine = fit_constant_vol(x, v, T, sigma0=atm)
        print(f"\n=== live T={T:.4f} ({row['expiry']:%Y-%m-%d}) band |k|<={band:.3f}, {len(g)} quotes ===", flush=True)
        m = InverseBSPINN(T, x, v, cfg=InverseBSConfig(train=_train_cfg(fast)))
        res = m.fit()
        rows.append(
            {
                "expiry": row["expiry"],
                "T": T,
                "n_quotes": len(g),
                "band_k": band,
                "atm_iv_brent": atm,
                "sigma_engine_lsq": sig_engine,
                "sigma_pinn": res.sigma,
                "pinn_minus_brent_volpt": 100 * (res.sigma - atm),
                "pinn_minus_engine_volpt": 100 * (res.sigma - sig_engine),
                "train_seconds": res.train_seconds,
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "pinn_vs_brent.csv", index=False)

    # HJB 센서 시점에서의 선도변동성 곡선
    curves = {"sensor_t": DEFAULT_SENSORS}
    curves["brent_atm"] = forward_vol_curve(valid["T"], valid["atm_iv"])
    if len(comp):
        curves["pinn"] = forward_vol_curve(comp["T"], comp["sigma_pinn"])
    vix = None
    try:
        vix = latest_vix_term()
        curves["vix"] = vix_forward_vol_curve(vix)
    except Exception as e:  # noqa: BLE001
        print(f"VIX term structure unavailable: {e!r}")
    fwd_df = pd.DataFrame(curves)
    fwd_df.to_csv(out / "forward_vol.csv", index=False)

    _plot.setup()
    import matplotlib.pyplot as plt

    # smile: 짧은 / 중간 / 긴 만기
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, i in enumerate(_pick(len(valid), 3)):
        e = valid.iloc[i]
        g = sm[(sm["expiry"] == e["expiry"]) & (sm["k"].abs() <= 0.25)]
        ax.plot(g["k"], 100 * g["iv"], color=_plot.SERIES[j], marker=_plot.MARKERS[j], markevery=max(1, len(g) // 12),
                label=f"{e['expiry']:%Y-%m-%d} (T = {e['T']:.2f})")
    ax.set_xlabel("log-moneyness k = ln(K/F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("SPX implied volatility smile (Brent)")
    ax.legend()
    _plot.save(fig, out / "smile.png")

    # ATM 기간구조: Brent, PINN, VIX 비교
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(valid["T"], 100 * valid["atm_iv"], color=_plot.SERIES[0], marker=_plot.MARKERS[0], label="Brent ATM")
    if len(comp):
        ax.plot(comp["T"], 100 * comp["sigma_pinn"], color=_plot.SERIES[1], marker=_plot.MARKERS[1], linestyle="none",
                markersize=8, label="Inverse PINN")
    if vix is not None:
        ax.plot(vix.index, 100 * vix.values, color=_plot.SERIES[2], marker=_plot.MARKERS[2], linestyle="--",
                linewidth=1.5, label=f"VIX indices ({vix.name:%Y-%m-%d})")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("volatility (%)")
    ax.set_title("At-the-money volatility term structure")
    ax.legend()
    _plot.save(fig, out / "term_structure.png")

    # 선도변동성 곡선 (HJB 모듈의 입력)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = {"brent_atm": "from Brent ATM", "pinn": "from inverse PINN", "vix": "from VIX indices"}
    for j, key in enumerate([k for k in ("brent_atm", "pinn", "vix") if k in fwd_df]):
        ax.step(fwd_df["sensor_t"], 100 * fwd_df[key], where="post", color=_plot.SERIES[j], marker=_plot.MARKERS[j],
                label=labels[key])
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("forward volatility sigma_fwd(t) (%)")
    ax.set_title("Forward volatility curve at HJB sensor times")
    ax.legend()
    _plot.save(fig, out / "forward_vol.png")

    if len(comp):
        print("\n" + comp.to_string(index=False))
    print("\n" + fwd_df.to_string(index=False))
    print(f"\nsaved to {out}")
    return comp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--fast", action="store_true", help="fewer training iterations (quick check)")
    ap.add_argument("--n-pinn", type=int, default=5, help="maturities fitted by the PINN in --live")
    ap.add_argument("--ticker", default="^SPX")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "module_a")
    args = ap.parse_args()
    if not (args.synthetic or args.live):
        ap.error("choose --synthetic and/or --live")
    if args.synthetic:
        run_synthetic(args.out / "synthetic", args.fast)
    if args.live:
        run_live(args.out, args.fast, args.n_pinn, args.ticker)


if __name__ == "__main__":
    main()
