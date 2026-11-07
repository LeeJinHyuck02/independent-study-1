"""Volatility term structure: ATM vol, skew and the forward volatility curve.

The HJB module needs the instantaneous volatility sigma_fwd(t) along the investment
horizon. From implied (or VIX) vols sigma(T_j) at maturities T_j:

    total variance        w_j = T_j sigma(T_j)^2      (must be non-decreasing)
    forward variance      sigma_fwd^2(t) = (w_j - w_{j-1}) / (T_j - T_{j-1}),  t in (T_{j-1}, T_j]

i.e. sigma_fwd^2 = dw/dt for a piecewise-linear w, flat beyond the last maturity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sensor times (years) where sigma_fwd(t) is sampled for the HJB / DeepONet input.
DEFAULT_SENSORS = np.array([1 / 52, 1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 8 / 12, 10 / 12, 1.0])


def _interp_at(k: np.ndarray, iv: np.ndarray, k0: float) -> float:
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    if len(k) == 0 or k0 < k[0] or k0 > k[-1]:
        return float("nan")
    return float(np.interp(k0, k, iv))


def term_structure(smile_df: pd.DataFrame, skew_moneyness: tuple[float, float] = (0.9, 1.1)) -> pd.DataFrame:
    """Per expiry: T, F, ATM implied vol (k = 0) and skew sigma(0.9F) - sigma(1.1F).

    ``smile_df`` is the output of :func:`mfin.engine.implied_vol.smile`.
    """
    k_lo, k_hi = np.log(skew_moneyness[0]), np.log(skew_moneyness[1])
    rows = []
    for exp, g in smile_df.groupby("expiry"):
        k, iv = g["k"].to_numpy(), g["iv"].to_numpy()
        rows.append(
            {
                "expiry": exp,
                "T": float(g["T"].iloc[0]),
                "F": float(g["F"].iloc[0]),
                "D": float(g["D"].iloc[0]),
                "n_options": len(g),
                "atm_iv": _interp_at(k, iv, 0.0),
                "skew": _interp_at(k, iv, k_lo) - _interp_at(k, iv, k_hi),
            }
        )
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def forward_variance_knots(taus, sigmas, min_fwd_var: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Maturity knots [0, T_1, ..., T_n] and the forward variance on each interval.

    Total variance is made non-decreasing (with a small positive floor on each
    increment) so that every forward variance is positive; calendar arbitrage in the
    raw quotes would otherwise give negative variance.
    """
    taus = np.asarray(taus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    ok = np.isfinite(taus) & np.isfinite(sigmas) & (taus > 0)
    taus, sigmas = taus[ok], sigmas[ok]
    if len(taus) == 0:
        raise ValueError("no valid (tau, sigma) points")
    order = np.argsort(taus)
    taus, sigmas = taus[order], sigmas[order]
    # merge duplicate maturities
    taus, inv = np.unique(taus, return_inverse=True)
    sigmas = np.array([sigmas[inv == i].mean() for i in range(len(taus))])

    knots = np.concatenate([[0.0], taus])
    w = np.concatenate([[0.0], taus * sigmas**2])
    for j in range(1, len(w)):
        w[j] = max(w[j], w[j - 1] + min_fwd_var * (knots[j] - knots[j - 1]))
    fwd_var = np.diff(w) / np.diff(knots)
    return knots, fwd_var


def forward_vol_curve(taus, sigmas, sensors=DEFAULT_SENSORS, min_fwd_var: float = 1e-6) -> np.ndarray:
    """Instantaneous forward volatility sigma_fwd(t) evaluated at ``sensors`` (years)."""
    knots, fwd_var = forward_variance_knots(taus, sigmas, min_fwd_var)
    t = np.asarray(sensors, dtype=float)
    # interval j covers (knots[j], knots[j+1]]; beyond the last knot use the last interval
    idx = np.clip(np.searchsorted(knots, t, side="left") - 1, 0, len(fwd_var) - 1)
    return np.sqrt(fwd_var[idx])


def vix_forward_vol_curve(vix_row: pd.Series, sensors=DEFAULT_SENSORS) -> np.ndarray:
    """Forward vol curve from one row of the VIX term structure (tenor -> vol)."""
    row = vix_row.dropna()
    return forward_vol_curve(row.index.to_numpy(dtype=float), row.to_numpy(dtype=float), sensors)
