"""Implied volatility by Brent root finding on the closed-form Black price.

This is the exact reference ("engine") for module A; the inverse PINN is compared
against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from mfin.engine.bs_analytic import black_call, black_put, black_vega_s


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    D: float = 1.0,
    kind: str = "call",
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Black implied volatility of one option; NaN if the price violates no-arbitrage
    bounds or no root lies in [lo, hi]."""
    if not (np.isfinite(price) and F > 0 and K > 0 and T > 0 and D > 0):
        return float("nan")
    x = F / K
    v = price / (D * K)
    if kind == "call":
        lower, upper, f = max(x - 1.0, 0.0), x, black_call
    else:
        lower, upper, f = max(1.0 - x, 0.0), 1.0, black_put
    if not (lower < v < upper):
        return float("nan")

    sqrt_t = np.sqrt(T)

    def g(s: float) -> float:
        return float(f(x, s * s)) - v

    a, b = lo * sqrt_t, hi * sqrt_t
    ga, gb = g(a), g(b)
    if ga * gb > 0:
        return float("nan")
    s = brentq(g, a, b, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=300)
    return s / sqrt_t


def implied_vols(prices, F, K, T, D=1.0, kind: str = "call") -> np.ndarray:
    """Vectorised wrapper around :func:`implied_vol` (loops over broadcast inputs)."""
    b = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (prices, F, K, T, D)))
    out = np.array([implied_vol(p, f, k, t, d, kind) for p, f, k, t, d in zip(*(a.ravel() for a in b))])
    return out.reshape(b[0].shape)


def smile(otm: pd.DataFrame) -> pd.DataFrame:
    """Implied vol for every OTM call-equivalent row (output of ``to_otm_calls``).

    Put-call parity means a put and the call built from it share one implied vol, so
    the call-equivalent price is inverted with the call formula.
    """
    s = otm.copy()
    s["iv"] = implied_vols(s["call_price"], s["F"], s["K"], s["T"], s["D"], "call")
    return s[np.isfinite(s["iv"])].reset_index(drop=True)


def fit_constant_vol(x, v, T: float, weights=None, sigma0: float = 0.2) -> float:
    """Least-squares constant sigma for normalised prices v at moneyness x, maturity T.

    Solves the same data-fitting problem as the inverse PINN but with the closed-form
    price, so the two can be compared on identical inputs.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.ones_like(v) if weights is None else np.sqrt(np.asarray(weights, dtype=float))

    def resid(p):
        return w * (black_call(x, p[0] ** 2 * T) - v)

    def jac(p):
        s = p[0] * np.sqrt(T)
        return (w * black_vega_s(x, s) * np.sqrt(T))[:, None]

    res = least_squares(resid, x0=[sigma0], jac=jac, bounds=([1e-4], [5.0]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return float(res.x[0])
