"""Closed-form Black-Scholes / Black (forward) prices and Greeks.

Forward-normalised form (used throughout module A)
-------------------------------------------------
With forward F, discount factor D, strike K, maturity T and volatility sigma,

    C = D K c(x, w),   x = F / K,   w = sigma^2 T  (total variance)
    c(x, w) = x N(d1) - N(d2),   d1 = (ln x + w/2) / sqrt(w),   d2 = d1 - sqrt(w)

The normalised undiscounted price v(tau, x) = c(x, sigma^2 tau) solves

    v_tau = 1/2 sigma^2 x^2 v_xx,   v(0, x) = (x - 1)^+,

which is the Black-Scholes equation after the change of variables
x = F/K, v = e^{r tau} V / K (rates and dividends are absorbed into F).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(z):
    return np.exp(-0.5 * np.square(z)) / _SQRT_2PI


def _d1_d2(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = np.sqrt(np.maximum(w, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(x) + 0.5 * w) / sw
    return d1, d1 - sw, sw


def black_call(x, w):
    """Normalised undiscounted call c(x, w); returns the payoff (x-1)^+ when w = 0."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    d1, d2, sw = _d1_d2(x, w)
    with np.errstate(invalid="ignore"):
        price = x * ndtr(d1) - ndtr(d2)
    payoff = np.maximum(x - 1.0, 0.0)
    out = np.where(sw > 0, price, payoff)
    return out[()] if out.ndim == 0 else out


def black_put(x, w):
    """Normalised undiscounted put p(x, w) = c(x, w) - (x - 1)."""
    return black_call(x, w) - (np.asarray(x, dtype=float) - 1.0)


def black_vega_s(x, s):
    """d c / d s with s = sigma sqrt(T) (vega per unit of total standard deviation)."""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    d1, _, _ = _d1_d2(x, s * s)
    return x * norm_pdf(d1)


def black_price(F, K, T, sigma, D=1.0, kind: str = "call"):
    """Discounted Black price from forward F."""
    x = np.asarray(F, dtype=float) / np.asarray(K, dtype=float)
    w = np.square(np.asarray(sigma, dtype=float)) * np.asarray(T, dtype=float)
    c = black_call(x, w) if kind == "call" else black_put(x, w)
    return np.asarray(D, dtype=float) * np.asarray(K, dtype=float) * c


def bs_price(S, K, T, r, sigma, q=0.0, kind: str = "call"):
    """Black-Scholes price with spot S, rate r and dividend yield q (continuous)."""
    F = np.asarray(S, dtype=float) * np.exp((np.asarray(r) - np.asarray(q)) * np.asarray(T))
    D = np.exp(-np.asarray(r) * np.asarray(T))
    return black_price(F, K, T, sigma, D, kind)


def _bs_d1(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = (np.asarray(a, dtype=float) for a in (S, K, T, r, sigma, q))
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_delta(S, K, T, r, sigma, q=0.0, kind: str = "call"):
    d1 = _bs_d1(S, K, T, r, sigma, q)
    disc_q = np.exp(-np.asarray(q) * np.asarray(T))
    return disc_q * (ndtr(d1) if kind == "call" else ndtr(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    d1 = _bs_d1(S, K, T, r, sigma, q)
    return np.exp(-np.asarray(q) * np.asarray(T)) * norm_pdf(d1) / (
        np.asarray(S) * np.asarray(sigma) * np.sqrt(np.asarray(T))
    )


def bs_vega(S, K, T, r, sigma, q=0.0):
    """d price / d sigma (per 1.00 of volatility, not per vol point)."""
    d1 = _bs_d1(S, K, T, r, sigma, q)
    return np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T)) * norm_pdf(d1) * np.sqrt(np.asarray(T))
