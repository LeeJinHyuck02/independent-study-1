"""Black 해석식 가격에 Brent 근 찾기를 적용해 내재변동성을 구한다.

모듈 A의 정확한 기준해("엔진")이며, 역방향 PINN은 이 결과와 비교한다.
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
    """옵션 하나의 Black 내재변동성. 가격이 무차익 범위를 벗어나거나 [lo, hi] 안에
    근이 없으면 NaN을 반환한다."""
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
    """:func:`implied_vol`의 배열 버전 (브로드캐스트한 입력을 하나씩 반복 계산한다)."""
    b = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (prices, F, K, T, D)))
    out = np.array([implied_vol(p, f, k, t, d, kind) for p, f, k, t, d in zip(*(a.ravel() for a in b))])
    return out.reshape(b[0].shape)


def smile(otm: pd.DataFrame) -> pd.DataFrame:
    """OTM 콜 환산 가격 행(``to_otm_calls``의 출력) 각각의 내재변동성.

    풋-콜 패리티에 의해 풋과 그 풋에서 만든 콜은 내재변동성이 같으므로, 콜 환산 가격을
    콜 공식으로 역산한다.
    """
    s = otm.copy()
    s["iv"] = implied_vols(s["call_price"], s["F"], s["K"], s["T"], s["D"], "call")
    return s[np.isfinite(s["iv"])].reset_index(drop=True)


def fit_constant_vol(x, v, T: float, weights=None, sigma0: float = 0.2) -> float:
    """머니니스 x, 만기 T의 정규화 가격 v에 최소제곱으로 맞춘 상수 sigma.

    역방향 PINN과 같은 데이터 적합 문제를 해석식 가격으로 푼다. 그래서 둘을 같은 입력에서
    비교할 수 있다.
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
