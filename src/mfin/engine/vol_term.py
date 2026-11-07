"""변동성 기간구조: ATM 변동성, skew, 선도변동성 곡선.

HJB 모듈에는 투자 기간에 걸친 순간 변동성 sigma_fwd(t)가 필요하다. 만기 T_j의 내재
변동성(또는 VIX) sigma(T_j)에서 다음과 같이 구한다.

    총분산        w_j = T_j sigma(T_j)^2      (감소하면 안 된다)
    선도분산      sigma_fwd^2(t) = (w_j - w_{j-1}) / (T_j - T_{j-1}),  t in (T_{j-1}, T_j]

즉 w를 구간별 선형으로 보고 sigma_fwd^2 = dw/dt로 둔다. 마지막 만기 이후는 일정하다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# HJB / DeepONet 입력용으로 sigma_fwd(t)를 읽는 센서 시점(년).
DEFAULT_SENSORS = np.array([1 / 52, 1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 8 / 12, 10 / 12, 1.0])


def _interp_at(k: np.ndarray, iv: np.ndarray, k0: float) -> float:
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    if len(k) == 0 or k0 < k[0] or k0 > k[-1]:
        return float("nan")
    return float(np.interp(k0, k, iv))


def term_structure(smile_df: pd.DataFrame, skew_moneyness: tuple[float, float] = (0.9, 1.1)) -> pd.DataFrame:
    """만기별 T, F, ATM 내재변동성(k = 0), skew sigma(0.9F) - sigma(1.1F).

    ``smile_df``는 :func:`mfin.engine.implied_vol.smile`의 출력이다.
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
    """만기 마디 [0, T_1, ..., T_n]와 각 구간의 선도분산.

    총분산이 감소하지 않도록 보정한다(각 증가분에 작은 양의 하한을 둔다). 그래야 모든
    선도분산이 양수가 된다. 원시 시세에 달력 차익이 있으면 분산이 음수로 나오기 때문이다.
    """
    taus = np.asarray(taus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    ok = np.isfinite(taus) & np.isfinite(sigmas) & (taus > 0)
    taus, sigmas = taus[ok], sigmas[ok]
    if len(taus) == 0:
        raise ValueError("no valid (tau, sigma) points")
    order = np.argsort(taus)
    taus, sigmas = taus[order], sigmas[order]
    # 중복 만기는 평균으로 합친다
    taus, inv = np.unique(taus, return_inverse=True)
    sigmas = np.array([sigmas[inv == i].mean() for i in range(len(taus))])

    knots = np.concatenate([[0.0], taus])
    w = np.concatenate([[0.0], taus * sigmas**2])
    for j in range(1, len(w)):
        w[j] = max(w[j], w[j - 1] + min_fwd_var * (knots[j] - knots[j - 1]))
    fwd_var = np.diff(w) / np.diff(knots)
    return knots, fwd_var


def forward_vol_curve(taus, sigmas, sensors=DEFAULT_SENSORS, min_fwd_var: float = 1e-6) -> np.ndarray:
    """``sensors`` 시점(년)에서의 순간 선도변동성 sigma_fwd(t)."""
    knots, fwd_var = forward_variance_knots(taus, sigmas, min_fwd_var)
    t = np.asarray(sensors, dtype=float)
    # 구간 j는 (knots[j], knots[j+1]]를 덮는다. 마지막 마디 이후는 마지막 구간 값을 쓴다
    idx = np.clip(np.searchsorted(knots, t, side="left") - 1, 0, len(fwd_var) - 1)
    return np.sqrt(fwd_var[idx])


def vix_forward_vol_curve(vix_row: pd.Series, sensors=DEFAULT_SENSORS) -> np.ndarray:
    """VIX 기간구조 한 행(만기 -> 변동성)으로 만든 선도변동성 곡선."""
    row = vix_row.dropna()
    return forward_vol_curve(row.index.to_numpy(dtype=float), row.to_numpy(dtype=float), sensors)
