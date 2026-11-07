"""SPX 옵션체인: 다운로드, 정제, 내재 선도가격, OTM 콜 환산 가격.

규약
-----------
- T는 만기까지 남은 시간(년, 달력일 / 365)이다. 체인 안의 가장 최근 시세 시각부터 잰다
  (그래서 주말에 받은 데이터는 금요일 장 마감부터 잰다).
- F는 만기별 선도가격, D는 할인계수다. 둘 다 풋-콜 패리티  C - P = D (F - K)에서 구한다.
  배당과 금리가 여기에 흡수되므로 배당수익률을 따로 둘 필요가 없다.
- 정규화 콜 가격  v = C / (D K)와 머니니스  x = F / K는 선도가격 형태 블랙-숄즈 방정식
  v_tau = 1/2 sigma^2 x^2 v_xx의 변수다.
"""

from __future__ import annotations

import re
import time
import warnings
from datetime import timedelta
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from mfin.data.cache import DataUnavailableError, cached, retry

NY = "America/New_York"
YEAR_SECONDS = 365.0 * 86400.0
_ROOT_RE = re.compile(r"^([A-Z]+)\d{6}[CP]\d+$")


def expiry_timestamp(expiry: str) -> pd.Timestamp:
    """만기일 문자열 -> 그날 뉴욕 시각 16:00."""
    return pd.Timestamp(f"{expiry} 16:00", tz=NY)


def select_expiries(
    expiries: Sequence[str],
    asof: pd.Timestamp,
    min_days: float = 7,
    max_days: float = 365,
    target_days: Sequence[float] | None = None,
) -> list[str]:
    """[min_days, max_days] 안의 만기들. ``target_days``가 주어지면 각 목표 일수에 가장
    가까운 만기만 고른다(중복 제거). 시간 순으로 반환한다."""
    days = {e: (expiry_timestamp(e) - asof).total_seconds() / 86400.0 for e in expiries}
    in_range = [e for e in expiries if min_days <= days[e] <= max_days]
    if not target_days:
        return sorted(in_range, key=days.get)
    chosen: list[str] = []
    for t in target_days:
        if not in_range:
            break
        e = min(in_range, key=lambda e: abs(days[e] - t))
        if e not in chosen:
            chosen.append(e)
    return sorted(chosen, key=days.get)


def _contract_root(symbol: str) -> str:
    m = _ROOT_RE.match(str(symbol))
    return m.group(1) if m else ""


def _download_chain(
    ticker: str,
    min_days: float,
    max_days: float,
    target_days: Sequence[float] | None,
    pause: float,
) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    expiries = list(retry(lambda: tk.options))
    if not expiries:
        return pd.DataFrame()
    hist = retry(lambda: tk.history(period="5d", auto_adjust=False))
    spot = float(hist["Close"].dropna().iloc[-1])
    asof = pd.Timestamp.now(tz=NY)

    frames = []
    for exp in select_expiries(expiries, asof, min_days, max_days, target_days):
        oc = retry(lambda exp=exp: tk.option_chain(exp))
        for kind, df in (("C", oc.calls), ("P", oc.puts)):
            if df is None or df.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "expiry": expiry_timestamp(exp),
                        "root": df["contractSymbol"].map(_contract_root),
                        "K": df["strike"].astype(float),
                        "type": kind,
                        "bid": pd.to_numeric(df["bid"], errors="coerce"),
                        "ask": pd.to_numeric(df["ask"], errors="coerce"),
                        "last": pd.to_numeric(df["lastPrice"], errors="coerce"),
                        "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0.0),
                        "open_interest": pd.to_numeric(df["openInterest"], errors="coerce").fillna(0.0),
                        "yahoo_iv": pd.to_numeric(df["impliedVolatility"], errors="coerce"),
                        "last_trade": pd.to_datetime(df["lastTradeDate"], utc=True),
                    }
                )
            )
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()

    chain = pd.concat(frames, ignore_index=True)
    # 만기는 "지금"이 아니라 가장 최근 시세 시각부터 잰다 (주말·공휴일 대비).
    quote_time = min(chain["last_trade"].max().tz_convert(NY), asof)
    chain["T"] = (chain["expiry"] - quote_time).dt.total_seconds() / YEAR_SECONDS
    chain["underlying"] = ticker
    chain["spot"] = spot
    chain["asof"] = asof
    chain["quote_time"] = quote_time
    return chain[chain["T"] > 0].reset_index(drop=True)


def fetch_chain(
    ticker: str = "^SPX",
    min_days: float = 7,
    max_days: float = 365,
    target_days: Sequence[float] | None = None,
    fallback: str | None = "SPY",
    max_age: timedelta | None = timedelta(hours=1),
    refresh: bool = False,
    pause: float = 0.3,
) -> pd.DataFrame:
    """옵션체인을 받는다(계약당 한 행). 실패하면 ``fallback`` 종목으로 대체한다.

    ^SPX 옵션은 유럽형이라 블랙-숄즈 모형과 맞는다. SPY 옵션은 미국형이다. ATM 근방에서는
    조기행사 프리미엄이 작지만, SPY는 대체용으로만 쓴다.
    """
    day = pd.Timestamp.now(tz=NY).strftime("%Y%m%d")
    tag = "all" if not target_days else "-".join(str(int(t)) for t in target_days)
    errors = []
    for tkr in [t for t in (ticker, fallback) if t]:
        key = f"chain_{tkr}_{day}_{min_days:g}-{max_days:g}_{tag}"
        try:
            df = cached(
                key,
                lambda tkr=tkr: _download_chain(tkr, min_days, max_days, target_days, pause),
                max_age,
                refresh,
            )
        except DataUnavailableError as e:
            errors.append(str(e))
            continue
        if tkr != ticker:
            warnings.warn(f"{ticker} chain unavailable; using {tkr} (American options)", stacklevel=2)
        return df
    raise DataUnavailableError("; ".join(errors))


def clean_chain(
    chain: pd.DataFrame,
    min_price: float = 0.01,
    require_activity: bool = True,
    max_last_age_days: float = 3.0,
) -> pd.DataFrame:
    """계약마다 쓸 수 있는 가격을 붙이고, 쓸 수 없는 행은 버린다.

    매수·매도 호가가 모두 있으면 가격 = 두 호가의 중간값(mid)이다. 장 시간 밖에는 Yahoo가
    bid = ask = 0으로 주는 경우가 많다. 이때 최근 체결가가 있으면 그것을 쓰고
    ``price_source == "last"``로 표시한다.
    """
    c = chain.copy()

    # 같은 만기에 오전 정산(SPX)과 오후 정산(SPXW) 계약이 함께 있을 수 있다.
    # 행사가가 중복되지 않도록 계약 수가 가장 많은 쪽만 남긴다.
    if "root" in c.columns:
        counts = c.groupby(["expiry", "root"]).size().rename("n").reset_index()
        keep = counts.sort_values("n", ascending=False).drop_duplicates("expiry")
        c = c.merge(keep[["expiry", "root"]], on=["expiry", "root"])

    bid, ask, last = c["bid"].fillna(0.0), c["ask"].fillna(0.0), c["last"].fillna(0.0)
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    last_age = (c["quote_time"] - c["last_trade"].dt.tz_convert(NY)).dt.total_seconds() / 86400.0
    recent_last = (last > 0) & (last_age <= max_last_age_days)

    c["mid"] = np.where(two_sided, 0.5 * (bid + ask), np.where(recent_last, last, np.nan))
    c["price_source"] = np.where(two_sided, "mid", np.where(recent_last, "last", "none"))
    c["spread"] = np.where(two_sided, ask - bid, np.nan)

    mask = (c["mid"] >= min_price) & (c["T"] > 0)
    if require_activity:
        mask &= (c["volume"] > 0) | (c["open_interest"] > 0)
    return c[mask].sort_values(["expiry", "type", "K"]).reset_index(drop=True)


def _plausible_discount(D: float, T: float, r: float | None, tol: float = 0.03) -> bool:
    """회귀로 구한 할인계수는 그에 해당하는 금리가 현실적일 때만 받아들인다.

    D의 범위만 보는 것으로는 부족하다. 10일 만기에서 D = 0.95는 연 190% 금리에 해당한다.
    시장 금리 r이 주어지면 해당 금리가 r과 ``tol`` 이내여야 한다.
    """
    if not (D > 0 and T > 0):
        return False
    r_impl = -np.log(D) / T
    if r is not None:
        return abs(r_impl - r) <= tol
    return -0.05 <= r_impl <= 0.20


def implied_forward(
    chain: pd.DataFrame,
    r: float | None = None,
    band: float = 0.05,
    n_min: int = 3,
) -> pd.DataFrame:
    """풋-콜 패리티로 만기별 선도가격 F와 할인계수 D를 구한다.

    현물가에서 ``band`` 이내이고 콜과 풋이 모두 있는 행사가에서 C - P = D F - D K는 K에
    대한 일차식이다. 그래서 최소제곱 직선의 기울기에서 D = -slope, 절편에서
    F = intercept / D를 얻는다. 이 결과는 해당 금리가 그럴듯할 때만 쓴다(오래되었거나 잡음이
    섞인 시세는 기울기를 틀어지게 한다). 그렇지 않으면 D = exp(-rT)로 두고, F는 ATM 근방
    행사가들의 K + (C - P)/D의 중앙값으로 한다.
    """
    rows = []
    for exp, g in chain.groupby("expiry"):
        T = float(g["T"].iloc[0])
        spot = float(g["spot"].iloc[0])
        calls = g[g["type"] == "C"].groupby("K")["mid"].mean()
        puts = g[g["type"] == "P"].groupby("K")["mid"].mean()
        common = calls.index.intersection(puts.index)
        if len(common) == 0:
            continue
        diff = (calls[common] - puts[common]).astype(float)
        near = diff[np.abs(diff.index / spot - 1.0) <= band]

        F = D = np.nan
        method = ""
        if len(near) >= n_min:
            slope, intercept = np.polyfit(near.index.values, near.values, 1)
            D_fit = -slope
            if _plausible_discount(D_fit, T, r):
                D, F, method = D_fit, intercept / D_fit, "regression"
        if not np.isfinite(F):
            D = float(np.exp(-(r if r is not None else 0.0) * T))
            pts = near if len(near) >= n_min else diff[diff.abs().nsmallest(min(5, len(diff))).index]
            F = float(np.median(pts.index.values + pts.values / D))
            method = "parity"
        rows.append({"expiry": exp, "T": T, "F": float(F), "D": float(D), "n_strikes": len(near), "method": method})
    return pd.DataFrame(rows).set_index("expiry").sort_values("T")


def to_otm_calls(chain: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """외가격(OTM) 옵션만 남기고 각각을 콜 환산 가격으로 나타낸다.

    K < F인 풋은 패리티  C = P + D (F - K)로 콜 가격으로 바꾸고, K >= F인 콜은 그대로 둔다.
    OTM 옵션이 유동성이 높고 조기행사 프리미엄 문제도 없다.
    x = F/K, k = ln(K/F), 정규화 콜 가격 v = C / (D K) 열을 추가한다.
    """
    c = chain.merge(forwards[["F", "D"]], left_on="expiry", right_index=True, how="inner")
    otm = ((c["type"] == "P") & (c["K"] < c["F"])) | ((c["type"] == "C") & (c["K"] >= c["F"]))
    c = c[otm].copy()
    c["call_price"] = np.where(c["type"] == "C", c["mid"], c["mid"] + c["D"] * (c["F"] - c["K"]))
    c["x"] = c["F"] / c["K"]
    c["k"] = np.log(c["K"] / c["F"])
    c["v"] = c["call_price"] / (c["D"] * c["K"])
    cols = [
        "underlying", "expiry", "T", "K", "F", "D", "type", "mid", "call_price",
        "x", "k", "v", "price_source", "spread", "volume", "open_interest",
    ]
    return c[[col for col in cols if col in c.columns]].sort_values(["T", "K"]).reset_index(drop=True)


def prepare_chain(chain: pd.DataFrame, r: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정제 -> 내재 선도가격 -> OTM 콜 환산. (otm, forwards)를 반환한다."""
    clean = clean_chain(chain)
    fwd = implied_forward(clean, r=r)
    return to_otm_calls(clean, fwd), fwd
