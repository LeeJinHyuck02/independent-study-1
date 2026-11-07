"""SPX option chains: download, cleaning, implied forwards and OTM call-equivalent prices.

Conventions
-----------
- T is the time to expiry in years (calendar days / 365), measured from the latest
  quote time in the chain (so weekend downloads are measured from Friday's close).
- F is the forward price and D the discount factor for each expiry, both implied
  from put-call parity  C - P = D (F - K).  This absorbs dividends and rates, so no
  separate dividend yield is needed.
- Normalised call price  v = C / (D K)  and moneyness  x = F / K  are the variables of
  the forward-form Black-Scholes equation  v_tau = 1/2 sigma^2 x^2 v_xx.
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
    """Expiry date string -> 16:00 New York time on that date."""
    return pd.Timestamp(f"{expiry} 16:00", tz=NY)


def select_expiries(
    expiries: Sequence[str],
    asof: pd.Timestamp,
    min_days: float = 7,
    max_days: float = 365,
    target_days: Sequence[float] | None = None,
) -> list[str]:
    """Expiries within [min_days, max_days]; if ``target_days`` is given, the expiry
    closest to each target (duplicates removed), in time order."""
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
    # Measure maturities from the latest quote, not from "now" (weekends/holidays).
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
    """Download an option chain (one row per contract), falling back to ``fallback``.

    ^SPX options are European, which matches the Black-Scholes model. SPY options are
    American; near the money the early-exercise premium is small, so SPY is only a
    fallback.
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
    """Add a usable price per contract and drop unusable rows.

    Price = bid/ask mid when both sides are quoted. Outside market hours Yahoo often
    reports bid = ask = 0; then the last trade price is used if it is recent, and the
    row is flagged with ``price_source == "last"``.
    """
    c = chain.copy()

    # The same expiry can list both AM-settled (SPX) and PM-settled (SPXW) contracts;
    # keep the root with the most contracts so strikes are not duplicated.
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


def implied_forward(
    chain: pd.DataFrame,
    r: float | None = None,
    band: float = 0.05,
    n_min: int = 3,
) -> pd.DataFrame:
    """Forward F and discount factor D per expiry from put-call parity.

    For strikes within ``band`` of spot that have both a call and a put,
    C - P = D F - D K is linear in K, so a least-squares line gives D = -slope and
    F = intercept / D. If the fit is not usable, D = exp(-rT) and F is the median of
    K + (C - P)/D over the strikes where |C - P| is smallest.
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
            if 0.8 < D_fit <= 1.02:
                D, F, method = D_fit, intercept / D_fit, "regression"
        if not np.isfinite(F):
            D = float(np.exp(-(r if r is not None else 0.0) * T))
            best = diff.abs().nsmallest(min(5, len(diff))).index
            F = float(np.median(best.values + diff[best].values / D))
            method = "parity"
        rows.append({"expiry": exp, "T": T, "F": float(F), "D": float(D), "n_strikes": len(near), "method": method})
    return pd.DataFrame(rows).set_index("expiry").sort_values("T")


def to_otm_calls(chain: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Keep out-of-the-money options and express each as a call-equivalent price.

    Puts with K < F are converted with parity  C = P + D (F - K); calls with K >= F are
    kept. OTM options are the liquid ones and carry no early-exercise premium issue.
    Adds x = F/K, k = ln(K/F) and the normalised call price v = C / (D K).
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
    """clean -> implied forwards -> OTM call-equivalents. Returns (otm, forwards)."""
    clean = clean_chain(chain)
    fwd = implied_forward(clean, r=r)
    return to_otm_calls(clean, fwd), fwd
