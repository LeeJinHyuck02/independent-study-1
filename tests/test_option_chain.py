"""알려진 smile로 가격을 매긴 SPX 유사 합성 체인으로 체인 처리 과정을 검사한다."""

import numpy as np
import pandas as pd
import pytest

from mfin.data.option_chain import NY, clean_chain, implied_forward, prepare_chain, to_otm_calls
from mfin.engine.bs_analytic import black_call, black_price
from mfin.engine.implied_vol import smile
from mfin.engine.vol_term import term_structure

S, R, Q = 6000.0, 0.04, 0.013
QUOTE = pd.Timestamp("2026-11-06 16:00", tz=NY)


def smile_vol(k):
    return 0.18 - 0.25 * k + 0.5 * k**2


def make_chain(maturities=(30 / 365, 0.5), root="SPXW"):
    rows = []
    for T in maturities:
        expiry = QUOTE + pd.Timedelta(days=T * 365)
        F, D = S * np.exp((R - Q) * T), np.exp(-R * T)
        for K in np.arange(4800.0, 7200.0 + 1, 25.0):
            sig = smile_vol(np.log(K / F))
            for kind, typ in (("call", "C"), ("put", "P")):
                p = float(black_price(F, K, T, sig, D, kind))
                rows.append(
                    {
                        "expiry": expiry, "root": root, "K": K, "type": typ, "bid": p, "ask": p, "last": p,
                        "volume": 10.0, "open_interest": 100.0, "yahoo_iv": np.nan,
                        "last_trade": QUOTE.tz_convert("UTC"), "T": T, "underlying": "^SPX", "spot": S,
                        "asof": QUOTE, "quote_time": QUOTE,
                    }
                )
    return pd.DataFrame(rows)


def test_implied_forward_and_discount():
    fwd = implied_forward(clean_chain(make_chain()), r=R)
    for _, row in fwd.iterrows():
        assert row["method"] == "regression"
        assert row["F"] == pytest.approx(S * np.exp((R - Q) * row["T"]), rel=1e-9)
        assert row["D"] == pytest.approx(np.exp(-R * row["T"]), rel=1e-9)


def test_implausible_discount_falls_back_to_rate():
    # 행사가에 따라 C - P를 기울인다 (오래된 시세가 그렇듯): 회귀로 구한 D는 약 190% 금리에 해당
    T = 10 / 365
    chain = make_chain(maturities=(T,))
    calls = chain["type"] == "C"
    chain.loc[calls, ["bid", "ask", "last"]] = chain.loc[calls, ["bid", "ask", "last"]].add(
        -0.05 * (chain.loc[calls, "K"] - S), axis=0
    ).clip(lower=0.05).to_numpy()
    fwd = implied_forward(clean_chain(chain), r=R)
    row = fwd.iloc[0]
    assert row["method"] == "parity"
    assert row["D"] == pytest.approx(np.exp(-R * T), rel=1e-12)
    assert row["F"] == pytest.approx(S * np.exp((R - Q) * T), rel=2e-3)


def test_otm_conversion_and_smile():
    otm, fwd = prepare_chain(make_chain(), r=R)
    # 풋에서 환산한 가격과 콜 가격이 모형 콜 가격과 같아야 한다
    expected = otm["D"] * otm["K"] * black_call(otm["x"], smile_vol(otm["k"]) ** 2 * otm["T"])
    assert np.allclose(otm["call_price"], expected, rtol=1e-9, atol=1e-9)
    assert set(otm.loc[otm["k"] < 0, "type"]) == {"P"} and set(otm.loc[otm["k"] > 0, "type"]) == {"C"}
    sm = smile(otm)
    good = sm["k"].abs() < 0.15  # 먼 날개 부분: 가격이 0에 가까워 변동성 정보가 거의 없다
    assert np.allclose(sm.loc[good, "iv"], smile_vol(sm.loc[good, "k"]), atol=1e-7)


def test_term_structure_atm_and_skew():
    otm, _ = prepare_chain(make_chain(), r=R)
    ts = term_structure(smile(otm))
    assert np.allclose(ts["atm_iv"], 0.18, atol=1e-4)
    assert np.allclose(ts["skew"], smile_vol(np.log(0.9)) - smile_vol(np.log(1.1)), atol=1e-3)


def test_clean_chain_price_source_and_dedup():
    chain = make_chain(maturities=(0.25,))
    # 주말형 시세: 호가 없음, 최근 체결가 있음 -> 체결가 사용
    chain.loc[0:9, ["bid", "ask"]] = 0.0
    # 오래된 체결가(10일 전)와 호가 없음 -> 제외
    chain.loc[10:19, ["bid", "ask"]] = 0.0
    chain.loc[10:19, "last_trade"] = (QUOTE - pd.Timedelta(days=10)).tz_convert("UTC")
    # 같은 만기의 두 번째, 더 작은 루트 (SPXW 옆의 오전 정산 SPX) -> 제거
    extra = chain.iloc[:5].copy()
    extra["root"] = "SPX"
    clean = clean_chain(pd.concat([chain, extra], ignore_index=True))

    assert set(clean["root"]) == {"SPXW"}
    assert (clean["price_source"] == "last").sum() == 10
    assert len(clean) == len(chain) - 10


def test_to_otm_calls_requires_forward():
    clean = clean_chain(make_chain(maturities=(0.25,)))
    fwd = implied_forward(clean, r=R)
    otm = to_otm_calls(clean, fwd)
    assert np.allclose(otm["v"], otm["call_price"] / (otm["D"] * otm["K"]))
