"""yfinance를 실제로 호출하는 스모크 테스트 (실행:  pytest -m network)."""

import pandas as pd
import pytest

pytestmark = pytest.mark.network


def _recent(days: int) -> str:
    return (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def test_prices_and_rate():
    from mfin.data.loaders import get_prices, get_risk_free

    px = get_prices(["SPY", "XLK"], start=_recent(30))
    assert not px.empty and list(px.columns) == ["SPY", "XLK"]
    r = get_risk_free(start=_recent(30))
    assert 0.0 <= r.iloc[-1] < 0.2


def test_volume():
    from mfin.data.loaders import get_volume

    vol = get_volume(["SPY", "XLK"], start=_recent(30))
    assert not vol.empty and list(vol.columns) == ["SPY", "XLK"]
    assert (vol.dropna() > 0).all().all()


def test_vix_term():
    from mfin.data.vix_term import latest_vix_term

    row = latest_vix_term()
    assert len(row) >= 3 and (row > 0.03).all() and (row < 1.5).all()


def test_option_chain():
    from mfin.data.option_chain import fetch_chain, prepare_chain

    chain = fetch_chain("^SPX", target_days=[30])
    assert set(chain["type"]) == {"C", "P"}
    otm, fwd = prepare_chain(chain, r=0.04)
    assert len(fwd) == 1 and len(otm) > 20
