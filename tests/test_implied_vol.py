import itertools

import numpy as np
import pytest

from mfin.engine.bs_analytic import black_price, black_vega_s
from mfin.engine.implied_vol import fit_constant_vol, implied_vol, implied_vols


@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip(kind):
    F, D = 100.0, 0.98
    for sig, x, T in itertools.product([0.05, 0.2, 0.5, 1.0], [0.9, 1.0, 1.1], [0.02, 0.5, 2.0]):
        K = F / x
        if black_vega_s(x, sig * np.sqrt(T)) < 1e-6:  # 가격에 변동성 정보가 거의 없다 (vega ~ 0)
            continue
        price = black_price(F, K, T, sig, D, kind)
        assert implied_vol(price, F, K, T, D, kind) == pytest.approx(sig, abs=1e-8)


def test_out_of_bounds_is_nan():
    F, K, T = 100.0, 100.0, 0.5
    assert np.isnan(implied_vol(-1.0, F, K, T))
    assert np.isnan(implied_vol(F + 1.0, F, K, T))  # 선도가격보다 비싸다 (콜 상한 위반)
    assert np.isnan(implied_vol(5.0, F, 90.0, T))  # 내재가치 10보다 싸다 (하한 위반)


def test_vectorised():
    K = np.array([90.0, 100.0, 110.0])
    prices = black_price(100.0, K, 0.25, 0.3)
    assert np.allclose(implied_vols(prices, 100.0, K, 0.25), 0.3, atol=1e-9)


def test_constant_vol_least_squares():
    T, sig = 0.4, 0.17
    x = np.exp(-np.linspace(-0.1, 0.1, 15))
    from mfin.engine.bs_analytic import black_call

    v = black_call(x, sig**2 * T)
    assert fit_constant_vol(x, v, T) == pytest.approx(sig, abs=1e-9)
