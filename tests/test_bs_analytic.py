import numpy as np
import pytest

from mfin.engine.bs_analytic import (
    black_call,
    black_price,
    black_put,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
)


def test_textbook_value():
    # Hull: S=K=100, T=1, r=5%, sigma=20%  ->  10.4506
    assert bs_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.450583572185565, abs=1e-12)


@pytest.mark.parametrize("S,K,T,r,q,sig", [(100, 90, 0.5, 0.03, 0.01, 0.25), (6000, 6300, 0.1, 0.04, 0.013, 0.15)])
def test_put_call_parity(S, K, T, r, q, sig):
    c = bs_price(S, K, T, r, sig, q, "call")
    p = bs_price(S, K, T, r, sig, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), rel=1e-12)


def test_zero_variance_is_payoff():
    x = np.linspace(0.5, 1.5, 11)
    assert np.allclose(black_call(x, 0.0), np.maximum(x - 1, 0))
    assert np.allclose(black_call(x, 1e-12), np.maximum(x - 1, 0), atol=1e-6)
    assert np.allclose(black_put(x, 0.0), np.maximum(1 - x, 0))


def test_normalised_form_matches_dimensional():
    F, K, T, sig, D = 6100.0, 6000.0, 0.3, 0.18, 0.99
    assert black_price(F, K, T, sig, D) == pytest.approx(D * K * black_call(F / K, sig**2 * T), rel=1e-14)


def test_greeks_against_finite_differences():
    S, K, T, r, q, sig, h = 100.0, 105.0, 0.75, 0.03, 0.01, 0.22, 1e-4
    price = lambda s=S, v=sig: bs_price(s, K, T, r, v, q)  # noqa: E731
    assert bs_delta(S, K, T, r, sig, q) == pytest.approx((price(S + h) - price(S - h)) / (2 * h), rel=1e-7)
    assert bs_gamma(S, K, T, r, sig, q) == pytest.approx(
        (price(S + h) - 2 * price() + price(S - h)) / h**2, rel=1e-4
    )
    assert bs_vega(S, K, T, r, sig, q) == pytest.approx((price(v=sig + h) - price(v=sig - h)) / (2 * h), rel=1e-7)
