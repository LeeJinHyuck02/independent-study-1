import numpy as np
import pytest
import torch

from mfin.pinn import InverseBSConfig, InverseBSPINN, price_error_vs_black, synthetic_quotes
from mfin.pinn.core import grad, to_tensor


def test_grad_helper():
    x = to_tensor(np.linspace(-1, 1, 7), requires_grad=True)
    y = x**3
    dy = grad(y, x)
    d2y = grad(dy, x)
    assert torch.allclose(dy, 3 * x**2) and torch.allclose(d2y, 6 * x)


@pytest.mark.parametrize("ref", [None, 0.3])
def test_initial_condition_is_exact_before_training(ref):
    x, v = synthetic_quotes(0.25, 0.2)
    m = InverseBSPINN(0.25, x, v, cfg=InverseBSConfig(ref_sigma=ref))
    xs = np.linspace(m.x_lo, m.x_hi, 101)
    assert np.allclose(m.price(np.zeros_like(xs), xs), np.maximum(xs - 1, 0), atol=1e-12)


def test_domain_scales_with_maturity():
    short = InverseBSPINN(0.05, *synthetic_quotes(0.05, 0.2, k_band=0.05))
    long = InverseBSPINN(1.0, *synthetic_quotes(1.0, 0.2))
    assert short.x_hi - short.x_lo < long.x_hi - long.x_lo
    assert short.x_lo <= 1 / np.exp(0.05) and short.x_hi >= np.exp(0.05)


def test_data_outside_domain_rejected():
    with pytest.raises(ValueError):
        InverseBSPINN(0.01, np.array([0.5, 1.0, 1.5]), np.array([0.0, 0.01, 0.5]))


@pytest.mark.slow
def test_inverse_pinn_recovers_sigma():
    T, sigma = 0.5, 0.2
    x, v = synthetic_quotes(T, sigma)
    m = InverseBSPINN(T, x, v)
    res = m.fit()
    assert abs(res.sigma - sigma) / sigma < 0.01
    assert price_error_vs_black(m, sigma)["rel_l2"] < 1e-3
