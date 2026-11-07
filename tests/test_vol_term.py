import numpy as np
import pandas as pd
import pytest

from mfin.engine.vol_term import DEFAULT_SENSORS, forward_variance_knots, forward_vol_curve, vix_forward_vol_curve


def test_constant_vol_gives_constant_forward_vol():
    taus = [0.1, 0.25, 0.5, 1.0]
    assert np.allclose(forward_vol_curve(taus, [0.2] * 4), 0.2)


def test_piecewise_forward_variance_recovered():
    # forward vol 0.1 on (0, 0.5], 0.3 on (0.5, 1]: w(0.5) = 0.005, w(1) = 0.05
    taus = [0.5, 1.0]
    sigmas = [np.sqrt(0.005 / 0.5), np.sqrt(0.05)]
    out = forward_vol_curve(taus, sigmas, sensors=[0.25, 0.5, 0.75, 1.0, 1.5])
    assert out == pytest.approx([0.1, 0.1, 0.3, 0.3, 0.3], abs=1e-12)


def test_calendar_arbitrage_is_repaired():
    # total variance decreases (0.045 -> 0.01): forward variance would be negative
    knots, fwd_var = forward_variance_knots([0.5, 1.0], [0.3, 0.1])
    assert np.all(fwd_var > 0)
    assert np.all(np.isfinite(forward_vol_curve([0.5, 1.0], [0.3, 0.1])))


def test_unsorted_duplicates_and_nans():
    out = forward_vol_curve([1.0, np.nan, 0.5, 0.5], [0.2, 0.3, 0.2, 0.2], sensors=[0.25, 0.75])
    assert out == pytest.approx([0.2, 0.2])


def test_vix_row():
    row = pd.Series({9 / 365: 0.15, 30 / 365: 0.16, 0.25: 0.18, 0.5: 0.19, 1.0: np.nan})
    out = vix_forward_vol_curve(row)
    assert out.shape == DEFAULT_SENSORS.shape
    assert np.all(out > 0.14) and np.all(out < 0.25)
