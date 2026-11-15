"""모듈 C 추정량: Ledoit–Wolf, EWMA, 베타, 모멘텀, 기대수익 모형, 미래참조 방지."""

import numpy as np
import pandas as pd
import pytest

from mfin.portfolio.estimators import (
    EstimatorConfig,
    betas_and_resid_vol,
    cross_sectional_z,
    estimate_state,
    ewma_vol,
    expected_return_model,
    ledoit_wolf_corr,
    momentum_12_1,
    vix_knots,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def synthetic_panel(n_rows=400, seed=0):
    """요인 구조를 가진 가격 패널 (자산 5개 + 벤치마크 SPY)."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, n_rows)
    betas = np.array([1.2, 0.9, 1.0, 1.4, 0.6])
    idio = rng.normal(0, 0.008, (n_rows, 5))
    rets = np.column_stack([mkt[:, None] * betas + idio, mkt])
    prices = 100 * np.cumprod(1 + rets, axis=0)
    idx = pd.bdate_range("2020-01-01", periods=n_rows)
    return pd.DataFrame(prices, index=idx, columns=TICKERS + ["SPY"])


def random_corr(rng, n):
    F = rng.normal(size=(n, 2))
    C = F @ F.T + np.diag(rng.uniform(0.2, 1.0, n))
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


# -------------------------------------------------------------------------- Ledoit–Wolf


def test_ledoit_wolf_properties():
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(np.zeros(5), random_corr(rng, 5), size=100)
    R, delta = ledoit_wolf_corr(X)
    assert 0.0 <= delta <= 1.0
    assert np.max(np.abs(R - R.T)) < 1e-14
    assert np.diag(R) == pytest.approx(1.0, abs=1e-14)
    assert np.linalg.eigvalsh(R).min() > 0
    R_hat = np.corrcoef(X, rowvar=False)
    rbar = R_hat[~np.eye(5, dtype=bool)].mean()
    target = (1 - rbar) * np.eye(5) + rbar
    assert R == pytest.approx((1 - delta) * R_hat + delta * target, abs=1e-14)


def test_ledoit_wolf_beats_sample_when_truth_is_constant_correlation():
    rng = np.random.default_rng(1)
    true = 0.4 * np.eye(5) + 0.6
    err_lw, err_s = [], []
    for _ in range(200):
        X = rng.multivariate_normal(np.zeros(5), true, size=60)
        err_lw.append(np.linalg.norm(ledoit_wolf_corr(X)[0] - true))
        err_s.append(np.linalg.norm(np.corrcoef(X, rowvar=False) - true))
    assert np.mean(err_lw) < np.mean(err_s)


def test_ledoit_wolf_shrinkage_vanishes_with_long_samples():
    rng = np.random.default_rng(2)
    X = rng.multivariate_normal(np.zeros(5), random_corr(rng, 5), size=20000)
    assert ledoit_wolf_corr(X)[1] < 0.05


# ---------------------------------------------------------------------- 변동성, 베타, 모멘텀


def test_ewma_scale_and_constant_series():
    rng = np.random.default_rng(3)
    b = rng.normal(0, 0.01, 252)
    vol = ewma_vol(np.column_stack([2 * b, b]), span=63)
    assert vol[0] / vol[1] == pytest.approx(2.0, abs=1e-12)
    c = np.full((100, 1), 0.01)
    assert ewma_vol(c, span=20)[0] == pytest.approx(0.01 * np.sqrt(252), rel=1e-12)


def test_beta_and_residual_vol_exact():
    rng = np.random.default_rng(4)
    b = rng.normal(0, 0.01, 252)
    beta, omega = betas_and_resid_vol(np.column_stack([1.5 * b, -0.5 * b + 0.001]), b)
    assert beta == pytest.approx([1.5, -0.5], abs=1e-12)
    assert omega == pytest.approx([0.0, 0.0], abs=1e-12)


def test_momentum_on_geometric_path():
    g = 1.001
    P = 100 * g ** np.arange(300)[:, None] * np.ones((1, 2))
    assert momentum_12_1(P, 252, 21) == pytest.approx(g ** (252 - 21) - 1, abs=1e-12)
    with pytest.raises(ValueError):
        momentum_12_1(P[:100])


def test_cross_sectional_z():
    z = cross_sectional_z([1.0, 2.0, 3.0, 100.0], clip=1.5)
    assert z.max() == 1.5
    assert cross_sectional_z([2.0, 2.0, 2.0]) == pytest.approx([0, 0, 0])


def test_expected_return_model():
    cfg = EstimatorConfig()
    beta = np.array([1.2, 0.8, 1.0])
    omega = np.array([0.10, 0.15, 0.08])
    mom = np.array([0.30, -0.10, 0.05])
    mu0, mu_inf, kappa = expected_return_model(0.03, beta, omega, mom, cfg)
    assert mu_inf == pytest.approx(0.03 + beta * 0.05, abs=1e-15)
    z = cross_sectional_z(mom)
    assert mu0 - mu_inf == pytest.approx(cfg.ic * omega * z * np.sqrt(12), abs=1e-15)
    assert z.sum() == pytest.approx(0.0, abs=1e-12)
    assert kappa == pytest.approx(np.log(2) / 0.5, abs=1e-15)


# -------------------------------------------------------------------------------- VIX


def test_vix_knots_drop_missing_tenors():
    row = pd.Series({9 / 365: np.nan, 30 / 365: 0.16, 0.25: 0.18, 0.5: np.nan, 1.0: 0.20})
    knots, fwd = vix_knots(row)
    assert knots == pytest.approx([0, 30 / 365, 0.25, 1.0])
    assert np.all(fwd > 0)


def test_vix_knots_require_30_day_index():
    with pytest.raises(ValueError):
        vix_knots(pd.Series({30 / 365: np.nan, 0.25: 0.18}))
    with pytest.raises(ValueError):
        vix_knots(pd.Series({9 / 365: np.nan, 30 / 365: np.nan}))


# ------------------------------------------------------------------------------- 상태 추정


def test_estimate_state_shapes_and_ewma_fallback():
    panel = synthetic_panel()
    st = estimate_state(panel, "SPY", 0.03, None)
    assert st.tickers == TICKERS and st.vol_source == "ewma"
    assert st.coeffs.n == 5 and st.prices.shape == (5,)
    assert st.coeffs.fwd_var[0] == pytest.approx(st.diagnostics["bench_vol_ewma"] ** 2)
    assert st.asof == panel.index[-1]
    assert st.coeffs.s == pytest.approx(st.sigma_ann / st.diagnostics["bench_vol_ewma"])


def test_estimate_state_uses_given_vol_knots():
    panel = synthetic_panel()
    knots = (np.array([0.0, 0.1, 1.0]), np.array([0.04, 0.03]))
    st = estimate_state(panel, "SPY", 0.03, knots, vol_source="vix")
    assert st.vol_source == "vix"
    assert st.coeffs.fwd_var == pytest.approx([0.04, 0.03])


def test_estimate_state_rejects_short_or_missing_data():
    panel = synthetic_panel()
    with pytest.raises(ValueError):
        estimate_state(panel.iloc[:200], "SPY", 0.03, None)
    bad = panel.copy()
    bad.iloc[-5, 0] = np.nan
    with pytest.raises(ValueError):
        estimate_state(bad, "SPY", 0.03, None)


def test_no_look_ahead():
    panel = synthetic_panel()
    i = 330
    a = estimate_state(panel.iloc[:i], "SPY", 0.03, None)
    future = panel.copy()
    future.iloc[i:] = 1e6
    b = estimate_state(future.iloc[:i], "SPY", 0.03, None)
    for name in ("s", "R", "mu0", "mu_inf", "fwd_var"):
        assert np.array_equal(getattr(a.coeffs, name), getattr(b.coeffs, name))
    assert np.array_equal(a.prices, b.prices)


def test_adv_from_volume():
    panel = synthetic_panel()
    vol = pd.DataFrame(np.tile(np.arange(1.0, 401.0)[:, None], (1, 6)), index=panel.index, columns=panel.columns)
    st = estimate_state(panel, "SPY", 0.03, None, volume=vol)
    assert st.adv_shares == pytest.approx(np.full(5, np.median(np.arange(338.0, 401.0))))
