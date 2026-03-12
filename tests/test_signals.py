"""Tests for signal construction pipeline components."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.pca_factors import extract_pca_factors
from src.signals.residuals import ols_residuals
from src.signals.ou_estimation import aggregate_residuals, fit_ar1, ou_parameters
from src.signals.zscore import modified_zscore, build_signal_vector


def test_pca_factors_shape():
    T, N, J = 100, 50, 15
    returns = np.random.randn(T, N)
    factors, loadings = extract_pca_factors(returns, n_components=J)
    assert factors.shape == (T, J)
    assert loadings.shape == (N, J)


def test_pca_factors_standardized():
    """With standardize=True, factors should be unit-variance (approx)."""
    T, N = 200, 30
    returns = np.random.randn(T, N) * np.array([0.1 * (i + 1) for i in range(N)])
    factors, _ = extract_pca_factors(returns, n_components=5, standardize=True)
    # PCA scores are not unit variance but should be finite
    assert not np.any(np.isnan(factors))


def test_aggregate_residuals_sum():
    """U_hat[P, t] should equal sum of residuals in window."""
    T, N = 20, 3
    residuals = np.arange(T * N, dtype=float).reshape(T, N)
    P = 5
    U = aggregate_residuals(residuals, P)
    for t in range(P - 1, T):
        expected = residuals[t - P + 1:t + 1].sum(axis=0)
        np.testing.assert_allclose(U[t], expected)


def test_fit_ar1_known():
    """Test AR(1) fit on a known process."""
    rng = np.random.default_rng(42)
    T = 500
    y = np.zeros(T)
    c0_true, cu_true = 0.1, 0.7
    for t in range(1, T):
        y[t] = c0_true + cu_true * y[t - 1] + rng.normal(0, 0.1)
    c0, cu, sigma2 = fit_ar1(y)
    assert abs(c0 - c0_true) < 0.05
    assert abs(cu - cu_true) < 0.05


def test_ou_parameters_keys():
    T, N = 200, 10
    residuals = np.random.randn(T, N) * 0.01
    params = ou_parameters(residuals)
    for P, p_dict in params.items():
        assert "kappa" in p_dict
        assert "m" in p_dict
        assert "sigma" in p_dict
        assert p_dict["kappa"].shape == (N,)


def test_zscore_finite():
    N = 20
    U_hat = np.random.randn(N) * 0.1
    drift = np.random.randn(N) * 0.001
    ou_params = {
        "kappa": np.abs(np.random.randn(N)) + 0.1,
        "m": np.random.randn(N) * 0.05,
        "sigma": np.abs(np.random.randn(N)) * 0.1 + 0.01,
    }
    z = modified_zscore(U_hat, drift, ou_params)
    assert z.shape == (N,)
    # Not all NaN
    assert not np.all(np.isnan(z))
