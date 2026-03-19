"""Tests for vectorized reservoir: equivalence with per-stock loop version."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.reservoir_fast import ESNReservoirFast


def test_vectorized_equals_loop():
    """forward_all_stocks_vectorized must be bit-identical to per-stock forward loop."""
    res = ESNReservoirFast(K=30, D=6, alpha=0.5, rho=0.4, gamma=0.01,
                           A_sparsity=0.2, C_sparsity=0.5, seed=42)
    T, N = 100, 20
    rng = np.random.default_rng(99)
    signals = rng.standard_normal((T, N, 6))
    validity = rng.random((T, N)) > 0.1  # ~10% invalid

    expected = res.forward_all_stocks(signals, validity)
    actual = res.forward_all_stocks_vectorized(signals, validity)

    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-10)


def test_vectorized_with_nan_signals():
    """valid=True but signal contains NaN — both versions must behave identically."""
    res = ESNReservoirFast(K=20, D=6, alpha=0.9, rho=0.3, gamma=0.005,
                           A_sparsity=0.15, C_sparsity=0.95, seed=7)
    T, N = 50, 10
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((T, N, 6))
    signals[3, 2, :] = np.nan   # valid=True but full NaN signal
    signals[10, 5, 1] = np.nan  # partial NaN
    validity = np.ones((T, N), dtype=bool)

    expected = res.forward_all_stocks(signals, validity)
    actual = res.forward_all_stocks_vectorized(signals, validity)

    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-10)


def test_vectorized_all_invalid():
    """All invalid — states should be all zeros."""
    res = ESNReservoirFast(K=20, D=6, alpha=0.0, rho=0.0, gamma=0.01,
                           A_sparsity=0.2, C_sparsity=0.5, seed=5)
    T, N = 20, 5
    rng = np.random.default_rng(77)
    signals = rng.standard_normal((T, N, 6))
    validity = np.zeros((T, N), dtype=bool)

    expected = res.forward_all_stocks(signals, validity)
    actual = res.forward_all_stocks_vectorized(signals, validity)

    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-10)
    assert np.allclose(actual, 0.0)


def test_vectorized_eod_rho_zero():
    """EOD horizon: rho=0, alpha=0 — pure feedforward, results must match."""
    res = ESNReservoirFast(K=30, D=6, alpha=0.0, rho=0.0, gamma=0.015,
                           A_sparsity=0.35, C_sparsity=0.25, seed=0)
    T, N = 80, 15
    rng = np.random.default_rng(123)
    signals = rng.standard_normal((T, N, 6))
    validity = rng.random((T, N)) > 0.2

    expected = res.forward_all_stocks(signals, validity)
    actual = res.forward_all_stocks_vectorized(signals, validity)

    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-10)


def test_vectorized_high_alpha():
    """High leak rate (alpha=0.99) — strong state persistence, results must match."""
    res = ESNReservoirFast(K=25, D=6, alpha=0.99, rho=0.5, gamma=0.01,
                           A_sparsity=0.3, C_sparsity=0.6, seed=11)
    T, N = 60, 12
    rng = np.random.default_rng(55)
    signals = rng.standard_normal((T, N, 6))
    validity = rng.random((T, N)) > 0.05

    expected = res.forward_all_stocks(signals, validity)
    actual = res.forward_all_stocks_vectorized(signals, validity)

    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-10)
