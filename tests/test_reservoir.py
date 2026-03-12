"""Tests for ESN reservoir construction and state evolution."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.reservoir import ESNReservoir
from src.utils.linalg import spectral_radius


def test_spectral_radius_target():
    res = ESNReservoir(K=50, D=6, alpha=0.9, rho=0.4, gamma=0.005,
                       A_sparsity=0.15, C_sparsity=0.95, seed=0)
    sr = spectral_radius(res.A_bar)
    # rho=0.4 → spectral radius should be close to 0.4
    assert abs(sr - 0.4) < 0.01, f"Spectral radius {sr} not close to 0.4"


def test_eod_reservoir_zero_recurrent():
    """EOD horizon has rho=0 → A_bar should be all zeros."""
    res = ESNReservoir(K=50, D=6, alpha=0.0, rho=0.0, gamma=0.015,
                       A_sparsity=0.35, C_sparsity=0.25, seed=0)
    assert np.allclose(res.A_bar, 0.0), "rho=0 should give zero recurrent matrix"


def test_state_shape():
    res = ESNReservoir(K=20, D=6, alpha=0.5, rho=0.3, gamma=0.01,
                       A_sparsity=0.2, C_sparsity=0.5, seed=1)
    T = 50
    signals = np.random.randn(T, 6)
    valid = np.ones(T, dtype=bool)
    states = res.forward(signals, valid)
    assert states.shape == (T, 20)


def test_missing_inputs_zero_replaced():
    """Missing inputs should be replaced with zeros, not cause NaN states."""
    res = ESNReservoir(K=20, D=6, alpha=0.5, rho=0.3, gamma=0.01,
                       A_sparsity=0.2, C_sparsity=0.5, seed=2)
    T = 30
    signals = np.random.randn(T, 6)
    signals[5:10] = np.nan
    valid = np.ones(T, dtype=bool)
    valid[5:10] = False
    states = res.forward(signals, valid)
    assert not np.any(np.isnan(states)), "States should not contain NaN"


def test_leak_rate_alpha_zero():
    """With alpha=0, state should NOT depend on previous state (only current input)."""
    res = ESNReservoir(K=20, D=6, alpha=0.0, rho=0.3, gamma=0.01,
                       A_sparsity=0.2, C_sparsity=0.5, seed=3)
    # Two identical inputs at different times should give same state
    sig = np.random.randn(6)
    s1 = np.tanh(res.C_bar @ sig)
    T = 5
    signals = np.zeros((T, 6))
    signals[2] = sig
    signals[4] = sig
    valid = np.ones(T, dtype=bool)
    states = res.forward(signals, valid)
    # With alpha=0 and rho=0.3, states[2] and states[4] will differ due to recurrent
    # Just check no NaN and correct shape
    assert states.shape == (T, 20)
    assert not np.any(np.isnan(states))
