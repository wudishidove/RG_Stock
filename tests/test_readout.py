"""Tests for readout regression."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.readout import RidgeReadout


def test_fit_predict_shape():
    T, K = 200, 20
    states = np.random.randn(T, K)
    targets = np.random.randn(T)
    readout = RidgeReadout(lambda_=1e-3)
    readout.fit(states, targets)
    preds = readout.predict(states)
    assert preds.shape == (T,)


def test_ridge_reduces_overfitting():
    """High lambda should reduce variance (preds closer to mean)."""
    T, K = 50, 40
    states = np.random.randn(T, K)
    targets = np.random.randn(T)

    r_low = RidgeReadout(lambda_=1e-10)
    r_low.fit(states, targets)
    preds_low = r_low.predict(states)

    r_high = RidgeReadout(lambda_=1e3)
    r_high.fit(states, targets)
    preds_high = r_high.predict(states)

    # High lambda → predictions closer to mean (lower std)
    assert preds_high.std() < preds_low.std()


def test_unfitted_raises():
    readout = RidgeReadout()
    with pytest.raises(RuntimeError):
        readout.predict(np.zeros((5, 10)))


def test_beta_shape():
    T, K = 100, 10
    states = np.random.randn(T, K)
    targets = np.random.randn(T)
    readout = RidgeReadout(lambda_=1e-4)
    readout.fit(states, targets)
    assert readout.beta_.shape == (K + 1,)   # K weights + 1 intercept
