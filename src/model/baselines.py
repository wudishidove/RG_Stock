"""
Baselines for comparison:
  1. Linear baseline: OLS directly on Z_t (no reservoir).
  2. Benchmark ridge: Ridge on Z_t, same windowing as ESN.
"""

from __future__ import annotations

import numpy as np


class LinearBaseline:
    """OLS regression directly on signal vector Z_t."""

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, signals: np.ndarray, targets: np.ndarray) -> "LinearBaseline":
        """
        Parameters
        ----------
        signals : (T_N, D) stacked signal vectors.
        targets : (T_N,) stacked target returns.
        """
        T_N, D = signals.shape
        X = np.column_stack([np.ones(T_N), signals])
        try:
            self.beta_ = np.linalg.lstsq(X, targets, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.beta_ = np.zeros(D + 1)
        self.is_fitted = True
        return self

    def predict(self, signals: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        if signals.ndim == 1:
            X = np.concatenate([[1.0], signals])
        else:
            X = np.column_stack([np.ones(len(signals)), signals])
        return X @ self.beta_


class BenchmarkRidge:
    """Ridge regression directly on signal vector (same CV scheme as ESN)."""

    def __init__(self, lambda_: float = 1e-4) -> None:
        self.lambda_ = lambda_
        self.beta_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, signals: np.ndarray, targets: np.ndarray) -> "BenchmarkRidge":
        T_N, D = signals.shape
        X = np.column_stack([np.ones(T_N), signals])
        XtX = X.T @ X
        penalty = np.eye(D + 1) * self.lambda_
        penalty[0, 0] = 0.0
        try:
            self.beta_ = np.linalg.solve(XtX + penalty, X.T @ targets)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(XtX + penalty, X.T @ targets, rcond=None)[0]
        self.is_fitted = True
        return self

    def predict(self, signals: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        if signals.ndim == 1:
            X = np.concatenate([[1.0], signals])
        else:
            X = np.column_stack([np.ones(len(signals)), signals])
        return X @ self.beta_
