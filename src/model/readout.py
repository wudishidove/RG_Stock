"""
Readout layer: linear regression from reservoir states to target returns.

Phase A: scalar ridge regression — (X'X + lambda*I)^{-1} X'y
Phase B (future): diagonal anisotropic ridge.

Pooled across all stocks in the training window.
"""

from __future__ import annotations

import numpy as np


class RidgeReadout:
    """Linear readout with scalar ridge penalty (Phase A)."""

    def __init__(self, lambda_: float = 1e-4) -> None:
        self.lambda_ = lambda_
        self.beta_: np.ndarray | None = None   # (K+1,) including intercept
        self.is_fitted: bool = False

    def fit(
        self,
        states: np.ndarray,   # (T_train * N_valid, K) stacked reservoir states
        targets: np.ndarray,  # (T_train * N_valid,) stacked target returns
    ) -> "RidgeReadout":
        """
        Fit pooled ridge regression.

        Design matrix X = [1, states] — intercept not penalized.
        """
        T_N, K = states.shape
        # Augment with intercept column
        X = np.column_stack([np.ones(T_N), states])   # (T_N, K+1)

        # Ridge: penalize all but intercept
        # (X'X + Lambda)^{-1} X'y  where Lambda[0,0] = 0, Lambda[k,k] = lambda_
        XtX = X.T @ X
        penalty = np.eye(K + 1) * self.lambda_
        penalty[0, 0] = 0.0   # don't penalize intercept
        try:
            self.beta_ = np.linalg.solve(XtX + penalty, X.T @ targets)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.lstsq(XtX + penalty, X.T @ targets, rcond=None)[0]
        self.is_fitted = True
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        Predict returns from reservoir states.

        Parameters
        ----------
        states : (T, K) or (K,) reservoir states.

        Returns
        -------
        (T,) or scalar predictions.
        """
        if not self.is_fitted or self.beta_ is None:
            raise RuntimeError("Readout not fitted")
        if states.ndim == 1:
            x = np.concatenate([[1.0], states])
            return float(self.beta_ @ x)
        X = np.column_stack([np.ones(len(states)), states])
        return X @ self.beta_
