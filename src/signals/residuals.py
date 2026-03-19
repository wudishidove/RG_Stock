"""
Rolling causal OLS factor regression to extract idiosyncratic residuals.
"""

from __future__ import annotations

import numpy as np


def ols_residuals(
    returns: np.ndarray,
    factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate r_i = a_i + B_i @ F_t + v_i via OLS.

    Parameters
    ----------
    returns : (T × N) idiosyncratic return slice.
    factors : (T × J) factor return slice.

    Returns
    -------
    drift     : (N,) intercept a_i
    loadings  : (N × J) factor loadings B_i
    residuals : (T × N) idiosyncratic residuals v_t
    """
    T, N = returns.shape
    T_f, J = factors.shape
    assert T == T_f, "returns and factors must have same length"

    # Design matrix: [1, F_1, ..., F_J]
    X = np.column_stack([np.ones(T), factors])   # (T × (J+1))

    # Per-time valid mask for factors
    factor_nan = np.any(np.isnan(factors), axis=1)   # (T,) True where factors NaN

    drift = np.full(N, np.nan)
    loadings = np.full((N, J), np.nan)
    residuals = np.full((T, N), np.nan)

    for i in range(N):
        y_i = returns[:, i]
        # Per-stock: use rows where both factors and this stock's return are valid
        valid_rows = ~np.isnan(y_i) & ~factor_nan
        if valid_rows.sum() < J + 2:
            continue
        X_valid = X[valid_rows]
        y_valid = y_i[valid_rows]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
        except np.linalg.LinAlgError:
            continue
        drift[i] = coeffs[0]
        loadings[i] = coeffs[1:]
        # Compute residuals for all rows (NaN rows propagate NaN naturally)
        residuals[:, i] = returns[:, i] - X @ np.concatenate([[coeffs[0]], coeffs[1:]])

    return drift, loadings, residuals
