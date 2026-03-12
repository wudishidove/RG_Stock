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

    # Valid stocks: no NaN in either returns or factors
    factor_nan = np.any(np.isnan(factors), axis=1)   # (T,)
    returns_nan = np.any(np.isnan(returns), axis=0)  # (N,) — stocks with any NaN

    drift = np.full(N, np.nan)
    loadings = np.full((N, J), np.nan)
    residuals = np.full((T, N), np.nan)

    if factor_nan.any():
        valid_t = ~factor_nan
        X_clean = X[valid_t]
        R_clean = returns[valid_t]
    else:
        X_clean = X
        R_clean = returns

    for i in range(N):
        if returns_nan[i]:
            continue
        y = R_clean[:, i]
        if np.any(np.isnan(y)):
            continue
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_clean, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        drift[i] = coeffs[0]
        loadings[i] = coeffs[1:]
        resid_full = returns[:, i] - X @ np.concatenate([[coeffs[0]], coeffs[1:]])
        residuals[:, i] = resid_full

    return drift, loadings, residuals
