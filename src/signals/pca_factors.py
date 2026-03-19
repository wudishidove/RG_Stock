"""
Causal rolling PCA market factor extraction.

All estimation strictly uses data before the current time point.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def extract_pca_factors(
    returns: np.ndarray,
    n_components: int = 15,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on a (T × N) return matrix and extract factor time series.

    Parameters
    ----------
    returns      : (T × N) array; rows = time, cols = stocks.
    n_components : number of principal components (J=15).
    standardize  : if True, standardize each stock before PCA (per Appendix B).

    Returns
    -------
    factors    : (T × J) array of factor returns.
    loadings   : (N × J) matrix of factor loadings.
    """
    T, N = returns.shape
    X = returns.copy()

    # Impute NaN with column means (instead of dropping entire columns)
    nan_mask = np.isnan(X)
    col_has_data = ~np.all(nan_mask, axis=0)  # columns with at least one non-NaN
    n_valid_cols = col_has_data.sum()

    if n_valid_cols < n_components:
        n_components = max(1, n_valid_cols)

    # Fill NaN with column mean for columns that have data
    col_means = np.nanmean(X[:, col_has_data], axis=0)
    X_valid = X[:, col_has_data].copy()
    for j in range(X_valid.shape[1]):
        col_nans = np.isnan(X_valid[:, j])
        if col_nans.any():
            X_valid[col_nans, j] = col_means[j]

    if standardize:
        std = X_valid.std(axis=0)
        std[std < 1e-12] = 1.0
        X_valid = X_valid / std

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_valid)   # (T × J)
    components = pca.components_.T       # (N_valid × J)

    # Expand back to full N (columns with no data get NaN loadings)
    full_loadings = np.full((N, n_components), np.nan)
    full_loadings[col_has_data] = components

    return scores, full_loadings


def rolling_pca_factors(
    returns: np.ndarray,
    timestamps: list,
    current_idx: int,
    lookback: int,
    n_components: int = 15,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extract PCA factors using only data in [current_idx - lookback, current_idx).
    Strictly causal: does NOT include current_idx.

    Returns (factors, loadings) or (None, None) if insufficient data.
    """
    start = max(0, current_idx - lookback)
    end = current_idx   # exclusive
    if end - start < n_components + 1:
        return None, None

    window = returns[start:end]
    return extract_pca_factors(window, n_components=n_components)
