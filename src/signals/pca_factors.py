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

    # Mask NaN columns
    valid_cols = ~np.any(np.isnan(X), axis=0)
    X_valid = X[:, valid_cols]

    if X_valid.shape[1] < n_components:
        n_components = max(1, X_valid.shape[1])

    if standardize:
        std = X_valid.std(axis=0)
        std[std < 1e-12] = 1.0
        X_valid = X_valid / std

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_valid)   # (T × J)
    components = pca.components_.T       # (N_valid × J)

    # Expand back to full N (missing stocks get NaN loadings)
    full_loadings = np.full((N, n_components), np.nan)
    full_loadings[valid_cols] = components

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
