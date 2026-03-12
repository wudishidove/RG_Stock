"""
End-to-end causal signal construction pipeline.

Orchestrates PCA → residuals → OU → z-score for each time point,
using only data strictly before the current time point.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .pca_factors import extract_pca_factors
from .residuals import ols_residuals
from .ou_estimation import ou_parameters, aggregate_residuals, _P_LEVELS
from .zscore import build_signal_vector

logger = logging.getLogger(__name__)

_D = 6   # signal dimension = number of P levels


def build_signals_causal(
    returns_panel: np.ndarray,          # (T × N) log returns
    session_boundary: np.ndarray,       # (T,) bool: True = first bar of session
    pca_lookback: int,                  # bars for PCA estimation
    n_pca_components: int = 15,
    p_levels: list[int] = _P_LEVELS,
    max_missing_fraction: float = 0.20,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (T × N × D) signal array causally.

    At each time t, only uses returns[0:t] for all estimations.

    Parameters
    ----------
    returns_panel      : (T × N) log return array (NaN for missing).
    session_boundary   : (T,) True at the first bar of each session.
    pca_lookback       : number of bars for PCA + OLS estimation.
    n_pca_components   : J in PCA.
    p_levels           : OU aggregation levels.
    max_missing_fraction: exclude stocks with too many missing bars.
    eps                : numerical floor.

    Returns
    -------
    signals  : (T × N × D) signal array; NaN where unavailable.
    validity : (T × N) bool mask; True where signal is valid.
    """
    T, N = returns_panel.shape
    D = len(p_levels)
    signals = np.full((T, N, D), np.nan)
    validity = np.zeros((T, N), dtype=bool)

    min_bars_needed = max(p_levels) + n_pca_components + 2

    for t in range(T):
        if t < min_bars_needed:
            continue

        start = max(0, t - pca_lookback)
        window = returns_panel[start:t]   # strictly before t
        Tw, _ = window.shape

        # Exclude stocks with too many missing values in window
        miss_frac = np.mean(np.isnan(window), axis=0)
        valid_stocks = miss_frac <= max_missing_fraction

        if valid_stocks.sum() < n_pca_components + 1:
            continue

        # Step 2: PCA on valid stocks
        try:
            factors, _ = extract_pca_factors(
                window[:, valid_stocks],
                n_components=n_pca_components,
            )
        except Exception as e:
            logger.debug("PCA failed at t=%d: %s", t, e)
            continue

        # Step 3: OLS residuals (full N, use factor scores from valid stocks only)
        # For simplicity, project all stocks onto the same factor scores
        try:
            drift, loadings, residuals = ols_residuals(window, factors)
        except Exception as e:
            logger.debug("OLS failed at t=%d: %s", t, e)
            continue

        # Step 4: OU parameter estimation
        try:
            ou_params_all = ou_parameters(residuals, p_levels=p_levels, eps=eps)
        except Exception as e:
            logger.debug("OU estimation failed at t=%d: %s", t, e)
            continue

        # Step 5: Compute aggregated residuals at current t (last row of window)
        U_hats: dict[int, np.ndarray] = {}
        for P in p_levels:
            from .ou_estimation import aggregate_residuals as _agg
            U_all = _agg(residuals, P)
            U_hats[P] = U_all[-1]   # value at last bar of window = t-1

        # Step 6: Modified z-score
        Z_t = build_signal_vector(U_hats, drift, ou_params_all, p_levels=p_levels, eps=eps)

        signals[t] = Z_t
        validity[t] = np.all(~np.isnan(Z_t), axis=1) & valid_stocks

    return signals, validity
