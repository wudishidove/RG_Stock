"""
Modified z-score construction (paper equation).

z_tilde[P, t, i] = (U_hat[P, t, i] - m_P[i]) / sigma_P[i]
                   - a[i] / (kappa_P[i] * sigma_P[i])
"""

from __future__ import annotations

import numpy as np

_P_LEVELS = [10, 20, 30, 60, 100, 150]
_EPS = 1e-10


def modified_zscore(
    U_hat: np.ndarray,          # (N,) aggregated residual at current t
    drift: np.ndarray,          # (N,) OLS drift a_i
    ou_params: dict[str, np.ndarray],  # {kappa, m, sigma} for this P
    eps: float = _EPS,
) -> np.ndarray:
    """
    Compute modified z-score for a single time point and single P level.

    Parameters
    ----------
    U_hat     : (N,) aggregated residuals at time t.
    drift     : (N,) estimated OLS intercept (drift a_i).
    ou_params : dict with keys 'kappa', 'm', 'sigma', each (N,).
    eps       : floor for sigma.

    Returns
    -------
    (N,) z-scores; NaN where parameters are unavailable.
    """
    kappa = ou_params["kappa"]
    m = ou_params["m"]
    sigma = np.maximum(ou_params["sigma"], eps)

    # OU mean-reversion signal
    z1 = (U_hat - m) / sigma

    # Drift correction
    denom = kappa * sigma
    denom_safe = np.where(np.abs(denom) < eps, np.nan, denom)
    z2 = drift / denom_safe

    return z1 - z2


def build_signal_vector(
    U_hats: dict[int, np.ndarray],  # P → (N,)
    drift: np.ndarray,               # (N,)
    ou_params_all: dict[int, dict[str, np.ndarray]],  # P → {kappa, m, sigma}
    p_levels: list[int] = _P_LEVELS,
    eps: float = _EPS,
) -> np.ndarray:
    """
    Build 6-dimensional signal matrix (N × 6) from all P levels.

    Returns
    -------
    (N × D) array where D = len(p_levels). Each column is the z-score for one P.
    """
    N = drift.shape[0]
    D = len(p_levels)
    Z = np.full((N, D), np.nan)

    for j, P in enumerate(p_levels):
        if P not in U_hats or P not in ou_params_all:
            continue
        Z[:, j] = modified_zscore(U_hats[P], drift, ou_params_all[P], eps=eps)

    return Z
