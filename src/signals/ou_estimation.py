"""
OU parameter estimation from aggregated idiosyncratic residuals.

Six aggregation levels P ∈ {10, 20, 30, 60, 100, 150} (in 10-min bars).
For each P, fit AR(1) on cumulative sum U_hat and extract kappa, m, sigma.
"""

from __future__ import annotations

import numpy as np

_P_LEVELS = [10, 20, 30, 60, 100, 150]
_EPS = 1e-10


def aggregate_residuals(residuals: np.ndarray, P: int) -> np.ndarray:
    """
    Compute aggregated residuals U_hat[P, t] = sum_{s=t-P}^{t} v_hat[s].

    Paper equation: P+1 terms (from t-P to t inclusive).

    Parameters
    ----------
    residuals : (T × N) idiosyncratic residuals.
    P         : aggregation window parameter.

    Returns
    -------
    (T × N) aggregated residuals (first P rows are NaN).
    """
    T, N = residuals.shape
    U = np.full_like(residuals, np.nan)
    if T < P + 1:
        return U
    # Cumulative sum over rolling window of size P+1 (paper: sum from t-P to t)
    cs = np.nancumsum(residuals, axis=0)
    U[P:] = cs[P:]
    U[P + 1:] -= cs[:T - P - 1]
    return U


def fit_ar1(series: np.ndarray) -> tuple[float, float, float]:
    """
    Fit AR(1): y_{t+1} = c0 + cu * y_t + eta.

    Returns (c0, cu, sigma_eta^2). Returns (nan, nan, nan) if fit fails.
    """
    y = series[~np.isnan(series)]
    if len(y) < 4:
        return np.nan, np.nan, np.nan
    y_lag = y[:-1]
    y_fwd = y[1:]
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_fwd, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    c0, cu = coeffs
    eta = y_fwd - X @ coeffs
    sigma2_eta = np.var(eta)
    return float(c0), float(cu), float(sigma2_eta)


def ou_parameters(
    residuals_window: np.ndarray,
    p_levels: list[int] = _P_LEVELS,
    eps: float = _EPS,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Estimate OU parameters for each aggregation level P.

    Parameters
    ----------
    residuals_window : (T × N) residuals in estimation window.
    p_levels         : list of P values.
    eps              : numerical stability floor.

    Returns
    -------
    dict: P → {kappa: (N,), m: (N,), sigma: (N,)}
    """
    T, N = residuals_window.shape
    result: dict[int, dict[str, np.ndarray]] = {}

    for P in p_levels:
        U = aggregate_residuals(residuals_window, P)   # (T × N)
        kappa = np.full(N, np.nan)
        m = np.full(N, np.nan)
        sigma = np.full(N, np.nan)

        for i in range(N):
            c0, cu, sigma2_eta = fit_ar1(U[:, i])
            if np.isnan(cu):
                continue
            # Stability safeguard
            cu_clipped = np.clip(cu, 0.01, 0.98)
            kappa_i = -np.log(cu_clipped)
            m_i = c0 / (1.0 - cu_clipped)
            if kappa_i < 0.01 or sigma2_eta < 0:
                continue
            sigma_i = np.sqrt(max(sigma2_eta, 0.0) / (2.0 * kappa_i))
            kappa[i] = kappa_i
            m[i] = m_i
            sigma[i] = max(sigma_i, 1e-4)

        result[P] = {"kappa": kappa, "m": m, "sigma": sigma}

    return result
