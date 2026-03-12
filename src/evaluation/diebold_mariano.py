"""
Diebold-Mariano test with Newey-West HAC standard errors.

Tests equal predictive ability: H0: E[d_t] = 0
where d_t = loss_model(t) - loss_baseline(t).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def newey_west_variance(series: np.ndarray, lags: int) -> float:
    """
    Newey-West HAC variance estimate for a time series.

    Parameters
    ----------
    series : 1D array of loss differentials.
    lags   : number of lags for HAC correction.

    Returns
    -------
    Variance estimate (scalar).
    """
    T = len(series)
    d = series - series.mean()
    gamma0 = np.dot(d, d) / T

    hac_var = gamma0
    for l in range(1, lags + 1):
        gamma_l = np.dot(d[l:], d[:-l]) / T
        weight = 1.0 - l / (lags + 1)   # Bartlett kernel
        hac_var += 2.0 * weight * gamma_l

    return max(hac_var, 1e-30)


def diebold_mariano_test(
    actual: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    valid_mask: np.ndarray | None = None,
    h: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Diebold-Mariano test.

    Parameters
    ----------
    actual       : (T, N) actual returns.
    pred_model   : (T, N) model predictions.
    pred_baseline: (T, N) baseline predictions.
    valid_mask   : (T, N) bool validity mask.
    h            : forecast horizon (in bars) — used to set HAC lags.
    alternative  : 'two-sided', 'less' (model better), 'greater'.

    Returns
    -------
    dict with 'dm_stat', 'p_value', 'n_obs'.
    """
    e2_model = (actual - pred_model) ** 2
    e2_base = (actual - pred_baseline) ** 2

    if valid_mask is not None:
        e2_model = np.where(valid_mask, e2_model, np.nan)
        e2_base = np.where(valid_mask, e2_base, np.nan)

    # Cross-sectional mean loss differential per time step
    d_t = np.nanmean(e2_model - e2_base, axis=1)   # (T,)
    d_t = d_t[~np.isnan(d_t)]

    T = len(d_t)
    if T < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "n_obs": T}

    lags = max(1, h - 1)
    hac_var = newey_west_variance(d_t, lags=lags)
    se = np.sqrt(hac_var / T)

    dm_stat = d_t.mean() / se if se > 0 else np.nan

    if alternative == "two-sided":
        p_value = 2.0 * stats.norm.sf(abs(dm_stat))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    else:
        p_value = stats.norm.sf(dm_stat)

    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "n_obs": int(T)}
