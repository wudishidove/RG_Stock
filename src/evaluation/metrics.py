"""
Evaluation metrics: MSFE, cumulative MSFE, OOS R².
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def msfe(
    actual: np.ndarray,
    predicted: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """
    Mean Squared Forecast Error, pooled over time and stocks.

    MSFE = (1/T) * sum_t [ (1/N_t) * sum_i (y_i - y_hat_i)^2 ]
    """
    e2 = (actual - predicted) ** 2
    if valid_mask is not None:
        e2 = np.where(valid_mask, e2, np.nan)

    # Cross-sectional mean per time step
    msfe_t = np.nanmean(e2, axis=1)   # (T,)
    return float(np.nanmean(msfe_t))


def cumulative_msfe_ratio(
    actual: np.ndarray,
    predicted_model: np.ndarray,
    predicted_baseline: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cumulative MSFE ratio: model / baseline at each time step.
    Values < 1 indicate model outperforms baseline.
    """
    e2_model = (actual - predicted_model) ** 2
    e2_base = (actual - predicted_baseline) ** 2

    if valid_mask is not None:
        e2_model = np.where(valid_mask, e2_model, np.nan)
        e2_base = np.where(valid_mask, e2_base, np.nan)

    msfe_t_model = np.nanmean(e2_model, axis=1)
    msfe_t_base = np.nanmean(e2_base, axis=1)

    cum_model = np.nancumsum(msfe_t_model)
    cum_base = np.nancumsum(msfe_t_base)

    ratio = np.where(cum_base > 0, cum_model / cum_base, np.nan)
    return ratio


def oos_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """
    Out-of-sample R²: 1 - SS_res / SS_tot.
    """
    y = actual.ravel()
    yhat = predicted.ravel()
    if valid_mask is not None:
        v = valid_mask.ravel()
        y = y[v]
        yhat = yhat[v]

    y = y[~np.isnan(y) & ~np.isnan(yhat)]
    yhat = yhat[~np.isnan(y) & ~np.isnan(yhat)]

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)
