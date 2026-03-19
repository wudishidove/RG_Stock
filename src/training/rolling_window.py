"""
Rolling window training loop.

At each prediction time t (paper Section 4.2):
1. Train window = [t - tau_h - M, t - tau_h)   (M bars)
2. Buffer       = [t - tau_h, t)                (tau_h bars, excluded)
3. Target       = return over [t, t+h]

tau_h >= h ensures no information leakage.
For pooled forecasting: stack all stocks × time steps in train window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cross_validation import daily_cv_lambda, _DEFAULT_LAMBDA_GRID

logger = logging.getLogger(__name__)


@dataclass
class RollingWindowConfig:
    train_window_bars: int   # M
    buffer_bars: int         # tau_h
    horizon_steps: int       # h
    bars_per_day: int = 39
    cv_lookback_days: int = 5
    cv_split: float = 0.7
    lambda_candidates: list[float] = field(
        default_factory=lambda: list(_DEFAULT_LAMBDA_GRID)
    )


def get_train_indices(
    t: int,
    cfg: RollingWindowConfig,
) -> tuple[int, int] | None:
    """
    Compute [train_start, train_end) index slice for prediction at time t.

    Paper: training over s from (t - tau_h - M) to (t - tau_h - 1).
    In Python slicing: [t - tau_h - M, t - tau_h).
    No need to subtract h because tau_h >= h (Table 7) already ensures
    no information leakage.

    Returns None if the window extends before the start of the series.
    """
    train_end = t - cfg.buffer_bars
    train_start = train_end - cfg.train_window_bars

    if train_start < 0 or train_end <= train_start:
        return None
    return train_start, train_end


def rolling_predictions(
    states: np.ndarray,        # (T, N, K) pre-computed reservoir states
    targets: np.ndarray,       # (T, N) target returns (shifted appropriately)
    valid_mask: np.ndarray,    # (T, N) bool
    cfg: RollingWindowConfig,
    lambda_: float = 1e-4,
    refit_every: int = 1,
    session_boundaries: np.ndarray | None = None,  # (T,) bool: True at first bar of each day
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate rolling predictions for all time steps and stocks.

    Returns
    -------
    predictions : (T, N) predicted returns (NaN where not predicted).
    pred_valid  : (T, N) bool — True where prediction was made.
    """
    T, N, K = states.shape
    predictions = np.full((T, N), np.nan)
    pred_valid = np.zeros((T, N), dtype=bool)

    last_beta = None
    current_lambda = lambda_

    # Pre-compute day boundaries for CV
    if session_boundaries is None:
        # Fallback: treat every bars_per_day bars as a new day
        session_boundaries = np.zeros(T, dtype=bool)
        session_boundaries[::cfg.bars_per_day] = True

    cv_lookback_bars = cfg.cv_lookback_days * cfg.bars_per_day

    for t in range(T):
        # Daily CV: select lambda at each new trading day
        if session_boundaries[t] and t >= cv_lookback_bars:
            cv_start = t - cv_lookback_bars
            cv_states = states[cv_start:t]
            cv_targets = targets[cv_start:t]
            cv_valid = valid_mask[cv_start:t]
            try:
                current_lambda = daily_cv_lambda(
                    cv_states, cv_targets, cv_valid,
                    lambda_candidates=cfg.lambda_candidates,
                    cv_split=cfg.cv_split,
                )
                logger.debug("CV at t=%d selected lambda=%.2e", t, current_lambda)
            except Exception:
                pass  # keep previous lambda

        idx = get_train_indices(t, cfg)
        if idx is None:
            continue

        train_start, train_end = idx

        # Refit readout
        if last_beta is None or t % refit_every == 0:
            train_states = states[train_start:train_end]     # (M, N, K)
            train_targets = targets[train_start:train_end]   # (M, N)
            train_valid = valid_mask[train_start:train_end]  # (M, N)

            M_actual = train_end - train_start
            flat_s = train_states.reshape(M_actual * N, K)
            flat_y = train_targets.reshape(M_actual * N)
            flat_v = train_valid.reshape(M_actual * N)

            X_fit = flat_s[flat_v]
            y_fit = flat_y[flat_v]

            if len(y_fit) < K + 2:
                continue

            # Standardize features for anisotropic ridge effect (paper: diagonal Lambda)
            scale = np.std(X_fit, axis=0)
            scale[scale < 1e-10] = 1.0
            X_fit_scaled = X_fit / scale

            X_aug = np.column_stack([np.ones(len(X_fit_scaled)), X_fit_scaled])
            XtX = X_aug.T @ X_aug
            penalty = np.eye(K + 1) * current_lambda
            penalty[0, 0] = 0.0
            try:
                beta_std = np.linalg.solve(XtX + penalty, X_aug.T @ y_fit)
            except np.linalg.LinAlgError:
                beta_std = np.linalg.lstsq(XtX + penalty, X_aug.T @ y_fit, rcond=None)[0]

            # Convert back to original scale
            last_beta = beta_std.copy()
            last_beta[1:] /= scale

        if last_beta is None:
            continue

        # Predict at time t for all stocks
        X_pred = np.column_stack([np.ones(N), states[t]])   # (N, K+1)
        preds = X_pred @ last_beta
        predictions[t] = preds
        pred_valid[t] = valid_mask[t]

    return predictions, pred_valid
