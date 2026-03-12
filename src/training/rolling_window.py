"""
Rolling window training loop.

At each prediction time t:
1. Train window = [t - M - tau_h - h, t - tau_h - h)
2. Buffer       = [t - tau_h - h, t - h)  (excluded from training)
3. Target       = return over [t, t+h]

For pooled forecasting: stack all stocks × time steps in train window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RollingWindowConfig:
    train_window_bars: int   # M
    buffer_bars: int         # tau_h
    horizon_steps: int       # h


def get_train_indices(
    t: int,
    cfg: RollingWindowConfig,
) -> tuple[int, int] | None:
    """
    Compute [train_start, train_end) index slice for prediction at time t.

    Returns None if the window extends before the start of the series.
    """
    # Target is at t+h; buffer starts at t-tau_h; train ends at t-tau_h-h
    h = max(cfg.horizon_steps, 1)
    train_end = t - cfg.buffer_bars - h + 1
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

    for t in range(T):
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

            X_aug = np.column_stack([np.ones(len(X_fit)), X_fit])
            XtX = X_aug.T @ X_aug
            penalty = np.eye(K + 1) * lambda_
            penalty[0, 0] = 0.0
            try:
                last_beta = np.linalg.solve(XtX + penalty, X_aug.T @ y_fit)
            except np.linalg.LinAlgError:
                last_beta = np.linalg.lstsq(XtX + penalty, X_aug.T @ y_fit, rcond=None)[0]

        if last_beta is None:
            continue

        # Predict at time t for all stocks
        X_pred = np.column_stack([np.ones(N), states[t]])   # (N, K+1)
        preds = X_pred @ last_beta
        predictions[t] = preds
        pred_valid[t] = valid_mask[t]

    return predictions, pred_valid
