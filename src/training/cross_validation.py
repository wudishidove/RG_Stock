"""
Daily cross-validation: 1-week lookback, 0.7/0.3 temporal split.

Run once per trading day to select readout regularization parameter.
"""

from __future__ import annotations

import numpy as np
import logging

logger = logging.getLogger(__name__)


def daily_cv_lambda(
    states: np.ndarray,        # (T_cv, N, K) reservoir states in CV window
    targets: np.ndarray,       # (T_cv, N)
    valid_mask: np.ndarray,    # (T_cv, N)
    lambda_candidates: list[float],
    cv_split: float = 0.7,
) -> float:
    """
    Select regularization lambda via temporal 70/30 split.

    Parameters
    ----------
    states           : reservoir states over CV lookback window.
    targets          : target returns.
    valid_mask       : valid observation mask.
    lambda_candidates: list of lambda values to search.
    cv_split         : fraction of window used for training (temporal order preserved).

    Returns
    -------
    Best lambda (minimizes validation MSFE).
    """
    T, N, K = states.shape
    split_idx = int(T * cv_split)

    if split_idx < K + 2 or T - split_idx < 2:
        return lambda_candidates[len(lambda_candidates) // 2]

    # Training portion
    train_states = states[:split_idx].reshape(split_idx * N, K)
    train_targets = targets[:split_idx].reshape(split_idx * N)
    train_valid = valid_mask[:split_idx].reshape(split_idx * N)

    # Validation portion
    val_states = states[split_idx:].reshape((T - split_idx) * N, K)
    val_targets = targets[split_idx:].reshape((T - split_idx) * N)
    val_valid = valid_mask[split_idx:].reshape((T - split_idx) * N)

    X_train = np.column_stack([np.ones(train_valid.sum()), train_states[train_valid]])
    y_train = train_targets[train_valid]
    X_val = np.column_stack([np.ones(val_valid.sum()), val_states[val_valid]])
    y_val = val_targets[val_valid]

    if len(y_train) < K + 2 or len(y_val) < 2:
        return lambda_candidates[len(lambda_candidates) // 2]

    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train

    best_lambda = lambda_candidates[0]
    best_msfe = np.inf

    for lam in lambda_candidates:
        penalty = np.eye(K + 1) * lam
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(XtX + penalty, Xty)
        except np.linalg.LinAlgError:
            continue
        preds = X_val @ beta
        msfe = np.mean((y_val - preds) ** 2)
        if msfe < best_msfe:
            best_msfe = msfe
            best_lambda = lam

    return best_lambda


_DEFAULT_LAMBDA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
