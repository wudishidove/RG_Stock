"""
Model Confidence Set (MCS) — Hansen, Lunde, Nason (2011).

Identifies the set of models not significantly dominated at confidence level alpha.
Uses the Range statistic (TR) and bootstrap p-values.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def mcs(
    losses: np.ndarray,
    alpha: float = 0.10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[int]:
    """
    Model Confidence Set via bootstrap.

    Parameters
    ----------
    losses      : (T, M) array of losses — rows = time, cols = models.
    alpha       : significance level (e.g., 0.10).
    n_bootstrap : number of bootstrap replications.
    seed        : random seed.

    Returns
    -------
    List of column indices (model indices) in the MCS.
    """
    rng = np.random.default_rng(seed)
    T, M = losses.shape

    surviving = list(range(M))

    while len(surviving) > 1:
        sub = losses[:, surviving]
        M_s = len(surviving)

        # Relative losses: d_{ij,t} = L_{i,t} - L_{j,t}
        # TR statistic = max_{i,j} |d_bar_{ij}| / se(d_bar_{ij})
        tr_stat, worst_idx = _tr_statistic(sub)

        # Bootstrap distribution of TR under H0
        boot_stats = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, T, size=T)
            boot_stat, _ = _tr_statistic(sub[idx])
            boot_stats[b] = boot_stat

        p_val = np.mean(boot_stats >= tr_stat)

        if p_val < alpha:
            # Eliminate worst model
            surviving.pop(worst_idx)
        else:
            break   # remaining models form the MCS

    return surviving


def _tr_statistic(losses: np.ndarray) -> tuple[float, int]:
    """Compute TR statistic and index of worst model."""
    T, M = losses.shape
    d_bar = np.zeros((M, M))
    d_var = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            if i != j:
                d = losses[:, i] - losses[:, j]
                d_bar[i, j] = d.mean()
                d_var[i, j] = d.var() / T

    # t-statistics for each pair
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(d_var > 0, d_bar / np.sqrt(d_var), 0.0)

    # TR = max over all pairs
    tr_flat = np.abs(t_stats).max()

    # Mean loss per model
    mean_losses = losses.mean(axis=0)
    worst_idx = int(np.argmax(mean_losses))

    return float(tr_flat), worst_idx
