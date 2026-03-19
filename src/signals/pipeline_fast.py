"""
Parallelised causal signal construction pipeline.

Same computation as pipeline.py but distributes the independent
time-step loop across multiple processes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp

import numpy as np

from .pca_factors import extract_pca_factors
from .residuals import ols_residuals
from .ou_estimation import ou_parameters, aggregate_residuals, _P_LEVELS
from .zscore import build_signal_vector

logger = logging.getLogger(__name__)

# ── worker shared state (set via initializer) ──────────────────────
_W: dict = {}


def _init_worker(
    returns_panel: np.ndarray,
    pca_lookback: int,
    n_pca_components: int,
    p_levels: list[int],
    max_missing_fraction: float,
    eps: float,
) -> None:
    _W["rp"] = returns_panel
    _W["pl"] = pca_lookback
    _W["npc"] = n_pca_components
    _W["plevs"] = p_levels
    _W["mmf"] = max_missing_fraction
    _W["eps"] = eps
    _W["min_bars"] = max(p_levels) + n_pca_components + 2


def _process_t(t: int):
    """Compute signals for a single time step (identical logic to pipeline.py)."""
    rp = _W["rp"]
    T, N = rp.shape
    if t < _W["min_bars"]:
        return None

    pca_lookback = _W["pl"]
    n_pca = _W["npc"]
    p_levels = _W["plevs"]
    mmf = _W["mmf"]
    eps = _W["eps"]

    start = max(0, t - pca_lookback)
    window = rp[start:t]  # strictly before t

    # Exclude stocks with too many missing values in window
    miss_frac = np.mean(np.isnan(window), axis=0)
    valid_stocks = miss_frac <= mmf

    if valid_stocks.sum() < n_pca + 1:
        return None

    # PCA
    try:
        factors, _ = extract_pca_factors(
            window[:, valid_stocks], n_components=n_pca
        )
    except Exception:
        return None

    # OLS residuals
    try:
        drift, loadings, residuals = ols_residuals(window, factors)
    except Exception:
        return None

    # OU parameter estimation
    try:
        ou_params_all = ou_parameters(residuals, p_levels=p_levels, eps=eps)
    except Exception:
        return None

    # Aggregated residuals at current t (last row of window)
    U_hats: dict[int, np.ndarray] = {}
    for P in p_levels:
        U_all = aggregate_residuals(residuals, P)
        U_hats[P] = U_all[-1]

    # Modified z-score
    Z_t = build_signal_vector(
        U_hats, drift, ou_params_all, p_levels=p_levels, eps=eps
    )
    validity_t = np.all(~np.isnan(Z_t), axis=1) & valid_stocks

    return (t, Z_t, validity_t)


# ── public API ─────────────────────────────────────────────────────

def build_signals_causal(
    returns_panel: np.ndarray,
    session_boundary: np.ndarray,
    pca_lookback: int,
    n_pca_components: int = 15,
    p_levels: list[int] = _P_LEVELS,
    max_missing_fraction: float = 0.20,
    eps: float = 1e-6,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for pipeline.build_signals_causal with multiprocessing.

    Each time step t is independent, so we distribute them across workers.
    Inner computation is identical → numerically identical output.
    """
    T, N = returns_panel.shape
    D = len(p_levels)
    signals = np.full((T, N, D), np.nan)
    validity = np.zeros((T, N), dtype=bool)

    min_bars_needed = max(p_levels) + n_pca_components + 2
    t_range = list(range(min_bars_needed, T))
    total = len(t_range)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    logger.info(
        "Parallel build_signals: %d workers, %d time steps to process", n_workers, total
    )

    done = 0
    with mp.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(
            returns_panel,
            pca_lookback,
            n_pca_components,
            p_levels,
            max_missing_fraction,
            eps,
        ),
    ) as pool:
        for result in pool.imap_unordered(_process_t, t_range, chunksize=32):
            done += 1
            if done % 2000 == 0:
                logger.info("Progress: %d / %d (%.0f%%)", done, total, 100 * done / total)
            if result is not None:
                t, Z_t, val_t = result
                signals[t] = Z_t
                validity[t] = val_t

    return signals, validity
