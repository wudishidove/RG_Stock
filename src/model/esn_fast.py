"""
Fast ESN using vectorized reservoir forward pass.

Drop-in replacement for ESN — same interface, same random matrices,
but compute_states uses GEMM instead of per-stock GEMV loops.
"""

from __future__ import annotations

import numpy as np

from .reservoir_fast import ESNReservoirFast
from .readout import RidgeReadout


class ESNFast:
    """ESN with vectorized reservoir. Same interface as ESN."""

    def __init__(
        self,
        K: int = 100,
        D: int = 6,
        alpha: float = 0.9,
        rho: float = 0.4,
        gamma: float = 0.005,
        A_sparsity: float = 0.15,
        C_sparsity: float = 0.95,
        lambda_ridge: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.reservoir = ESNReservoirFast(
            K=K, D=D, alpha=alpha, rho=rho, gamma=gamma,
            A_sparsity=A_sparsity, C_sparsity=C_sparsity, seed=seed,
        )
        self.readout = RidgeReadout(lambda_=lambda_ridge)

    def compute_states(
        self,
        signals: np.ndarray,   # (T, N, D)
        validity: np.ndarray,  # (T, N)
    ) -> np.ndarray:
        """Compute (T, N, K) reservoir states using vectorized forward pass."""
        return self.reservoir.forward_all_stocks_vectorized(signals, validity)

    def fit_readout(
        self,
        states: np.ndarray,        # (T_train, N, K)
        targets: np.ndarray,       # (T_train, N)
        valid_mask: np.ndarray,    # (T_train, N) bool
    ) -> "ESNFast":
        T, N, K = states.shape
        flat_states = states.reshape(T * N, K)
        flat_targets = targets.reshape(T * N)
        flat_valid = valid_mask.reshape(T * N)

        X = flat_states[flat_valid]
        y = flat_targets[flat_valid]

        if len(y) < K + 2:
            return self

        self.readout.fit(X, y)
        return self

    def predict(
        self,
        states: np.ndarray,   # (T, N, K) or (N, K)
    ) -> np.ndarray:
        if states.ndim == 3:
            T, N, K = states.shape
            flat = states.reshape(T * N, K)
            preds = self.readout.predict(flat)
            return preds.reshape(T, N)
        elif states.ndim == 2:
            return self.readout.predict(states)
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")
