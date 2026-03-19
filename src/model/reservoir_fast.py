"""
Vectorized ESN reservoir: processes all N stocks simultaneously per time step.

Instead of N individual matrix-vector multiplies (GEMV),
uses a single matrix-matrix multiply (GEMM) per time step.
"""

from __future__ import annotations

import numpy as np

from .reservoir import ESNReservoir


class ESNReservoirFast(ESNReservoir):
    """ESNReservoir with vectorized forward pass across all stocks."""

    def forward_all_stocks_vectorized(
        self,
        signals: np.ndarray,      # (T, N, D)
        validity: np.ndarray,     # (T, N) bool
    ) -> np.ndarray:
        """
        Compute reservoir states for all N stocks simultaneously.

        Equivalent to forward_all_stocks() but uses GEMM instead of
        N individual GEMV calls, yielding significant speedup.

        Returns (T, N, K) state array.
        """
        T, N, D = signals.shape
        all_states = np.zeros((T, N, self.K))
        X = np.zeros((N, self.K))  # (N, K) current state for all stocks

        A_T = self.A_bar.T  # (K, K) — pre-transpose for X @ A_T
        C_T = self.C_bar.T  # (D, K) — pre-transpose for Z @ C_T

        for t in range(T):
            Z = signals[t].copy()         # (N, D)
            Z[~validity[t]] = 0.0         # zero out invalid stocks
            np.nan_to_num(Z, copy=False)  # zero out NaN in-place

            pre = X @ A_T + Z @ C_T      # (N, K) — GEMM
            X = self.alpha * X + (1.0 - self.alpha) * self._activation(pre)
            all_states[t] = X

        return all_states
