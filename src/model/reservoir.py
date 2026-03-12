"""
ESN reservoir: random matrix construction, state evolution, state decay.

State update: X_t = alpha * X_{t-1} + (1-alpha) * tanh(rho * A_bar @ X_{t-1} + gamma * C_bar @ Z_t)

Key property: reservoir weights are fixed after construction.
All reservoir states can be computed in a single forward pass.
"""

from __future__ import annotations

import numpy as np

from ..utils.linalg import (
    sparse_gaussian_matrix,
    sparse_uniform_matrix,
    normalize_spectral_radius,
    normalize_spectral_norm,
)


class ESNReservoir:
    """Fixed random reservoir for Echo State Network."""

    def __init__(
        self,
        K: int,
        D: int,
        alpha: float,
        rho: float,
        gamma: float,
        A_sparsity: float,
        C_sparsity: float,
        seed: int = 42,
        activation: str = "tanh",
    ) -> None:
        """
        Parameters
        ----------
        K          : reservoir size.
        D          : input dimension (signal dimension = 6).
        alpha      : leak rate ∈ [0, 1].
        rho        : target spectral radius of A.
        gamma      : input scaling.
        A_sparsity : density of recurrent matrix.
        C_sparsity : density of input matrix.
        seed       : random seed for reproducibility.
        activation : nonlinear activation function name.
        """
        self.K = K
        self.D = D
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.A_sparsity = A_sparsity
        self.C_sparsity = C_sparsity
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Build and normalize recurrent matrix A
        A_star = sparse_gaussian_matrix(K, K, A_sparsity, rng)
        if rho > 1e-10:
            self.A_bar = normalize_spectral_radius(A_star, target_rho=rho)
        else:
            self.A_bar = np.zeros((K, K))  # EOD: rho=0, no recurrent connection

        # Build and normalize input matrix C
        C_star = sparse_uniform_matrix(K, D, C_sparsity, rng)
        self.C_bar = normalize_spectral_norm(C_star, target_gamma=gamma)

        self._activation = np.tanh if activation == "tanh" else np.tanh

    def forward(
        self,
        signals: np.ndarray,     # (T, D) signal sequence for one stock
        valid_mask: np.ndarray,  # (T,) bool: True = valid signal
        x0: np.ndarray | None = None,  # (K,) initial state
    ) -> np.ndarray:
        """
        Compute reservoir states for a single stock's signal sequence.

        Missing signals (valid_mask=False) are replaced with zero vector.
        The state continues to evolve (state decay through zero input).

        Parameters
        ----------
        signals    : (T, D) signal array (may contain NaN for missing).
        valid_mask : (T,) True where signals are valid.
        x0         : initial reservoir state (zeros if None).

        Returns
        -------
        states : (T, K) reservoir state array.
        """
        T = signals.shape[0]
        states = np.zeros((T, self.K))
        x = np.zeros(self.K) if x0 is None else x0.copy()

        for t in range(T):
            z = signals[t] if valid_mask[t] else np.zeros(self.D)
            # Replace NaN within valid signals with 0 (extra safety)
            z = np.where(np.isnan(z), 0.0, z)

            pre = self.A_bar @ x + self.C_bar @ z
            x = self.alpha * x + (1.0 - self.alpha) * self._activation(pre)
            states[t] = x

        return states

    def forward_all_stocks(
        self,
        signals: np.ndarray,      # (T, N, D)
        validity: np.ndarray,     # (T, N) bool
    ) -> np.ndarray:
        """
        Compute reservoir states for all N stocks independently.

        Returns (T, N, K) state array.
        """
        T, N, D = signals.shape
        all_states = np.zeros((T, N, self.K))
        for i in range(N):
            all_states[:, i, :] = self.forward(signals[:, i, :], validity[:, i])
        return all_states
