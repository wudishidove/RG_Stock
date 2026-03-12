"""Linear algebra utilities for ESN reservoir construction."""

import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs


def spectral_radius(A: np.ndarray) -> float:
    """Compute the spectral radius (largest absolute eigenvalue) of matrix A."""
    if A.shape[0] <= 1:
        return float(np.abs(A).max())
    try:
        # Use sparse eigensolver for efficiency; k=1 largest magnitude
        vals = eigs(A, k=1, which="LM", return_eigenvectors=False)
        return float(np.abs(vals).max())
    except Exception:
        vals = np.linalg.eigvals(A)
        return float(np.abs(vals).max())


def sparse_gaussian_matrix(n_rows: int, n_cols: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a sparse matrix with Gaussian nonzero entries.

    Nonzero positions drawn uniformly; values ~ N(0,1).
    Returns a dense numpy array (K is small, typically 100).
    """
    A = np.zeros((n_rows, n_cols))
    n_nonzero = max(1, int(density * n_rows * n_cols))
    idx = rng.choice(n_rows * n_cols, size=n_nonzero, replace=False)
    rows, cols = np.unravel_index(idx, (n_rows, n_cols))
    A[rows, cols] = rng.standard_normal(n_nonzero)
    return A


def sparse_uniform_matrix(n_rows: int, n_cols: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a sparse matrix with Uniform(-1,1) nonzero entries."""
    C = np.zeros((n_rows, n_cols))
    n_nonzero = max(1, int(density * n_rows * n_cols))
    idx = rng.choice(n_rows * n_cols, size=n_nonzero, replace=False)
    rows, cols = np.unravel_index(idx, (n_rows, n_cols))
    C[rows, cols] = rng.uniform(-1.0, 1.0, n_nonzero)
    return C


def normalize_spectral_radius(A: np.ndarray, target_rho: float, eps: float = 1e-10) -> np.ndarray:
    """Normalize A so its spectral radius equals target_rho."""
    sr = spectral_radius(A)
    if sr < eps:
        return A
    return A * (target_rho / sr)


def normalize_spectral_norm(C: np.ndarray, target_gamma: float, eps: float = 1e-10) -> np.ndarray:
    """Normalize C by its spectral norm (largest singular value), then scale by gamma."""
    sv = np.linalg.norm(C, ord=2)
    if sv < eps:
        return C
    return C * (target_gamma / sv)
