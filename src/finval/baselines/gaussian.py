"""Multivariate Gaussian baseline.

Fits a multivariate normal to the real returns and samples from it.
This is the naive "Gaussian i.i.d." benchmark. A model that can't beat
this on temporal metrics is not capturing any time structure.
"""

from __future__ import annotations

import numpy as np


def gaussian_baseline(
    real: np.ndarray,
    n_samples: int | None = None,
    n_paths: int | None = None,
    path_length: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate samples from a multivariate Gaussian fit to real returns.

    Two output modes:

    1. Flat mode (n_samples given): returns shape (n_samples, n_features),
       matching the format expected by distribution / dependence metrics.

    2. Path mode (n_paths and path_length given): returns shape
       (n_paths, path_length, n_features), matching the format expected
       by temporal and path metrics. Each row within a path is an
       independent draw — there is no temporal structure.

    Args:
        real: (n_obs, n_features) real return series used to fit mean+cov.
        n_samples: Number of flat samples to return.
        n_paths: Number of paths to return (requires path_length).
        path_length: Length of each path.
        seed: RNG seed.

    Returns:
        numpy array of shape (n_samples, n_features) or (n_paths, path_length, n_features).
    """
    real = np.asarray(real)
    if real.ndim != 2:
        raise ValueError(f"real must be 2D, got shape {real.shape}")

    mean = np.nanmean(real, axis=0)
    # Handle NaN by dropping rows before covariance
    clean = real[~np.any(np.isnan(real), axis=1)]
    if len(clean) < 2:
        raise ValueError("need at least 2 clean rows to fit covariance")
    cov = np.cov(clean, rowvar=False)
    # Regularize to ensure positive-definite
    d = cov.shape[0]
    cov = cov + 1e-10 * np.eye(d)

    rng = np.random.default_rng(seed)

    if n_samples is not None:
        return rng.multivariate_normal(mean, cov, size=n_samples)

    if n_paths is not None and path_length is not None:
        return rng.multivariate_normal(mean, cov, size=(n_paths, path_length))

    raise ValueError("must specify either n_samples or (n_paths and path_length)")
