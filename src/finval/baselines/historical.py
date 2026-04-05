"""Historical resampling baselines.

- `historical_bootstrap` samples rows i.i.d. from real returns. Preserves
  the empirical marginal and joint distribution perfectly but destroys
  all temporal structure.

- `block_bootstrap` samples contiguous blocks of rows. Preserves short-
  range temporal dependence (within a block) at the cost of some joint
  distribution fidelity at block boundaries. This is the strongest simple
  baseline for temporal stylized facts.
"""

from __future__ import annotations

import numpy as np


def historical_bootstrap(
    real: np.ndarray,
    n_samples: int | None = None,
    n_paths: int | None = None,
    path_length: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """I.i.d. bootstrap: sample rows of real with replacement.

    Perfect marginal and joint distribution match in expectation.
    Zero temporal structure (independent rows).
    """
    real = np.asarray(real)
    if real.ndim != 2:
        raise ValueError(f"real must be 2D, got shape {real.shape}")

    rng = np.random.default_rng(seed)
    n = len(real)

    if n_samples is not None:
        idx = rng.integers(0, n, size=n_samples)
        return real[idx]

    if n_paths is not None and path_length is not None:
        idx = rng.integers(0, n, size=(n_paths, path_length))
        return real[idx]

    raise ValueError("must specify either n_samples or (n_paths and path_length)")


def block_bootstrap(
    real: np.ndarray,
    n_paths: int,
    path_length: int,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Moving-block bootstrap: sample contiguous blocks of rows.

    Preserves temporal dependence up to block_size. A block size of
    ~20 trading days is typical for financial data. Only supports path
    output (the concept doesn't make sense for flat samples).

    Returns shape (n_paths, path_length, n_features).
    """
    real = np.asarray(real)
    if real.ndim != 2:
        raise ValueError(f"real must be 2D, got shape {real.shape}")
    if len(real) < block_size:
        raise ValueError(f"real has {len(real)} rows, need >= block_size={block_size}")

    rng = np.random.default_rng(seed)
    n = len(real)
    n_features = real.shape[1]
    n_blocks = (path_length + block_size - 1) // block_size

    paths = np.empty((n_paths, path_length, n_features), dtype=real.dtype)
    for p in range(n_paths):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        blocks = [real[s : s + block_size] for s in starts]
        full = np.concatenate(blocks, axis=0)
        paths[p] = full[:path_length]
    return paths
