"""Bootstrap confidence intervals for metrics.

Block bootstrap is appropriate for time series (preserves local dependence).
Naive i.i.d. bootstrap is appropriate for flattened-sample metrics like
marginal KS or Pearson correlation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def iid_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    synthetic: np.ndarray,
    real: np.ndarray,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a metric under i.i.d. resampling.

    This is valid for cross-sectional metrics where rows are treated as
    exchangeable samples (e.g., marginal KS, pairwise correlation error).

    Args:
        metric_fn: Callable taking (synthetic, real) arrays and returning
            a scalar metric value.
        synthetic: (n_samples, n_features) synthetic data.
        real: (n_samples, n_features) real data.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Desired confidence level, e.g. 0.95 for a 95% CI.
        seed: RNG seed for reproducibility.

    Returns:
        (ci_low, ci_high) for the metric.
    """
    rng = np.random.default_rng(seed)
    n_syn = len(synthetic)
    n_real = len(real)

    values: list[float] = []
    for _ in range(n_bootstrap):
        idx_syn = rng.integers(0, n_syn, size=n_syn)
        idx_real = rng.integers(0, n_real, size=n_real)
        try:
            v = metric_fn(synthetic[idx_syn], real[idx_real])
            if np.isfinite(v):
                values.append(float(v))
        except Exception:
            continue

    if not values:
        return (float("nan"), float("nan"))

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(values, alpha * 100))
    hi = float(np.percentile(values, (1 - alpha) * 100))
    return (lo, hi)


def block_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    synthetic: np.ndarray,
    real: np.ndarray,
    block_size: int = 20,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap CI using moving-block bootstrap for time series.

    Preserves local temporal dependence by resampling contiguous blocks.
    Appropriate for metrics that depend on the order of samples (ACF,
    volatility clustering, leverage effect).

    Args:
        metric_fn: Callable taking (synthetic, real) arrays and returning
            a scalar metric value.
        synthetic: (n_timesteps, n_features) synthetic time series.
        real: (n_timesteps, n_features) real time series.
        block_size: Length of each resampling block.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Desired confidence level.
        seed: RNG seed.

    Returns:
        (ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)

    def resample(x: np.ndarray) -> np.ndarray:
        n = len(x)
        n_blocks = (n + block_size - 1) // block_size
        starts = rng.integers(0, max(n - block_size + 1, 1), size=n_blocks)
        out = np.concatenate([x[s : s + block_size] for s in starts], axis=0)
        return out[:n]

    values: list[float] = []
    for _ in range(n_bootstrap):
        try:
            v = metric_fn(resample(synthetic), resample(real))
            if np.isfinite(v):
                values.append(float(v))
        except Exception:
            continue

    if not values:
        return (float("nan"), float("nan"))

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(values, alpha * 100))
    hi = float(np.percentile(values, (1 - alpha) * 100))
    return (lo, hi)
