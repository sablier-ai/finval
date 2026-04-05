"""Tests for path-level metrics."""

from __future__ import annotations

import numpy as np

from finval.metrics.paths import compute_drawdown_distribution


def _returns_to_levels(returns: np.ndarray) -> np.ndarray:
    """Cumulative product for log-return price levels."""
    return np.exp(np.cumsum(returns, axis=1))


def test_drawdown_matching_distributions_passes(rng):
    """Two identical-distribution return processes should have matching drawdowns."""
    syn_ret = rng.standard_t(df=5, size=(100, 100, 3)) * 0.01
    real_ret = rng.standard_t(df=5, size=(100, 100, 3)) * 0.01
    result = compute_drawdown_distribution(
        _returns_to_levels(syn_ret), _returns_to_levels(real_ret)
    )
    assert result.passed


def test_drawdown_detects_scale_mismatch(rng):
    """Synthetic with 3x scale has much larger drawdowns."""
    real_ret = rng.standard_t(df=5, size=(100, 100, 3)) * 0.01
    syn_ret = rng.standard_t(df=5, size=(100, 100, 3)) * 0.03
    result = compute_drawdown_distribution(
        _returns_to_levels(syn_ret), _returns_to_levels(real_ret)
    )
    assert not result.passed
