"""Tests for temporal metrics."""

from __future__ import annotations

import numpy as np

from finval.metrics.temporal import (
    compute_acf_returns,
    compute_cross_correlation,
    compute_leverage_effect,
    compute_volatility_clustering,
)


def test_acf_returns_iid_passes(rng):
    """Two i.i.d. Gaussian path sets should have near-zero ACF."""
    syn = rng.standard_normal((50, 100, 3)) * 0.01
    real = rng.standard_normal((50, 100, 3)) * 0.01
    result = compute_acf_returns(syn, real)
    assert result.passed


def test_acf_returns_detects_autocorrelation_mismatch(rng):
    """AR(1) synthetic vs i.i.d. real — should fail ACF."""
    n_paths, horizon, d = 50, 100, 2
    real = rng.standard_normal((n_paths, horizon, d)) * 0.01

    # Synthetic with strong positive autocorrelation
    syn = np.zeros_like(real)
    for p in range(n_paths):
        syn[p, 0] = rng.standard_normal(d) * 0.01
        for t in range(1, horizon):
            syn[p, t] = 0.8 * syn[p, t - 1] + rng.standard_normal(d) * 0.01

    result = compute_acf_returns(syn, real)
    assert not result.passed


def test_volatility_clustering_matches_gaussian(rng):
    """Two i.i.d. samples have no vol clustering — should pass trivially."""
    syn = rng.standard_normal((50, 100, 2)) * 0.01
    real = rng.standard_normal((50, 100, 2)) * 0.01
    result = compute_volatility_clustering(syn, real)
    assert result.passed


def test_leverage_effect_runs(rng):
    """Leverage effect on i.i.d. data should be near zero for both."""
    syn = rng.standard_normal((50, 100, 2)) * 0.01
    real = rng.standard_normal((50, 100, 2)) * 0.01
    result = compute_leverage_effect(syn, real)
    assert result.passed


def test_cross_correlation_matching_passes(rng):
    syn = rng.standard_normal((50, 100, 3)) * 0.01
    real = rng.standard_normal((50, 100, 3)) * 0.01
    result = compute_cross_correlation(syn, real)
    assert result.passed


def test_cross_correlation_detects_mismatch(rng):
    """Real has strong cross-feature correlation; synthetic is independent."""
    real = rng.standard_normal((50, 100, 3)) * 0.01
    real[:, :, 1] = 0.9 * real[:, :, 0] + 0.1 * real[:, :, 1]
    real[:, :, 2] = 0.9 * real[:, :, 0] + 0.1 * real[:, :, 2]
    syn = rng.standard_normal((50, 100, 3)) * 0.01
    result = compute_cross_correlation(syn, real)
    assert not result.passed


def test_short_horizon_returns_error(rng):
    """Temporal metrics require sufficient horizon."""
    syn = rng.standard_normal((20, 10, 2)) * 0.01
    real = rng.standard_normal((20, 10, 2)) * 0.01
    result = compute_acf_returns(syn, real)
    assert result.quality == "poor"
    assert "too short" in result.interpretation.lower()
