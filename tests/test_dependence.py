"""Tests for dependence metrics."""

from __future__ import annotations

import numpy as np

from finval.metrics.dependence import (
    compute_copula_distance,
    compute_correlation_breakdown,
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_tail_dependence,
)


def test_pearson_matching_correlation_passes(real_returns_2d, matched_synthetic_2d):
    result = compute_pearson_correlation(matched_synthetic_2d, real_returns_2d)
    assert result.passed
    assert result.value < 0.20


def test_pearson_independent_vs_correlated_fails(rng):
    # Real data has strong correlation, synthetic is independent
    n = 2000
    real = rng.standard_normal((n, 3))
    real[:, 1] = 0.9 * real[:, 0] + 0.1 * real[:, 1]
    real[:, 2] = 0.9 * real[:, 0] + 0.1 * real[:, 2]
    syn = rng.standard_normal((n, 3))  # independent
    result = compute_pearson_correlation(syn, real)
    assert not result.passed


def test_spearman_matching_correlation_passes(real_returns_2d, matched_synthetic_2d):
    result = compute_spearman_correlation(matched_synthetic_2d, real_returns_2d)
    assert result.passed


def test_copula_distance_matching_passes(real_returns_2d, matched_synthetic_2d):
    result = compute_copula_distance(matched_synthetic_2d, real_returns_2d, grid_size=10)
    assert result.passed


def test_copula_distance_different_dependence_detects(rng):
    # Real: strong positive dependence. Synthetic: strong negative.
    n = 2000
    base = rng.standard_normal(n)
    real = np.column_stack([base, base + 0.3 * rng.standard_normal(n)])
    syn = np.column_stack([base, -base + 0.3 * rng.standard_normal(n)])
    result = compute_copula_distance(syn, real, grid_size=10)
    # Should detect the difference (not necessarily "poor" — copula shape
    # is mirrored but magnitude is similar), just not excellent
    assert result.quality != "excellent"


def test_tail_dependence_matching_passes(real_returns_2d, matched_synthetic_2d):
    upper, lower = compute_tail_dependence(matched_synthetic_2d, real_returns_2d)
    assert upper.passed
    assert lower.passed


def test_tail_dependence_independent_vs_crashes_together(rng):
    # Real data: assets crash together (lower tail dependence)
    n = 5000
    shock = rng.standard_t(df=3, size=n)  # common crash factor
    idio = rng.standard_normal((n, 3)) * 0.5
    real = np.column_stack([shock + idio[:, i] for i in range(3)])

    # Synthetic: independent — no tail dependence
    syn = rng.standard_t(df=3, size=(n, 3))

    _, lower = compute_tail_dependence(syn, real)
    # Real has strong lower tail dependence, synthetic has none
    assert lower.value > 0.05  # should detect the difference


def test_tail_dependence_values_in_valid_range(real_returns_2d, matched_synthetic_2d):
    upper, lower = compute_tail_dependence(matched_synthetic_2d, real_returns_2d)
    # Errors are absolute differences of probabilities, so in [0, 1]
    assert 0 <= upper.value <= 1
    assert 0 <= lower.value <= 1


def test_correlation_breakdown_runs(real_returns_2d, matched_synthetic_2d):
    result = compute_correlation_breakdown(matched_synthetic_2d, real_returns_2d)
    assert np.isfinite(result.value)
    assert result.passed


def test_single_feature_returns_error():
    real = np.random.randn(500, 1)
    syn = np.random.randn(500, 1)
    result = compute_pearson_correlation(syn, real)
    assert result.quality == "poor"
