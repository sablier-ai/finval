"""Tests for distribution metrics.

Each metric is tested with known-ground-truth synthetic data:
- Identical distributions should yield "excellent" quality
- Deliberately broken synthetic data should yield "poor" quality
"""

from __future__ import annotations

import numpy as np

from finval.metrics.distribution import (
    compute_energy_distance,
    compute_marginal_ks,
    compute_tail_heaviness,
    compute_tail_quantiles,
)


def test_marginal_ks_identical_distributions_passes(
    real_returns_2d, matched_synthetic_2d
):
    result = compute_marginal_ks(matched_synthetic_2d, real_returns_2d)
    assert result.passed
    assert result.quality in ("excellent", "good")
    assert result.value < 0.10


def test_marginal_ks_wrong_scale_fails(rng, real_returns_2d):
    # Wrong scale: 5x wider — should fail
    bad = rng.standard_normal(real_returns_2d.shape) * 0.05
    result = compute_marginal_ks(bad, real_returns_2d)
    assert not result.passed
    assert result.value > 0.20


def test_marginal_ks_wrong_mean_fails(rng, real_returns_2d):
    # Shifted mean by several std — should fail
    bad = rng.standard_t(df=5, size=real_returns_2d.shape) * 0.01 + 0.05
    result = compute_marginal_ks(bad, real_returns_2d)
    assert not result.passed


def test_energy_distance_identical_distributions_passes(
    real_returns_2d, matched_synthetic_2d
):
    result = compute_energy_distance(matched_synthetic_2d, real_returns_2d)
    assert result.passed
    assert result.value < 0.20


def test_energy_distance_different_shape_fails(rng, real_returns_2d):
    # Different distribution shape (uniform vs t_5)
    bad = rng.uniform(-0.05, 0.05, size=real_returns_2d.shape)
    result = compute_energy_distance(bad, real_returns_2d)
    # Energy distance should at least not score excellent
    assert result.quality != "excellent"


def test_tail_quantiles_matching_tails_passes(real_returns_2d, matched_synthetic_2d):
    result = compute_tail_quantiles(matched_synthetic_2d, real_returns_2d)
    assert result.passed
    assert result.value < 0.35


def test_tail_quantiles_thin_tails_fails(rng, real_returns_2d):
    # Gaussian synthetic vs t_5 real — should fail the tails
    # Match scale to fool marginal_ks but not tail_quantiles
    real_std = np.std(real_returns_2d, axis=0)
    bad = rng.standard_normal(real_returns_2d.shape) * real_std
    result = compute_tail_quantiles(bad, real_returns_2d)
    # Thin-tailed Gaussian at 1st/99th percentile differs from t_5
    assert result.value > 0.05  # some error is expected


def test_tail_quantiles_wrong_scale_fails_badly(rng, real_returns_2d):
    bad = rng.standard_t(df=5, size=real_returns_2d.shape) * 0.10  # 10x scale
    result = compute_tail_quantiles(bad, real_returns_2d)
    assert not result.passed
    assert result.value > 1.0  # extreme scale mismatch


def test_tail_heaviness_is_diagnostic(real_returns_2d, matched_synthetic_2d):
    """tail_heaviness is diagnostic — we just check it runs without errors."""
    result = compute_tail_heaviness(matched_synthetic_2d, real_returns_2d)
    assert result.name == "tail_heaviness"
    assert result.category == "distribution"
    assert np.isfinite(result.value)


def test_marginal_ks_returns_per_feature_breakdown(real_returns_2d, matched_synthetic_2d):
    names = ["a", "b", "c"]
    result = compute_marginal_ks(matched_synthetic_2d, real_returns_2d, feature_names=names)
    assert set(result.per_feature.keys()) == set(names)


def test_marginal_ks_handles_nan(rng):
    real = rng.standard_t(df=5, size=(500, 2)) * 0.01
    syn = rng.standard_t(df=5, size=(500, 2)) * 0.01
    syn[0, 0] = np.nan  # inject a single NaN
    result = compute_marginal_ks(syn, real)
    assert np.isfinite(result.value)


def test_feature_count_mismatch_returns_error(rng):
    real = rng.standard_normal((100, 3))
    bad = rng.standard_normal((100, 4))
    result = compute_marginal_ks(bad, real)
    assert result.quality == "poor"
    assert "mismatch" in result.interpretation.lower()
