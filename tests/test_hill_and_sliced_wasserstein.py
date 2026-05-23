"""Tests for Hill tail-index and sliced multivariate Wasserstein.

Reference checks:

  - Hill estimator: a sample from Pareto(alpha) has true xi = 1/alpha.
    Matched (alpha, alpha) samples should produce small error;
    Pareto vs Gaussian should produce large error.
  - Sliced Wasserstein: identical empirical distributions give SW -> 0;
    a shifted distribution gives SW proportional to the shift; scale
    differences are picked up too.
"""

from __future__ import annotations

import numpy as np

from finval.metrics.distribution import (
    _hill_estimator,
    compute_hill_tail_index,
    compute_sliced_wasserstein,
)


# -------------------------------------------------------------------------
# Hill estimator
# -------------------------------------------------------------------------


def test_hill_pareto_recovers_inverse_alpha():
    # Pareto(alpha) survival function is x^{-alpha} so xi = 1/alpha.
    # alpha=2 -> xi = 0.5; with n=10_000, k=500 (5% tail), the
    # estimator's standard error is roughly xi / sqrt(k) ~ 0.022.
    rng = np.random.default_rng(0)
    alpha = 2.0
    n = 10_000
    # scipy-free Pareto sampling: 1 / U^{1/alpha} where U ~ Uniform(0,1)
    x = 1.0 / rng.uniform(size=n) ** (1.0 / alpha)
    k = 500
    xi_hat = _hill_estimator(x, k)
    assert abs(xi_hat - 0.5) < 0.05, f"Hill on Pareto(2): xi_hat={xi_hat}, expected ~0.5"


def test_hill_matched_distributions_score_excellent():
    rng = np.random.default_rng(1)
    n, d = 5000, 3
    # All features fat-tailed but identical
    syn = rng.standard_t(df=5, size=(n, d))
    real = rng.standard_t(df=5, size=(n, d))
    res = compute_hill_tail_index(syn, real)
    assert res.passed
    # Same family, big n -> small error
    assert res.value < 0.10, f"matched Hill error = {res.value}"


def test_hill_pareto_vs_gaussian_detected():
    rng = np.random.default_rng(2)
    n, d = 5000, 2
    # Synthetic = Gaussian, real = Pareto-like fat tails.
    syn = rng.standard_normal((n, d))
    # Build symmetric Pareto: sign * Pareto(2) magnitude
    mag = 1.0 / rng.uniform(size=(n, d)) ** 0.5
    sign = rng.choice([-1.0, 1.0], size=(n, d))
    real = sign * mag
    res = compute_hill_tail_index(syn, real)
    assert not res.passed
    # Pareto xi_real ~ 0.5; Gaussian xi_syn is much smaller -> large gap
    assert res.value > 0.10, f"Pareto vs Gaussian Hill error = {res.value}"


def test_hill_handles_too_few_obs():
    rng = np.random.default_rng(3)
    # 30 observations is below min_k * 2 = 50 default
    syn = rng.standard_normal((30, 2))
    real = rng.standard_normal((30, 2))
    res = compute_hill_tail_index(syn, real)
    assert res.quality == "poor"
    assert not res.passed


def test_hill_upper_only_versus_both_sides():
    rng = np.random.default_rng(4)
    n = 5000
    syn = rng.standard_t(df=5, size=(n, 2))
    real = rng.standard_t(df=5, size=(n, 2))
    res_both = compute_hill_tail_index(syn, real, sides=("upper", "lower"))
    res_upper = compute_hill_tail_index(syn, real, sides=("upper",))
    # Both should pass for matched t_5; the upper-only run scores fewer
    # pairs but should still have a reasonable error magnitude.
    assert res_both.metadata["n_pairs_scored"] == 2 * res_upper.metadata["n_pairs_scored"]


def test_hill_per_feature_breakdown():
    rng = np.random.default_rng(5)
    n = 5000
    syn = rng.standard_t(df=5, size=(n, 3))
    real = rng.standard_t(df=5, size=(n, 3))
    feature_names = ["alpha", "beta", "gamma"]
    res = compute_hill_tail_index(syn, real, feature_names=feature_names)
    assert set(res.per_feature.keys()) <= set(feature_names)
    # Each per-feature has detail dict in metadata
    for name, detail in res.metadata["per_feature_detail"].items():
        assert "xi_upper_synthetic" in detail
        assert "xi_lower_synthetic" in detail


# -------------------------------------------------------------------------
# Sliced Wasserstein
# -------------------------------------------------------------------------


def test_sliced_wasserstein_identical_distributions_near_zero():
    rng = np.random.default_rng(10)
    n, d = 2000, 4
    syn = rng.standard_normal((n, d))
    real = rng.standard_normal((n, d))
    res = compute_sliced_wasserstein(syn, real, random_state=0)
    # Distinct draws from the same distribution: small but non-zero
    assert res.value < 0.10
    assert res.quality == "excellent"


def test_sliced_wasserstein_shifted_detected():
    rng = np.random.default_rng(11)
    n, d = 2000, 4
    syn = rng.standard_normal((n, d)) + 1.0  # +1 sigma shift
    real = rng.standard_normal((n, d))
    res = compute_sliced_wasserstein(syn, real, random_state=0)
    # Shift dominates scaling: should fail
    assert not res.passed
    assert res.value > 0.40


def test_sliced_wasserstein_scale_mismatch_detected():
    rng = np.random.default_rng(12)
    n, d = 2000, 4
    syn = rng.standard_normal((n, d)) * 0.3  # 3x narrower
    real = rng.standard_normal((n, d))
    res = compute_sliced_wasserstein(syn, real, random_state=0)
    assert res.value > 0.10


def test_sliced_wasserstein_reproducible():
    # Same seed -> same value
    rng = np.random.default_rng(13)
    n, d = 1000, 3
    syn = rng.standard_normal((n, d))
    real = rng.standard_normal((n, d))
    res1 = compute_sliced_wasserstein(syn, real, random_state=0, n_projections=64)
    res2 = compute_sliced_wasserstein(syn, real, random_state=0, n_projections=64)
    assert res1.value == res2.value


def test_sliced_wasserstein_different_sample_sizes():
    # n_syn != n_real should still work via quantile-resampling
    rng = np.random.default_rng(14)
    syn = rng.standard_normal((1000, 3))
    real = rng.standard_normal((2000, 3))
    res = compute_sliced_wasserstein(syn, real, random_state=0)
    # Same distribution, different sample sizes: small error
    assert res.value < 0.20


def test_sliced_wasserstein_p1_and_p2_both_work():
    rng = np.random.default_rng(15)
    syn = rng.standard_normal((500, 2))
    real = rng.standard_normal((500, 2))
    res1 = compute_sliced_wasserstein(syn, real, p=1, random_state=0)
    res2 = compute_sliced_wasserstein(syn, real, p=2, random_state=0)
    assert res1.passed and res2.passed
    # Both should be small; n=500 has non-negligible sampling variance,
    # so loosen the threshold above the "excellent" cutoff.
    assert res1.value < 0.15
    assert res2.value < 0.15


def test_sliced_wasserstein_too_few_samples_errors():
    res = compute_sliced_wasserstein(
        np.zeros((10, 3)), np.zeros((10, 3))
    )
    assert res.quality == "poor"
    assert not res.passed


def test_sliced_wasserstein_dimension_mismatch_errors():
    rng = np.random.default_rng(16)
    res = compute_sliced_wasserstein(
        rng.standard_normal((500, 3)),
        rng.standard_normal((500, 5)),
    )
    assert res.quality == "poor"
    assert not res.passed
