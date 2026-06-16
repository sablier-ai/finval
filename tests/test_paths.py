"""Tests for path-level metrics."""

from __future__ import annotations

import numpy as np

from finval.metrics.paths import (
    compute_covariance_calibration,
    compute_drawdown_distribution,
    compute_tail_dependence_asymmetry,
)


def _returns_to_levels(returns: np.ndarray) -> np.ndarray:
    """Cumulative product for log-return price levels."""
    return np.exp(np.cumsum(returns, axis=1))


def _clayton_like_asym(rng, n, d=3):
    """A multivariate sample with LOWER-tail clustering (A = λ_L - λ_U > 0).

    Built by mixing a heavy common crash factor into the lower tail only:
    a shared Student-t shock drives joint crashes, idiosyncratic noise breaks
    up the rallies. Mimics the real-equity crash-co-movement asymmetry that an
    elliptical (radially symmetric) generator cannot reproduce.
    """
    crash = -np.abs(rng.standard_t(df=2.5, size=(n, 1)))  # common, lower-tail-only
    idio = rng.standard_normal((n, d))
    return 0.6 * crash + 0.9 * idio


def test_tail_asymmetry_matching_passes(rng):
    """Two draws from the same asymmetric process should match (low bias)."""
    real = _clayton_like_asym(rng, 8000)
    syn = _clayton_like_asym(rng, 8000)
    result = compute_tail_dependence_asymmetry(syn, real, quantile=0.10)
    assert result.passed
    # Real asymmetry should be meaningfully positive (crashes cluster).
    assert result.metadata["mean_asymmetry_real"] > 0.05
    # An honest resample carries little directional bias.
    assert result.metadata["systematic_bias"] < result.metadata["mean_asymmetry_real"]


def test_tail_asymmetry_elliptical_synth_penalized(rng):
    """A radially-symmetric (Gaussian) synth erases A; bias ≈ the real asymmetry.

    The Gaussian is graded strictly worse than a matched asymmetric resample and
    its panel-level bias is large (it cannot be the better of the two), even if a
    weakly-asymmetric panel only pushes it to 'acceptable' rather than 'poor'.
    """
    real = _clayton_like_asym(rng, 8000)
    matched = _clayton_like_asym(rng, 8000)
    cov = np.corrcoef(real, rowvar=False)
    gauss = rng.multivariate_normal(np.zeros(real.shape[1]), cov, size=8000)

    r_match = compute_tail_dependence_asymmetry(matched, real, quantile=0.10)
    r_gauss = compute_tail_dependence_asymmetry(gauss, real, quantile=0.10)

    # Gaussian erases the asymmetry → much larger panel bias than a matched draw.
    assert r_gauss.value > r_match.value
    # And it captures essentially none of the real (positive) asymmetry.
    assert r_gauss.metadata["mean_asymmetry_synth"] < 0.5 * real_asym(r_gauss)


def real_asym(result) -> float:
    return result.metadata["mean_asymmetry_real"]


def test_tail_asymmetry_accepts_3d_paths(real_paths_3d, matched_synthetic_paths_3d):
    """3D path input is flattened to cross-sectional rows and runs."""
    result = compute_tail_dependence_asymmetry(
        matched_synthetic_paths_3d, real_paths_3d, quantile=0.10
    )
    assert np.isfinite(result.value)


def test_cov_calibration_matching_passes(real_returns_2d, matched_synthetic_2d):
    """Matched second-moment structure → dispersion ratios ≈ 1 → passes."""
    result = compute_covariance_calibration(matched_synthetic_2d, real_returns_2d)
    assert result.passed


def test_cov_calibration_underdispersed_corr_fails(rng):
    """Shrinking the SPREAD of pairwise correlations (under-dispersion) is caught."""
    n = 6000
    d = 6
    f = rng.standard_normal((n, 1))
    # Real: heterogeneous loadings on a common factor → some pairs highly
    # correlated, some weakly → WIDE spread of off-diagonal correlations.
    loadings = np.array([1.4, 1.1, 0.8, 0.2, 0.05, 0.0])
    real = f * loadings + rng.standard_normal((n, d))
    # Synth: homogeneous moderate loadings → all pairwise correlations bunched
    # near one value → SHRUNK spread (under-dispersed) at a similar mean level.
    syn = f * 0.55 + rng.standard_normal((n, d))
    result = compute_covariance_calibration(syn, real)
    assert not result.passed
    assert result.metadata["corr_dispersion_ratio"] < 0.8


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
