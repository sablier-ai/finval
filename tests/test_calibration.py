"""Tests for calibration metrics."""

from __future__ import annotations

import numpy as np

from finval.metrics.calibration import (
    compute_coverage,
    compute_crps,
    compute_pit_uniformity,
)


def _make_wellcalibrated_forecasts(
    rng: np.random.Generator,
    n_obs: int = 200,
    n_samples: int = 100,
    n_features: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate well-calibrated forecasts: samples and actuals from the same N(0, 1) distribution."""
    actuals = rng.standard_normal((n_obs, n_features))
    samples = rng.standard_normal((n_obs, n_samples, n_features))
    return samples, actuals


def _make_overconfident_forecasts(
    rng: np.random.Generator,
    n_obs: int = 200,
    n_samples: int = 100,
    n_features: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Overconfident: forecast distribution is too narrow."""
    actuals = rng.standard_normal((n_obs, n_features))
    # Forecast samples have std = 0.3, but actuals have std = 1.0
    samples = rng.standard_normal((n_obs, n_samples, n_features)) * 0.3
    return samples, actuals


def test_pit_uniformity_wellcalibrated(rng):
    samples, actuals = _make_wellcalibrated_forecasts(rng)
    result = compute_pit_uniformity(samples, actuals)
    assert result.passed


def test_pit_uniformity_overconfident_fails(rng):
    samples, actuals = _make_overconfident_forecasts(rng)
    result = compute_pit_uniformity(samples, actuals)
    assert not result.passed


def test_crps_gaussian_near_theoretical_optimum(rng):
    """E[CRPS/sigma] for well-calibrated N(0,1) forecasts = 1/sqrt(pi) ~= 0.564."""
    samples, actuals = _make_wellcalibrated_forecasts(rng, n_obs=500, n_samples=200)
    result = compute_crps(samples, actuals)
    # Expected value ~0.564, allow ±25% slack for finite sample
    assert 0.40 < result.value < 0.75


def test_crps_overconfident_worse(rng):
    samples, actuals = _make_overconfident_forecasts(rng, n_obs=500, n_samples=200)
    result = compute_crps(samples, actuals)
    # Overconfident forecasts score much higher than well-calibrated (~0.564)
    well_samples, well_actuals = _make_wellcalibrated_forecasts(
        rng, n_obs=500, n_samples=200
    )
    well_result = compute_crps(well_samples, well_actuals)
    assert result.value > well_result.value + 0.05


def test_coverage_90_wellcalibrated_near_nominal(rng):
    samples, actuals = _make_wellcalibrated_forecasts(rng, n_obs=500, n_samples=200)
    result = compute_coverage(samples, actuals, level=0.90)
    assert result.value < 0.10
    assert result.metadata["empirical"] == result.metadata["empirical"]  # sanity


def test_coverage_90_overconfident_much_too_low(rng):
    samples, actuals = _make_overconfident_forecasts(rng, n_obs=500, n_samples=200)
    result = compute_coverage(samples, actuals, level=0.90)
    assert not result.passed
    assert result.metadata["empirical"] < 0.80  # much below nominal 90%


def test_coverage_multiple_levels(rng):
    samples, actuals = _make_wellcalibrated_forecasts(rng, n_obs=500, n_samples=200)
    for level in (0.50, 0.90, 0.95):
        result = compute_coverage(samples, actuals, level=level)
        assert result.name == f"coverage_{int(level * 100)}"


def test_calibration_wrong_shape_errors(rng):
    # 2D samples where we expect 3D
    samples = rng.standard_normal((100, 2))
    actuals = rng.standard_normal((100, 2))
    result = compute_pit_uniformity(samples, actuals)
    assert result.quality == "poor"
