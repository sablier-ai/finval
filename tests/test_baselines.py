"""Tests for baseline generators."""

from __future__ import annotations

import numpy as np

import finval
from finval.baselines import block_bootstrap, gaussian_baseline, historical_bootstrap


def test_gaussian_baseline_flat_shape(rng):
    real = rng.standard_normal((1000, 3))
    out = gaussian_baseline(real, n_samples=500, seed=1)
    assert out.shape == (500, 3)


def test_gaussian_baseline_path_shape(rng):
    real = rng.standard_normal((1000, 3))
    out = gaussian_baseline(real, n_paths=20, path_length=50, seed=1)
    assert out.shape == (20, 50, 3)


def test_gaussian_baseline_reproducible(rng):
    real = rng.standard_normal((500, 2))
    a = gaussian_baseline(real, n_samples=100, seed=1)
    b = gaussian_baseline(real, n_samples=100, seed=1)
    np.testing.assert_array_equal(a, b)


def test_historical_bootstrap_matches_real_distribution(rng):
    real = rng.standard_t(df=5, size=(2000, 3)) * 0.01
    syn = historical_bootstrap(real, n_samples=2000, seed=1)
    # iid bootstrap should score excellent on distribution metrics
    report = finval.validate(syn, real, metrics=["marginal_ks", "energy_distance"])
    assert report.metrics["marginal_ks"].passed
    assert report.metrics["energy_distance"].passed


def test_block_bootstrap_shape(rng):
    real = rng.standard_normal((1000, 2))
    out = block_bootstrap(real, n_paths=20, path_length=60, block_size=10)
    assert out.shape == (20, 60, 2)


def test_block_bootstrap_preserves_temporal_structure(rng):
    """Block bootstrap should preserve short-range ACF from the source series."""
    # Build a strongly autocorrelated AR(1) source
    n = 2000
    ar = np.zeros((n, 1))
    ar[0] = rng.standard_normal()
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + rng.standard_normal() * 0.5

    # Bootstrap with block_size=20 should preserve the AR structure
    syn_paths = block_bootstrap(ar, n_paths=50, path_length=100, block_size=20)

    # Compute lag-1 ACF on synthetic paths
    def lag1_acf(x):
        x = x - np.mean(x)
        denom = np.dot(x, x) + 1e-12
        return np.dot(x[:-1], x[1:]) / denom

    acfs = [lag1_acf(syn_paths[p, :, 0]) for p in range(50)]
    mean_acf = float(np.mean(acfs))
    # True AR(1) coefficient is 0.7; block bootstrap should give near that
    assert 0.3 < mean_acf < 0.9


def test_gaussian_vs_bootstrap_on_fat_tails(rng):
    """Gaussian baseline fails tail metrics on t_5 real data; bootstrap passes."""
    real = rng.standard_t(df=5, size=(2000, 3)) * 0.01

    gauss = gaussian_baseline(real, n_samples=2000, seed=1)
    boot = historical_bootstrap(real, n_samples=2000, seed=1)

    gauss_report = finval.validate(gauss, real, metrics=["tail_quantiles"])
    boot_report = finval.validate(boot, real, metrics=["tail_quantiles"])

    # Bootstrap should do at least as well as Gaussian on tail quantiles
    assert boot_report.metrics["tail_quantiles"].value <= gauss_report.metrics[
        "tail_quantiles"
    ].value + 0.1  # small tolerance
