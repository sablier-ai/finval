"""v0.4.0 generative-health (density/coverage vs bootstrap) — the generator-vs-replay axis.

A faithful generator covers the real manifold (low coverage_deficit) and stays on it (low
plausibility_deficit). A mode-collapsed generator misses real regions → high coverage_deficit
AND worse than the block-bootstrap replay. An off-manifold generator → high plausibility_deficit.
"""

from __future__ import annotations

import numpy as np

from finval import validate_generative
from finval.metrics.generative import density_coverage, path_features

CENTERS = np.array([[0.0, 0.0], [6.0, 6.0], [-6.0, 6.0]])  # 3 well-separated modes


def _mixture(n, rng, centers=CENTERS, scale=0.3):
    comp = rng.integers(0, len(centers), size=n)
    return centers[comp] + rng.normal(0.0, scale, size=(n, centers.shape[1]))


def test_faithful_generator_passes_both():
    rng = np.random.default_rng(0)
    real = _mixture(800, rng)
    synth = _mixture(800, np.random.default_rng(1))
    rep = validate_generative(synth, real, k=5)
    assert rep.metrics["coverage_deficit"].passed
    assert rep.metrics["plausibility_deficit"].passed


def test_mode_collapse_is_caught_and_worse_than_replay():
    rng = np.random.default_rng(2)
    real = _mixture(800, rng)
    # collapse to a single mode → two of three real regions unrepresented
    collapsed = CENTERS[0] + np.random.default_rng(3).normal(0.0, 0.3, size=(800, 2))
    faithful = _mixture(800, np.random.default_rng(4))

    rc = validate_generative(collapsed, real, k=5)
    rf = validate_generative(faithful, real, k=5)
    # collapse has far higher coverage deficit than a faithful generator
    assert rc.metrics["coverage_deficit"].value > rf.metrics["coverage_deficit"].value + 0.3
    assert not rc.metrics["coverage_deficit"].passed  # ~0.67 → poor
    # and it is WORSE than the block-bootstrap replay (positive vs-bootstrap delta)
    assert rc.metrics["coverage_deficit"].metadata["coverage_deficit_vs_bootstrap"] > 0.1


def test_offmanifold_high_plausibility_deficit():
    rng = np.random.default_rng(5)
    real = _mixture(800, rng)
    shifted = _mixture(800, np.random.default_rng(6)) + 25.0   # entirely off the real manifold
    faithful = _mixture(800, np.random.default_rng(7))
    r_off = validate_generative(shifted, real, k=5)
    r_ok = validate_generative(faithful, real, k=5)
    assert r_off.metrics["plausibility_deficit"].value > r_ok.metrics["plausibility_deficit"].value + 0.3
    assert not r_off.metrics["plausibility_deficit"].passed


def test_bootstrap_reference_is_near_zero_deficit():
    # an i.i.d. resample of real (what the baseline is) should itself score ~perfectly
    rng = np.random.default_rng(8)
    real = _mixture(800, rng)
    resample = real[np.random.default_rng(9).integers(0, len(real), size=800)]
    rep = validate_generative(resample, real, k=5)
    assert rep.metrics["coverage_deficit"].value < 0.15
    assert rep.metrics["plausibility_deficit"].value < 0.15


def test_paths_input_featurizes_and_runs():
    rng = np.random.default_rng(10)
    real_paths = rng.normal(0.0, 0.01, size=(120, 30, 3))
    synth_paths = rng.normal(0.0, 0.01, size=(120, 30, 3))
    rep = validate_generative(synth_paths, real_paths, k=4)
    assert np.isfinite(rep.metrics["coverage_deficit"].value)
    assert rep.metrics["coverage_deficit"].metadata["input"] == "paths"
    # path_features shape: 4 summaries × 3 features
    assert path_features(real_paths).shape == (120, 12)


def test_density_coverage_directly():
    rng = np.random.default_rng(11)
    real = _mixture(600, rng)
    dens, cov = density_coverage(_mixture(600, np.random.default_rng(12)), real, k=5)
    assert 0.0 <= cov <= 1.0 and dens >= 0.0
    assert cov > 0.7  # faithful → high recall
