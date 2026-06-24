"""v0.4.0 joint-lens omnibus — the classifier two-sample test (C2ST).

If a kNN classifier can't beat chance, real and synthetic are indistinguishable (c2st ~ 0,
high p). If they differ anywhere, c2st climbs toward 1 with a significant p. This is the
catch-all for differences no named statistic anticipated.
"""

from __future__ import annotations

import numpy as np

from finval import validate
from finval.metrics.joint import compute_c2st


def test_c2st_indistinguishable():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (800, 4))
    synth = rng.normal(0, 1, (800, 4))                # same distribution
    r = compute_c2st(synth, real, seed=0)
    assert r.value < 0.20 and r.passed
    assert r.metadata["p_value"] > 0.05              # not significantly separable


def test_c2st_distinguishable():
    rng = np.random.default_rng(1)
    real = rng.normal(0, 1, (800, 4))
    synth = rng.normal(0.0, 1, (800, 4)) + 1.5        # clearly shifted off real
    r = compute_c2st(synth, real, seed=0)
    assert r.value > 0.5 and not r.passed
    assert r.metadata["p_value"] < 0.05              # significantly separable


def test_c2st_catches_nonlinear_difference():
    # same marginals & ~same correlation, but different DEPENDENCE (a logistic C2ST would miss this)
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, (1000, 2))
    real = z.copy()
    synth = rng.normal(0, 1, (1000, 2))               # independent vs ... make real dependent
    real[:, 1] = 0.9 * real[:, 0] + 0.44 * rng.normal(0, 1, 1000)  # correlated
    r = compute_c2st(synth, real, seed=0)
    assert r.value > 0.2                               # the dependence difference is detected


def test_c2st_in_validate_panel():
    rng = np.random.default_rng(3)
    real = rng.normal(0, 0.01, (600, 3))
    synth = rng.normal(0, 0.01, (600, 3))
    rep = validate(synth, real, metrics=["c2st"])
    assert "c2st" in rep.metrics and rep.metrics["c2st"].category == "joint"


def test_c2st_paths_featurized():
    rng = np.random.default_rng(4)
    real = rng.normal(0, 0.01, (150, 30, 3))
    synth = rng.normal(0, 0.01, (150, 30, 3))
    r = compute_c2st(synth, real, k=5, seed=0)        # 3D → path-features internally
    assert np.isfinite(r.value)
