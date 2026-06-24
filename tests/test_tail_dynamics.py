"""v0.4.0 Phase-2 tail/dynamics localizers — each catches what the headline metric only shadows."""

from __future__ import annotations

import numpy as np

from finval import validate, validate_paths
from finval.metrics.tail_dynamics import (
    compute_coskewness,
    compute_extreme_clustering,
    compute_far_tail_quantiles,
    compute_long_memory,
    compute_marginal_skew,
    compute_variance_term_structure,
)


def _iid_paths(n, H, D, seed, vol=0.012):
    return np.random.default_rng(seed).normal(0, vol, (n, H, D))


def _garch_paths(n, H, D, seed, alpha=0.85):
    rng = np.random.default_rng(seed)
    out = np.empty((n, H, D))
    for p in range(n):
        s = np.full(D, 0.012)
        for t in range(H):
            e = rng.normal(0, 1, D)
            out[p, t] = s * e
            s = np.sqrt(0.012**2 * (1 - alpha) + alpha * (s * e) ** 2)
    return out


def _ar1_paths(n, H, D, seed, phi=0.4, vol=0.01):
    rng = np.random.default_rng(seed)
    out = np.empty((n, H, D))
    for p in range(n):
        x = np.zeros(D)
        for t in range(H):
            x = phi * x + rng.normal(0, vol, D)
            out[p, t] = x
    return out


def _longmem_paths(n, H, D, seed, phi=0.98):
    """Highly persistent log-vol → |returns| ACF decays slowly = genuine long memory (GARCH
    has only short/exponential memory, so it can't drive a long-lag ACF test)."""
    rng = np.random.default_rng(seed)
    base = np.log(0.012)
    out = np.empty((n, H, D))
    for p in range(n):
        logv = np.full(D, base)
        for t in range(H):
            logv = phi * logv + (1 - phi) * base + rng.normal(0, 0.3, D)
            out[p, t] = np.exp(logv) * rng.normal(0, 1, D)
    return out


def test_far_tail_quantiles_detects_heavier_tail():
    rng = np.random.default_rng(0)
    real = rng.standard_normal((4000, 3))
    matched = np.random.default_rng(1).standard_normal((4000, 3))
    heavy = np.random.default_rng(2).standard_t(2.5, (4000, 3))   # much heavier far tail
    assert compute_far_tail_quantiles(heavy, real).value > compute_far_tail_quantiles(matched, real).value


def test_marginal_skew_detects_asymmetry():
    rng = np.random.default_rng(3)
    real = rng.standard_normal((2000, 2))                          # symmetric
    skewed = np.random.default_rng(4).exponential(1.0, (2000, 2))  # right-skewed
    assert compute_marginal_skew(skewed, real).value > compute_marginal_skew(real, real).value
    assert compute_marginal_skew(real, real).value < 0.2          # matched ~ small


def test_coskewness_detects_asymmetric_comovement():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((3000, 2))
    real = np.column_stack([z[:, 0], z[:, 1] + 0.6 * z[:, 0] ** 2])   # j's mean tied to i's variance
    synth = rng.standard_normal((3000, 2))                            # no coskew
    assert compute_coskewness(synth, real).value > compute_coskewness(real, real).value


def test_variance_term_structure_detects_scaling_mismatch():
    real = _ar1_paths(200, 40, 3, seed=6, phi=0.4)        # autocorrelated → variance scales super-linearly
    iid = _iid_paths(200, 40, 3, seed=7)                  # variance scales linearly
    matched = _ar1_paths(200, 40, 3, seed=8, phi=0.4)
    assert (compute_variance_term_structure(iid, real).value
            > compute_variance_term_structure(matched, real).value)


def test_extreme_clustering_detects_bursts():
    real = _garch_paths(150, 50, 2, seed=9)              # vol clusters → exceedances cluster
    iid = _iid_paths(150, 50, 2, seed=10)               # no clustering
    matched = _garch_paths(150, 50, 2, seed=11)
    assert (compute_extreme_clustering(iid, real).value
            > compute_extreme_clustering(matched, real).value)


def test_long_memory_detects_persistence():
    real = _longmem_paths(150, 120, 2, seed=12)          # genuine long memory (persistent log-vol)
    iid = _iid_paths(150, 120, 2, seed=13)               # no memory
    matched = _longmem_paths(150, 120, 2, seed=14)
    assert compute_long_memory(iid, real).value > compute_long_memory(matched, real).value


def test_localizers_in_panels():
    rng = np.random.default_rng(15)
    flat_r = rng.normal(0, 0.01, (1000, 3))
    flat_s = rng.normal(0, 0.01, (1000, 3))
    frep = validate(flat_s, flat_r, metrics=["far_tail_quantiles", "marginal_skew", "coskewness"])
    assert {"far_tail_quantiles", "marginal_skew", "coskewness"} <= set(frep.metrics)

    pr, ps = _garch_paths(80, 40, 3, 16), _garch_paths(80, 40, 3, 17)
    prep = validate_paths(ps, pr, metrics=["variance_term_structure", "extreme_clustering", "long_memory"])
    assert {"variance_term_structure", "extreme_clustering", "long_memory"} <= set(prep.metrics)
