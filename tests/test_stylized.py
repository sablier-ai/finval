"""v0.4.0 gap-close: stylized-fact / SOTA metrics. Each must DETECT the failure it targets
(mismatch > matched), where the existing suite would miss it."""

from __future__ import annotations

import numpy as np

from finval import validate, validate_paths
from finval.metrics.stylized import (
    compute_aggregational_gaussianity,
    compute_conditional_heavy_tails,
    compute_hill_tail_index,
    compute_regime_persistence,
    compute_signature_distance,
    compute_time_reversal_asymmetry,
    compute_variogram_score,
)


def _iid(n, H, D, seed, vol=0.012):
    return np.random.default_rng(seed).normal(0, vol, (n, H, D))


def _gjr_garch(n, H, D, seed, alpha=0.05, gamma=0.10, beta=0.80, t_df=None):
    """GJR-GARCH (stationary: alpha+beta+gamma/2 < 1): leverage (neg returns boost future vol) →
    time-IRREVERSIBLE. Optional Student-t innovations (heavy conditional tails)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, H, D))
    w = max(1e-8, 0.012**2 * (1 - beta - alpha - gamma / 2))
    for p in range(n):
        s2 = np.full(D, 0.012**2)
        rprev = np.zeros(D)
        for t in range(H):
            e = rng.standard_t(t_df, D) / np.sqrt(t_df / (t_df - 2)) if t_df else rng.normal(0, 1, D)
            r = np.sqrt(np.maximum(s2, 1e-12)) * e
            out[p, t] = r
            s2 = np.maximum(w + (alpha + gamma * (rprev < 0)) * rprev**2 + beta * s2, 1e-12)
            rprev = r
    return out


def test_time_reversal_detects_irreversibility():
    real = _gjr_garch(150, 50, 2, seed=0)        # leverage → irreversible
    iid = _iid(150, 50, 2, seed=1)               # reversible
    matched = _gjr_garch(150, 50, 2, seed=2)
    assert (compute_time_reversal_asymmetry(iid, real).value
            > compute_time_reversal_asymmetry(matched, real).value)


def test_regime_persistence_detects_short_crises():
    real = _gjr_garch(150, 60, 2, seed=3, alpha=0.03, gamma=0.04, beta=0.90)  # highly persistent vol
    iid = _iid(150, 60, 2, seed=4)                                            # no persistence
    matched = _gjr_garch(150, 60, 2, seed=5, alpha=0.03, gamma=0.04, beta=0.90)
    assert (compute_regime_persistence(iid, real).value
            > compute_regime_persistence(matched, real).value)


def test_aggregational_gaussianity_detects_kurtosis_curve_mismatch():
    # Laplace = excess kurtosis 3, ALL moments finite → stable sample kurtosis (Student-t(3)'s 4th
    # moment is infinite → unusable). Daily-heavy, decays toward Gaussian on aggregation.
    rng = np.random.default_rng(6)
    real = rng.laplace(0, 0.012 / np.sqrt(2), (300, 40, 2))
    gauss = _iid(300, 40, 2, seed=7)                                  # ~0 kurtosis at all horizons
    matched = np.random.default_rng(8).laplace(0, 0.012 / np.sqrt(2), (300, 40, 2))
    assert (compute_aggregational_gaussianity(gauss, real).value
            > compute_aggregational_gaussianity(matched, real).value)


def test_conditional_heavy_tails_detects_gaussian_residuals():
    # t_df=10 → finite, stable residual kurtosis (df<=4 is infinite-kurtosis noise). Heavy
    # innovations → fat devol'd residuals; gaussian innovations → ~Gaussian residuals.
    real = _gjr_garch(250, 80, 2, seed=9, t_df=10)
    gauss_innov = _gjr_garch(250, 80, 2, seed=10)
    matched = _gjr_garch(250, 80, 2, seed=11, t_df=10)
    assert (compute_conditional_heavy_tails(gauss_innov, real).value
            > compute_conditional_heavy_tails(matched, real).value)


def test_hill_tail_index_detects_wrong_tail_shape():
    rng = np.random.default_rng(12)
    real = rng.standard_t(2.5, (3000, 2))            # power-law tail
    gauss = rng.standard_normal((3000, 2))           # exponential tail (different shape)
    matched = np.random.default_rng(13).standard_t(2.5, (3000, 2))
    assert compute_hill_tail_index(gauss, real).value > compute_hill_tail_index(matched, real).value


def test_variogram_detects_dependence_error_energy_misses():
    # same marginals, different DEPENDENCE — the case energy distance is documented to under-detect
    rng = np.random.default_rng(14)
    z = rng.standard_normal((3000, 2))
    real = np.column_stack([z[:, 0], 0.9 * z[:, 0] + np.sqrt(1 - 0.81) * z[:, 1]])  # corr ~0.9
    indep = rng.standard_normal((3000, 2))                                          # corr ~0, same marginals
    matched = np.column_stack([
        (m := np.random.default_rng(15).standard_normal((3000, 2)))[:, 0],
        0.9 * m[:, 0] + np.sqrt(1 - 0.81) * m[:, 1],
    ])
    assert compute_variogram_score(indep, real).value > compute_variogram_score(matched, real).value


def test_signature_detects_lead_lag_with_identical_marginals():
    rng = np.random.default_rng(16)
    a = rng.normal(0, 0.01, (200, 40))
    real = np.stack([a, np.concatenate([np.zeros((200, 1)), a[:, :-1]], axis=1)], axis=2)  # ch1 lags ch0
    ai = rng.normal(0, 0.01, (200, 40))
    indep = np.stack([ai, rng.normal(0, 0.01, (200, 40))], axis=2)            # independent, same marginals
    a2 = np.random.default_rng(17).normal(0, 0.01, (200, 40))
    matched = np.stack([a2, np.concatenate([np.zeros((200, 1)), a2[:, :-1]], axis=1)], axis=2)
    assert (compute_signature_distance(indep, real).value
            > compute_signature_distance(matched, real).value)


def test_new_metrics_in_panels():
    rng = np.random.default_rng(18)
    fr, fs = rng.normal(0, 0.01, (1200, 3)), rng.normal(0, 0.01, (1200, 3))
    frep = validate(fs, fr, metrics=["hill_tail_index", "variogram_score"])
    assert {"hill_tail_index", "variogram_score"} <= set(frep.metrics)
    pr, ps = _gjr_garch(80, 40, 3, 19), _gjr_garch(80, 40, 3, 20)
    prep = validate_paths(ps, pr, metrics=["time_reversal_asymmetry", "aggregational_gaussianity",
                                           "conditional_heavy_tails", "regime_persistence", "signature_distance"])
    assert {"time_reversal_asymmetry", "aggregational_gaussianity", "conditional_heavy_tails",
            "regime_persistence", "signature_distance"} <= set(prep.metrics)
