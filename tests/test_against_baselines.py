"""v0.4.0 model-vs-baseline deltas — a metric only matters where the model beats replay.

The wrapper runs the pooled panel on the model AND on dumb baselines (block-bootstrap,
Gaussian) and reports per-metric deltas (delta < 0 = model beats the baseline). Model-agnostic:
nothing here knows what generated `synthetic`.
"""

from __future__ import annotations

import numpy as np

from finval import validate_against_baselines


def _ar1_paths(n_paths, H, D, rng, phi=0.0, vol=0.01):
    """Paths with optional vol-clustering (AR(1) in squared returns) — temporal structure
    a Gaussian baseline can't reproduce."""
    out = np.empty((n_paths, H, D))
    for p in range(n_paths):
        s = np.full(D, vol)
        for t in range(H):
            shock = rng.normal(0, 1, D)
            out[p, t] = s * shock
            s = np.sqrt((1 - phi) * vol**2 + phi * (s * shock) ** 2)  # GARCH-ish
    return out


def test_structure_and_keys():
    rng = np.random.default_rng(0)
    real = _ar1_paths(120, 30, 3, rng, phi=0.6)
    synth = _ar1_paths(120, 30, 3, np.random.default_rng(1), phi=0.6)
    res = validate_against_baselines(synth, real, metrics=["energy_distance", "acf_returns", "c2st"])
    assert set(res) == {"model", "baselines", "deltas", "summary"}
    assert "block_bootstrap" in res["deltas"] and "gaussian" in res["deltas"]
    for name in ("block_bootstrap", "gaussian"):
        assert set(res["summary"][name]) == {"n_better", "n_worse", "mean_delta"}


def test_flat_input_uses_validate():
    rng = np.random.default_rng(2)
    real = rng.normal(0, 0.01, (800, 3))
    synth = rng.normal(0, 0.01, (800, 3))
    res = validate_against_baselines(synth, real, metrics=["energy_distance", "marginal_ks"])
    # flat path → baselines are i.i.d. + gaussian; deltas computed for the requested metrics
    assert "energy_distance" in res["deltas"]["gaussian"]
    assert np.isfinite(res["deltas"]["gaussian"]["energy_distance"])


def test_model_with_vol_clustering_beats_gaussian_on_temporal():
    # a generator WITH vol-clustering should beat the (structureless) Gaussian baseline on
    # volatility_clustering — a negative delta on that metric.
    rng = np.random.default_rng(3)
    real = _ar1_paths(150, 40, 2, rng, phi=0.7)
    synth = _ar1_paths(150, 40, 2, np.random.default_rng(4), phi=0.7)   # same DGP → captures clustering
    res = validate_against_baselines(synth, real, metrics=["volatility_clustering"])
    assert res["deltas"]["gaussian"]["volatility_clustering"] < 0   # model beats Gaussian on clustering
