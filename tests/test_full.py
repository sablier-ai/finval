"""v0.4.0 validate_full — the comprehensive multi-lens scorer (per-lens vector + hard gates).

Runs whatever entry points the inputs allow, renormalizes over present lenses, and gates on
critical failure modes. Model-agnostic: only arrays in.
"""

from __future__ import annotations

import numpy as np

import finval


def _two_mode_paths(n_each, H, D, seed, lo=0.01, hi=0.05):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, lo, (n_each, H, D)), rng.normal(0, hi, (n_each, H, D))])


def _cond_data(seed=0, sensitive=True):
    rng = np.random.default_rng(seed)
    n, NS, F = 80, 60, 3
    lab = np.array(["calm"] * n + ["stress"] * n)
    sig = np.where(lab == "calm", 0.01, 0.05)[:, None]
    actuals = rng.normal(0, 1, (2 * n, F)) * sig
    fsig = sig[:, :, None] if sensitive else np.full((2 * n, 1, 1), 0.03)
    samples = rng.normal(0, 1, (2 * n, NS, F)) * fsig
    return samples, actuals, lab


def test_full_all_lenses_present():
    real = _two_mode_paths(100, 30, 3, seed=0)
    synth = _two_mode_paths(100, 30, 3, seed=1)
    fs, act, lab = _cond_data(sensitive=True)
    rep = finval.validate_full(synth, real, forecast_samples=fs, actuals=act, regime_labels=lab)
    # all six lenses scorable with these inputs
    assert {"marginal", "dependence", "temporal", "joint", "generative", "conditional"} <= set(rep.per_lens)
    assert 0.0 <= rep.overall_score <= 1.0
    assert isinstance(rep.summary(), str) and "per-lens" in rep.summary()
    assert "overall_score" in rep.to_dict()


def test_full_partial_inputs_conditional_from_paths_only():
    real = _two_mode_paths(100, 30, 3, seed=2)
    synth = _two_mode_paths(100, 30, 3, seed=3)
    rep = finval.validate_full(synth, real)              # no forecast inputs
    # regime_conditional is a PATH metric → it scores the conditional lens from paths alone...
    assert "regime_conditional" in rep.metrics
    # ...but the forecast-requiring conditional metrics did NOT run
    assert "conditional_sensitivity" not in rep.metrics
    assert "crps" not in rep.metrics
    assert {"marginal", "temporal", "joint", "generative"} <= set(rep.per_lens)
    assert 0.0 <= rep.overall_score <= 1.0


def test_full_hard_gate_on_mode_collapse():
    real = _two_mode_paths(100, 30, 3, seed=4, lo=0.01, hi=0.06)
    collapse = np.random.default_rng(5).normal(0, 0.01, (200, 30, 3))   # only the low-vol mode
    rep = finval.validate_full(collapse, real)
    assert "coverage_deficit" in rep.metrics
    assert rep.gated                                     # a hard gate fired
    assert rep.overall_quality == "poor"                # gated → poor regardless of weighted score


def test_full_is_model_agnostic_signature():
    # works with bare arrays, no model object anywhere
    real = np.random.default_rng(6).normal(0, 0.01, (120, 4))
    synth = np.random.default_rng(7).normal(0, 0.01, (120, 4))
    rep = finval.validate_full(synth, real)             # 2D rows → no temporal/conditional (need paths/forecasts)
    assert "temporal" not in rep.per_lens and "conditional" not in rep.per_lens
    assert {"marginal", "joint", "generative"} <= set(rep.per_lens)
    assert sum(rep.lens_weights_used.values()) < 1.0   # overall renormalized over present lenses
