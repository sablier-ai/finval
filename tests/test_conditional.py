"""v0.4.0 regime-sensitivity diagnostics — and proof they don't touch the scored panel.

The core claim: a regime-BLIND climatology generator passes the pooled panel but must FAIL
`conditional_sensitivity`; a regime-sensitive one must pass. And `validate`/`validate_paths`/
`validate_calibration` must be byte-identical to v0.3.0 (the new metrics are additive only).
"""

from __future__ import annotations

import numpy as np

import finval
from finval import validate, validate_conditional
from finval.core.thresholds import default_absolute_weights
from finval.metrics.conditional import conditional_sensitivity, regime_stratified_calibration
from finval.validate import _resolve_metrics

CALM, STRESS = 0.01, 0.05
N_PER, N_SAMP, N_FEAT = 80, 120, 3


def _regime_data(sensitive: bool, seed: int = 0):
    """Two regimes (calm/stress) with genuinely different realized vol. A *sensitive* model
    forecasts the right vol per regime; a *climatology* model forecasts the same vol always."""
    rng = np.random.default_rng(seed)
    labels = np.array(["calm"] * N_PER + ["stress"] * N_PER)
    sig = np.where(labels == "calm", CALM, STRESS)[:, None]          # (n_obs, 1)
    actuals = rng.normal(0.0, 1.0, (2 * N_PER, N_FEAT)) * sig
    if sensitive:
        fsig = sig[:, :, None]                                       # forecast vol tracks regime
    else:
        fsig = np.full((2 * N_PER, 1, 1), (CALM + STRESS) / 2)       # climatology: one vol always
    samples = rng.normal(0.0, 1.0, (2 * N_PER, N_SAMP, N_FEAT)) * fsig
    return samples, actuals, labels


def test_sensitivity_catches_climatology():
    s_samp, s_act, lab = _regime_data(sensitive=True, seed=1)
    c_samp, c_act, _ = _regime_data(sensitive=False, seed=1)
    sensitive = conditional_sensitivity(s_samp, s_act, lab)
    climatology = conditional_sensitivity(c_samp, c_act, lab)

    # the regime-sensitive model is scored far better than the climatology model
    assert sensitive.value < climatology.value
    # climatology must be flagged (its forecast doesn't move across regimes)
    assert climatology.quality == "poor" and not climatology.passed
    assert climatology.metadata["ratio"] < 0.5          # D_model << D_real
    # the sensitive model passes (forecast moves ~as much as reality)
    assert sensitive.passed
    assert sensitive.metadata["d_real"] > 0.0


def test_within_regime_gap_surfaces_hidden_miscalibration():
    # calm forecasts well-calibrated; stress forecasts far too NARROW (under-dispersed) →
    # pooled CRPS looks ok-ish, but the stress bucket is badly miscalibrated.
    rng = np.random.default_rng(2)
    labels = np.array(["calm"] * N_PER + ["stress"] * N_PER)
    sig = np.where(labels == "calm", CALM, STRESS)[:, None]
    actuals = rng.normal(0.0, 1.0, (2 * N_PER, N_FEAT)) * sig
    fsig = np.where(labels == "calm", CALM, CALM)[:, None, None]      # stress under-dispersed
    samples = rng.normal(0.0, 1.0, (2 * N_PER, N_SAMP, N_FEAT)) * fsig

    out = regime_stratified_calibration(samples, actuals, labels)
    assert "crps@calm" in out and "crps@stress" in out
    assert out["crps@stress"].value > out["crps@calm"].value          # stress is worse
    gap = out["within_regime_calibration_gap"]
    assert gap.value > 0.0                                            # pooling hid the stress failure


def test_validate_conditional_report():
    samp, act, lab = _regime_data(sensitive=True, seed=3)
    report = validate_conditional(samp, act, lab)
    assert "conditional_sensitivity" in report.metrics
    assert "within_regime_calibration_gap" in report.metrics
    assert any(k.startswith("crps@") for k in report.metrics)
    assert 0.0 <= report.overall_score <= 1.0


# ---- isolation: the scored panel is provably untouched ------------------------------------
def test_new_metrics_are_not_in_the_scored_panel():
    assert "conditional_sensitivity" not in _resolve_metrics("all")
    assert "conditional_sensitivity" not in default_absolute_weights()
    assert "within_regime_calibration_gap" not in default_absolute_weights()


def test_default_validate_does_not_run_new_metrics():
    rng = np.random.default_rng(4)
    real = rng.normal(0, 0.01, (1000, 3))
    syn = rng.normal(0, 0.01, (1000, 3))
    report = validate(syn, real)
    assert "conditional_sensitivity" not in report.metrics
    assert "within_regime_calibration_gap" not in report.metrics


def test_version_bumped():
    assert finval.__version__ == "0.5.0"


def test_multi_axis_catches_partial_conditioning():
    # Two independent axes: vol (the model tracks it) and trend (the model is BLIND to it,
    # though reality's MEAN depends on it). Multi-axis must flag the trend blindness.
    rng = np.random.default_rng(20)
    n, NS, F = 200, 120, 3
    vol_hi = np.arange(2 * n) >= n                      # calm | stress
    trend_up = (np.arange(2 * n) % 2 == 0)             # up | down, independent of vol
    vol = np.where(vol_hi, 0.05, 0.01)[:, None]
    mu = np.where(trend_up, +0.02, -0.02)[:, None]
    actuals = rng.normal(0, 1, (2 * n, F)) * vol + mu  # vol←vol-axis, MEAN←trend-axis
    samples = rng.normal(0, 1, (2 * n, NS, F)) * vol[:, :, None]   # model tracks vol, blind to trend
    labels = {"vol": np.where(vol_hi, "high", "low"), "trend": np.where(trend_up, "up", "down")}

    rep = validate_conditional(samples, actuals, labels)
    assert "conditional_sensitivity@vol" in rep.metrics
    assert "conditional_sensitivity@trend" in rep.metrics
    # responsive on vol, blind on trend
    assert rep.metrics["conditional_sensitivity@vol"].value < rep.metrics["conditional_sensitivity@trend"].value
    assert not rep.metrics["conditional_sensitivity@trend"].passed
    # headline = the worst (least-responsive) axis = trend
    assert rep.metrics["conditional_sensitivity"].value == rep.metrics["conditional_sensitivity@trend"].value


def test_single_axis_array_still_works():
    s_samp, s_act, lab = _regime_data(sensitive=True, seed=1)
    rep = validate_conditional(s_samp, s_act, lab)            # array, not dict (back-compat)
    assert "conditional_sensitivity" in rep.metrics
    assert "conditional_sensitivity@vol" in rep.metrics


def test_undefined_regime_is_not_a_model_failure():
    """A degenerate regime split (only one populated regime, or real outcomes that don't
    separate across regimes) is UNDEFINED — the property can't be measured on this data —
    NOT a model failure. It must be applicable=False, not graded "poor", and never gate.
    Regression for the cross-dataset OverflowError where inf poisoned seed-aggregation."""
    rng = np.random.default_rng(7)
    fs = rng.normal(0, 1, (60, 100, N_FEAT))
    act = rng.normal(0, 1, (60, N_FEAT))
    labels = np.array(["calm"] * 60)                       # one regime → responsiveness undefined
    res = conditional_sensitivity(fs, act, labels)
    assert res.applicable is False
    assert res.quality != "poor"                           # not a measured failure
    assert res.metadata.get("undefined_reason")
    # to_dict surfaces the flag so downstream consumers can exclude it
    assert res.to_dict()["applicable"] is False


def test_validate_full_not_gated_by_undefined_conditional():
    """validate_full must not put `conditional_sensitivity` in failing_gates when the axis is
    undefined on this data (the model-agnostic guarantee: don't penalize a model for the
    user's data lacking regime structure)."""
    rng = np.random.default_rng(8)
    fs = rng.normal(0, 1, (60, 100, N_FEAT))
    act = rng.normal(0, 1, (60, N_FEAT))
    labels = np.array(["calm"] * 60)
    rep = finval.validate_full(forecast_samples=fs, actuals=act, regime_labels=labels)
    assert "conditional_sensitivity" not in rep.failing_gates


def test_conditional_headline_na_when_most_axes_undefined():
    """Long-horizon thin-data: many axes requested but most go undefined (regimes below floor) →
    don't score conditioning off the single surviving noisy axis; headline = not-applicable."""
    rng = np.random.default_rng(11)
    n = 90
    fs = rng.normal(0, 1, (n, 80, N_FEAT)); act = rng.normal(0, 1, (n, N_FEAT))
    # 4 degenerate axes (one populated regime → undefined) + 1 valid 2-regime axis
    one = np.array(["a"] * n)
    sig = np.where(np.arange(n) < n // 2, 0.01, 0.06)[:, None]
    act_v = rng.normal(0, 1, (n, N_FEAT)) * sig
    valid = np.where(np.arange(n) < n // 2, "lo", "hi")
    labels = {"vol": one, "trend": one, "drawdown": one, "vol_term": one, "dispersion": valid}
    rep = validate_conditional(fs, act_v, labels)
    cs = rep.metrics["conditional_sensitivity"]
    assert cs.applicable is False, f"expected N/A with only 1/5 axes measurable, got {cs.value}"
