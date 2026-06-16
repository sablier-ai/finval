"""Quality thresholds for all finval metrics.

Each metric has three cutoffs: excellent, good, acceptable. Values below
the corresponding cutoff earn that grade; values above "acceptable" are
graded "poor".

All metrics in finval are normalized so that **lower is better**.

Thresholds are empirically calibrated against real financial data and
standard baselines (GBM, historical bootstrap). They are reasonable
defaults but should be tightened for higher-stakes applications via
the `thresholds` argument to each metric function.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# DISTRIBUTION METRICS
# ---------------------------------------------------------------------------

DISTRIBUTION_THRESHOLDS: dict[str, dict[str, float]] = {
    # Kolmogorov-Smirnov statistic, mean across features.
    # KS in [0, 1]; 0.05 is approximately the 5% critical value at n~1000.
    "marginal_ks": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # Normalized energy distance — measures multivariate distribution match.
    "energy_distance": {"excellent": 0.10, "good": 0.20, "acceptable": 0.40},
    # Normalized tail quantile error: mean |q_syn - q_real| / real_std
    # at the 1st, 5th, 95th, 99th percentiles, averaged across features.
    # More robust than kurtosis (SE ~ sqrt(24/n), heavily outlier-driven).
    "tail_quantiles": {"excellent": 0.10, "good": 0.20, "acceptable": 0.35},
    # Excess kurtosis error (diagnostic only — unstable).
    "tail_heaviness": {"excellent": 1.0, "good": 2.0, "acceptable": 4.0},
}

# ---------------------------------------------------------------------------
# DEPENDENCE METRICS
# ---------------------------------------------------------------------------

DEPENDENCE_THRESHOLDS: dict[str, dict[str, float]] = {
    # Pairwise correlation matrix error (mean across pairs, lower-triangle).
    "pearson_corr": {"excellent": 0.10, "good": 0.20, "acceptable": 0.30},
    "spearman_corr": {"excellent": 0.10, "good": 0.20, "acceptable": 0.30},
    # Empirical copula distance (Cramer-von Mises type).
    "copula_distance": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # Tail dependence coefficients (lambda_U, lambda_L) error, mean across pairs.
    "tail_dependence_upper": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    "tail_dependence_lower": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # Difference between stress-period and calm-period correlation error.
    "correlation_breakdown": {"excellent": 0.15, "good": 0.25, "acceptable": 0.40},
}

# ---------------------------------------------------------------------------
# TEMPORAL METRICS
# ---------------------------------------------------------------------------

TEMPORAL_THRESHOLDS: dict[str, dict[str, float]] = {
    # ACF error at lags {1, 5, 10, 20}. Real returns have near-zero ACF.
    "acf_returns": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # ACF of squared returns (volatility clustering) at lags {1..5}.
    "volatility_clustering": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # Leverage effect: corr(r_t, |r_{t+k}|) at lags {1, 2, 5, 10}. Negative
    # for equities; the model should capture both sign and magnitude.
    "leverage_effect": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # Cross-correlation error (mean across feature pairs).
    "cross_correlation": {"excellent": 0.10, "good": 0.20, "acceptable": 0.30},
}

# ---------------------------------------------------------------------------
# CALIBRATION METRICS
# ---------------------------------------------------------------------------

CALIBRATION_THRESHOLDS: dict[str, dict[str, float]] = {
    # Coverage error |actual - nominal| for each interval level.
    "coverage_50": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    "coverage_90": {"excellent": 0.05, "good": 0.10, "acceptable": 0.15},
    "coverage_95": {"excellent": 0.03, "good": 0.08, "acceptable": 0.12},
    # PIT uniformity KS statistic vs Uniform[0, 1].
    "pit_uniformity": {"excellent": 0.05, "good": 0.10, "acceptable": 0.20},
    # Continuous Ranked Probability Score, normalized by real std.
    # Under a well-calibrated Gaussian forecast, E[CRPS/sigma] = 1/sqrt(pi) ~= 0.564.
    # Thresholds are slightly tighter than this floor to reward sharper forecasts
    # (mean centered closer to actual); poor models quickly exceed 0.75.
    "crps": {"excellent": 0.58, "good": 0.65, "acceptable": 0.80},
}

# ---------------------------------------------------------------------------
# PATH-LEVEL METRICS
# ---------------------------------------------------------------------------

PATH_THRESHOLDS: dict[str, dict[str, float]] = {
    # KS statistic on max drawdown distribution across paths.
    "drawdown_distribution": {"excellent": 0.10, "good": 0.20, "acceptable": 0.35},
}

# v0.3.0: the conditional axis — the only metric that is regime-conditional rather
# than pooled. value = within-regime energy distance (stress-weighted) + 0.5*mass
# error. Thresholds calibrated against the model corpus (see FINVAL_V2_DECISION.md).
CONDITIONAL_THRESHOLDS: dict[str, dict[str, float]] = {
    # Calibrated on the model corpus: real-vs-real = 0; FLOW/GARCH/DCC ~0.23-0.30
    # ("good" — they get within-regime shape right but under-produce stress ~10x, so
    # not "excellent"); regime-collapsing deep-gen models (zero high-vol paths) ~1-3.5
    # ("poor"). No current model earns "excellent" — that needs the stress FREQUENCY right.
    "regime_conditional": {"excellent": 0.20, "good": 0.45, "acceptable": 0.75},
}

# v0.3.0: memorization / data-copying. value = max(0, 1 - synth/real NN-distance
# ratio); 0 = generalizing (synth NN distances ~ real NN distances), → 1 = copying.
MEMORIZATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "memorization": {"excellent": 0.10, "good": 0.25, "acceptable": 0.50},
}

# v0.3.0: lower-vs-upper tail-dependence ASYMMETRY bias at q=0.10. value =
# |mean(A_syn) - mean(A_real)| where A = lambda_L - lambda_U (positive in real
# equities: crashes cluster more than rallies). Scored on the panel-MEAN of A,
# not per-pair, because per-pair lambda noise swamps the signal at realistic n;
# the mean cancels that noise and isolates the systematic, directional defect.
# Calibrated on the 50-name reference panel (1253 daily obs): honest real-vs-real
# resamples floor at median ~0.022 / p90 ~0.044 → "excellent"/"good"; a Gaussian
# (or any elliptical/Student-t) fit to the real covariance drives mean(A)->~0 so
# its bias ≈ the full real asymmetry (~0.048) → "acceptable"; a model that INVERTS
# the asymmetry (mean A_syn < 0 while real > 0) scores ~0.10+ → "poor". Units are
# raw lambda-coefficient differences (probabilities), panel-agnostic, set off the
# real-vs-real noise floor — NOT off any model. Tighten via `thresholds` for
# larger panels/samples (the floor shrinks with sqrt(n_pairs * n_obs)).
TAIL_ASYMMETRY_THRESHOLDS: dict[str, dict[str, float]] = {
    "tail_dependence_asymmetry": {"excellent": 0.025, "good": 0.045, "acceptable": 0.07},
}

# v0.3.0: covariance dispersion calibration. value = mean of |log(ratio)| over the
# correlation-spread ratio and the (log-)variance-spread ratio of synth vs real.
# 0 = spreads match; |log| grows symmetrically whether the model under- or
# over-disperses. A perfectly calibrated resample sits at sampling noise (~0.05);
# the measured FLOW defect (corr under-dispersed ~28% -> |log .72|=.33, var
# over-dispersed ~42% -> |log 1.42|=.35) averages ~0.34 -> "poor". An elliptical
# fit to the right covariance matches dispersion well (it reproduces second
# moments), so this metric is genuinely orthogonal to tail_dependence_asymmetry
# and does NOT single out FLOW — it just measures whichever side gets spread wrong.
COV_CALIBRATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "covariance_calibration": {"excellent": 0.10, "good": 0.20, "acceptable": 0.30},
}

# ---------------------------------------------------------------------------
# CONSOLIDATED DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    **DISTRIBUTION_THRESHOLDS,
    **DEPENDENCE_THRESHOLDS,
    **TEMPORAL_THRESHOLDS,
    **CALIBRATION_THRESHOLDS,
    **PATH_THRESHOLDS,
    **CONDITIONAL_THRESHOLDS,
    **MEMORIZATION_THRESHOLDS,
    **TAIL_ASYMMETRY_THRESHOLDS,
    **COV_CALIBRATION_THRESHOLDS,
}


def quality_from_value(
    value: float,
    thresholds: dict[str, float],
) -> tuple[str, bool]:
    """Assign a quality grade and pass flag given a metric value and thresholds.

    All metrics are "lower is better". Returns ("poor", False) for NaN/Inf.

    Args:
        value: Metric value (lower is better).
        thresholds: Dict with keys "excellent", "good", "acceptable".

    Returns:
        (quality_grade, passed) where passed is True iff quality != "poor".
    """
    import math

    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return ("poor", False)

    excellent = thresholds.get("excellent", 0.10)
    good = thresholds.get("good", 0.20)
    acceptable = thresholds.get("acceptable", 0.30)

    if value < excellent:
        return ("excellent", True)
    if value < good:
        return ("good", True)
    if value < acceptable:
        return ("acceptable", True)
    return ("poor", False)


def get_thresholds(metric_name: str) -> dict[str, float]:
    """Get default thresholds for a metric by name."""
    if metric_name not in DEFAULT_THRESHOLDS:
        raise KeyError(f"No default thresholds for metric {metric_name!r}")
    return dict(DEFAULT_THRESHOLDS[metric_name])


# ---------------------------------------------------------------------------
# CATEGORY AND WEIGHTING DEFAULTS
# ---------------------------------------------------------------------------

# Which category each metric belongs to
METRIC_CATEGORY: dict[str, str] = {
    # Distribution
    "marginal_ks": "distribution",
    "energy_distance": "distribution",
    "tail_quantiles": "distribution",
    "tail_heaviness": "distribution",
    # Dependence
    "pearson_corr": "dependence",
    "spearman_corr": "dependence",
    "copula_distance": "dependence",
    "tail_dependence_upper": "dependence",
    "tail_dependence_lower": "dependence",
    "correlation_breakdown": "dependence",
    # v0.3.0 — dependence-axis additions
    "tail_dependence_asymmetry": "dependence",
    "covariance_calibration": "dependence",
    # Temporal
    "acf_returns": "temporal",
    "volatility_clustering": "temporal",
    "leverage_effect": "temporal",
    "cross_correlation": "temporal",
    # Calibration
    "coverage_50": "calibration",
    "coverage_90": "calibration",
    "coverage_95": "calibration",
    "pit_uniformity": "calibration",
    "crps": "calibration",
    # Path-level
    "drawdown_distribution": "path",
    # Conditional (regime-conditional, not pooled) — v0.3.0
    "regime_conditional": "conditional",
    # Memorization / generalization — v0.3.0
    "memorization": "memorization",
}

# Relative importance of each category within the overall score
# v0.3.0: distribution lifted 0.15->0.20 so extreme-quantile fidelity (the
# failure a derivatives book prices on, and which de-quantization now makes
# legible) carries real weight; temporal/calibration trimmed to compensate.
# v0.3.0: + the `conditional` axis (regime-conditional fidelity — the measured
# option-pricing gap). Its 0.12 is pulled from calibration/temporal/path; the
# distribution lift and dependence (tail bump) are preserved.
CATEGORY_WEIGHTS: dict[str, float] = {
    "distribution": 0.20,
    "dependence": 0.25,
    "temporal": 0.15,
    "calibration": 0.15,
    "path": 0.08,
    "conditional": 0.12,
    "memorization": 0.05,
}

# Metric weight within its category (sums to ~1 per category)
METRIC_WEIGHTS_IN_CATEGORY: dict[str, float] = {
    # Distribution (20% total) — v0.3.0: tail_quantiles 0.25->0.40 (extreme
    # quantile fidelity is the hard part everyone fails and what options price on)
    "marginal_ks": 0.35,
    "energy_distance": 0.25,
    "tail_quantiles": 0.40,
    # Dependence (25% total) — v0.3.0: both tail_dependence (crash co-movement) bumped.
    # v0.3.0: + tail_dependence_asymmetry (the elliptical blind spot) and
    # covariance_calibration (dispersion of the cov matrix); existing six trimmed
    # proportionally so the category still sums to ~1.
    "pearson_corr": 0.16,
    "spearman_corr": 0.10,
    "copula_distance": 0.19,
    "tail_dependence_upper": 0.10,
    "tail_dependence_lower": 0.18,  # crash co-movement — critical for risk
    "correlation_breakdown": 0.08,
    "tail_dependence_asymmetry": 0.10,  # crash-vs-rally asymmetry elliptical models miss
    "covariance_calibration": 0.09,  # var/corr dispersion calibration
    # Temporal (20% total)
    "acf_returns": 0.35,
    "volatility_clustering": 0.35,
    "leverage_effect": 0.20,
    "cross_correlation": 0.10,
    # Calibration (30% total)
    "coverage_50": 0.10,
    "coverage_90": 0.30,
    "coverage_95": 0.10,
    "pit_uniformity": 0.25,
    "crps": 0.25,
    # Path (8% total)
    "drawdown_distribution": 1.0,
    # Conditional (12% total) — v0.3.0
    "regime_conditional": 1.0,
    # Memorization (5% total) — v0.3.0
    "memorization": 1.0,
}


def default_absolute_weights() -> dict[str, float]:
    """Return the default absolute weights (category x within-category)."""
    out: dict[str, float] = {}
    for metric, w_in_cat in METRIC_WEIGHTS_IN_CATEGORY.items():
        cat = METRIC_CATEGORY[metric]
        out[metric] = round(CATEGORY_WEIGHTS[cat] * w_in_cat, 4)
    return out
