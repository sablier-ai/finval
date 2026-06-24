"""Main validation entry points.

`finval.validate(synthetic, real)` is the one-liner for 2D (flat) data.
`finval.validate_paths(synthetic_paths, real_paths)` is for 3D (path) data.

Both return a `ValidationReport` that aggregates all applicable metrics
with a weighted overall score.
"""

from __future__ import annotations

import logging

import numpy as np

from finval.core.result import MetricResult, ValidationReport
from finval.core.thresholds import (
    CATEGORY_WEIGHTS,
    METRIC_CATEGORY,
    default_absolute_weights,
)
from finval.metrics.calibration import (
    compute_coverage,
    compute_crps,
    compute_pit_uniformity,
)
from finval.metrics.dependence import (
    compute_copula_distance,
    compute_correlation_breakdown,
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_tail_dependence,
)
from finval.metrics.distribution import (
    compute_energy_distance,
    compute_marginal_ks,
    compute_tail_heaviness,
    compute_tail_quantiles,
)
from finval.metrics.joint import compute_c2st
from finval.metrics.stylized import (
    compute_aggregational_gaussianity,
    compute_conditional_heavy_tails,
    compute_hill_tail_index,
    compute_regime_persistence,
    compute_signature_distance,
    compute_time_reversal_asymmetry,
    compute_variogram_score,
)
from finval.metrics.tail_dynamics import (
    compute_coskewness,
    compute_extreme_clustering,
    compute_far_tail_quantiles,
    compute_long_memory,
    compute_marginal_skew,
    compute_variance_term_structure,
)
from finval.metrics.paths import (
    compute_covariance_calibration,
    compute_drawdown_distribution,
    compute_memorization,
    compute_regime_conditional,
    compute_tail_dependence_asymmetry,
)
from finval.metrics.temporal import (
    compute_acf_returns,
    compute_cross_correlation,
    compute_leverage_effect,
    compute_volatility_clustering,
)

logger = logging.getLogger(__name__)


# Metrics that apply to flat (2D) data: (n_samples, n_features)
FLAT_METRICS = (
    "marginal_ks",
    "energy_distance",
    "tail_quantiles",
    "pearson_corr",
    "spearman_corr",
    "copula_distance",
    "tail_dependence_upper",
    "tail_dependence_lower",
    "correlation_breakdown",
    # v0.3.0: cross-sectional dependence metrics — flat-input, also fed the
    # flattened rows by validate_paths (like the other dependence metrics).
    "tail_dependence_asymmetry",
    "covariance_calibration",
    # v0.4.0: joint-lens omnibus catch-all (classifier two-sample test).
    "c2st",
    # v0.4.0 Phase-2 localizers (marginal + dependence tail/shape).
    "far_tail_quantiles",
    "marginal_skew",
    "coskewness",
    # v0.4.0 gap-close (stylized-fact / SOTA): flat-input.
    "hill_tail_index",
    "variogram_score",
)

# Metrics that require path (3D) data
PATH_METRICS = (
    "acf_returns",
    "volatility_clustering",
    "leverage_effect",
    "cross_correlation",
    "drawdown_distribution",
    "regime_conditional",
    "memorization",
    # v0.4.0 Phase-2 localizers (temporal dynamics).
    "variance_term_structure",
    "extreme_clustering",
    "long_memory",
    # v0.4.0 gap-close (stylized-fact / SOTA): path-input.
    "time_reversal_asymmetry",
    "regime_persistence",
    "aggregational_gaussianity",
    "conditional_heavy_tails",
    "signature_distance",
)

# Metrics that require per-observation forecast samples
CALIBRATION_METRICS = (
    "pit_uniformity",
    "crps",
    "coverage_50",
    "coverage_90",
    "coverage_95",
)


def _run_flat_metrics(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None,
    include: set[str],
) -> dict[str, MetricResult]:
    """Run all 2D-input metrics and return a dict of results."""
    results: dict[str, MetricResult] = {}

    if "marginal_ks" in include:
        results["marginal_ks"] = compute_marginal_ks(synthetic, real, feature_names)
    if "energy_distance" in include:
        results["energy_distance"] = compute_energy_distance(synthetic, real, feature_names)
    if "tail_quantiles" in include:
        results["tail_quantiles"] = compute_tail_quantiles(synthetic, real, feature_names)
    if "far_tail_quantiles" in include:
        results["far_tail_quantiles"] = compute_far_tail_quantiles(synthetic, real, feature_names)
    if "marginal_skew" in include:
        results["marginal_skew"] = compute_marginal_skew(synthetic, real, feature_names)
    if "hill_tail_index" in include:
        results["hill_tail_index"] = compute_hill_tail_index(synthetic, real, feature_names)
    if synthetic.shape[1] >= 2:
        if "pearson_corr" in include:
            results["pearson_corr"] = compute_pearson_correlation(synthetic, real, feature_names)
        if "spearman_corr" in include:
            results["spearman_corr"] = compute_spearman_correlation(synthetic, real, feature_names)
        if "copula_distance" in include:
            results["copula_distance"] = compute_copula_distance(synthetic, real, feature_names)
        if "tail_dependence_upper" in include or "tail_dependence_lower" in include:
            up, lo = compute_tail_dependence(synthetic, real, feature_names)
            if "tail_dependence_upper" in include:
                results["tail_dependence_upper"] = up
            if "tail_dependence_lower" in include:
                results["tail_dependence_lower"] = lo
        if "correlation_breakdown" in include:
            results["correlation_breakdown"] = compute_correlation_breakdown(
                synthetic, real, feature_names
            )
        if "tail_dependence_asymmetry" in include:
            results["tail_dependence_asymmetry"] = compute_tail_dependence_asymmetry(
                synthetic, real, feature_names
            )
        if "covariance_calibration" in include:
            results["covariance_calibration"] = compute_covariance_calibration(
                synthetic, real, feature_names
            )
        if "coskewness" in include:
            results["coskewness"] = compute_coskewness(synthetic, real, feature_names)
        if "variogram_score" in include:
            results["variogram_score"] = compute_variogram_score(synthetic, real, feature_names)

    if "c2st" in include:
        results["c2st"] = compute_c2st(synthetic, real, feature_names)

    return results


def _run_path_metrics(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None,
    include: set[str],
) -> dict[str, MetricResult]:
    """Run all 3D-input metrics on path data."""
    results: dict[str, MetricResult] = {}

    if "acf_returns" in include:
        results["acf_returns"] = compute_acf_returns(
            synthetic_paths, real_paths, feature_names
        )
    if "volatility_clustering" in include:
        results["volatility_clustering"] = compute_volatility_clustering(
            synthetic_paths, real_paths, feature_names
        )
    if "leverage_effect" in include:
        results["leverage_effect"] = compute_leverage_effect(
            synthetic_paths, real_paths, feature_names
        )
    if synthetic_paths.shape[2] >= 2 and "cross_correlation" in include:
        results["cross_correlation"] = compute_cross_correlation(
            synthetic_paths, real_paths, feature_names
        )
    if "drawdown_distribution" in include:
        results["drawdown_distribution"] = compute_drawdown_distribution(
            synthetic_paths, real_paths, feature_names
        )
    if synthetic_paths.shape[2] >= 1 and "regime_conditional" in include:
        results["regime_conditional"] = compute_regime_conditional(
            synthetic_paths, real_paths, feature_names
        )
    if "memorization" in include:
        results["memorization"] = compute_memorization(
            synthetic_paths, real_paths, feature_names
        )
    if "variance_term_structure" in include:
        results["variance_term_structure"] = compute_variance_term_structure(
            synthetic_paths, real_paths, feature_names
        )
    if "extreme_clustering" in include:
        results["extreme_clustering"] = compute_extreme_clustering(
            synthetic_paths, real_paths, feature_names
        )
    if "long_memory" in include:
        results["long_memory"] = compute_long_memory(
            synthetic_paths, real_paths, feature_names
        )
    if "time_reversal_asymmetry" in include:
        results["time_reversal_asymmetry"] = compute_time_reversal_asymmetry(
            synthetic_paths, real_paths, feature_names
        )
    if "regime_persistence" in include:
        results["regime_persistence"] = compute_regime_persistence(
            synthetic_paths, real_paths, feature_names
        )
    if "aggregational_gaussianity" in include:
        results["aggregational_gaussianity"] = compute_aggregational_gaussianity(
            synthetic_paths, real_paths, feature_names
        )
    if "conditional_heavy_tails" in include:
        results["conditional_heavy_tails"] = compute_conditional_heavy_tails(
            synthetic_paths, real_paths, feature_names
        )
    if "signature_distance" in include:
        results["signature_distance"] = compute_signature_distance(
            synthetic_paths, real_paths, feature_names
        )

    return results


def _run_calibration_metrics(
    samples: np.ndarray,
    actuals: np.ndarray,
    feature_names: list[str] | None,
    include: set[str],
) -> dict[str, MetricResult]:
    """Run calibration metrics on forecast samples + actuals."""
    results: dict[str, MetricResult] = {}

    if "pit_uniformity" in include:
        results["pit_uniformity"] = compute_pit_uniformity(samples, actuals, feature_names)
    if "crps" in include:
        results["crps"] = compute_crps(samples, actuals, feature_names)
    for level in (0.50, 0.90, 0.95):
        metric = f"coverage_{int(level * 100)}"
        if metric in include:
            results[metric] = compute_coverage(samples, actuals, feature_names, level=level)

    return results


def _resolve_metrics(metrics: str | list[str] | None) -> set[str]:
    """Resolve a metrics selector to a concrete set of metric names."""
    if metrics is None or metrics == "all":
        return set(FLAT_METRICS) | set(PATH_METRICS) | set(CALIBRATION_METRICS) | {
            "tail_heaviness"
        }
    if isinstance(metrics, str):
        metrics = [metrics]
    return set(metrics)


def _build_report(
    results: dict[str, MetricResult],
    weights: dict[str, float] | None = None,
) -> ValidationReport:
    """Assemble a ValidationReport from a dict of MetricResults."""
    if weights is None:
        weights = default_absolute_weights()
    # Only include weights for metrics we actually computed
    used_weights = {k: v for k, v in weights.items() if k in results}
    return ValidationReport(
        metrics=results,
        weights=used_weights,
        category_weights=dict(CATEGORY_WEIGHTS),
    )


def validate(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    metrics: str | list[str] | None = "all",
    weights: dict[str, float] | None = None,
) -> ValidationReport:
    """Validate synthetic data against real data (flat, 2D).

    Runs all applicable flat-input metrics (distribution + dependence)
    and returns an aggregated ValidationReport.

    Args:
        synthetic: (n_samples_syn, n_features) synthetic returns.
        real: (n_samples_real, n_features) real returns.
        feature_names: Optional list of feature names for nicer output.
        metrics: Which metrics to run. One of:
            - "all": all applicable (default)
            - list of metric names: e.g. ["marginal_ks", "pearson_corr"]
        weights: Override default metric weights for aggregation.

    Returns:
        ValidationReport with overall_score, overall_quality, and per-metric results.

    Example:
        >>> import numpy as np
        >>> import finval
        >>> real = np.random.randn(1000, 3) * 0.01
        >>> synthetic = np.random.randn(1000, 3) * 0.01
        >>> report = finval.validate(synthetic, real)
        >>> print(report.summary())
    """
    synthetic = np.asarray(synthetic)
    real = np.asarray(real)
    if synthetic.ndim != 2 or real.ndim != 2:
        raise ValueError(
            f"validate() requires 2D arrays; got synthetic {synthetic.shape}, real {real.shape}. "
            "For path data, use validate_paths()."
        )
    if synthetic.shape[1] != real.shape[1]:
        raise ValueError(
            f"feature mismatch: synthetic has {synthetic.shape[1]}, real has {real.shape[1]}"
        )

    include = _resolve_metrics(metrics) & set(FLAT_METRICS)
    results = _run_flat_metrics(synthetic, real, feature_names, include)
    return _build_report(results, weights)


def validate_paths(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    metrics: str | list[str] | None = "all",
    weights: dict[str, float] | None = None,
    include_flat: bool = True,
) -> ValidationReport:
    """Validate path-level synthetic data against real paths.

    Runs temporal + path metrics on the 3D arrays directly, and
    optionally also runs distribution / dependence metrics on the
    flattened rows (synthetic_paths.reshape(-1, n_features)).

    Args:
        synthetic_paths: (n_paths_syn, path_length, n_features) synthetic
            paths. Can be returns or price levels — but the metrics chosen
            must match (drawdown_distribution requires levels, temporal
            metrics expect returns).
        real_paths: (n_paths_real, path_length, n_features) real paths.
        feature_names: Optional feature names.
        metrics: Which metrics to run ("all" or list).
        weights: Override default weights.
        include_flat: If True, also run flat metrics on reshaped data.

    Returns:
        ValidationReport.
    """
    synthetic_paths = np.asarray(synthetic_paths)
    real_paths = np.asarray(real_paths)
    if synthetic_paths.ndim != 3 or real_paths.ndim != 3:
        raise ValueError(
            f"validate_paths() requires 3D arrays; got synthetic {synthetic_paths.shape}, "
            f"real {real_paths.shape}. For flat data, use validate()."
        )
    if synthetic_paths.shape[2] != real_paths.shape[2]:
        raise ValueError(
            f"feature mismatch: synthetic {synthetic_paths.shape[2]}, real {real_paths.shape[2]}"
        )

    requested = _resolve_metrics(metrics)
    results: dict[str, MetricResult] = {}

    # Path-level metrics
    path_include = requested & set(PATH_METRICS)
    results.update(_run_path_metrics(synthetic_paths, real_paths, feature_names, path_include))

    # Flat metrics on reshaped data
    if include_flat:
        flat_include = requested & set(FLAT_METRICS)
        if flat_include:
            syn_flat = synthetic_paths.reshape(-1, synthetic_paths.shape[2])
            real_flat = real_paths.reshape(-1, real_paths.shape[2])
            results.update(_run_flat_metrics(syn_flat, real_flat, feature_names, flat_include))

    return _build_report(results, weights)


def validate_calibration(
    forecast_samples: np.ndarray,
    actuals: np.ndarray,
    feature_names: list[str] | None = None,
    metrics: str | list[str] | None = "all",
    weights: dict[str, float] | None = None,
) -> ValidationReport:
    """Validate probabilistic forecast calibration.

    Args:
        forecast_samples: (n_obs, n_samples_per_obs, n_features) — for
            each observation, `n_samples_per_obs` independent samples
            drawn from the model's predictive distribution.
        actuals: (n_obs, n_features) — the realized values.
        feature_names: Optional feature names.
        metrics: Which calibration metrics to run.
        weights: Override default weights.

    Returns:
        ValidationReport with calibration metrics only.
    """
    forecast_samples = np.asarray(forecast_samples)
    actuals = np.asarray(actuals)
    if forecast_samples.ndim != 3:
        raise ValueError(
            f"forecast_samples must be 3D (n_obs, n_samples, n_features), got {forecast_samples.shape}"
        )
    if actuals.ndim != 2:
        raise ValueError(f"actuals must be 2D (n_obs, n_features), got {actuals.shape}")

    include = _resolve_metrics(metrics) & set(CALIBRATION_METRICS)
    results = _run_calibration_metrics(forecast_samples, actuals, feature_names, include)
    return _build_report(results, weights)


def validate_conditional(
    forecast_samples: np.ndarray,
    actuals: np.ndarray,
    regime_labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> ValidationReport:
    """Regime-sensitivity DIAGNOSTIC (v0.4.0) — the axis the pooled panel can't see.

    The pooled metrics (``validate``/``validate_paths``) are *unconditional*: a regime-BLIND
    climatology generator that emits the same distribution regardless of its conditioning can
    pass them all. This entry point tests the two things they miss:

      - ``conditional_sensitivity``: does the forecast distribution actually *move* across
        regimes, as much as reality does? (catches climatology directly.)
      - ``regime_stratified_calibration``: is the model calibrated *within* each regime, not
        just on average? (+ a ``within_regime_calibration_gap`` summary.)

    This is **purely additive and isolated**: it lives outside the scored panel, defines its
    own weights locally, and cannot change ``validate``/``validate_paths``/
    ``validate_calibration`` outputs (those are byte-identical to v0.3.0).

    Args:
        forecast_samples: (n_obs, n_samples_per_obs, n_features) predictive samples.
        actuals: (n_obs, n_features) realized values.
        regime_labels: EITHER (n_obs,) labels for a single conditioning axis (back-compat,
            labeled "vol"), OR a dict ``{axis_name: (n_obs,) labels}`` for MULTIPLE axes
            (vol / trend / drawdown / vol-term-structure / cross-asset). Each label set is
            PIT-derived from the conditioning available *at forecast time*; labeling by the
            realized outcome would be look-ahead.
        feature_names: Optional feature names.

    Returns:
        ValidationReport whose ``overall_score`` summarizes regime-sensitivity health
        (weighted over the headline ``conditional_sensitivity`` + ``within_regime_calibration_gap``).
        With multiple axes, the headline = the WORST (least-responsive) axis (climatology on
        ANY axis is a flag), with per-axis ``conditional_sensitivity@<axis>`` carried alongside;
        per-regime ``crps@<r>`` etc. (on the primary axis) are diagnostics (weight 0).
    """
    import dataclasses

    from finval.metrics.conditional import (
        multi_axis_conditional_sensitivity,
        regime_stratified_calibration,
    )

    forecast_samples = np.asarray(forecast_samples)
    actuals = np.asarray(actuals)
    if forecast_samples.ndim != 3:
        raise ValueError(
            f"forecast_samples must be 3D (n_obs, n_samples, n_features), got {forecast_samples.shape}"
        )
    if actuals.ndim != 2:
        raise ValueError(f"actuals must be 2D (n_obs, n_features), got {actuals.shape}")

    # Single-axis array (back-compat, labeled "vol") or multi-axis dict.
    if isinstance(regime_labels, dict):
        if not regime_labels:
            raise ValueError("regime_labels dict is empty")
        label_sets = {str(k): np.asarray(v) for k, v in regime_labels.items()}
        primary = "vol" if "vol" in label_sets else next(iter(label_sets))
    else:
        label_sets = {"vol": np.asarray(regime_labels)}
        primary = "vol"
    for ax, lab in label_sets.items():
        if lab.ravel().shape[0] != actuals.shape[0]:
            raise ValueError(f"regime_labels['{ax}'] must have one label per observation (n_obs)")

    per_axis = multi_axis_conditional_sensitivity(forecast_samples, actuals, label_sets, feature_names)
    results: dict[str, MetricResult] = dict(per_axis)
    # Headline = the worst (highest-value = least-responsive) finite axis.
    finite = [m for m in per_axis.values() if np.isfinite(m.value)]
    if finite:
        worst = max(finite, key=lambda m: m.value)
        results["conditional_sensitivity"] = dataclasses.replace(
            worst,
            name="conditional_sensitivity",
            interpretation=f"worst-axis [{worst.name.split('@')[-1]}] — {worst.interpretation}",
        )
    results.update(
        regime_stratified_calibration(forecast_samples, actuals, label_sets[primary], feature_names)
    )
    # Self-contained weights for THIS report only (the headline + the calibration gap).
    local_weights = {"conditional_sensitivity": 0.6, "within_regime_calibration_gap": 0.4}
    used = {k: v for k, v in local_weights.items() if k in results}
    return ValidationReport(metrics=results, weights=used, category_weights={"regime_sensitivity": 1.0})


def _generate_baseline(name: str, real: np.ndarray, synthetic: np.ndarray, seed: int) -> np.ndarray:
    """Generate a baseline replay of `real` matched to `synthetic`'s shape. 3D → block /
    Gaussian paths; 2D → i.i.d. / Gaussian rows ('block_bootstrap' degrades to i.i.d. for
    flat input, since blocks need a time axis)."""
    from finval.baselines.gaussian import gaussian_baseline
    from finval.baselines.historical import block_bootstrap, historical_bootstrap

    real2d = real.reshape(-1, real.shape[-1]) if real.ndim == 3 else real
    if synthetic.ndim == 3:
        n_paths, H = synthetic.shape[0], synthetic.shape[1]
        if name == "block_bootstrap":
            return block_bootstrap(real2d, n_paths=n_paths, path_length=H, seed=seed)
        if name == "gaussian":
            return gaussian_baseline(real2d, n_paths=n_paths, path_length=H, seed=seed)
    else:
        n = len(synthetic)
        if name in ("block_bootstrap", "historical_bootstrap"):
            return historical_bootstrap(real2d, n_samples=n, seed=seed)
        if name == "gaussian":
            return gaussian_baseline(real2d, n_samples=n, seed=seed)
    raise ValueError(f"unknown baseline {name!r} for ndim={synthetic.ndim}")


def validate_against_baselines(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    baselines: tuple[str, ...] = ("block_bootstrap", "gaussian"),
    metrics: str | list[str] | None = "all",
    seed: int = 42,
) -> dict:
    """Model-vs-baseline deltas: run the pooled panel on the model under test AND on dumb
    replay baselines, and report the per-metric **deltas** — the number that guides research.

    A metric only matters where the model beats a baseline a replay machine already maxes.
    For each baseline, ``delta = model_value - baseline_value`` (all metrics lower-is-better,
    so **delta < 0 = model beats the baseline**). Covers the pooled lenses
    (distribution / dependence / temporal / joint); the generative lens self-contextualizes
    against bootstrap already, and the conditional lens needs regime labels (use
    ``validate_conditional``). DCC / GARCH-copula can be added as baseline generators later.

    Returns ``{model, baselines: {name: report}, deltas: {name: {metric: delta}},
    summary: {name: {n_better, n_worse, mean_delta}}}``.
    """
    synthetic = np.asarray(synthetic)
    real = np.asarray(real)
    runner = validate_paths if synthetic.ndim == 3 else validate
    model_rep = runner(synthetic, real, feature_names=feature_names, metrics=metrics)

    out: dict = {"model": model_rep, "baselines": {}, "deltas": {}, "summary": {}}
    for name in baselines:
        b_synth = _generate_baseline(name, real, synthetic, seed)
        b_rep = runner(b_synth, real, feature_names=feature_names, metrics=metrics)
        out["baselines"][name] = b_rep
        deltas = {
            m: float(model_rep.metrics[m].value - b_rep.metrics[m].value)
            for m in model_rep.metrics
            if m in b_rep.metrics
            and np.isfinite(model_rep.metrics[m].value)
            and np.isfinite(b_rep.metrics[m].value)
        }
        out["deltas"][name] = deltas
        vals = list(deltas.values())
        out["summary"][name] = {
            "n_better": int(sum(d < 0 for d in vals)),   # model beats baseline (lower is better)
            "n_worse": int(sum(d > 0 for d in vals)),
            "mean_delta": float(np.mean(vals)) if vals else float("nan"),
        }
    return out


def validate_full(
    synthetic: np.ndarray | None = None,
    real: np.ndarray | None = None,
    *,
    forecast_samples: np.ndarray | None = None,
    actuals: np.ndarray | None = None,
    regime_labels: np.ndarray | dict | None = None,
    feature_names: list[str] | None = None,
    generative: bool = True,
    baseline_block: int = 20,
    seed: int = 42,
):
    """Comprehensive multi-lens scorer (v0.4.0) — runs every entry point applicable to the
    inputs given and assembles a per-lens vector + overall + hard gates.

      - ``(synthetic, real)``            → pooled panel (marginal/dependence/temporal/joint) + generative
      - ``(forecast_samples, actuals)``  → calibration (folded into the conditional lens)
      - ``(+ regime_labels)``            → conditional responsiveness + regime-stratified calibration

    Lenses with no inputs are omitted and the overall renormalizes over the lenses present
    (partial-input runs are honest). The returned ``FullReport.overall_score`` is intentionally
    a *different, more complete* number than a pooled ``ValidationReport.overall_score``; any
    "poor" hard gate sets ``gated`` and forces ``overall_quality`` to "poor".

    Model-agnostic: it only ever sees arrays.
    """
    from finval.core.result import FullReport
    from finval.core.thresholds import HARD_GATES, LENS_SCORED, LENS_WEIGHTS
    from finval.metrics.generative import validate_generative

    metrics: dict[str, MetricResult] = {}

    if synthetic is not None and real is not None:
        syn = np.asarray(synthetic)
        rl = np.asarray(real)
        pooled = (validate_paths if syn.ndim == 3 else validate)(syn, rl, feature_names=feature_names)
        metrics.update(pooled.metrics)
        if generative:
            metrics.update(
                validate_generative(
                    syn, rl, baseline_block=baseline_block, feature_names=feature_names, seed=seed
                ).metrics
            )

    if forecast_samples is not None and actuals is not None:
        if regime_labels is not None:
            metrics.update(
                validate_conditional(forecast_samples, actuals, regime_labels, feature_names).metrics
            )
        metrics.update(
            validate_calibration(forecast_samples, actuals, feature_names=feature_names).metrics
        )

    # Per-lens sub-scores: weighted mean of the present, finite, scored metrics in each lens.
    lens_num: dict[str, float] = {}
    lens_den: dict[str, float] = {}
    for name, (lens, w) in LENS_SCORED.items():
        m = metrics.get(name)
        if m is not None and m.value is not None and np.isfinite(m.value):
            lens_num[lens] = lens_num.get(lens, 0.0) + m.score * w
            lens_den[lens] = lens_den.get(lens, 0.0) + w
    per_lens = {lens: lens_num[lens] / lens_den[lens] for lens in lens_den if lens_den[lens] > 0}

    # Overall = lens-weighted, renormalized over the lenses actually present.
    lw_used = {lens: LENS_WEIGHTS[lens] for lens in per_lens}
    tot = sum(lw_used.values())
    overall = sum(lw_used[lens] * per_lens[lens] for lens in per_lens) / tot if tot > 0 else 0.0

    # A hard gate fires only on a MEASURED poor — never on an inapplicable/errored metric
    # (non-finite value), so a model is never gated for a property unmeasurable on this data.
    failing = [
        g for g in HARD_GATES
        if g in metrics and metrics[g].quality == "poor"
        and metrics[g].applicable and np.isfinite(metrics[g].value)
    ]
    return FullReport(
        metrics=metrics, per_lens=per_lens, lens_weights_used=lw_used,
        overall_score=float(overall), failing_gates=failing,
    )
