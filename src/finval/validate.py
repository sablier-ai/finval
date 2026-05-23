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
    compute_hill_tail_index,
    compute_marginal_ks,
    compute_sliced_wasserstein,
    compute_tail_heaviness,
    compute_tail_quantiles,
)
from finval.metrics.paths import compute_drawdown_distribution
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
    "hill_tail_index",
    "sliced_wasserstein",
    "pearson_corr",
    "spearman_corr",
    "copula_distance",
    "tail_dependence_upper",
    "tail_dependence_lower",
    "correlation_breakdown",
)

# Metrics that require path (3D) data
PATH_METRICS = (
    "acf_returns",
    "volatility_clustering",
    "leverage_effect",
    "cross_correlation",
    "drawdown_distribution",
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
    if "hill_tail_index" in include:
        results["hill_tail_index"] = compute_hill_tail_index(synthetic, real, feature_names)
    if "sliced_wasserstein" in include:
        results["sliced_wasserstein"] = compute_sliced_wasserstein(synthetic, real, feature_names)
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
