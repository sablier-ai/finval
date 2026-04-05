"""Distribution metrics: marginal and joint distribution fidelity.

All metrics accept 2D arrays of shape (n_samples, n_features) and return
a MetricResult. The synthetic and real arrays do not need to have the
same number of rows, but must have the same number of features.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import DISTRIBUTION_THRESHOLDS, quality_from_value

logger = logging.getLogger(__name__)


def compute_marginal_ks(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Kolmogorov-Smirnov test on each marginal, aggregated by mean.

    For each feature independently, performs a 2-sample KS test between
    synthetic and real values. Uses the mean KS statistic across features
    (not max) because max is noise-dominated when any single feature has
    a naturally hard-to-match distribution in the validation window.

    Args:
        synthetic: (n_samples_syn, n_features) array.
        real: (n_samples_real, n_features) array.
        feature_names: Optional list of feature names for per-feature output.
        thresholds: Override default thresholds.

    Returns:
        MetricResult with mean KS statistic, per-feature breakdown, and p-values.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        if synthetic.ndim != 2 or real.ndim != 2:
            return create_error_metric("marginal_ks", "inputs must be 2D", "distribution")
        if synthetic.shape[1] != real.shape[1]:
            return create_error_metric(
                "marginal_ks",
                f"feature mismatch: synthetic has {synthetic.shape[1]}, real has {real.shape[1]}",
                "distribution",
            )

        n_features = synthetic.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        ks_stats: dict[str, float] = {}
        p_values: dict[str, float] = {}

        for i, name in enumerate(names[:n_features]):
            syn_col = synthetic[:, i]
            real_col = real[:, i]
            syn_clean = syn_col[~np.isnan(syn_col)]
            real_clean = real_col[~np.isnan(real_col)]

            if len(syn_clean) < 10 or len(real_clean) < 10:
                ks_stats[name] = 1.0
                p_values[name] = 0.0
                continue

            stat, pval = stats.ks_2samp(syn_clean, real_clean)
            ks_stats[name] = float(stat)
            p_values[name] = float(pval)

        if not ks_stats:
            return create_error_metric("marginal_ks", "no features computed", "distribution")

        mean_ks = float(np.mean(list(ks_stats.values())))
        max_ks = float(np.max(list(ks_stats.values())))

        th = thresholds or DISTRIBUTION_THRESHOLDS["marginal_ks"]
        quality, passed = quality_from_value(mean_ks, th)

        worst = sorted(ks_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        worst_name, worst_val = worst[0]

        return MetricResult(
            name="marginal_ks",
            value=mean_ks,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="distribution",
            interpretation=(
                f"Mean KS {mean_ks:.4f}, max {max_ks:.4f}. Worst: {worst_name} ({worst_val:.4f})"
            ),
            per_feature=ks_stats,
            metadata={"max_ks": max_ks, "p_values": p_values, "n_features": n_features},
        )

    except Exception as e:
        logger.warning("marginal_ks failed: %s", e)
        return create_error_metric("marginal_ks", str(e), "distribution")


def compute_energy_distance(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 5000,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Energy distance between synthetic and real distributions.

    Energy distance is a proper measure of multivariate distribution
    difference: E(X, Y) = 2 E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||].
    It is zero iff the two distributions are equal.

    Result is normalized by the mean per-feature standard deviation of
    real data so that the metric is scale-invariant.

    Args:
        synthetic: (n_samples_syn, n_features) array.
        real: (n_samples_real, n_features) array.
        feature_names: Optional feature names.
        max_samples: Subsample both sides to this many rows for O(n^2) cost.
        thresholds: Override default thresholds.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)

        # Deterministic subsample (take the first max_samples rows)
        synthetic = synthetic[:max_samples]
        real = real[:max_samples]

        # Drop rows with any NaN
        synthetic = synthetic[~np.any(np.isnan(synthetic), axis=1)]
        real = real[~np.any(np.isnan(real), axis=1)]

        if len(synthetic) < 50 or len(real) < 50:
            return create_error_metric(
                "energy_distance", "need >=50 clean samples per side", "distribution"
            )

        xy = cdist(synthetic, real, "euclidean")
        xx = cdist(synthetic, synthetic, "euclidean")
        yy = cdist(real, real, "euclidean")
        n, m = len(synthetic), len(real)

        energy = (
            2.0 * np.mean(xy)
            - np.sum(xx) / (n * max(n - 1, 1))
            - np.sum(yy) / (m * max(m - 1, 1))
        )
        energy = max(0.0, float(energy))

        scale = float(np.mean(np.std(real, axis=0))) + 1e-10
        normalized = energy / scale

        th = thresholds or DISTRIBUTION_THRESHOLDS["energy_distance"]
        quality, passed = quality_from_value(normalized, th)

        return MetricResult(
            name="energy_distance",
            value=normalized,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="distribution",
            interpretation=(
                f"Normalized energy {normalized:.4f} (raw {energy:.4f}, scale {scale:.4f})"
            ),
            metadata={"raw_energy": energy, "scale": scale, "n_syn": n, "n_real": m},
        )

    except Exception as e:
        logger.warning("energy_distance failed: %s", e)
        return create_error_metric("energy_distance", str(e), "distribution")


def compute_tail_quantiles(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99),
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Normalized error between synthetic and real tail quantiles.

    For each feature, compares the specified tail quantiles of synthetic
    vs real, normalized by the real standard deviation. Much more robust
    than kurtosis (whose SE ~ sqrt(24/n) is dominated by single outliers).

    Args:
        synthetic: (n_samples_syn, n_features) array.
        real: (n_samples_real, n_features) array.
        feature_names: Optional feature names.
        quantiles: Quantiles at which to compare (default: 1st/5th/95th/99th).
        thresholds: Override default thresholds.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        n_features = synthetic.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        per_feature: dict[str, float] = {}
        per_feature_detail: dict[str, dict] = {}

        for i, name in enumerate(names[:n_features]):
            syn_col = synthetic[:, i]
            real_col = real[:, i]
            syn_clean = syn_col[~np.isnan(syn_col)]
            real_clean = real_col[~np.isnan(real_col)]

            if len(syn_clean) < 50 or len(real_clean) < 50:
                per_feature[name] = 1.0
                continue

            real_std = max(float(np.std(real_clean)), 1e-10)
            errs: list[float] = []
            detail: dict[str, dict] = {}

            for q in quantiles:
                syn_q = float(np.percentile(syn_clean, q * 100))
                real_q = float(np.percentile(real_clean, q * 100))
                err = abs(syn_q - real_q) / real_std
                errs.append(err)
                detail[f"q{q:.2f}"] = {
                    "synthetic": syn_q,
                    "real": real_q,
                    "normalized_error": err,
                }

            per_feature[name] = float(np.mean(errs))
            per_feature_detail[name] = detail

        errors = [v for v in per_feature.values() if np.isfinite(v)]
        mean_error = float(np.mean(errors)) if errors else 1.0

        th = thresholds or DISTRIBUTION_THRESHOLDS["tail_quantiles"]
        quality, passed = quality_from_value(mean_error, th)

        worst = sorted(per_feature.items(), key=lambda x: x[1], reverse=True)[:3]

        return MetricResult(
            name="tail_quantiles",
            value=mean_error,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="distribution",
            interpretation=(
                f"Mean normalized tail quantile error {mean_error:.4f}. "
                f"Worst: {worst[0][0]} ({worst[0][1]:.4f})"
                if worst
                else f"Mean error {mean_error:.4f}"
            ),
            per_feature=per_feature,
            metadata={"quantiles": list(quantiles), "per_feature_detail": per_feature_detail},
        )

    except Exception as e:
        logger.warning("tail_quantiles failed: %s", e)
        return create_error_metric("tail_quantiles", str(e), "distribution")


def compute_tail_heaviness(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Excess kurtosis error, mean across features.

    Warning: kurtosis is the 4th central moment with standard error
    ~sqrt(24/n). It is unstable below ~1000 observations and dominated
    by single outliers. Prefer `tail_quantiles` for scored evaluation.
    Kept as a diagnostic for users who specifically want a kurtosis match.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        n_features = synthetic.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        errors: dict[str, float] = {}

        for i, name in enumerate(names[:n_features]):
            syn_clean = synthetic[:, i][~np.isnan(synthetic[:, i])]
            real_clean = real[:, i][~np.isnan(real[:, i])]

            if len(syn_clean) < 30 or len(real_clean) < 30:
                errors[name] = 10.0
                continue

            syn_kurt = stats.kurtosis(syn_clean, fisher=True)
            real_kurt = stats.kurtosis(real_clean, fisher=True)
            err = float(abs(syn_kurt - real_kurt))
            if not np.isfinite(err):
                err = 10.0
            errors[name] = err

        vals = [v for v in errors.values() if np.isfinite(v)]
        mean_err = float(np.mean(vals)) if vals else 10.0
        max_err = float(np.max(vals)) if vals else 10.0

        th = thresholds or DISTRIBUTION_THRESHOLDS["tail_heaviness"]
        quality, passed = quality_from_value(mean_err, th)

        return MetricResult(
            name="tail_heaviness",
            value=mean_err,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="distribution",
            interpretation=f"Mean kurtosis error {mean_err:.2f}, max {max_err:.2f}",
            per_feature=errors,
            metadata={
                "note": "diagnostic only — kurtosis SE ~ sqrt(24/n) is unstable below n=1000",
            },
        )

    except Exception as e:
        logger.warning("tail_heaviness failed: %s", e)
        return create_error_metric("tail_heaviness", str(e), "distribution")
