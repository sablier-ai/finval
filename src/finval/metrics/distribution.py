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


def _hill_estimator(x: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index xi for an upper tail with k order stats.

    Hill (1975): given x_(1) >= x_(2) >= ... >= x_(n) (the upper order
    statistics of a sample assumed to have a regularly-varying right
    tail with index xi > 0), the estimator is

        xi_hat(k) = (1/k) * sum_{i=1..k} log( x_(i) / x_(k+1) )

    The estimator targets POSITIVE x with a heavy right tail. For
    financial return tails (both signs) the caller passes either the
    positive part of returns (for the upper tail) or the absolute
    value of the negative part (for the lower tail).

    Returns ``np.nan`` if there are fewer than ``k + 1`` strictly
    positive observations or if the resulting threshold is <= 0.
    """
    pos = x[x > 0]
    if pos.size < k + 1:
        return float("nan")
    # Descending sort, take top k+1
    pos_sorted = np.sort(pos)[::-1]
    threshold = pos_sorted[k]
    if threshold <= 0:
        return float("nan")
    return float(np.mean(np.log(pos_sorted[:k] / threshold)))


def compute_hill_tail_index(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    tail_fraction: float = 0.05,
    min_k: int = 25,
    sides: tuple[str, ...] = ("upper", "lower"),
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Hill tail-index error between synthetic and real returns.

    For each feature and each requested tail (``upper``, ``lower``,
    or both), compute the Hill estimator of the tail index xi on the
    top ``tail_fraction`` of observations. The metric value is the
    mean absolute error in xi across all (feature, side) pairs.

    The xi parameter captures the power-law decay of the tail: heavier
    tails have larger xi. For SPY daily returns xi is typically
    ~0.25-0.40; matching it is a much stronger constraint than matching
    kurtosis because the Hill estimator has bounded sampling variance
    (unlike the 4th moment).

    Args:
        synthetic: (n_samples_syn, n_features) array.
        real: (n_samples_real, n_features) array.
        feature_names: Optional list of feature names for breakdown.
        tail_fraction: Fraction of observations to keep in each tail
            for the Hill estimator. Default 0.05 (5% of n).
        min_k: Minimum number of order statistics required. If the
            number of strictly positive observations on a side is
            below this, that side is skipped for that feature.
        sides: Which tails to score. ``("upper",)`` or ``("lower",)``
            for asymmetric inspection; default is both.
        thresholds: Override default thresholds.

    Returns:
        ``MetricResult`` with the mean |xi_syn - xi_real| error and
        per-feature breakdown (both sides where computed).
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        if synthetic.ndim != 2 or real.ndim != 2:
            return create_error_metric(
                "hill_tail_index", "inputs must be 2D", "distribution"
            )
        if synthetic.shape[1] != real.shape[1]:
            return create_error_metric(
                "hill_tail_index",
                f"feature mismatch: synthetic has {synthetic.shape[1]}, real has {real.shape[1]}",
                "distribution",
            )

        n_features = synthetic.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        per_feature: dict[str, float] = {}
        per_feature_detail: dict[str, dict[str, float]] = {}
        all_errors: list[float] = []

        for i, name in enumerate(names[:n_features]):
            syn_col = synthetic[:, i][~np.isnan(synthetic[:, i])]
            real_col = real[:, i][~np.isnan(real[:, i])]

            if len(syn_col) < min_k * 2 or len(real_col) < min_k * 2:
                continue

            detail: dict[str, float] = {}
            errors_here: list[float] = []
            for side in sides:
                if side == "upper":
                    syn_side = syn_col
                    real_side = real_col
                elif side == "lower":
                    # Lower tail: reflect (use -x) so the Hill estimator,
                    # which targets the upper tail, sees the lower tail.
                    syn_side = -syn_col
                    real_side = -real_col
                else:
                    continue
                k_syn = max(min_k, int(tail_fraction * np.sum(syn_side > 0)))
                k_real = max(min_k, int(tail_fraction * np.sum(real_side > 0)))
                xi_syn = _hill_estimator(syn_side, k_syn)
                xi_real = _hill_estimator(real_side, k_real)
                if not (np.isfinite(xi_syn) and np.isfinite(xi_real)):
                    continue
                err = abs(xi_syn - xi_real)
                errors_here.append(err)
                detail[f"xi_{side}_synthetic"] = float(xi_syn)
                detail[f"xi_{side}_real"] = float(xi_real)
                detail[f"xi_{side}_error"] = float(err)
                detail[f"k_{side}_synthetic"] = int(k_syn)
                detail[f"k_{side}_real"] = int(k_real)

            if errors_here:
                per_feature[name] = float(np.mean(errors_here))
                per_feature_detail[name] = detail
                all_errors.extend(errors_here)

        if not all_errors:
            return create_error_metric(
                "hill_tail_index",
                "no usable feature × side combinations (need ~10x tail_fraction more obs)",
                "distribution",
            )

        mean_err = float(np.mean(all_errors))
        th = thresholds or DISTRIBUTION_THRESHOLDS["hill_tail_index"]
        quality, passed = quality_from_value(mean_err, th)

        worst = sorted(per_feature.items(), key=lambda x: x[1], reverse=True)
        worst_str = (
            f"{worst[0][0]} ({worst[0][1]:.3f})" if worst else "n/a"
        )
        return MetricResult(
            name="hill_tail_index",
            value=mean_err,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="distribution",
            interpretation=(
                f"Mean |xi_syn - xi_real| = {mean_err:.4f} across "
                f"{len(all_errors)} (feature, side) pairs; "
                f"tail_fraction={tail_fraction}, sides={sides}. Worst: {worst_str}"
            ),
            per_feature=per_feature,
            metadata={
                "tail_fraction": float(tail_fraction),
                "sides": list(sides),
                "n_pairs_scored": len(all_errors),
                "per_feature_detail": per_feature_detail,
            },
        )

    except Exception as e:
        logger.warning("hill_tail_index failed: %s", e)
        return create_error_metric("hill_tail_index", str(e), "distribution")


def compute_sliced_wasserstein(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    n_projections: int = 256,
    p: int = 2,
    max_samples: int = 5000,
    random_state: int | None = 0,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Sliced multivariate Wasserstein-p distance.

    Approximates the full multivariate Wasserstein by averaging 1-D
    Wasserstein distances along ``n_projections`` random directions on
    the unit sphere (Bonneel et al. 2015). Has the same metric
    properties as full Wasserstein (under sufficient projections) at
    a tiny fraction of the cost: an exact 1-D Wasserstein has a closed
    form via sorted samples.

    Pros over the existing ``energy_distance`` metric:
      * Wasserstein is a *proper* distance (energy is a divergence).
      * Linear in sample size per projection (no n^2 distance matrix).
      * Cleanly normalised by the mean per-feature std of the real
        data, so the value is scale-invariant.

    Args:
        synthetic: (n_samples_syn, n_features) array.
        real: (n_samples_real, n_features) array.
        feature_names: Optional names (currently unused; aggregate only).
        n_projections: Number of random directions to average over.
        p: Order of the Wasserstein distance (1 or 2; default 2).
        max_samples: Subsample both sides to this many rows (cost is
            ``O(n log n * n_projections)``).
        random_state: Seed for the projection directions (default 0
            for reproducibility).
        thresholds: Override default thresholds.

    Returns:
        ``MetricResult`` with normalized sliced Wasserstein distance.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        if synthetic.ndim != 2 or real.ndim != 2:
            return create_error_metric(
                "sliced_wasserstein", "inputs must be 2D", "distribution"
            )
        if synthetic.shape[1] != real.shape[1]:
            return create_error_metric(
                "sliced_wasserstein",
                f"feature mismatch: synthetic has {synthetic.shape[1]}, real has {real.shape[1]}",
                "distribution",
            )
        if p not in (1, 2):
            return create_error_metric(
                "sliced_wasserstein", f"p must be 1 or 2, got {p}", "distribution"
            )

        # Drop rows with any NaN, then subsample.
        synthetic = synthetic[~np.any(np.isnan(synthetic), axis=1)][:max_samples]
        real = real[~np.any(np.isnan(real), axis=1)][:max_samples]
        if len(synthetic) < 50 or len(real) < 50:
            return create_error_metric(
                "sliced_wasserstein",
                "need >= 50 clean samples per side",
                "distribution",
            )

        d = synthetic.shape[1]
        rng = np.random.default_rng(random_state)
        # Random unit vectors on S^{d-1} via normalised Gaussian.
        thetas = rng.standard_normal((n_projections, d))
        thetas /= np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12

        # Pre-sort projected samples per projection via vectorised matmul.
        syn_proj = synthetic @ thetas.T  # (n_syn, n_projections)
        real_proj = real @ thetas.T      # (n_real, n_projections)

        # 1-D Wasserstein-p between two empirical distributions reduces
        # to the L_p distance between sorted samples after resampling
        # both to a common grid. We use a common quantile grid based on
        # min(n_syn, n_real) so we don't need to interpolate.
        m = min(syn_proj.shape[0], real_proj.shape[0])
        # Quantile points: (i + 0.5) / m for i = 0..m-1 — midpoints.
        q = (np.arange(m, dtype=float) + 0.5) / m
        # Sort each column independently and compute empirical quantiles.
        syn_sorted = np.sort(syn_proj, axis=0)
        real_sorted = np.sort(real_proj, axis=0)
        # Use linear interp to resample to common length m if needed.
        if syn_sorted.shape[0] != m:
            syn_sorted = _column_interp(syn_sorted, m)
        if real_sorted.shape[0] != m:
            real_sorted = _column_interp(real_sorted, m)
        # Per-projection W_p
        if p == 1:
            w_per_proj = np.mean(np.abs(syn_sorted - real_sorted), axis=0)
        else:  # p == 2
            w_per_proj = np.sqrt(
                np.mean((syn_sorted - real_sorted) ** 2, axis=0)
            )
        raw_sw = float(np.mean(w_per_proj))

        # Normalise by mean per-feature std of real (scale-invariant).
        scale = float(np.mean(np.std(real, axis=0))) + 1e-12
        value = raw_sw / scale

        th = thresholds or DISTRIBUTION_THRESHOLDS["sliced_wasserstein"]
        quality, passed = quality_from_value(value, th)

        return MetricResult(
            name="sliced_wasserstein",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="distribution",
            interpretation=(
                f"Sliced W_{p} = {value:.4f} (raw {raw_sw:.4f}, scale {scale:.4f}); "
                f"n_projections={n_projections}, n_syn={len(synthetic)}, "
                f"n_real={len(real)}"
            ),
            metadata={
                "p": p,
                "n_projections": n_projections,
                "raw": raw_sw,
                "scale": scale,
                "n_syn": int(len(synthetic)),
                "n_real": int(len(real)),
            },
        )

    except Exception as e:
        logger.warning("sliced_wasserstein failed: %s", e)
        return create_error_metric("sliced_wasserstein", str(e), "distribution")


def _column_interp(sorted_arr: np.ndarray, m: int) -> np.ndarray:
    """Interpolate a column-sorted (n, k) array to common length m.

    Used when the synthetic and real sample sizes differ — we resample
    the sorted projections to a common grid via linear interpolation
    of the empirical quantile function.
    """
    n_in = sorted_arr.shape[0]
    if n_in == m:
        return sorted_arr
    src = (np.arange(n_in, dtype=float) + 0.5) / n_in
    tgt = (np.arange(m, dtype=float) + 0.5) / m
    out = np.empty((m, sorted_arr.shape[1]), dtype=float)
    for j in range(sorted_arr.shape[1]):
        out[:, j] = np.interp(tgt, src, sorted_arr[:, j])
    return out
