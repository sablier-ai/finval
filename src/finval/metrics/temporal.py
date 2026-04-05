"""Temporal metrics: autocorrelation and volatility dynamics.

Temporal metrics operate on path-level arrays of shape
(n_paths, path_length, n_features), computing autocorrelation-based
statistics that capture financial stylized facts:

- ACF of returns: should be near zero (efficient markets)
- Volatility clustering: ACF of squared returns should be positive
- Leverage effect: corr(r_t, |r_{t+k}|) is negative for equities
- Cross-correlation: contemporaneous cross-asset dependence

All temporal metrics require sufficient path length (>= max_lag + a few).
"""

from __future__ import annotations

import logging

import numpy as np

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import TEMPORAL_THRESHOLDS, quality_from_value

logger = logging.getLogger(__name__)


def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function up to max_lag for a 1D series."""
    n = len(x)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom < 1e-12:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        num = np.dot(x[: n - k], x[k:])
        acf[k] = num / denom
    return acf


def _check_path_shape(paths: np.ndarray, name: str) -> str | None:
    """Return an error string if path array shape is wrong, else None."""
    if paths.ndim != 3:
        return f"{name} must have shape (n_paths, horizon, n_features), got {paths.shape}"
    return None


def compute_acf_returns(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    lags: tuple[int, ...] = (1, 5, 10, 20),
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """ACF error at specified lags, averaged across paths and features.

    Real financial returns have near-zero ACF (weak market efficiency).
    The model should reproduce this. Large positive ACF in synthetic
    indicates trend artifacts; large negative indicates mean-reversion
    that doesn't exist in reality.

    Args:
        synthetic_paths: (n_paths, horizon, n_features) array of returns.
        real_paths: (n_paths, horizon, n_features) array of returns.
        feature_names: Optional feature names.
        lags: Lags at which to compare ACF.
        thresholds: Override default thresholds.
    """
    try:
        synthetic_paths = np.asarray(synthetic_paths)
        real_paths = np.asarray(real_paths)
        err = _check_path_shape(synthetic_paths, "synthetic_paths") or _check_path_shape(
            real_paths, "real_paths"
        )
        if err:
            return create_error_metric("acf_returns", err, "temporal")

        max_lag = max(lags)
        n_features = synthetic_paths.shape[2]
        horizon = synthetic_paths.shape[1]
        if horizon < max_lag + 5:
            return create_error_metric(
                "acf_returns",
                f"horizon {horizon} too short for lags up to {max_lag} (need {max_lag + 5})",
                "temporal",
            )

        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        per_feature: dict[str, float] = {}
        all_errors: list[float] = []

        for i, name in enumerate(names[:n_features]):
            # ACF for each path then average
            syn_acf = np.mean(
                [_compute_acf(synthetic_paths[p, :, i], max_lag) for p in range(len(synthetic_paths))],
                axis=0,
            )
            real_acf = np.mean(
                [_compute_acf(real_paths[p, :, i], max_lag) for p in range(len(real_paths))],
                axis=0,
            )
            errs = [abs(syn_acf[k] - real_acf[k]) for k in lags]
            err_val = float(np.mean(errs))
            per_feature[name] = err_val
            all_errors.append(err_val)

        mean_error = float(np.mean(all_errors)) if all_errors else 1.0

        th = thresholds or TEMPORAL_THRESHOLDS["acf_returns"]
        quality, passed = quality_from_value(mean_error, th)

        return MetricResult(
            name="acf_returns",
            value=mean_error,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="temporal",
            interpretation=f"Mean ACF error at lags {list(lags)}: {mean_error:.4f}",
            per_feature=per_feature,
            metadata={"lags": list(lags)},
        )

    except Exception as e:
        logger.warning("acf_returns failed: %s", e)
        return create_error_metric("acf_returns", str(e), "temporal")


def compute_volatility_clustering(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    lags: tuple[int, ...] = (1, 2, 3, 4, 5),
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """ACF of squared returns (volatility clustering) at short lags.

    Real financial returns show strong positive ACF of |r| and r^2 at
    short lags — volatility clusters. Synthetic data that looks right
    marginally but has no vol clustering is a giveaway.
    """
    try:
        synthetic_paths = np.asarray(synthetic_paths)
        real_paths = np.asarray(real_paths)
        err = _check_path_shape(synthetic_paths, "synthetic_paths") or _check_path_shape(
            real_paths, "real_paths"
        )
        if err:
            return create_error_metric("volatility_clustering", err, "temporal")

        max_lag = max(lags)
        n_features = synthetic_paths.shape[2]
        horizon = synthetic_paths.shape[1]
        if horizon < max_lag + 5:
            return create_error_metric(
                "volatility_clustering",
                f"horizon {horizon} too short for lags up to {max_lag}",
                "temporal",
            )

        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        per_feature: dict[str, float] = {}
        all_errors: list[float] = []

        for i, name in enumerate(names[:n_features]):
            syn_acf = np.mean(
                [
                    _compute_acf(synthetic_paths[p, :, i] ** 2, max_lag)
                    for p in range(len(synthetic_paths))
                ],
                axis=0,
            )
            real_acf = np.mean(
                [
                    _compute_acf(real_paths[p, :, i] ** 2, max_lag)
                    for p in range(len(real_paths))
                ],
                axis=0,
            )
            errs = [abs(syn_acf[k] - real_acf[k]) for k in lags]
            err_val = float(np.mean(errs))
            per_feature[name] = err_val
            all_errors.append(err_val)

        mean_error = float(np.mean(all_errors)) if all_errors else 1.0

        th = thresholds or TEMPORAL_THRESHOLDS["volatility_clustering"]
        quality, passed = quality_from_value(mean_error, th)

        return MetricResult(
            name="volatility_clustering",
            value=mean_error,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="temporal",
            interpretation=f"Mean ACF(r^2) error at lags {list(lags)}: {mean_error:.4f}",
            per_feature=per_feature,
            metadata={"lags": list(lags)},
        )

    except Exception as e:
        logger.warning("volatility_clustering failed: %s", e)
        return create_error_metric("volatility_clustering", str(e), "temporal")


def compute_leverage_effect(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    lags: tuple[int, ...] = (1, 2, 5, 10),
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Leverage effect: corr(r_t, |r_{t+k}|) at several lags.

    For equities this is typically negative: a negative return predicts
    higher future volatility more than a positive return of the same
    magnitude does. This is one of the most robust stylized facts and
    distinguishes realistic financial paths from naive simulations.

    Uses mean error across features (not max) because a single feature
    with atypical leverage in the validation window shouldn't tank the score.
    """
    try:
        synthetic_paths = np.asarray(synthetic_paths)
        real_paths = np.asarray(real_paths)
        err = _check_path_shape(synthetic_paths, "synthetic_paths") or _check_path_shape(
            real_paths, "real_paths"
        )
        if err:
            return create_error_metric("leverage_effect", err, "temporal")

        max_lag = max(lags)
        n_features = synthetic_paths.shape[2]
        horizon = synthetic_paths.shape[1]
        if horizon < max_lag + 10:
            return create_error_metric(
                "leverage_effect",
                f"horizon {horizon} too short for lags up to {max_lag}",
                "temporal",
            )

        def leverage_curve(paths: np.ndarray, feat_i: int) -> np.ndarray:
            """Return leverage correlation at each lag, averaged over paths."""
            out = np.zeros(max_lag + 1)
            for k in lags:
                corrs: list[float] = []
                for p in range(len(paths)):
                    r = paths[p, :, feat_i]
                    if len(r) < k + 5:
                        continue
                    r_t = r[:-k]
                    abs_r_tk = np.abs(r[k:])
                    if np.std(r_t) < 1e-10 or np.std(abs_r_tk) < 1e-10:
                        continue
                    corrs.append(float(np.corrcoef(r_t, abs_r_tk)[0, 1]))
                out[k] = float(np.mean(corrs)) if corrs else 0.0
            return out

        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        per_feature: dict[str, float] = {}
        all_errors: list[float] = []

        for i, name in enumerate(names[:n_features]):
            syn_lev = leverage_curve(synthetic_paths, i)
            real_lev = leverage_curve(real_paths, i)
            errs = [abs(syn_lev[k] - real_lev[k]) for k in lags]
            err_val = float(np.mean(errs))
            per_feature[name] = err_val
            all_errors.append(err_val)

        mean_error = float(np.mean(all_errors)) if all_errors else 1.0

        th = thresholds or TEMPORAL_THRESHOLDS["leverage_effect"]
        quality, passed = quality_from_value(mean_error, th)

        return MetricResult(
            name="leverage_effect",
            value=mean_error,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="temporal",
            interpretation=f"Mean leverage correlation error at lags {list(lags)}: {mean_error:.4f}",
            per_feature=per_feature,
            metadata={"lags": list(lags)},
        )

    except Exception as e:
        logger.warning("leverage_effect failed: %s", e)
        return create_error_metric("leverage_effect", str(e), "temporal")


def compute_cross_correlation(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Contemporaneous cross-feature correlation error (mean across pairs).

    Flattens all paths to (n_total_steps, n_features), computes the
    correlation matrix, and compares vs real. Mean error across pairs
    (max over n*(n-1)/2 pairs is noise-dominated for small OOS samples).
    """
    try:
        synthetic_paths = np.asarray(synthetic_paths)
        real_paths = np.asarray(real_paths)
        err = _check_path_shape(synthetic_paths, "synthetic_paths") or _check_path_shape(
            real_paths, "real_paths"
        )
        if err:
            return create_error_metric("cross_correlation", err, "temporal")

        n_features = synthetic_paths.shape[2]
        if n_features < 2:
            return create_error_metric("cross_correlation", "need >=2 features", "temporal")

        syn_flat = synthetic_paths.reshape(-1, n_features)
        real_flat = real_paths.reshape(-1, n_features)

        syn_flat = syn_flat[~np.any(np.isnan(syn_flat), axis=1)]
        real_flat = real_flat[~np.any(np.isnan(real_flat), axis=1)]

        if len(syn_flat) < 50 or len(real_flat) < 50:
            return create_error_metric(
                "cross_correlation", "insufficient non-NaN rows", "temporal"
            )

        syn_corr = np.corrcoef(syn_flat, rowvar=False)
        real_corr = np.corrcoef(real_flat, rowvar=False)

        mask = ~np.eye(n_features, dtype=bool)
        errors = np.abs(syn_corr - real_corr)[mask]
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        th = thresholds or TEMPORAL_THRESHOLDS["cross_correlation"]
        quality, passed = quality_from_value(mean_err, th)

        return MetricResult(
            name="cross_correlation",
            value=mean_err,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="temporal",
            interpretation=f"Mean cross-correlation error {mean_err:.4f}, max {max_err:.4f}",
            metadata={"max_error": max_err},
        )

    except Exception as e:
        logger.warning("cross_correlation failed: %s", e)
        return create_error_metric("cross_correlation", str(e), "temporal")
