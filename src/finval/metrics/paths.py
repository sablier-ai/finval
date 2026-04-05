"""Path-level metrics: properties of full simulated trajectories.

These metrics operate on price paths, i.e., cumulative sums or products
of returns. They measure realistic drawdown behavior, which is critical
for risk applications.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import PATH_THRESHOLDS, quality_from_value

logger = logging.getLogger(__name__)


def _max_drawdown(levels: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a price path."""
    running_max = np.maximum.accumulate(levels)
    drawdowns = (running_max - levels) / np.maximum(running_max, 1e-10)
    return float(np.max(drawdowns))


def compute_drawdown_distribution(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """KS test on the distribution of max drawdowns across paths.

    For each feature, computes the max drawdown of every path and
    compares the synthetic vs real distributions via 2-sample KS.
    Uses the mean KS across features.

    Input `paths` should be price levels (not returns). For return-space
    input, convert first: `levels = np.exp(np.cumsum(returns, axis=1))`.

    Args:
        synthetic_paths: (n_paths_syn, path_length, n_features) price levels.
        real_paths: (n_paths_real, path_length, n_features) price levels.
        feature_names: Optional feature names.
        thresholds: Override default thresholds.
    """
    try:
        synthetic_paths = np.asarray(synthetic_paths)
        real_paths = np.asarray(real_paths)
        if synthetic_paths.ndim != 3 or real_paths.ndim != 3:
            return create_error_metric(
                "drawdown_distribution",
                "paths must have shape (n_paths, path_length, n_features)",
                "path",
            )

        n_features = synthetic_paths.shape[2]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        per_feature_ks: dict[str, float] = {}
        all_ks: list[float] = []

        for i, name in enumerate(names):
            syn_dd = np.array(
                [_max_drawdown(synthetic_paths[p, :, i]) for p in range(len(synthetic_paths))]
            )
            real_dd = np.array(
                [_max_drawdown(real_paths[p, :, i]) for p in range(len(real_paths))]
            )
            syn_dd = syn_dd[np.isfinite(syn_dd)]
            real_dd = real_dd[np.isfinite(real_dd)]

            if len(syn_dd) < 10 or len(real_dd) < 10:
                per_feature_ks[name] = 1.0
                continue

            ks_stat, _ = stats.ks_2samp(syn_dd, real_dd)
            per_feature_ks[name] = float(ks_stat)
            all_ks.append(float(ks_stat))

        if not all_ks:
            return create_error_metric(
                "drawdown_distribution", "no valid drawdown data", "path"
            )

        mean_ks = float(np.mean(all_ks))

        th = thresholds or PATH_THRESHOLDS["drawdown_distribution"]
        quality, passed = quality_from_value(mean_ks, th)

        return MetricResult(
            name="drawdown_distribution",
            value=mean_ks,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="path",
            interpretation=f"Mean max-drawdown KS {mean_ks:.4f}",
            per_feature=per_feature_ks,
        )

    except Exception as e:
        logger.warning("drawdown_distribution failed: %s", e)
        return create_error_metric("drawdown_distribution", str(e), "path")
