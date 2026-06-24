"""Calibration metrics: probabilistic forecast quality.

Calibration metrics evaluate the full predictive distribution rather
than point forecasts. They answer: "if the model says there's a 90%
chance the return will be in [-0.02, 0.02], does that actually happen
90% of the time?"

Input format for calibration is different from other metrics:

- `samples`: (n_observations, n_samples_per_obs, n_features) array of
    forecast samples. For each observation, the model has drawn
    n_samples_per_obs independent samples from its predictive
    distribution.
- `actuals`: (n_observations, n_features) array of actual realized values.

These metrics apply to one-step-ahead forecasts; multi-step calibration
is a harder problem and is not covered in v0.1.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import CALIBRATION_THRESHOLDS, quality_from_value

logger = logging.getLogger(__name__)


def compute_pit_uniformity(
    samples: np.ndarray,
    actuals: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Probability Integral Transform uniformity test.

    If the forecast distribution is well-calibrated, PIT(y) should be
    uniform on [0, 1], where PIT(y) = F(y) and F is the forecast CDF.
    Empirically PIT_i = (#samples <= actual_i) / n_samples_per_obs.

    Aggregates PIT values across all observations and features, then
    runs a KS test against Uniform[0, 1].
    """
    try:
        samples = np.asarray(samples)
        actuals = np.asarray(actuals)
        if samples.ndim != 3:
            return create_error_metric(
                "pit_uniformity",
                "samples must have shape (n_obs, n_samples, n_features)",
                "calibration",
            )
        if actuals.ndim != 2:
            return create_error_metric(
                "pit_uniformity",
                "actuals must have shape (n_obs, n_features)",
                "calibration",
            )

        n_obs, n_samples, n_features = samples.shape
        if actuals.shape != (n_obs, n_features):
            return create_error_metric(
                "pit_uniformity",
                f"shape mismatch: samples {samples.shape}, actuals {actuals.shape}",
                "calibration",
            )

        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        pit_values: list[float] = []
        per_feature_ks: dict[str, float] = {}

        for i, name in enumerate(names):
            feature_pits: list[float] = []
            for obs in range(n_obs):
                actual = actuals[obs, i]
                sample_col = samples[obs, :, i]
                if np.isnan(actual) or np.any(np.isnan(sample_col)):
                    continue
                pit = float(np.mean(sample_col <= actual))
                # Add tiny jitter to avoid exact 0/1 piling up
                pit = min(max(pit, 1e-6), 1 - 1e-6)
                feature_pits.append(pit)

            if feature_pits:
                ks_stat, _ = stats.kstest(feature_pits, "uniform")
                per_feature_ks[name] = float(ks_stat)
                pit_values.extend(feature_pits)

        if not pit_values:
            return create_error_metric(
                "pit_uniformity", "no valid PIT values computed", "calibration"
            )

        overall_ks, overall_p = stats.kstest(pit_values, "uniform")
        overall_ks = float(overall_ks)

        th = thresholds or CALIBRATION_THRESHOLDS["pit_uniformity"]
        quality, passed = quality_from_value(overall_ks, th)

        return MetricResult(
            name="pit_uniformity",
            value=overall_ks,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="calibration",
            interpretation=(
                f"PIT uniformity KS {overall_ks:.4f} (p={overall_p:.3f}), "
                f"n={len(pit_values)}, mean={np.mean(pit_values):.3f}, "
                f"std={np.std(pit_values):.3f}"
            ),
            per_feature=per_feature_ks,
            metadata={
                "n_pit_values": len(pit_values),
                "pit_mean": float(np.mean(pit_values)),
                "pit_std": float(np.std(pit_values)),
                "overall_p_value": float(overall_p),
            },
        )

    except Exception as e:
        logger.warning("pit_uniformity failed: %s", e)
        return create_error_metric("pit_uniformity", str(e), "calibration")


def compute_crps(
    samples: np.ndarray,
    actuals: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Continuous Ranked Probability Score, normalized by real std.

    CRPS is a proper scoring rule: it is minimized in expectation by the
    true predictive distribution. Formula:
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent samples from the forecast distribution F.

    We compute this in O(n log n) per observation via the sorted-sample
    formula, then normalize by the std of actuals for scale-invariance.

    Expected CRPS under a well-calibrated forecast where both the forecast
    distribution and the realizations are N(0, 1) is 1/sqrt(pi) ~= 0.564.
    Higher values indicate the forecast is either biased or too wide/narrow.
    """
    try:
        samples = np.asarray(samples)
        actuals = np.asarray(actuals)
        if samples.ndim != 3 or actuals.ndim != 2:
            return create_error_metric(
                "crps",
                "samples must be 3D, actuals 2D",
                "calibration",
            )

        n_obs, n_samples, n_features = samples.shape
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        per_feature_crps: dict[str, float] = {}
        all_crps: list[float] = []

        for i, name in enumerate(names):
            feature_crps: list[float] = []
            for obs in range(n_obs):
                x = samples[obs, :, i]
                y = actuals[obs, i]
                if np.isnan(y) or np.any(np.isnan(x)):
                    continue

                # O(n log n) CRPS via the "fair" (unbiased) sorted-sample formula
                # of Zamo & Naveau (2018). For X_1..X_n i.i.d. from the forecast,
                #   CRPS = E|X - y| - 0.5 * E|X - X'|
                # The unbiased estimator of E|X - X'| from n samples excludes the
                # i = j diagonal and divides by n(n-1):
                #   E|X - X'| ~ (2 / (n(n-1))) * sum_{i=1..n} (2i - n - 1) * x_(i)
                # This matches the `properscoring` package and the backend
                # implementation. The biased /n^2 version overestimates CRPS by
                # ~1/n which matters for small sample counts.
                xs = np.sort(x)
                n = len(xs)
                term1 = float(np.mean(np.abs(xs - y)))
                if n > 1:
                    weights = 2 * np.arange(1, n + 1) - n - 1
                    term2 = float(2.0 * np.sum(weights * xs) / (n * (n - 1)))
                else:
                    term2 = 0.0
                crps = max(0.0, term1 - 0.5 * term2)
                feature_crps.append(crps)

            if feature_crps:
                # Normalize by std of actuals for this feature
                feat_std = float(np.nanstd(actuals[:, i])) + 1e-10
                mean_crps = float(np.mean(feature_crps)) / feat_std
                per_feature_crps[name] = mean_crps
                all_crps.append(mean_crps)

        if not all_crps:
            return create_error_metric("crps", "no valid CRPS values", "calibration")

        overall = float(np.mean(all_crps))

        th = thresholds or CALIBRATION_THRESHOLDS["crps"]
        quality, passed = quality_from_value(overall, th)

        return MetricResult(
            name="crps",
            value=overall,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="calibration",
            interpretation=(
                f"Normalized CRPS {overall:.4f} "
                f"(well-calibrated Gaussian floor ~0.564 = 1/√π; pass band 0.58/0.65/0.80)"
            ),
            per_feature=per_feature_crps,
        )

    except Exception as e:
        logger.warning("crps failed: %s", e)
        return create_error_metric("crps", str(e), "calibration")


def compute_coverage(
    samples: np.ndarray,
    actuals: np.ndarray,
    feature_names: list[str] | None = None,
    level: float = 0.90,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Empirical coverage of an interval-level% prediction interval.

    For each observation, computes the [(1-level)/2, 1-(1-level)/2]
    quantile interval of the forecast samples and checks whether the
    actual falls inside. Returns the absolute deviation from the nominal
    coverage rate.

    Args:
        samples: (n_obs, n_samples, n_features) forecast samples.
        actuals: (n_obs, n_features) realized values.
        feature_names: Optional feature names.
        level: Nominal coverage level (0.50, 0.90, 0.95 typical).
        thresholds: Override default thresholds. Defaults to
            CALIBRATION_THRESHOLDS[f"coverage_{int(level*100)}"].
    """
    try:
        samples = np.asarray(samples)
        actuals = np.asarray(actuals)
        if samples.ndim != 3 or actuals.ndim != 2:
            return create_error_metric(
                f"coverage_{int(level*100)}",
                "samples must be 3D, actuals 2D",
                "calibration",
            )

        n_obs, n_samples, n_features = samples.shape
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        alpha = (1 - level) / 2
        lo_q = alpha * 100
        hi_q = (1 - alpha) * 100

        per_feature_coverage: dict[str, float] = {}
        all_coverage: list[float] = []
        per_feature_error: list[float] = []

        for i, name in enumerate(names):
            hits: list[int] = []
            for obs in range(n_obs):
                x = samples[obs, :, i]
                y = actuals[obs, i]
                if np.isnan(y) or np.any(np.isnan(x)):
                    continue
                lo = np.percentile(x, lo_q)
                hi = np.percentile(x, hi_q)
                hits.append(1 if (lo <= y <= hi) else 0)

            if hits:
                cov = float(np.mean(hits))
                per_feature_coverage[name] = cov
                all_coverage.append(cov)
                per_feature_error.append(abs(cov - level))

        if not all_coverage:
            return create_error_metric(
                f"coverage_{int(level*100)}", "no valid coverage obs", "calibration"
            )

        mean_cov = float(np.mean(all_coverage))
        # Score on the MEAN of per-feature absolute coverage errors — NOT
        # |mean_coverage - level|. Averaging coverage across features first lets
        # an over-covering feature (+0.1) and an under-covering feature (-0.1)
        # cancel to a perfect-looking 0 error while both are miscalibrated. The
        # mean-abs-error penalises each feature's miscalibration honestly.
        error = float(np.mean(per_feature_error))

        metric_name = f"coverage_{int(level*100)}"
        th = thresholds or CALIBRATION_THRESHOLDS[metric_name]
        quality, passed = quality_from_value(error, th)

        return MetricResult(
            name=metric_name,
            value=error,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="calibration",
            interpretation=(
                f"{int(level*100)}% interval empirical coverage: "
                f"{mean_cov:.3f} (nominal {level:.2f}, error {error:.4f})"
            ),
            per_feature=per_feature_coverage,
            metadata={"nominal": level, "empirical": mean_cov},
        )

    except Exception as e:
        logger.warning("coverage_%d failed: %s", int(level * 100), e)
        return create_error_metric(f"coverage_{int(level*100)}", str(e), "calibration")
