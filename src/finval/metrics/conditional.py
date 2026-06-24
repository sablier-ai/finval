"""Regime-sensitivity diagnostics (v0.4.0) — the axis the pooled panel can't see.

The core finval metrics are *unconditional* two-sample comparisons: they ask "does the
synthetic distribution match the real distribution?" pooled across time. A generator that
is **regime-BLIND** — one that emits the same climatology distribution regardless of the
state it was conditioned on — can pass every pooled metric (right marginals, right tails,
even the right regime *mixture* via `regime_conditional`) while being useless for "what
happens next given where we are now."

These diagnostics test the two things the pooled panel misses:

  1. `conditional_sensitivity` — does the model's forecast distribution actually *move*
     across regimes, and by as much as reality does? (catches climatology directly.)
  2. `regime_stratified_calibration` — is the model calibrated *within* each regime, not
     just on average? (a calm-over-dispersed / stress-under-dispersed model averages out
     to "fine" when pooled — so you must not pool.)

These are **purely additive and NON-SCORING**: they live outside the weighted panel
(`validate`/`validate_paths`/`validate_calibration` are byte-identical to v0.3.0). They
require a `regime_labels` input — one label per observation, derived from the
*conditioning information available at forecast time* (e.g. trailing-vol tercile of the
anchor history). The caller supplies it because it depends on the conditioning history,
which finval does not see. Labeling by the realized outcome would be look-ahead.

Input shape mirrors the calibration path:
- `forecast_samples`: (n_obs, n_samples_per_obs, n_features)
- `actuals`:          (n_obs, n_features)
- `regime_labels`:    (n_obs,)  hashable label per observation
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

from finval.core.result import MetricResult, create_error_metric, create_undefined_metric
from finval.core.thresholds import quality_from_value
from finval.metrics.calibration import (
    compute_coverage,
    compute_crps,
    compute_pit_uniformity,
)

logger = logging.getLogger(__name__)

# Diagnostic thresholds defined HERE (not in core/thresholds.py) so the scored panel and
# its weights stay provably untouched. Lower is better for both, per finval convention.
SENSITIVITY_THRESHOLDS = {"excellent": 0.35, "good": 0.70, "acceptable": 1.20}  # |log ratio|
GAP_THRESHOLDS = {"excellent": 0.10, "good": 0.25, "acceptable": 0.50}          # CRPS units

_MIN_OBS_PER_REGIME = 15


def _energy(a: np.ndarray, b: np.ndarray, max_n: int = 2000) -> float:
    """Raw (unnormalized) energy distance between two point clouds; NaN if too few rows.

    E(X,Y) = 2·E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖, the within-terms using the unbiased n(n−1)
    normalization. Same estimator as `distribution.compute_energy_distance`, exposed raw
    so we can take a *ratio* of two energy distances (no scale normalization needed — the
    ratio is unitless)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.any(np.isnan(a), axis=1)][:max_n]
    b = b[~np.any(np.isnan(b), axis=1)][:max_n]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    xy = cdist(a, b, "euclidean")
    xx = cdist(a, a, "euclidean")
    yy = cdist(b, b, "euclidean")
    n, m = len(a), len(b)
    e = 2.0 * np.mean(xy) - np.sum(xx) / (n * max(n - 1, 1)) - np.sum(yy) / (m * max(m - 1, 1))
    return max(0.0, float(e))


def _group_by_regime(regime_labels: np.ndarray) -> dict:
    """{label: row-index array} for regimes with >= _MIN_OBS_PER_REGIME observations."""
    labels = np.asarray(regime_labels).ravel()
    groups: dict = {}
    for u in dict.fromkeys(labels.tolist()):  # preserve first-seen order, dedup
        idx = np.where(labels == u)[0]
        if len(idx) >= _MIN_OBS_PER_REGIME:
            groups[u] = idx
    return groups


def _mean_pairwise_between(pools: list[np.ndarray]) -> float:
    """Mean energy distance over all unordered regime pairs (NaN pairs dropped)."""
    vals = []
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            e = _energy(pools[i], pools[j])
            if np.isfinite(e):
                vals.append(e)
    return float(np.mean(vals)) if vals else float("nan")


def conditional_sensitivity(
    forecast_samples: np.ndarray,
    actuals: np.ndarray,
    regime_labels: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Does the model's conditional forecast distribution move across regimes — as much as
    reality does?

    Let D_model = mean pairwise energy distance between regimes' pooled *forecast* samples,
    and D_real = the same between regimes' realized *actuals*. A regime-sensitive, correct
    model has D_model ≈ D_real. A **climatology** model has D_model ≈ 0 (its forecast barely
    changes across regimes) while D_real > 0.

        value = |log((D_model + eps) / (D_real + eps))|     (lower is better; 0 = perfect)

    Same "ratio-of-dispersions, log, abs" idiom as `covariance_calibration`. A climatology
    generator drives D_model→0 → value→large → quality "poor": the pooled panel would pass
    it, this catches it.
    """
    try:
        fs = np.asarray(forecast_samples, dtype=float)
        act = np.asarray(actuals, dtype=float)
        if fs.ndim != 3 or act.ndim != 2:
            return create_error_metric(
                "conditional_sensitivity",
                "forecast_samples must be 3D (n_obs, n_samples, n_features), actuals 2D",
                "regime_sensitivity",
            )
        n_obs, n_samples, n_features = fs.shape
        if act.shape != (n_obs, n_features):
            return create_error_metric(
                "conditional_sensitivity",
                f"shape mismatch: forecast {fs.shape}, actuals {act.shape}",
                "regime_sensitivity",
            )

        groups = _group_by_regime(regime_labels)
        if len(groups) < 2:
            return create_undefined_metric(
                "conditional_sensitivity",
                f"need >=2 regimes with >={_MIN_OBS_PER_REGIME} obs on this data; got {len(groups)}",
                "regime_sensitivity",
            )

        real_pools = [act[idx] for idx in groups.values()]
        model_pools = [fs[idx].reshape(-1, n_features) for idx in groups.values()]

        d_real = _mean_pairwise_between(real_pools)
        d_model = _mean_pairwise_between(model_pools)
        if not np.isfinite(d_real) or not np.isfinite(d_model):
            return create_undefined_metric(
                "conditional_sensitivity",
                "could not estimate between-regime divergence on this data (pools too small)",
                "regime_sensitivity",
            )

        eps = 1e-9 + 1e-3 * d_real  # floor relative to the real regime effect
        if d_real < 1e-9:
            return create_undefined_metric(
                "conditional_sensitivity",
                "real outcomes do not separate across regimes (D_real≈0) — no regime structure "
                "in this data to condition on; metric undefined",
                "regime_sensitivity",
            )
        value = abs(float(np.log((d_model + eps) / (d_real + eps))))

        th = thresholds or SENSITIVITY_THRESHOLDS
        quality, passed = quality_from_value(value, th)
        ratio = (d_model + eps) / (d_real + eps)
        reading = (
            "climatology — forecast barely moves across regimes"
            if ratio < 0.5 else
            "over-reactive — forecast moves more than reality" if ratio > 2.0 else
            "regime-sensitive, scale ≈ reality"
        )
        return MetricResult(
            name="conditional_sensitivity",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="regime_sensitivity",
            interpretation=(
                f"|log(D_model/D_real)|={value:.3f} "
                f"(D_model={d_model:.5f}, D_real={d_real:.5f}, ratio={ratio:.2f}) — {reading}"
            ),
            metadata={
                "d_model": d_model,
                "d_real": d_real,
                "ratio": float(ratio),
                "n_regimes": len(groups),
                "regimes": [str(k) for k in groups],
            },
        )

    except Exception as e:  # noqa: BLE001
        logger.warning("conditional_sensitivity failed: %s", e)
        return create_error_metric("conditional_sensitivity", str(e), "regime_sensitivity")


def regime_stratified_calibration(
    forecast_samples: np.ndarray,
    actuals: np.ndarray,
    regime_labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, MetricResult]:
    """CRPS / PIT / coverage computed WITHIN each regime (don't pool), plus a summary
    `within_regime_calibration_gap` = worst-regime CRPS − pooled CRPS.

    Pooling hides regime-specific miscalibration: a model over-dispersed in calm and
    under-dispersed in stress can show a "fine" pooled CRPS while being wrong in both
    buckets. The gap surfaces exactly how much the pooled number is hiding. All results
    carry `@<regime>` suffixes and are pure diagnostics (never weighted into the panel).
    """
    out: dict[str, MetricResult] = {}
    try:
        fs = np.asarray(forecast_samples, dtype=float)
        act = np.asarray(actuals, dtype=float)
        groups = _group_by_regime(regime_labels)
        if len(groups) < 1:
            return {
                "within_regime_calibration_gap": create_undefined_metric(
                    "within_regime_calibration_gap",
                    f"no regime has >={_MIN_OBS_PER_REGIME} obs on this data",
                    "regime_sensitivity",
                )
            }

        pooled = compute_crps(fs, act, feature_names)
        per_regime_crps: list[float] = []
        for r, idx in groups.items():
            fs_r, act_r = fs[idx], act[idx]
            for fn, mname, kw in (
                (compute_crps, "crps", {}),
                (compute_pit_uniformity, "pit_uniformity", {}),
                (compute_coverage, "coverage_90", {"level": 0.90}),
            ):
                res = fn(fs_r, act_r, feature_names, **kw)
                res.name = f"{mname}@{r}"
                out[res.name] = res
                if mname == "crps" and np.isfinite(res.value):
                    per_regime_crps.append(float(res.value))

        if per_regime_crps and pooled.value is not None and np.isfinite(pooled.value):
            gap = max(0.0, max(per_regime_crps) - float(pooled.value))
            quality, passed = quality_from_value(gap, GAP_THRESHOLDS)
            out["within_regime_calibration_gap"] = MetricResult(
                name="within_regime_calibration_gap",
                value=gap,
                quality=quality,
                passed=passed,
                thresholds=GAP_THRESHOLDS,
                category="regime_sensitivity",
                interpretation=(
                    f"worst-regime CRPS {max(per_regime_crps):.3f} − pooled {float(pooled.value):.3f} "
                    f"= {gap:.3f} hidden by pooling across {len(groups)} regimes"
                ),
                metadata={"pooled_crps": float(pooled.value), "worst_regime_crps": max(per_regime_crps)},
            )
        return out

    except Exception as e:  # noqa: BLE001
        logger.warning("regime_stratified_calibration failed: %s", e)
        return {
            "within_regime_calibration_gap": create_error_metric(
                "within_regime_calibration_gap", str(e), "regime_sensitivity"
            )
        }


def multi_axis_conditional_sensitivity(
    forecast_samples: np.ndarray,
    actuals: np.ndarray,
    label_sets: dict[str, np.ndarray],
    feature_names: list[str] | None = None,
) -> dict[str, MetricResult]:
    """Responsiveness across MULTIPLE conditioning axes — not just vol-regime.

    A model can be regime-sensitive on one axis (e.g. vol) and pure climatology on others
    (trend, drawdown, vol-term-structure, cross-asset divergence). `conditional_sensitivity`
    on a single axis can't see that; this runs it per axis. `label_sets` maps an axis name to
    its per-observation regime labels (each PIT-derived from the conditioning at forecast
    time — the caller's responsibility). Returns `{conditional_sensitivity@<axis>: MetricResult}`.
    """
    out: dict[str, MetricResult] = {}
    for axis, labels in label_sets.items():
        res = conditional_sensitivity(forecast_samples, actuals, labels, feature_names)
        res.name = f"conditional_sensitivity@{axis}"
        out[res.name] = res
    return out
