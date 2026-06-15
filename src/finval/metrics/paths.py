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
from finval.core.thresholds import (
    CONDITIONAL_THRESHOLDS,
    MEMORIZATION_THRESHOLDS,
    PATH_THRESHOLDS,
    quality_from_value,
)
from finval.metrics.distribution import compute_energy_distance

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


def compute_regime_conditional(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    n_regimes: int = 3,
    regime_weights: tuple[float, ...] = (0.25, 0.30, 0.45),
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Regime-conditional distributional fidelity (the only metric that is
    conditional rather than pooled).

    Unconditional metrics can all pass while the generator gets the
    regime-conditional distribution wrong: it can reproduce the POOLED return
    distribution yet under-produce — or under-disperse — the high-volatility
    regime. That is exactly the failure that makes a generator underprice options
    in a stress regime, and no pooled metric can see it.

    Each path is assigned a realized-volatility level (RMS return over the
    window). Tercile edges are taken from the REAL paths and applied to both
    sides, giving low/mid/high regimes. Two errors are combined (lower is better):

    - **regime MASS error** — total-variation distance between the synth and real
      regime histograms: does the generator produce the right FREQUENCY of stress
      paths? An under-dispersing model puts too few paths in the high-vol bin.
    - **within-regime SHAPE error** — the scale-normalized energy distance between
      real and synth returns INSIDE each regime, stress-weighted toward high vol.
      An empty/sparse synth regime (regime collapse) is scored at the worst band.

    ``value = within_regime_shape + 0.5 * mass_error`` — both are O(1), lower=better.

    Args:
        synthetic_paths / real_paths: (n_paths, horizon, n_features) of RETURNS.
        n_regimes: number of volatility regimes (default 3 = terciles).
        regime_weights: per-regime weights (low..high); high vol carries the most.
        thresholds: override default thresholds.
    """
    COLLAPSE_ED = 1.0  # within-regime penalty when a synth regime is empty/too sparse
    try:
        s = np.asarray(synthetic_paths, dtype=float)
        r = np.asarray(real_paths, dtype=float)
        if s.ndim != 3 or r.ndim != 3:
            return create_error_metric(
                "regime_conditional", "paths must be (n_paths, horizon, n_features)", "conditional"
            )

        def realized_vol(p):  # RMS return per path over (time, feature)
            return np.sqrt(np.nanmean(p ** 2, axis=(1, 2)))

        rv_r, rv_s = realized_vol(r), realized_vol(s)
        rv_r_clean = rv_r[np.isfinite(rv_r)]
        if len(rv_r_clean) < n_regimes * 5 or not np.isfinite(rv_s).any():
            return create_error_metric(
                "regime_conditional", "too few finite paths to form regimes", "conditional"
            )

        # Regime edges from REAL (the ground truth), applied to both sides.
        edges = np.quantile(rv_r_clean, np.linspace(0, 1, n_regimes + 1)[1:-1])
        reg_r = np.digitize(rv_r, edges)
        reg_s = np.digitize(rv_s, edges)

        f_r = np.array([np.mean(reg_r == k) for k in range(n_regimes)])
        f_s = np.array([np.mean(reg_s == k) for k in range(n_regimes)])
        mass_error = 0.5 * float(np.sum(np.abs(f_s - f_r)))  # total-variation distance

        w = np.asarray(regime_weights[:n_regimes], dtype=float)
        w = w / w.sum()
        D = s.shape[2]
        per_regime: dict[str, dict] = {}
        within = 0.0
        for k in range(n_regimes):
            real_k = r[reg_r == k].reshape(-1, D)
            syn_k = s[reg_s == k].reshape(-1, D)
            real_k = real_k[~np.any(np.isnan(real_k), axis=1)]
            syn_k = syn_k[~np.any(np.isnan(syn_k), axis=1)]
            if len(syn_k) < 50 or len(real_k) < 50:
                ed = COLLAPSE_ED  # regime collapse / too sparse to compare → worst band
            else:
                ed = float(compute_energy_distance(syn_k, real_k).value)
            per_regime[f"regime_{k}"] = {
                "energy_distance": ed,
                "frac_real": float(f_r[k]),
                "frac_synth": float(f_s[k]),
                "weight": float(w[k]),
            }
            within += w[k] * ed

        value = float(within + 0.5 * mass_error)
        th = thresholds or CONDITIONAL_THRESHOLDS["regime_conditional"]
        quality, passed = quality_from_value(value, th)

        return MetricResult(
            name="regime_conditional",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="conditional",
            interpretation=(
                f"Regime-conditional divergence {value:.4f} "
                f"(within-regime {within:.3f}, mass {mass_error:.3f}); "
                f"high-vol synth frac {f_s[-1]:.2f} vs real {f_r[-1]:.2f}"
            ),
            metadata={
                "within_regime": within,
                "mass_error": mass_error,
                "regime_edges": [float(x) for x in edges],
                "per_regime": per_regime,
            },
        )

    except Exception as e:
        logger.warning("regime_conditional failed: %s", e)
        return create_error_metric("regime_conditional", str(e), "conditional")


def compute_memorization(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    max_paths: int = 2000,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Memorization / data-copying diagnostic via nearest-neighbor distances.

    A generator that MEMORIZES its data can score perfectly on every other metric —
    copies *are* realistic — while having zero generalization value and leaking the
    training set. Nothing else in the suite distinguishes a generalizing generator
    from a memorizing one, so this is its own axis.

    Each path is standardized per-feature (by the real per-feature std) and flattened
    to a vector. We compare the nearest-neighbor distance from each SYNTH path to the
    REAL set against the leave-one-out nearest-neighbor distance WITHIN the real set.
    A healthy generator samples the same distribution without copying, so synth→real
    NN distances are comparable to real→real NN distances (ratio ≈ 1). A memorizer
    places samples on top of real points, so the ratio → 0.

    ``value = max(0, 1 - median(d_synth→real) / median(d_real→real))`` — 0 = no
    copying, → 1 = severe copying (lower is better).

    NOTE: for a true memorization test pass the data the model was TRAINED on as
    ``real``. With a held-out reference this instead flags trivial reproduction of
    that reference.
    """
    from scipy.spatial.distance import cdist

    try:
        s = np.asarray(synthetic_paths, dtype=float)
        r = np.asarray(real_paths, dtype=float)
        if s.ndim != 3 or r.ndim != 3:
            return create_error_metric("memorization", "need 3D (N,H,D) path data", "memorization")

        std = r.reshape(-1, r.shape[2]).std(axis=0) + 1e-12
        sf = (s / std).reshape(len(s), -1)[:max_paths]
        rf = (r / std).reshape(len(r), -1)[:max_paths]
        sf = sf[np.all(np.isfinite(sf), axis=1)]
        rf = rf[np.all(np.isfinite(rf), axis=1)]
        if len(sf) < 10 or len(rf) < 10:
            return create_error_metric("memorization", "need >=10 clean paths per side", "memorization")

        d_sr = cdist(sf, rf).min(axis=1)                 # each synth → nearest real
        rr = cdist(rf, rf)
        np.fill_diagonal(rr, np.inf)
        d_rr = rr.min(axis=1)                            # each real → nearest OTHER real (LOO)

        med_rr = float(np.median(d_rr))
        ratio = float(np.median(d_sr) / (med_rr + 1e-12))
        value = float(max(0.0, 1.0 - ratio))
        near_dup = float(np.mean(d_sr < np.quantile(d_rr, 0.05)))  # synth near-duplicates of a real path

        th = thresholds or MEMORIZATION_THRESHOLDS["memorization"]
        quality, passed = quality_from_value(value, th)

        return MetricResult(
            name="memorization",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="memorization",
            interpretation=(
                f"NN copying score {value:.3f} (synth/real NN ratio {ratio:.2f}, "
                f"near-duplicate frac {near_dup:.2f})"
            ),
            metadata={
                "nn_ratio": ratio,
                "near_duplicate_fraction": near_dup,
                "median_synth_to_real": float(np.median(d_sr)),
                "median_real_to_real": med_rr,
            },
        )

    except Exception as e:
        logger.warning("memorization failed: %s", e)
        return create_error_metric("memorization", str(e), "memorization")
