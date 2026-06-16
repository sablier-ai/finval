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
    COV_CALIBRATION_THRESHOLDS,
    MEMORIZATION_THRESHOLDS,
    PATH_THRESHOLDS,
    TAIL_ASYMMETRY_THRESHOLDS,
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


def _to_uniform(x: np.ndarray) -> np.ndarray:
    """Per-feature rank transform to pseudo-observations in (0, 1).

    Matches the pseudo-obs convention used by the dependence metrics
    (rankdata / (n + 1)); NaNs are ignored by ranking only finite rows
    per column, leaving NaN in place so callers can mask per-pair.
    """
    out = np.full_like(x, np.nan, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        m = np.isfinite(col)
        n_finite = int(m.sum())
        if n_finite == 0:
            continue
        out[m, j] = stats.rankdata(col[m]) / (n_finite + 1)
    return out


def _empirical_tail_coeff(u: np.ndarray, v: np.ndarray, q: float, tail: str) -> float:
    """Empirical tail-dependence coefficient at a finite quantile q.

    lower: P(V < q | U < q);  upper: P(V > 1-q | U > 1-q). Returns NaN
    when the conditioning tail is empty (so the pair can be dropped rather
    than counted as a spurious 0). Mirrors dependence._tail_coefficient but
    NaN-signals the empty-tail case instead of returning 0.
    """
    m = np.isfinite(u) & np.isfinite(v)
    u, v = u[m], v[m]
    if len(u) < 50:
        return float("nan")
    if tail == "upper":
        cond = u > (1 - q)
        joint = cond & (v > (1 - q))
    else:
        cond = u < q
        joint = cond & (v < q)
    n_cond = int(cond.sum())
    if n_cond == 0:
        return float("nan")
    return float(joint.sum()) / n_cond


def compute_tail_dependence_asymmetry(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    quantile: float = 0.10,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Lower-vs-upper tail-dependence ASYMMETRY fidelity (the elliptical blind spot).

    Real multivariate returns crash together more than they rally together:
    the lower tail-dependence coefficient lambda_L exceeds the upper lambda_U,
    so the per-pair asymmetry ``A = lambda_L - lambda_U`` is systematically
    POSITIVE. Any elliptical generator (Gaussian, Student-t, and the
    Gaussian/Ledoit-Wolf baselines) is radially symmetric and produces
    ``A = 0`` for every pair by construction — it cannot be wrong on
    ``tail_dependence_lower`` and ``tail_dependence_upper`` *individually* in a
    way the existing pooled levels reliably catch, yet it gets the crash-vs-rally
    ASYMMETRY exactly backwards (flat). This metric scores that one number
    directly, so a symmetric copula can no longer hide behind level errors that
    happen to cancel.

    For each feature pair we compute the real and synthetic asymmetries
    ``A_real`` and ``A_syn`` from rank pseudo-observations at a finite tail
    ``quantile`` (default 10% — large enough to keep the conditioning tail
    populated at realistic sample sizes so the lambda estimates aren't pure
    noise), then score the mean absolute error
    ``|A_syn - A_real|`` across pairs (lower is better, 0 = perfect). Pairs whose
    conditioning tail is empty on either side are dropped (NaN), not scored as 0,
    so the metric is honest under sparse extremes. Because A is a DIFFERENCE of
    two coefficients, it is sign-aware: a model that flips the asymmetry
    (rally-clustering instead of crash-clustering) is penalized, not rewarded.

    Agnostic: operates on rank pseudo-obs of any multivariate panel, makes no
    distributional assumption, and does not reference any particular generator.
    On 3D path input the paths are flattened to cross-sectional rows
    ``(N*H, D)``; on 2D input it is used directly.

    Args:
        synthetic_paths / real_paths: (n_paths, horizon, n_features) RETURNS,
            or (n_samples, n_features) flat returns.
        quantile: tail probability q for the conditioning event (default 0.05).
        thresholds: override default thresholds.
    """
    try:
        s = np.asarray(synthetic_paths, dtype=float)
        r = np.asarray(real_paths, dtype=float)
        if s.ndim == 3:
            s = s.reshape(-1, s.shape[2])
        if r.ndim == 3:
            r = r.reshape(-1, r.shape[2])
        if s.ndim != 2 or r.ndim != 2:
            return create_error_metric(
                "tail_dependence_asymmetry", "need 2D or 3D return data", "dependence"
            )
        n_features = s.shape[1]
        if n_features < 2:
            return create_error_metric(
                "tail_dependence_asymmetry", "need >=2 features", "dependence"
            )

        su, ru = _to_uniform(s), _to_uniform(r)

        per_pair: dict[str, dict] = {}
        errs: list[float] = []
        sum_a_real = 0.0
        sum_a_syn = 0.0
        n_valid = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                rl = _empirical_tail_coeff(ru[:, i], ru[:, j], quantile, "lower")
                rup = _empirical_tail_coeff(ru[:, i], ru[:, j], quantile, "upper")
                sl = _empirical_tail_coeff(su[:, i], su[:, j], quantile, "lower")
                sup = _empirical_tail_coeff(su[:, i], su[:, j], quantile, "upper")
                if not (np.isfinite(rl) and np.isfinite(rup) and np.isfinite(sl) and np.isfinite(sup)):
                    continue
                a_real = rl - rup
                a_syn = sl - sup
                err = abs(a_syn - a_real)
                per_pair[f"{i}<->{j}"] = {
                    "asym_real": a_real,
                    "asym_synth": a_syn,
                    "error": err,
                }
                errs.append(err)
                sum_a_real += a_real
                sum_a_syn += a_syn
                n_valid += 1

        if not errs:
            return create_error_metric(
                "tail_dependence_asymmetry", "no pair had a populated tail on both sides", "dependence"
            )

        mean_a_real = sum_a_real / n_valid
        mean_a_synth = sum_a_syn / n_valid

        # The scored statistic is the PANEL-LEVEL asymmetry bias
        #   systematic_bias = |mean(A_syn) - mean(A_real)|.
        # This is deliberate. The per-pair |A_syn - A_real| (reported as a
        # diagnostic) is dominated by finite-tail lambda noise at realistic sample
        # sizes — on the 50-name reference panel the honest real-vs-real per-pair
        # MAE floor (~0.07) is LARGER than the real asymmetry signal (~0.05), so it
        # cannot separate an elliptical model from an honest one. Averaging the
        # signed asymmetry across all pairs cancels that idiosyncratic noise and
        # leaves the systematic, directional defect: a radially-symmetric generator
        # (Gaussian/Student-t) drives mean(A_syn)->0 while real mean(A) stays
        # positive, so its bias ≈ the full real asymmetry; an honest resample has no
        # directional bias and floors near zero. The metric stays in raw
        # lambda-difference units and assumes no particular generator is correct.
        per_pair_mae = float(np.mean(errs))
        systematic_bias = float(abs(mean_a_synth - mean_a_real))
        value = systematic_bias

        th = thresholds or TAIL_ASYMMETRY_THRESHOLDS["tail_dependence_asymmetry"]
        quality, passed = quality_from_value(value, th)

        return MetricResult(
            name="tail_dependence_asymmetry",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="dependence",
            interpretation=(
                f"Tail-asymmetry (lambda_L - lambda_U) error {value:.4f} at q={quantile:.2f} "
                f"(per-pair MAE {per_pair_mae:.3f}, panel bias {systematic_bias:.3f}); "
                f"mean A real {mean_a_real:+.3f} vs synth {mean_a_synth:+.3f} "
                f"over {n_valid} pairs"
            ),
            per_pair=per_pair,
            metadata={
                "quantile": quantile,
                "per_pair_mae": per_pair_mae,
                "systematic_bias": systematic_bias,
                "mean_asymmetry_real": mean_a_real,
                "mean_asymmetry_synth": mean_a_synth,
                "n_pairs_scored": n_valid,
            },
        )

    except Exception as e:
        logger.warning("tail_dependence_asymmetry failed: %s", e)
        return create_error_metric("tail_dependence_asymmetry", str(e), "dependence")


def compute_covariance_calibration(
    synthetic_paths: np.ndarray,
    real_paths: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Variance / correlation DISPERSION calibration of the synthetic covariance.

    A generator can match the AVERAGE level of variances and correlations while
    getting their cross-sectional SPREAD wrong — under-dispersing correlations
    (too few near +/-1) and over-dispersing variances (too many extreme).
    A covariance matrix that is right on average but wrong in dispersion produces
    portfolio risk that is systematically mis-stated (min-variance hedges that
    don't hedge), and no pooled level metric (pearson_corr is a mean abs error,
    it cannot see a shrunk-vs-fanned spread that leaves the mean intact) catches
    it. This metric scores the two dispersion ratios directly.

    From each side we take the per-feature variances and the off-diagonal
    Pearson correlations, and compare the DISPERSION (standard deviation across
    features / across pairs) of synthetic vs real:

    - ``corr_dispersion_ratio  = std(corr_synth_offdiag)  / std(corr_real_offdiag)``
    - ``var_dispersion_ratio   = std(log var_synth)       / std(log var_real)``

    (variance dispersion is taken in log space so it is scale-free and symmetric
    in over/under). For each ratio the calibration error is ``|log(ratio)|`` —
    0 when the spread matches, growing symmetrically whether the model
    under-disperses (ratio<1) or over-disperses (ratio>1). The reported value is
    the mean of the two log-ratio errors (lower is better). A ratio of e.g. 0.72
    (corr under-dispersed 28%) or 1.42 (var over-dispersed 42%) maps to
    |log 0.72| = 0.33 / |log 1.42| = 0.35 — both land in the "poor" band by
    default, which is the intended sensitivity for that measured defect.

    Agnostic: pure second-moment dispersion of any multivariate panel; no
    generator-specific structure and no assumption that any particular model is
    the reference. On 3D input the paths are flattened to ``(N*H, D)`` rows.

    Args:
        synthetic_paths / real_paths: (n_paths, horizon, n_features) RETURNS,
            or (n_samples, n_features) flat returns.
        thresholds: override default thresholds.
    """
    try:
        s = np.asarray(synthetic_paths, dtype=float)
        r = np.asarray(real_paths, dtype=float)
        if s.ndim == 3:
            s = s.reshape(-1, s.shape[2])
        if r.ndim == 3:
            r = r.reshape(-1, r.shape[2])
        if s.ndim != 2 or r.ndim != 2:
            return create_error_metric(
                "covariance_calibration", "need 2D or 3D return data", "dependence"
            )
        n_features = s.shape[1]
        if n_features < 2:
            return create_error_metric(
                "covariance_calibration", "need >=2 features", "dependence"
            )

        def _moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = x[np.all(np.isfinite(x), axis=1)]
            if len(x) < 20:
                raise ValueError("need >=20 clean rows")
            var = np.var(x, axis=0)
            c = np.corrcoef(x, rowvar=False)
            offdiag = c[~np.eye(n_features, dtype=bool)]
            return var, offdiag

        var_r, corr_r = _moments(r)
        var_s, corr_s = _moments(s)

        # Correlation dispersion (spread of off-diagonal correlations).
        corr_disp_r = float(np.std(corr_r))
        corr_disp_s = float(np.std(corr_s))
        corr_ratio = corr_disp_s / (corr_disp_r + 1e-12)

        # Variance dispersion in log space (scale-free, symmetric over/under).
        eps = 1e-30
        logvar_r = np.log(var_r + eps)
        logvar_s = np.log(var_s + eps)
        var_disp_r = float(np.std(logvar_r))
        var_disp_s = float(np.std(logvar_s))
        var_ratio = var_disp_s / (var_disp_r + 1e-12)

        # If a real dispersion is ~0 (e.g. a single feature, or identical vars),
        # that axis is undefined — fall back to the other axis alone.
        corr_def = corr_disp_r > 1e-9
        var_def = var_disp_r > 1e-9
        errs = []
        if corr_def:
            errs.append(abs(float(np.log(max(corr_ratio, 1e-12)))))
        if var_def:
            errs.append(abs(float(np.log(max(var_ratio, 1e-12)))))
        if not errs:
            return create_error_metric(
                "covariance_calibration", "real dispersions degenerate (no spread to match)", "dependence"
            )
        value = float(np.mean(errs))

        th = thresholds or COV_CALIBRATION_THRESHOLDS["covariance_calibration"]
        quality, passed = quality_from_value(value, th)

        return MetricResult(
            name="covariance_calibration",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="dependence",
            interpretation=(
                f"Dispersion calibration {value:.4f}: corr-spread ratio {corr_ratio:.3f} "
                f"(synth/real), var-spread ratio {var_ratio:.3f}"
            ),
            metadata={
                "corr_dispersion_ratio": corr_ratio,
                "var_dispersion_ratio": var_ratio,
                "corr_dispersion_real": corr_disp_r,
                "corr_dispersion_synth": corr_disp_s,
                "var_dispersion_real": var_disp_r,
                "var_dispersion_synth": var_disp_s,
            },
        )

    except Exception as e:
        logger.warning("covariance_calibration failed: %s", e)
        return create_error_metric("covariance_calibration", str(e), "dependence")
