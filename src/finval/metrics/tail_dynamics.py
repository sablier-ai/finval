"""Phase-2 tail & dynamics localizers (v0.4.0).

Six diagnostics that deepen the marginal / temporal / dependence lenses where the headline
metrics only see a shadow — especially the chronic tail under-dispersion of many generators:

  marginal (flat):
    far_tail_quantiles  — quantile error BEYOND the 1/99 that `tail_quantiles` covers (0.5/99.5).
    marginal_skew       — directional asymmetry of the marginal (equity returns are left-skewed).
  dependence (flat):
    coskewness          — multivariate 3rd co-moment E[z_i^2 z_j] (asymmetric comovement the
                          pairwise copula under-resolves).
  temporal (paths):
    variance_term_structure — does multi-day variance scale like reality (random-walk vs not)?
    extreme_clustering      — do extreme moves cluster in time (crashes come in bursts)?
    long_memory             — long-lag ACF of |returns| (volatility long memory).

All are **localizers**: computed + reported, weight 0 in the scored aggregate (they tell you
*where* a lens broke). All "lower is better". Pure numpy. Model-agnostic.
"""

from __future__ import annotations

import logging

import numpy as np

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import quality_from_value

logger = logging.getLogger(__name__)

# Calibrated against the real-vs-real sampling-noise floor (random-split of an equity-macro
# panel, 2026-06-22; excellent ~ floor, acceptable ~ partway to the Gaussian baseline). The
# PATH-metric floors (var-term/extreme/long-memory) are from overlapping windows and are
# approximate — refine against a many-independent-path corpus. See tools/calibrate.py.
FAR_TAIL_THRESHOLDS = {"excellent": 0.40, "good": 0.60, "acceptable": 0.85}   # floor ~0.39
SKEW_THRESHOLDS = {"excellent": 0.25, "good": 0.55, "acceptable": 1.00}        # floor ~0.22
COSKEW_THRESHOLDS = {"excellent": 0.20, "good": 0.30, "acceptable": 0.45}      # floor ~0.19
VAR_TERM_THRESHOLDS = {"excellent": 0.25, "good": 0.40, "acceptable": 0.60}    # floor ~0.23 (approx)
EXTREME_CLUSTER_THRESHOLDS = {"excellent": 0.08, "good": 0.15, "acceptable": 0.25}  # floor ~0.08 (approx)
LONG_MEMORY_THRESHOLDS = {"excellent": 0.05, "good": 0.10, "acceptable": 0.20}      # floor ~0.04 (approx)


def _clean2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[~np.any(np.isnan(X), axis=1)]


def _names(n: int, feature_names: list[str] | None) -> list[str]:
    return feature_names or [f"feature_{i}" for i in range(n)]


def _pooled_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _result(name, value, thresholds, category, interp, **meta) -> MetricResult:
    quality, passed = quality_from_value(value, thresholds)
    return MetricResult(name=name, value=value, quality=quality, passed=passed,
                        thresholds=thresholds, category=category, interpretation=interp,
                        metadata=meta)


# ---- marginal (flat) ----------------------------------------------------------------------
def compute_far_tail_quantiles(synthetic, real, feature_names=None,
                               quantiles=(0.005, 0.995), thresholds=None) -> MetricResult:
    """Quantile error in the FAR tail (the 0.5/99.5 pct — beyond the 1/99 `tail_quantiles`
    covers), normalized by real std. Catches far-tail collapse — the rare-crash magnitude a
    99th-pct check misses. The 0.5/99.5 pct is the robust far-tail level (the 0.1 pct is too
    noisy to calibrate at realistic sample sizes); a proper GPD/Hill EVT fit is a later refinement."""
    try:
        S, R = _clean2d(synthetic), _clean2d(real)
        if len(S) < 500 or len(R) < 500:
            return create_error_metric("far_tail_quantiles", "need >=500 clean rows/side", "distribution")
        names = _names(R.shape[1], feature_names)
        per_feature, errs = {}, []
        for j, nm in enumerate(names):
            sd = float(np.std(R[:, j])) + 1e-12
            e = float(np.mean([abs(np.quantile(S[:, j], q) - np.quantile(R[:, j], q)) / sd for q in quantiles]))
            per_feature[nm] = e
            errs.append(e)
        v = float(np.mean(errs))
        m = _result("far_tail_quantiles", v, thresholds or FAR_TAIL_THRESHOLDS, "distribution",
                    f"far-tail (q={quantiles}) quantile error {v:.3f} (norm by real std)")
        m.per_feature = per_feature
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("far_tail_quantiles failed: %s", e)
        return create_error_metric("far_tail_quantiles", str(e), "distribution")


def compute_marginal_skew(synthetic, real, feature_names=None, thresholds=None) -> MetricResult:
    """|skew_syn - skew_real| per feature, averaged — directional asymmetry (left-skew of equities)."""
    try:
        S, R = _clean2d(synthetic), _clean2d(real)
        if len(S) < 30 or len(R) < 30:
            return create_error_metric("marginal_skew", "need >=30 clean rows/side", "distribution")

        def skew(X):
            mu = X.mean(0); sd = X.std(0) + 1e-12
            return (((X - mu) / sd) ** 3).mean(0)

        d = np.abs(skew(S) - skew(R))
        names = _names(R.shape[1], feature_names)
        v = float(np.mean(d))
        m = _result("marginal_skew", v, thresholds or SKEW_THRESHOLDS, "distribution",
                    f"mean |Δskew| {v:.3f}")
        m.per_feature = {nm: float(d[j]) for j, nm in enumerate(names)}
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("marginal_skew failed: %s", e)
        return create_error_metric("marginal_skew", str(e), "distribution")


# ---- dependence (flat) --------------------------------------------------------------------
def compute_coskewness(synthetic, real, feature_names=None, max_pairs=200, thresholds=None) -> MetricResult:
    """Mean |coskew_syn - coskew_real| over ordered pairs, coskew_{ij} = E[z_i^2 z_j] (z = per-feature
    standardized). Captures asymmetric comovement (i's volatility tied to j's return) the pairwise
    copula under-resolves."""
    try:
        S, R = _clean2d(synthetic), _clean2d(real)
        d = R.shape[1]
        if d < 2 or len(S) < 50 or len(R) < 50:
            return create_error_metric("coskewness", "need >=2 features and >=50 rows/side", "dependence")
        Zs = (S - S.mean(0)) / (S.std(0) + 1e-12)
        Zr = (R - R.mean(0)) / (R.std(0) + 1e-12)
        rng = np.random.default_rng(0)
        pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
        if len(pairs) > max_pairs:
            pairs = [pairs[t] for t in rng.choice(len(pairs), max_pairs, replace=False)]
        diffs = [abs(float((Zs[:, i] ** 2 * Zs[:, j]).mean() - (Zr[:, i] ** 2 * Zr[:, j]).mean()))
                 for i, j in pairs]
        v = float(np.mean(diffs))
        return _result("coskewness", v, thresholds or COSKEW_THRESHOLDS, "dependence",
                       f"mean |Δ E[z_i² z_j]| {v:.3f} over {len(pairs)} pairs", n_pairs=len(pairs))
    except Exception as e:  # noqa: BLE001
        logger.warning("coskewness failed: %s", e)
        return create_error_metric("coskewness", str(e), "dependence")


# ---- temporal (paths) ---------------------------------------------------------------------
def compute_variance_term_structure(synth_paths, real_paths, feature_names=None, thresholds=None) -> MetricResult:
    """Mean |log(var_syn(h)/var_real(h))| over horizons h and features, where var(h) is the
    cross-path variance of the h-step cumulative return. Catches wrong multi-day variance scaling
    (random-walk vs autocorrelated) that breaks long-horizon scenarios."""
    try:
        S = np.asarray(synth_paths, dtype=float)
        R = np.asarray(real_paths, dtype=float)
        if S.ndim != 3 or R.ndim != 3:
            return create_error_metric("variance_term_structure", "need 3D paths", "temporal")
        vs = np.nanvar(np.nancumsum(S, axis=1), axis=0)   # (H, D)
        vr = np.nanvar(np.nancumsum(R, axis=1), axis=0)
        v = float(np.nanmean(np.abs(np.log((vs + 1e-12) / (vr + 1e-12)))))
        return _result("variance_term_structure", v, thresholds or VAR_TERM_THRESHOLDS, "temporal",
                       f"mean |log var-ratio| across horizon {v:.3f}")
    except Exception as e:  # noqa: BLE001
        logger.warning("variance_term_structure failed: %s", e)
        return create_error_metric("variance_term_structure", str(e), "temporal")


def compute_extreme_clustering(synth_paths, real_paths, feature_names=None, q=0.95, thresholds=None) -> MetricResult:
    """|Δ| lag-1 autocorrelation of the exceedance indicator (|r| > per-feature q-quantile),
    pooled over paths. Positive autocorr = extremes cluster (crashes come in bursts)."""
    try:
        S = np.asarray(synth_paths, dtype=float)
        R = np.asarray(real_paths, dtype=float)
        if S.ndim != 3 or R.ndim != 3 or S.shape[1] < 3:
            return create_error_metric("extreme_clustering", "need 3D paths with horizon>=3", "temporal")

        def exceed_acf(P, j):
            thr = np.nanquantile(np.abs(P[:, :, j]), q)
            ind = (np.abs(P[:, :, j]) > thr).astype(float)   # (n_paths, H)
            return _pooled_corr(ind[:, :-1], ind[:, 1:])

        names = _names(R.shape[2], feature_names)
        d = np.abs(np.array([exceed_acf(S, j) - exceed_acf(R, j) for j in range(R.shape[2])]))
        v = float(np.mean(d))
        m = _result("extreme_clustering", v, thresholds or EXTREME_CLUSTER_THRESHOLDS, "temporal",
                    f"mean |Δ exceedance-acf(1)| {v:.3f} (q={q})")
        m.per_feature = {nm: float(d[j]) for j, nm in enumerate(names)}
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("extreme_clustering failed: %s", e)
        return create_error_metric("extreme_clustering", str(e), "temporal")


def compute_long_memory(synth_paths, real_paths, feature_names=None, lags=(20, 50, 100), thresholds=None) -> MetricResult:
    """|Δ| long-lag ACF of |returns| (volatility long memory), averaged over lags and features.
    Complements volatility_clustering (short lags) — real vol decays slowly (long memory)."""
    try:
        S = np.asarray(synth_paths, dtype=float)
        R = np.asarray(real_paths, dtype=float)
        if S.ndim != 3 or R.ndim != 3:
            return create_error_metric("long_memory", "need 3D paths", "temporal")
        H = R.shape[1]
        use = [lag for lag in lags if lag < H - 2]
        if not use:
            return create_error_metric("long_memory", f"horizon {H} too short for lags {lags}", "temporal")

        def abs_acf(P, j, lag):
            x = np.abs(P[:, :, j])
            return _pooled_corr(x[:, :-lag], x[:, lag:])

        errs = [abs(abs_acf(S, j, lag) - abs_acf(R, j, lag)) for j in range(R.shape[2]) for lag in use]
        v = float(np.mean(errs))
        return _result("long_memory", v, thresholds or LONG_MEMORY_THRESHOLDS, "temporal",
                       f"mean |Δ |r|-acf| at long lags {tuple(use)} = {v:.3f}")
    except Exception as e:  # noqa: BLE001
        logger.warning("long_memory failed: %s", e)
        return create_error_metric("long_memory", str(e), "temporal")
