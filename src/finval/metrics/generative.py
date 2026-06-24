"""Generative-health metrics (v0.4.0) — the generator-vs-replay axis.

Distributional fidelity (distribution/dependence/temporal) asks "do the SAME statistics
match?". This lens asks the orthogonal question a statistic can't: is the generator a
HEALTHY sampler of the real manifold, or is it
  (a) collapsing to a few modes  → misses rare regimes (e.g. never a 2008/2020),
  (b) drifting OFF the real manifold → implausible samples that still match marginals, or
  (c) merely replaying training data → no value over a block-bootstrap.

We use the manifold **density & coverage** of Naeem et al. (2020), "Reliable Fidelity and
Diversity Metrics for Generative Models" — the outlier-robust successor to improved
precision/recall (Kynkaanniemi 2019). With each real point's k-NN radius inside the real
set as the local scale:

  density  = (1 / k·|S|) · #{(s, r) : ||s - r|| <= radius_k(r)}
             plausibility — ~1 when S sits in real-dense regions, <1 = off-manifold.
  coverage = (1 / |R|) · #{r : exists s with ||s - r|| <= radius_k(r)}
             recall — 1 = every real region is represented, low = mode collapse.

Reported as finval "lower is better" deficits (1 - coverage, max(0, 1 - density)) AND —
crucially — as the **delta vs a block-bootstrap of the real data** (the replay baseline a
generator must beat to justify itself). Block-bootstrap is *real data replayed*, so it is
near-unbeatable on UNCONDITIONAL coverage/density: a generator that only ties it adds no
unconditional value, and its value (if any) is conditional (see conditional_sensitivity).

3D path inputs are reduced to an interpretable PATH-FEATURE space (terminal cumulative
return, realized vol, max drawdown, return skew — per feature) before the manifold metrics,
both because raw (H x D)-dim manifolds are unreliable with few paths (curse of
dimensionality) and because "cover the space of plausible PATH SHAPES" is the
decision-relevant question (the FID-style "metrics in feature space" approach).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

from finval.baselines.historical import block_bootstrap, historical_bootstrap
from finval.core.result import MetricResult, ValidationReport, create_error_metric

logger = logging.getLogger(__name__)

# Local thresholds (lower is better) — kept here, not in core/thresholds.py, until the
# 0.4.0 corpus calibration pass sets the real-vs-real floor (same discipline as 0.3.0).
COVERAGE_DEFICIT_THRESHOLDS = {"excellent": 0.10, "good": 0.20, "acceptable": 0.35}
PLAUSIBILITY_DEFICIT_THRESHOLDS = {"excellent": 0.10, "good": 0.20, "acceptable": 0.35}

_MAX_PTS = 2000  # cap for the O(n^2) cdist


def _clean(X: np.ndarray, max_pts: int = _MAX_PTS) -> np.ndarray:
    """Drop NaN/inf rows and deterministically cap to max_pts."""
    X = np.asarray(X, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    return X[:max_pts]


def path_features(paths: np.ndarray) -> np.ndarray:
    """Reduce (n_paths, horizon, n_features) RETURN paths to an interpretable per-path
    feature vector: [terminal cum-return, realized vol, max drawdown, return skew] per
    feature → (n_paths, 4 * n_features). The space the manifold metrics operate in for
    path inputs."""
    p = np.asarray(paths, dtype=float)
    if p.ndim != 3:
        raise ValueError(f"path_features expects 3D paths, got {p.shape}")
    cum = np.nansum(p, axis=1)                                  # terminal cumulative return
    vol = np.nanstd(p, axis=1)                                  # realized vol
    cumpath = np.nancumsum(p, axis=1)
    runmax = np.maximum.accumulate(cumpath, axis=1)
    mdd = np.nanmin(cumpath - runmax, axis=1)                   # max drawdown (<= 0)
    mu = np.nanmean(p, axis=1, keepdims=True)
    sd = np.nanstd(p, axis=1, keepdims=True) + 1e-12
    skew = np.nanmean(((p - mu) / sd) ** 3, axis=1)             # return skew
    return np.concatenate([cum, vol, mdd, skew], axis=1)


def _to_points(arr: np.ndarray) -> np.ndarray:
    """2D → rows as points; 3D → per-path feature vectors as points."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return path_features(a)
    raise ValueError(f"expected 2D or 3D array, got {a.shape}")


def _knn_radius(X: np.ndarray, k: int) -> np.ndarray:
    """Distance from each row of X to its k-th nearest neighbour within X (excl. self)."""
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    kk = min(k, D.shape[1] - 1) if D.shape[1] > 1 else 0
    return D[:, kk]


def density_coverage(synth_pts: np.ndarray, real_pts: np.ndarray, k: int = 5) -> tuple[float, float]:
    """Naeem density & coverage between two point clouds (already feature-space points).

    Standardizes both clouds by the REAL per-dimension mean/std so no single feature
    dominates the Euclidean manifold. Returns (density, coverage)."""
    S, R = _clean(synth_pts), _clean(real_pts)
    if len(S) < k + 1 or len(R) < k + 1:
        raise ValueError(f"need > k={k} clean points per side; got synth={len(S)}, real={len(R)}")
    mu = R.mean(axis=0)
    sd = R.std(axis=0) + 1e-12
    S = (S - mu) / sd
    R = (R - mu) / sd
    radii = _knn_radius(R, k)                       # (|R|,)
    within = cdist(S, R) <= radii[None, :]          # (|S|, |R|): synth s inside real r's k-ball
    density = float(within.sum() / (k * len(S)))
    coverage = float(within.any(axis=0).mean())     # fraction of real with >=1 synth in-ball
    return density, coverage


def _baseline_like(real: np.ndarray, synthetic: np.ndarray, *, block: int, seed: int) -> np.ndarray:
    """A block-bootstrap (3D) / i.i.d.-bootstrap (2D) replay of `real`, matched to the
    SHAPE of `synthetic` — the honest replay reference."""
    r = np.asarray(real, dtype=float)
    s = np.asarray(synthetic, dtype=float)
    real_2d = r.reshape(-1, r.shape[-1]) if r.ndim == 3 else r
    if s.ndim == 3:
        n_paths, H = s.shape[0], s.shape[1]
        return block_bootstrap(real_2d, n_paths=n_paths, path_length=H, block_size=block, seed=seed)
    return historical_bootstrap(real_2d, n_samples=len(s), seed=seed)


def _deficit_metric(name: str, value: float, thresholds: dict, interp: str, meta: dict) -> MetricResult:
    from finval.core.thresholds import quality_from_value

    quality, passed = quality_from_value(value, thresholds)
    return MetricResult(
        name=name, value=value, quality=quality, passed=passed, thresholds=thresholds,
        category="generative", interpretation=interp, metadata=meta,
    )


def validate_generative(
    synthetic: np.ndarray,
    real: np.ndarray,
    *,
    k: int = 5,
    baseline_block: int = 20,
    feature_names: list[str] | None = None,
    seed: int = 42,
) -> ValidationReport:
    """Generative-health DIAGNOSTIC (v0.4.0): is the generator a healthy sampler of the
    real manifold, and does it beat a block-bootstrap replay?

    Scores `coverage_deficit` (1 - recall; mode collapse / missing regimes) and
    `plausibility_deficit` (max(0, 1 - density); off-manifold drift). Also computes the
    SAME for a shape-matched block-bootstrap of `real` and reports the **deltas**
    (synthetic - bootstrap): negative = the generator beats replay; ~0 = it only ties
    (no unconditional value); positive = worse than replay.

    Args:
        synthetic: (n, d) rows or (n_paths, horizon, d) return paths.
        real: same ndim as synthetic; the reference sample.
        k: k-NN neighbourhood for the manifold (Naeem default 5).
        baseline_block: block size for the bootstrap replay baseline.
        feature_names: optional (for reporting).
        seed: RNG seed for the baseline.

    Returns:
        ValidationReport (category "generative"); overall_score = weighted coverage +
        plausibility deficits; bootstrap deltas live in metadata + the report metrics.
    """
    try:
        syn = np.asarray(synthetic, dtype=float)
        rl = np.asarray(real, dtype=float)
        if syn.ndim != rl.ndim or syn.ndim not in (2, 3):
            return _err_report("synthetic and real must share ndim (2D rows or 3D paths)")
        if syn.shape[-1] != rl.shape[-1]:
            return _err_report(f"feature mismatch: synthetic {syn.shape[-1]}, real {rl.shape[-1]}")

        syn_pts, real_pts = _to_points(syn), _to_points(rl)
        density, coverage = density_coverage(syn_pts, real_pts, k=k)

        # The replay reference: block-bootstrap of real, matched to synthetic's shape.
        base = _baseline_like(rl, syn, block=baseline_block, seed=seed)
        b_density, b_coverage = density_coverage(_to_points(base), real_pts, k=k)

        cov_def = max(0.0, 1.0 - coverage)
        plaus_def = max(0.0, 1.0 - density)
        b_cov_def = max(0.0, 1.0 - b_coverage)
        b_plaus_def = max(0.0, 1.0 - b_density)
        d_cov = cov_def - b_cov_def            # <0 = better recall than replay
        d_plaus = plaus_def - b_plaus_def      # <0 = more plausible than replay

        shared_meta = {
            "density": density, "coverage": coverage,
            "bootstrap_density": b_density, "bootstrap_coverage": b_coverage,
            "coverage_deficit_vs_bootstrap": d_cov,
            "plausibility_deficit_vs_bootstrap": d_plaus,
            "k": k, "n_synth": int(len(_clean(syn_pts))), "n_real": int(len(_clean(real_pts))),
            "input": "paths" if syn.ndim == 3 else "rows",
        }
        metrics = {
            "coverage_deficit": _deficit_metric(
                "coverage_deficit", cov_def, COVERAGE_DEFICIT_THRESHOLDS,
                f"recall {coverage:.3f} (deficit {cov_def:.3f}); bootstrap {b_coverage:.3f} "
                f"→ vs-replay delta {d_cov:+.3f} ({'beats' if d_cov < -1e-3 else 'ties' if abs(d_cov) <= 1e-3 else 'worse than'} replay)",
                shared_meta,
            ),
            "plausibility_deficit": _deficit_metric(
                "plausibility_deficit", plaus_def, PLAUSIBILITY_DEFICIT_THRESHOLDS,
                f"density {density:.3f} (deficit {plaus_def:.3f}); bootstrap {b_density:.3f} "
                f"→ vs-replay delta {d_plaus:+.3f}",
                shared_meta,
            ),
        }
        weights = {"coverage_deficit": 0.5, "plausibility_deficit": 0.5}
        return ValidationReport(metrics=metrics, weights=weights, category_weights={"generative": 1.0})

    except Exception as e:  # noqa: BLE001
        logger.warning("validate_generative failed: %s", e)
        return _err_report(str(e))


def _err_report(msg: str) -> ValidationReport:
    return ValidationReport(
        metrics={"coverage_deficit": create_error_metric("coverage_deficit", msg, "generative")},
        weights={}, category_weights={"generative": 1.0},
    )
