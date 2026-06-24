"""Joint-lens omnibus: the Classifier Two-Sample Test (C2ST).

Every other metric checks a *named* statistic (a marginal, a correlation, a tail coefficient).
C2ST is the catch-all for the **unknown unknowns**: if a classifier can tell real from
synthetic better than chance, the two distributions differ *somewhere* — in a way no
hand-picked statistic had to anticipate. It's the single most powerful "did we miss
something?" detector (Lopez-Paz & Oquab 2017).

We use a **k-NN leave-one-out** classifier (Schilling 1986 / Henze nearest-neighbour
two-sample test), not a parametric one:
  - nonparametric → catches *nonlinear* differences a logistic classifier would miss,
  - needs no training/test split (LOO is built in) and no extra dependency (pure numpy/scipy),
  - the same neighbour graph gives a cheap **permutation p-value** (shuffle labels, re-vote).

Real and synthetic are balanced (equal n) and pooled-standardized so the chance accuracy is
0.5. Reported as ``c2st = |2·accuracy − 1|`` (0 = indistinguishable = good, 1 = perfectly
separable = bad), with the permutation p-value in metadata (significant separability = a flag).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import quality_from_value

logger = logging.getLogger(__name__)

# Thresholds on |2·acc − 1|. Calibrated against the real-vs-real floor (random-split of an
# equity-macro panel, 2026-06-22): floor ~0.002 (two real halves are indistinguishable),
# Gaussian baseline ~0.24. NB: a TIME-split of real data scores ~0.31 — c2st correctly
# detects market non-stationarity between periods, so a high c2st on held-out time is signal,
# not noise. See tools/calibrate.py.
C2ST_THRESHOLDS = {"excellent": 0.05, "good": 0.15, "acceptable": 0.30}


def _points(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 3:  # paths → per-path feature vectors (same space as generative)
        from finval.metrics.generative import path_features

        a = path_features(a)
    elif a.ndim != 2:
        raise ValueError(f"c2st expects 2D rows or 3D paths, got {a.shape}")
    return a[np.all(np.isfinite(a), axis=1)]


def compute_c2st(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    k: int = 5,
    max_per_class: int = 1500,
    n_perm: int = 100,
    seed: int = 0,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """k-NN leave-one-out classifier two-sample test. ``value = |2·acc − 1|`` (lower better)."""
    try:
        S, R = _points(synthetic), _points(real)
        n = min(len(S), len(R), max_per_class)
        if n < k + 5:
            return create_error_metric("c2st", f"need >= {k + 5} clean points per side; got {n}", "joint")
        rng = np.random.default_rng(seed)
        # balance classes (deterministic subsample of the larger side)
        S = S[rng.permutation(len(S))[:n]]
        R = R[rng.permutation(len(R))[:n]]
        X = np.vstack([R, S])
        y = np.concatenate([np.ones(n), np.zeros(n)])          # 1 = real, 0 = synth
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)     # pooled-standardize

        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        nn_idx = np.argsort(D, axis=1)[:, :k]                  # k nearest (self excluded)
        nn_labels = y[nn_idx]                                  # (2n, k); k odd → no ties
        pred = (nn_labels.mean(axis=1) >= 0.5).astype(float)
        acc = float((pred == y).mean())
        c2st = abs(2.0 * acc - 1.0)

        # permutation null: re-vote with shuffled labels on the SAME neighbour graph
        if n_perm and n_perm > 0:
            null = np.empty(n_perm)
            for b in range(n_perm):
                yp = rng.permutation(y)
                predp = (yp[nn_idx].mean(axis=1) >= 0.5).astype(float)
                null[b] = abs(2.0 * float((predp == yp).mean()) - 1.0)
            pval = float((np.sum(null >= c2st) + 1) / (n_perm + 1))
        else:
            pval = float("nan")

        th = thresholds or C2ST_THRESHOLDS
        quality, passed = quality_from_value(c2st, th)
        sig = "" if not (pval == pval) else (
            " — significantly separable" if pval < 0.05 else " — not significantly separable"
        )
        return MetricResult(
            name="c2st",
            value=c2st,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="joint",
            interpretation=f"kNN C2ST acc={acc:.3f} → |2·acc−1|={c2st:.3f} (p={pval:.3f}){sig}",
            metadata={"accuracy": acc, "p_value": pval, "k": k, "n_per_class": n, "n_perm": n_perm},
        )

    except Exception as e:  # noqa: BLE001
        logger.warning("c2st failed: %s", e)
        return create_error_metric("c2st", str(e), "joint")
