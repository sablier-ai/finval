"""Stylized-fact & SOTA gap metrics (v0.4.0, gap-close).

Added after a SOTA audit of synthetic-financial-time-series evaluation. Each catches a failure
the rest of the suite provably misses:

  time_reversal_asymmetry  — the Zumbach effect / time-irreversibility (Cont fact #11). Real
                             markets aren't time-reversible (past returns drive future vol
                             asymmetrically); most generators are. Leverage(forward) − leverage(backward).
  aggregational_gaussianity — Cont fact #4: excess kurtosis must DECAY as returns aggregate.
                             We had variance term-structure; this is the kurtosis/shape one.
  conditional_heavy_tails  — Cont fact #7: tails stay heavy AFTER de-volatilizing (kurtosis of
                             rolling-vol-standardized residuals). Catches "fat tails only via the
                             vol process".
  regime_persistence       — dwell-time of the high-vol state. Catches crises that mean-revert
                             too fast (right mixture, wrong persistence).
  hill_tail_index          — the tail EXPONENT (power-law α, target ≈3 / Cont fact #2), not just
                             heaviness. Catches wrong tail SHAPE.
  variogram_score          — Scheuerer-Hamill: the energy distance is nearly correlation-BLIND;
                             the variogram (pairwise-difference moments) is dependence-sensitive.
                             Hardens the joint omnibus.
  signature_distance       — truncated-signature (level-2) path distance: higher-order
                             path-ordering / lead-lag interactions no marginal/ACF/energy metric
                             sees. Path order matters. (Truncation level reported — higher levels
                             decay factorially, so this is a level-2 lower bound, not the full law.)

All "lower is better", pure numpy, model-agnostic. Thresholds are rough pending the calibration
pass (tools/calibrate.py).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import quality_from_value

logger = logging.getLogger(__name__)

# Calibrated vs the real-vs-real floor (random-split equity-macro panel, tools/calibrate.py).
# The kurtosis-based metrics (aggregational, conditional-heavy) have HIGH floors — sample
# kurtosis is intrinsically noisy — so their bands are wide. signature_distance has a TINY
# scale / low discrimination on daily data (the research-predicted low power of a level-2
# truncation): treat it as a weak diagnostic until path-scaling/lead-lag/higher levels are tuned.
TIME_REVERSAL_THRESHOLDS = {"excellent": 0.05, "good": 0.10, "acceptable": 0.20}   # floor ~0.04
AGG_GAUSS_THRESHOLDS = {"excellent": 3.6, "good": 6.0, "acceptable": 9.0}           # floor ~3.6 (noisy)
COND_HEAVY_THRESHOLDS = {"excellent": 4.6, "good": 10.0, "acceptable": 18.0}        # floor ~4.6 (noisy)
REGIME_PERSIST_THRESHOLDS = {"excellent": 0.05, "good": 0.12, "acceptable": 0.25}   # floor ~0.04
HILL_THRESHOLDS = {"excellent": 0.06, "good": 0.12, "acceptable": 0.25}             # floor ~0.04
VARIOGRAM_THRESHOLDS = {"excellent": 0.05, "good": 0.10, "acceptable": 0.20}        # floor ~0.01
SIGNATURE_THRESHOLDS = {"excellent": 0.01, "good": 0.03, "acceptable": 0.08}        # floor ~0.00 (low power)


def _names(n, feature_names):
    return feature_names or [f"feature_{i}" for i in range(n)]


def _result(name, value, th, category, interp, **meta):
    q, p = quality_from_value(value, th)
    return MetricResult(name=name, value=value, quality=q, passed=p, thresholds=th,
                        category=category, interpretation=interp, metadata=meta)


def _corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---- temporal -----------------------------------------------------------------------------
def compute_time_reversal_asymmetry(synth_paths, real_paths, feature_names=None, lags=(1, 2, 5), thresholds=None):
    """Zumbach / time-irreversibility: forward leverage corr(r_t, |r_{t+k}|) MINUS backward
    corr(r_{t+k}, |r_t|). Zero for any time-reversible process; real markets are not. Scores
    |asymmetry_syn − asymmetry_real| averaged over features and lags."""
    try:
        S, R = np.asarray(synth_paths, float), np.asarray(real_paths, float)
        if S.ndim != 3 or R.ndim != 3 or S.shape[1] < max(lags) + 2:
            return create_error_metric("time_reversal_asymmetry", "need 3D paths, horizon > max lag+1", "temporal")

        def asym(P, j, k):
            r = P[:, :, j]
            fwd = _corr(r[:, :-k], np.abs(r[:, k:]))   # past return vs future |return|
            bwd = _corr(r[:, k:], np.abs(r[:, :-k]))   # future return vs past |return|
            return fwd - bwd

        names = _names(R.shape[2], feature_names)
        per = {nm: float(np.mean([abs(asym(S, j, k) - asym(R, j, k)) for k in lags]))
               for j, nm in enumerate(names)}
        v = float(np.mean(list(per.values())))
        m = _result("time_reversal_asymmetry", v, thresholds or TIME_REVERSAL_THRESHOLDS, "temporal",
                    f"|Δ time-reversal asymmetry| {v:.3f} (lev_fwd−lev_bwd, lags {lags})")
        m.per_feature = per
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("time_reversal_asymmetry failed: %s", e)
        return create_error_metric("time_reversal_asymmetry", str(e), "temporal")


def compute_regime_persistence(synth_paths, real_paths, feature_names=None, q=0.8, thresholds=None):
    """Dwell-time of the high-|return| state: mean run length of consecutive exceedances within a
    path. Catches crises that mean-revert too fast. |mean_run_syn − mean_run_real| / real."""
    try:
        S, R = np.asarray(synth_paths, float), np.asarray(real_paths, float)
        if S.ndim != 3 or R.ndim != 3 or S.shape[1] < 5:
            return create_error_metric("regime_persistence", "need 3D paths, horizon>=5", "temporal")

        def mean_run(P, j):
            thr = np.nanquantile(np.abs(P[:, :, j]), q)
            runs = []
            for row in (np.abs(P[:, :, j]) > thr):
                c = 0
                for x in row:
                    if x:
                        c += 1
                    elif c:
                        runs.append(c); c = 0
                if c:
                    runs.append(c)
            return float(np.mean(runs)) if runs else 0.0

        names = _names(R.shape[2], feature_names)
        per = {}
        for j, nm in enumerate(names):
            mr = mean_run(R, j)
            per[nm] = abs(mean_run(S, j) - mr) / (mr + 1e-9)
        v = float(np.mean(list(per.values())))
        m = _result("regime_persistence", v, thresholds or REGIME_PERSIST_THRESHOLDS, "temporal",
                    f"|Δ high-vol dwell-time| {v:.3f} (rel., q={q})")
        m.per_feature = per
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("regime_persistence failed: %s", e)
        return create_error_metric("regime_persistence", str(e), "temporal")


# ---- marginal / distribution --------------------------------------------------------------
def _exkurt(x):
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    z = (x - x.mean()) / (x.std() + 1e-12)
    return float((z ** 4).mean() - 3.0)


def compute_aggregational_gaussianity(synth_paths, real_paths, feature_names=None, taus=(1, 5, 10), thresholds=None):
    """Cont fact #4: excess kurtosis of τ-aggregated returns must decay toward 0 as τ grows.
    Compares the kurtosis-vs-τ curve. |exkurt_syn(τ) − exkurt_real(τ)| over features and horizons."""
    try:
        S, R = np.asarray(synth_paths, float), np.asarray(real_paths, float)
        if S.ndim != 3 or R.ndim != 3:
            return create_error_metric("aggregational_gaussianity", "need 3D paths", "distribution")
        H = R.shape[1]
        use = [t for t in taus if t <= H]
        if not use:
            return create_error_metric("aggregational_gaussianity", f"horizon {H} < min tau", "distribution")

        def agg_kurt(P, j, tau):
            nb = P.shape[1] // tau
            agg = P[:, : nb * tau, j].reshape(P.shape[0], nb, tau).sum(axis=2)  # τ-summed returns
            return _exkurt(agg.ravel())

        errs = [abs(agg_kurt(S, j, t) - agg_kurt(R, j, t)) for j in range(R.shape[2]) for t in use]
        v = float(np.mean(errs))
        return _result("aggregational_gaussianity", v, thresholds or AGG_GAUSS_THRESHOLDS, "distribution",
                       f"mean |Δ excess-kurtosis| across aggregation τ={tuple(use)} = {v:.3f}")
    except Exception as e:  # noqa: BLE001
        logger.warning("aggregational_gaussianity failed: %s", e)
        return create_error_metric("aggregational_gaussianity", str(e), "distribution")


def compute_conditional_heavy_tails(synth_paths, real_paths, feature_names=None, window=5, thresholds=None):
    """Cont fact #7: after de-volatilizing (divide by trailing rolling std), residuals stay
    heavy-tailed. Compares excess kurtosis of rolling-vol-standardized residuals. Catches a model
    whose fat tails come ONLY from its volatility process (devol'd residuals collapse to Gaussian)."""
    try:
        S, R = np.asarray(synth_paths, float), np.asarray(real_paths, float)
        if S.ndim != 3 or R.ndim != 3 or S.shape[1] < window + 3:
            return create_error_metric("conditional_heavy_tails", "need 3D paths, horizon>window+2", "distribution")

        def resid_kurt(P, j):
            r = P[:, :, j]
            res = []
            for row in r:
                for t in range(window, len(row)):
                    sd = row[t - window:t].std()
                    if sd > 1e-12:
                        res.append(row[t] / sd)
            return _exkurt(np.array(res))

        names = _names(R.shape[2], feature_names)
        per = {nm: abs(resid_kurt(S, j) - resid_kurt(R, j)) for j, nm in enumerate(names)}
        v = float(np.mean(list(per.values())))
        m = _result("conditional_heavy_tails", v, thresholds or COND_HEAVY_THRESHOLDS, "distribution",
                    f"mean |Δ devol'd-residual excess-kurtosis| {v:.3f} (window={window})")
        m.per_feature = per
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("conditional_heavy_tails failed: %s", e)
        return create_error_metric("conditional_heavy_tails", str(e), "distribution")


def compute_hill_tail_index(synthetic, real, feature_names=None, tail_frac=0.05, thresholds=None):
    """Hill estimator of the tail index ξ=1/α on both tails (real α≈3 → ξ≈0.33). Catches wrong
    tail SHAPE (exponential vs power-law) that a quantile check passes. |ξ_syn − ξ_real| averaged
    over features and both tails. (Report is a robust k-fraction Hill, not a single-k point.)"""
    try:
        S = np.asarray(synthetic, float); R = np.asarray(real, float)
        S = S[np.all(np.isfinite(S), axis=1)]; R = R[np.all(np.isfinite(R), axis=1)]
        if len(S) < 500 or len(R) < 500:
            return create_error_metric("hill_tail_index", "need >=500 rows/side", "distribution")

        def hill(x, right):
            v = x if right else -x
            k = max(10, int(len(v) * tail_frac))
            top = np.sort(v)[-k - 1:]
            u = top[0]
            exc = top[1:][top[1:] > max(u, 1e-12)]
            if u <= 1e-12 or len(exc) < 5:
                return np.nan
            return float(np.mean(np.log(exc / u)))

        names = _names(R.shape[1], feature_names)
        per, errs = {}, []
        for j, nm in enumerate(names):
            ds = []
            for right in (True, False):
                hs, hr = hill(S[:, j], right), hill(R[:, j], right)
                if np.isfinite(hs) and np.isfinite(hr):
                    ds.append(abs(hs - hr))
            if ds:
                per[nm] = float(np.mean(ds)); errs.append(per[nm])
        if not errs:
            return create_error_metric("hill_tail_index", "no finite tail-index estimates", "distribution")
        v = float(np.mean(errs))
        m = _result("hill_tail_index", v, thresholds or HILL_THRESHOLDS, "distribution",
                    f"mean |Δ Hill tail-index ξ=1/α| {v:.3f} (real α≈3 → ξ≈0.33)")
        m.per_feature = per
        return m
    except Exception as e:  # noqa: BLE001
        logger.warning("hill_tail_index failed: %s", e)
        return create_error_metric("hill_tail_index", str(e), "distribution")


# ---- dependence / joint -------------------------------------------------------------------
def compute_variogram_score(synthetic, real, feature_names=None, p=0.5, thresholds=None):
    """Scheuerer-Hamill variogram-of-order-p, as a two-sample dependence-fidelity distance:
    compares E|z_i − z_j|^p of synth vs real over feature pairs (z standardized by real). The
    energy distance is nearly correlation-blind; this pairwise-difference moment IS dependence-
    sensitive, hardening the joint omnibus. mean over pairs |E_syn − E_real|."""
    try:
        S = np.asarray(synthetic, float); R = np.asarray(real, float)
        S = S[np.all(np.isfinite(S), axis=1)]; R = R[np.all(np.isfinite(R), axis=1)]
        d = R.shape[1]
        if d < 2 or len(S) < 50 or len(R) < 50:
            return create_error_metric("variogram_score", "need >=2 features and >=50 rows/side", "dependence")
        mu, sd = R.mean(0), R.std(0) + 1e-12
        Zs, Zr = (S - mu) / sd, (R - mu) / sd
        errs = []
        for i in range(d):
            for j in range(i + 1, d):
                vs = np.mean(np.abs(Zs[:, i] - Zs[:, j]) ** p)
                vr = np.mean(np.abs(Zr[:, i] - Zr[:, j]) ** p)
                errs.append(abs(vs - vr))
        v = float(np.mean(errs))
        return _result("variogram_score", v, thresholds or VARIOGRAM_THRESHOLDS, "dependence",
                       f"mean |Δ E|z_i−z_j|^{p}| over {len(errs)} pairs = {v:.3f}", n_pairs=len(errs))
    except Exception as e:  # noqa: BLE001
        logger.warning("variogram_score failed: %s", e)
        return create_error_metric("variogram_score", str(e), "dependence")


def _level2_signature(paths, max_n=1500):
    """Truncated (level-2) signature feature vector per path of a time-augmented, standardized
    path: [S1 (C), S2 (C*C)]. S1 = total increment; S2_{ij} = Σ_t (partial-sum_i before t)·dX_j(t)."""
    P = np.asarray(paths, float)
    n, H, D = P.shape
    if n > max_n:
        P = P[np.random.default_rng(0).choice(n, max_n, replace=False)]
        n = max_n
    t = np.linspace(0.0, 1.0, H)[None, :, None] * np.ones((P.shape[0], 1, 1))
    aug = np.concatenate([t, P], axis=2)                       # time augmentation → (n,H,D+1)
    C = aug.shape[2]
    feats = np.empty((P.shape[0], C + C * C))
    for a in range(P.shape[0]):
        dX = np.diff(aug[a], axis=0)                            # (H-1, C)
        run = np.cumsum(dX, axis=0) - dX                       # partial sum strictly before step t
        S1 = dX.sum(axis=0)                                    # (C,)
        S2 = run.T @ dX                                        # (C, C)
        feats[a] = np.concatenate([S1, S2.ravel()])
    return feats


def compute_signature_distance(synth_paths, real_paths, feature_names=None, thresholds=None):
    """Level-2 truncated-signature path distance: normalized energy distance between the signature
    feature clouds (time-augmented, standardized paths). Captures higher-order path-ordering and
    cross-channel lead-lag interactions invisible to marginal/ACF/energy metrics. NB: level-2 is a
    lower bound on the full law (higher signature levels decay factorially); the level is reported."""
    try:
        S, R = np.asarray(synth_paths, float), np.asarray(real_paths, float)
        if S.ndim != 3 or R.ndim != 3:
            return create_error_metric("signature_distance", "need 3D paths", "joint")
        fs, fr = _level2_signature(S), _level2_signature(R)
        mu, sd = fr.mean(0), fr.std(0) + 1e-12
        fs, fr = (fs - mu) / sd, (fr - mu) / sd                # standardize sig-features by real
        # normalized energy distance between the two signature clouds
        xy = cdist(fs, fr); xx = cdist(fs, fs); yy = cdist(fr, fr)
        nS, nR = len(fs), len(fr)
        e = 2 * xy.mean() - xx.sum() / (nS * max(nS - 1, 1)) - yy.sum() / (nR * max(nR - 1, 1))
        scale = float(np.mean(np.linalg.norm(fr, axis=1))) + 1e-9
        v = max(0.0, float(e)) / scale
        return _result("signature_distance", v, thresholds or SIGNATURE_THRESHOLDS, "joint",
                       f"level-2 signature energy distance {v:.3f} (time-aug; level-2 lower bound)",
                       truncation_level=2, n_sig_features=fr.shape[1])
    except Exception as e:  # noqa: BLE001
        logger.warning("signature_distance failed: %s", e)
        return create_error_metric("signature_distance", str(e), "joint")
