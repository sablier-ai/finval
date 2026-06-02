"""Dependence metrics: cross-feature correlation and copula structure.

Pearson and Spearman capture linear/monotonic dependence. The empirical
copula distance captures full rank-based dependence. Tail dependence
coefficients (lambda_U, lambda_L) capture conditional co-movement in the
extremes. Correlation breakdown captures how much dependence changes
between stress and calm periods.

All pairwise metrics use **mean error** across pairs rather than max,
because max over n*(n-1)/2 pairs is dominated by sampling noise when
any individual pair has naturally wide correlation CIs.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import DEPENDENCE_THRESHOLDS, quality_from_value

logger = logging.getLogger(__name__)


def _correlation_matrix(x: np.ndarray, method: str) -> np.ndarray:
    """Correlation matrix, robust to NaNs.

    Fast path (no NaNs present): whole-matrix ``np.corrcoef`` / ``spearmanr`` —
    identical numbers to before, so clean-data results are unchanged. When any
    NaN is present, fall back to **pairwise-complete** estimation (mask each
    pair's missing rows independently) so a single ragged feature with a short
    history doesn't poison the whole correlation row. Pairs with <10 overlapping
    finite observations (or a non-finite estimate) contribute rho=0.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown method {method!r}")
    n_features = x.shape[1]

    if not np.isnan(x).any():
        if method == "pearson":
            return np.asarray(np.corrcoef(x, rowvar=False))
        corr, _ = stats.spearmanr(x)
        if n_features == 2:
            corr = np.array([[1.0, corr], [corr, 1.0]])
        return np.asarray(corr)

    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    c = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            xi, xj = x[:, i], x[:, j]
            m = ~(np.isnan(xi) | np.isnan(xj))
            if int(m.sum()) < 10:
                rho = 0.0
            else:
                rho = float(corr_fn(xi[m], xj[m])[0])
                if not np.isfinite(rho):
                    rho = 0.0
            c[i, j] = c[j, i] = rho
    return c


def _pairwise_correlation_error(
    synthetic: np.ndarray,
    real: np.ndarray,
    method: str,
) -> tuple[float, float, dict[str, float]]:
    """Helper: compute mean and max pairwise correlation error."""
    n_features = synthetic.shape[1]

    syn_corr = _correlation_matrix(synthetic, method)
    real_corr = _correlation_matrix(real, method)

    mask = ~np.eye(n_features, dtype=bool)
    errors = np.abs(syn_corr - real_corr)[mask]
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    # Per-pair breakdown (upper triangle only)
    per_pair: dict[str, float] = {}
    for i in range(n_features):
        for j in range(i + 1, n_features):
            key = f"{i}<->{j}"
            per_pair[key] = float(abs(syn_corr[i, j] - real_corr[i, j]))

    return mean_err, max_err, per_pair


def compute_pearson_correlation(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Mean absolute error between synthetic and real Pearson correlation matrices."""
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        if synthetic.shape[1] < 2:
            return create_error_metric(
                "pearson_corr", "need >=2 features", "dependence"
            )

        mean_err, max_err, per_pair = _pairwise_correlation_error(synthetic, real, "pearson")
        th = thresholds or DEPENDENCE_THRESHOLDS["pearson_corr"]
        quality, passed = quality_from_value(mean_err, th)

        return MetricResult(
            name="pearson_corr",
            value=mean_err,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="dependence",
            interpretation=f"Mean Pearson correlation error {mean_err:.4f}, max {max_err:.4f}",
            per_pair=per_pair,
            metadata={"max_error": max_err},
        )

    except Exception as e:
        logger.warning("pearson_corr failed: %s", e)
        return create_error_metric("pearson_corr", str(e), "dependence")


def compute_spearman_correlation(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Mean absolute error between synthetic and real Spearman rank correlations."""
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        if synthetic.shape[1] < 2:
            return create_error_metric("spearman_corr", "need >=2 features", "dependence")

        mean_err, max_err, per_pair = _pairwise_correlation_error(synthetic, real, "spearman")
        th = thresholds or DEPENDENCE_THRESHOLDS["spearman_corr"]
        quality, passed = quality_from_value(mean_err, th)

        return MetricResult(
            name="spearman_corr",
            value=mean_err,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="dependence",
            interpretation=f"Mean Spearman correlation error {mean_err:.4f}, max {max_err:.4f}",
            per_pair=per_pair,
            metadata={"max_error": max_err},
        )

    except Exception as e:
        logger.warning("spearman_corr failed: %s", e)
        return create_error_metric("spearman_corr", str(e), "dependence")


def compute_copula_distance(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    grid_size: int = 20,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Cramer-von Mises distance between pairwise empirical copulas.

    For each feature pair, transforms both to pseudo-observations (ranks
    scaled to (0, 1)), evaluates the empirical copula on a grid_size x
    grid_size grid, and computes the root-mean-square difference. Returns
    the mean across pairs. Copulas isolate the dependence structure from
    the marginals.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        n_features = synthetic.shape[1]
        if n_features < 2:
            return create_error_metric("copula_distance", "need >=2 features", "dependence")

        def to_pseudo(x: np.ndarray) -> np.ndarray:
            n = len(x)
            out = np.empty_like(x, dtype=float)
            for j in range(x.shape[1]):
                out[:, j] = stats.rankdata(x[:, j]) / (n + 1)
            return out

        syn_u = to_pseudo(synthetic)
        real_u = to_pseudo(real)

        grid = np.linspace(0.05, 0.95, grid_size)
        u_grid, v_grid = np.meshgrid(grid, grid)
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()

        per_pair: dict[str, float] = {}
        errors: list[float] = []

        # Copula distance is O(n_pairs * grid_size^2); at large universes
        # (F=1000 -> ~500k pairs) full enumeration is intractable. Subsample a
        # fixed, seeded set of pairs so the metric scales without losing
        # determinism. (Mirrors the engine's MAX_PAIRS cap.)
        MAX_PAIRS = 5000
        all_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
        if len(all_pairs) > MAX_PAIRS:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(all_pairs), size=MAX_PAIRS, replace=False)
            pair_list = [all_pairs[k] for k in sel]
        else:
            pair_list = all_pairs

        for i, j in pair_list:
            syn_pair = syn_u[:, [i, j]]
            real_pair = real_u[:, [i, j]]

            syn_c = np.array(
                [
                    np.mean((syn_pair[:, 0] <= u) & (syn_pair[:, 1] <= v))
                    for u, v in zip(u_flat, v_flat, strict=False)
                ]
            )
            real_c = np.array(
                [
                    np.mean((real_pair[:, 0] <= u) & (real_pair[:, 1] <= v))
                    for u, v in zip(u_flat, v_flat, strict=False)
                ]
            )

            dist = float(np.sqrt(np.mean((syn_c - real_c) ** 2)))
            per_pair[f"{i}<->{j}"] = dist
            errors.append(dist)

        mean_err = float(np.mean(errors)) if errors else 0.0
        max_err = float(np.max(errors)) if errors else 0.0

        th = thresholds or DEPENDENCE_THRESHOLDS["copula_distance"]
        quality, passed = quality_from_value(mean_err, th)

        return MetricResult(
            name="copula_distance",
            value=mean_err,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="dependence",
            interpretation=f"Mean copula CvM distance {mean_err:.4f}, max {max_err:.4f}",
            per_pair=per_pair,
            metadata={"max_error": max_err, "grid_size": grid_size},
        )

    except Exception as e:
        logger.warning("copula_distance failed: %s", e)
        return create_error_metric("copula_distance", str(e), "dependence")


def _tail_coefficient(u: np.ndarray, v: np.ndarray, quantile: float, tail: str) -> float:
    """Empirical tail dependence coefficient at a finite quantile."""
    n = len(u)
    if n < 50:
        return 0.0

    if tail == "upper":
        threshold = 1 - quantile
        in_tail_u = u > threshold
        in_tail_v = v > threshold
    else:
        threshold = quantile
        in_tail_u = u < threshold
        in_tail_v = v < threshold

    n_in_u = int(np.sum(in_tail_u))
    if n_in_u == 0:
        return 0.0
    n_both = int(np.sum(in_tail_u & in_tail_v))
    return n_both / n_in_u


def compute_tail_dependence(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    quantile: float = 0.05,
    thresholds: dict[str, float] | None = None,
) -> tuple[MetricResult, MetricResult]:
    """Tail dependence coefficients (lambda_U, lambda_L), mean error across pairs.

    - Lower: lambda_L = P(V < q | U < q)   — crash co-movement
    - Upper: lambda_U = P(V > 1-q | U > 1-q) — rally co-movement

    Returns two MetricResults: (upper, lower). These are critical for risk
    because linear correlation underestimates co-movement in the tails.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        n_features = synthetic.shape[1]
        if n_features < 2:
            upper = create_error_metric("tail_dependence_upper", "need >=2 features", "dependence")
            lower = create_error_metric("tail_dependence_lower", "need >=2 features", "dependence")
            return upper, lower

        def to_uniform(x: np.ndarray) -> np.ndarray:
            n = len(x)
            out = np.empty_like(x, dtype=float)
            for j in range(x.shape[1]):
                out[:, j] = stats.rankdata(x[:, j]) / (n + 1)
            return out

        syn_u = to_uniform(synthetic)
        real_u = to_uniform(real)

        upper_pair: dict[str, dict] = {}
        lower_pair: dict[str, dict] = {}
        upper_errs: list[float] = []
        lower_errs: list[float] = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                key = f"{i}<->{j}"

                syn_up = _tail_coefficient(syn_u[:, i], syn_u[:, j], quantile, "upper")
                real_up = _tail_coefficient(real_u[:, i], real_u[:, j], quantile, "upper")
                up_err = abs(syn_up - real_up)
                upper_pair[key] = {"synthetic": syn_up, "real": real_up, "error": up_err}
                upper_errs.append(up_err)

                syn_lo = _tail_coefficient(syn_u[:, i], syn_u[:, j], quantile, "lower")
                real_lo = _tail_coefficient(real_u[:, i], real_u[:, j], quantile, "lower")
                lo_err = abs(syn_lo - real_lo)
                lower_pair[key] = {"synthetic": syn_lo, "real": real_lo, "error": lo_err}
                lower_errs.append(lo_err)

        upper_mean = float(np.mean(upper_errs)) if upper_errs else 0.0
        upper_max = float(np.max(upper_errs)) if upper_errs else 0.0
        lower_mean = float(np.mean(lower_errs)) if lower_errs else 0.0
        lower_max = float(np.max(lower_errs)) if lower_errs else 0.0

        up_th = thresholds or DEPENDENCE_THRESHOLDS["tail_dependence_upper"]
        lo_th = thresholds or DEPENDENCE_THRESHOLDS["tail_dependence_lower"]
        up_q, up_p = quality_from_value(upper_mean, up_th)
        lo_q, lo_p = quality_from_value(lower_mean, lo_th)

        upper_result = MetricResult(
            name="tail_dependence_upper",
            value=upper_mean,
            quality=up_q,
            passed=up_p,
            thresholds=up_th,
            category="dependence",
            interpretation=(
                f"Upper tail (rally) lambda_U error: mean {upper_mean:.4f}, max {upper_max:.4f} "
                f"at quantile q={1-quantile:.2f}"
            ),
            per_pair=upper_pair,
            metadata={"max_error": upper_max, "quantile": 1 - quantile},
        )

        lower_result = MetricResult(
            name="tail_dependence_lower",
            value=lower_mean,
            quality=lo_q,
            passed=lo_p,
            thresholds=lo_th,
            category="dependence",
            interpretation=(
                f"Lower tail (crash) lambda_L error: mean {lower_mean:.4f}, max {lower_max:.4f} "
                f"at quantile q={quantile:.2f}"
            ),
            per_pair=lower_pair,
            metadata={"max_error": lower_max, "quantile": quantile},
        )

        return upper_result, lower_result

    except Exception as e:
        logger.warning("tail_dependence failed: %s", e)
        return (
            create_error_metric("tail_dependence_upper", str(e), "dependence"),
            create_error_metric("tail_dependence_lower", str(e), "dependence"),
        )


def compute_correlation_breakdown(
    synthetic: np.ndarray,
    real: np.ndarray,
    feature_names: list[str] | None = None,
    stress_percentile: float = 10.0,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Correlation breakdown between stress and calm regimes.

    "Correlations go to one in a crisis" — real markets show much stronger
    correlations during drawdowns. A good generative model should reproduce
    this asymmetry. Defines stress as periods where the cross-sectional
    mean return is in the bottom `stress_percentile`%, calm as the rest,
    then compares synthetic vs real correlation matrices in each regime.

    Returns the max of (stress_error, calm_error), each computed as the
    mean pairwise absolute difference.
    """
    try:
        synthetic = np.asarray(synthetic)
        real = np.asarray(real)
        n_features = synthetic.shape[1]
        if n_features < 2:
            return create_error_metric(
                "correlation_breakdown", "need >=2 features", "dependence"
            )

        def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            avg = np.mean(x, axis=1)
            t = np.percentile(avg, stress_percentile)
            return x[avg < t], x[avg >= t]

        def mean_corr_err(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 20 or len(b) < 20:
                return 0.0
            ca = np.corrcoef(a, rowvar=False)
            cb = np.corrcoef(b, rowvar=False)
            mask = ~np.eye(n_features, dtype=bool)
            return float(np.mean(np.abs(ca - cb)[mask]))

        syn_stress, syn_calm = split(synthetic)
        real_stress, real_calm = split(real)

        stress_err = mean_corr_err(syn_stress, real_stress)
        calm_err = mean_corr_err(syn_calm, real_calm)
        value = max(stress_err, calm_err)

        th = thresholds or DEPENDENCE_THRESHOLDS["correlation_breakdown"]
        quality, passed = quality_from_value(value, th)

        return MetricResult(
            name="correlation_breakdown",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="dependence",
            interpretation=(
                f"Stress-regime correlation error {stress_err:.4f}, calm {calm_err:.4f}. "
                f"Max {value:.4f}"
            ),
            metadata={
                "stress_error": stress_err,
                "calm_error": calm_err,
                "stress_percentile": stress_percentile,
                "n_stress_syn": int(len(syn_stress)),
                "n_stress_real": int(len(real_stress)),
            },
        )

    except Exception as e:
        logger.warning("correlation_breakdown failed: %s", e)
        return create_error_metric("correlation_breakdown", str(e), "dependence")
