"""Backtest-overfitting metrics.

This module implements the family of metrics that quantify how much a
backtest's apparent performance is due to selection bias and overfitting
rather than real skill. They are the metrics that Marcos Lopez de Prado
has spent much of the last decade arguing should be applied to *every*
strategy backtest.

Two metrics are exposed:

  - ``compute_deflated_sharpe`` — the Deflated Sharpe Ratio of Bailey
    and Lopez de Prado (2014). Adjusts the observed Sharpe for the
    number of trials and the higher moments of the return distribution.

  - ``compute_pbo`` — the Probability of Backtest Overfitting of
    Bailey, Borwein, Lopez de Prado and Zhu (2015), computed via
    Combinatorially Symmetric Cross-Validation (CSCV).

Both functions return a ``MetricResult``. Following the finval
"lower-is-better" convention:

  - For DSR we report ``1 - DSR_probability`` so that a backtest whose
    Sharpe is *unlikely* to be real receives a high value.
  - For PBO we report the probability directly — a backtest selection
    procedure that overfits has PBO close to 0.5 or higher.

References:
  Bailey, Lopez de Prado (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and Non-Normality." Journal of Portfolio Management
    40(5): 94-107.
  Bailey, Borwein, Lopez de Prado, Zhu (2015).
    "The Probability of Backtest Overfitting." Journal of Computational
    Finance 20(4): 39-69.
  Lopez de Prado (2018).
    *Advances in Financial Machine Learning*, ch. 14.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Optional

import numpy as np
from scipy import stats
from scipy.stats import norm

from finval.core.result import MetricResult, create_error_metric
from finval.core.thresholds import BACKTEST_THRESHOLDS, quality_from_value

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant, used in the false-strategy expression for
# E[max{Z_1, ..., Z_N}] of N iid standard Normals.
_EULER_MASCHERONI = 0.5772156649015329


def _expected_max_sharpe_under_null(n_trials: int, sr_variance_across_trials: float) -> float:
    """Expected Sharpe of the best of N iid trials under the null SR=0.

    From Bailey & Lopez de Prado (2014), the "false strategy theorem":

        E[max{SR_n}] ~ sqrt(V) * Z

    where ``V = Var(SR_n)`` across trials and ``Z`` is the expected
    maximum of ``N`` iid standard Normals, approximated by Embrechts'
    asymptotic expansion:

        Z ~ (1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))

    For N == 1, returns 0 (no selection bias).
    """
    if n_trials <= 1:
        return 0.0
    z = (
        (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    return float(math.sqrt(max(sr_variance_across_trials, 0.0)) * z)


def compute_deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    *,
    sr_benchmark: float = 0.0,
    var_trials: Optional[float] = None,
    annualization_factor: float = 1.0,
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Tests whether the observed Sharpe of a single strategy is large
    enough to be real, after adjusting for:

      1. The number of trials (``n_trials``) that were searched before
         settling on this strategy. The more trials, the higher the
         observed Sharpe must be to survive deflation.
      2. The non-Normality of the return distribution (via skew and
         kurtosis). Fat-tailed strategies are penalised because the
         Sharpe estimator has higher variance under non-Normality.

    The DSR is a probability in ``[0, 1]``; high values mean the
    Sharpe is unlikely to be a statistical artefact. We follow
    finval's "lower-is-better" convention and store
    ``value = 1 - DSR``.

    Args:
        returns: 1-D array of strategy returns at any fixed frequency
            (daily, monthly, ...). Must be at least 30 observations.
        n_trials: Number of independent strategy variants that were
            evaluated before this one was selected. **Honest reporting
            of this is the whole point of DSR.** A single hand-coded
            strategy is 1; a grid-search over K parameters is K.
        sr_benchmark: Null Sharpe to deflate against (default 0). Set
            to a positive value to test against an actively-managed
            benchmark.
        var_trials: Variance of the Sharpe ratio *across trials*
            (``Var(SR_n)`` in MLDP's notation). When unknown, we use
            the variance of the single observed SR estimator as a
            conservative proxy. Pass an explicit value if you have
            the cross-trial empirical variance.
        annualization_factor: Multiplier for the human-readable
            annualised Sharpe (``sqrt(252)`` for daily, ``sqrt(12)``
            for monthly, etc.). Does **not** affect the DSR itself.
        thresholds: Override default thresholds.

    Returns:
        ``MetricResult`` with ``value = 1 - DSR`` and rich metadata
        (DSR probability, observed Sharpe, expected max under null,
        skew, kurtosis, sample size, n_trials).
    """
    try:
        returns = np.asarray(returns, dtype=float).ravel()
        returns = returns[~np.isnan(returns)]
        T = int(returns.size)
        if T < 30:
            return create_error_metric(
                "deflated_sharpe",
                f"need >= 30 observations after dropping NaNs, got {T}",
                "backtest",
            )
        if n_trials < 1:
            return create_error_metric(
                "deflated_sharpe",
                f"n_trials must be >= 1, got {n_trials}",
                "backtest",
            )

        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        # Reject near-degenerate series. The threshold is relative to
        # |mu| so that a strategy with very small but real volatility
        # still passes, while a constant series (numerical-noise std)
        # fails loudly. With abs(mu) + 1 in the denominator we also
        # catch the pure-zero-mean constant case.
        if sigma <= 1e-12 * (abs(mu) + 1.0):
            return create_error_metric(
                "deflated_sharpe",
                f"return std too small ({sigma:.2e}) — series is nearly constant",
                "backtest",
            )

        sr_hat = mu / sigma  # per-period Sharpe
        # Use FULL kurtosis (gamma_4), not excess, per the MLDP 2014 formula.
        skew = float(stats.skew(returns, bias=False))
        full_kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

        # Variance of SR estimator (Mertens 2002 / Lo 2002 generalised):
        #   sigma^2(SR_hat) = (1 - gamma_3 * SR + (gamma_4 - 1)/4 * SR^2) / (T - 1)
        sr_variance = (
            1.0 - skew * sr_hat + ((full_kurt - 1.0) / 4.0) * sr_hat * sr_hat
        ) / max(T - 1, 1)
        if sr_variance <= 0.0:
            return create_error_metric(
                "deflated_sharpe",
                f"non-positive SR variance estimate ({sr_variance:.4f}) — "
                "higher-moment correction broke; report skew/kurt and try again",
                "backtest",
            )
        sr_std = math.sqrt(sr_variance)

        # Expected max Sharpe under N IID trials with variance V across
        # trials. If V is not supplied, the single-estimator variance is
        # a reasonable proxy when trials are similar in scale.
        V = float(var_trials) if var_trials is not None else sr_variance
        expected_max_sr = _expected_max_sharpe_under_null(n_trials, V)

        # Probabilistic Sharpe Ratio formula generalised to the deflated
        # benchmark = max(sr_benchmark, expected_max_sr).
        benchmark = max(float(sr_benchmark), expected_max_sr)
        psr_arg = (sr_hat - benchmark) / sr_std
        dsr_prob = float(norm.cdf(psr_arg))

        # Lower-is-better: 1 - DSR (probability the strategy is *not* skill).
        value = 1.0 - dsr_prob
        # Clip away tiny numerical excursions outside [0, 1].
        value = float(min(1.0, max(0.0, value)))

        th = thresholds or BACKTEST_THRESHOLDS["deflated_sharpe"]
        quality, passed = quality_from_value(value, th)

        sr_hat_annual = sr_hat * annualization_factor
        return MetricResult(
            name="deflated_sharpe",
            value=value,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="backtest",
            interpretation=(
                f"DSR = {dsr_prob:.4f} (1-DSR = {value:.4f}); "
                f"SR_hat = {sr_hat_annual:.3f} (annualized), "
                f"N_trials = {n_trials}, E[max SR | null] = {expected_max_sr:.4f} "
                f"per-period; sample size T = {T}"
            ),
            metadata={
                "dsr_probability": dsr_prob,
                "sharpe_hat": sr_hat,
                "sharpe_hat_annualized": sr_hat_annual,
                "expected_max_sharpe_under_null": expected_max_sr,
                "sr_benchmark_effective": benchmark,
                "skew": skew,
                "kurtosis_full": full_kurt,
                "kurtosis_excess": full_kurt - 3.0,
                "sr_std": sr_std,
                "n_observations": T,
                "n_trials": n_trials,
                "annualization_factor": annualization_factor,
            },
        )

    except Exception as e:
        logger.warning("deflated_sharpe failed: %s", e)
        return create_error_metric("deflated_sharpe", str(e), "backtest")


def compute_pbo(
    returns_matrix: np.ndarray,
    *,
    n_splits: int = 16,
    score_fn: str = "sharpe",
    thresholds: dict[str, float] | None = None,
) -> MetricResult:
    """Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2015).

    Computed via Combinatorially Symmetric Cross-Validation (CSCV):

      1. Split the ``T`` time periods into ``n_splits`` equal blocks.
      2. For every combination of ``n_splits / 2`` blocks (the "train"
         set), compute each strategy's score (Sharpe by default) on
         the chosen blocks (in-sample / IS) and the remaining blocks
         (out-of-sample / OOS).
      3. Pick the best strategy in IS; record the relative rank of its
         OOS score among all strategies' OOS scores.
      4. PBO = fraction of combinations where the IS-best strategy's
         OOS rank is below the median.

    A PBO close to 0 means the IS-best strategy is *also* OOS-best most
    of the time — i.e., the selection procedure is not overfitting. A
    PBO near or above 0.5 means the selection picks losers as often as
    winners — overfitting is severe.

    Args:
        returns_matrix: ``(T, N)`` array of strategy returns. ``T`` is
            time, ``N`` is the number of strategy variants explored.
            All variants should be evaluated over the same time index.
        n_splits: Number of equal-sized time blocks ``S``. Must be
            even. Default 16 (gives C(16,8) = 12,870 train/test
            combinations).
        score_fn: ``"sharpe"`` (default) or ``"mean"``. Sharpe is the
            standard choice in the literature.
        thresholds: Override default thresholds.

    Returns:
        ``MetricResult`` with ``value = PBO`` and rich metadata.

    Notes:
        - The cost is ``O(C(S, S/2) * N)`` in scoring (the two matrix
          reductions per combination are cheap). At ``S = 16, N <
          10_000`` this runs in seconds on a single CPU.
        - When two or more strategies tie on the IS score the first
          ``argmax`` is taken; this matches MLDP's published reference
          implementation but may give a tiny systematic bias when ties
          are systematic.
    """
    try:
        M = np.asarray(returns_matrix, dtype=float)
        if M.ndim != 2:
            return create_error_metric(
                "pbo", f"returns_matrix must be 2D (T, N), got shape {M.shape}", "backtest"
            )
        T, N = M.shape
        if N < 2:
            return create_error_metric(
                "pbo", f"need >= 2 strategies, got N={N}", "backtest"
            )
        if n_splits < 2 or n_splits % 2 != 0:
            return create_error_metric(
                "pbo",
                f"n_splits must be an even integer >= 2, got {n_splits}",
                "backtest",
            )
        rows_per_split = T // n_splits
        if rows_per_split < 2:
            return create_error_metric(
                "pbo",
                f"need >= {n_splits * 2} time rows for n_splits={n_splits}, got T={T}",
                "backtest",
            )
        if score_fn not in ("sharpe", "mean"):
            return create_error_metric(
                "pbo", f"score_fn must be 'sharpe' or 'mean', got {score_fn!r}", "backtest"
            )

        # Drop the incomplete tail so blocks are equal.
        M = M[: rows_per_split * n_splits]
        blocks = np.split(M, n_splits, axis=0)  # list of (rows_per_split, N)

        def _score(block_concat: np.ndarray) -> np.ndarray:
            if score_fn == "mean":
                return block_concat.mean(axis=0)
            # sharpe — guard against zero std
            mu = block_concat.mean(axis=0)
            sd = block_concat.std(axis=0, ddof=1)
            sd = np.where(sd > 1e-12, sd, np.inf)  # zero-vol strategies → -inf score
            return mu / sd

        half = n_splits // 2
        logits = np.empty(0, dtype=float)
        omegas = np.empty(0, dtype=float)
        logits_list: list[float] = []
        omegas_list: list[float] = []

        for is_combo in combinations(range(n_splits), half):
            is_set = set(is_combo)
            oos_combo = [i for i in range(n_splits) if i not in is_set]

            is_data = np.concatenate([blocks[i] for i in is_combo], axis=0)
            oos_data = np.concatenate([blocks[i] for i in oos_combo], axis=0)

            is_score = _score(is_data)
            oos_score = _score(oos_data)

            n_star = int(np.argmax(is_score))

            # OOS rank: position of strategy n_star among the N OOS
            # scores, sorted ascending. Rank 1 = worst, N = best.
            # We use the "fractional rank" rank/(N+1) so it lies in
            # (0, 1), avoiding logit blow-up at the extremes.
            rank_lt = int(np.sum(oos_score < oos_score[n_star]))
            rank_eq = int(np.sum(oos_score == oos_score[n_star]))
            # Average rank when ties (handles degenerate score functions
            # without bias).
            rank = rank_lt + 0.5 * (rank_eq + 1)
            omega = rank / (N + 1)
            omega = min(max(omega, 1.0 / (N + 2)), 1.0 - 1.0 / (N + 2))
            lam = math.log(omega / (1.0 - omega))
            logits_list.append(lam)
            omegas_list.append(omega)

        logits = np.asarray(logits_list, dtype=float)
        omegas = np.asarray(omegas_list, dtype=float)
        pbo = float(np.mean(logits < 0.0))

        th = thresholds or BACKTEST_THRESHOLDS["pbo"]
        quality, passed = quality_from_value(pbo, th)

        return MetricResult(
            name="pbo",
            value=pbo,
            quality=quality,
            passed=passed,
            thresholds=th,
            category="backtest",
            interpretation=(
                f"PBO = {pbo:.4f} from {len(logits)} CSCV combinations "
                f"(n_splits={n_splits}); median OOS-rank-ratio of IS-best "
                f"strategy = {float(np.median(omegas)):.3f} "
                f"(score = {score_fn})"
            ),
            metadata={
                "n_strategies": int(N),
                "n_time_obs": int(T),
                "n_splits": int(n_splits),
                "n_combinations": int(len(logits)),
                "mean_logit": float(np.mean(logits)),
                "median_oos_relative_rank": float(np.median(omegas)),
                "score_fn": score_fn,
            },
        )

    except Exception as e:
        logger.warning("pbo failed: %s", e)
        return create_error_metric("pbo", str(e), "backtest")
