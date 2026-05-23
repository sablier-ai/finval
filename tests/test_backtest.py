"""Tests for backtest-overfitting metrics + CPCV utility.

Reference checks (so future refactors are caught):

  * DSR is monotone-decreasing in n_trials when SR is held fixed.
  * DSR -> ~ 0.5 when the observed Sharpe equals the expected max
    under the null (i.e., the strategy is exactly at the false-strategy
    expectation).
  * Fat-tailed strategies are penalised: t-distributed returns with
    matched mean/std and the same nominal Sharpe produce a lower DSR
    than Gaussian returns.
  * PBO ~ 0.5 for pure-noise strategies (no real differentiation).
  * PBO ~ 0 when there is a dominant strategy IS and OOS.
  * PBO ~ 1 for the pathological "rank-inversion" case (IS winner is
    OOS loser).
  * CPCV split counts match C(n_splits, n_test_splits); train/test are
    disjoint; embargo creates the requested gap.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from finval.core.cpcv import iter_cpcv_splits, make_cpcv_splits, n_cpcv_paths
from finval.metrics.backtest import compute_deflated_sharpe, compute_pbo


# -------------------------------------------------------------------------
# Deflated Sharpe Ratio
# -------------------------------------------------------------------------


def _gaussian_returns(rng: np.random.Generator, T: int, sr_period: float) -> np.ndarray:
    """Return T Gaussian draws with per-period Sharpe approximately ``sr_period``.

    We fix sigma = 1 and set mu = sr_period * sigma so the SR estimator
    centres on sr_period in expectation. Tests use moderate T (~1000) so
    that finite-sample noise around the target SR is small.
    """
    return rng.standard_normal(T) + sr_period


def test_dsr_high_sharpe_single_trial_passes(rng):
    # A strategy with per-period Sharpe ~ 0.10 over T=1000 has annualized
    # Sharpe ~ 0.10 * sqrt(252) ~ 1.59 — strong by industry standards.
    # With n_trials=1 (no selection bias) DSR should be high (~ 1).
    returns = _gaussian_returns(rng, T=1000, sr_period=0.10)
    res = compute_deflated_sharpe(returns, n_trials=1)
    assert res.metadata["dsr_probability"] > 0.95
    assert res.value < 0.05  # 1 - DSR
    assert res.quality == "excellent"
    assert res.passed


def test_dsr_high_sharpe_many_trials_fails(rng):
    # Same nominal strategy, but discovered after 10_000 trials.
    # The expected max Sharpe under the null at 10_000 iid Normals
    # is around sqrt(2 log 10_000)/sqrt(T-1), which deflates the
    # observed Sharpe substantially.
    returns = _gaussian_returns(rng, T=1000, sr_period=0.10)
    res = compute_deflated_sharpe(returns, n_trials=10_000)
    # DSR should now be far from 1; the deflation should push value up.
    assert res.metadata["dsr_probability"] < 0.95
    assert res.value > 0.05
    # Sanity: expected max under null is positive
    assert res.metadata["expected_max_sharpe_under_null"] > 0


def test_dsr_monotone_in_n_trials(rng):
    # Holding the strategy fixed, DSR should be a non-increasing
    # function of n_trials (more trials -> more deflation).
    returns = _gaussian_returns(rng, T=1000, sr_period=0.10)
    probs = []
    for n in (1, 10, 100, 1000, 10_000):
        res = compute_deflated_sharpe(returns, n_trials=n)
        probs.append(res.metadata["dsr_probability"])
    # Strictly non-increasing
    for a, b in zip(probs, probs[1:]):
        assert b <= a + 1e-9, f"DSR not monotone: probs={probs}"


def test_dsr_zero_sharpe_around_half(rng):
    # A strategy with zero edge: SR_hat ~ 0. With n_trials=1, the
    # benchmark is 0 and the deflation argument is ~ 0, so DSR ~ 0.5.
    # We average over a small ensemble to reduce one-off sampling bias.
    probs = []
    for seed in range(20):
        rng_local = np.random.default_rng(seed)
        returns = _gaussian_returns(rng_local, T=2000, sr_period=0.0)
        res = compute_deflated_sharpe(returns, n_trials=1)
        probs.append(res.metadata["dsr_probability"])
    assert 0.35 < np.mean(probs) < 0.65


def test_dsr_fat_tails_penalised():
    # Two strategies with identical empirical SR but different higher
    # moments. The t_5 version should have a *lower* DSR than the
    # Gaussian one.
    rng = np.random.default_rng(7)
    T = 2000
    sr_target = 0.08

    # Gaussian
    gauss = rng.standard_normal(T) + sr_target
    # t_5 — scale to match unit variance, then add target mean
    t5 = rng.standard_t(df=5, size=T) / np.sqrt(5 / 3)  # unit variance
    t5 = t5 + sr_target

    # Match mean/std exactly so both have the same empirical SR
    gauss_sr = gauss.mean() / gauss.std(ddof=1)
    t5_sr = t5.mean() / t5.std(ddof=1)
    # Rescale t5 to match gauss_sr exactly (cheat is fine for the test;
    # we only care that DSR sees the higher-moment penalty)
    t5_adj = t5 * (gauss_sr / max(t5_sr, 1e-12))
    t5_adj = (
        t5_adj - t5_adj.mean()
        + gauss_sr * t5_adj.std(ddof=1)
    )

    res_gauss = compute_deflated_sharpe(gauss, n_trials=100)
    res_t5 = compute_deflated_sharpe(t5_adj, n_trials=100)

    # Same nominal SR (within float tolerance)
    assert abs(res_gauss.metadata["sharpe_hat"] - res_t5.metadata["sharpe_hat"]) < 1e-2
    # Fat-tailed strategy must have higher SR_std (more uncertainty)
    assert res_t5.metadata["sr_std"] > res_gauss.metadata["sr_std"]
    # Higher SR_std => smaller PSR argument => lower DSR probability
    assert (
        res_t5.metadata["dsr_probability"]
        < res_gauss.metadata["dsr_probability"] + 1e-6
    )


def test_dsr_short_series_returns_error():
    # < 30 observations should fail loudly.
    res = compute_deflated_sharpe(np.zeros(20) + 0.01, n_trials=10)
    assert res.quality == "poor"
    assert not res.passed
    assert "30" in res.interpretation


def test_dsr_zero_variance_returns_error():
    res = compute_deflated_sharpe(np.ones(100) * 0.001, n_trials=1)
    assert res.quality == "poor"


def test_dsr_serialises_to_dict(rng):
    res = compute_deflated_sharpe(_gaussian_returns(rng, 500, 0.05), n_trials=10)
    d = res.to_dict()
    # spot-check structure
    assert d["category"] == "backtest"
    assert "metadata" in d
    assert "dsr_probability" in d["metadata"]


# -------------------------------------------------------------------------
# Probability of Backtest Overfitting
# -------------------------------------------------------------------------


def test_pbo_pure_noise_is_about_half():
    # 30 strategies, 1000 time steps, pure iid Gaussian noise per
    # strategy. No real edge -> the IS-best is just lucky -> PBO ~ 0.5.
    rng = np.random.default_rng(0)
    M = rng.standard_normal((1000, 30))
    res = compute_pbo(M, n_splits=10)  # C(10,5)=252 combinations
    # Concentration band around 0.5; needs to be robust across re-seeds.
    assert 0.35 < res.value < 0.65, f"pure noise PBO={res.value}"


def test_pbo_dominant_strategy_near_zero():
    # One strategy with a real, persistent edge — the rest are noise.
    rng = np.random.default_rng(1)
    T, N = 1000, 30
    M = rng.standard_normal((T, N)) * 0.01  # daily returns ~ 1%
    M[:, 0] += 0.005  # +0.5% per period real edge on strategy 0
    res = compute_pbo(M, n_splits=10)
    # IS-best will be strategy 0 essentially every split, and it will
    # also be OOS-best — PBO should be very small.
    assert res.value < 0.10, f"dominant-strategy PBO={res.value}"
    assert res.metadata["median_oos_relative_rank"] > 0.85


def test_pbo_rank_inversion_near_one():
    # Construct a pathological case: each strategy is positive on
    # half the time and negative on the other half. Splitting in two
    # contiguous halves makes the IS winner = OOS loser exactly.
    T, N = 200, 8
    M = np.zeros((T, N))
    for n in range(N):
        # Strategy n is positive on the first half if n is even,
        # negative on the first half if n is odd, and the opposite
        # on the second half. With n_splits=2 and IS={block 0}, the
        # IS winner is whichever 'even' strategy has the highest
        # mean on the first half — that same strategy is the worst
        # on the second half (its sign flips).
        first_half = (-1.0 if n % 2 else 1.0) * (1.0 + 0.01 * n)
        second_half = -first_half
        M[: T // 2, n] = first_half
        M[T // 2 :, n] = second_half
        # add tiny random noise so argmax is well-defined
    M += np.random.default_rng(123).standard_normal(M.shape) * 1e-6
    res = compute_pbo(M, n_splits=2)
    # n_splits=2 has only C(2,1)=2 combinations, both perfectly
    # symmetric — PBO should be exactly 1.
    assert res.value == pytest.approx(1.0)


def test_pbo_too_few_strategies_errors():
    res = compute_pbo(np.zeros((1000, 1)))
    assert not res.passed
    assert res.quality == "poor"


def test_pbo_odd_n_splits_errors():
    res = compute_pbo(np.zeros((1000, 5)), n_splits=7)
    assert not res.passed
    assert "even" in res.interpretation


def test_pbo_too_few_time_obs_errors():
    res = compute_pbo(np.zeros((10, 5)), n_splits=16)
    assert not res.passed


# -------------------------------------------------------------------------
# CPCV splits
# -------------------------------------------------------------------------


def test_cpcv_split_count_matches_combinatorial():
    splits = make_cpcv_splits(n_samples=1000, n_splits=10, n_test_splits=2)
    assert len(splits) == comb(10, 2) == 45


def test_cpcv_train_and_test_are_disjoint():
    splits = make_cpcv_splits(n_samples=500, n_splits=8, n_test_splits=2)
    for train_idx, test_idx in splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(test_idx) > 0
        assert len(train_idx) > 0


def test_cpcv_with_zero_embargo_covers_all_indices():
    n = 600
    splits = make_cpcv_splits(n_samples=n, n_splits=10, n_test_splits=2, embargo=0)
    # Union of train + test in each split is all indices
    for train_idx, test_idx in splits:
        u = np.union1d(train_idx, test_idx)
        assert len(u) == n
        assert u[0] == 0 and u[-1] == n - 1


def test_cpcv_embargo_creates_gap():
    # With embargo=5 the train indices closest to each test block
    # should be at least 5 positions away from the test boundary.
    n = 1000
    embargo = 7
    splits = make_cpcv_splits(n_samples=n, n_splits=10, n_test_splits=2, embargo=embargo)
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        # For every train index t and adjacent test boundary, the gap
        # should be > embargo on at least one side.
        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())
        # Sample a few test boundary positions and check the band
        for boundary in (test_idx[0], test_idx[-1]):
            band = range(
                max(0, int(boundary) - embargo + 1),
                min(n, int(boundary) + embargo),
            )
            # The embargo band can contain test indices (the test block
            # itself overlaps) but it must NOT contain any train index.
            for b in band:
                if b in test_set:
                    continue
                assert b not in train_set, (
                    f"train index {b} within embargo={embargo} of test boundary {boundary}"
                )


def test_cpcv_path_count_formula():
    # Each observation appears in exactly C(n_splits-1, n_test_splits-1)
    # test folds. Use the helper and verify by direct count.
    n, S, k = 200, 8, 3
    expected_per_obs = n_cpcv_paths(S, k)
    splits = make_cpcv_splits(n, S, k)
    appearance = np.zeros(n, dtype=int)
    for _train_idx, test_idx in splits:
        appearance[test_idx] += 1
    # All observations should appear the same number of times when
    # partitions are equal-sized; here n%S=0 so they are exactly equal.
    assert np.all(appearance == expected_per_obs)


def test_cpcv_invalid_arguments_raise():
    with pytest.raises(ValueError):
        make_cpcv_splits(100, n_splits=1, n_test_splits=1)  # n_splits too small
    with pytest.raises(ValueError):
        make_cpcv_splits(100, n_splits=10, n_test_splits=10)  # test == splits
    with pytest.raises(ValueError):
        make_cpcv_splits(5, n_splits=10, n_test_splits=2)  # n < n_splits
    with pytest.raises(ValueError):
        make_cpcv_splits(100, n_splits=10, n_test_splits=2, embargo=-1)


def test_iter_cpcv_splits_equivalent_to_make_cpcv_splits():
    a = list(iter_cpcv_splits(200, 6, 2))
    b = make_cpcv_splits(200, 6, 2)
    assert len(a) == len(b)
    for (ta, tea), (tb, teb) in zip(a, b):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(tea, teb)
