"""Combinatorial Purged Cross-Validation (CPCV) splits.

CPCV (Lopez de Prado, *Advances in Financial Machine Learning*, Ch. 12)
generates train/test index splits that respect the temporal structure of
financial data:

  - Time is partitioned into ``n_splits`` contiguous groups.
  - Every combination of ``n_test_splits`` groups (out of the total) is
    designated, in turn, as the test set; the remaining groups form the
    training set.
  - An ``embargo`` band on either side of each test group is excluded
    from training to remove leakage through serial dependence near the
    boundary.

Each observation therefore appears in exactly ``C(n_splits-1,
n_test_splits-1)`` distinct test folds and ``C(n_splits-1,
n_test_splits)`` distinct training folds — producing many backtest
"paths" through history rather than a single in/out split. This is the
input shape consumed by ``compute_pbo`` (Probability of Backtest
Overfitting).

The implementation here returns explicit ``(train_idx, test_idx)``
``np.ndarray`` pairs so it can be consumed by any backtester (no
assumptions about labels, classifiers, or the strategy code itself).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterator

import numpy as np


def make_cpcv_splits(
    n_samples: int,
    n_splits: int = 10,
    n_test_splits: int = 2,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate all CPCV train/test index splits.

    Args:
        n_samples: Total number of time-ordered observations.
        n_splits: Number of contiguous partitions of the timeline (>= 2).
        n_test_splits: Number of partitions used for test in each fold
            (1 <= n_test_splits < n_splits). For the standard MLDP
            "CPCV(N,k)" notation, this is ``k``.
        embargo: Number of observations to drop from training on either
            side of each test partition. Use this to remove
            autocorrelation leakage at the train/test boundary.

    Returns:
        A list of ``(train_idx, test_idx)`` tuples. Length is
        ``C(n_splits, n_test_splits)``. Each ``train_idx`` is a sorted
        array of ints; ``test_idx`` is the concatenation of the test
        partitions in time order.

    Raises:
        ValueError: If parameters are inconsistent (e.g. partitions
            larger than ``n_samples``, or ``n_test_splits`` outside
            ``[1, n_splits - 1]``).

    Notes:
        - Partitions are as equal as possible. The first
          ``n_samples % n_splits`` partitions get one extra observation.
        - With embargo > 0, the training set near each test partition
          shrinks; on very tight schedules this may produce empty
          training sets — the caller should filter.
        - No purging by label here (no label horizon argument); the
          caller is responsible for choosing an embargo that covers the
          relevant decay of serial dependence. For daily equity returns,
          5-10 observations is a typical choice.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not (1 <= n_test_splits < n_splits):
        raise ValueError(
            f"n_test_splits must satisfy 1 <= n_test_splits < n_splits; "
            f"got n_test_splits={n_test_splits}, n_splits={n_splits}"
        )
    if n_samples < n_splits:
        raise ValueError(
            f"n_samples ({n_samples}) must be >= n_splits ({n_splits})"
        )
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    # Partition n_samples into n_splits as equal as possible.
    base = n_samples // n_splits
    extra = n_samples % n_splits
    sizes = [base + (1 if i < extra else 0) for i in range(n_splits)]
    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_combo in combinations(range(n_splits), n_test_splits):
        # Test indices: concatenation of test partitions in time order.
        test_idx = np.concatenate(
            [np.arange(starts[i], ends[i]) for i in test_combo]
        ).astype(np.int64)

        # Train mask: everything not in test, minus embargo around each
        # test partition. We embargo on BOTH sides because returns
        # autocorrelate symmetrically around the boundary at the lags
        # the embargo is designed to cover.
        train_mask = np.ones(n_samples, dtype=bool)
        for i in test_combo:
            lo = max(0, starts[i] - embargo)
            hi = min(n_samples, ends[i] + embargo)
            train_mask[lo:hi] = False
        train_idx = np.flatnonzero(train_mask).astype(np.int64)

        splits.append((train_idx, test_idx))

    return splits


def n_cpcv_paths(n_splits: int, n_test_splits: int) -> int:
    """Number of distinct backtest paths CPCV produces.

    Each observation appears in ``C(n_splits-1, n_test_splits-1)`` test
    folds. The total number of paths through history is therefore
    ``C(n_splits, n_test_splits) * n_test_splits / n_splits``.
    """
    if n_splits < 2 or not (1 <= n_test_splits < n_splits):
        return 0
    return comb(n_splits - 1, n_test_splits - 1)


def iter_cpcv_splits(
    n_samples: int,
    n_splits: int = 10,
    n_test_splits: int = 2,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Iterator variant of ``make_cpcv_splits`` — yields one split at a time.

    Useful when ``C(n_splits, n_test_splits)`` is large and you do not
    want to materialise all splits in memory at once (each pair is
    ``O(n_samples)`` ints).
    """
    for split in make_cpcv_splits(n_samples, n_splits, n_test_splits, embargo):
        yield split
