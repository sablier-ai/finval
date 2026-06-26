"""Tests for the v0.5.0 opt-in sub-window validation + per-metric effective_n.

The single most important property is the **byte-identical default**: passing
``subwindow=None`` (or omitting it entirely) must reproduce the v0.4.0 scores
exactly — finval is the FinBench leaderboard backend and existing scores must
not move when the new param is unused.
"""

from __future__ import annotations

import numpy as np
import pytest

import finval
from finval.validate import FULL_HORIZON_METRICS, PATH_METRICS


@pytest.fixture
def long_real_paths(rng: np.random.Generator) -> np.ndarray:
    """Long-horizon path data: 60 paths × 120 steps × 3 features (returns)."""
    return rng.standard_t(df=5, size=(60, 120, 3)) * 0.01


@pytest.fixture
def long_syn_paths(rng: np.random.Generator) -> np.ndarray:
    """Matched long-horizon synthetic paths."""
    return rng.standard_t(df=5, size=(60, 120, 3)) * 0.01


# --- (a) Reproducibility: subwindow=None is byte-identical to omitting the param -----------
def test_subwindow_none_is_byte_identical_to_default(long_real_paths, long_syn_paths):
    """The whole point: subwindow=None must reproduce the default run EXACTLY.

    Same overall_score and same per-metric scores/values — proves existing
    FinBench leaderboard numbers do not move when the new param is unused.
    """
    base = finval.validate_paths(long_syn_paths, long_real_paths)
    none = finval.validate_paths(long_syn_paths, long_real_paths, subwindow=None)

    # Same set of metrics, identical overall score.
    assert set(base.metrics) == set(none.metrics)
    assert base.overall_score == none.overall_score
    assert base.overall_quality == none.overall_quality

    # Identical per-metric value AND score (the numbers FinBench publishes).
    for name, m in base.metrics.items():
        m2 = none.metrics[name]
        assert m.value == m2.value, f"{name} value moved"
        assert m.score == m2.score, f"{name} score moved"
        assert m.quality == m2.quality, f"{name} quality moved"
        assert m.passed == m2.passed, f"{name} passed moved"


def test_subwindow_larger_than_horizon_is_noop(long_real_paths, long_syn_paths):
    """subwindow >= path_length cannot partition → falls back to the default run."""
    base = finval.validate_paths(long_syn_paths, long_real_paths)
    # H = 120; a window >= H leaves nothing to sub-window.
    big = finval.validate_paths(long_syn_paths, long_real_paths, subwindow=200)
    assert base.overall_score == big.overall_score
    for name, m in base.metrics.items():
        assert m.value == big.metrics[name].value


# --- (b) Sub-window power: sub-windowed metrics report higher effective_n ------------------
def test_subwindow_lifts_effective_n_for_subwindowed_metrics(long_real_paths, long_syn_paths):
    """A sub-windowed path metric is computed on ~ (H // W)x more real windows."""
    n_paths, H = long_real_paths.shape[0], long_real_paths.shape[1]
    W = 30
    nb = H // W  # 4 non-overlapping windows per path

    base = finval.validate_paths(long_syn_paths, long_real_paths)
    sub = finval.validate_paths(long_syn_paths, long_real_paths, subwindow=W)

    # volatility_clustering is short-scale → sub-windowed.
    assert "volatility_clustering" in sub.metrics
    assert base.metrics["volatility_clustering"].effective_n == n_paths
    assert sub.metrics["volatility_clustering"].effective_n == n_paths * nb
    # The power gain is exactly the (H // W) factor.
    assert sub.metrics["volatility_clustering"].effective_n == base.metrics[
        "volatility_clustering"
    ].effective_n * nb


# --- (c) Full-horizon metrics preserved under sub-window mode ------------------------------
def test_full_horizon_metrics_stay_on_full_paths(long_real_paths, long_syn_paths):
    """FULL_HORIZON_METRICS run on the full paths (effective_n == n_paths), and
    long_memory is applicable at the full horizon but NOT on a short sub-window."""
    n_paths = long_real_paths.shape[0]
    W = 20  # < long_memory's smallest lag (20) - 2 → long_memory dies on sub-windows

    sub = finval.validate_paths(long_syn_paths, long_real_paths, subwindow=W)

    # long_memory is full-horizon: present, applicable, finite, scored on full paths.
    lm = sub.metrics["long_memory"]
    assert lm.effective_n == n_paths
    assert lm.applicable
    assert np.isfinite(lm.value)

    # Every full-horizon metric that ran keeps effective_n == n_paths (full paths).
    for name in FULL_HORIZON_METRICS:
        if name in sub.metrics:
            assert sub.metrics[name].effective_n == n_paths, name

    # Sanity: on a W=20 sub-window long_memory would be non-applicable (lags too long),
    # which is exactly why it must stay on the full paths.
    from finval.metrics.tail_dynamics import compute_long_memory
    from finval.validate import _subwindow

    sub_real = _subwindow(long_real_paths, W)
    sub_syn = _subwindow(long_syn_paths, W)
    lm_on_window = compute_long_memory(sub_syn, sub_real)
    assert not np.isfinite(lm_on_window.value)  # horizon too short for its long lags


# --- (d) effective_n present everywhere, with the correct count ----------------------------
def test_effective_n_present_in_default_mode(long_real_paths, long_syn_paths):
    """Every metric exposes effective_n in the default run with the right count."""
    n_paths = long_real_paths.shape[0]
    flat_rows = long_real_paths.shape[0] * long_real_paths.shape[1]

    rep = finval.validate_paths(long_syn_paths, long_real_paths)
    for name, m in rep.metrics.items():
        assert m.effective_n is not None, name
        if name in PATH_METRICS:
            assert m.effective_n == n_paths, name
        else:  # flat metric on the reshaped rows
            assert m.effective_n == flat_rows, name


def test_effective_n_present_in_subwindow_mode(long_real_paths, long_syn_paths):
    """Every metric exposes the correct effective_n under subwindow=W."""
    n_paths, H, _ = long_real_paths.shape
    W = 30
    nb = H // W
    flat_rows = n_paths * H  # flatten is invariant to sub-windowing

    rep = finval.validate_paths(long_syn_paths, long_real_paths, subwindow=W)
    for name, m in rep.metrics.items():
        assert m.effective_n is not None, name
        if name in FULL_HORIZON_METRICS:
            assert m.effective_n == n_paths, name
        elif name in PATH_METRICS:
            assert m.effective_n == n_paths * nb, name
        else:  # flat metric
            assert m.effective_n == flat_rows, name


def test_effective_n_present_in_flat_validate(real_returns_2d, matched_synthetic_2d):
    """validate() (2D flat) stamps effective_n = real row count on every metric."""
    rep = finval.validate(matched_synthetic_2d, real_returns_2d)
    n = real_returns_2d.shape[0]
    for name, m in rep.metrics.items():
        assert m.effective_n == n, name


def test_effective_n_serialized_in_to_dict(long_real_paths, long_syn_paths):
    """effective_n round-trips through MetricResult.to_dict()."""
    rep = finval.validate_paths(long_syn_paths, long_real_paths, subwindow=30)
    d = rep.to_dict()
    for name, md in d["metrics"].items():
        assert "effective_n" in md, name


def test_validate_full_forwards_subwindow(long_real_paths, long_syn_paths):
    """validate_full forwards subwindow to the pooled path run; default stays byte-identical."""
    base = finval.validate_full(long_syn_paths, long_real_paths, generative=False)
    none = finval.validate_full(
        long_syn_paths, long_real_paths, generative=False, subwindow=None
    )
    assert base.overall_score == none.overall_score

    sub = finval.validate_full(
        long_syn_paths, long_real_paths, generative=False, subwindow=30
    )
    nb = long_real_paths.shape[1] // 30
    # The sub-windowed pooled metric carries the lifted effective_n into the FullReport.
    assert sub.metrics["volatility_clustering"].effective_n == long_real_paths.shape[0] * nb
