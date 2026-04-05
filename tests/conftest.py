"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def real_returns_2d(rng: np.random.Generator) -> np.ndarray:
    """Realistic fat-tailed 3-feature return series with correlation."""
    n = 2000
    x = rng.standard_t(df=5, size=(n, 3)) * 0.01
    x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]  # induce correlation
    return x


@pytest.fixture
def matched_synthetic_2d(rng: np.random.Generator) -> np.ndarray:
    """Synthetic with same structure as real_returns_2d — should pass all metrics."""
    n = 2000
    x = rng.standard_t(df=5, size=(n, 3)) * 0.01
    x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
    return x


@pytest.fixture
def real_paths_3d(rng: np.random.Generator) -> np.ndarray:
    """Path-level return data: 200 paths × 60 steps × 3 features."""
    return rng.standard_t(df=5, size=(200, 60, 3)) * 0.01


@pytest.fixture
def matched_synthetic_paths_3d(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_t(df=5, size=(200, 60, 3)) * 0.01
