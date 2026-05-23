"""finval — rigorous validation for synthetic financial time series.

Quick start:

    import finval

    # synthetic and real are (n_samples, n_features) arrays of returns
    report = finval.validate(synthetic, real)
    print(report.summary())
    report.to_dict()

For path-level validation with drawdowns and calibration:

    # paths are (n_paths, horizon, n_features) arrays
    report = finval.validate_paths(synthetic_paths, real_returns)

For backtest-overfitting checks (used independently of synthetic data):

    from finval import compute_deflated_sharpe, compute_pbo, make_cpcv_splits

    # DSR of a single strategy after K trials
    res = compute_deflated_sharpe(strategy_returns, n_trials=K)

    # PBO from a (T, N) matrix of N strategy variants' returns
    res = compute_pbo(returns_matrix, n_splits=16)

    # CPCV splits for purged walk-forward
    for train_idx, test_idx in make_cpcv_splits(T, n_splits=10, n_test_splits=2, embargo=5):
        ...
"""

from finval.core.cpcv import iter_cpcv_splits, make_cpcv_splits, n_cpcv_paths
from finval.core.result import MetricResult, ValidationReport
from finval.metrics.backtest import compute_deflated_sharpe, compute_pbo
from finval.validate import validate, validate_paths

__version__ = "0.1.0"

__all__ = [
    "MetricResult",
    "ValidationReport",
    "validate",
    "validate_paths",
    # Backtest-overfitting
    "compute_deflated_sharpe",
    "compute_pbo",
    "make_cpcv_splits",
    "iter_cpcv_splits",
    "n_cpcv_paths",
    "__version__",
]
