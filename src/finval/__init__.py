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
"""

from finval.core.result import MetricResult, ValidationReport
from finval.validate import validate, validate_paths

__version__ = "0.1.0"

__all__ = [
    "MetricResult",
    "ValidationReport",
    "validate",
    "validate_paths",
    "__version__",
]
