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

from finval.core.result import FullReport, MetricResult, ValidationReport
from finval.metrics.generative import validate_generative
from finval.validate import (
    validate,
    validate_against_baselines,
    validate_calibration,
    validate_conditional,
    validate_full,
    validate_paths,
)

__version__ = "0.4.0"

__all__ = [
    "MetricResult",
    "ValidationReport",
    "FullReport",
    "validate",
    "validate_paths",
    "validate_calibration",
    "validate_conditional",
    "validate_generative",
    "validate_against_baselines",
    "validate_full",
    "__version__",
]
