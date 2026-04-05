"""Validation metrics for synthetic financial time series.

Each metric is a pure function of (synthetic, real) arrays that returns
a MetricResult. Metrics are organized by what they measure:

- distribution: marginal and joint distribution fidelity
- dependence: cross-feature dependence structure (including tails)
- temporal: autocorrelation and volatility dynamics
- calibration: probabilistic forecast calibration
- paths: path-level properties (drawdowns, etc.)
"""

from finval.metrics.calibration import (
    compute_coverage,
    compute_crps,
    compute_pit_uniformity,
)
from finval.metrics.dependence import (
    compute_copula_distance,
    compute_correlation_breakdown,
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_tail_dependence,
)
from finval.metrics.distribution import (
    compute_energy_distance,
    compute_marginal_ks,
    compute_tail_heaviness,
    compute_tail_quantiles,
)
from finval.metrics.paths import compute_drawdown_distribution
from finval.metrics.temporal import (
    compute_acf_returns,
    compute_cross_correlation,
    compute_leverage_effect,
    compute_volatility_clustering,
)

__all__ = [
    # Distribution
    "compute_marginal_ks",
    "compute_energy_distance",
    "compute_tail_quantiles",
    "compute_tail_heaviness",
    # Dependence
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    "compute_copula_distance",
    "compute_tail_dependence",
    "compute_correlation_breakdown",
    # Temporal
    "compute_acf_returns",
    "compute_volatility_clustering",
    "compute_leverage_effect",
    "compute_cross_correlation",
    # Calibration
    "compute_pit_uniformity",
    "compute_crps",
    "compute_coverage",
    # Paths
    "compute_drawdown_distribution",
]
