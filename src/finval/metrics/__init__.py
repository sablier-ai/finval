"""Validation metrics for synthetic financial time series.

Most metrics are pure functions of (synthetic, real) arrays and return
a MetricResult. The ``backtest`` module is the exception — its inputs
are strategy returns / IS-OOS matrices, since backtest-overfitting is
a property of the *selection procedure*, not the synthetic generator.

Categories:

- distribution: marginal and joint distribution fidelity
- dependence: cross-feature dependence structure (including tails)
- temporal: autocorrelation and volatility dynamics
- calibration: probabilistic forecast calibration
- paths: path-level properties (drawdowns, etc.)
- backtest: deflated Sharpe + probability of backtest overfitting
"""

from finval.metrics.backtest import compute_deflated_sharpe, compute_pbo
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
    compute_hill_tail_index,
    compute_marginal_ks,
    compute_sliced_wasserstein,
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
    "compute_hill_tail_index",
    "compute_sliced_wasserstein",
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
    # Backtest
    "compute_deflated_sharpe",
    "compute_pbo",
]
