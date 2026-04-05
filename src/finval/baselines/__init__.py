"""Baseline generators for benchmarking synthetic financial data.

A baseline is a simple generator that produces synthetic data from real
data. Baselines are not meant to be good models — they're meant to be
reference points so users can answer "is my fancy model actually better
than X?" where X is a trivially simple generator.

Three baselines ship with finval:

- `gaussian`: Multivariate Gaussian with the empirical mean and covariance
  of the real data. No temporal structure, no tails, no vol clustering.
  This is the minimum bar any generative model should clear.

- `historical_bootstrap`: Random sampling (with replacement) from real
  returns. Reproduces marginals and joint distribution exactly in
  expectation, but destroys all temporal structure (no ACF, no leverage).

- `block_bootstrap`: Moving-block bootstrap preserves local dependence.
  This is the strongest simple baseline — it reproduces short-range
  temporal structure and most stylized facts at the cost of exact
  marginal duplication. A generative model that beats block bootstrap
  on all metrics is genuinely doing something new.
"""

from finval.baselines.gaussian import gaussian_baseline
from finval.baselines.historical import block_bootstrap, historical_bootstrap

__all__ = [
    "gaussian_baseline",
    "historical_bootstrap",
    "block_bootstrap",
]
