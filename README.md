# finval

**Rigorous validation for synthetic financial time series.**

`finval` is a Python library for assessing the quality of synthetic
market data against real data. It was built because no existing library
covers the financial stylized facts that matter: fat tails, volatility
clustering, leverage effect, crash co-movement, and probabilistic
forecast calibration.

> ⚠️ **Alpha.** The API may still change. Feedback welcome.

## Why finval?

General-purpose synthetic data libraries (`sdmetrics`, `synthcity`,
`tsgm`) treat time series as generic sequences. They don't know what
"leverage effect" is, don't check PIT uniformity, and don't compute tail
dependence coefficients. For financial applications — risk management,
backtesting, derivatives — you need a suite that tests the things that
actually matter for market data.

`finval` ships 19 metrics across 5 categories, each with thresholds
calibrated against real financial data and justified by the statistical
literature.

## Installation

```bash
pip install finval
```

## Quickstart

```python
import numpy as np
import finval

# 2D data: (n_samples, n_features) returns
real = np.random.randn(1000, 3) * 0.01
synthetic = np.random.randn(1000, 3) * 0.01

report = finval.validate(synthetic, real)
print(report.summary())
print(f"Overall quality: {report.overall_quality}")
print(f"Pass rate: {report.pass_rate:.0%}")

# 3D data: (n_paths, horizon, n_features) for path-level validation
real_paths = np.random.randn(100, 60, 3) * 0.01
syn_paths = np.random.randn(100, 60, 3) * 0.01

report = finval.validate_paths(syn_paths, real_paths)
print(report.summary())
```

## Metrics

### Distribution (15% of overall score)
- **marginal_ks** — Kolmogorov-Smirnov test on each feature's marginal
- **energy_distance** — multivariate distribution difference
- **tail_quantiles** — 1st/5th/95th/99th percentile comparison (robust alternative to kurtosis)
- **tail_heaviness** — excess kurtosis error (diagnostic only)

### Dependence (25%)
- **pearson_corr** — linear correlation matrix error
- **spearman_corr** — rank correlation matrix error
- **copula_distance** — Cramér-von Mises distance between empirical copulas
- **tail_dependence_upper** — rally co-movement (λ_U)
- **tail_dependence_lower** — crash co-movement (λ_L)
- **correlation_breakdown** — stress vs calm regime correlation shift

### Temporal (20%)
- **acf_returns** — autocorrelation of returns (should be ~0)
- **volatility_clustering** — autocorrelation of squared returns
- **leverage_effect** — corr(r_t, |r_{t+k}|) (negative for equities)
- **cross_correlation** — contemporaneous cross-asset correlation

### Calibration (30%)
- **pit_uniformity** — KS test on probability integral transforms
- **crps** — continuous ranked probability score
- **coverage_50 / 90 / 95** — empirical vs nominal interval coverage

### Path-level (10%)
- **drawdown_distribution** — KS test on max drawdown distribution

## Baselines

Compare your model against simple reference generators to calibrate what
"good" means for your data:

```python
from finval.baselines import gaussian_baseline, historical_bootstrap, block_bootstrap

# Gaussian: matches mean+cov, no temporal structure
gauss = gaussian_baseline(real, n_samples=1000)

# i.i.d. bootstrap: matches joint distribution exactly, zero temporal
boot = historical_bootstrap(real, n_samples=1000)

# Block bootstrap: preserves short-range temporal structure
blocks = block_bootstrap(real, n_paths=100, path_length=60, block_size=20)

# Validate each
for name, syn in [("gaussian", gauss), ("iid", boot)]:
    r = finval.validate(syn, real)
    print(f"{name}: {r.overall_quality} ({r.overall_score:.0%})")
```

## Design principles

1. **Reliable over comprehensive.** Each metric is chosen because it's
   robust and informative, not because it's impressive.

2. **Mean over max for pairwise metrics.** Max over n(n-1)/2 feature
   pairs is dominated by sampling noise. `finval` uses mean error, which
   is harder to fool and more stable run-to-run.

3. **Lower is always better.** Every metric is normalized so that zero
   is perfect and higher is worse. No flipped signs to remember.

4. **Financial stylized facts first.** Leverage effect, vol clustering,
   fat tails, crash co-movement — these aren't optional for financial
   data.

5. **Proper scoring rules.** CRPS and PIT uniformity are proper scoring
   rules, not just rank-order checks. Your model is evaluated against
   the ground truth the statistics literature actually endorses.

## License

MIT
