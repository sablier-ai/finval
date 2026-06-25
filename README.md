# finval

**Rigorous validation for synthetic financial time series.**

`finval` is a Python library for assessing the quality of synthetic
market data against real data. It was built because no existing library
covers the financial stylized facts that matter: fat tails, volatility
clustering, leverage effect, crash co-movement, and probabilistic
forecast calibration.

`finval` is the scoring backend behind
**[FinBench](https://github.com/sablier-ai/finbench)**, the public
leaderboard for multivariate financial time-series generation.

> **Current release: `0.4.0`** (`0.2.0`/`0.3.0` preserved at their tags for
> reproducibility). 0.4.0 reorganizes scoring into **6 weighted lenses** —
> marginal (0.15), dependence (0.20), temporal (0.13), joint (0.10),
> **conditional (0.22)** and **generative (0.20)** — plus **7 hard gates** that
> fail a model outright regardless of the weighted score (`memorization`,
> `tail_quantiles`, `tail_dependence_lower`, `drawdown_distribution`,
> `conditional_sensitivity`, `c2st`, `coverage_deficit`). New in 0.4.0: a
> regime-stratified **`conditional_sensitivity`** axis (catches a
> calibrated-but-*climatological* generator whose forecast barely moves across
> regimes), a **generative** lens (Naeem density/coverage scored as the delta vs
> a block-bootstrap *replay* of the real data — a generator must beat replay to
> justify itself), a **joint** C2ST omnibus, and **graceful degradation**
> (metrics flagged non-`applicable` on thin-data / long-horizon inputs rather
> than scored as failures). The library is in active use (FinBench v2 production
> scoring); pin an exact version if you need score-stability across releases.

## Why finval?

General-purpose synthetic data libraries (`sdmetrics`, `synthcity`,
`tsgm`) treat time series as generic sequences. They don't know what
"leverage effect" is, don't check PIT uniformity, and don't compute tail
dependence coefficients. For financial applications — risk management,
backtesting, derivatives — you need a suite that tests the things that
actually matter for market data.

`finval` ships **26 scored metrics across 6 weighted lenses** (marginal,
dependence, temporal, joint, conditional, generative) **+ 7 hard gates +
diagnostic localizers**, each with thresholds calibrated against real financial
data and justified by the statistical literature. They split across three entry
points by input shape:

- **11 flat metrics** — 3 distributional (`marginal_ks`,
  `energy_distance`, `tail_quantiles`) and 8 dependence (`pearson_corr`,
  `spearman_corr`, `copula_distance`, `tail_dependence_upper`,
  `tail_dependence_lower`, `correlation_breakdown`,
  `tail_dependence_asymmetry`, `covariance_calibration`) — run by
  `validate(...)` on 2D flat data.
- **7 path-level metrics** (`acf_returns`, `volatility_clustering`,
  `leverage_effect`, `cross_correlation`, `drawdown_distribution`,
  `regime_conditional`, `memorization`) — run by `validate_paths(...)`
  on 3D sample paths, which also reshapes the paths and runs the 11
  flat metrics on them (so `validate_paths` produces **18** scores
  total).
- **5 calibration metrics** (`pit_uniformity`, `crps`, `coverage_50`,
  `coverage_90`, `coverage_95`) — run by `validate_calibration(...)` on
  per-observation forecast distributions paired with realized actuals.

Implementation note: the 23 weighted metrics come from **20 underlying
compute functions** producing **23 individual numeric outputs** —
`compute_tail_dependence` returns upper + lower (2 scores) and
`compute_coverage` returns the three levels (3 scores). Every output is
weighted into `overall_score`.

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

As of 0.4.0 the weighted score is a **six-lens** mean (weights below); a set of
**diagnostic localizers** is computed + reported at weight 0; and **7 hard gates**
fail a model outright. `validate_full(...)` runs every lens it has inputs for.

### Marginal (15%)
- **marginal_ks** — Kolmogorov-Smirnov test on each feature's marginal
- **energy_distance** — multivariate distribution difference
- **tail_quantiles** *(gate)* — 1st/5th/95th/99th percentile comparison (robust alternative to kurtosis)
- **tail_heaviness** — tail-index fidelity

### Dependence (20%)
- **pearson_corr** / **spearman_corr** — linear / rank correlation matrix error
- **copula_distance** — Cramér-von Mises distance between empirical copulas
- **tail_dependence_upper** (λ_U) / **tail_dependence_lower** *(gate)* (λ_L) — rally / crash co-movement
- **correlation_breakdown** — stress vs calm regime correlation shift
- **tail_dependence_asymmetry** — fidelity of the crash-vs-rally gap A = λ_L − λ_U (0 by construction for any elliptical/Gaussian model, so it catches what the individual λ levels miss)
- **covariance_calibration** — variance/correlation dispersion ratio (catches a covariance right on average but wrong in spread)

### Temporal (13%)
- **acf_returns** — autocorrelation of returns (should be ~0)
- **volatility_clustering** — autocorrelation of squared returns
- **leverage_effect** — corr(r_t, |r_{t+k}|) (negative for equities)
- **cross_correlation** — contemporaneous cross-asset correlation

### Joint (10%)
- **c2st** *(gate)* — Classifier Two-Sample Test: train a classifier to tell synthetic from real; ~0.5 accuracy = indistinguishable. An omnibus that catches joint-distribution defects the per-axis metrics miss.

### Conditional (22%)
- **regime_conditional** — regime-conditional distributional fidelity: paths bucketed into low/mid/high realized-vol regimes (tercile edges from the real paths) and scored on stress-path frequency + within-regime shape.
- **conditional_sensitivity** *(gate)* — regime-stratified energy-distance ratio: does the forecast distribution actually *move* across vol/trend regimes, or is the model a calibrated **climatology** that ignores the conditioning? The pooled lenses are blind to this.
- **pit_uniformity** / **crps** / **coverage_50/90/95** *(coverage_deficit is a gate)* — per-observation forecast-distribution calibration.

### Generative (20%)
- **Naeem density / coverage** — manifold realism scored as the **delta vs a block-bootstrap replay** of the real data (a generator must beat replay to justify itself, not merely tie it). Surfaces `coverage_deficit` *(gate)* + `plausibility_deficit`.

### Hard gates (fail outright, regardless of the weighted score)
`memorization` (data-copying: synth→real vs real→real NN distances — pass the *training* set as `real`), `tail_quantiles`, `tail_dependence_lower`, `drawdown_distribution`, `conditional_sensitivity`, `c2st`, `coverage_deficit`.

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

## Changelog

- **0.4.0** — Scoring reorganized into **6 weighted lenses** (marginal 0.15,
  dependence 0.20, temporal 0.13, joint 0.10, conditional 0.22, generative 0.20)
  + **7 hard gates**. New: `conditional_sensitivity` (regime-stratified — catches a
  climatological generator whose forecast ignores the conditioning), a **generative**
  lens (Naeem density/coverage vs a block-bootstrap *replay* baseline), a **joint**
  `c2st` omnibus, and **graceful degradation** (metrics flagged non-`applicable` on
  thin-data / long-horizon inputs instead of scored as failures). `validate_full(...)`
  is the all-lenses entry point.
- **0.3.0** — Two new dependence metrics: `tail_dependence_asymmetry`
  (scores whether synthetic paths reproduce the real lower-vs-upper
  tail-dependence asymmetry `A = λ_L − λ_U` that elliptical/Gaussian
  baselines get as 0) and `covariance_calibration` (scores the
  variance/correlation dispersion ratios of synthetic vs real, catching a
  covariance that is right on average but wrong in spread). Two new
  category axes: `regime_conditional` (12% — regime-conditional fidelity,
  the measured option-pricing gap that no pooled metric sees) and
  `memorization` (5% — nearest-neighbor data-copying diagnostic). Scoring
  is now de-quantized (continuous bands rather than discrete tiers).
  Category weights rebalanced: distribution 0.15→0.20, temporal
  0.20→0.15, calibration 0.30→0.15, path 0.10→0.08, with the new
  conditional/memorization axes carved out. The dependence and path
  metrics run under `validate_paths(...)`; the two dependence metrics
  also run under `validate(...)`.

## License

MIT
