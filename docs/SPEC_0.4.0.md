# finval 0.4.0 — Validation Spec (the target contents)

**Status:** in-progress, **UNRELEASED** (working-tree version `0.4.0`; nothing published to PyPI; no tag/Release). This doc is the authoritative target for what 0.4.0 ships.

## Purpose

Define the complete set of properties we validate about synthetic data from **any** generative
model — derived **top-down** from what we want from the generated output and the downstream uses
(backtest augmentation, forecasting, scenario simulation) and the strategies run on the generated
paths — and map each to: current code state, scored-vs-diagnostic role, the failure mode it catches,
and the downstream use that needs it. **finval is model-agnostic** (it only ever sees `(synthetic,
real)` arrays — never a model); our own FLOW work is the *motivating use case and corpus*, not part
of the library.

## Governing principles (read before adding anything)

1. **Collectively exhaustive over failure modes, not MECE.** A metric earns its place if it catches a failure a sibling misses — *overlap is allowed*. The only thing forbidden is **deadweight**: a metric that catches nothing another doesn't. Every row below names the failure mode it catches and the use that needs it; if it can't, it's cut.
2. **Three layers, all required.** **A** = is the data realistic (intrinsic proxies). **B** = does that hold across the operating envelope (generalization). **C** = does it serve the decision (ground truth). A localizes *why* C fails; C is what actually matters; B checks both survive real conditions.
3. **Category weights cap each lens; weight within a lens by importance; gate the critical ones.** The category-weight structure (each lens = a fixed share of the total) means a metric-dense lens can't dominate the scalar — so we keep *multiple* scored metrics per lens wherever they catch distinct failures (coverage > MECE). Only genuinely diagnostic / derived / noisy metrics are demoted to **weight-0 localizers** (still computed + reported — full visibility). Decision-critical metrics are additionally **hard gates** (a "poor" flags the model regardless of aggregate).
4. **finval is the library; the leaderboard is finbench's concern — design finval right.** We are free to re-weight, reorganize categories, and change the scored aggregate. The 0.3.0 → 0.4.0 bump *is* the signal that the methodology changed; finbench pins a finval version and **re-runs its leaderboard against 0.4.0 when it adopts** — a deliberate, finbench-side cost paid later, not a constraint on finval's design. We do NOT contort finval to keep an old leaderboard byte-reproducible.
5. **Always report model-minus-baseline.** Every scored axis is also computed for the baselines (`baselines/historical.py` = block-bootstrap, `baselines/gaussian.py`; add DCC/GARCH) and the **delta** reported. A property only matters where the model beats the dumb baseline; coverage of a property the baseline already maxes does not guide research.

Status legend: **HAVE-S** = exists, scored · **HAVE-L** = exists, localizer (weight 0) · **BUILD** = new · **EXT** = extend existing · **finbench** = lives in the benchmark repo, not finval.

---

## LAYER A — realism of the generated data

### Lens 1 — Marginals (each feature's own distribution)

| metric | status | role | uniquely catches | downstream need |
|---|---|---|---|---|
| `marginal_ks` | HAVE-S | scored | body misfit (center/scale/shape) | all |
| `tail_quantiles` (1/5/95/99) | HAVE-S | scored | tail location the body-KS smooths over | risk, scenario |
| `tail_heaviness` (excess kurtosis) | HAVE-L | localizer | shape/under-dispersion (noisy → not scored) | risk |
| **far-tail EVT** (GPD tail-index / block-max KS) | **BUILD** | localizer (`validate`) | the >99% rare-crash collapse quantile-at-99 misses | VaR/ES, stress |
| **marginal skew error** | **BUILD** | localizer (`validate`) | directional asymmetry of the marginal | risk |

### Lens 2 — Cross-sectional dependence (contemporaneous co-movement)

| metric | status | role | uniquely catches | downstream need |
|---|---|---|---|---|
| `pearson_corr` | HAVE-S | scored | gross linear comovement (cheap, interpretable) | portfolio |
| `spearman_corr` | HAVE-S | scored | monotone nonlinearity linear misses | portfolio |
| `copula_distance` | HAVE-S | scored | full dependence shape | multi-asset |
| `tail_dependence_upper/lower` | HAVE-S | scored | joint crashes (corr can be right, this wrong) | portfolio risk, stress |
| `tail_dependence_asymmetry` (λ_L−λ_U) | HAVE-S | scored | downside-clustering = the non-elliptical edge | tail-hedge, stress |
| `covariance_calibration` | HAVE-S | scored | dispersion of the cov matrix (the under-/over-dispersion) | risk |
| `correlation_breakdown` (stress vs calm) | HAVE-S | scored† | corr-spikes-in-stress | stress, portfolio |
| **coskewness / cokurtosis error** | **BUILD** | localizer (`validate`) | multivariate higher-order shape the grid-copula under-resolves | options, dispersion |

†`correlation_breakdown` is regime-split → conceptually Layer-A-conditional; kept in the dependence category for leaderboard continuity, cross-referenced from Lens 5.

### Lens 3 — Temporal dynamics (path evolution within a feature)

| metric | status | role | uniquely catches | downstream need |
|---|---|---|---|---|
| `acf_returns` | HAVE-S | scored | spurious path momentum | momentum strats |
| `volatility_clustering` | HAVE-S | scored | GARCH persistence | vol-targeting |
| `leverage_effect` | HAVE-S | scored | return→vol asymmetry | options, risk |
| `cross_correlation` | HAVE-S | scored | lagged/contemporaneous cross-feature lead-lag | multi-asset |
| `drawdown_distribution` | HAVE-S | scored | path-tail shape (max DD) | risk, stop-loss |
| **variance term-structure** (realized-vol vs H) | **BUILD** | localizer (`validate_paths`) | multi-day variance scaling (long-H breaks) | long-H scenario, option term |
| **extreme clustering** (inter-exceedance times / extremal index) | **BUILD** | localizer (`validate_paths`) | crash bunching | stress, tail strats |
| **long memory** (Hurst / long-lag ACF) | **BUILD** | localizer (`validate_paths`) | persistence misfit | long-H |

### Lens 4 — Joint / omnibus

| metric | status | role | uniquely catches | downstream need |
|---|---|---|---|---|
| `energy_distance` | HAVE-S | scored | multivariate interactions the decomposition misses | all |
| **C2ST** (classifier two-sample test) | **BUILD** | localizer (`validate`/`validate_paths`) | **any unanticipated systematic real-vs-synth difference** (unknown-unknowns) | all |

C2ST = cross-validated classifier (gradient-boosted trees on rows/path-features) trained real-vs-synth; metric = `|2·AUC − 1|` (0 = indistinguishable). The single highest-value "did we miss something" detector; weight-0 so it never alters the frozen aggregate, but a high value is a hard flag.

### Lens 5 — Conditional fidelity (the climatology axis)

| metric | status | role | uniquely catches | downstream need |
|---|---|---|---|---|
| `conditional_sensitivity` (vol-regime) | HAVE-S(in `validate_conditional`) | scored (own entry) | climatology — forecast doesn't move | forecasting, scenario |
| regime-stratified CRPS/PIT/coverage + `within_regime_calibration_gap` | HAVE-S(in `validate_conditional`) | scored/localizer | stress miscalibration hidden by pooling | forecasting, risk |
| `regime_conditional` (within-regime joint + mixture) | HAVE-S | scored | wrong regime mixture / within-regime shape | scenario |
| conditional CRPS-**skill** vs climatology | HAVE (in `flow_conditional_run.py`, not a finval metric) | BUILD→localizer | conditioning that doesn't help | forecasting |
| **multi-axis sensitivity** (trend, drawdown, vol-term-structure, x-asset divergence, cycle) | **EXT** `conditional_sensitivity` to accept multiple label sets | scored per axis | partial conditioning (responds to vol only) | forecasting, scenario |
| **interventional response** ("rates +200bp") | **BUILD (research, later)** | localizer | no counterfactual behavior | scenario sim |

### Lens 6 — Generative health (generator-vs-replay)

| metric | status | role | uniquely catches | downstream need |
|---|---|---|---|---|
| `memorization` (NN-distance ratio) | HAVE-S | scored | copying training data | backtest aug |
| **recall / coverage** (real modes incl. rare crises represented) | **BUILD** (`validate_generative`) | scored | never generates a 2008/2020 | scenario, augmentation |
| **precision** (synth modes are real) | **BUILD** (`validate_generative`) | localizer | implausible generated modes | all |
| **novelty–plausibility vs bootstrap** | **BUILD** (`validate_generative`) | scored | **no value over replay (the reason to exist)** | the whole pitch |
| **diversity / effective mode count** | **BUILD** (`validate_generative`) | localizer | mode collapse | augmentation |

Generative health gets a new entry point `validate_generative(synthetic, real, *, baseline="bootstrap")` because it needs the real *set* + a baseline generator, not just two samples. recall/precision via improved precision-recall (Kynkäänniemi-style k-NN manifold) or Vendi/coverage; novelty-plausibility = distribution of synth→nearest-real distance × on-manifold fraction, reported as **model − bootstrap**.

### Lens 7 — Integrity / validity (hard pass-fail) — *0.4.x, derivatives-driven*

| check | status | role | catches | need |
|---|---|---|---|---|
| support-validity (bounded features in bounds) | **BUILD** | gate | out-of-support garbage | all (bounded series) |
| no-arbitrage (option surfaces) | **BUILD (later)** | gate | arbitrageable paths | derivatives |
| economic coherence (sign/monotonicity) | **BUILD (later)** | gate | incoherent scenarios | scenario, derivatives |

### Lens 8 — Operational / self-knowledge — *later*

| check | status | catches | need |
|---|---|---|---|
| reliability-envelope calibration (does "reliable" mean reliable) | **BUILD (later)** | overconfident reliability flags | production trust |
| generation reproducibility (seed-variance of downstream numbers) | **BUILD (later)** | unstable risk numbers | production |

---

## LAYER B — generalization across the operating envelope

Not new metrics — a **protocol**: run the Layer-A metrics across a grid and report **curves + worst-case + degradation slope**, not point estimates.

| axis | grid | catches |
|---|---|---|
| asset class | equity / rates / FX / crypto / commodities | works on equity, fails on rates |
| dimension D | 3 → 50 → 1000 | great at D=7, collapses at D=50 (the DCC hint) |
| horizon H | trained → 4× | variance term-structure breaks long |
| era / regime | train era vs OOS era | non-stationarity |
| seed | ≥5 | non-reproducible generated distribution |

Deliverable: a `validate_envelope(...)` harness (orchestration; **0.4.x**, can lag the metrics) that runs the panel over the grid and returns curves + worst-case per metric.

---

## LAYER C — downstream-decision fidelity (ground truth) — **finbench**

The layer that matters most and is least built. Principle: **each strategy/decision family is a distinct probe of the distribution; coverage over probes = coverage over decision-relevant failure modes.** Generalize TSTR from one momentum family to a battery.

| use case | must preserve | probe / KPI | status |
|---|---|---|---|
| backtest augmentation | strategy **rank AND magnitude**, across a battery (momentum, mean-reversion, carry, vol-target, tail-hedge, cross-sectional, options) | per-strategy Sharpe/DD/turnover synth-vs-real; rank-ρ (TSTR) + magnitude error | TSTR rank exists (1 family); **BUILD** battery + magnitude |
| overfit detection | a real-overfit strategy looks overfit on synth | synth-null deflation correctness | partial (fund `eval/verifier.py`); **BUILD** as finbench probe |
| forecasting | calibrated + skillful conditional forecasts | conditional CRPS-skill, PIT, coverage on the target | overlaps Lens 5; **BUILD** as forecasting KPI |
| scenario simulation | stress severity & co-movement; hit a specified scenario; plausible tails | stress-period match; conditional/interventional fidelity; tail plausibility | **BUILD** |
| risk (VaR/ES) | realized exceedance matches nominal | backtested VaR/ES exceedance | **BUILD** |

---

## Scoring model for 0.4.0

0.4.0 restructures the category map so the scored aggregate reflects every lens — including the two we were blind to. This *changes* the overall score vs 0.3.0 (intended; that's what the version bump means). **Calibration folds into Conditional** (calibration is inherently about predictive distributions given context). **Generative becomes first-class** (was `memorization` alone at 0.05). `energy_distance` moves to a small **Joint** lens with C2ST; `drawdown_distribution` folds into Temporal.

**Proposed category weights** (a DESIGN proposal — calibrate against the model corpus before locking, same discipline as 0.3.0's thresholds):

| category (lens) | weight | scored metrics | weight-0 localizers | hard gate |
|---|---|---|---|---|
| marginal | 0.15 | marginal_ks, tail_quantiles | tail_heaviness, far-tail EVT, skew | tail_quantiles, far-tail |
| dependence | 0.20 | copula_distance, tail_dep U/L, asymmetry, cov_calibration, pearson, spearman | correlation_breakdown, coskew | tail_dep_lower |
| temporal | 0.13 | vol_clustering, acf_returns, leverage, cross_correlation, drawdown | term-structure, extreme-clustering, long-memory | drawdown |
| joint | 0.10 | energy_distance | C2ST | C2ST |
| conditional | 0.22 | conditional_sensitivity, regime_conditional, crps, pit, coverage_90 | coverage_50/95, CRPS-skill, regime-stratified set, multi-axis sensitivities | conditional_sensitivity |
| generative | 0.20 | novelty-plausibility-vs-bootstrap, recall, memorization | precision, diversity | memorization, recall |

(Integrity = pass/fail **gates**, not weighted into the score.) Within-category weights are TBD by the calibration pass; multiple scored metrics per lens is deliberate (coverage), and the category share caps each lens so no lens bloats the scalar.

- **Per-lens vector** is the primary research-guidance object — drive on the 6-vector, not the scalar.
- **`validate_full(...)`** (BUILD): runs every applicable entry point (`validate`/`validate_paths` + `validate_conditional` + `validate_generative`), **renormalizing category weights over whatever inputs were provided** (pooled-only vs +forecast-samples vs +baseline — the existing `overall_score` already renormalizes to present metrics, so partial-input runs are honest). Returns `{overall, per_lens_vector, gates, flow_minus_baseline, full vector}`.
- **model-minus-baseline**: a reporting wrapper computing the panel for the model under test and for `baselines/{historical,gaussian}` (+ DCC/GARCH to add), emitting per-axis deltas — the number that actually guides research.

## Calibration (2026-06-22)

Thresholds for the new 0.4.0 metrics are calibrated against the **real-vs-real sampling-noise floor** via `tools/calibrate.py` (model-agnostic: random-split a real panel → run the panel on two same-distribution halves → the floor; Gaussian baseline = the "bad" reference). Set `excellent ≈ floor`. Run on an equity-macro panel; floors: c2st 0.002, coverage_deficit 0.0, plausibility_deficit 0.0, energy 0.0 (real-vs-real correctly ≈ perfect); far_tail 0.39, marginal_skew 0.22, coskew 0.19, variance_term 0.23*, extreme_clustering 0.08*, long_memory 0.04* (*path metrics from overlapping windows — approximate; refine against a many-independent-path corpus). Two findings worth keeping: (1) **far_tail at the 0.1 pct is uncalibratable** at realistic sample sizes (floor 1.3) → moved to 0.5/99.5 pct; (2) a **time-split** (not random) scores c2st ~0.31 — c2st *correctly detects market non-stationarity between periods*, so a high c2st on held-out time is signal, not a calibration error.

**Still a proposal (NOT yet calibrated):** the LENS_WEIGHTS and within-lens weights (relative importance is a downstream-utility judgment — calibrate against the decision battery in finbench / Layer C, not real-vs-real), and the conditional-lens thresholds (need model forecasts, not just real data — calibrate from the AWS run).

## Gap-close (SOTA audit, 2026-06-22)

A SOTA literature audit (Cont's stylized facts; Tail-GAN; Sig-Wasserstein; variogram score; TSGBench) found genuine holes. Added to `metrics/stylized.py` as calibrated localizers (each catches a failure the rest of the suite provably misses):

| metric | lens | catches | source |
|---|---|---|---|
| `time_reversal_asymmetry` | temporal | Zumbach / time-irreversibility (leverage fwd≠bwd) — the fact generators most silently fail | Zumbach 2007; Cont #11 |
| `aggregational_gaussianity` | marginal | kurtosis must DECAY on aggregation (we had variance term-structure, not shape) | Cont #4 |
| `conditional_heavy_tails` | marginal | tails stay heavy after de-volatilizing (not just from the vol process) | Cont #7 |
| `regime_persistence` | temporal | high-vol dwell-time (crises that mean-revert too fast) | regime-switching lit |
| `hill_tail_index` | marginal | tail EXPONENT α≈3 (shape), not just heaviness | Cont #2 |
| `variogram_score` | dependence | dependence error the energy-distance omnibus is documented to MISS | Scheuerer-Hamill 2015 |
| `signature_distance` | joint | higher-order path-ordering / lead-lag (level-2 truncation; **low power** on daily — tune path-scaling/lead-lag before trusting) | Sig-WGAN; sig-MMD power critique |

All thresholds calibrated vs the real-vs-real floor (the kurtosis metrics have wide bands — sample kurtosis is intrinsically noisy; signature is a weak diagnostic pending tuning). **Conceptual sufficiency note:** even with these, "aces finval ⟹ what we need" is guaranteed only by (a) a stringent C2ST → ~0.5 (no unmeasured difference detectable) AND (b) the downstream battery (Layer C, finbench, not yet built). The named metrics exist to *explain* the C2ST signal; residual C2ST after they pass = a still-unnamed gap.

**Still missing (deliberately):** strategy-level VaR/ES preservation (Tail-GAN) = the downstream sufficiency layer → **finbench Layer C**, not finval. Privacy suite (MIA/DCR) → only if the SDK ships synthetic data. Both out of finval scope.

---

## Build sequence (by downstream urgency)

**Phase 1 — fill the worst blind spots, cheap, in finval** (serves augmentation + forecasting directly; these are where a generator's value lives and the panel is blind):
1. `validate_generative`: **novelty–plausibility vs bootstrap** + **recall/coverage** + diversity. (the generator's reason-to-exist.)
2. **Multi-axis** `conditional_sensitivity` (trend / drawdown / vol-term / x-asset, not just vol).
3. **C2ST** omnibus localizer in `validate`/`validate_paths`.
4. **model-minus-baseline** reporting wrapper (reuse `baselines/`).

**Phase 2 — tails & dynamics, in finval** (serves scenario sim + risk):
5. far-tail EVT + marginal skew (Lens 1 localizers).
6. variance term-structure + extreme clustering + long memory (Lens 3 localizers).
7. coskew/cokurt (Lens 2 localizer).

**Phase 3 — ground truth, in finbench** (the decision battery — the thing that actually matters):
8. downstream battery: rank **+ magnitude** across strategy families; VaR/ES backtest; scenario-severity; overfit-deflation correctness.

**Phase 4 — later** (scope-disciplined, not in the first 0.4.0 cut):
9. `validate_envelope` grid harness (Layer B automation).
10. Integrity gates (support-validity now; no-arb/economic when derivatives land).
11. Operational: reliability-envelope calibration, generation reproducibility.
12. Interventional/counterfactual conditional response (research protocol).

`validate_full` + hard-gates land at the end of Phase 1 (once `validate_generative` exists), so the vector + gates are usable early.

---

## Explicitly OUT of the 0.4.0 first cut (scope discipline)

- **Deleting** any existing metric that still catches a distinct failure (coverage > MECE — keep it, even if it overlaps). Re-weighting and re-organizing the scored aggregate IS in scope; the downstream consequence (finbench re-runs its leaderboard against 0.4.0) is finbench's, paid when it adopts — not a reason to freeze finval.
- No-arbitrage / economic-coherence gates (need the derivatives data contract first).
- Interventional response, reliability self-calibration, full Layer-B automation (Phase 4).

## Open decisions

- C2ST classifier family (GBT vs logistic-on-features) + the path-feature embedding for the 3D case.
- recall/precision estimator (improved-P&R k-NN vs Vendi vs coverage) and the manifold k.
- The canonical baseline set for the model-minus-baseline delta (bootstrap + Gaussian have impls; add DCC + GARCH-copula).
- Whether `validate_full`'s combined score is published anywhere or stays research-internal (recommend internal until validated).
- Hard-gate thresholds (which metrics protected, at what grade).
