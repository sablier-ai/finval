"""Model-agnostic threshold-calibration harness.

Given a real return/price panel, estimate every metric's **real-vs-real floor** — run the panel
on two disjoint halves of the REAL data, i.e. what "indistinguishable from reality" looks like
under finite-sample noise — and the **Gaussian-baseline** level (a dumb model). Thresholds should
sit: excellent ~ floor, acceptable ~ partway toward the baseline. finval ships reasonable defaults;
re-run this on your own corpus to recalibrate. Nothing here is model-specific.

Usage:  PYTHONPATH=src python tools/calibrate.py <panel.parquet> [--horizon 20] [--prices]
"""

from __future__ import annotations

import argparse

import numpy as np


def main(path: str, horizon: int, prices: bool) -> None:
    import pandas as pd

    import finval
    from finval.baselines.gaussian import gaussian_baseline

    df = pd.read_parquet(path)
    arr = df.select_dtypes("number").to_numpy(dtype=float)
    rets = np.diff(np.log(np.clip(arr, 1e-9, None)), axis=0) if prices else arr
    rets = rets[np.all(np.isfinite(rets), axis=1)]
    n, d = rets.shape
    print(f"loaded {path}: {n} rows x {d} features (prices={prices}) | finval {finval.__version__}")

    floor: dict = {}
    base: dict = {}
    rng = np.random.default_rng(0)

    # --- flat (rows) real-vs-real floor + gaussian baseline ---
    # RANDOM split (not time-order): two halves of the SAME distribution → the true
    # sampling-noise floor, uncontaminated by market non-stationarity.
    shuf = rets[rng.permutation(n)]
    h = n // 2
    A, B = shuf[:h], shuf[h : 2 * h]
    for k, m in finval.validate(A, B, metrics="all").metrics.items():
        floor[k] = m.value
    g = gaussian_baseline(rets, n_samples=h, seed=0)
    for k, m in finval.validate(g, rets[:h], metrics="all").metrics.items():
        base[k] = m.value

    # --- paths (OVERLAPPING windows → many paths for a stable path-metric floor) ---
    stride = max(1, horizon // 4)
    starts = list(range(0, n - horizon, stride))
    W = len(starts)
    if W >= 40:
        paths = np.stack([rets[s : s + horizon] for s in starts])  # (W, horizon, d)
        paths = paths[rng.permutation(W)]                          # shuffle window order (kill non-stationarity)
        half = W // 2
        pa, pb = paths[:half], paths[half : 2 * half]              # two same-distribution halves
        for rep in (finval.validate_paths(pa, pb, metrics="all"), finval.validate_generative(pa, pb)):
            for k, m in rep.metrics.items():
                floor[k] = m.value
        bp = gaussian_baseline(rets, n_paths=W, path_length=horizon, seed=0)
        for rep in (finval.validate_paths(bp, paths, metrics="all"), finval.validate_generative(bp, paths)):
            for k, m in rep.metrics.items():
                base[k] = m.value
    else:
        print(f"(only {W} windows at horizon={horizon}; skipping path metrics)")

    print(f"\n{'metric':30s} {'real-vs-real (floor)':>22s} {'gaussian (bad)':>16s}")
    print("-" * 70)
    for k in sorted(set(floor) | set(base)):
        fv = floor.get(k, float("nan"))
        bv = base.get(k, float("nan"))
        print(f"{k:30s} {fv:>22.4f} {bv:>16.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--prices", action="store_true", help="panel holds prices (→ log-returns); else returns")
    a = ap.parse_args()
    main(a.panel, a.horizon, a.prices)
