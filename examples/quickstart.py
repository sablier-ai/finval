"""finval quickstart example.

Demonstrates the three main use cases:

1. Flat 2D validation (distribution + dependence)
2. Path-level 3D validation (adds temporal + path metrics)
3. Baseline comparison (gaussian vs bootstrap vs your model)

Run from the finval repo root:

    python examples/quickstart.py
"""

import numpy as np

import finval
from finval.baselines import (
    block_bootstrap,
    gaussian_baseline,
    historical_bootstrap,
)


def build_real_returns(n: int = 2000, seed: int = 0) -> np.ndarray:
    """Synthesize realistic-ish multi-asset returns with fat tails,
    cross-asset correlation, and a crash regime."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(df=5, size=(n, 3)) * 0.01
    # Induce correlation: feature 1 is mostly driven by feature 0
    x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
    # Inject a few crash days where all assets move together
    crash_days = rng.choice(n, size=n // 100, replace=False)
    x[crash_days] -= 0.04
    return x


def main() -> None:
    print("=" * 72)
    print("finval quickstart")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # 1. Flat validation
    # ----------------------------------------------------------------------
    real = build_real_returns(n=2000, seed=0)
    synthetic_good = build_real_returns(n=2000, seed=1)  # same distribution, new seed

    print("\n1) Flat validation (matched synthetic):")
    report = finval.validate(synthetic_good, real, feature_names=["asset_a", "asset_b", "asset_c"])
    print(report.summary())

    # ----------------------------------------------------------------------
    # 2. Baseline comparison
    # ----------------------------------------------------------------------
    print("\n2) Baseline comparison:")
    for name, syn in [
        ("gaussian_iid", gaussian_baseline(real, n_samples=2000, seed=2)),
        ("iid_bootstrap", historical_bootstrap(real, n_samples=2000, seed=3)),
        ("your_model", synthetic_good),
    ]:
        r = finval.validate(syn, real)
        print(f"  {name:20s} {r.overall_quality:10s} {r.overall_score:5.0%}")

    # ----------------------------------------------------------------------
    # 3. Path-level validation
    # ----------------------------------------------------------------------
    # Build return paths from the flat series by slicing non-overlapping windows
    real_paths = real[: 60 * 30].reshape(30, 60, 3)
    syn_paths = synthetic_good[: 60 * 30].reshape(30, 60, 3)

    print("\n3) Path-level validation:")
    report = finval.validate_paths(syn_paths, real_paths)
    print(report.summary())

    # ----------------------------------------------------------------------
    # 4. Block bootstrap as a strong baseline for path metrics
    # ----------------------------------------------------------------------
    print("\n4) Block bootstrap beats gaussian on path metrics:")
    block_paths = block_bootstrap(real, n_paths=30, path_length=60, block_size=20, seed=4)
    gauss_paths = gaussian_baseline(real, n_paths=30, path_length=60, seed=5)

    for name, paths in [("gaussian_paths", gauss_paths), ("block_bootstrap", block_paths)]:
        r = finval.validate_paths(paths, real_paths)
        print(f"  {name:20s} {r.overall_quality:10s} {r.overall_score:5.0%}")


if __name__ == "__main__":
    main()
