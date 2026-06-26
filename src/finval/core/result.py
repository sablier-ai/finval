"""Result types for validation metrics.

A `MetricResult` captures the output of a single metric: value, quality
grade, thresholds used, and optional bootstrap confidence interval and
per-feature/per-pair breakdown.

A `ValidationReport` aggregates multiple MetricResults into a single object
with an overall quality grade and weighted pass rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

QUALITY_LEVELS = ("excellent", "good", "acceptable", "poor")
QUALITY_SCORES = {"excellent": 1.0, "good": 0.75, "acceptable": 0.5, "poor": 0.0}


@dataclass
class MetricResult:
    """Result of a single validation metric.

    Attributes:
        name: Metric identifier, e.g. "marginal_ks".
        value: Scalar metric value (lower is always better by convention).
        quality: One of "excellent", "good", "acceptable", "poor".
        passed: True if quality is not "poor".
        thresholds: The threshold dict used to assign quality.
        category: Metric category, e.g. "distribution", "dependence".
        interpretation: Human-readable one-line summary.
        ci_low: Lower bound of bootstrap confidence interval (optional).
        ci_high: Upper bound of bootstrap confidence interval (optional).
        per_feature: Per-feature breakdown (optional).
        per_pair: Per-pair breakdown for dependence metrics (optional).
        metadata: Additional metric-specific information.
        effective_n: Number of independent REAL (sub)windows the metric was
            computed on — the reference sample size that bounds its statistical
            power. v0.5.0. ``None`` until stamped by the entry point that ran
            the metric (the metric functions don't set it). For path metrics
            this is ``n_real_paths`` (or ``n_real_paths * (H // W)`` under the
            opt-in ``subwindow=W`` mode); for flat metrics it is the flattened
            real row count. NOTE: finval reports the count it was *given*. If the
            caller passes OVERLAPPING real windows, the truly-independent N is
            lower than ``effective_n`` — guaranteeing independence is the
            caller's responsibility, not the library's.
    """

    name: str
    value: float
    quality: str
    passed: bool
    thresholds: dict[str, float] = field(default_factory=dict)
    category: str = "uncategorized"
    interpretation: str = ""
    ci_low: float | None = None
    ci_high: float | None = None
    per_feature: dict[str, Any] = field(default_factory=dict)
    per_pair: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # v0.5.0: independent-real-sample count (see attribute docstring above).
    effective_n: int | None = None
    # False = the property could NOT be measured on this data (e.g. the regimes don't
    # separate the real outcomes, so conditional-sensitivity is undefined) — as opposed
    # to "measured and poor". An inapplicable metric must NEVER gate or score as a model
    # failure; consumers exclude it. Keeps a model-agnostic library honest: a model isn't
    # "poor" because the USER's data lacks the structure a metric needs.
    applicable: bool = True

    def __post_init__(self) -> None:
        # Sanitize NaN/Inf to ensure serializable output
        if self.value is not None and (np.isnan(self.value) or np.isinf(self.value)):
            self.value = float("inf")
            if self.applicable:
                self.quality = "poor"          # genuine failure → worst grade
                self.passed = False
            # else: undefined/not-applicable — keep the caller's neutral grade so it
            # neither gates nor counts as a model failure (excluded via the inf value).
        if self.quality not in QUALITY_LEVELS:
            raise ValueError(f"quality must be one of {QUALITY_LEVELS}, got {self.quality!r}")

    @property
    def score(self) -> float:
        """Continuous, severity-aware score in [0, 1] for weighted aggregation.

        v0.3.0 de-quantization. Earlier versions collapsed ``value`` onto the
        four grade scores {1.0, 0.75, 0.5, 0.0}, discarding ALL within-grade
        severity — e.g. a ``tail_quantiles`` of 0.351 and 5.0 both scored 0.0,
        and 0.582 (66% past the gate) scored the same as 0.351. That made the
        scalar blind to exactly the failures it should penalize most and gave
        the research archive nothing to rank on inside a grade.

        This maps ``value`` continuously through the excellent/good/acceptable
        thresholds (lower is always better): piecewise-linear with knots
        (0→1.0, excellent→0.9, good→0.7, acceptable→0.5), then a smooth
        exponential decay toward 0 beyond the acceptable gate so severity keeps
        registering. Falls back to the discrete grade map when thresholds are
        absent or non-standard, so non-monotone/custom metrics are unaffected.
        """
        if self.value is None or not np.isfinite(self.value):
            return 0.0
        th = self.thresholds
        if not th or any(k not in th for k in ("excellent", "good", "acceptable")):
            return QUALITY_SCORES[self.quality]
        e, g, a = float(th["excellent"]), float(th["good"]), float(th["acceptable"])
        if not (0.0 <= e < g < a):  # non-standard ordering — don't guess
            return QUALITY_SCORES[self.quality]
        v = float(self.value)
        if v <= a:
            return float(np.interp(v, [0.0, e, g, a], [1.0, 0.9, 0.7, 0.5]))
        scale = max(a - e, 1e-9)  # severity decay length set by the metric's own band
        return float(0.5 * np.exp(-(v - a) / scale))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        out: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "quality": self.quality,
            "passed": self.passed,
            "applicable": self.applicable,
            "score": self.score,
            "thresholds": self.thresholds,
            "category": self.category,
            "interpretation": self.interpretation,
        }
        if self.effective_n is not None:
            out["effective_n"] = self.effective_n
        if self.ci_low is not None:
            out["ci_low"] = self.ci_low
            out["ci_high"] = self.ci_high
        if self.per_feature:
            out["per_feature"] = self.per_feature
        if self.per_pair:
            out["per_pair"] = self.per_pair
        if self.metadata:
            out["metadata"] = self.metadata
        return out


def create_error_metric(
    name: str,
    error: str,
    category: str = "uncategorized",
) -> MetricResult:
    """Build a poor-quality result representing a computation failure (a genuine error —
    bad input shape, a crash — that the caller should fix; scores 0 and may gate)."""
    return MetricResult(
        name=name,
        value=float("inf"),
        quality="poor",
        passed=False,
        category=category,
        interpretation=f"Failed: {error}",
        metadata={"error": error},
    )


def create_undefined_metric(
    name: str,
    reason: str,
    category: str = "uncategorized",
) -> MetricResult:
    """Build a NOT-APPLICABLE result: the property genuinely cannot be measured on this data
    (not a model failure). Excluded from lens scores and hard gates (inf value + applicable=
    False), so a model is never penalized for the user's data lacking the structure a metric
    needs. Distinct from `create_error_metric` (a real computation error)."""
    return MetricResult(
        name=name,
        value=float("nan"),          # sanitized to inf in __post_init__; applicable=False keeps grade neutral
        quality="acceptable",
        passed=True,
        applicable=False,
        category=category,
        interpretation=f"Not applicable: {reason}",
        metadata={"undefined_reason": reason},
    )


@dataclass
class ValidationReport:
    """Aggregated report of multiple validation metrics.

    A ValidationReport groups metrics by category, computes a weighted
    overall score, and assigns an overall quality grade.

    Attributes:
        metrics: Dict mapping metric name to MetricResult.
        weights: Dict mapping metric name to absolute weight (fractions
            should sum to ~1 over included metrics; missing metrics
            contribute 0).
        category_weights: Dict mapping category to category weight.
        overall_score: Weighted sum of metric scores in [0, 1].
        overall_quality: One of "excellent" (>=0.85), "good" (>=0.65),
            "acceptable" (>=0.45), "poor" otherwise.
    """

    metrics: dict[str, MetricResult] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    category_weights: dict[str, float] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Weighted average of metric scores.

        The public APIs (``validate``, ``validate_paths``,
        ``validate_calibration``) filter ``weights`` to only the
        metrics that actually ran, so the denominator is always
        self-consistent with the numerator — no silent zero-fill
        penalty for metrics that don't apply to the input shape.

        If a caller constructs a ``ValidationReport`` directly with
        ``weights`` for metrics not present in ``metrics``, those
        weights still contribute to the denominator with zero score.
        That behavior is intentional for incomplete-run detection but
        only applies when bypassing the public APIs.
        """
        if not self.metrics or not self.weights:
            return 0.0
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0.0
        weighted = sum(
            m.score * self.weights.get(m.name, 0.0) for m in self.metrics.values()
        )
        return weighted / total_weight

    @property
    def overall_quality(self) -> str:
        """Quality grade based on overall_score."""
        s = self.overall_score
        if s >= 0.85:
            return "excellent"
        if s >= 0.65:
            return "good"
        if s >= 0.45:
            return "acceptable"
        return "poor"

    @property
    def pass_rate(self) -> float:
        """Fraction of metrics with quality >= acceptable."""
        if not self.metrics:
            return 0.0
        passed = sum(1 for m in self.metrics.values() if m.passed)
        return passed / len(self.metrics)

    def by_category(self) -> dict[str, list[MetricResult]]:
        """Group metrics by category."""
        out: dict[str, list[MetricResult]] = {}
        for m in self.metrics.values():
            out.setdefault(m.category, []).append(m)
        return out

    def summary(self) -> str:
        """Human-readable one-page summary."""
        lines = [
            f"finval ValidationReport — {self.overall_quality.upper()} ({self.overall_score:.0%})",
            f"  metrics: {len(self.metrics)}, passed: {int(self.pass_rate * len(self.metrics))}/{len(self.metrics)}",
            "",
        ]
        for category, mlist in self.by_category().items():
            lines.append(f"  [{category}]")
            for m in sorted(mlist, key=lambda x: x.name):
                status = "PASS" if m.passed else "FAIL"
                ci = ""
                if m.ci_low is not None:
                    ci = f" ({m.ci_low:.3f}–{m.ci_high:.3f})"
                lines.append(
                    f"    {status}  {m.name:28s}  value={m.value:7.4f}{ci}  {m.quality}"
                )
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "overall_score": self.overall_score,
            "overall_quality": self.overall_quality,
            "pass_rate": self.pass_rate,
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "weights": dict(self.weights),
            "category_weights": dict(self.category_weights),
        }


@dataclass
class FullReport:
    """Comprehensive multi-lens report (v0.4.0, from ``validate_full``).

    The primary research-guidance object: a **per-lens vector** (marginal / dependence /
    temporal / joint / conditional / generative), an overall weighted over the lenses that
    were actually scorable (renormalized — partial inputs are honest), and the **hard gates**.
    ``overall_score`` is intentionally a *different, more complete* number than a pooled
    ``ValidationReport.overall_score``; if any hard gate is "poor", the model is ``gated`` and
    ``overall_quality`` is "poor" regardless of the weighted score.
    """

    metrics: dict[str, MetricResult] = field(default_factory=dict)
    per_lens: dict[str, float] = field(default_factory=dict)
    lens_weights_used: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    failing_gates: list[str] = field(default_factory=list)

    @property
    def gated(self) -> bool:
        return len(self.failing_gates) > 0

    @property
    def overall_quality(self) -> str:
        if self.gated:
            return "poor"
        s = self.overall_score
        if s >= 0.85:
            return "excellent"
        if s >= 0.65:
            return "good"
        if s >= 0.45:
            return "acceptable"
        return "poor"

    def summary(self) -> str:
        head = f"finval FullReport — {self.overall_quality.upper()} ({self.overall_score:.0%})"
        if self.gated:
            head += f"  ⚠ GATED: {', '.join(self.failing_gates)}"
        lines = [head, "  per-lens:"]
        for lens, s in sorted(self.per_lens.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {lens:12s} {s:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "overall_quality": self.overall_quality,
            "gated": self.gated,
            "failing_gates": list(self.failing_gates),
            "per_lens": dict(self.per_lens),
            "lens_weights_used": dict(self.lens_weights_used),
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
        }
