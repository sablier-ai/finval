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

    def __post_init__(self) -> None:
        # Sanitize NaN/Inf to ensure serializable output
        if self.value is not None and (np.isnan(self.value) or np.isinf(self.value)):
            self.value = float("inf")
            self.quality = "poor"
            self.passed = False
        if self.quality not in QUALITY_LEVELS:
            raise ValueError(f"quality must be one of {QUALITY_LEVELS}, got {self.quality!r}")

    @property
    def score(self) -> float:
        """Numeric score in [0, 1] for weighted aggregation."""
        return QUALITY_SCORES[self.quality]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        out: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "quality": self.quality,
            "passed": self.passed,
            "score": self.score,
            "thresholds": self.thresholds,
            "category": self.category,
            "interpretation": self.interpretation,
        }
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
    """Build a poor-quality result representing a computation failure."""
    return MetricResult(
        name=name,
        value=float("inf"),
        quality="poor",
        passed=False,
        category=category,
        interpretation=f"Failed: {error}",
        metadata={"error": error},
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
