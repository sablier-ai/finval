"""Integration tests for the top-level validate() / validate_paths() API."""

from __future__ import annotations

import numpy as np

import finval


def test_validate_returns_validation_report(real_returns_2d, matched_synthetic_2d):
    report = finval.validate(matched_synthetic_2d, real_returns_2d)
    assert isinstance(report, finval.ValidationReport)
    assert report.overall_quality in ("excellent", "good", "acceptable")
    assert len(report.metrics) >= 5


def test_validate_matched_data_scores_high(real_returns_2d, matched_synthetic_2d):
    report = finval.validate(matched_synthetic_2d, real_returns_2d)
    assert report.overall_score > 0.65  # at least "good"


def test_validate_broken_data_scores_low(rng, real_returns_2d):
    # 10x scale mismatch
    bad = rng.standard_normal(real_returns_2d.shape) * 0.1
    report = finval.validate(bad, real_returns_2d)
    assert report.overall_score < 0.50


def test_validate_paths_returns_all_path_metrics(real_paths_3d, matched_synthetic_paths_3d):
    report = finval.validate_paths(matched_synthetic_paths_3d, real_paths_3d)
    assert "acf_returns" in report.metrics
    assert "volatility_clustering" in report.metrics
    assert "leverage_effect" in report.metrics
    assert "cross_correlation" in report.metrics


def test_validate_paths_includes_flat_metrics_by_default(
    real_paths_3d, matched_synthetic_paths_3d
):
    report = finval.validate_paths(matched_synthetic_paths_3d, real_paths_3d)
    assert "marginal_ks" in report.metrics
    assert "pearson_corr" in report.metrics


def test_validate_paths_without_flat(real_paths_3d, matched_synthetic_paths_3d):
    report = finval.validate_paths(
        matched_synthetic_paths_3d, real_paths_3d, include_flat=False
    )
    assert "marginal_ks" not in report.metrics
    assert "acf_returns" in report.metrics


def test_validate_rejects_3d_input(real_paths_3d, matched_synthetic_paths_3d):
    import pytest

    with pytest.raises(ValueError, match="2D"):
        finval.validate(matched_synthetic_paths_3d, real_paths_3d)


def test_validate_paths_rejects_2d_input(real_returns_2d, matched_synthetic_2d):
    import pytest

    with pytest.raises(ValueError, match="3D"):
        finval.validate_paths(matched_synthetic_2d, real_returns_2d)


def test_metric_subset(real_returns_2d, matched_synthetic_2d):
    report = finval.validate(
        matched_synthetic_2d, real_returns_2d, metrics=["marginal_ks", "pearson_corr"]
    )
    assert set(report.metrics.keys()) == {"marginal_ks", "pearson_corr"}


def test_report_summary_is_string(real_returns_2d, matched_synthetic_2d):
    report = finval.validate(matched_synthetic_2d, real_returns_2d)
    s = report.summary()
    assert isinstance(s, str)
    assert "ValidationReport" in s


def test_report_to_dict_is_json_serializable(real_returns_2d, matched_synthetic_2d):
    import json

    report = finval.validate(matched_synthetic_2d, real_returns_2d)
    d = report.to_dict()
    json_str = json.dumps(d, default=float)  # numpy types → float
    assert len(json_str) > 0
