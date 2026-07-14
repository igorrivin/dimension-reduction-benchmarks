"""Tests for the pure FP32/BF16 comparison helpers."""

import math

import pytest
from experiments.analyze_ae_precision import (
    BF16_TRAIN_FP32_EVAL,
    PRACTICAL_MARGINS,
    _verify_precision_metadata,
    compare_named_values,
    compare_values,
    descriptive_summary,
    practical_noninferiority,
    ratio_summary,
)


def test_descriptive_and_ratio_summaries_use_paired_sample_statistics():
    summary = descriptive_summary([1.0, 2.0, 3.0])
    assert summary == {
        "n": 3,
        "mean": 2.0,
        "sample_sd": 1.0,
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
    }

    ratios = ratio_summary([2.0, 8.0, 18.0], [1.0, 2.0, 3.0])
    assert ratios["values"] == [2.0, 4.0, 6.0]
    assert ratios["median"] == 4.0
    assert ratios["geometric_mean"] == pytest.approx(math.pow(48.0, 1.0 / 3.0))


def test_compare_values_reports_bf16_minus_fp32_paired_deltas():
    comparison = compare_values([1.0, 2.0, 4.0], [1.5, 1.0, 5.0])
    assert comparison["fp32"]["mean"] == pytest.approx(7.0 / 3.0)
    assert comparison["bf16"]["mean"] == pytest.approx(2.5)
    assert comparison["bf16_minus_fp32"]["mean"] == pytest.approx(1.0 / 6.0)
    assert comparison["bf16_minus_fp32"]["sample_sd"] == pytest.approx(
        descriptive_summary([0.5, -1.0, 1.0])["sample_sd"]
    )


def test_named_comparison_supports_the_evaluation_isolation_contrast():
    comparison = compare_named_values(
        [2.0, 4.0, 6.0],
        [3.0, 3.0, 9.0],
        reference_name="bf16_coupled",
        candidate_name="bf16_train_fp32_eval",
    )
    assert comparison["bf16_coupled"]["mean"] == 4.0
    assert comparison["bf16_train_fp32_eval"]["mean"] == 5.0
    assert comparison["bf16_train_fp32_eval_minus_bf16_coupled"]["mean"] == 1.0


def test_isolation_arm_requires_explicit_resolved_precision_metadata():
    payload = {"metadata": {"configuration": {"ae_evaluation_precision": "fp32"}}}
    expected = {
        "training_precision": "bfloat16_autocast",
        "validation_precision": "float32",
        "encoding_precision": "float32",
        "evaluation_precision": "float32",
        "evaluation_precision_policy": "fp32",
    }
    rows = [{"seed": seed, "details": expected.copy()} for seed in range(3)]
    resolved = _verify_precision_metadata(payload, rows, BF16_TRAIN_FP32_EVAL)
    assert resolved["training_precision"] == "bfloat16_autocast"
    assert resolved["evaluation_precision"] == "float32"

    rows[1]["details"]["evaluation_precision"] = "bfloat16"
    with pytest.raises(ValueError, match="details.evaluation_precision mismatch"):
        _verify_precision_metadata(payload, rows, BF16_TRAIN_FP32_EVAL)


def _statistics_for_noninferiority(dataset, overrides=None):
    overrides = {} if overrides is None else overrides
    names = [
        "native_reconstruction_nmse",
        "trustworthiness",
        "continuity",
        "neighbor_overlap",
        "distance_spearman",
        "normalized_stress",
        "decoder_nmse",
    ]
    names += (
        ["category_accuracy", "category_balanced_accuracy"]
        if dataset == "arxiv"
        else ["group_hit_rate", "group_recall", "group_purity"]
    )
    result = {}
    for name in names:
        fp32 = overrides.get(name, (1.0, 1.0, 1.0))[0]
        bf16 = overrides.get(name, (1.0, 1.0, 1.0))[1]
        result[name] = compare_values([fp32] * 3, [bf16] * 3)
    return result


@pytest.mark.parametrize("dataset", ["arxiv", "wikipedia"])
def test_practical_noninferiority_passes_values_at_the_prespecified_boundaries(dataset):
    overrides = {
        "native_reconstruction_nmse": (1.0, 1.02),
        "trustworthiness": (1.0, 0.99),
        "continuity": (1.0, 0.99),
        "neighbor_overlap": (1.0, 0.99),
        "distance_spearman": (1.0, 0.99),
        "normalized_stress": (1.0, 1.02),
        "decoder_nmse": (1.0, 1.02),
    }
    if dataset == "arxiv":
        overrides.update(
            {
                "category_accuracy": (1.0, 0.98),
                "category_balanced_accuracy": (1.0, 0.98),
            }
        )
    else:
        overrides.update(
            {
                "group_hit_rate": (1.0, 0.98),
                "group_recall": (1.0, 0.98),
                "group_purity": (1.0, 0.99),
            }
        )
    result = practical_noninferiority(
        dataset,
        _statistics_for_noninferiority(dataset, overrides),
        [1.05, 1.0, 0.95],
    )
    assert result["all_passed"]
    assert all(check["passed"] for check in result["checks"])
    assert PRACTICAL_MARGINS["native_nmse_mean_ratio_maximum"] == 1.02


def test_practical_noninferiority_identifies_mean_and_paired_failures():
    statistics = _statistics_for_noninferiority(
        "arxiv",
        {
            "native_reconstruction_nmse": (1.0, 1.021),
            "distance_spearman": (1.0, 0.989),
        },
    )
    result = practical_noninferiority("arxiv", statistics, [1.0, 1.051, 1.0])
    failures = {check["name"] for check in result["checks"] if not check["passed"]}
    assert not result["all_passed"]
    assert failures == {
        "native_nmse_ratio_of_means",
        "maximum_paired_native_nmse_ratio",
        "mean_delta_distance_spearman",
    }
