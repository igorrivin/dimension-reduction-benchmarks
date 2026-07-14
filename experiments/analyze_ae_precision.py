"""Audit and summarize the large-corpus FP32 and BF16 AE precision experiment.

The analysis is intentionally deterministic: it contains no timestamp, hashes every
input payload, validates that each precision pair used the same data and evaluation
samples, and writes a fixed-order paired CSV plus a sorted JSON summary.

Example
-------
python experiments/analyze_ae_precision.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (0, 1, 2)
HASH_METRICS = (
    "metric_sample_index_sha256",
    "distance_pair_sha256",
    "decoder_sample_index_sha256",
)
MATCHED_CONFIGURATION_FIELDS = (
    "ae_batch_size",
    "ae_epochs",
    "ae_hidden_dimensions",
    "ae_learning_rate",
    "ae_min_delta",
    "ae_patience",
    "ae_validation",
    "ae_weight_decay",
    "category_min_count",
    "dataset_name",
    "decoder_sample",
    "device",
    "distance_pairs",
    "evaluation_seed",
    "label_column",
    "label_policy",
    "metric_jobs",
    "metric_sample",
    "neighbors",
    "prepare_block_rows",
    "sample_seed",
    "seeds",
    "shard_glob",
    "sizes",
    "timeout",
)


@dataclass(frozen=True)
class Cell:
    dataset: str
    target_dimension: int
    fp32_results: str
    bf16_coupled_results: str
    bf16_train_fp32_eval_results: str
    fp32_topology: str
    bf16_coupled_topology: str
    bf16_train_fp32_eval_topology: str


@dataclass(frozen=True)
class Measure:
    name: str
    section: str
    source_name: str | None = None

    @property
    def key(self) -> str:
        return self.name if self.source_name is None else self.source_name


CELLS = (
    Cell(
        "arxiv",
        2,
        "artifacts/large-arxiv-d2/results.json",
        "artifacts/large-arxiv-ae-bf16/results.json",
        "artifacts/large-arxiv-ae-bf16-train-fp32-eval/results.json",
        "artifacts/large-arxiv-d2/topology/raw.json",
        "artifacts/large-arxiv-ae-bf16/topology-d2/raw.json",
        "artifacts/large-arxiv-ae-bf16-train-fp32-eval/topology-d2/raw.json",
    ),
    Cell(
        "arxiv",
        8,
        "artifacts/large-arxiv-d8/results.json",
        "artifacts/large-arxiv-ae-bf16/results.json",
        "artifacts/large-arxiv-ae-bf16-train-fp32-eval/results.json",
        "artifacts/large-arxiv-d8/topology/raw.json",
        "artifacts/large-arxiv-ae-bf16/topology-d8/raw.json",
        "artifacts/large-arxiv-ae-bf16-train-fp32-eval/topology-d8/raw.json",
    ),
    Cell(
        "wikipedia",
        2,
        "artifacts/large-wikipedia-d2/results.json",
        "artifacts/large-wikipedia-ae-bf16/results.json",
        "artifacts/large-wikipedia-ae-bf16-train-fp32-eval/results.json",
        "artifacts/large-wikipedia-d2/topology/raw.json",
        "artifacts/large-wikipedia-ae-bf16/topology-d2/raw.json",
        "artifacts/large-wikipedia-ae-bf16-train-fp32-eval/topology-d2/raw.json",
    ),
    Cell(
        "wikipedia",
        8,
        "artifacts/large-wikipedia-d8/results.json",
        "artifacts/large-wikipedia-ae-bf16/results.json",
        "artifacts/large-wikipedia-ae-bf16-train-fp32-eval/results.json",
        "artifacts/large-wikipedia-d8/topology/raw.json",
        "artifacts/large-wikipedia-ae-bf16/topology-d8/raw.json",
        "artifacts/large-wikipedia-ae-bf16-train-fp32-eval/topology-d8/raw.json",
    ),
)

FP32 = "fp32"
BF16_COUPLED = "bf16_coupled"
BF16_TRAIN_FP32_EVAL = "bf16_train_fp32_eval"
BF16_ARMS = (BF16_COUPLED, BF16_TRAIN_FP32_EVAL)

MEASURES = (
    Measure("total_seconds", "details", "fit_seconds"),
    Measure("staging_seconds", "details"),
    Measure("selection_seconds", "details"),
    Measure("refit_seconds", "details"),
    Measure("training_seconds", "derived"),
    Measure("encode_reconstruction_seconds", "details"),
    Measure("native_metric_finalize_seconds", "details"),
    Measure("selected_epochs", "details"),
    Measure("best_validation_mse", "details"),
    Measure("native_reconstruction_mse", "details"),
    Measure("native_reconstruction_nmse", "details"),
    Measure("native_reconstruction_cosine", "details"),
    Measure("trustworthiness", "metrics", "sample_conditional_trustworthiness"),
    Measure("continuity", "metrics", "sample_conditional_continuity"),
    Measure("neighbor_overlap", "metrics", "sample_conditional_neighbor_overlap"),
    Measure("distance_spearman", "metrics", "sampled_pair_distance_spearman"),
    Measure("normalized_stress", "metrics", "sampled_pair_normalized_stress"),
    Measure("decoder_nmse", "metrics", "sampled_local_decodability_nmse"),
    Measure("category_accuracy", "metrics", "category_knn_accuracy"),
    Measure("category_balanced_accuracy", "metrics", "category_knn_balanced_accuracy"),
    Measure("group_purity", "metrics", "group_neighbor_purity"),
    Measure("group_hit_rate", "metrics", "group_neighbor_hit_rate"),
    Measure("group_recall", "metrics", "group_neighbor_recall"),
    Measure("peak_incremental_gpu_bytes", "memory"),
    Measure("peak_rss_bytes", "memory"),
)

PRACTICAL_MARGINS = {
    "native_nmse_mean_ratio_maximum": 1.02,
    "native_nmse_paired_ratio_maximum": 1.05,
    "higher_is_better_geometry_mean_delta_minimum": -0.01,
    "stress_and_decoder_nmse_mean_delta_maximum": 0.02,
    "arxiv_label_mean_delta_minimum": -0.02,
    "wikipedia_hit_recall_mean_delta_minimum": -0.02,
    "wikipedia_purity_mean_delta_minimum": -0.01,
}


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without reading the whole payload into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    """Match the topology evaluator's canonical, platform-independent index hash."""
    values = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def descriptive_summary(values: Iterable[float]) -> dict[str, float | int]:
    """Return fixed descriptive statistics, using sample SD for replicated values."""
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot summarize an empty sequence")
    if not all(math.isfinite(value) for value in data):
        raise ValueError("summary values must all be finite")
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "sample_sd": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "minimum": min(data),
        "maximum": max(data),
    }


def ratio_summary(numerators: Sequence[float], denominators: Sequence[float]) -> dict[str, Any]:
    """Summarize positive paired ratios, including their geometric mean."""
    if len(numerators) != len(denominators) or not numerators:
        raise ValueError("ratio inputs must be non-empty and have equal length")
    ratios = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        numerator = float(numerator)
        denominator = float(denominator)
        if numerator <= 0.0 or denominator <= 0.0:
            raise ValueError("ratio inputs must be positive")
        ratios.append(numerator / denominator)
    summary = descriptive_summary(ratios)
    summary["geometric_mean"] = math.exp(statistics.fmean(math.log(value) for value in ratios))
    summary["values"] = ratios
    return summary


def compare_named_values(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    """Describe two named arms and their paired candidate-minus-reference differences."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("comparison inputs must be non-empty and have equal length")
    deltas = [float(right) - float(left) for left, right in zip(reference, candidate, strict=True)]
    return {
        reference_name: descriptive_summary(reference),
        candidate_name: descriptive_summary(candidate),
        f"{candidate_name}_minus_{reference_name}": descriptive_summary(deltas),
    }


def compare_values(
    fp32: Sequence[float],
    candidate: Sequence[float],
    candidate_name: str = "bf16",
) -> dict[str, Any]:
    """Describe FP32, one candidate arm, and paired candidate-minus-FP32 differences."""
    return compare_named_values(
        fp32,
        candidate,
        reference_name=FP32,
        candidate_name=candidate_name,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _require_equal(label: str, left: Any, right: Any) -> Any:
    if left != right:
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")
    return left


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _ae_rows(
    payload: dict[str, Any],
    *,
    dataset: str,
    target_dimension: int,
    arm: str,
) -> list[dict[str, Any]]:
    metadata = payload.get("metadata", {})
    configuration = metadata.get("configuration", {})
    if arm not in (FP32, *BF16_ARMS):
        raise ValueError(f"unsupported precision arm {arm!r}")
    expected_amp = arm != FP32
    if metadata.get("dataset") != dataset:
        raise ValueError(f"metadata dataset is not {dataset!r}")
    if configuration.get("dataset_name") != dataset:
        raise ValueError(f"configuration dataset is not {dataset!r}")
    if configuration.get("ae_amp") is not expected_amp:
        raise ValueError(f"configuration ae_amp is not {expected_amp}")
    if list(configuration.get("seeds", [])) != list(SEEDS):
        raise ValueError(f"expected seeds {SEEDS}, got {configuration.get('seeds')!r}")
    configured_dimensions = configuration.get("target_dimensions", [])
    expected_dimensions = [target_dimension] if arm == FP32 else [2, 8]
    if configured_dimensions != expected_dimensions:
        raise ValueError(
            f"expected target dimensions {expected_dimensions}, got {configured_dimensions!r}"
        )

    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("method") == "autoencoder" and row.get("target_dimension") == target_dimension
    ]
    if len(rows) != len(SEEDS):
        raise ValueError(
            f"expected {len(SEEDS)} AE rows for {dataset} d={target_dimension}, got {len(rows)}"
        )
    rows.sort(key=lambda row: row["seed"])
    if [row.get("seed") for row in rows] != list(SEEDS):
        raise ValueError(f"AE rows do not contain exactly seeds {SEEDS}")
    for row in rows:
        if row.get("dataset") != dataset:
            raise ValueError("row dataset does not match its cell")
        if row.get("target_dimension") != target_dimension:
            raise ValueError("row target dimension does not match its cell")
        if row.get("status") != "ok" or row.get("error") is not None:
            raise ValueError(f"AE seed {row.get('seed')} did not finish successfully")
        if row.get("details", {}).get("amp_bfloat16") is not expected_amp:
            raise ValueError(f"AE seed {row.get('seed')} has the wrong AMP flag")
        selected_epochs = row.get("details", {}).get("selected_epochs")
        if (
            not isinstance(selected_epochs, int)
            or not 1 <= selected_epochs <= configuration["ae_epochs"]
        ):
            raise ValueError(f"invalid selected epoch count {selected_epochs!r}")
    return rows


def _verify_precision_metadata(
    payload: dict[str, Any], rows: list[dict[str, Any]], arm: str
) -> dict[str, Any]:
    """Resolve and verify the computation precision represented by an input arm."""
    configuration = payload["metadata"]["configuration"]
    if arm == BF16_TRAIN_FP32_EVAL:
        _require_equal(
            "configuration.ae_evaluation_precision",
            configuration.get("ae_evaluation_precision"),
            "fp32",
        )
        expected_details = {
            "training_precision": "bfloat16_autocast",
            "validation_precision": "float32",
            "encoding_precision": "float32",
            "evaluation_precision": "float32",
            "evaluation_precision_policy": "fp32",
        }
        for row in rows:
            for field, expected in expected_details.items():
                _require_equal(
                    f"seed {row['seed']} details.{field}",
                    row["details"].get(field),
                    expected,
                )
        return {
            **expected_details,
            "configuration_ae_evaluation_precision": "fp32",
            "metadata_basis": "explicit resolved precision metadata",
        }

    _require_equal(
        "legacy configuration.ae_evaluation_precision",
        configuration.get("ae_evaluation_precision"),
        None,
    )
    legacy_fields = (
        "training_precision",
        "validation_precision",
        "encoding_precision",
        "evaluation_precision",
        "evaluation_precision_policy",
    )
    for row in rows:
        for field in legacy_fields:
            _require_equal(
                f"legacy seed {row['seed']} details.{field}", row["details"].get(field), None
            )
    if arm == FP32:
        return {
            "training_precision": "float32",
            "validation_precision": "float32",
            "encoding_precision": "float32",
            "evaluation_precision": "float32",
            "metadata_basis": "legacy ae_amp=false flag and archived harness semantics",
        }
    if arm == BF16_COUPLED:
        return {
            "training_precision": "bfloat16_autocast",
            "validation_precision": "bfloat16_autocast",
            "encoding_precision": "bfloat16_autocast",
            "evaluation_precision": "bfloat16_autocast",
            "metadata_basis": "legacy ae_amp=true flag and archived harness semantics",
        }
    raise ValueError(f"unsupported precision arm {arm!r}")


def _prepared_record(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = payload.get("metadata", {}).get("prepared", [])
    if len(prepared) != 1:
        raise ValueError(f"expected one prepared-data record, got {len(prepared)}")
    return prepared[0]


def _verify_pair_provenance(
    fp32_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    fp32_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    candidate_name: str,
) -> dict[str, Any]:
    fp32_metadata = fp32_payload["metadata"]
    candidate_metadata = candidate_payload["metadata"]
    fp32_config = fp32_metadata["configuration"]
    candidate_config = candidate_metadata["configuration"]

    matched_configuration = {}
    for field in MATCHED_CONFIGURATION_FIELDS:
        matched_configuration[field] = _require_equal(
            f"configuration.{field}", fp32_config.get(field), candidate_config.get(field)
        )
    harness_sha256 = {
        FP32: fp32_metadata.get("harness_sha256"),
        candidate_name: candidate_metadata.get("harness_sha256"),
    }
    if not all(isinstance(value, str) and len(value) == 64 for value in harness_sha256.values()):
        raise ValueError("each results payload must contain a SHA-256 harness identifier")
    source_fingerprint = _require_equal(
        "source.fingerprint",
        fp32_metadata.get("source", {}).get("fingerprint"),
        candidate_metadata.get("source", {}).get("fingerprint"),
    )
    for field in ("dtype", "features", "rows"):
        _require_equal(
            f"source.{field}",
            fp32_metadata.get("source", {}).get(field),
            candidate_metadata.get("source", {}).get(field),
        )
    label_fingerprint = _require_equal(
        "labels.fingerprint",
        fp32_metadata.get("labels", {}).get("fingerprint"),
        candidate_metadata.get("labels", {}).get("fingerprint"),
    )
    fp32_prepared = _prepared_record(fp32_payload)
    candidate_prepared = _prepared_record(candidate_payload)
    prepared_sha256 = _require_equal(
        "prepared.normalized_sha256",
        fp32_prepared.get("normalized_sha256"),
        candidate_prepared.get("normalized_sha256"),
    )
    source_index_sha256 = _require_equal(
        "prepared.source_index_sha256",
        fp32_prepared.get("source_index_sha256"),
        candidate_prepared.get("source_index_sha256"),
    )
    for prepared in (fp32_prepared, candidate_prepared):
        _require_equal(
            "prepared.source_fingerprint", prepared.get("source_fingerprint"), source_fingerprint
        )

    paired_metric_hashes: dict[str, str] = {}
    for fp32_row, candidate_row in zip(fp32_rows, candidate_rows, strict=True):
        seed = fp32_row["seed"]
        _require_equal("paired seed", seed, candidate_row["seed"])
        for field in ("features", "n_samples", "rows"):
            _require_equal(
                f"seed {seed} row.{field}", fp32_row.get(field), candidate_row.get(field)
            )
        for field in (
            "cuda_compute_capability",
            "gpu_name",
            "gpu_total_memory_bytes",
            "machine",
            "platform",
        ):
            _require_equal(
                f"seed {seed} hardware.{field}",
                fp32_row.get("hardware", {}).get(field),
                candidate_row.get("hardware", {}).get(field),
            )
        for field in ("numpy", "torch"):
            _require_equal(
                f"seed {seed} versions.{field}",
                fp32_row.get("versions", {}).get(field),
                candidate_row.get("versions", {}).get(field),
            )
        for precision, row in ((FP32, fp32_row), (candidate_name, candidate_row)):
            _require_equal(
                f"{precision} prepared_sha256", row.get("prepared_sha256"), prepared_sha256
            )
            _require_equal(
                f"{precision} source_index_sha256",
                row.get("source_index_sha256"),
                source_index_sha256,
            )
        for field in HASH_METRICS:
            value = _require_equal(
                f"seed {seed} metrics.{field}",
                fp32_row["metrics"].get(field),
                candidate_row["metrics"].get(field),
            )
            if field in paired_metric_hashes:
                _require_equal(f"cross-seed metrics.{field}", paired_metric_hashes[field], value)
            else:
                paired_metric_hashes[field] = value

    return {
        "harness_sha256": harness_sha256,
        "harness_identical_to_fp32": (harness_sha256[FP32] == harness_sha256[candidate_name]),
        "source_fingerprint": source_fingerprint,
        "label_fingerprint": label_fingerprint,
        "prepared_normalized_sha256": prepared_sha256,
        "source_index_sha256": source_index_sha256,
        **paired_metric_hashes,
        "matched_configuration": matched_configuration,
    }


def _measure_value(row: dict[str, Any], measure: Measure) -> float | None:
    if measure.section == "derived":
        details = row["details"]
        return float(details["selection_seconds"] + details["refit_seconds"])
    value = row.get(measure.section, {}).get(measure.key)
    return None if value is None else float(value)


def _cell_series(
    fp32_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    candidate_name: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    statistics_by_measure: dict[str, dict[str, Any]] = {}
    paired_rows = [{"seed": seed} for seed in SEEDS]
    for measure in MEASURES:
        fp32 = [_measure_value(row, measure) for row in fp32_rows]
        candidate = [_measure_value(row, measure) for row in candidate_rows]
        if all(value is None for value in fp32 + candidate):
            continue
        if any(value is None for value in fp32 + candidate):
            raise ValueError(f"measure {measure.name} is missing from only part of a paired cell")
        fp32_values = [float(value) for value in fp32 if value is not None]
        candidate_values = [float(value) for value in candidate if value is not None]
        statistics_by_measure[measure.name] = compare_values(
            fp32_values, candidate_values, candidate_name
        )
        for index, (left, right) in enumerate(zip(fp32_values, candidate_values, strict=True)):
            paired_rows[index][f"{measure.name}_fp32"] = left
            paired_rows[index][f"{measure.name}_candidate"] = right
            paired_rows[index][f"{measure.name}_candidate_minus_fp32"] = right - left
    return statistics_by_measure, paired_rows


def practical_noninferiority(
    dataset: str,
    statistics_by_measure: dict[str, dict[str, Any]],
    native_nmse_paired_ratios: Sequence[float],
    candidate_name: str = "bf16",
) -> dict[str, Any]:
    """Apply the fixed practical margins; this is not a formal NI test."""

    def delta(name: str) -> float:
        return float(statistics_by_measure[name][f"{candidate_name}_minus_{FP32}"]["mean"])

    checks: list[dict[str, Any]] = []

    def add(name: str, observed: float, operator: str, threshold: float) -> None:
        if operator == "<=":
            passed = observed <= threshold or math.isclose(observed, threshold, abs_tol=1e-12)
        elif operator == ">=":
            passed = observed >= threshold or math.isclose(observed, threshold, abs_tol=1e-12)
        else:
            raise ValueError(f"unsupported comparison operator {operator!r}")
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": passed,
            }
        )

    nmse_fp32 = statistics_by_measure["native_reconstruction_nmse"]["fp32"]["mean"]
    nmse_candidate = statistics_by_measure["native_reconstruction_nmse"][candidate_name]["mean"]
    add(
        "native_nmse_ratio_of_means",
        nmse_candidate / nmse_fp32,
        "<=",
        PRACTICAL_MARGINS["native_nmse_mean_ratio_maximum"],
    )
    add(
        "maximum_paired_native_nmse_ratio",
        max(native_nmse_paired_ratios),
        "<=",
        PRACTICAL_MARGINS["native_nmse_paired_ratio_maximum"],
    )
    for measure in ("trustworthiness", "continuity", "neighbor_overlap", "distance_spearman"):
        add(
            f"mean_delta_{measure}",
            delta(measure),
            ">=",
            PRACTICAL_MARGINS["higher_is_better_geometry_mean_delta_minimum"],
        )
    for measure in ("normalized_stress", "decoder_nmse"):
        add(
            f"mean_delta_{measure}",
            delta(measure),
            "<=",
            PRACTICAL_MARGINS["stress_and_decoder_nmse_mean_delta_maximum"],
        )
    if dataset == "arxiv":
        for measure in ("category_accuracy", "category_balanced_accuracy"):
            add(
                f"mean_delta_{measure}",
                delta(measure),
                ">=",
                PRACTICAL_MARGINS["arxiv_label_mean_delta_minimum"],
            )
    elif dataset == "wikipedia":
        for measure in ("group_hit_rate", "group_recall"):
            add(
                f"mean_delta_{measure}",
                delta(measure),
                ">=",
                PRACTICAL_MARGINS["wikipedia_hit_recall_mean_delta_minimum"],
            )
        add(
            "mean_delta_group_purity",
            delta("group_purity"),
            ">=",
            PRACTICAL_MARGINS["wikipedia_purity_mean_delta_minimum"],
        )
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")

    return {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "interpretation": (
            "Prespecified practical/descriptive non-inferiority margins; with three seeds, "
            "these are not formal statistical non-inferiority or equivalence tests."
        ),
    }


def _resolve_recorded_path(recorded: str, raw_path: Path) -> Path:
    candidate = Path(recorded)
    if candidate.is_file():
        return candidate
    fallback = raw_path.parent / candidate.name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"topology sample file does not exist: {recorded}")


def _topology_samples(payload: dict[str, Any], raw_path: Path) -> dict[int, dict[str, Any]]:
    samples = payload.get("provenance", {}).get("samples", {})
    result: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        record = samples.get(str(seed))
        if not isinstance(record, dict):
            raise ValueError(f"topology provenance is missing sample seed {seed}")
        sample_path = _resolve_recorded_path(record["path"], raw_path)
        indices = np.load(sample_path, allow_pickle=False)
        canonical_hash = index_sha256(indices)
        file_hash = sha256_file(sample_path)
        _require_equal(
            f"topology seed {seed} canonical index hash", canonical_hash, record["index_sha256"]
        )
        _require_equal(f"topology seed {seed} sample file hash", file_hash, record["sha256"])
        _require_equal(f"topology seed {seed} sample size", int(indices.size), record["size"])
        result[seed] = {
            "index_sha256": canonical_hash,
            "file_sha256": file_hash,
            "size": int(indices.size),
        }
    return result


def _ae_topology_records(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    records = [
        record
        for record in payload.get("records", [])
        if str(record.get("method", "")).startswith("autoencoder-s")
    ]
    result = {(record["method"], int(record["seed"])): record for record in records}
    expected = {
        (f"autoencoder-s{fit_seed}", sample_seed) for fit_seed in SEEDS for sample_seed in SEEDS
    }
    if set(result) != expected or len(records) != len(expected):
        raise ValueError("topology AE records are not the complete 3 x 3 fit/sample grid")
    return result


def _compare_topology(
    reference_path: Path,
    candidate_path: Path,
    target_dimension: int,
    *,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    if not reference_path.is_file() or not candidate_path.is_file():
        return {
            "status": "not_compared",
            f"{reference_name}_raw_exists": reference_path.is_file(),
            f"{candidate_name}_raw_exists": candidate_path.is_file(),
            "reason": "Both topology raw.json payloads are required for a paired comparison.",
        }
    reference = _load_json(reference_path)
    candidate = _load_json(candidate_path)
    reference_config = reference.get("configuration", {})
    candidate_config = candidate.get("configuration", {})
    _require_equal(
        "topology target dimension", reference_config.get("target_dimension"), target_dimension
    )
    _require_equal(
        "topology target dimension", candidate_config.get("target_dimension"), target_dimension
    )
    for field in (
        "diameter_block_size",
        "l2_atol",
        "n_steps",
        "sample_size",
        "sampling",
        "seeds",
        "target_dimension",
    ):
        _require_equal(
            f"topology configuration.{field}",
            reference_config.get(field),
            candidate_config.get(field),
        )

    reference_samples = _topology_samples(reference, reference_path)
    candidate_samples = _topology_samples(candidate, candidate_path)
    for seed in SEEDS:
        _require_equal(
            f"topology seed {seed} canonical sample hash",
            reference_samples[seed]["index_sha256"],
            candidate_samples[seed]["index_sha256"],
        )
        _require_equal(
            f"topology seed {seed} sample file hash",
            reference_samples[seed]["file_sha256"],
            candidate_samples[seed]["file_sha256"],
        )

    reference_records = _ae_topology_records(reference)
    candidate_records = _ae_topology_records(candidate)
    comparisons = {}
    for metric in ("dtw_beta0", "dtw_beta1"):
        left = []
        right = []
        for key in sorted(reference_records):
            reference_record = reference_records[key]
            candidate_record = candidate_records[key]
            sample_seed = key[1]
            expected_sample_hash = reference_samples[sample_seed]["index_sha256"]
            for precision, record in (
                (reference_name, reference_record),
                (candidate_name, candidate_record),
            ):
                _require_equal(
                    f"{precision} topology record sample hash",
                    record.get("sample_index_sha256"),
                    expected_sample_hash,
                )
                _require_equal(
                    f"{precision} topology record source sample hash",
                    record.get("source_sample_index_sha256"),
                    expected_sample_hash,
                )
            left.append(float(reference_record[metric]))
            right.append(float(candidate_record[metric]))
        comparison = compare_named_values(
            left,
            right,
            reference_name=reference_name,
            candidate_name=candidate_name,
        )
        comparison[f"{candidate_name}_over_{reference_name}_paired_ratio"] = ratio_summary(
            right, left
        )
        comparison[f"{candidate_name}_over_{reference_name}_ratio_of_pooled_means"] = (
            comparison[candidate_name]["mean"] / comparison[reference_name]["mean"]
        )
        comparisons[metric] = comparison

    return {
        "status": "compared",
        "interpretation": (
            "Topology DTW comparisons are descriptive only; no practical pass/fail margin "
            "was prespecified because the sampled Betti curves are sensitive to sampling "
            "and filtration grids."
        ),
        "inputs": {
            reference_name: {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            candidate_name: {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
        },
        "verified_sample_hashes": {str(seed): reference_samples[seed] for seed in SEEDS},
        "pooled_over_fit_and_topology_seeds": comparisons,
    }


def _named_statistics(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reference_name: str,
    candidate_name: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for measure in MEASURES:
        left = [_measure_value(row, measure) for row in reference_rows]
        right = [_measure_value(row, measure) for row in candidate_rows]
        if all(value is None for value in left + right):
            continue
        if any(value is None for value in left + right):
            raise ValueError(f"measure {measure.name} is missing from one precision arm")
        result[measure.name] = compare_named_values(
            [float(value) for value in left if value is not None],
            [float(value) for value in right if value is not None],
            reference_name=reference_name,
            candidate_name=candidate_name,
        )
    return result


def _relativize_topology_inputs(topology: dict[str, Any], root: Path) -> None:
    if topology["status"] != "compared":
        return
    for input_record in topology["inputs"].values():
        input_record["path"] = _relative(Path(input_record["path"]), root)


def _arm_comparison(
    *,
    cell: Cell,
    root: Path,
    arm: str,
    fp32_payload: dict[str, Any],
    arm_payload: dict[str, Any],
    fp32_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    fp32_topology: str,
    arm_topology: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    provenance = _verify_pair_provenance(fp32_payload, arm_payload, fp32_rows, arm_rows, arm)
    statistics_by_measure, paired_rows = _cell_series(fp32_rows, arm_rows, arm)
    fp32_nmse = [row["details"]["native_reconstruction_nmse"] for row in fp32_rows]
    arm_nmse = [row["details"]["native_reconstruction_nmse"] for row in arm_rows]
    native_nmse_ratio = ratio_summary(arm_nmse, fp32_nmse)
    fp32_training = [
        row["details"]["selection_seconds"] + row["details"]["refit_seconds"] for row in fp32_rows
    ]
    arm_training = [
        row["details"]["selection_seconds"] + row["details"]["refit_seconds"] for row in arm_rows
    ]
    training_speedup = ratio_summary(fp32_training, arm_training)
    total_speedup = ratio_summary(
        [row["details"]["fit_seconds"] for row in fp32_rows],
        [row["details"]["fit_seconds"] for row in arm_rows],
    )
    noninferiority = practical_noninferiority(
        cell.dataset,
        statistics_by_measure,
        native_nmse_ratio["values"],
        arm,
    )
    precision = _verify_precision_metadata(arm_payload, arm_rows, arm)
    for index, paired in enumerate(paired_rows):
        paired.update(
            {
                "dataset": cell.dataset,
                "target_dimension": cell.target_dimension,
                "seed": SEEDS[index],
                "comparison_arm": arm,
                "training_precision": precision["training_precision"],
                "evaluation_precision": precision["evaluation_precision"],
                "native_nmse_ratio_candidate_over_fp32": native_nmse_ratio["values"][index],
                "native_nmse_paired_ratio_pass": (
                    native_nmse_ratio["values"][index]
                    <= PRACTICAL_MARGINS["native_nmse_paired_ratio_maximum"]
                ),
                "training_speedup_fp32_over_candidate": training_speedup["values"][index],
                "total_speedup_fp32_over_candidate": total_speedup["values"][index],
            }
        )
    topology = _compare_topology(
        root / fp32_topology,
        root / arm_topology,
        cell.target_dimension,
        reference_name=FP32,
        candidate_name=arm,
    )
    _relativize_topology_inputs(topology, root)
    return {
        "resolved_precision": precision,
        "verified_provenance": provenance,
        "statistics": statistics_by_measure,
        "derived": {
            f"native_nmse_ratio_{arm}_over_{FP32}": native_nmse_ratio,
            f"training_speedup_{FP32}_over_{arm}": training_speedup,
            f"total_speedup_{FP32}_over_{arm}": total_speedup,
            "selected_epochs": {
                FP32: [row["details"]["selected_epochs"] for row in fp32_rows],
                arm: [row["details"]["selected_epochs"] for row in arm_rows],
            },
        },
        "practical_noninferiority": noninferiority,
        "topology": topology,
    }, paired_rows


def _analyze_cell(cell: Cell, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = {
        FP32: root / cell.fp32_results,
        BF16_COUPLED: root / cell.bf16_coupled_results,
        BF16_TRAIN_FP32_EVAL: root / cell.bf16_train_fp32_eval_results,
    }
    payloads = {name: _load_json(path) for name, path in paths.items()}
    rows = {
        name: _ae_rows(
            payload,
            dataset=cell.dataset,
            target_dimension=cell.target_dimension,
            arm=name,
        )
        for name, payload in payloads.items()
    }
    _verify_precision_metadata(payloads[FP32], rows[FP32], FP32)
    topology_paths = {
        FP32: cell.fp32_topology,
        BF16_COUPLED: cell.bf16_coupled_topology,
        BF16_TRAIN_FP32_EVAL: cell.bf16_train_fp32_eval_topology,
    }
    comparisons = {}
    paired_rows = []
    for arm in BF16_ARMS:
        comparison, arm_paired_rows = _arm_comparison(
            cell=cell,
            root=root,
            arm=arm,
            fp32_payload=payloads[FP32],
            arm_payload=payloads[arm],
            fp32_rows=rows[FP32],
            arm_rows=rows[arm],
            fp32_topology=topology_paths[FP32],
            arm_topology=topology_paths[arm],
        )
        comparisons[arm] = comparison
        paired_rows.extend(arm_paired_rows)

    isolation_topology = _compare_topology(
        root / topology_paths[BF16_COUPLED],
        root / topology_paths[BF16_TRAIN_FP32_EVAL],
        cell.target_dimension,
        reference_name=BF16_COUPLED,
        candidate_name=BF16_TRAIN_FP32_EVAL,
    )
    _relativize_topology_inputs(isolation_topology, root)
    topology_attribution = []
    for metric in ("dtw_beta0", "dtw_beta1"):
        coupled = comparisons[BF16_COUPLED]["topology"]["pooled_over_fit_and_topology_seeds"][
            metric
        ]
        isolated = comparisons[BF16_TRAIN_FP32_EVAL]["topology"][
            "pooled_over_fit_and_topology_seeds"
        ][metric]
        fp32_mean = coupled[FP32]["mean"]
        coupled_shift = coupled[BF16_COUPLED]["mean"] - fp32_mean
        isolated_shift = isolated[BF16_TRAIN_FP32_EVAL]["mean"] - fp32_mean
        topology_attribution.append(
            {
                "metric": metric,
                "fp32_mean": fp32_mean,
                "coupled_bf16_shift_from_fp32": coupled_shift,
                "fp32_evaluation_isolation_shift_from_fp32": isolated_shift,
                "isolation_minus_coupled_mean": isolated_shift - coupled_shift,
                "isolation_is_closer_to_fp32": abs(isolated_shift) < abs(coupled_shift),
            }
        )

    selected_epochs_match = all(
        left["details"]["selected_epochs"] == right["details"]["selected_epochs"]
        for left, right in zip(rows[BF16_COUPLED], rows[BF16_TRAIN_FP32_EVAL], strict=True)
    )
    return {
        "dataset": cell.dataset,
        "target_dimension": cell.target_dimension,
        "seeds": list(SEEDS),
        "inputs": {
            name: {"path": _relative(path, root), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "comparisons_to_fp32": comparisons,
        "isolation_contrast": {
            "statistics": _named_statistics(
                rows[BF16_COUPLED],
                rows[BF16_TRAIN_FP32_EVAL],
                BF16_COUPLED,
                BF16_TRAIN_FP32_EVAL,
            ),
            "selected_epochs_identical": selected_epochs_match,
            "topology": isolation_topology,
            "topology_shift_attribution": topology_attribution,
        },
    }, paired_rows


def _csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    leading = [
        "dataset",
        "target_dimension",
        "seed",
        "comparison_arm",
        "training_precision",
        "evaluation_precision",
    ]
    derived = [
        "native_nmse_ratio_candidate_over_fp32",
        "native_nmse_paired_ratio_pass",
        "training_speedup_fp32_over_candidate",
        "total_speedup_fp32_over_candidate",
    ]
    measure_columns = []
    for measure in MEASURES:
        if not any(f"{measure.name}_fp32" in row for row in rows):
            continue
        measure_columns.extend(
            (
                f"{measure.name}_fp32",
                f"{measure.name}_candidate",
                f"{measure.name}_candidate_minus_fp32",
            )
        )
    return leading + derived + measure_columns


def analyze(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell_summaries = []
    paired_rows = []
    for cell in CELLS:
        summary, rows = _analyze_cell(cell, root)
        cell_summaries.append(summary)
        paired_rows.extend(rows)

    failed_checks = []
    attribution = []
    for cell in cell_summaries:
        attribution.extend(cell["isolation_contrast"]["topology_shift_attribution"])
        for arm in BF16_ARMS:
            checks = cell["comparisons_to_fp32"][arm]["practical_noninferiority"]["checks"]
            for check in checks:
                if not check["passed"]:
                    failed_checks.append(
                        {
                            "dataset": cell["dataset"],
                            "target_dimension": cell["target_dimension"],
                            "arm": arm,
                            **check,
                        }
                    )
    closer = sum(item["isolation_is_closer_to_fp32"] for item in attribution)
    coupled_abs = [abs(item["coupled_bf16_shift_from_fp32"]) for item in attribution]
    isolated_abs = [abs(item["fp32_evaluation_isolation_shift_from_fp32"]) for item in attribution]
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": 2,
        "analysis": "paired FP32 and two-arm BF16 autoencoder precision sensitivity",
        "precision_protocol": {
            FP32: "float32 training, validation, encoding, and evaluation",
            BF16_COUPLED: (
                "bfloat16 autocast training, validation, and encoding in the archived coupled arm"
            ),
            BF16_TRAIN_FP32_EVAL: (
                "bfloat16 autocast training with explicitly resolved float32 validation, "
                "encoding, and evaluation"
            ),
            "pairing": "same corpus, target dimension, seed, and evaluation samples",
            "timing_note": (
                "Speedups pair archived FP32 runs with later BF16 runs; they are not alternating "
                "contemporaneous timing replicates."
            ),
            "training_seconds_definition": "selection_seconds + refit_seconds",
            "total_seconds_definition": "details.fit_seconds",
            "speedup_definition": "FP32 seconds / candidate seconds",
            "primary_speedup_summaries": "paired median and paired geometric mean",
            "epoch_note": (
                "selected_epochs is the chosen full-data refit duration; the archived harness "
                "does not record selection-phase executed epochs or optimizer steps."
            ),
        },
        "practical_noninferiority_protocol": {
            "margins": PRACTICAL_MARGINS,
            "scope": (
                "The same practical/descriptive checks are applied separately to both BF16 arms; "
                "they are not formal statistical non-inferiority or equivalence tests."
            ),
            "topology": "descriptive only; no pass/fail DTW margin",
        },
        "provenance": {
            "analysis_script": _relative(script_path, root),
            "analysis_script_sha256": sha256_file(script_path),
        },
        "cells": cell_summaries,
        "output_quantization_assessment": {
            "topology_metric_cells": len(attribution),
            "fp32_evaluation_is_closer_to_fp32_count": closer,
            "fp32_evaluation_is_not_closer_count": len(attribution) - closer,
            "mean_absolute_coupled_bf16_shift": statistics.fmean(coupled_abs),
            "mean_absolute_fp32_evaluation_isolation_shift": statistics.fmean(isolated_abs),
            "conclusion": "not_supported_descriptively",
            "interpretation": (
                "FP32 evaluation moves the pooled topology DTW mean closer to FP32 in only "
                f"{closer} of {len(attribution)} corpus/width/Betti comparisons; the topology "
                "shifts therefore are not explained by quantization of the encoded output."
            ),
        },
        "overall": {
            "cells": len(cell_summaries),
            "paired_arm_comparisons": len(paired_rows),
            "all_input_and_pairing_checks_passed": True,
            "all_practical_noninferiority_checks_passed": not failed_checks,
            "failed_practical_checks": failed_checks,
            "topology_arm_comparisons": sum(
                cell["comparisons_to_fp32"][arm]["topology"]["status"] == "compared"
                for cell in cell_summaries
                for arm in BF16_ARMS
            ),
            "all_bf16_arm_selected_epochs_identical": all(
                cell["isolation_contrast"]["selected_epochs_identical"] for cell in cell_summaries
            ),
        },
    }
    return payload, paired_rows


def write_outputs(
    payload: dict[str, Any], paired_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    paired_path = output_dir / "paired-results.csv"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    columns = _csv_columns(paired_rows)
    with paired_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(paired_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing artifacts/ (default: inferred from this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: <repo-root>/artifacts/large-ae-precision)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "artifacts/large-ae-precision"
    )
    payload, paired_rows = analyze(root)
    write_outputs(payload, paired_rows, output_dir)
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'paired-results.csv'}")
    print(
        "practical checks: "
        f"{'PASS' if payload['overall']['all_practical_noninferiority_checks_passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
