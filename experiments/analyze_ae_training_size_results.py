#!/usr/bin/env python3
"""Create public-safe summaries and learning curves for the AE pool-size study.

The required input is the JSON emitted by ``compare_ae_training_sizes.py``.
The geometry result from ``evaluate_ae_training_size_geometry.py`` and one or
more multicutoff retrieval packages may be joined by ``(init_seed, pool_size)``.

The analyzer deliberately distinguishes three reconstruction estimands:

* tune-set quality after exactly one pass over each pool;
* tune-set quality after the common fixed optimizer-update budget; and
* legacy evaluation-set quality at the tune-selected checkpoint.

Outputs contain hashes and experimental identities, but never absolute input,
checkpoint, layout, or output paths.  The output directory receives
``aggregate.csv``, ``paired-contrasts.csv``, ``summary.json``, and matching PNG
and PDF learning-curve figures.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAINING_PROTOCOL = "deep1b-ae-training-pool-sensitivity-v1"
GEOMETRY_PROTOCOL = "deep1b-ae-training-pool-geometry-v1"
RETRIEVAL_PROTOCOL = "deep1b-ae-training-size-multicutoff-retrieval-v1"
PROTOCOL = "deep1b-ae-training-size-public-analysis-v1"
SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 128 * 2**20


@dataclass(frozen=True)
class MetricSpec:
    """Public definition and plotting metadata for one scalar response."""

    metric: str
    metric_group: str
    estimand: str
    display_name: str
    direction: str
    definition: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, block_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input is unexpectedly large: {path.name}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain one object: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _object(payload: Mapping[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _metric_specs() -> dict[str, MetricSpec]:
    specs = [
        MetricSpec(
            "one_pass_tune_reconstruction_nmse",
            "reconstruction",
            "one_pass_tune",
            "One-pass tune NMSE",
            "lower",
            "Tune-set reconstruction MSE divided by mean-predictor MSE immediately "
            "after one complete pass over the arm's training pool.",
        ),
        MetricSpec(
            "one_pass_tune_reconstruction_cosine",
            "reconstruction",
            "one_pass_tune",
            "One-pass tune cosine",
            "higher",
            "Mean tune-set cosine similarity immediately after one complete pass over "
            "the arm's training pool.",
        ),
        MetricSpec(
            "fixed_budget_tune_reconstruction_nmse",
            "reconstruction",
            "fixed_budget_tune",
            "Fixed-budget tune NMSE",
            "lower",
            "Tune-set normalized reconstruction MSE after the common fixed optimizer-update "
            "budget, irrespective of the checkpoint selected for final evaluation.",
        ),
        MetricSpec(
            "fixed_budget_tune_reconstruction_cosine",
            "reconstruction",
            "fixed_budget_tune",
            "Fixed-budget tune cosine",
            "higher",
            "Mean tune-set cosine similarity after the common fixed optimizer-update budget.",
        ),
        MetricSpec(
            "selected_checkpoint_legacy_eval_reconstruction_nmse",
            "reconstruction",
            "selected_checkpoint_legacy_eval",
            "Selected-checkpoint legacy-eval NMSE",
            "lower",
            "Normalized reconstruction MSE on the common legacy evaluation cohort, measured "
            "once after tune-only checkpoint selection.",
        ),
        MetricSpec(
            "selected_checkpoint_legacy_eval_reconstruction_cosine",
            "reconstruction",
            "selected_checkpoint_legacy_eval",
            "Selected-checkpoint legacy-eval cosine",
            "higher",
            "Mean cosine similarity on the common legacy evaluation cohort, measured once "
            "after tune-only checkpoint selection.",
        ),
        MetricSpec(
            "sample_conditional_trustworthiness",
            "geometry",
            "selected_checkpoint_legacy_eval_sample",
            "Trustworthiness",
            "higher",
            "Sample-conditional trustworthiness on the fixed geometry sample.",
        ),
        MetricSpec(
            "sample_conditional_continuity",
            "geometry",
            "selected_checkpoint_legacy_eval_sample",
            "Continuity",
            "higher",
            "Sample-conditional continuity on the fixed geometry sample.",
        ),
        MetricSpec(
            "sample_conditional_neighbor_overlap",
            "geometry",
            "selected_checkpoint_legacy_eval_sample",
            "Neighbor overlap",
            "higher",
            "Mean fixed-sample original/encoded k-nearest-neighbor overlap.",
        ),
        MetricSpec(
            "sampled_pair_distance_spearman",
            "geometry",
            "selected_checkpoint_legacy_eval_sample",
            "Pair-distance Spearman",
            "higher",
            "Spearman correlation of original and encoded distances on fixed sampled pairs.",
        ),
        MetricSpec(
            "sampled_pair_normalized_stress",
            "geometry",
            "selected_checkpoint_legacy_eval_sample",
            "Normalized stress",
            "lower",
            "Scale-fitted normalized stress on fixed sampled pairs.",
        ),
        MetricSpec(
            "sampled_local_decodability_nmse",
            "geometry",
            "selected_checkpoint_legacy_eval_sample",
            "Local decodability NMSE",
            "lower",
            "Distance-weighted kNN reconstruction NMSE on the fixed decoder sample.",
        ),
    ]
    return {spec.metric: spec for spec in specs}


def _retrieval_spec(cutoff: int) -> MetricSpec:
    return MetricSpec(
        f"recall_at_{cutoff}",
        "retrieval",
        "selected_checkpoint_deep1b_retrieval",
        f"Recall@{cutoff}",
        "higher",
        f"Strict original-ID intersection recall at cutoff {cutoff} on the common public "
        "Deep1B query prefix.",
    )


def _training_arms(
    payload: dict[str, Any], input_sha256: str, specs: dict[str, MetricSpec]
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if payload.get("protocol") != TRAINING_PROTOCOL or payload.get("status") != "ok":
        raise ValueError("training input is not a successful training-pool-size result")
    source = _object(payload, "source", context="training")
    split = _object(payload, "split", context="training")
    training_split = _object(split, "training", context="training.split")
    test_split = _object(split, "test", context="training.split")
    configuration = _object(payload, "protocol_configuration", context="training")
    fixed_update_budget = _positive_integer(
        configuration.get("update_budget"),
        name="training.protocol_configuration.update_budget",
    )
    raw_pool_sizes = training_split.get("pool_sizes")
    if not isinstance(raw_pool_sizes, list) or not raw_pool_sizes:
        raise ValueError("training.split.training.pool_sizes must be a non-empty list")
    pool_sizes = [_positive_integer(value, name="training pool size") for value in raw_pool_sizes]
    if len(pool_sizes) < 2 or pool_sizes != sorted(set(pool_sizes)):
        raise ValueError("training pool sizes must contain at least two strictly increasing values")
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("training.arms must be a non-empty list")

    arms: dict[tuple[int, int], dict[str, Any]] = {}
    observations = []
    for raw_arm in raw_arms:
        if not isinstance(raw_arm, dict):
            raise ValueError("every training arm must be an object")
        seed = _nonnegative_integer(raw_arm.get("init_seed"), name="training arm init_seed")
        pool = _positive_integer(raw_arm.get("pool_size"), name="training arm pool_size")
        identity = (seed, pool)
        if identity in arms or pool not in pool_sizes:
            raise ValueError(f"duplicate or unknown training arm {identity}")
        first_pass = _object(raw_arm, "first_pass", context=f"training arm {identity}")
        fixed_budget = _object(raw_arm, "fixed_budget", context=f"training arm {identity}")
        selection = _object(raw_arm, "selection", context=f"training arm {identity}")
        test = _object(raw_arm, "test", context=f"training arm {identity}")
        first_tune = _object(first_pass, "tune", context=f"training arm {identity}.first_pass")
        fixed_tune = _object(fixed_budget, "tune", context=f"training arm {identity}.fixed_budget")
        if not math.isclose(
            _finite(
                first_pass.get("equivalent_passes"),
                name=f"training arm {identity} first-pass equivalent passes",
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"training arm {identity} first-pass record is not one pass")
        if fixed_budget.get("update") != fixed_update_budget:
            raise ValueError(f"training arm {identity} fixed-budget update is inconsistent")
        if (
            test.get("evaluated_once") is not True
            or test.get("checkpoint_selected_without_test") is not True
        ):
            raise ValueError(f"training arm {identity} legacy evaluation policy is inconsistent")
        selected_update = _positive_integer(
            selection.get("best_update"), name=f"training arm {identity} selected update"
        )
        arm = {
            "init_seed": seed,
            "pool_size": pool,
            "selected_update": selected_update,
            "initial_state_sha256": _sha256(
                raw_arm.get("initial_state_sha256"),
                name=f"training arm {identity} initial state hash",
            ),
        }
        arms[identity] = arm
        values = {
            "one_pass_tune_reconstruction_nmse": first_tune.get("reconstruction_nmse"),
            "one_pass_tune_reconstruction_cosine": first_tune.get("reconstruction_cosine"),
            "fixed_budget_tune_reconstruction_nmse": fixed_tune.get("reconstruction_nmse"),
            "fixed_budget_tune_reconstruction_cosine": fixed_tune.get("reconstruction_cosine"),
            "selected_checkpoint_legacy_eval_reconstruction_nmse": test.get("reconstruction_nmse"),
            "selected_checkpoint_legacy_eval_reconstruction_cosine": test.get(
                "reconstruction_cosine"
            ),
        }
        for metric, value in values.items():
            observations.append(
                {
                    "init_seed": seed,
                    "pool_size": pool,
                    "metric": metric,
                    "value": _finite(value, name=f"training arm {identity} {metric}"),
                }
            )

    seeds = sorted({seed for seed, _ in arms})
    expected = {(seed, pool) for seed in seeds for pool in pool_sizes}
    if set(arms) != expected:
        raise ValueError("training result is not a complete seed-by-pool block")
    for seed in seeds:
        hashes = {arms[(seed, pool)]["initial_state_sha256"] for pool in pool_sizes}
        if len(hashes) != 1:
            raise ValueError(f"training arms for seed {seed} do not share initialization")

    public = {
        "file_sha256": input_sha256,
        "protocol": TRAINING_PROTOCOL,
        "arm_count": len(arms),
        "source": {
            "rows": _positive_integer(source.get("rows"), name="training.source.rows"),
            "dimension": _positive_integer(
                source.get("dimension"), name="training.source.dimension"
            ),
            "file_sha256": _sha256(source.get("file_sha256"), name="training.source.file_sha256"),
            "input_transformation": source.get("input_transformation"),
        },
        "legacy_evaluation": {
            "rows": _positive_integer(test_split.get("rows"), name="training.split.test.rows"),
            "indices_sha256": _sha256(
                test_split.get("indices_sha256"), name="training.split.test.indices_sha256"
            ),
        },
        "architecture": configuration.get("architecture"),
        "target_dimension": _positive_integer(
            configuration.get("target_dimension"),
            name="training.protocol_configuration.target_dimension",
        ),
        "fixed_update_budget": fixed_update_budget,
        "pool_sizes": pool_sizes,
        "init_seeds": seeds,
    }
    return arms, observations, public


def _geometry_observations(
    payload: dict[str, Any],
    input_sha256: str,
    training_sha256: str,
    training_arms: Mapping[tuple[int, int], dict[str, Any]],
    legacy_evaluation_indices_sha256: str,
    specs: dict[str, MetricSpec],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if payload.get("protocol") != GEOMETRY_PROTOCOL or payload.get("status") != "ok":
        raise ValueError("geometry input is not a successful training-size geometry result")
    link = _object(payload, "training_result", context="geometry")
    if link.get("file_sha256") != training_sha256:
        raise ValueError("geometry input does not identify the supplied training result hash")
    published = _object(payload, "published_protocol", context="geometry")
    if not bool(published.get("only_common_test_rows_encoded")):
        raise ValueError("geometry input did not encode only the common legacy cohort")
    geometry_test_hash = _sha256(
        published.get("test_indices_sha256"), name="geometry legacy-evaluation index hash"
    )
    if geometry_test_hash != legacy_evaluation_indices_sha256:
        raise ValueError("geometry input identifies another legacy evaluation cohort")
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list):
        raise ValueError("geometry.arms must be a list")
    observations = []
    identities: set[tuple[int, int]] = set()
    geometry_metrics = [metric for metric, spec in specs.items() if spec.metric_group == "geometry"]
    for arm in raw_arms:
        if not isinstance(arm, dict):
            raise ValueError("every geometry arm must be an object")
        seed = _nonnegative_integer(arm.get("init_seed"), name="geometry arm init_seed")
        pool = _positive_integer(arm.get("pool_size"), name="geometry arm pool_size")
        identity = (seed, pool)
        if identity in identities or identity not in training_arms:
            raise ValueError(f"duplicate or unknown geometry arm {identity}")
        identities.add(identity)
        training_record = _object(arm, "training_result", context=f"geometry arm {identity}")
        if training_record.get("best_update") != training_arms[identity]["selected_update"]:
            raise ValueError(f"geometry selected update disagrees for arm {identity}")
        metrics = _object(arm, "metrics", context=f"geometry arm {identity}")
        for metric in geometry_metrics:
            observations.append(
                {
                    "init_seed": seed,
                    "pool_size": pool,
                    "metric": metric,
                    "value": _finite(metrics.get(metric), name=f"geometry arm {identity} {metric}"),
                }
            )
    if identities != set(training_arms):
        raise ValueError("geometry input is not aligned with every training arm")
    return observations, {
        "file_sha256": input_sha256,
        "protocol": GEOMETRY_PROTOCOL,
        "arm_count": len(identities),
        "training_result_sha256_verified": True,
        "legacy_evaluation_indices_sha256": geometry_test_hash,
    }


def _possible_training_hash(payload: Mapping[str, Any]) -> str | None:
    candidates = []
    for field in ("training_result", "verification"):
        value = payload.get(field)
        if isinstance(value, dict):
            for key in (
                "file_sha256",
                "training_result_sha256",
                "training_results_sha256",
            ):
                if value.get(key) is not None:
                    candidates.append(value[key])
    if payload.get("training_result_sha256") is not None:
        candidates.append(payload["training_result_sha256"])
    if not candidates:
        return None
    checked = [_sha256(value, name="retrieval training-result hash") for value in candidates]
    if len(set(checked)) != 1:
        raise ValueError("retrieval package contains conflicting training-result hashes")
    return checked[0]


def _retrieval_observations(
    packages: list[tuple[dict[str, Any], str]],
    training_sha256: str,
    training_arms: Mapping[tuple[int, int], dict[str, Any]],
    specs: dict[str, MetricSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations = []
    provenance = []
    seen: set[tuple[int, int, int]] = set()
    common_query_count = None
    for package_number, (payload, input_sha256) in enumerate(packages, start=1):
        if payload.get("protocol") != RETRIEVAL_PROTOCOL or payload.get("status") != "ok":
            raise ValueError("retrieval input is not a successful multicutoff result")
        package_verification = _object(payload, "verification", context="retrieval")
        if package_verification.get("all_encoding_provenance_verified") is not True:
            raise ValueError("retrieval input did not verify every encoding provenance record")
        linked_hash = _possible_training_hash(payload)
        if linked_hash is not None and linked_hash != training_sha256:
            raise ValueError("retrieval input identifies a different training result hash")
        cutoffs_raw = payload.get("cutoffs")
        if not isinstance(cutoffs_raw, list) or not cutoffs_raw:
            raise ValueError("retrieval.cutoffs must be a non-empty list")
        cutoffs = sorted(
            {_positive_integer(value, name="retrieval cutoff") for value in cutoffs_raw}
        )
        query_count = _positive_integer(payload.get("query_count"), name="retrieval.query_count")
        if common_query_count is None:
            common_query_count = query_count
        elif query_count != common_query_count:
            raise ValueError("retrieval packages do not use one common query count")
        raw_arms = payload.get("arms")
        if not isinstance(raw_arms, list) or not raw_arms:
            raise ValueError("retrieval.arms must be a non-empty list")
        package_identities = set()
        for arm in raw_arms:
            if not isinstance(arm, dict):
                raise ValueError("every retrieval arm must be an object")
            selection = _object(arm, "selection", context="retrieval arm")
            seed = _nonnegative_integer(selection.get("init_seed"), name="retrieval init_seed")
            pool = _positive_integer(selection.get("pool_size"), name="retrieval pool_size")
            identity = (seed, pool)
            if identity not in training_arms:
                raise ValueError(f"unknown retrieval arm {identity}")
            if selection.get("selected_update") != training_arms[identity]["selected_update"]:
                raise ValueError(f"retrieval selected update disagrees for arm {identity}")
            verification = _object(arm, "verification", context=f"retrieval arm {identity}")
            arm_training_hash = _sha256(
                verification.get("training_result_sha256"),
                name=f"retrieval arm {identity} training-result hash",
            )
            if arm_training_hash != training_sha256:
                raise ValueError(f"retrieval arm {identity} identifies another training result")
            _sha256(
                verification.get("checkpoint_state_dict_sha256"),
                name=f"retrieval arm {identity} checkpoint-state hash",
            )
            _sha256(
                verification.get("encoding_provenance_sha256"),
                name=f"retrieval arm {identity} encoding-provenance hash",
            )
            quality = _object(arm, "quality", context=f"retrieval arm {identity}")
            recall = _object(quality, "recall_at", context=f"retrieval arm {identity}.quality")
            hits = quality.get("recall_hits")
            denominators = quality.get("recall_denominator")
            if not isinstance(hits, dict) or not isinstance(denominators, dict):
                raise ValueError(f"retrieval arm {identity} has no hit/denominator maps")
            for cutoff in cutoffs:
                duplicate_key = (seed, pool, cutoff)
                if duplicate_key in seen:
                    raise ValueError(f"duplicate retrieval observation {duplicate_key}")
                seen.add(duplicate_key)
                key = str(cutoff)
                value = _finite(recall.get(key), name=f"retrieval arm {identity} recall@{cutoff}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"retrieval arm {identity} recall@{cutoff} is outside [0,1]")
                hit_count = _nonnegative_integer(
                    hits.get(key), name=f"retrieval arm {identity} hits@{cutoff}"
                )
                denominator = _positive_integer(
                    denominators.get(key),
                    name=f"retrieval arm {identity} denominator@{cutoff}",
                )
                if denominator != query_count * cutoff:
                    raise ValueError(
                        f"retrieval arm {identity} denominator@{cutoff} disagrees with queries"
                    )
                if hit_count > denominator or not math.isclose(
                    value, hit_count / denominator, rel_tol=1e-12, abs_tol=1e-15
                ):
                    raise ValueError(f"retrieval arm {identity} recall@{cutoff} is inconsistent")
                metric = f"recall_at_{cutoff}"
                specs.setdefault(metric, _retrieval_spec(cutoff))
                observations.append(
                    {
                        "init_seed": seed,
                        "pool_size": pool,
                        "metric": metric,
                        "value": value,
                    }
                )
            package_identities.add(identity)
        provenance.append(
            {
                "package": package_number,
                "file_sha256": input_sha256,
                "protocol": RETRIEVAL_PROTOCOL,
                "arm_count": len(package_identities),
                "cutoffs": cutoffs,
                "query_count": query_count,
                "training_result_sha256_verified": True,
                "top_level_training_result_sha256_present": linked_hash is not None,
            }
        )
    return observations, provenance


def _aggregate(
    observations: list[dict[str, Any]], specs: Mapping[str, MetricSpec], pool_sizes: list[int]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for observation in observations:
        grouped[(observation["pool_size"], observation["metric"])].append(observation["value"])
    order = {metric: position for position, metric in enumerate(specs)}
    rows = []
    for (pool, metric), values in sorted(
        grouped.items(), key=lambda item: (pool_sizes.index(item[0][0]), order[item[0][1]])
    ):
        spec = specs[metric]
        rows.append(
            {
                "pool_size": pool,
                "metric_group": spec.metric_group,
                "estimand": spec.estimand,
                "metric": spec.metric,
                "display_name": spec.display_name,
                "direction": spec.direction,
                "n": len(values),
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return rows


def _paired_contrasts(
    observations: list[dict[str, Any]], specs: Mapping[str, MetricSpec], pool_sizes: list[int]
) -> list[dict[str, Any]]:
    values = {
        (item["init_seed"], item["pool_size"], item["metric"]): item["value"]
        for item in observations
    }
    metrics = {item["metric"] for item in observations}
    seeds = sorted({item["init_seed"] for item in observations})
    reference_pool = max(pool_sizes)
    order = {metric: position for position, metric in enumerate(specs)}
    rows = []
    for pool in pool_sizes:
        if pool == reference_pool:
            continue
        for metric in sorted(metrics, key=order.__getitem__):
            paired = [
                (values[(seed, pool, metric)], values[(seed, reference_pool, metric)])
                for seed in seeds
                if (seed, pool, metric) in values and (seed, reference_pool, metric) in values
            ]
            if not paired:
                continue
            deltas = [left - right for left, right in paired]
            spec = specs[metric]
            mean_delta = statistics.fmean(deltas)
            rows.append(
                {
                    "pool_size": pool,
                    "reference_pool_size": reference_pool,
                    "metric_group": spec.metric_group,
                    "estimand": spec.estimand,
                    "metric": spec.metric,
                    "display_name": spec.display_name,
                    "direction": spec.direction,
                    "n_pairs": len(paired),
                    "mean_at_pool_paired": statistics.fmean(left for left, _ in paired),
                    "mean_at_reference_paired": statistics.fmean(right for _, right in paired),
                    "mean_delta_pool_minus_reference": mean_delta,
                    "sample_sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
                    "mean_favorable_delta": (
                        mean_delta if spec.direction == "higher" else -mean_delta
                    ),
                }
            )
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(path, encoded)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode())


def _short_pool_size(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def _plot_learning_curves(
    aggregate: list[dict[str, Any]], destination_png: Path, destination_pdf: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["pool_size"], row["metric"]): row for row in aggregate}
    pools = sorted({row["pool_size"] for row in aggregate})
    available = {row["metric"] for row in aggregate}
    panels = [
        (
            "Reconstruction NMSE",
            [
                ("one_pass_tune_reconstruction_nmse", "One-pass tune"),
                ("fixed_budget_tune_reconstruction_nmse", "Fixed-budget tune"),
                (
                    "selected_checkpoint_legacy_eval_reconstruction_nmse",
                    "Selected checkpoint, legacy eval",
                ),
            ],
        )
    ]
    neighborhood = [
        ("sample_conditional_trustworthiness", "Trustworthiness"),
        ("sample_conditional_continuity", "Continuity"),
        ("sample_conditional_neighbor_overlap", "Neighbor overlap"),
        ("sampled_pair_distance_spearman", "Pair Spearman"),
    ]
    distortion = [
        ("sampled_pair_normalized_stress", "Normalized stress"),
        ("sampled_local_decodability_nmse", "Decoder NMSE"),
    ]
    retrieval = sorted(
        (
            (metric, metric.replace("recall_at_", "Recall@"))
            for metric in available
            if metric.startswith("recall_at_")
        ),
        key=lambda item: int(item[0].removeprefix("recall_at_")),
    )
    if any(metric in available for metric, _ in neighborhood):
        panels.append(("Geometry preservation (higher is better)", neighborhood))
    if any(metric in available for metric, _ in distortion):
        panels.append(("Geometry distortion (lower is better)", distortion))
    if retrieval:
        panels.append(("Strict Deep1B retrieval", retrieval))

    columns = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.4 * columns, 4.2 * rows), squeeze=False)
    flat_axes = list(axes.flat)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    markers = ("o", "s", "^", "D", "v")
    for axis, (panel_title, series) in zip(flat_axes, panels, strict=False):
        for position, (metric, label) in enumerate(series):
            points = [lookup[(pool, metric)] for pool in pools if (pool, metric) in lookup]
            if not points:
                continue
            x = [point["pool_size"] for point in points]
            y = [point["mean"] for point in points]
            yerr = [point["sample_sd"] or 0.0 for point in points]
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                label=label,
                color=colors[position % len(colors)],
                marker=markers[position % len(markers)],
                linewidth=1.8,
                markersize=5,
                capsize=3,
            )
        axis.set_xscale("log")
        axis.set_xticks(pools, [_short_pool_size(pool) for pool in pools])
        axis.set_xlabel("Unique training-pool rows")
        axis.set_title(panel_title)
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)
    for axis in flat_axes[len(panels) :]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Points are means; error bars are sample SD across paired initialization seeds.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    metadata = {
        "Title": title,
        "Author": "dimension-reduction-benchmarks contributors",
        "Subject": "Deep1B autoencoder training-pool-size sensitivity",
    }
    for destination, output_format in ((destination_png, "png"), (destination_pdf, "pdf")):
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            fig.savefig(
                temporary,
                format=output_format,
                dpi=180 if output_format == "png" else None,
                metadata=metadata,
                bbox_inches="tight",
            )
            os.replace(temporary, destination)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    plt.close(fig)


def _assert_public_safe(value: Any, location: str = "root") -> None:
    """Reject path-like strings before publishing the summary JSON."""

    if isinstance(value, dict):
        for key, item in value.items():
            _assert_public_safe(item, f"{location}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            _assert_public_safe(item, f"{location}[{position}]")
    elif isinstance(value, str):
        if value.startswith(("/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError(f"public summary contains an absolute path at {location}")


def _analysis_summary(
    aggregate: list[dict[str, Any]], contrasts: list[dict[str, Any]], pool_sizes: list[int]
) -> dict[str, Any]:
    selected_metric = "selected_checkpoint_legacy_eval_reconstruction_nmse"
    selected = [row for row in aggregate if row["metric"] == selected_metric]
    best = min(selected, key=lambda row: row["mean"])
    smallest = min(pool_sizes)
    smallest_contrasts = [row for row in contrasts if row["pool_size"] == smallest]
    return {
        "best_mean_selected_checkpoint_legacy_eval_nmse": {
            "pool_size": best["pool_size"],
            "mean": best["mean"],
            "sample_sd": best["sample_sd"],
            "n": best["n"],
            "selection_is_descriptive_not_inferential": True,
        },
        "smallest_vs_largest_paired": smallest_contrasts,
        "interpretation_guardrails": [
            "Pool-size contrasts are paired within initialization seed.",
            "Sample SD describes the observed initialization seeds; it is not a confidence interval.",
            "One-pass tune, fixed-budget tune, and selected-checkpoint legacy-eval rows are distinct estimands.",
            "Optional geometry metrics condition on fixed legacy-evaluation subsets.",
            "Optional retrieval recall is a strict original-ID metric on public queries.",
        ],
    }


def analyze(args) -> dict[str, Any]:
    started = time.perf_counter()
    training, training_sha256 = _load_json(args.training_results)
    specs = _metric_specs()
    training_arms, observations, training_public = _training_arms(training, training_sha256, specs)
    geometry_public = None
    if args.geometry_results is not None:
        geometry, geometry_sha256 = _load_json(args.geometry_results)
        geometry_observations, geometry_public = _geometry_observations(
            geometry,
            geometry_sha256,
            training_sha256,
            training_arms,
            training_public["legacy_evaluation"]["indices_sha256"],
            specs,
        )
        observations.extend(geometry_observations)

    retrieval_packages = [_load_json(path) for path in args.retrieval_results]
    retrieval_public = []
    if retrieval_packages:
        retrieval_observations, retrieval_public = _retrieval_observations(
            retrieval_packages, training_sha256, training_arms, specs
        )
        observations.extend(retrieval_observations)

    pool_sizes = training_public["pool_sizes"]
    aggregate = _aggregate(observations, specs, pool_sizes)
    contrasts = _paired_contrasts(observations, specs, pool_sizes)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "aggregate_csv": output_dir / "aggregate.csv",
        "paired_contrasts_csv": output_dir / "paired-contrasts.csv",
        "summary_json": output_dir / "summary.json",
        "learning_curves_png": output_dir / "learning-curves.png",
        "learning_curves_pdf": output_dir / "learning-curves.pdf",
    }
    if not args.force:
        existing = [path.name for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"analysis outputs exist; pass --force: {', '.join(existing)}")

    _atomic_csv(outputs["aggregate_csv"], aggregate)
    _atomic_csv(outputs["paired_contrasts_csv"], contrasts)
    _plot_learning_curves(
        aggregate,
        outputs["learning_curves_png"],
        outputs["learning_curves_pdf"],
        args.title,
    )
    artifacts = {
        name: {"filename": path.name, "sha256": _sha256_file(path)}
        for name, path in outputs.items()
        if name != "summary_json"
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "generated_at_utc": _utc_now(),
        "title": args.title,
        "public_safety": {
            "absolute_paths_omitted": True,
            "checkpoint_paths_omitted": True,
            "layout_paths_omitted": True,
            "source_local_path_omitted": True,
        },
        "analyzer": {
            "source_sha256": _sha256_file(Path(__file__).resolve()),
            "schema_version": SCHEMA_VERSION,
        },
        "inputs": {
            "training": training_public,
            "geometry": geometry_public,
            "retrieval": retrieval_public,
        },
        "design": {
            "pairing_unit": "initialization seed",
            "pool_sizes": pool_sizes,
            "init_seeds": training_public["init_seeds"],
            "largest_pool_reference": max(pool_sizes),
            "mean_and_sample_sd_by_pool": True,
            "paired_delta_definition": "pool value minus largest-pool value within seed",
            "favorable_delta_definition": "delta for higher-is-better metrics; negated delta for lower-is-better metrics",
        },
        "metric_catalog": [asdict(spec) for spec in specs.values()],
        "aggregates": aggregate,
        "paired_contrasts": contrasts,
        "analysis": _analysis_summary(aggregate, contrasts, pool_sizes),
        "artifacts": artifacts,
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    _assert_public_safe(summary)
    _atomic_json(outputs["summary_json"], summary)
    return summary


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--training-results", required=True, type=Path)
    parser.add_argument("--geometry-results", type=Path)
    parser.add_argument("--retrieval-results", type=Path, nargs="*", default=())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--title", default="Deep1B autoencoder training-pool-size sensitivity")
    args = parser.parse_args(argv)
    args.training_results = args.training_results.expanduser().resolve()
    if args.geometry_results is not None:
        args.geometry_results = args.geometry_results.expanduser().resolve()
    args.retrieval_results = tuple(path.expanduser().resolve() for path in args.retrieval_results)
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.title.strip():
        parser.error("--title must not be empty")
    _assert_public_safe(args.title, "title")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args)
    print(
        f"wrote {len(summary['aggregates'])} aggregate rows and "
        f"{len(summary['paired_contrasts'])} paired contrasts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
