#!/usr/bin/env python3
"""Evaluate geometry of completed Deep1B AE training-size checkpoints.

This is a deliberately post-hoc, layout-free evaluator.  It reconstructs the
published 65,536-row Deep1B validation cohort from its seed, verifies its pinned
SHA-256 identity, loads every tune-selected checkpoint from
``compare_ae_training_sizes.py``, and encodes only those common held-out rows.
The paper's sample-conditional geometry metrics are then evaluated on fixed
subsets of that cohort.

Example::

    python experiments/evaluate_ae_training_size_geometry.py \
      --results /tmp/ae-training-size-final/results.json \
      --checkpoint-dir /tmp/ae-training-size-final/checkpoints \
      --data deep-base-prefix-10m.fbin \
      --output-dir /tmp/ae-training-size-final/geometry

No full-corpus layout, retrieval result, or topology artifact is produced.
The only layouts exist transiently in memory and contain the common test rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from experiments.compare_ae_training_sizes import (
        DEFAULT_TEST_ROWS,
        DEFAULT_UNIVERSE_ROWS,
        PUBLISHED_TEST_INDICES_SHA256,
        PUBLISHED_TEST_SEED,
        _sha256_indices,
        _state_sha256,
    )
    from experiments.compare_ae_training_sizes import (
        PROTOCOL as TRAINING_PROTOCOL,
    )
    from experiments.compare_large_embeddings import (
        _sha256_bytes,
        deterministic_sample,
        distance_fidelity,
        index_sha256,
        local_decodability_nmse,
        sample_conditional_metrics,
        sample_pairs,
        sampled_pair_distances,
    )
    from experiments.compare_yandex_streaming import BoundedFbinReader, _build_autoencoder
    from experiments.yandex_fbin import read_fbin_header
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from compare_ae_training_sizes import (  # type: ignore[no-redef]
        DEFAULT_TEST_ROWS,
        DEFAULT_UNIVERSE_ROWS,
        PUBLISHED_TEST_INDICES_SHA256,
        PUBLISHED_TEST_SEED,
        _sha256_indices,
        _state_sha256,
    )
    from compare_ae_training_sizes import (
        PROTOCOL as TRAINING_PROTOCOL,
    )
    from compare_large_embeddings import (  # type: ignore[no-redef]
        _sha256_bytes,
        deterministic_sample,
        distance_fidelity,
        index_sha256,
        local_decodability_nmse,
        sample_conditional_metrics,
        sample_pairs,
        sampled_pair_distances,
    )
    from compare_yandex_streaming import (  # type: ignore[no-redef]
        BoundedFbinReader,
        _build_autoencoder,
    )
    from yandex_fbin import read_fbin_header  # type: ignore[no-redef]


PROTOCOL = "deep1b-ae-training-pool-geometry-v1"
SCHEMA_VERSION = 1
DEFAULT_EVALUATION_SEED = 2718
DEFAULT_NEIGHBORS = 16
DEFAULT_METRIC_SAMPLE = 5_000
DEFAULT_DECODER_SAMPLE = 10_000
DEFAULT_DISTANCE_PAIRS = 200_000
GEOMETRY_METRIC_KEYS = (
    "sample_conditional_trustworthiness",
    "sample_conditional_continuity",
    "sample_conditional_neighbor_overlap",
    "sampled_pair_distance_spearman",
    "sampled_pair_normalized_stress",
    "sampled_local_decodability_nmse",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, serialized.encode())


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty geometry CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode())


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError(f"JSON input is unexpectedly large: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain one object: {path}")
    return payload


def _validate_published_configuration(args) -> None:
    if args.allow_nonpublished_cohort:
        return
    expected = {
        "evaluation_seed": DEFAULT_EVALUATION_SEED,
        "neighbors": DEFAULT_NEIGHBORS,
        "metric_sample": DEFAULT_METRIC_SAMPLE,
        "decoder_sample": DEFAULT_DECODER_SAMPLE,
        "distance_pairs": DEFAULT_DISTANCE_PAIRS,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise ValueError(
            "published mode fixes all metric samples; use the exact defaults "
            f"({expected}), observed {observed}"
        )


def _validate_training_result(payload: dict[str, Any], args) -> tuple[dict, list[dict]]:
    if payload.get("status") != "ok" or payload.get("protocol") != TRAINING_PROTOCOL:
        raise ValueError("--results is not a successful AE training-pool-size result")
    source = payload.get("source")
    split = payload.get("split")
    configuration = payload.get("protocol_configuration")
    arms = payload.get("arms")
    if not all(isinstance(item, dict) for item in (source, split, configuration)):
        raise ValueError("training result is missing source, split, or protocol configuration")
    if not isinstance(arms, list) or not arms or not all(isinstance(arm, dict) for arm in arms):
        raise ValueError("training result has no arms")
    test = split.get("test")
    if not isinstance(test, dict):
        raise ValueError("training result has no test split")

    published_fields = (
        int(source.get("universe_rows", -1)) == DEFAULT_UNIVERSE_ROWS
        and int(test.get("rows", -1)) == DEFAULT_TEST_ROWS
        and int(test.get("seed", -1)) == PUBLISHED_TEST_SEED
        and test.get("indices_sha256") == PUBLISHED_TEST_INDICES_SHA256
    )
    if not args.allow_nonpublished_cohort and not published_fields:
        raise ValueError("training result does not identify the pinned published test cohort")
    if int(test.get("rows", -1)) < 3:
        raise ValueError("training result test cohort is too small")
    if int(source.get("dimension", -1)) < 2:
        raise ValueError("training result source dimension is invalid")
    if int(configuration.get("target_dimension", -1)) < 1:
        raise ValueError("training result target dimension is invalid")
    hidden = configuration.get("hidden_dimensions")
    if not isinstance(hidden, list) or not hidden or any(int(width) < 1 for width in hidden):
        raise ValueError("training result hidden dimensions are invalid")

    identities: set[tuple[int, int]] = set()
    for arm in arms:
        identity = (int(arm.get("init_seed", -1)), int(arm.get("pool_size", -1)))
        if min(identity) < 0 or identity in identities:
            raise ValueError("training result arm identities must be unique non-negative pairs")
        identities.add(identity)
        selection = arm.get("selection")
        if not isinstance(selection, dict) or int(selection.get("best_update", -1)) < 1:
            raise ValueError(f"training arm {identity} has no valid tune-selected update")
    return configuration, arms


def reconstruct_test_indices(training: dict[str, Any], *, require_published: bool) -> np.ndarray:
    source = training["source"]
    recorded = training["split"]["test"]
    universe_rows = int(source["universe_rows"])
    test_rows = int(recorded["rows"])
    test_seed = int(recorded["seed"])

    # Import locally to make the dependency on the original bounded sampler explicit.
    try:
        from experiments.compare_yandex_streaming import bounded_validation_indices
    except ModuleNotFoundError:
        from compare_yandex_streaming import bounded_validation_indices  # type: ignore

    indices = bounded_validation_indices(universe_rows, test_rows, test_seed)
    observed_hash = _sha256_indices(indices)
    if observed_hash != recorded.get("indices_sha256"):
        raise ValueError("recomputed test indices disagree with the training result SHA-256")
    if require_published and observed_hash != PUBLISHED_TEST_INDICES_SHA256:
        raise ValueError("recomputed test indices disagree with the pinned published SHA-256")
    return indices


def _validate_source(data: Path, training: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    info = read_fbin_header(data)
    expected = training["source"]
    observed_bytes = int(data.stat().st_size)
    if (
        int(info.rows) != int(expected.get("rows", -1))
        or int(info.dimension) != int(expected.get("dimension", -1))
        or observed_bytes != int(expected.get("file_bytes", -1))
    ):
        raise ValueError("Deep1B fbin metadata disagrees with the completed training result")
    started = time.perf_counter()
    digest = _sha256_file(data)
    hash_seconds = time.perf_counter() - started
    if digest != expected.get("file_sha256"):
        raise ValueError("Deep1B fbin SHA-256 disagrees with the completed training result")
    return info, {
        "path": str(data.resolve()),
        "rows": int(info.rows),
        "dimension": int(info.dimension),
        "file_bytes": observed_bytes,
        "file_sha256": digest,
        "file_sha256_seconds": hash_seconds,
        "identity_matches_training_result": True,
        "input_transformation": expected.get("input_transformation"),
    }


def make_metric_cohorts(test_indices: np.ndarray, args) -> dict[str, Any]:
    rows = int(test_indices.size)
    if min(args.metric_sample, args.decoder_sample) > rows:
        raise ValueError("metric and decoder samples must fit in the common test cohort")
    geometry = deterministic_sample(rows, args.metric_sample, args.evaluation_seed)
    decoder = deterministic_sample(rows, args.decoder_sample, args.evaluation_seed + 1)
    pairs = sample_pairs(rows, args.distance_pairs, args.evaluation_seed + 2)
    pair_hash = _sha256_bytes(
        np.ascontiguousarray(pairs[0], dtype="<i8").tobytes(),
        np.ascontiguousarray(pairs[1], dtype="<i8").tobytes(),
    )
    global_pairs = (test_indices[pairs[0]], test_indices[pairs[1]])
    global_pair_hash = _sha256_bytes(
        np.ascontiguousarray(global_pairs[0], dtype="<i8").tobytes(),
        np.ascontiguousarray(global_pairs[1], dtype="<i8").tobytes(),
    )
    return {
        "geometry_indices": geometry,
        "decoder_indices": decoder,
        "pairs": pairs,
        "provenance": {
            "evaluation_seed": int(args.evaluation_seed),
            "neighbors": int(args.neighbors),
            "geometry_sample": {
                "rows": int(geometry.size),
                "local_index_sha256": index_sha256(geometry),
                "global_deep1b_index_sha256": _sha256_indices(test_indices[geometry]),
            },
            "decoder_sample": {
                "rows": int(decoder.size),
                "local_index_sha256": index_sha256(decoder),
                "global_deep1b_index_sha256": _sha256_indices(test_indices[decoder]),
                "split_seed": int(args.evaluation_seed),
            },
            "distance_pairs": {
                "count": int(args.distance_pairs),
                "local_endpoint_pair_sha256": pair_hash,
                "global_deep1b_endpoint_pair_sha256": global_pair_hash,
            },
        },
    }


def _load_checkpoint(torch, path: Path, arm: dict[str, Any]) -> tuple[dict, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older local PyTorch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(f"checkpoint has no model_state_dict: {path}")
    expected = {
        "protocol": TRAINING_PROTOCOL,
        "init_seed": int(arm["init_seed"]),
        "pool_size": int(arm["pool_size"]),
        "selected_update": int(arm["selection"]["best_update"]),
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"checkpoint metadata disagrees with arm: {observed} != {expected}")
    state = payload["model_state_dict"]
    return state, {
        "path": str(path.resolve()),
        "file_bytes": int(path.stat().st_size),
        "file_sha256": _sha256_file(path),
        "model_state_sha256": _state_sha256(torch, state),
        "metadata_verified": True,
    }


def _encode_test_rows(
    torch, model, X: np.ndarray, device, batch_size: int
) -> tuple[np.ndarray, float]:
    output_dimension = int(model.encoder[-1].out_features)
    layout = np.empty((X.shape[0], output_dimension), dtype=np.float32)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            batch = torch.from_numpy(np.ascontiguousarray(X[start:stop])).to(device)
            encoded = model.encoder(batch).float()
            if not bool(torch.all(torch.isfinite(encoded)).item()):
                raise ValueError("checkpoint encoder produced non-finite coordinates")
            layout[start:stop] = encoded.cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return layout, time.perf_counter() - started


def _evaluate_metrics(X: np.ndarray, layout: np.ndarray, cohorts: dict, args) -> dict[str, Any]:
    metrics = sample_conditional_metrics(
        X,
        layout,
        cohorts["geometry_indices"],
        args.neighbors,
        n_jobs=args.metric_jobs,
    )
    metrics["sampled_local_decodability_nmse"] = float(
        local_decodability_nmse(
            X,
            layout,
            cohorts["decoder_indices"],
            args.neighbors,
            args.evaluation_seed,
            n_jobs=args.metric_jobs,
        )
    )
    reference, embedded = sampled_pair_distances(X, layout, cohorts["pairs"])
    correlation, stress = distance_fidelity(reference, embedded)
    metrics.update(
        {
            "sampled_pair_distance_spearman": correlation,
            "sampled_pair_normalized_stress": stress,
            "decoder_sample_rows": int(cohorts["decoder_indices"].size),
            "decoder_sample_index_sha256": index_sha256(cohorts["decoder_indices"]),
            "distance_pair_count": int(cohorts["pairs"][0].size),
            "distance_pair_sha256": cohorts["provenance"]["distance_pairs"][
                "local_endpoint_pair_sha256"
            ],
        }
    )
    return metrics


def _pair_arms(arms: list[dict[str, Any]], pool_sizes: list[int]) -> list[dict[str, Any]]:
    groups = []
    for init_seed in sorted({int(arm["init_seed"]) for arm in arms}):
        paired = sorted(
            (arm for arm in arms if int(arm["init_seed"]) == init_seed),
            key=lambda arm: int(arm["pool_size"]),
        )
        hashes = {arm["initial_state_sha256"] for arm in paired}
        observed_sizes = [int(arm["pool_size"]) for arm in paired]
        if len(hashes) != 1:
            raise ValueError(f"seed {init_seed} is not paired by common initialization")
        if observed_sizes != pool_sizes:
            raise ValueError(f"seed {init_seed} does not contain every common pool-size arm")
        groups.append(
            {
                "pair_id": f"initialization-seed-{init_seed}",
                "init_seed": init_seed,
                "initial_state_sha256": next(iter(hashes)),
                "pool_sizes": observed_sizes,
                "arm_ids": [arm["arm_id"] for arm in paired],
                "common_initialization_verified": True,
                "complete_pool_size_block_verified": True,
            }
        )
    return groups


def _attach_paired_deltas(arms: list[dict[str, Any]]) -> None:
    for init_seed in sorted({int(arm["init_seed"]) for arm in arms}):
        paired = [arm for arm in arms if int(arm["init_seed"]) == init_seed]
        reference = max(paired, key=lambda arm: int(arm["pool_size"]))
        for arm in paired:
            arm["paired_comparison"] = {
                "reference_arm_id": reference["arm_id"],
                "reference_pool_size": int(reference["pool_size"]),
                "metric_delta_arm_minus_reference": {
                    key: float(arm["metrics"][key] - reference["metrics"][key])
                    for key in GEOMETRY_METRIC_KEYS
                },
            }


def _summary_row(arm: dict[str, Any]) -> dict[str, Any]:
    training = arm["training_result"]
    metrics = arm["metrics"]
    deltas = arm["paired_comparison"]["metric_delta_arm_minus_reference"]
    row = {
        "arm_id": arm["arm_id"],
        "pair_id": arm["pair_id"],
        "init_seed": arm["init_seed"],
        "pool_size": arm["pool_size"],
        "best_update": training["best_update"],
        "training_result_test_nmse": training["test_nmse"],
        "training_result_test_cosine": training["test_cosine"],
        "checkpoint_sha256": arm["checkpoint"]["file_sha256"],
        "model_state_sha256": arm["checkpoint"]["model_state_sha256"],
        "test_layout_sha256": arm["encoding"]["layout_sha256"],
        "encoded_rows": arm["encoding"]["rows"],
        "encoding_seconds": arm["encoding"]["seconds"],
        "paired_reference_pool_size": arm["paired_comparison"]["reference_pool_size"],
    }
    for key in GEOMETRY_METRIC_KEYS:
        row[key] = metrics[key]
        row[f"delta_to_largest_pool__{key}"] = deltas[key]
    return row


def run_evaluation(args) -> dict[str, Any]:
    import scipy
    import sklearn
    import torch

    started_at = _utc_now()
    total_started = time.perf_counter()
    _validate_published_configuration(args)
    training = _load_json_object(args.results)
    configuration, training_arms = _validate_training_result(training, args)
    test_indices = reconstruct_test_indices(
        training, require_published=not args.allow_nonpublished_cohort
    )
    info, source = _validate_source(args.data, training)
    if test_indices[-1] >= int(info.rows):
        raise ValueError("recomputed test index exceeds the supplied fbin")

    with BoundedFbinReader(args.data, info, rows=int(info.rows)) as reader:
        X = reader.read_rows(test_indices)
    if X.shape != (test_indices.size, int(info.dimension)) or not np.all(np.isfinite(X)):
        raise ValueError("held-out source matrix is invalid")
    test_matrix_sha256 = hashlib.sha256(np.ascontiguousarray(X, dtype="<f4").tobytes()).hexdigest()
    row_norms = np.linalg.norm(X.astype(np.float64), axis=1)
    cohorts = make_metric_cohorts(test_indices, args)

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)

    input_dimension = int(info.dimension)
    target_dimension = int(configuration["target_dimension"])
    hidden_dimensions = tuple(int(width) for width in configuration["hidden_dimensions"])
    arms: list[dict[str, Any]] = []
    for training_arm in training_arms:
        init_seed = int(training_arm["init_seed"])
        pool_size = int(training_arm["pool_size"])
        checkpoint_path = args.checkpoint_dir / f"best-seed-{init_seed}-pool-{pool_size}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing completed arm checkpoint: {checkpoint_path}")
        state, checkpoint = _load_checkpoint(torch, checkpoint_path, training_arm)
        model = _build_autoencoder(torch, input_dimension, target_dimension, hidden_dimensions).to(
            device
        )
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"checkpoint model shape disagrees with training result: {error}"
            ) from error
        layout, encoding_seconds = _encode_test_rows(
            torch, model, X, device, args.encoding_batch_size
        )
        metrics_started = time.perf_counter()
        metrics = _evaluate_metrics(X, layout, cohorts, args)
        metrics_seconds = time.perf_counter() - metrics_started
        arm_id = f"seed-{init_seed}-pool-{pool_size}"
        arms.append(
            {
                "arm_id": arm_id,
                "pair_id": f"initialization-seed-{init_seed}",
                "init_seed": init_seed,
                "pool_size": pool_size,
                "initial_state_sha256": training_arm["initial_state_sha256"],
                "pool_indices_sha256": training_arm["pool_indices_sha256"],
                "training_result": {
                    "best_update": int(training_arm["selection"]["best_update"]),
                    "test_nmse": float(training_arm["test"]["reconstruction_nmse"]),
                    "test_cosine": float(training_arm["test"]["reconstruction_cosine"]),
                    "test_checkpoint_selected_without_test": bool(
                        training_arm["test"]["checkpoint_selected_without_test"]
                    ),
                },
                "checkpoint": checkpoint,
                "encoding": {
                    "rows": int(layout.shape[0]),
                    "dimensions": int(layout.shape[1]),
                    "dtype": str(layout.dtype),
                    "layout_sha256": hashlib.sha256(
                        np.ascontiguousarray(layout, dtype="<f4").tobytes()
                    ).hexdigest(),
                    "seconds": encoding_seconds,
                    "scope": "common held-out test cohort only",
                    "retention": "in-memory until this arm's metrics completed",
                },
                "metrics": metrics,
                "timing_seconds": {"geometry_metrics": metrics_seconds},
            }
        )
        del layout, model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pool_sizes = sorted({int(arm["pool_size"]) for arm in arms})
    pairing = _pair_arms(arms, pool_sizes)
    _attach_paired_deltas(arms)
    training_result_hash = _sha256_file(args.results)
    compare_module_path = Path(__file__).with_name("compare_large_embeddings.py")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "published_protocol": {
            "required": not args.allow_nonpublished_cohort,
            "verified": not args.allow_nonpublished_cohort,
            "test_rows": int(test_indices.size),
            "test_seed": int(training["split"]["test"]["seed"]),
            "test_indices_sha256": _sha256_indices(test_indices),
            "expected_test_indices_sha256": (
                PUBLISHED_TEST_INDICES_SHA256 if not args.allow_nonpublished_cohort else None
            ),
            "only_common_test_rows_encoded": True,
            "full_layouts_written": False,
            "retrieval_evaluated": False,
            "topology_evaluated": False,
        },
        "training_result": {
            "path": str(args.results.resolve()),
            "file_sha256": training_result_hash,
            "protocol": training["protocol"],
            "started_at_utc": training.get("started_at_utc"),
            "finished_at_utc": training.get("finished_at_utc"),
            "arm_count": len(training_arms),
        },
        "source": source,
        "test_data": {
            "rows": int(X.shape[0]),
            "dimension": int(X.shape[1]),
            "matrix_sha256": test_matrix_sha256,
            "global_indices_sha256": _sha256_indices(test_indices),
            "global_minimum_index": int(test_indices[0]),
            "global_maximum_index": int(test_indices[-1]),
            "row_l2_norm": {
                "minimum": float(np.min(row_norms)),
                "maximum": float(np.max(row_norms)),
                "mean": float(np.mean(row_norms)),
            },
            "input_transformation": "none",
        },
        "metric_protocol": {
            **cohorts["provenance"],
            "definitions_reused_from": "experiments.compare_large_embeddings",
            "compare_large_embeddings_sha256": _sha256_file(compare_module_path),
            "sample_conditioning": "all metric positions index the common held-out cohort",
            "distance": "Euclidean in the untransformed input and encoded coordinates",
            "local_decodability": "whitened-layout distance-weighted kNN regression",
        },
        "pairing": {
            "unit": "common initialization state within init_seed",
            "reference_for_deltas": "largest training pool within the same initialization",
            "pool_sizes": pool_sizes,
            "groups": pairing,
        },
        "arms": arms,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "timing_seconds": {"total": time.perf_counter() - total_started},
    }


def execute(args) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "geometry-results.json"
    csv_path = output_dir / "geometry-results.csv"
    if not args.force and (result_path.exists() or csv_path.exists()):
        raise FileExistsError("geometry output exists; pass --force to replace it")
    result = run_evaluation(args)
    _atomic_csv(csv_path, [_summary_row(arm) for arm in result["arms"]])
    result["artifacts"] = {
        "csv_path": str(csv_path),
        "csv_sha256": _sha256_file(csv_path),
        "json_path": str(result_path),
        "retained_layout_paths": [],
        "checkpoint_files_modified": False,
    }
    _atomic_json(result_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--encoding-batch-size", type=int, default=8192)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--metric-sample", type=int, default=DEFAULT_METRIC_SAMPLE)
    parser.add_argument("--decoder-sample", type=int, default=DEFAULT_DECODER_SAMPLE)
    parser.add_argument("--distance-pairs", type=int, default=DEFAULT_DISTANCE_PAIRS)
    parser.add_argument("--metric-jobs", type=int, default=1)
    parser.add_argument(
        "--allow-nonpublished-cohort",
        action="store_true",
        help="permit smaller non-published cohorts and metric samples for smoke tests only",
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.encoding_batch_size,
            args.neighbors,
            args.metric_sample,
            args.decoder_sample,
            args.distance_pairs,
            args.metric_jobs,
        )
        < 1
    ):
        parser.error(
            "batch sizes, sample counts, neighbors, pair counts, and jobs must be positive"
        )
    for name in ("results", "checkpoint_dir", "data", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def _test_args(**overrides):
    defaults = {
        "results": Path("results.json"),
        "checkpoint_dir": Path("checkpoints"),
        "data": Path("data.fbin"),
        "output_dir": Path("geometry"),
        "force": False,
        "device": "cpu",
        "encoding_batch_size": 16,
        "evaluation_seed": DEFAULT_EVALUATION_SEED,
        "neighbors": 4,
        "metric_sample": 12,
        "decoder_sample": 16,
        "distance_pairs": 100,
        "metric_jobs": 1,
        "allow_nonpublished_cohort": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute(args)
    print(f"wrote {len(result['arms'])} geometry arms to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
