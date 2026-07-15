#!/usr/bin/env python3
"""Deterministic Deep1B autoencoder optimizer-update learning curves.

The training-size experiment showed that a small, recycled pool can match a
10-million-row pool when both receive the same number of optimizer updates.
This harness asks the complementary question: how many updates are actually
needed?  For each pool and initialization seed it trains exactly one model
trajectory and inspects a predeclared sequence of update counts.

The protocol deliberately separates discovery from confirmation.  Discovery
uses only tune reconstruction and tune-cohort geometry; the fixed legacy
cohort is never materialized.  Confirmation accepts three locked update counts
(candidate, local step, and arbitrary reference), derives the lower/upper
bracket, includes update 1204 only as a replay anchor, and evaluates only those
locked checkpoints on fresh seeds and the legacy cohort.  Requested small
checkpoint files may be retained for a later retrieval pass.  Encoded layouts
are always hashed in memory and discarded.

Example::

    python experiments/compare_ae_update_budgets.py \
      --data deep-base-prefix-10m.fbin --phase discovery --device cuda \
      --output-dir artifacts/deep1b-ae-update-budget
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import random
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# Deterministic CUDA GEMMs require this before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    from experiments.compare_ae_training_sizes import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_INIT_SEEDS,
        DEFAULT_POOL_SEED,
        DEFAULT_TEST_ROWS,
        DEFAULT_TUNE_ROWS,
        DEFAULT_TUNE_SEED,
        DEFAULT_UNIVERSE_ROWS,
        PUBLISHED_TEST_INDICES_SHA256,
        PUBLISHED_TEST_SEED,
        _arguments_json,
        _atomic_csv,
        _atomic_json,
        _epoch_order,
        _evaluate,
        _resolve_device,
        _sha256_file,
        _sha256_indices,
        _state_sha256,
        _sync,
        inspect_source,
        load_resident_source,
        make_experiment_indices,
    )
    from experiments.compare_yandex_streaming import _build_autoencoder
    from experiments.evaluate_ae_training_size_geometry import (
        DEFAULT_DECODER_SAMPLE,
        DEFAULT_DISTANCE_PAIRS,
        DEFAULT_EVALUATION_SEED,
        DEFAULT_METRIC_SAMPLE,
        DEFAULT_NEIGHBORS,
        GEOMETRY_METRIC_KEYS,
        _encode_test_rows,
        _evaluate_metrics,
        _validate_published_configuration,
        make_metric_cohorts,
    )
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from compare_ae_training_sizes import (  # type: ignore[no-redef]
        DEFAULT_BATCH_SIZE,
        DEFAULT_INIT_SEEDS,
        DEFAULT_POOL_SEED,
        DEFAULT_TEST_ROWS,
        DEFAULT_TUNE_ROWS,
        DEFAULT_TUNE_SEED,
        DEFAULT_UNIVERSE_ROWS,
        PUBLISHED_TEST_INDICES_SHA256,
        PUBLISHED_TEST_SEED,
        _arguments_json,
        _atomic_csv,
        _atomic_json,
        _epoch_order,
        _evaluate,
        _resolve_device,
        _sha256_file,
        _sha256_indices,
        _state_sha256,
        _sync,
        inspect_source,
        load_resident_source,
        make_experiment_indices,
    )
    from compare_yandex_streaming import _build_autoencoder  # type: ignore[no-redef]
    from evaluate_ae_training_size_geometry import (  # type: ignore[no-redef]
        DEFAULT_DECODER_SAMPLE,
        DEFAULT_DISTANCE_PAIRS,
        DEFAULT_EVALUATION_SEED,
        DEFAULT_METRIC_SAMPLE,
        DEFAULT_NEIGHBORS,
        GEOMETRY_METRIC_KEYS,
        _encode_test_rows,
        _evaluate_metrics,
        _validate_published_configuration,
        make_metric_cohorts,
    )


PROTOCOL = "deep1b-ae-update-budget-sensitivity-v1"
SCHEMA_VERSION = 1
DEFAULT_POOL_SIZE = 65_536
DEFAULT_POOL_SIZES = (65_536, 1_048_576)
DEFAULT_CHECKPOINT_UPDATES = (
    0,
    8,
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1_024,
    1_204,
    1_808,
    2_408,
    3_612,
    4_816,
)
REPLAY_ANCHOR_UPDATE = 1_204
DEFAULT_CONFIRMATION_SEEDS = tuple(range(1_732, 1_744))

# Positive values always mean degradation relative to the same seed/pool's
# reference checkpoint.  Relative margins are fractions, not percentages.
NONINFERIORITY_MARGINS = {
    "reconstruction_nmse": {"margin": 0.01, "scale": "relative", "better": "lower"},
    "reconstruction_cosine": {"margin": 0.005, "scale": "absolute", "better": "higher"},
    "sample_conditional_trustworthiness": {
        "margin": 0.002,
        "scale": "absolute",
        "better": "higher",
    },
    "sample_conditional_continuity": {
        "margin": 0.002,
        "scale": "absolute",
        "better": "higher",
    },
    "sample_conditional_neighbor_overlap": {
        "margin": 0.005,
        "scale": "absolute",
        "better": "higher",
    },
    "sampled_pair_distance_spearman": {
        "margin": 0.010,
        "scale": "absolute",
        "better": "higher",
    },
    "sampled_pair_normalized_stress": {
        "margin": 0.005,
        "scale": "absolute",
        "better": "lower",
    },
    "sampled_local_decodability_nmse": {
        "margin": 0.01,
        "scale": "relative",
        "better": "lower",
    },
}


def signed_degradation(metric: str, candidate: float, reference: float) -> float:
    """Return a degradation score where positive is always worse."""

    rule = NONINFERIORITY_MARGINS[metric]
    if rule["scale"] == "relative":
        if reference == 0:
            raise ValueError(f"cannot compute relative degradation for zero {metric} reference")
        ratio = candidate / reference - 1.0
        return float(ratio if rule["better"] == "lower" else -ratio)
    difference = candidate - reference
    return float(difference if rule["better"] == "lower" else -difference)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _gather_numpy(torch, resident, indices: np.ndarray, batch_size: int) -> np.ndarray:
    """Copy declared resident rows to one compact host matrix."""

    result = np.empty((indices.size, resident.shape[1]), dtype=np.float32)
    for start in range(0, indices.size, batch_size):
        stop = min(start + batch_size, indices.size)
        selected = torch.as_tensor(indices[start:stop], device=resident.device)
        result[start:stop] = resident.index_select(0, selected).cpu().numpy()
    _sync(torch, resident.device)
    if not np.all(np.isfinite(result)):
        raise ValueError("fixed legacy evaluation matrix contains non-finite values")
    return result


def _matrix_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def _configure_determinism(torch, device) -> None:
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.cuda.reset_peak_memory_stats(device)


def validate_args(args, source) -> None:
    if not 0 < args.universe_rows <= source.rows:
        raise ValueError("universe_rows must fit in the validated source row count")
    if not 0 < args.target_dimension < source.dimension:
        raise ValueError("target_dimension must be smaller than the source dimension")
    if not args.init_seeds or len(set(args.init_seeds)) != len(args.init_seeds):
        raise ValueError("init_seeds must be non-empty and unique")
    if tuple(sorted(set(args.checkpoint_updates))) != tuple(args.checkpoint_updates):
        raise ValueError("checkpoint_updates must be unique and strictly increasing")
    if not args.checkpoint_updates or args.checkpoint_updates[0] < 0:
        raise ValueError("checkpoint_updates must be non-empty and non-negative")
    if args.checkpoint_updates[-1] < 1:
        raise ValueError("checkpoint_updates must include at least one trained checkpoint")
    for name in ("legacy_updates", "geometry_updates", "retain_updates"):
        values = tuple(getattr(args, name))
        if tuple(sorted(set(values))) != values:
            raise ValueError(f"{name} must be unique and strictly increasing")
        if not set(values).issubset(args.checkpoint_updates):
            raise ValueError(f"{name} must be a subset of checkpoint_updates")
    if args.reference_update not in args.checkpoint_updates:
        raise ValueError("reference_update must be present in checkpoint_updates")
    if not args.pool_sizes or tuple(sorted(set(args.pool_sizes))) != tuple(args.pool_sizes):
        raise ValueError("pool_sizes must be non-empty, unique, and strictly increasing")
    positive = (
        *args.pool_sizes,
        args.batch_size,
        args.evaluation_batch_size,
        args.load_block_rows,
        args.test_rows,
        args.tune_rows,
        args.encoding_batch_size,
        args.neighbors,
        args.metric_sample,
        args.decoder_sample,
        args.distance_pairs,
        args.metric_jobs,
    )
    if min(positive) < 1:
        raise ValueError("row counts, sample counts, batch sizes, and jobs must be positive")
    if any(size < args.batch_size or size % args.batch_size for size in args.pool_sizes):
        raise ValueError("every pool size must be at least one batch and divisible by batch_size")
    if args.test_rows + args.tune_rows + max(args.pool_sizes) > args.universe_rows:
        raise ValueError("test, tune, and training pool exceed the fixed universe")
    geometry_rows = args.tune_rows if args.phase == "discovery" else args.test_rows
    if max(args.metric_sample, args.decoder_sample) > geometry_rows:
        raise ValueError("geometry and decoder samples must fit in the active evaluation cohort")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not args.hidden_dimensions or any(width < 1 for width in args.hidden_dimensions):
        raise ValueError("hidden dimensions must be non-empty and positive")
    seeds = (
        args.test_seed,
        args.tune_seed,
        args.pool_seed,
        args.shuffle_seed_offset,
        args.evaluation_seed,
        *args.init_seeds,
    )
    if min(seeds) < 0:
        raise ValueError("all seeds and seed offsets must be non-negative")
    if args.cuda_reserve_gib < 0:
        raise ValueError("cuda_reserve_gib must be non-negative")
    if args.phase == "discovery":
        if args.legacy_updates:
            raise ValueError("discovery forbids all legacy-cohort evaluation")
        if any(
            value is not None
            for value in (
                args.locked_candidate_update,
                args.locked_step_updates,
                args.locked_reference_update,
            )
        ):
            raise ValueError("locked confirmation updates are invalid during discovery")
        if args.geometry_updates and args.reference_update not in args.geometry_updates:
            raise ValueError("discovery geometry_updates must include the reference update")
    else:
        locked = (
            args.locked_candidate_update,
            args.locked_step_updates,
            args.locked_reference_update,
        )
        if any(value is None for value in locked):
            raise ValueError("confirmation requires locked candidate, step, and reference updates")
        candidate = int(args.locked_candidate_update)
        step = int(args.locked_step_updates)
        reference = int(args.locked_reference_update)
        lower = max(0, candidate - step)
        if step < 1 or not lower < candidate < candidate + step < reference:
            raise ValueError(
                "locked confirmation updates must satisfy " "max(0,B-step) < B < B+step < reference"
            )
        if args.reference_update != reference:
            raise ValueError("confirmation reference_update must equal locked_reference_update")
        expected = tuple(
            sorted(
                {
                    lower,
                    candidate,
                    candidate + step,
                    REPLAY_ANCHOR_UPDATE,
                    reference,
                }
            )
        )
        if args.checkpoint_updates != expected:
            raise ValueError(
                "confirmation checkpoint_updates must be exactly B-step, B, B+step, "
                "the 1204 replay anchor, and reference"
            )
        if args.legacy_updates != args.checkpoint_updates:
            raise ValueError("confirmation must evaluate every locked checkpoint on legacy")
        if not {candidate, reference}.issubset(args.geometry_updates):
            raise ValueError("confirmation geometry must include candidate and reference")
    if args.retain_updates and args.checkpoint_dir is None:
        raise ValueError("retain_updates requires checkpoint_dir")
    if args.protocol_document is not None and not args.protocol_document.is_file():
        raise ValueError("protocol_document must be a readable file")


def _evaluate_checkpoint(
    torch,
    *,
    model,
    resident,
    active_geometry_matrix: np.ndarray | None,
    active_geometry_role: str,
    tune_indices: np.ndarray,
    legacy_indices: np.ndarray,
    geometry_cohorts: dict[str, Any],
    args,
    device,
    pool_size: int,
    pool_indices_sha256: str,
    init_seed: int,
    update: int,
    optimizer_interval_seconds: float,
    optimizer_cumulative_seconds: float,
    training_sse: float,
    training_elements: int,
) -> dict[str, Any]:
    tune = None
    tune_seconds = 0.0
    if args.phase == "discovery":
        tune_started = time.perf_counter()
        tune = _evaluate(torch, model, resident, tune_indices, device, args.evaluation_batch_size)
        tune_seconds = time.perf_counter() - tune_started

    legacy = None
    legacy_seconds = 0.0
    if update in args.legacy_updates:
        legacy_started = time.perf_counter()
        legacy_metrics = _evaluate(
            torch, model, resident, legacy_indices, device, args.evaluation_batch_size
        )
        legacy_seconds = time.perf_counter() - legacy_started
        legacy = {
            **legacy_metrics,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_early_stopping": False,
            "role": "locked confirmation evaluation only",
        }

    geometry = None
    encoding = None
    encoding_seconds = 0.0
    geometry_seconds = 0.0
    if update in args.geometry_updates:
        if active_geometry_matrix is None:
            raise RuntimeError("geometry checkpoint requested without an active cohort matrix")
        layout, encoding_seconds = _encode_test_rows(
            torch, model, active_geometry_matrix, device, args.encoding_batch_size
        )
        layout_sha256 = _matrix_sha256(layout)
        geometry_started = time.perf_counter()
        geometry = _evaluate_metrics(active_geometry_matrix, layout, geometry_cohorts, args)
        geometry_seconds = time.perf_counter() - geometry_started
        encoding = {
            "rows": int(active_geometry_matrix.shape[0]),
            "dimensions": int(args.target_dimension),
            "dtype": "float32",
            "layout_sha256": layout_sha256,
            "retained": False,
            "scope": active_geometry_role,
        }
        del layout

    model_state_sha256 = _state_sha256(torch, model.state_dict())
    retained_checkpoint = None
    if update in args.retain_updates:
        checkpoint_path = args.checkpoint_dir / (
            f"{args.phase}-seed-{init_seed}-pool-{pool_size}-update-{update}.pt"
        )
        if checkpoint_path.exists() and not args.force:
            raise FileExistsError(f"retained checkpoint exists: {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp")
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "phase": args.phase,
            "init_seed": int(init_seed),
            "pool_size": int(pool_size),
            "pool_indices_sha256": pool_indices_sha256,
            "update": int(update),
            "model_state_sha256": model_state_sha256,
            "input_dimension": int(resident.shape[1]),
            "target_dimension": int(args.target_dimension),
            "hidden_dimensions": list(args.hidden_dimensions),
            "protocol_document_sha256": getattr(args, "protocol_document_sha256", None),
            "model_state_dict": state,
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        retained_checkpoint = {
            "path": str(checkpoint_path.resolve()),
            "file_bytes": int(checkpoint_path.stat().st_size),
            "file_sha256": _sha256_file(checkpoint_path),
            "model_state_sha256": model_state_sha256,
            "metadata_embedded": True,
        }

    return {
        "update": int(update),
        "presentations": int(update * args.batch_size),
        "equivalent_pool_passes": float(update * args.batch_size / pool_size),
        "completed_pool_passes": int(update // (pool_size // args.batch_size)),
        "partial_pass_updates": int(update % (pool_size // args.batch_size)),
        "unique_training_rows_presented": int(min(pool_size, update * args.batch_size)),
        "presentations_per_unique_pool_row": float(update * args.batch_size / pool_size),
        "model_state_sha256": model_state_sha256,
        "training": {
            "cumulative_mse": (
                float(training_sse / training_elements) if training_elements else None
            ),
            "cumulative_elements": int(training_elements),
        },
        "tune": tune,
        "fixed_legacy_evaluation": legacy,
        "geometry": geometry,
        "geometry_cohort_role": active_geometry_role if geometry is not None else None,
        "encoding": encoding,
        "retained_checkpoint": retained_checkpoint,
        "timing_seconds": {
            "optimizer_since_previous_checkpoint": float(optimizer_interval_seconds),
            "optimizer_cumulative": float(optimizer_cumulative_seconds),
            "tune_reconstruction": float(tune_seconds),
            "legacy_reconstruction": float(legacy_seconds),
            "legacy_encoding": float(encoding_seconds),
            "geometry_metrics": float(geometry_seconds),
        },
    }


def run_trajectory(
    torch,
    *,
    resident,
    indices,
    active_geometry_matrix: np.ndarray | None,
    active_geometry_role: str,
    geometry_cohorts: dict[str, Any],
    source_dimension: int,
    args,
    init_seed: int,
    pool_size: int,
    device,
) -> dict[str, Any]:
    """Train one uninterrupted trajectory and inspect only scheduled updates."""

    pool = indices.training_order[:pool_size]
    pool_indices_sha256 = _sha256_indices(pool)
    pool_tensor = torch.as_tensor(pool, device=resident.device)
    updates_per_pass = pool_size // args.batch_size

    random.seed(init_seed)
    np.random.seed(init_seed % 2**32)
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    model = _build_autoencoder(
        torch, source_dimension, args.target_dimension, tuple(args.hidden_dimensions)
    ).to(device)
    initial_state_sha256 = _state_sha256(torch, model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = torch.nn.MSELoss(reduction="mean")

    trajectory_started = time.perf_counter()
    optimizer_cumulative_seconds = 0.0
    optimizer_interval_seconds = 0.0
    training_sse = 0.0
    training_elements = 0
    checkpoints: list[dict[str, Any]] = []

    def inspect(update: int) -> None:
        nonlocal optimizer_interval_seconds
        checkpoints.append(
            _evaluate_checkpoint(
                torch,
                model=model,
                resident=resident,
                active_geometry_matrix=active_geometry_matrix,
                active_geometry_role=active_geometry_role,
                tune_indices=indices.tune,
                legacy_indices=indices.test,
                geometry_cohorts=geometry_cohorts,
                args=args,
                device=device,
                pool_size=pool_size,
                pool_indices_sha256=pool_indices_sha256,
                init_seed=init_seed,
                update=update,
                optimizer_interval_seconds=optimizer_interval_seconds,
                optimizer_cumulative_seconds=optimizer_cumulative_seconds,
                training_sse=training_sse,
                training_elements=training_elements,
            )
        )
        optimizer_interval_seconds = 0.0

    scheduled = set(args.checkpoint_updates)
    if 0 in scheduled:
        inspect(0)
    maximum_update = args.checkpoint_updates[-1]
    update = 0
    epoch = 0
    while update < maximum_update:
        order = _epoch_order(
            torch,
            pool_size,
            init_seed + args.shuffle_seed_offset,
            epoch,
            resident.device,
        )
        for batch_number in range(updates_per_pass):
            if update >= maximum_update:
                break
            started = time.perf_counter()
            positions = order[batch_number * args.batch_size : (batch_number + 1) * args.batch_size]
            row_indices = pool_tensor.index_select(0, positions)
            batch = resident.index_select(0, row_indices).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_function(reconstructed.float(), batch.float())
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            _sync(torch, device)
            elapsed = time.perf_counter() - started
            optimizer_cumulative_seconds += elapsed
            optimizer_interval_seconds += elapsed
            elements = int(batch.numel())
            training_sse += float(loss.item()) * elements
            training_elements += elements
            update += 1
            if update in scheduled:
                inspect(update)
        epoch += 1

    observed = [checkpoint["update"] for checkpoint in checkpoints]
    if observed != list(args.checkpoint_updates):
        raise RuntimeError(f"observed checkpoint grid {observed} != declared grid")
    retained = [
        checkpoint["retained_checkpoint"]
        for checkpoint in checkpoints
        if checkpoint["retained_checkpoint"] is not None
    ]
    return {
        "trajectory_id": f"pool-{pool_size}-initialization-seed-{init_seed}",
        "init_seed": int(init_seed),
        "initial_state_sha256": initial_state_sha256,
        "pool_size": int(pool_size),
        "pool_indices_sha256": pool_indices_sha256,
        "updates_per_pool_pass": int(updates_per_pass),
        "maximum_update": int(maximum_update),
        "optimizer_steps_executed": int(update),
        "single_uninterrupted_training_trajectory": True,
        "model_reinitialized_between_checkpoints": False,
        "checkpoint_states_retained": bool(retained),
        "retained_checkpoints": retained,
        "checkpoints": checkpoints,
        "timing_seconds": {
            "optimizer_total": float(optimizer_cumulative_seconds),
            "trajectory_total": float(time.perf_counter() - trajectory_started),
        },
    }


def _checkpoint_metric_values(checkpoint: dict[str, Any], phase: str) -> dict[str, float]:
    reconstruction = (
        checkpoint["tune"] if phase == "discovery" else checkpoint["fixed_legacy_evaluation"]
    )
    values: dict[str, float] = {}
    if reconstruction is not None:
        values.update(
            {
                "reconstruction_nmse": float(reconstruction["reconstruction_nmse"]),
                "reconstruction_cosine": float(reconstruction["reconstruction_cosine"]),
            }
        )
    if checkpoint["geometry"] is not None:
        values.update({key: float(checkpoint["geometry"][key]) for key in GEOMETRY_METRIC_KEYS})
    return values


def attach_reference_degradations(
    trajectory: dict[str, Any], *, phase: str, reference_update: int
) -> None:
    """Attach same-seed/pool signed degradation relative to a fixed reference."""

    by_update = {item["update"]: item for item in trajectory["checkpoints"]}
    if reference_update not in by_update:
        raise ValueError("reference update is absent from the checkpoint grid")
    reference_values = _checkpoint_metric_values(by_update[reference_update], phase)
    for checkpoint in trajectory["checkpoints"]:
        observed = _checkpoint_metric_values(checkpoint, phase)
        common = sorted(set(observed).intersection(reference_values))
        checkpoint["comparison_to_reference"] = {
            "reference_update": int(reference_update),
            "positive_means_worse": True,
            "signed_degradation": {
                key: signed_degradation(key, observed[key], reference_values[key]) for key in common
            },
            "margin": {key: float(NONINFERIORITY_MARGINS[key]["margin"]) for key in common},
        }


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in result["trajectories"]:
        for checkpoint in run["checkpoints"]:
            row = {
                "trajectory_id": run["trajectory_id"],
                "init_seed": run["init_seed"],
                "pool_size": run["pool_size"],
                "update": checkpoint["update"],
                "presentations": checkpoint["presentations"],
                "equivalent_pool_passes": checkpoint["equivalent_pool_passes"],
                "model_state_sha256": checkpoint["model_state_sha256"],
                "tune_nmse": (
                    checkpoint["tune"]["reconstruction_nmse"]
                    if checkpoint["tune"] is not None
                    else None
                ),
                "tune_cosine": (
                    checkpoint["tune"]["reconstruction_cosine"]
                    if checkpoint["tune"] is not None
                    else None
                ),
                "legacy_nmse": (
                    checkpoint["fixed_legacy_evaluation"]["reconstruction_nmse"]
                    if checkpoint["fixed_legacy_evaluation"] is not None
                    else None
                ),
                "legacy_cosine": (
                    checkpoint["fixed_legacy_evaluation"]["reconstruction_cosine"]
                    if checkpoint["fixed_legacy_evaluation"] is not None
                    else None
                ),
                "layout_sha256": (
                    checkpoint["encoding"]["layout_sha256"]
                    if checkpoint["encoding"] is not None
                    else None
                ),
                "optimizer_interval_seconds": checkpoint["timing_seconds"][
                    "optimizer_since_previous_checkpoint"
                ],
                "optimizer_cumulative_seconds": checkpoint["timing_seconds"][
                    "optimizer_cumulative"
                ],
            }
            for key in GEOMETRY_METRIC_KEYS:
                row[key] = (
                    checkpoint["geometry"][key] if checkpoint["geometry"] is not None else None
                )
                row[f"degradation_to_reference__{key}"] = checkpoint["comparison_to_reference"][
                    "signed_degradation"
                ].get(key)
            for key in ("reconstruction_nmse", "reconstruction_cosine"):
                row[f"degradation_to_reference__{key}"] = checkpoint["comparison_to_reference"][
                    "signed_degradation"
                ].get(key)
            rows.append(row)
    return rows


def run_experiment(args) -> dict[str, Any]:
    import scipy
    import sklearn
    import torch

    started_at = _utc_now()
    total_started = time.perf_counter()
    source = inspect_source(args.data)
    validate_args(args, source)
    _validate_published_configuration(args)
    protocol_document = None
    args.protocol_document_sha256 = None
    if args.protocol_document is not None:
        args.protocol_document_sha256 = _sha256_file(args.protocol_document)
        protocol_document = {
            "path": str(args.protocol_document.resolve()),
            "file_bytes": int(args.protocol_document.stat().st_size),
            "file_sha256": args.protocol_document_sha256,
            "binding": "frozen before the run",
        }
    indices = make_experiment_indices(
        universe_rows=args.universe_rows,
        test_rows=args.test_rows,
        tune_rows=args.tune_rows,
        maximum_pool_size=max(args.pool_sizes),
        test_seed=args.test_seed,
        tune_seed=args.tune_seed,
        pool_seed=args.pool_seed,
    )
    test_hash = _sha256_indices(indices.test)
    published_split = (
        args.universe_rows == DEFAULT_UNIVERSE_ROWS
        and args.test_rows == DEFAULT_TEST_ROWS
        and args.test_seed == PUBLISHED_TEST_SEED
    )
    if published_split and test_hash != PUBLISHED_TEST_INDICES_SHA256:
        raise RuntimeError("published Deep1B legacy indices do not match their pinned SHA-256")
    if not args.allow_nonpublished_cohort and not published_split:
        raise ValueError("published mode requires the fixed 10M Deep1B legacy cohort")

    intersections = {
        "legacy_tune_intersection": int(np.intersect1d(indices.test, indices.tune).size),
        "legacy_training_intersection": int(
            np.intersect1d(indices.test, indices.training_order).size
        ),
        "tune_training_intersection": int(
            np.intersect1d(indices.tune, indices.training_order).size
        ),
    }
    combined = np.concatenate((indices.test, indices.tune, indices.training_order))
    if any(intersections.values()) or np.unique(combined).size != combined.size:
        raise RuntimeError("legacy, tune, and training rows are not disjoint")

    device = _resolve_device(torch, args.device)
    _configure_determinism(torch, device)
    resident, residency = load_resident_source(torch, source, args, device)
    active_indices = indices.tune if args.phase == "discovery" else indices.test
    active_geometry_role = (
        "disjoint tune cohort" if args.phase == "discovery" else "fixed legacy cohort"
    )
    matrix_started = time.perf_counter()
    active_geometry_matrix = (
        _gather_numpy(torch, resident, active_indices, args.evaluation_batch_size)
        if args.geometry_updates
        else None
    )
    matrix_seconds = time.perf_counter() - matrix_started
    geometry_cohorts = make_metric_cohorts(active_indices, args)

    trajectories = []
    for seed in args.init_seeds:
        paired_initial_state = None
        for pool_size in args.pool_sizes:
            trajectory = run_trajectory(
                torch,
                resident=resident,
                indices=indices,
                active_geometry_matrix=active_geometry_matrix,
                active_geometry_role=active_geometry_role,
                geometry_cohorts=geometry_cohorts,
                source_dimension=source.dimension,
                args=args,
                init_seed=seed,
                pool_size=pool_size,
                device=device,
            )
            if paired_initial_state is None:
                paired_initial_state = trajectory["initial_state_sha256"]
            elif trajectory["initial_state_sha256"] != paired_initial_state:
                raise RuntimeError("same-seed pool trajectories did not share initialization")
            attach_reference_degradations(
                trajectory, phase=args.phase, reference_update=args.reference_update
            )
            trajectories.append(trajectory)

    source_hash_started = time.perf_counter()
    source_sha256 = _sha256_file(source.path)
    source_hash_seconds = time.perf_counter() - source_hash_started
    geometry_module = Path(__file__).with_name("evaluate_ae_training_size_geometry.py")
    replay_entries = []
    for trajectory in trajectories:
        anchor = next(
            (
                checkpoint
                for checkpoint in trajectory["checkpoints"]
                if checkpoint["update"] == REPLAY_ANCHOR_UPDATE
            ),
            None,
        )
        replay_entries.append(
            {
                "init_seed": trajectory["init_seed"],
                "pool_size": trajectory["pool_size"],
                "initial_state_sha256": trajectory["initial_state_sha256"],
                "pool_indices_sha256": trajectory["pool_indices_sha256"],
                "anchor_present": anchor is not None,
                "model_state_sha256": (
                    anchor["model_state_sha256"] if anchor is not None else None
                ),
                "tune": anchor["tune"] if anchor is not None else None,
            }
        )
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "phase": args.phase,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "question": (
            "earliest practically non-inferior update budget relative to the configured "
            "finite reference"
        ),
        "arguments": _arguments_json(args),
        "protocol_document": protocol_document,
        "source": {
            "path": str(source.path),
            "format": source.format,
            "rows": source.rows,
            "dimension": source.dimension,
            "dtype": source.dtype,
            "file_bytes": source.file_bytes,
            "file_sha256": source_sha256,
            "file_sha256_seconds": source_hash_seconds,
            "universe_policy": "fixed leading prefix",
            "universe_rows": int(args.universe_rows),
            "input_transformation": "none",
        },
        "split": {
            "fixed_legacy_evaluation": {
                "rows": int(indices.test.size),
                "seed": int(args.test_seed),
                "indices_sha256": test_hash,
                "matrix_sha256": (
                    _matrix_sha256(active_geometry_matrix)
                    if args.phase == "confirmation" and active_geometry_matrix is not None
                    else None
                ),
                "published_identity_expected": (
                    PUBLISHED_TEST_INDICES_SHA256 if published_split else None
                ),
                "published_identity_verified": bool(published_split),
                "materialized": args.phase == "confirmation" and bool(args.geometry_updates),
                "reconstruction_updates": list(args.legacy_updates),
                "used_during_discovery": False,
                "role": "locked confirmation cohort; never used for discovery",
            },
            "tune": {
                "rows": int(indices.tune.size),
                "seed": int(args.tune_seed),
                "indices_sha256": _sha256_indices(indices.tune),
                "matrix_sha256": (
                    _matrix_sha256(active_geometry_matrix)
                    if args.phase == "discovery" and active_geometry_matrix is not None
                    else None
                ),
                "geometry_updates": (
                    list(args.geometry_updates) if args.phase == "discovery" else []
                ),
            },
            "training_pool": {
                "rows": list(args.pool_sizes),
                "pool_seed": int(args.pool_seed),
                "maximum_prefix_indices_sha256": _sha256_indices(indices.training_order),
                "prefix_indices_sha256": {
                    str(size): _sha256_indices(indices.training_order[:size])
                    for size in args.pool_sizes
                },
                "nested_prefix_protocol": True,
            },
            "disjointness": {
                **intersections,
                "combined_rows": int(combined.size),
                "exact_unique_rows": int(np.unique(combined).size),
                "legacy_used_for_training": False,
            },
        },
        "protocol_configuration": {
            "target_dimension": int(args.target_dimension),
            "hidden_dimensions": list(args.hidden_dimensions),
            "architecture": (
                f"{source.dimension}-"
                + "-".join(map(str, args.hidden_dimensions))
                + f"-{args.target_dimension}-symmetric"
            ),
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in _build_autoencoder(
                        torch,
                        source.dimension,
                        args.target_dimension,
                        tuple(args.hidden_dimensions),
                    ).parameters()
                )
            ),
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "loss": "elementwise reconstruction MSE",
            "batch_size": int(args.batch_size),
            "pool_sizes": list(args.pool_sizes),
            "checkpoint_updates_predeclared": list(args.checkpoint_updates),
            "maximum_updates": int(args.checkpoint_updates[-1]),
            "reference_update": int(args.reference_update),
            "reference_policy": "configured terminal or locked checkpoint; never hard-coded",
            "replay_anchor_update": REPLAY_ANCHOR_UPDATE,
            "replay_anchor_role": "compatibility check against the prior 1204-update run",
            "init_seeds": list(args.init_seeds),
            "training_trajectories": len(args.init_seeds) * len(args.pool_sizes),
            "one_trajectory_per_seed_and_pool": True,
            "model_reinitialized_only_between_seed_pool_trajectories": True,
            "phase": args.phase,
            "legacy_used_for_discovery_selection_or_stopping": False,
            "precision": "CUDA FP32" if device.type == "cuda" else "CPU FP32 (test mode)",
            "checkpoint_files_written": bool(args.retain_updates),
            "full_layouts_written": False,
        },
        "decision_protocol": {
            "estimand": "practical non-inferiority to the configured same-seed/pool reference",
            "not_claimed": "symmetric equivalence or a plateau beyond the maximum tested update",
            "positive_signed_degradation_means_worse": True,
            "noninferiority_margins": NONINFERIORITY_MARGINS,
            "discovery_rule": {
                "seeds": [1729, 1730, 1731],
                "mean_degradation_strictly_below_margin": True,
                "per_seed_degradation_limit": "strictly below two times each margin",
                "sustained_pass": "candidate and next two predeclared checkpoints must pass",
                "bracketing": "report the last failing checkpoint and first sustained pass",
                "fine_grid": "16 updates when boundary <=256; otherwise 32",
                "terminal_flatness": (
                    "mean absolute paired metric changes on each metric's declared "
                    "relative or absolute scale"
                ),
                "extension_policy": (
                    "if terminal flatness fails, extend to the next frozen terminal "
                    "(initially 9632) and recompute comparisons against it"
                ),
                "cohort": "tune only",
            },
            "confirmation_rule": {
                "fresh_seed_default": list(DEFAULT_CONFIRMATION_SEEDS),
                "comparison": "locked candidate versus configured reference",
                "criterion": "upper one-sided 95% paired-t harm CI strictly below margin",
                "cohort": "fixed legacy only",
                "locked_candidate_update": args.locked_candidate_update,
                "locked_step_updates": args.locked_step_updates,
                "locked_reference_update": args.locked_reference_update,
            },
            "retrieval_followup": {
                "seed": 1732,
                "queries": 1000,
                "states": "locked candidate and replay/reference checkpoints only",
            },
        },
        "replay_audit": {
            "anchor_update": REPLAY_ANCHOR_UPDATE,
            "purpose": "exact comparison with the prior training-size trajectories",
            "current_run_identity_exposed": True,
            "external_prior_result_checked": False,
            "claim": "none until an external prior-result comparison succeeds",
            "entries": replay_entries,
        },
        "geometry_protocol": {
            **geometry_cohorts["provenance"],
            "metric_keys": list(GEOMETRY_METRIC_KEYS),
            "definitions_reused_from": "experiments.evaluate_ae_training_size_geometry",
            "geometry_evaluator_sha256": _sha256_file(geometry_module),
            "cohort_role": active_geometry_role,
            "cohort_global_indices_sha256": _sha256_indices(active_indices),
            "evaluated_updates": list(args.geometry_updates),
            "layout_retention": "hashed in memory and discarded after every checkpoint",
        },
        "residency": residency,
        "trajectories": trajectories,
        "artifacts": {
            "checkpoint_paths": [
                retained["path"]
                for trajectory in trajectories
                for retained in trajectory["retained_checkpoints"]
            ],
            "retained_layout_paths": [],
        },
        "resources": {"cuda": cuda_memory},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "training_device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_matmul_tf32_allowed": (
                torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
            ),
            "cudnn_tf32_allowed": (
                torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark if device.type == "cuda" else None,
            "cudnn_deterministic": (
                torch.backends.cudnn.deterministic if device.type == "cuda" else None
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "timing_seconds": {
            "active_geometry_matrix_materialization": float(matrix_seconds),
            "total": float(time.perf_counter() - total_started),
        },
    }


def execute(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "update-budget-results.json"
    csv_path = output_dir / "update-budget-results.csv"
    if not args.force and (result_path.exists() or csv_path.exists()):
        raise FileExistsError("update-budget output exists; pass --force to replace it")
    result = run_experiment(args)
    _atomic_csv(csv_path, _summary_rows(result))
    result["artifacts"].update(
        {
            "csv_path": str(csv_path),
            "csv_sha256": _sha256_file(csv_path),
            "json_path": str(result_path),
        }
    )
    _atomic_json(result_path, result)
    return result


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected one or more non-negative integers")
    return parsed


def _optional_update_grid(value: str) -> tuple[int, ...] | None:
    lowered = value.strip().lower()
    if lowered == "all":
        return None
    if lowered == "none":
        return ()
    return _comma_separated_ints(value)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/deep1b-ae-update-budget")
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--phase", choices=("discovery", "confirmation"), default="discovery")
    parser.add_argument("--protocol-document", type=Path)
    parser.add_argument("--universe-rows", type=int, default=DEFAULT_UNIVERSE_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--tune-rows", type=int, default=DEFAULT_TUNE_ROWS)
    parser.add_argument("--test-seed", type=int, default=PUBLISHED_TEST_SEED)
    parser.add_argument("--tune-seed", type=int, default=DEFAULT_TUNE_SEED)
    parser.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED)
    parser.add_argument("--shuffle-seed-offset", type=int, default=10_000_019)
    parser.add_argument("--pool-sizes", type=_comma_separated_ints, default=DEFAULT_POOL_SIZES)
    parser.add_argument(
        "--checkpoint-updates",
        type=_comma_separated_ints,
        default=DEFAULT_CHECKPOINT_UPDATES,
    )
    parser.add_argument(
        "--reference-update",
        type=int,
        help="same-seed/pool comparison reference (default: maximum checkpoint)",
    )
    parser.add_argument(
        "--legacy-updates",
        type=_optional_update_grid,
        default=None,
        help="all, none, or a comma-separated checkpoint subset",
    )
    parser.add_argument(
        "--geometry-updates",
        type=_optional_update_grid,
        default=None,
        help="all, none, or a comma-separated checkpoint subset",
    )
    parser.add_argument(
        "--retain-updates",
        type=_optional_update_grid,
        default=(),
        help="none, all, or checkpoint subset to retain for later retrieval",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--locked-candidate-update", type=int)
    parser.add_argument("--locked-step-updates", type=int)
    parser.add_argument("--locked-reference-update", type=int)
    parser.add_argument("--init-seeds", type=_comma_separated_ints)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--target-dimension", type=int, default=8)
    parser.add_argument("--hidden-dimensions", type=_comma_separated_ints, default=(256, 128, 32))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--source-residency", choices=("auto", "device", "host"), default="auto")
    parser.add_argument("--cuda-reserve-gib", type=float, default=2.0)
    parser.add_argument("--load-block-rows", type=int, default=131_072)
    parser.add_argument("--encoding-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--metric-sample", type=int, default=DEFAULT_METRIC_SAMPLE)
    parser.add_argument("--decoder-sample", type=int, default=DEFAULT_DECODER_SAMPLE)
    parser.add_argument("--distance-pairs", type=int, default=DEFAULT_DISTANCE_PAIRS)
    parser.add_argument("--metric-jobs", type=int, default=1)
    parser.add_argument(
        "--allow-nonpublished-cohort",
        action="store_true",
        help="permit small synthetic cohorts and metric samples for smoke tests only",
    )
    args = parser.parse_args(argv)
    if args.init_seeds is None:
        args.init_seeds = (
            DEFAULT_INIT_SEEDS if args.phase == "discovery" else DEFAULT_CONFIRMATION_SEEDS
        )
    if args.reference_update is None:
        args.reference_update = args.checkpoint_updates[-1]
    if args.legacy_updates is None:
        args.legacy_updates = () if args.phase == "discovery" else args.checkpoint_updates
    if args.geometry_updates is None:
        args.geometry_updates = args.checkpoint_updates
    if args.retain_updates is None:
        args.retain_updates = args.checkpoint_updates
    for name in ("data", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.protocol_document is not None:
        args.protocol_document = args.protocol_document.expanduser().resolve()
    return args


def _test_args(**overrides):
    defaults = {
        "data": Path("unused.npy"),
        "output_dir": Path("unused-output"),
        "force": False,
        "phase": "discovery",
        "protocol_document": None,
        "universe_rows": 80,
        "test_rows": 24,
        "tune_rows": 16,
        "test_seed": 7,
        "tune_seed": 11,
        "pool_seed": 13,
        "shuffle_seed_offset": 10_000_019,
        "pool_sizes": (16, 32),
        "checkpoint_updates": (0, 1, 2, 4, 8),
        "reference_update": 8,
        "legacy_updates": (),
        "geometry_updates": (0, 1, 2, 4, 8),
        "retain_updates": (),
        "checkpoint_dir": None,
        "locked_candidate_update": None,
        "locked_step_updates": None,
        "locked_reference_update": None,
        "init_seeds": (101, 202),
        "batch_size": 4,
        "evaluation_batch_size": 16,
        "target_dimension": 2,
        "hidden_dimensions": (10, 4),
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "device": "cpu",
        "source_residency": "auto",
        "cuda_reserve_gib": 0.0,
        "load_block_rows": 16,
        "encoding_batch_size": 16,
        "evaluation_seed": 17,
        "neighbors": 3,
        "metric_sample": 12,
        "decoder_sample": 16,
        "distance_pairs": 64,
        "metric_jobs": 1,
        "allow_nonpublished_cohort": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute(args)
    print(
        f"wrote {len(result['trajectories'])} trajectories on "
        f"{len(args.checkpoint_updates)} checkpoints to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
