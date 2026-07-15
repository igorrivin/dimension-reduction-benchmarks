#!/usr/bin/env python3
"""Controlled Deep1B autoencoder training-pool-size experiment.

The protocol separates three questions which are otherwise easy to conflate:

* what one pass over a training pool buys;
* what the same number of optimizer updates buys for every pool size; and
* which checkpoint is preferred by a tuning set that is not the final test set.

By default the final test rows are exactly the validation rows used by the
published 10-million-row Deep1B autoencoder run.  A new, disjoint tuning set is
sampled once, and the training pools are nested prefixes of one uniformly
shuffled complement.  Test data are evaluated exactly once per arm, after the
best tuning checkpoint has been selected.

Example with five initialization seeds (25 arms)::

    python experiments/compare_ae_training_sizes.py \
      --data deep10M_base.fbin --device cuda \
      --init-seeds 1729,1730,1731,1732,1733 \
      --output-dir artifacts/deep1b-ae-training-size

The harness writes compact JSON and CSV summaries, never full layouts.  Model
checkpoints are written only when ``--checkpoint-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# Required by deterministic CUDA matrix multiplications.  This must be set
# before PyTorch initializes CUDA (and therefore before the local torch import
# in ``run_experiment``).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    from experiments.compare_yandex_streaming import (
        BoundedFbinReader,
        _build_autoencoder,
        bounded_validation_indices,
    )
    from experiments.yandex_fbin import read_fbin_header
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from compare_yandex_streaming import (  # type: ignore[no-redef]
        BoundedFbinReader,
        _build_autoencoder,
        bounded_validation_indices,
    )
    from yandex_fbin import read_fbin_header  # type: ignore[no-redef]


PROTOCOL = "deep1b-ae-training-pool-sensitivity-v1"
SCHEMA_VERSION = 1
DEFAULT_UNIVERSE_ROWS = 10_000_000
DEFAULT_BATCH_SIZE = 8_192
DEFAULT_POOL_SIZES = (65_536, 262_144, 1_048_576, 4_194_304, 9_863_168)
DEFAULT_UPDATE_BUDGET = 1_204
DEFAULT_TUNE_INTERVAL_UPDATES = 32
DEFAULT_INIT_SEEDS = (1729, 1730, 1731)
DEFAULT_TEST_ROWS = 65_536
DEFAULT_TUNE_ROWS = 65_536
PUBLISHED_TEST_SEED = 1729 + 1_000_003
PUBLISHED_TEST_INDICES_SHA256 = "74821e8792f51cf70591121217b1ce0bd6c1f4ebbbb47283503e5bbf0d907548"
DEFAULT_TUNE_SEED = 2_718_281
DEFAULT_POOL_SEED = 3_141_593


@dataclass(frozen=True)
class ExperimentIndices:
    """Fixed test/tune rows and one randomized order defining nested pools."""

    test: np.ndarray
    tune: np.ndarray
    training_order: np.ndarray
    unused: np.ndarray


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    format: str
    rows: int
    dimension: int
    dtype: str
    file_bytes: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_indices(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(torch, state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _clone_state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


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
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(path, encoded)


def _draw_disjoint_uniform(
    universe_rows: int, count: int, excluded: np.ndarray, seed: int
) -> np.ndarray:
    """Uniform rejection sample from the complement using O(count) memory."""

    if count < 1 or count > universe_rows - excluded.size:
        raise ValueError("sample count must fit in the complement")
    forbidden = {int(value) for value in np.asarray(excluded, dtype=np.int64)}
    selected: set[int] = set()
    rng = np.random.default_rng(seed)
    while len(selected) < count:
        candidate = int(rng.integers(0, universe_rows))
        if candidate not in forbidden:
            selected.add(candidate)
    return np.fromiter(sorted(selected), dtype=np.int64, count=count)


def make_experiment_indices(
    *,
    universe_rows: int,
    test_rows: int,
    tune_rows: int,
    maximum_pool_size: int,
    test_seed: int = PUBLISHED_TEST_SEED,
    tune_seed: int = DEFAULT_TUNE_SEED,
    pool_seed: int = DEFAULT_POOL_SEED,
) -> ExperimentIndices:
    """Create one auditable split and a uniform order for nested training pools."""

    if universe_rows < 3:
        raise ValueError("universe_rows must be at least three")
    if min(test_rows, tune_rows, maximum_pool_size) < 1:
        raise ValueError("test, tune, and maximum pool sizes must be positive")
    if test_rows + tune_rows + maximum_pool_size > universe_rows:
        raise ValueError("test, tune, and maximum training pool do not fit in the universe")
    test = bounded_validation_indices(universe_rows, test_rows, test_seed)
    tune = _draw_disjoint_uniform(universe_rows, tune_rows, test, tune_seed)

    eligible_mask = np.ones(universe_rows, dtype=bool)
    eligible_mask[test] = False
    eligible_mask[tune] = False
    eligible = np.flatnonzero(eligible_mask)
    del eligible_mask
    rng = np.random.default_rng(pool_seed)
    rng.shuffle(eligible)
    training_order = np.ascontiguousarray(eligible[:maximum_pool_size], dtype=np.int64)
    unused = np.sort(np.ascontiguousarray(eligible[maximum_pool_size:], dtype=np.int64))
    return ExperimentIndices(test=test, tune=tune, training_order=training_order, unused=unused)


def inspect_source(path: Path) -> SourceInfo:
    """Strictly inspect a two-dimensional float32 NPY or Big-ANN fbin source."""

    path = Path(path).resolve()
    if path.suffix.lower() == ".fbin":
        info = read_fbin_header(path)
        return SourceInfo(
            path=path,
            format="fbin",
            rows=int(info.rows),
            dimension=int(info.dimension),
            dtype="little-endian float32",
            file_bytes=int(info.expected_size),
        )
    if path.suffix.lower() != ".npy":
        raise ValueError("--data must have an .fbin or .npy suffix")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not inspect NPY source {path}: {error}") from error
    if array.ndim != 2:
        raise ValueError(f"NPY source must be two-dimensional, found shape {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"NPY source must have native float32 dtype, found {array.dtype}")
    return SourceInfo(
        path=path,
        format="npy",
        rows=int(array.shape[0]),
        dimension=int(array.shape[1]),
        dtype="native float32",
        file_bytes=int(path.stat().st_size),
    )


def _iter_source_blocks(info: SourceInfo, rows: int, block_rows: int):
    if info.format == "fbin":
        fbin_info = read_fbin_header(info.path)
        with BoundedFbinReader(info.path, fbin_info, rows=rows) as reader:
            for start in range(0, rows, block_rows):
                stop = min(rows, start + block_rows)
                yield start, stop, reader.read_block(start, stop)
        return
    array = np.load(info.path, mmap_mode="r", allow_pickle=False)
    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        yield start, stop, np.array(array[start:stop], dtype=np.float32, order="C", copy=True)


def _resolve_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_resident_source(torch, info: SourceInfo, args, training_device):
    """Load the fixed universe once, preferring CUDA residency when it safely fits."""

    source_bytes = args.universe_rows * info.dimension * np.dtype(np.float32).itemsize
    resident_device = training_device
    fallback_reason = None
    if training_device.type == "cuda" and args.source_residency == "auto":
        free_bytes, _ = torch.cuda.mem_get_info(training_device)
        reserve = int(args.cuda_reserve_gib * 2**30)
        if source_bytes + reserve > free_bytes:
            resident_device = torch.device("cpu")
            fallback_reason = (
                f"source bytes plus {reserve} reserve exceed {int(free_bytes)} free CUDA bytes"
            )
    elif training_device.type == "cuda" and args.source_residency == "host":
        resident_device = torch.device("cpu")
        fallback_reason = "host residency explicitly requested"

    started = time.perf_counter()
    resident = torch.empty(
        (args.universe_rows, info.dimension), dtype=torch.float32, device=resident_device
    )
    for start, stop, block in _iter_source_blocks(info, args.universe_rows, args.load_block_rows):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"source rows [{start}, {stop}) contain non-finite values")
        resident[start:stop].copy_(torch.from_numpy(block).to(resident_device))
    _sync(torch, resident_device)
    return resident, {
        "requested": args.source_residency,
        "actual_device": str(resident_device),
        "bytes": int(source_bytes),
        "load_seconds": time.perf_counter() - started,
        "cuda_fallback_reason": fallback_reason,
        "entire_universe_resident": True,
    }


def _evaluate(torch, model, resident, indices: np.ndarray, device, batch_size: int) -> dict:
    """Evaluate MSE, mean-predictor-normalized MSE, and cosine in FP32."""

    sse = 0.0
    cosine_sum = 0.0
    rows = 0
    feature_sum = np.zeros(resident.shape[1], dtype=np.float64)
    feature_square_sum = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            selected = torch.as_tensor(indices[start : start + batch_size], device=resident.device)
            batch = resident.index_select(0, selected).to(device)
            reconstructed = model(batch).float()
            difference = reconstructed - batch
            sse += float(torch.sum(difference * difference).item())
            numerator = torch.sum(reconstructed * batch, dim=1)
            denominator = torch.linalg.vector_norm(reconstructed, dim=1) * torch.linalg.vector_norm(
                batch, dim=1
            )
            cosine_sum += float(torch.sum(numerator / denominator.clamp_min(1e-12)).item())
            host = batch.detach().cpu().numpy().astype(np.float64, copy=False)
            feature_sum += np.sum(host, axis=0, dtype=np.float64)
            feature_square_sum += float(np.sum(host * host, dtype=np.float64))
            rows += int(batch.shape[0])
    _sync(torch, device)
    elements = rows * resident.shape[1]
    mse = sse / elements
    centered_sse = feature_square_sum - float(np.sum(feature_sum * feature_sum)) / rows
    baseline_mse = max(np.finfo(np.float64).eps, centered_sse / elements)
    return {
        "rows": rows,
        "reconstruction_mse": mse,
        "reconstruction_nmse": mse / baseline_mse,
        "reconstruction_cosine": cosine_sum / rows,
        "mean_predictor_mse": baseline_mse,
    }


def _epoch_order(torch, pool_size: int, seed: int, epoch: int, device):
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(
        int(np.random.SeedSequence([seed, epoch, pool_size]).generate_state(1)[0])
    )
    return torch.randperm(pool_size, generator=generator, device=device)


def _checkpoint_payload(arm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "init_seed": arm["init_seed"],
        "pool_size": arm["pool_size"],
        "selected_update": arm["selection"]["best_update"],
        "model_state_dict": state,
    }


def run_arm(torch, resident, indices: ExperimentIndices, source_dimension: int, args, init_seed):
    """Run one paired pool-size arm and evaluate the final test exactly once."""

    pool_size = int(args.current_pool_size)
    pool = indices.training_order[:pool_size]
    updates_per_pass = pool_size // args.batch_size
    device = _resolve_device(torch, args.device)
    pool_tensor = torch.as_tensor(pool, device=resident.device)

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

    arm_started = time.perf_counter()
    training_seconds = 0.0
    tune_seconds = 0.0
    training_sse = 0.0
    training_elements = 0
    best_tune_nmse = float("inf")
    best_update = None
    best_state = None
    selection_trace: list[dict[str, Any]] = []
    first_pass_record = None
    fixed_budget_record = None
    update = 0
    epoch = 0

    while update < args.update_budget:
        order = _epoch_order(
            torch, pool_size, init_seed + args.shuffle_seed_offset, epoch, resident.device
        )
        for batch_number in range(updates_per_pass):
            if update >= args.update_budget:
                break
            train_started = time.perf_counter()
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
            training_seconds += time.perf_counter() - train_started
            elements = int(batch.numel())
            training_sse += float(loss.item()) * elements
            training_elements += elements
            update += 1

            first_pass_end = update == updates_per_pass
            selection_candidate = (
                update % args.tune_interval_updates == 0 or update == args.update_budget
            )
            needs_evaluation = first_pass_end or selection_candidate
            if needs_evaluation:
                evaluation_started = time.perf_counter()
                tune_metrics = _evaluate(
                    torch, model, resident, indices.tune, device, args.evaluation_batch_size
                )
                tune_seconds += time.perf_counter() - evaluation_started
                checkpoint = {
                    "update": update,
                    "presentations": update * args.batch_size,
                    "equivalent_passes": update / updates_per_pass,
                    "completed_passes": update // updates_per_pass,
                    "partial_pass_updates": update % updates_per_pass,
                    "unique_training_rows_presented": pool_size,
                    "presentations_per_unique_training_row": (update * args.batch_size / pool_size),
                    "cumulative_training_mse": training_sse / training_elements,
                    "tune": tune_metrics,
                }
                if first_pass_record is None and first_pass_end:
                    first_pass_record = dict(checkpoint)
                if update == args.update_budget:
                    fixed_budget_record = dict(checkpoint)
                if selection_candidate:
                    selection_trace.append(checkpoint)
                    if tune_metrics["reconstruction_nmse"] < best_tune_nmse:
                        best_tune_nmse = tune_metrics["reconstruction_nmse"]
                        best_update = update
                        best_state = _clone_state_to_cpu(model.state_dict())
        epoch += 1

    if first_pass_record is None or fixed_budget_record is None or best_state is None:
        raise RuntimeError("the configured budget did not produce all required checkpoints")
    model.load_state_dict(best_state)
    test_started = time.perf_counter()
    test_metrics = _evaluate(
        torch, model, resident, indices.test, device, args.evaluation_batch_size
    )
    test_seconds = time.perf_counter() - test_started
    best_trace = next(item for item in selection_trace if item["update"] == best_update)

    arm = {
        "init_seed": int(init_seed),
        "pool_size": pool_size,
        "pool_indices_sha256": _sha256_indices(pool),
        "initial_state_sha256": initial_state_sha256,
        "updates_per_pass": updates_per_pass,
        "first_pass": first_pass_record,
        "fixed_budget": {
            **fixed_budget_record,
            "update_budget": int(args.update_budget),
            "unique_training_rows": pool_size,
            "exact_full_batches": int(args.update_budget),
        },
        "selection": {
            "criterion": "minimum tune reconstruction NMSE on the common fixed-update grid",
            "interval_updates": int(args.tune_interval_updates),
            "candidate_count": len(selection_trace),
            "candidate_updates": [item["update"] for item in selection_trace],
            "best_update": int(best_update),
            "best_tune_checkpoint": best_trace,
            "first_pass_is_candidate_only_when_on_common_grid_or_final": True,
        },
        "test": {
            **test_metrics,
            "evaluated_once": True,
            "checkpoint_selected_without_test": True,
        },
        "timing_seconds": {
            "optimizer_training": training_seconds,
            "tune_evaluations": tune_seconds,
            "single_final_test_evaluation": test_seconds,
            "arm_total": time.perf_counter() - arm_started,
        },
        "precision": {
            "training": "float32",
            "parameters": sorted(
                {str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()}
            ),
            "loss": "float32",
            "evaluation": "float32",
        },
    }
    return arm, best_state


def _summary_row(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "init_seed": arm["init_seed"],
        "pool_size": arm["pool_size"],
        "pool_indices_sha256": arm["pool_indices_sha256"],
        "initial_state_sha256": arm["initial_state_sha256"],
        "updates_per_pass": arm["updates_per_pass"],
        "first_pass_tune_nmse": arm["first_pass"]["tune"]["reconstruction_nmse"],
        "first_pass_tune_cosine": arm["first_pass"]["tune"]["reconstruction_cosine"],
        "fixed_budget_updates": arm["fixed_budget"]["update"],
        "fixed_budget_presentations": arm["fixed_budget"]["presentations"],
        "fixed_budget_equivalent_passes": arm["fixed_budget"]["equivalent_passes"],
        "fixed_budget_tune_nmse": arm["fixed_budget"]["tune"]["reconstruction_nmse"],
        "fixed_budget_tune_cosine": arm["fixed_budget"]["tune"]["reconstruction_cosine"],
        "best_update": arm["selection"]["best_update"],
        "best_tune_nmse": arm["selection"]["best_tune_checkpoint"]["tune"]["reconstruction_nmse"],
        "test_nmse": arm["test"]["reconstruction_nmse"],
        "test_cosine": arm["test"]["reconstruction_cosine"],
        "optimizer_training_seconds": arm["timing_seconds"]["optimizer_training"],
        "tune_evaluation_seconds": arm["timing_seconds"]["tune_evaluations"],
        "test_evaluation_seconds": arm["timing_seconds"]["single_final_test_evaluation"],
        "arm_total_seconds": arm["timing_seconds"]["arm_total"],
    }


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result CSV")
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode())


def validate_args(args, source: SourceInfo) -> None:
    if not 0 < args.universe_rows <= source.rows:
        raise ValueError("universe_rows must fit in the validated source row count")
    if not 0 < args.target_dimension < source.dimension:
        raise ValueError("target_dimension must be smaller than the source dimension")
    if len(args.init_seeds) < 1 or len(set(args.init_seeds)) != len(args.init_seeds):
        raise ValueError("init_seeds must be non-empty and unique")
    if not args.pool_sizes or any(size < 1 for size in args.pool_sizes):
        raise ValueError("pool_sizes must be non-empty and positive")
    if (
        min(
            args.batch_size,
            args.evaluation_batch_size,
            args.load_block_rows,
            args.update_budget,
            args.tune_interval_updates,
            args.test_rows,
            args.tune_rows,
        )
        < 1
    ):
        raise ValueError("row counts, batch sizes, and update budget must be positive")
    if tuple(sorted(set(args.pool_sizes))) != tuple(args.pool_sizes):
        raise ValueError("pool_sizes must be unique and strictly increasing")
    if any(size % args.batch_size for size in args.pool_sizes):
        raise ValueError("every training pool size must be divisible by batch_size")
    if args.update_budget * args.batch_size < max(args.pool_sizes):
        raise ValueError("update budget must permit at least one full pass over every pool")
    if args.test_rows + args.tune_rows + max(args.pool_sizes) > args.universe_rows:
        raise ValueError("test, tune, and maximum training pool exceed the fixed universe")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if any(width < 1 for width in args.hidden_dimensions):
        raise ValueError("hidden dimensions must be positive")
    if (
        min(
            args.test_seed,
            args.tune_seed,
            args.pool_seed,
            args.shuffle_seed_offset,
            *args.init_seeds,
        )
        < 0
    ):
        raise ValueError("all seeds and seed offsets must be non-negative")


def _arguments_json(args) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "current_pool_size"
    }


def run_experiment(args) -> dict[str, Any]:
    import torch

    started_at = _utc_now()
    total_started = time.perf_counter()
    source = inspect_source(args.data)
    validate_args(args, source)
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
    published_default = (
        args.universe_rows == DEFAULT_UNIVERSE_ROWS
        and args.test_rows == DEFAULT_TEST_ROWS
        and args.test_seed == PUBLISHED_TEST_SEED
    )
    if published_default and test_hash != PUBLISHED_TEST_INDICES_SHA256:
        raise RuntimeError("published Deep1B test indices do not match their pinned SHA-256")

    test_tune_intersection = int(np.intersect1d(indices.test, indices.tune).size)
    test_training_intersection = int(np.intersect1d(indices.test, indices.training_order).size)
    tune_training_intersection = int(np.intersect1d(indices.tune, indices.training_order).size)
    split_union = np.concatenate((indices.test, indices.tune, indices.training_order))
    exact_unique = int(np.unique(split_union).size)
    expected_unique = int(split_union.size)
    if (
        any((test_tune_intersection, test_training_intersection, tune_training_intersection))
        or exact_unique != expected_unique
    ):
        raise RuntimeError("test, tune, and training rows are not disjoint")

    device = _resolve_device(torch, args.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.cuda.reset_peak_memory_stats(device)
    resident, residency = load_resident_source(torch, source, args, device)
    arms = []
    checkpoint_paths = []
    for init_seed in args.init_seeds:
        paired_initial_hash = None
        for pool_size in args.pool_sizes:
            args.current_pool_size = pool_size
            arm, best_state = run_arm(torch, resident, indices, source.dimension, args, init_seed)
            if paired_initial_hash is None:
                paired_initial_hash = arm["initial_state_sha256"]
            elif arm["initial_state_sha256"] != paired_initial_hash:
                raise RuntimeError("paired pool-size arms did not receive identical initialization")
            if args.checkpoint_dir is not None:
                checkpoint_path = Path(args.checkpoint_dir) / (
                    f"best-seed-{init_seed}-pool-{pool_size}.pt"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(_checkpoint_payload(arm, best_state), checkpoint_path)
                arm["checkpoint_path"] = str(checkpoint_path.resolve())
                checkpoint_paths.append(str(checkpoint_path.resolve()))
            else:
                arm["checkpoint_path"] = None
            arms.append(arm)

    source_sha_started = time.perf_counter()
    source_sha256 = _sha256_file(source.path)
    source_sha_seconds = time.perf_counter() - source_sha_started
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "arguments": _arguments_json(args),
        "source": {
            "path": str(source.path),
            "format": source.format,
            "rows": source.rows,
            "dimension": source.dimension,
            "dtype": source.dtype,
            "file_bytes": source.file_bytes,
            "file_sha256": source_sha256,
            "file_sha256_seconds": source_sha_seconds,
            "universe_policy": "fixed leading prefix",
            "universe_rows": args.universe_rows,
            "input_transformation": "none",
        },
        "split": {
            "test": {
                "rows": int(indices.test.size),
                "seed": int(args.test_seed),
                "indices_sha256": test_hash,
                "published_10m_validation_identity_expected": (
                    PUBLISHED_TEST_INDICES_SHA256 if published_default else None
                ),
                "published_10m_validation_identity_verified": bool(published_default),
                "minimum_index": int(indices.test[0]),
                "maximum_index": int(indices.test[-1]),
            },
            "tune": {
                "rows": int(indices.tune.size),
                "seed": int(args.tune_seed),
                "indices_sha256": _sha256_indices(indices.tune),
                "minimum_index": int(indices.tune[0]),
                "maximum_index": int(indices.tune[-1]),
            },
            "training": {
                "pool_seed": int(args.pool_seed),
                "nested_uniform_order_sha256": _sha256_indices(indices.training_order),
                "pool_sizes": list(args.pool_sizes),
                "pool_indices_sha256": {
                    str(size): _sha256_indices(indices.training_order[:size])
                    for size in args.pool_sizes
                },
                "unused_eligible_rows": int(indices.unused.size),
                "unused_indices_sha256": _sha256_indices(indices.unused),
            },
            "disjointness": {
                "test_tune_intersection": test_tune_intersection,
                "test_training_intersection": test_training_intersection,
                "tune_training_intersection": tune_training_intersection,
                "combined_rows": expected_unique,
                "exact_unique_rows": exact_unique,
                "test_used_for_training_or_selection": False,
            },
        },
        "protocol_configuration": {
            "target_dimension": args.target_dimension,
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
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "elementwise reconstruction MSE",
            "batch_size": args.batch_size,
            "update_budget": args.update_budget,
            "tune_interval_updates": args.tune_interval_updates,
            "presentations_per_fixed_budget_arm": args.update_budget * args.batch_size,
            "init_seeds": list(args.init_seeds),
            "precision": "CUDA FP32" if device.type == "cuda" else "CPU FP32 (test mode)",
            "test_evaluation_policy": "once per arm after tune-only checkpoint selection",
            "full_layouts_written": False,
            "checkpoint_writes_requested": args.checkpoint_dir is not None,
        },
        "residency": residency,
        "arms": arms,
        "artifacts": {"checkpoint_paths": checkpoint_paths},
        "resources": {"cuda": cuda_memory},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
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
        "timing_seconds": {"total": time.perf_counter() - total_started},
    }
    return result


def execute(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    if not args.force and (result_path.exists() or csv_path.exists()):
        raise FileExistsError("summary output exists; pass --force to replace it")
    result = run_experiment(args)
    _atomic_json(result_path, result)
    _atomic_csv(csv_path, [_summary_row(arm) for arm in result["arms"]])
    return result


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected one or more non-negative integers")
    return parsed


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/deep1b-ae-training-size")
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--universe-rows", type=int, default=DEFAULT_UNIVERSE_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--tune-rows", type=int, default=DEFAULT_TUNE_ROWS)
    parser.add_argument("--test-seed", type=int, default=PUBLISHED_TEST_SEED)
    parser.add_argument("--tune-seed", type=int, default=DEFAULT_TUNE_SEED)
    parser.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED)
    parser.add_argument("--shuffle-seed-offset", type=int, default=10_000_019)
    parser.add_argument("--pool-sizes", type=_comma_separated_ints, default=DEFAULT_POOL_SIZES)
    parser.add_argument("--init-seeds", type=_comma_separated_ints, default=DEFAULT_INIT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--update-budget", type=int, default=DEFAULT_UPDATE_BUDGET)
    parser.add_argument("--tune-interval-updates", type=int, default=DEFAULT_TUNE_INTERVAL_UPDATES)
    parser.add_argument("--target-dimension", type=int, default=8)
    parser.add_argument("--hidden-dimensions", type=_comma_separated_ints, default=(256, 128, 32))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--source-residency", choices=("auto", "device", "host"), default="auto")
    parser.add_argument("--cuda-reserve-gib", type=float, default=2.0)
    parser.add_argument("--load-block-rows", type=int, default=131_072)
    args = parser.parse_args(argv)
    if args.cuda_reserve_gib < 0:
        parser.error("--cuda-reserve-gib must be non-negative")
    args.data = args.data.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.resolve()
    return args


def _test_args(**overrides):
    defaults = {
        "data": Path("unused.npy"),
        "output_dir": Path("unused-output"),
        "force": False,
        "checkpoint_dir": None,
        "universe_rows": 64,
        "test_rows": 8,
        "tune_rows": 8,
        "test_seed": PUBLISHED_TEST_SEED,
        "tune_seed": DEFAULT_TUNE_SEED,
        "pool_seed": DEFAULT_POOL_SEED,
        "shuffle_seed_offset": 10_000_019,
        "pool_sizes": (8, 16, 48),
        "init_seeds": (101, 202, 303),
        "batch_size": 4,
        "evaluation_batch_size": 8,
        "update_budget": 12,
        "tune_interval_updates": 4,
        "target_dimension": 2,
        "hidden_dimensions": (10, 4),
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "device": "cpu",
        "source_residency": "auto",
        "cuda_reserve_gib": 0.0,
        "load_block_rows": 16,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute(args)
    print(f"wrote {len(result['arms'])} arms to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
