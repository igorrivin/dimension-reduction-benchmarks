#!/usr/bin/env python3
"""Bounded-memory streaming autoencoder experiment for Yandex ``.fbin`` data.

This is a fixed-budget scaling protocol, not the validation-selection/full-refit
protocol in :mod:`experiments.compare_large_embeddings`.  A small deterministic
holdout is excluded from every training pass.  Training visits a deterministic
shuffle of source blocks and performs a bounded shuffle inside each block; it
never constructs a corpus-sized permutation or copies the complete corpus to a
GPU.  The final layout is encoded sequentially into an atomically published NPY
file without a corpus-sized input or output mapping.

Example
-------
Train for two passes over a 10-million-row Deep prefix::

    python experiments/compare_yandex_streaming.py \
      --data deep1B_base.fbin --rows 10000000 --passes 2 \
      --target-dimension 8 --bf16 --device cuda \
      --output-dir artifacts/yandex-deep-10m-ae

``--max-updates`` may cap a run before all requested passes complete.  The JSON
result distinguishes examples consumed from source rows materialized and reports
both completed and equivalent passes, making partial runs auditable.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
import traceback
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from experiments.yandex_fbin import read_fbin_header
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from yandex_fbin import read_fbin_header


PROTOCOL = "streaming-fixed-budget-v1"
SCHEMA_VERSION = 1


class BoundedFbinReader:
    """Strict positional reader with no persistent corpus-sized memory mapping."""

    def __init__(self, path: Path, info, *, rows: int):
        self.path = Path(path)
        self.info = info
        self.rows = int(rows)
        self.dimension = int(info.dimension)
        self.row_bytes = self.dimension * np.dtype("<f4").itemsize
        if not 0 < self.rows <= int(info.rows):
            raise ValueError(f"reader row limit {self.rows} is outside the fbin row count")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        self._fd = os.open(self.path, flags)
        self.mapped_bytes = 0

    def close(self) -> None:
        descriptor, self._fd = self._fd, -1
        if descriptor >= 0:
            os.close(descriptor)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):  # pragma: no cover - defensive cleanup on exceptional unwinding.
        with contextlib.suppress(OSError):
            self.close()

    def _pread_exact(self, offset: int, length: int) -> bytearray:
        buffer = bytearray(length)
        view = memoryview(buffer)
        cursor = 0
        while cursor < length:
            try:
                if hasattr(os, "preadv"):
                    received = os.preadv(self._fd, [view[cursor:]], offset + cursor)
                else:  # pragma: no cover - Linux benchmark hosts expose preadv.
                    chunk = os.pread(self._fd, length - cursor, offset + cursor)
                    received = len(chunk)
                    view[cursor : cursor + received] = chunk
            except InterruptedError:
                continue
            if received <= 0:
                raise ValueError(
                    f"short fbin positional read at byte {offset + cursor}: "
                    f"expected {length - cursor} more bytes"
                )
            cursor += received
        return buffer

    def read_header(self) -> bytes:
        return bytes(self._pread_exact(0, 8))

    def read_block(self, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= self.rows:
            raise IndexError(f"invalid bounded fbin block [{start}, {stop})")
        payload = self._pread_exact(8 + start * self.row_bytes, (stop - start) * self.row_bytes)
        block = np.frombuffer(payload, dtype="<f4").reshape(stop - start, self.dimension)
        if not block.dtype.isnative:
            block = block.astype(np.float32, copy=True)
        if not np.all(np.isfinite(block)):
            raise ValueError(f"source block [{start}, {stop}) contains non-finite values")
        return block

    def read_rows(self, indices: np.ndarray) -> np.ndarray:
        """Read sorted arbitrary rows with bounded direct reads of consecutive runs."""

        selected = np.asarray(indices, dtype=np.int64)
        if (
            selected.ndim != 1
            or selected.size < 1
            or np.any(np.diff(selected) <= 0)
            or selected[0] < 0
            or selected[-1] >= self.rows
        ):
            raise ValueError("direct-read indices must be sorted, unique, and in bounds")
        output = np.empty((selected.size, self.dimension), dtype=np.float32)
        output_cursor = 0
        run_start = 0
        for position in range(1, selected.size + 1):
            if position < selected.size and selected[position] == selected[position - 1] + 1:
                continue
            first = int(selected[run_start])
            stop = int(selected[position - 1]) + 1
            block = self.read_block(first, stop)
            length = stop - first
            output[output_cursor : output_cursor + length] = block
            output_cursor += length
            run_start = position
        return output


@dataclass(frozen=True)
class BlockSelection:
    """One bounded training block and its shuffled absolute row indices."""

    block_id: int
    start: int
    stop: int
    indices: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, serialized.encode("utf-8"))


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


def _sha256_indices(indices: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def _source_snapshot(path: Path, header_bytes: int = 8) -> dict[str, Any]:
    """Capture identity fields that reveal replacement or ordinary in-place mutation."""

    stat = path.stat()
    with path.open("rb") as handle:
        header = handle.read(header_bytes)
    return {
        "resolved_path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _load_source_manifest(path: Path, info) -> dict[str, Any]:
    """Load and structurally validate the downloader provenance sidecar, if present."""

    manifest_path = Path(f"{path}.manifest.json")
    if not manifest_path.exists():
        return {"present": False, "path": str(manifest_path.resolve())}
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError(f"source manifest is unexpectedly large: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid source manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("prefix"), dict):
        raise ValueError(f"source manifest {manifest_path} has no prefix object")
    prefix = payload["prefix"]
    expected = {
        "rows": int(info.rows),
        "dimension": int(info.dimension),
        "file_bytes": int(info.expected_size),
    }
    observed = {key: prefix.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"source manifest prefix metadata disagrees with fbin: {observed} != {expected}"
        )
    expected_sha256 = prefix.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"source manifest {manifest_path} has no valid prefix SHA-256")
    return {
        "present": True,
        "path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_fbin_sha256": expected_sha256,
        "source": payload.get("source"),
    }


def conservative_host_memory_plan(
    *,
    rows: int,
    input_dimension: int,
    target_dimension: int,
    validation_rows: int,
    block_rows: int,
    batch_size: int,
    budget_bytes: int,
) -> dict[str, Any]:
    """Conservatively bound explicit host buffers created by this harness."""

    effective_block = min(rows, block_rows)
    effective_batch = min(effective_block, batch_size)
    components = {
        "source_block_float32": effective_block * input_dimension * 4,
        "little_endian_digest_copy_upper_bound": effective_block * input_dimension * 4,
        "source_block_finiteness_mask": effective_block * input_dimension,
        # arange, filtered indices, absolute-to-local conversion, and keep mask.
        "block_index_working_sets": effective_block * (8 + 8 + 8 + 1),
        "host_training_batch_float32": effective_batch * input_dimension * 4,
        "host_encoded_batch_float32": effective_batch * target_dimension * 4,
        "heldout_fancy_and_owned_float32": validation_rows * input_dimension * 8,
        # astype(float64) and subtraction output can coexist during centering.
        "heldout_simultaneous_float64_workspaces": validation_rows * input_dimension * 16,
        "heldout_indices_int64": validation_rows * 8,
        # Conservative CPython set slot + integer object + sorted-list pointer bound.
        "floyd_sampler_python_set_upper_bound": validation_rows * 80,
        "fixed_runtime_slack": 64 * 1024 * 1024,
    }
    estimated = sum(components.values())
    if estimated > budget_bytes:
        raise ValueError(
            "configured validation/block/batch buffers exceed --host-memory-budget-gib: "
            f"estimated {estimated} bytes, budget {budget_bytes} bytes"
        )
    return {
        "budget_bytes": int(budget_bytes),
        "conservative_estimated_peak_buffer_bytes": int(estimated),
        "components_bytes": {key: int(value) for key, value in components.items()},
        "persistent_corpus_mapping_bytes": 0,
        "persistent_layout_mapping_bytes": 0,
        "input_access": "os.preadv into one owned bounded source block",
        "estimate_excludes_framework_runtime_and_file-backed_page-cache_rss": True,
    }


def _layout_disk_preflight(
    destination: Path, rows: int, target_dimension: int, reserve_bytes: int
) -> dict[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(destination.parent)
    estimated_file_bytes = rows * target_dimension * 4 + 4096
    required_free = estimated_file_bytes + reserve_bytes
    if usage.free < required_free:
        raise OSError(
            errno.ENOSPC,
            f"layout needs about {estimated_file_bytes} bytes plus {reserve_bytes} bytes "
            f"reserve, but only {usage.free} bytes are free",
            str(destination.parent),
        )
    return {
        "estimated_layout_file_bytes": int(estimated_file_bytes),
        "reserve_bytes": int(reserve_bytes),
        "required_free_bytes": int(required_free),
        "free_bytes_before_training": int(usage.free),
    }


def bounded_validation_indices(total_rows: int, validation_rows: int, seed: int) -> np.ndarray:
    """Sample a fixed holdout with O(validation_rows), not O(total_rows), memory.

    Floyd's algorithm draws a uniform sample without replacement while storing
    only the selected values.  Sorting the bounded result supports fast exclusion
    from sequential source blocks.
    """

    if total_rows < 2:
        raise ValueError("at least two source rows are required")
    if not 0 < validation_rows < total_rows:
        raise ValueError("validation_rows must be in [1, total_rows - 1]")
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    for upper in range(total_rows - validation_rows, total_rows):
        candidate = int(rng.integers(0, upper + 1))
        selected.add(upper if candidate in selected else candidate)
    result = np.fromiter(sorted(selected), dtype=np.int64, count=validation_rows)
    if result.size != validation_rows:
        raise RuntimeError("bounded validation sampler produced an invalid sample")
    return result


def deterministic_block_order(
    total_rows: int, block_rows: int, seed: int, pass_index: int
) -> Iterator[int]:
    """Yield a deterministic O(1)-state permutation of source block identifiers."""

    if total_rows < 1 or block_rows < 1:
        raise ValueError("total_rows and block_rows must be positive")
    if pass_index < 0:
        raise ValueError("pass_index must be non-negative")
    block_count = math.ceil(total_rows / block_rows)
    if block_count == 1:
        yield 0
        return
    rng = np.random.default_rng(np.random.SeedSequence([seed, pass_index, block_count]))
    offset = int(rng.integers(0, block_count))
    step = int(rng.integers(1, block_count))
    while math.gcd(step, block_count) != 1:
        step += 1
        if step == block_count:
            step = 1
    for position in range(block_count):
        yield (offset + position * step) % block_count


def iter_training_blocks(
    total_rows: int,
    block_rows: int,
    validation_indices: np.ndarray,
    *,
    seed: int,
    pass_index: int,
) -> Iterator[BlockSelection]:
    """Yield block-local shuffles after excluding the fixed validation sample.

    The largest index arrays created by this iterator contain at most
    ``block_rows`` entries.  Absolute row indices make the exclusion invariant
    straightforward to audit in tests and logs.
    """

    validation = np.asarray(validation_indices, dtype=np.int64)
    if validation.ndim != 1 or validation.size < 1:
        raise ValueError("validation_indices must be a non-empty vector")
    if np.any(np.diff(validation) <= 0) or validation[0] < 0 or validation[-1] >= total_rows:
        raise ValueError("validation_indices must be sorted, unique, and in bounds")

    for block_id in deterministic_block_order(total_rows, block_rows, seed, pass_index):
        start = block_id * block_rows
        stop = min(total_rows, start + block_rows)
        indices = np.arange(start, stop, dtype=np.int64)
        left = int(np.searchsorted(validation, start, side="left"))
        right = int(np.searchsorted(validation, stop, side="left"))
        if right > left:
            excluded = validation[left:right] - start
            keep = np.ones(stop - start, dtype=bool)
            keep[excluded] = False
            indices = indices[keep]
        local_rng = np.random.default_rng(
            np.random.SeedSequence([seed, pass_index, block_id, 0xA17E])
        )
        local_rng.shuffle(indices)
        yield BlockSelection(block_id=block_id, start=start, stop=stop, indices=indices)


def _auto_hidden_dimensions(input_dimension: int) -> tuple[int, ...]:
    return (256, 128, 32) if input_dimension <= 512 else (512, 256, 64)


def _build_autoencoder(torch, input_dimension: int, target_dimension: int, hidden: Sequence[int]):
    class StreamingAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder: list[Any] = []
            previous = input_dimension
            for width in hidden:
                encoder.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            encoder.append(torch.nn.Linear(previous, target_dimension))
            self.encoder = torch.nn.Sequential(*encoder)

            decoder: list[Any] = []
            previous = target_dimension
            for width in reversed(hidden):
                decoder.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            decoder.append(torch.nn.Linear(previous, input_dimension))
            self.decoder = torch.nn.Sequential(*decoder)

        def forward(self, batch):
            return self.decoder(self.encoder(batch))

    return StreamingAutoencoder()


def _resolve_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested)
    return device


def _validate_precision(torch, device, bf16: bool) -> None:
    if not bf16 or device.type != "cuda":
        return
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    try:
        supported = checker(including_emulation=False) if checker is not None else False
    except TypeError:
        supported = checker() if checker is not None else False
    if not supported:
        raise RuntimeError("native CUDA BF16 support is required for --bf16")


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def _host_block(source: BoundedFbinReader, start: int, stop: int) -> np.ndarray:
    return source.read_block(start, stop)


def _evaluate_holdout(torch, model, validation: np.ndarray, device, batch_size: int) -> dict:
    """Evaluate the fixed holdout entirely in FP32."""

    sse = 0.0
    cosine_sum = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, validation.shape[0], batch_size):
            host = np.ascontiguousarray(validation[start : start + batch_size], dtype=np.float32)
            batch = torch.from_numpy(host).to(device)
            reconstructed = model(batch).float()
            if not bool(torch.all(torch.isfinite(reconstructed)).item()):
                raise RuntimeError("non-finite held-out reconstruction")
            difference = reconstructed - batch.float()
            sse += float(torch.sum(difference * difference).item())
            numerator = torch.sum(reconstructed * batch.float(), dim=1)
            denominator = torch.linalg.vector_norm(reconstructed, dim=1) * torch.linalg.vector_norm(
                batch.float(), dim=1
            )
            cosine_sum += float(torch.sum(numerator / denominator.clamp_min(1e-12)).item())
            rows += batch.shape[0]
    _sync(torch, device)
    mse = sse / validation.size
    centered = validation.astype(np.float64) - np.mean(validation, axis=0, dtype=np.float64)
    baseline_mse = max(
        np.finfo(np.float64).eps,
        float(np.sum(centered * centered, dtype=np.float64) / validation.size),
    )
    return {
        "heldout_reconstruction_mse": mse,
        "heldout_reconstruction_nmse": mse / baseline_mse,
        "heldout_reconstruction_cosine": cosine_sum / rows,
        "heldout_mean_predictor_mse": baseline_mse,
    }


def _stream_layout(
    torch,
    model,
    source: BoundedFbinReader,
    destination: Path,
    *,
    target_dimension: int,
    block_rows: int,
    batch_size: int,
    device,
    source_header: bytes,
) -> tuple[int, float, str]:
    """Encode in FP32 and publish NPY bytes without a corpus-sized output mapping."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    started = time.perf_counter()
    encoded_rows = 0
    source_digest = hashlib.sha256(source_header)
    try:
        with temporary.open("wb") as handle:
            np.lib.format.write_array_header_2_0(
                handle,
                {
                    "descr": np.dtype("<f4").str,
                    "fortran_order": False,
                    "shape": (source.rows, target_dimension),
                },
            )
            model.eval()
            with torch.no_grad():
                for block_start in range(0, source.rows, block_rows):
                    block_stop = min(source.rows, block_start + block_rows)
                    block = _host_block(source, block_start, block_stop)
                    little_endian_block = np.ascontiguousarray(block, dtype="<f4")
                    source_digest.update(memoryview(little_endian_block).cast("B"))
                    for local_start in range(0, block.shape[0], batch_size):
                        local_stop = min(block.shape[0], local_start + batch_size)
                        batch = torch.from_numpy(block[local_start:local_stop]).to(device)
                        encoded = model.encoder(batch).float()
                        if not bool(torch.all(torch.isfinite(encoded)).item()):
                            raise RuntimeError("non-finite streamed layout coordinates")
                        coordinates = np.ascontiguousarray(encoded.cpu().numpy(), dtype="<f4")
                        handle.write(memoryview(coordinates).cast("B"))
                        encoded_rows += local_stop - local_start
                        del batch, encoded, coordinates
                    del block, little_endian_block
            _sync(torch, device)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    return encoded_rows, time.perf_counter() - started, source_digest.hexdigest()


def _namespace_json(args) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"func"}
    }


def _failure_category(error: BaseException) -> str:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if isinstance(error, (FileNotFoundError, ValueError, TypeError)):
        return "invalid_input_or_integrity"
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "insufficient_disk"
    if "out of memory" in message or "outofmemory" in name:
        return "out_of_memory"
    if "bf16" in message or "cuda was requested" in message:
        return "unsupported_device_or_precision"
    return "runtime_error"


def _generation_layout_path(pointer: Path, run_id: str) -> Path:
    return pointer.with_name(f"{pointer.stem}.{run_id}{pointer.suffix}")


def _publish_layout_pointer(pointer: Path, generation: Path) -> None:
    """Atomically update a lightweight latest-layout symlink after result publication."""

    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
    relative_target = os.path.relpath(generation, start=pointer.parent)
    try:
        os.symlink(relative_target, temporary)
        os.replace(temporary, pointer)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def run_experiment(args) -> dict[str, Any]:
    """Run one streaming fixed-budget experiment and return its result object."""

    import torch

    started_wall = _utc_now()
    total_started = time.perf_counter()
    data_path = Path(args.data).resolve()
    source_snapshot_before = _source_snapshot(data_path)
    info = read_fbin_header(args.data)
    if _source_snapshot(data_path) != source_snapshot_before:
        raise RuntimeError("source fbin changed while its header was being validated")
    source_manifest = _load_source_manifest(data_path, info)
    if args.rows > info.rows:
        raise ValueError(f"--rows={args.rows} exceeds the validated source row count {info.rows}")
    if args.rows < 2:
        raise ValueError("--rows must be at least two")
    if args.validation_rows >= args.rows:
        raise ValueError("--validation-rows must be smaller than --rows")
    if not 0 < args.target_dimension < info.dimension:
        raise ValueError("--target-dimension must be positive and smaller than the input dimension")
    layout_pointer_path = Path(args.layout_path).resolve()
    layout_path = Path(getattr(args, "generation_layout_path", layout_pointer_path)).resolve()
    query_data_value = getattr(args, "query_data", None)
    query_path = Path(query_data_value).resolve() if query_data_value is not None else None
    query_info = None
    query_snapshot_before = None
    query_manifest = None
    query_layout_pointer_path = None
    query_layout_path = None
    if query_path is not None:
        query_snapshot_before = _source_snapshot(query_path)
        query_info = read_fbin_header(query_path)
        if _source_snapshot(query_path) != query_snapshot_before:
            raise RuntimeError("query fbin changed while its header was being validated")
        if query_info.dimension != info.dimension:
            raise ValueError(
                f"query dimension {query_info.dimension} does not match base dimension {info.dimension}"
            )
        query_manifest = _load_source_manifest(query_path, query_info)
        query_layout_pointer_path = Path(args.query_layout_path).resolve()
        query_layout_path = Path(
            getattr(args, "generation_query_layout_path", query_layout_pointer_path)
        ).resolve()

    base_layout_paths = {layout_pointer_path, layout_path}
    query_layout_paths: set[Path] = set()
    protected_paths = set(base_layout_paths)
    if query_layout_pointer_path is not None and query_layout_path is not None:
        query_layout_paths.update((query_layout_pointer_path, query_layout_path))
        protected_paths.update(query_layout_paths)
    if data_path in protected_paths or (query_path is not None and query_path in protected_paths):
        raise ValueError("the streamed layout path must not overwrite the source fbin")
    if base_layout_paths.intersection(query_layout_paths):
        raise ValueError("base and query layout pointer/generation paths must be distinct")
    host_memory_budget_bytes = int(getattr(args, "host_memory_budget_gib", 2.0) * 2**30)
    host_memory_plan = conservative_host_memory_plan(
        rows=args.rows,
        input_dimension=info.dimension,
        target_dimension=args.target_dimension,
        validation_rows=args.validation_rows,
        block_rows=args.block_rows,
        batch_size=args.batch_size,
        budget_bytes=host_memory_budget_bytes,
    )
    disk_reserve_bytes = int(getattr(args, "disk_reserve_gib", 1.0) * 2**30)
    query_layout_estimate = (
        int(query_info.rows) * args.target_dimension * 4 + 4096 if query_info is not None else 0
    )
    disk_preflight = _layout_disk_preflight(
        layout_path,
        args.rows,
        args.target_dimension,
        disk_reserve_bytes + query_layout_estimate,
    )
    query_disk_preflight = (
        _layout_disk_preflight(
            query_layout_path, query_info.rows, args.target_dimension, disk_reserve_bytes
        )
        if query_info is not None and query_layout_path is not None
        else None
    )
    source = BoundedFbinReader(data_path, info, rows=args.rows)

    validation_indices = bounded_validation_indices(
        args.rows, args.validation_rows, args.seed + args.validation_seed_offset
    )
    validation_started = time.perf_counter()
    validation = source.read_rows(validation_indices)
    if not np.all(np.isfinite(validation)):
        raise ValueError("held-out validation sample contains non-finite values")
    validation_load_seconds = time.perf_counter() - validation_started

    device = _resolve_device(torch, args.device)
    _validate_precision(torch, device, args.bf16)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    hidden = tuple(args.hidden_dimensions) or _auto_hidden_dimensions(info.dimension)
    model = _build_autoencoder(torch, info.dimension, args.target_dimension, hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("autoencoder parameters must remain FP32")

    eligible_rows = args.rows - args.validation_rows
    optimizer_updates = 0
    training_examples = 0
    materialized_rows = 0
    completed_passes = 0
    passes_started = 0
    training_loss_sum = 0.0
    training_elements = 0
    _sync(torch, device)
    training_started = time.perf_counter()
    stop = False
    for pass_index in range(args.passes):
        passes_started += 1
        pass_examples = 0
        model.train()
        for selection in iter_training_blocks(
            args.rows,
            args.block_rows,
            validation_indices,
            seed=args.seed,
            pass_index=pass_index,
        ):
            if args.max_updates is not None and optimizer_updates >= args.max_updates:
                stop = True
                break
            if selection.indices.size == 0:
                continue
            block = _host_block(source, selection.start, selection.stop)
            materialized_rows += block.shape[0]
            local_indices = selection.indices - selection.start
            for batch_start in range(0, local_indices.size, args.batch_size):
                if args.max_updates is not None and optimizer_updates >= args.max_updates:
                    stop = True
                    break
                chosen = local_indices[batch_start : batch_start + args.batch_size]
                host_batch = np.ascontiguousarray(block[chosen], dtype=np.float32)
                batch = torch.from_numpy(host_batch).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.bf16,
                ):
                    reconstructed = model(batch)
                # Keep loss arithmetic outside autocast.  Its FP32 gradient then
                # flows through the eligible BF16-autocast model operations.
                difference = reconstructed.float() - batch.float()
                loss = torch.mean(difference * difference)
                if loss.dtype != torch.float32 or not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("autoencoder training produced a non-finite FP32 loss")
                loss.backward()
                optimizer.step()
                count = int(batch.shape[0])
                optimizer_updates += 1
                training_examples += count
                pass_examples += count
                training_loss_sum += float(loss.item()) * batch.numel()
                training_elements += batch.numel()
                del batch, reconstructed, difference, loss, host_batch
            if stop:
                del block
                break
            del block
        if pass_examples == eligible_rows:
            completed_passes += 1
        if stop:
            break
    _sync(torch, device)
    training_seconds = time.perf_counter() - training_started
    if optimizer_updates < 1:
        raise RuntimeError("the training budget produced zero optimizer updates")

    metric_started = time.perf_counter()
    metrics = _evaluate_holdout(torch, model, validation, device, args.batch_size)
    metric_seconds = time.perf_counter() - metric_started

    source_header = source.read_header()
    encoded_rows, layout_seconds, consumed_source_sha256 = _stream_layout(
        torch,
        model,
        source,
        layout_path,
        target_dimension=args.target_dimension,
        block_rows=args.block_rows,
        batch_size=args.batch_size,
        device=device,
        source_header=source_header,
    )
    source.close()
    if encoded_rows != args.rows:
        raise RuntimeError(f"streamed layout encoded {encoded_rows} rows, expected {args.rows}")
    source_snapshot_after = _source_snapshot(data_path)
    if source_snapshot_after != source_snapshot_before:
        raise RuntimeError("source fbin changed during the streaming experiment")
    manifest_sha_verified = False
    if source_manifest["present"] and args.rows == info.rows:
        if consumed_source_sha256 != source_manifest["expected_fbin_sha256"]:
            raise ValueError("source fbin content does not match its download manifest SHA-256")
        manifest_sha_verified = True

    query_result = None
    if (
        query_path is not None
        and query_info is not None
        and query_snapshot_before is not None
        and query_manifest is not None
        and query_layout_path is not None
        and query_layout_pointer_path is not None
    ):
        query_source = BoundedFbinReader(query_path, query_info, rows=query_info.rows)
        query_header = query_source.read_header()
        query_rows, query_seconds, query_sha256 = _stream_layout(
            torch,
            model,
            query_source,
            query_layout_path,
            target_dimension=args.target_dimension,
            block_rows=args.block_rows,
            batch_size=args.batch_size,
            device=device,
            source_header=query_header,
        )
        query_source.close()
        if query_rows != query_info.rows:
            raise RuntimeError(
                f"streamed query layout encoded {query_rows} rows, expected {query_info.rows}"
            )
        if _source_snapshot(query_path) != query_snapshot_before:
            raise RuntimeError("query fbin changed during FP32 streamed encoding")
        query_manifest_verified = False
        if query_manifest["present"]:
            if query_sha256 != query_manifest["expected_fbin_sha256"]:
                raise ValueError("query fbin content does not match its manifest SHA-256")
            query_manifest_verified = True
        query_result = {
            "source": {
                "path": str(query_path),
                "rows": int(query_info.rows),
                "dimension": int(query_info.dimension),
                "dtype": "little-endian float32",
                "validated_file_bytes": int(query_info.expected_size),
                "identity_snapshot": query_snapshot_before,
                "content_sha256": query_sha256,
                "download_manifest": {
                    **query_manifest,
                    "content_sha256_verified": query_manifest_verified,
                },
                "access": "bounded os.preadv blocks; no persistent full-file mapping",
            },
            "encoding_precision": "float32",
            "layout_generation_path": str(query_layout_path),
            "layout_pointer_path": str(query_layout_pointer_path),
            "layout_rows": int(query_rows),
            "layout_dimension": int(args.target_dimension),
            "encoding_seconds": query_seconds,
            "rows_per_second": query_rows / query_seconds,
            "source_bytes_per_second": query_info.data_bytes / query_seconds,
            "disk_preflight": query_disk_preflight,
        }

    optimizer_state_dtypes = sorted(
        {
            str(value.dtype).removeprefix("torch.")
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        }
    )
    if optimizer_state_dtypes and optimizer_state_dtypes != ["float32"]:
        raise RuntimeError(f"optimizer state escaped FP32: {optimizer_state_dtypes}")
    cuda_memory = {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "allocated_bytes_at_end": None,
        "reserved_bytes_at_end": None,
    }
    if device.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocated_bytes_at_end": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes_at_end": int(torch.cuda.memory_reserved(device)),
        }

    row_bytes = info.dimension * np.dtype("<f4").itemsize
    training_payload_bytes = training_examples * row_bytes
    materialized_payload_bytes = materialized_rows * row_bytes
    validation_payload_bytes = args.validation_rows * row_bytes
    encoding_payload_bytes = args.rows * row_bytes
    total_seconds = time.perf_counter() - total_started
    parameter_dtypes = sorted(
        {str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()}
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "run_id": getattr(args, "run_id", None),
        "status": "ok",
        "started_at_utc": started_wall,
        "finished_at_utc": _utc_now(),
        "arguments": _namespace_json(args),
        "source": {
            "path": str(Path(args.data).resolve()),
            "declared_rows": int(info.rows),
            "rows_used": int(args.rows),
            "dimension": int(info.dimension),
            "dtype": "little-endian float32",
            "validated_file_bytes": int(info.expected_size),
            "identity_snapshot": source_snapshot_before,
            "header_plus_selected_payload_sha256": consumed_source_sha256,
            "download_manifest": {
                **source_manifest,
                "content_sha256_verified": manifest_sha_verified,
                "verification_scope": (
                    "complete fbin"
                    if manifest_sha_verified
                    else "not verified (manifest absent or experiment uses a strict prefix)"
                ),
            },
            "row_bytes": int(row_bytes),
            "row_limit_policy": "required strict prefix; complete fbin length validated",
            "access": "bounded os.preadv blocks; no persistent full-file mapping",
        },
        "query": query_result,
        "validation": {
            "rows": int(args.validation_rows),
            "indices_sha256": _sha256_indices(validation_indices),
            "minimum_index": int(validation_indices[0]),
            "maximum_index": int(validation_indices[-1]),
            "sampling_algorithm": "Floyd uniform sample without replacement",
            "excluded_from_all_training_passes": True,
            **metrics,
        },
        "model": {
            "input_dimension": int(info.dimension),
            "target_dimension": int(args.target_dimension),
            "hidden_dimensions": list(hidden),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "precision": {
            "training": "bfloat16_autocast" if args.bf16 else "float32",
            "loss": "float32",
            "parameters": parameter_dtypes,
            "optimizer_state": optimizer_state_dtypes,
            "validation": "float32",
            "encoding": "float32",
            "bf16_requested": bool(args.bf16),
        },
        "streaming": {
            "block_order": "seeded affine permutation; O(1) permutation state",
            "local_shuffle": "seeded per-block permutation",
            "maximum_block_rows": int(args.block_rows),
            "maximum_batch_rows": int(args.batch_size),
            "corpus_sized_permutation": False,
            "full_corpus_device_copy": False,
            "persistent_input_mapping_bytes": 0,
            "persistent_layout_mapping_bytes": 0,
            "input_transformation": "none (Deep fbin vectors are consumed as stored)",
            "layout_atomic_npy": str(layout_path.resolve()),
            "layout_generation_path": str(layout_path.resolve()),
            "layout_pointer_path": str(layout_pointer_path.resolve()),
            "publication_order": "immutable generation, result JSON, then atomic latest symlink",
            "host_memory_preflight": host_memory_plan,
            "disk_preflight": disk_preflight,
        },
        "budget": {
            "requested_passes": int(args.passes),
            "maximum_updates": args.max_updates,
            "passes_started": int(passes_started),
            "completed_passes": int(completed_passes),
            "equivalent_passes": training_examples / eligible_rows,
            "eligible_rows_per_pass": int(eligible_rows),
            "optimizer_updates": int(optimizer_updates),
            "training_examples": int(training_examples),
            "training_elements": int(training_elements),
            "training_payload_bytes": int(training_payload_bytes),
            "source_rows_materialized_for_training": int(materialized_rows),
            "source_payload_bytes_materialized_for_training": int(materialized_payload_bytes),
            "validation_payload_bytes": int(validation_payload_bytes),
            "encoding_payload_bytes": int(encoding_payload_bytes),
            "total_logical_source_payload_bytes_accessed": int(
                materialized_payload_bytes + validation_payload_bytes + encoding_payload_bytes
            ),
            "stopped_by_update_cap": bool(
                args.max_updates is not None
                and optimizer_updates >= args.max_updates
                and completed_passes < args.passes
            ),
        },
        "training": {
            "mean_elementwise_mse": training_loss_sum / training_elements,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "timing_seconds": {
            "validation_load": validation_load_seconds,
            "training": training_seconds,
            "heldout_evaluation": metric_seconds,
            "layout_encoding": layout_seconds,
            "total": total_seconds,
        },
        "throughput": {
            "training_examples_per_second": training_examples / training_seconds,
            "training_payload_bytes_per_second": training_payload_bytes / training_seconds,
            "materialized_source_bytes_per_training_second": (
                materialized_payload_bytes / training_seconds
            ),
            "layout_rows_per_second": encoded_rows / layout_seconds,
            "layout_source_bytes_per_second": encoding_payload_bytes / layout_seconds,
        },
        "resources": {
            "peak_rss_bytes": _rss_bytes(),
            "cuda": cuda_memory,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "device": str(device),
            "seed": int(args.seed),
            "validation_seed_offset": int(args.validation_seed_offset),
            "torch_deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        },
    }
    return result


def execute(args) -> int:
    """Write either a complete success record or an explicit atomic failure record."""

    result_path = Path(args.result_path)
    layout_path = Path(args.layout_path)
    query_requested = getattr(args, "query_data", None) is not None
    query_layout_path = Path(args.query_layout_path) if query_requested else None
    result_preexisted = result_path.exists()
    layout_preexisted = os.path.lexists(layout_path)
    query_layout_preexisted = (
        os.path.lexists(query_layout_path) if query_layout_path is not None else False
    )
    if not args.force and (result_preexisted or layout_preexisted or query_layout_preexisted):
        raise FileExistsError("result or layout already exists; pass --force to replace it")
    if result_preexisted and result_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("refusing to replace an unexpectedly large existing result JSON")
    run_id = uuid.uuid4().hex
    generation_path = _generation_layout_path(layout_path, run_id)
    query_generation_path = (
        _generation_layout_path(query_layout_path, run_id)
        if query_layout_path is not None
        else None
    )
    setattr(args, "run_id", run_id)
    setattr(args, "generation_layout_path", generation_path)
    if query_generation_path is not None:
        setattr(args, "generation_query_layout_path", query_generation_path)
    started = _utc_now()
    try:
        result = run_experiment(args)
        result["streaming"]["layout_pointer_publication"] = {"status": "pending"}
        if result["query"] is not None:
            result["query"]["layout_pointer_publication"] = {"status": "pending"}
        _atomic_json(result_path, result)
    except (KeyboardInterrupt, SystemExit):
        with contextlib.suppress(FileNotFoundError):
            generation_path.unlink()
        if query_generation_path is not None:
            with contextlib.suppress(FileNotFoundError):
                query_generation_path.unlink()
        raise
    except Exception as error:
        # The immutable generation belongs only to this attempt.  A pre-existing
        # good latest-layout pointer is never touched until a new result succeeds.
        with contextlib.suppress(FileNotFoundError):
            generation_path.unlink()
        if query_generation_path is not None:
            with contextlib.suppress(FileNotFoundError):
                query_generation_path.unlink()
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "failed",
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "arguments": _namespace_json(args),
            "failure": {
                "category": _failure_category(error),
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        failure_path = (
            result_path.with_name(f"{result_path.stem}.{run_id}.failed{result_path.suffix}")
            if result_preexisted
            else result_path
        )
        _atomic_json(failure_path, failure)
        return 1

    # The immutable generation and canonical result are already valid.  Updating
    # the convenience pointer is therefore deliberately nonfatal: deleting the
    # generation here would invalidate the successfully published result.
    try:
        _publish_layout_pointer(layout_path, generation_path)
        pointer_record = {"status": "ok", "path": str(layout_path.absolute())}
    except Exception as error:  # pragma: no cover - exact filesystem errors vary.
        pointer_record = {
            "status": "failed_nonfatal",
            "type": type(error).__name__,
            "message": str(error),
            "generation_remains_canonical": True,
        }
    result["streaming"]["layout_pointer_publication"] = pointer_record
    if query_layout_path is not None and query_generation_path is not None:
        try:
            _publish_layout_pointer(query_layout_path, query_generation_path)
            query_pointer_record = {
                "status": "ok",
                "path": str(query_layout_path.absolute()),
            }
        except Exception as error:  # pragma: no cover - filesystem-dependent.
            query_pointer_record = {
                "status": "failed_nonfatal",
                "type": type(error).__name__,
                "message": str(error),
                "generation_remains_canonical": True,
            }
        result["query"]["layout_pointer_publication"] = query_pointer_record
    try:
        _atomic_json(result_path, result)
    except OSError as error:  # The earlier pending-state result remains usable.
        print(
            f"warning: could not finalize pointer status in {result_path}: {error}", file=sys.stderr
        )
    return 0


def _parse_dimensions(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        dimensions = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "hidden dimensions must be comma-separated integers"
        ) from error
    if any(dimension < 1 for dimension in dimensions):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive")
    return dimensions


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, type=Path, help="strict Yandex/Big-ANN fbin")
    parser.add_argument(
        "--query-data",
        type=Path,
        help="optional strict fbin query set encoded in FP32 by the trained encoder",
    )
    parser.add_argument(
        "--rows",
        required=True,
        type=int,
        help="required source-prefix row limit; the harness never silently uses all rows",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/yandex-streaming-ae"))
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--layout-path", type=Path)
    parser.add_argument("--query-layout-path", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-dimension", type=int, default=8)
    parser.add_argument("--hidden-dimensions", type=_parse_dimensions, default=())
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--validation-rows", type=int, default=65_536)
    parser.add_argument("--block-rows", type=int, default=131_072)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--host-memory-budget-gib",
        type=float,
        default=2.0,
        help="hard budget for conservatively estimated harness-owned host buffers",
    )
    parser.add_argument(
        "--disk-reserve-gib",
        type=float,
        default=1.0,
        help="free space that must remain beyond the estimated generation layout size",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--validation-seed-offset", type=int, default=1_000_003)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="BF16 autocast for training only; loss, parameters, optimizer, validation, and encoding stay FP32",
    )
    args = parser.parse_args(argv)
    positive = {
        "rows": args.rows,
        "target-dimension": args.target_dimension,
        "passes": args.passes,
        "validation-rows": args.validation_rows,
        "block-rows": args.block_rows,
        "batch-size": args.batch_size,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        parser.error(f"positive values required for: {', '.join(invalid)}")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive when supplied")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if args.host_memory_budget_gib <= 0 or args.disk_reserve_gib < 0:
        parser.error("host-memory-budget-gib must be positive and disk-reserve-gib non-negative")
    if args.seed < 0 or args.validation_seed_offset < 0:
        parser.error("seed and validation-seed-offset must be non-negative")
    args.output_dir = args.output_dir.resolve()
    args.result_path = (args.result_path or args.output_dir / "result.json").resolve()
    args.layout_path = (args.layout_path or args.output_dir / "layout.npy").resolve()
    resolved_data = args.data.resolve()
    if args.query_data is not None:
        args.query_data = args.query_data.resolve()
        args.query_layout_path = (
            args.query_layout_path or args.output_dir / "query-layout.npy"
        ).resolve()
        paths = {
            resolved_data,
            args.query_data,
            args.result_path,
            args.layout_path,
            args.query_layout_path,
        }
        if len(paths) != 5:
            parser.error("base/query data, result, and both layout paths must be distinct")
        if args.query_layout_path.suffix != ".npy":
            parser.error("--query-layout-path must have an .npy suffix")
    elif args.query_layout_path is not None:
        parser.error("--query-layout-path requires --query-data")
    if len({resolved_data, args.result_path, args.layout_path}) != 3:
        parser.error("data, result, and layout paths must be distinct")
    if args.layout_path.suffix != ".npy":
        parser.error("--layout-path must have an .npy suffix")
    return args


def _test_args(**overrides):
    """Build a complete namespace for focused CPU tests without invoking argparse."""

    defaults = {
        "data": Path("unused.fbin"),
        "query_data": None,
        "rows": 32,
        "output_dir": Path("unused"),
        "result_path": Path("unused/result.json"),
        "layout_path": Path("unused/layout.npy"),
        "query_layout_path": None,
        "force": False,
        "target_dimension": 2,
        "hidden_dimensions": (8, 4),
        "passes": 1,
        "max_updates": None,
        "validation_rows": 4,
        "block_rows": 8,
        "batch_size": 4,
        "host_memory_budget_gib": 2.0,
        "disk_reserve_gib": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "seed": 1729,
        "validation_seed_offset": 1_000_003,
        "device": "cpu",
        "bf16": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main(argv: Sequence[str] | None = None) -> int:
    return execute(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
