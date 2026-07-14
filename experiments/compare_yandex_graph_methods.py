#!/usr/bin/env python3
"""Run one isolated DiRe or RAPIDS UMAP fit on a Yandex ``fbin`` prefix.

The command deliberately represents one experimental cell: one input prefix, one row
limit, one method, one target dimension, and one seed.  The public process launches a
fresh worker so that CUDA OOMs, host OOM kills, and timeouts can be recorded without
losing the cell result.  Both the NPY layout and JSON result are published atomically.

Examples
--------
RAPIDS 26.06 UMAP with partitioned NN-descent::

    conda run -n rapids-26.06 python experiments/compare_yandex_graph_methods.py \
      --data /data/yandex/DEEP100M/base.100M.fbin --method umap --rows 100000000 \
      --target-dimension 2 --output-dir artifacts/yandex/umap-n100m-d2 \
      --umap-knn-n-clusters 16 --umap-knn-overlap-factor 2 --timeout 21600

DiRe with an explicitly selected cuVS index::

    conda run -n rapids-26.06 python experiments/compare_yandex_graph_methods.py \
      --data /data/yandex/DEEP100M/base.100M.fbin --method dire --rows 100000000 \
      --target-dimension 2 --output-dir artifacts/yandex/dire-n100m-d2 \
      --dire-cuvs-index ivf_pq --dire-repo ../dire-rapids --timeout 21600

The harness opens the selected fbin prefix as a read-only host memmap.  It does not
normalize, concatenate, or otherwise make a complete host-side copy.  Backends may, of
course, stage data internally.  UMAP is forced to ``build_algo='nn_descent'``; all
RAPIDS 26.06 NN-descent and partitioning controls are exposed and recorded.
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
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dimension_reduction_benchmarks.dire_backend import (
    git_revision,
    implementation_identity,
    load_dire,
)

try:
    from experiments.yandex_fbin import open_fbin_memmap, read_fbin_header
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from yandex_fbin import open_fbin_memmap, read_fbin_header

METHODS = ("dire", "umap")
STATUSES = (
    "ok",
    "cuda_oom",
    "host_oom_or_sigkill",
    "timeout",
    "unsupported_device",
    "unsupported_scale",
    "integrity_failed",
    "metric_failed",
)
CUVS_INDEX_TYPES = ("auto", "flat", "ivf_flat", "ivf_pq", "cagra")
DEFAULT_DIRE_NEG_RATIO = 8
SCHEMA_VERSION = 1
WORKER_PHASES = {
    "worker_started": 0,
    "input_validation": 1,
    "preflight_complete": 2,
    "fit_started": 3,
    "fit_returned": 4,
    "layout_validated": 5,
    "layout_published": 6,
    "worker_failed": 7,
}


class IntegrityFailure(RuntimeError):
    """The input or layout did not satisfy the experiment contract."""


class MetricFailure(RuntimeError):
    """The post-fit layout summary could not be evaluated."""


@dataclass(frozen=True)
class ChildOutcome:
    """Observable result of an isolated worker process."""

    returncode: int | None
    wall_seconds: float
    timed_out: bool
    termination_signal: int | None = None
    launch_error: str | None = None


def _absolute_path(path: str | Path) -> Path:
    """Make a lexical absolute path without dereferencing its final symlink."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _write_heartbeat(
    args: argparse.Namespace,
    phase: str,
    *,
    attempted_fit: bool,
    last_phase: str | None = None,
) -> None:
    if phase not in WORKER_PHASES:
        raise ValueError(f"unknown worker phase {phase!r}")
    if not args.worker_heartbeat:
        return
    _atomic_json(
        args.worker_heartbeat,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "pid": os.getpid(),
            "phase": phase,
            "phase_index": WORKER_PHASES[phase],
            "last_phase": last_phase,
            "attempted_fit": attempted_fit,
            "method": args.method,
            "rows": args.rows,
            "target_dimension": args.target_dimension,
            "layout_path": str(_absolute_path(args.layout_path)),
            "updated_at": _utc_now(),
        },
    )


def _read_heartbeat(path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        heartbeat = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "method": args.method,
        "rows": args.rows,
        "target_dimension": args.target_dimension,
        "layout_path": str(_absolute_path(args.layout_path)),
    }
    if any(heartbeat.get(key) != value for key, value in expected.items()):
        return None
    if heartbeat.get("phase") not in WORKER_PHASES:
        return None
    return heartbeat


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_snapshot(path: str | Path) -> dict[str, Any]:
    """Capture cheap identity fields that reveal replacement or ordinary mutation."""
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    with source.open("rb") as handle:
        header = handle.read(8)
    return {
        "resolved_path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _assert_source_unchanged(path: str | Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        observed = _source_snapshot(path)
    except OSError as exc:
        raise IntegrityFailure(f"source fbin became unavailable: {exc}") from exc
    if observed != expected:
        raise IntegrityFailure("source fbin changed during the isolated experiment")
    return observed


def _load_source_manifest(path: str | Path, info) -> dict[str, Any]:
    """Validate downloader provenance when the cache sidecar is available."""
    source = Path(path).expanduser().resolve()
    manifest_path = Path(f"{source}.manifest.json")
    if not manifest_path.exists():
        return {"present": False, "path": str(manifest_path)}
    if manifest_path.stat().st_size > 1024 * 1024:
        raise IntegrityFailure(f"source manifest is unexpectedly large: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IntegrityFailure(f"invalid source manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("prefix"), dict):
        raise IntegrityFailure(f"source manifest {manifest_path} has no prefix object")
    prefix = payload["prefix"]
    expected = {
        "rows": int(info.rows),
        "dimension": int(info.dimension),
        "file_bytes": int(info.expected_size),
    }
    observed = {key: prefix.get(key) for key in expected}
    if observed != expected:
        raise IntegrityFailure(
            f"source manifest prefix metadata disagrees with fbin: {observed} != {expected}"
        )
    expected_sha256 = prefix.get("sha256")
    try:
        valid_sha256 = (
            isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
            and int(expected_sha256, 16) >= 0
        )
    except ValueError:
        valid_sha256 = False
    if not valid_sha256:
        raise IntegrityFailure(f"source manifest {manifest_path} has no valid prefix SHA-256")
    return {
        "present": True,
        "path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_fbin_sha256": expected_sha256,
        "full_fbin_sha256_verified_this_run": False,
        "source": payload.get("source"),
    }


def _layout_disk_preflight(
    destination: str | Path,
    rows: int,
    target_dimension: int,
    reserve_bytes: int,
) -> dict[str, int]:
    """Require room for a complete atomic generation plus an explicit reserve."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path.parent)
    estimated_file_bytes = rows * target_dimension * np.dtype(np.float32).itemsize + 4096
    required_free = estimated_file_bytes + reserve_bytes
    if usage.free < required_free:
        raise OSError(
            errno.ENOSPC,
            f"layout needs about {estimated_file_bytes} bytes plus {reserve_bytes} bytes "
            f"reserve, but only {usage.free} bytes are free",
            str(path.parent),
        )
    return {
        "estimated_layout_file_bytes": int(estimated_file_bytes),
        "reserve_bytes": int(reserve_bytes),
        "required_free_bytes": int(required_free),
        "free_bytes_before_fit": int(usage.free),
    }


def _generation_layout_path(pointer: Path, run_id: str) -> Path:
    return pointer.with_name(f"{pointer.stem}.{run_id}{pointer.suffix}")


def _publish_layout_pointer(pointer: Path, generation: Path) -> None:
    """Atomically update the convenience layout symlink after result publication."""
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.tmp-{uuid.uuid4().hex}")
    relative_target = os.path.relpath(generation, start=pointer.parent)
    try:
        os.symlink(relative_target, temporary)
        os.replace(temporary, pointer)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _paths_are_existing_aliases(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, OSError):
        return False


def _validate_publication_paths(args: argparse.Namespace) -> None:
    source = Path(args.data).expanduser().resolve()
    outputs = {
        "output directory": Path(args.output_dir).expanduser().resolve(),
        "result": _absolute_path(args.result_path),
        "layout": _absolute_path(args.layout_path),
        "log": _absolute_path(args.log_path),
    }
    collisions = [name for name, path in outputs.items() if path == source]
    if collisions:
        raise IntegrityFailure(
            f"publication path must not overwrite source fbin: {', '.join(collisions)}"
        )
    file_outputs = {name: path for name, path in outputs.items() if name != "output directory"}
    if len(set(file_outputs.values())) != len(file_outputs):
        raise IntegrityFailure("result, layout, and log paths must be distinct")
    named_files = {"source": source, **file_outputs}
    items = list(named_files.items())
    for index, (first_name, first_path) in enumerate(items):
        for second_name, second_path in items[index + 1 :]:
            if _paths_are_existing_aliases(first_path, second_path):
                raise IntegrityFailure(
                    f"{first_name} and {second_name} are aliases of the same file"
                )
    if outputs["layout"].suffix.lower() != ".npy":
        raise IntegrityFailure("layout path must end in .npy")


def _quarantine_generation(generation: Path, run_id: str) -> list[str]:
    """Move completed or partial orphan layouts aside for postmortem inspection."""
    quarantined: list[str] = []
    candidates = []
    if os.path.lexists(generation):
        candidates.append(generation)
    candidates.extend(sorted(generation.parent.glob(f".{generation.name}.tmp-*")))
    for index, candidate in enumerate(candidates):
        suffix = "" if index == 0 else f".partial-{index}"
        quarantine = generation.with_name(
            f"{generation.stem}.{run_id}.orphaned{suffix}{generation.suffix}"
        )
        try:
            os.replace(candidate, quarantine)
            quarantined.append(str(quarantine.resolve()))
        except OSError:
            # Never risk deleting an artifact whose ownership/state is uncertain.
            quarantined.append(str(candidate.resolve()))
    return quarantined


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Publish strict JSON without ever exposing a partial destination file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _logical_array_sha256(array: np.ndarray, block_rows: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], block_rows):
        stop = min(start + block_rows, array.shape[0])
        block = np.ascontiguousarray(array[start:stop], dtype="<f4")
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def _save_layout_atomic(
    path: str | Path, layout: np.ndarray, *, block_rows: int = 1_048_576
) -> dict[str, Any]:
    """Block-copy a float32 layout to an atomic NPY and validate its header."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    try:
        output = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=layout.shape
        )
        try:
            for start in range(0, layout.shape[0], block_rows):
                stop = min(start + block_rows, layout.shape[0])
                block = np.ascontiguousarray(layout[start:stop], dtype="<f4")
                output[start:stop] = block
                digest.update(block.tobytes(order="C"))
            output.flush()
        finally:
            del output
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        staged = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if staged.shape != layout.shape or staged.dtype != np.dtype(np.float32):
            raise IntegrityFailure("staged layout has an invalid NPY header")
        expected_minimum = int(layout.size * np.dtype(np.float32).itemsize)
        if temporary.stat().st_size < expected_minimum:
            raise IntegrityFailure("staged layout is shorter than its logical payload")
        del staged
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()

    reopened = np.load(destination, mmap_mode="r", allow_pickle=False)
    if reopened.shape != layout.shape or reopened.dtype != np.dtype(np.float32):
        raise IntegrityFailure("published layout has an invalid NPY header")
    return {
        "path": str(destination.resolve()),
        "rows": int(reopened.shape[0]),
        "target_dimension": int(reopened.shape[1]),
        "dtype": str(reopened.dtype),
        "file_bytes": int(destination.stat().st_size),
        "logical_data_sha256": digest.hexdigest(),
    }


def _parse_device_ids(value: str) -> str | list[int] | None:
    value = value.strip().lower()
    if value in {"", "none"}:
        return None
    if value == "all":
        return "all"
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "device IDs must be 'all', 'none', or CSV integers"
        ) from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("device IDs must be unique non-negative integers")
    return result


def selected_device_id(args: argparse.Namespace) -> int:
    """Return the single device that this harness can measure correctly."""
    device_ids = args.umap_device_ids if args.method == "umap" else None
    if isinstance(device_ids, list) and len(device_ids) == 1:
        return device_ids[0]
    return 0


def umap_build_kwds(args: argparse.Namespace) -> dict[str, int | float]:
    """Return the exact RAPIDS 26.06 scalable graph-build configuration."""
    return {
        "knn_n_clusters": args.umap_knn_n_clusters,
        "knn_overlap_factor": args.umap_knn_overlap_factor,
        "nnd_graph_degree": args.umap_nnd_graph_degree,
        "nnd_intermediate_graph_degree": args.umap_nnd_intermediate_graph_degree,
        "nnd_max_iterations": args.umap_nnd_max_iterations,
        "nnd_termination_threshold": args.umap_nnd_termination_threshold,
    }


def _dire_named_memory(
    rows: int,
    features: int,
    target_dimension: int,
    neighbors: int,
    *,
    neg_ratio: int,
) -> dict[str, int]:
    configured_negatives = min(neg_ratio * neighbors, rows - 1)
    default_negatives = min(DEFAULT_DIRE_NEG_RATIO * neighbors, rows - 1)
    input_bytes = rows * features * np.dtype(np.float32).itemsize
    knn_indices = rows * neighbors * np.dtype(np.int64).itemsize
    knn_distances = rows * neighbors * np.dtype(np.float32).itemsize
    layout = rows * target_dimension * np.dtype(np.float32).itemsize
    configured_negative = rows * configured_negatives * np.dtype(np.int64).itemsize
    default_negative = rows * default_negatives * np.dtype(np.int64).itemsize
    return {
        "input_float32_bytes": input_bytes,
        "knn_indices_int64_bytes": knn_indices,
        "knn_distances_float32_bytes": knn_distances,
        "knn_graph_bytes": knn_indices + knn_distances,
        "layout_float32_bytes": layout,
        "configured_negative_indices_int64_bytes": configured_negative,
        "default_negative_indices_int64_bytes": default_negative,
        "configured_negative_samples_per_row": configured_negatives,
        "default_negative_samples_per_row": default_negatives,
        "configured_named_tensor_total_bytes": (
            input_bytes + knn_indices + knn_distances + layout + configured_negative
        ),
    }


def _umap_named_memory(
    rows: int,
    features: int,
    target_dimension: int,
    neighbors: int,
    args: argparse.Namespace,
) -> dict[str, int]:
    build = umap_build_kwds(args)
    clusters = int(build["knn_n_clusters"])
    overlap = int(build["knn_overlap_factor"])
    resident_fraction = 1.0 if clusters == 1 else min(1.0, overlap / clusters)
    partition_rows = int(math.ceil(rows * resident_fraction))
    graph_degree = int(build["nnd_intermediate_graph_degree"])
    return {
        "host_mapped_input_bytes": rows * features * np.dtype(np.float32).itemsize,
        "estimated_partition_rows": partition_rows,
        "partition_input_float32_bytes": (
            partition_rows * features * np.dtype(np.float32).itemsize
        ),
        "intermediate_graph_indices_int64_bytes": (
            rows * graph_degree * np.dtype(np.int64).itemsize
        ),
        "knn_distances_float32_bytes": rows * neighbors * np.dtype(np.float32).itemsize,
        "layout_float32_bytes": rows * target_dimension * np.dtype(np.float32).itemsize,
    }


def _host_buffer_advisory(args: argparse.Namespace, rows: int, features: int) -> dict[str, Any]:
    validation_rows = min(rows, args.validation_block_rows)
    write_rows = min(rows, args.layout_write_block_rows)
    target = args.target_dimension
    components = {
        "backend_layout_float32": rows * target * 4,
        "validation_float64_square_block": validation_rows * target * 8,
        "validation_finiteness_mask": validation_rows * target,
        "write_contiguous_float32_block": write_rows * target * 4,
        "write_digest_bytes_copy": write_rows * target * 4,
    }
    return {
        "advisory_only": True,
        "components_bytes": {key: int(value) for key, value in components.items()},
        "bounded_validation_rows": int(validation_rows),
        "bounded_write_rows": int(write_rows),
        "estimated_explicit_peak_bytes": int(sum(components.values())),
        "source_memmap_virtual_bytes": int(rows * features * 4),
        "backend_internal_host_allocations_excluded": True,
        "file_backed_page_cache_rss_excluded": True,
    }


def gpu_memory_info(device_id: int = 0) -> dict[str, int] | None:
    """Return free/total device bytes without making availability a requirement."""
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        if device_id >= cp.cuda.runtime.getDeviceCount():
            return None
        free, total = cp.cuda.Device(device_id).mem_info
        return {
            "device_id": int(device_id),
            "free_bytes": int(free),
            "total_bytes": int(total),
        }
    except Exception:
        return None


def memory_preflight(
    args: argparse.Namespace,
    *,
    rows: int,
    features: int,
    gpu_memory: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Describe named allocations and warn without pretending a fit was attempted."""
    if args.method == "dire":
        named = _dire_named_memory(
            rows,
            features,
            args.target_dimension,
            args.neighbors,
            neg_ratio=args.dire_neg_ratio,
        )
        named_total = named["configured_named_tensor_total_bytes"]
        scope = (
            "Current cuVS-backed DiRe vectorized-force path: input, host/device kNN "
            "payload, layout, and the full configured negative-index tensor"
        )
    else:
        named = _umap_named_memory(rows, features, args.target_dimension, args.neighbors, args)
        named_total = sum(
            value
            for key, value in named.items()
            if key.endswith("_bytes") and key != "host_mapped_input_bytes"
        )
        scope = (
            "UMAP named tensor approximation using the configured partition fraction; "
            "the backend controls actual graph/index workspace"
        )
    budget = int(math.ceil(named_total * args.preflight_safety_factor))
    device_id = selected_device_id(args)
    device = gpu_memory if gpu_memory is not None else gpu_memory_info(device_id)
    threshold = None
    warning = False
    messages: list[str] = []
    if device is None:
        messages.append("GPU free memory was unavailable; no capacity comparison was made")
    else:
        threshold = int(device["free_bytes"] * args.preflight_vram_fraction)
        warning = budget > threshold
        if warning:
            messages.append(
                "estimated budget exceeds the configured fraction of currently free VRAM"
            )
    if args.method == "dire":
        messages.append(
            "The estimate excludes cuVS index workspace and transient force tensors; "
            "the full negative-index allocation is nevertheless included explicitly"
        )
    else:
        messages.append(
            "RAPIDS may stream partitions but still owns graph and optimizer allocations"
        )
    return {
        "method": args.method,
        "scope": scope,
        "named_allocations": named,
        "named_tensor_bytes": int(named_total),
        "safety_factor": args.preflight_safety_factor,
        "estimated_budget_bytes": budget,
        "gpu_memory_at_preflight": device,
        "monitored_device_id": device_id,
        "vram_fraction": args.preflight_vram_fraction,
        "warning_threshold_bytes": threshold,
        "warning": warning,
        "messages": messages,
        "fit_blocked": False,
        "fit_blocked_reason": None,
        "device_advisory": {
            "advisory_only": True,
            "named_tensor_bytes": int(named_total),
            "estimated_budget_bytes": budget,
            "gpu_memory": device,
            "warning": warning,
        },
        "host_advisory": _host_buffer_advisory(args, rows, features),
        "disk_requirement": None,
    }


def _attach_disk_preflight(
    preflight: dict[str, Any],
    args: argparse.Namespace,
    layout_path: str | Path,
) -> None:
    reserve_bytes = int(args.disk_reserve_gib * 2**30)
    try:
        record = _layout_disk_preflight(
            layout_path,
            args.rows,
            args.target_dimension,
            reserve_bytes,
        )
    except OSError as exc:
        preflight["disk_requirement"] = {
            "enforced": True,
            "status": "failed",
            "errno": exc.errno,
            "message": str(exc),
        }
        raise
    preflight["disk_requirement"] = {
        "enforced": True,
        "status": "ok",
        **record,
    }


class PeakMonitor:
    """Best-effort process RSS and NVML GPU-memory sampler."""

    def __init__(self, interval: float = 0.1, device_id: int = 0):
        self.interval = interval
        self.device_id = device_id
        self.peak_rss_bytes = 0
        self.peak_gpu_bytes: int | None = None
        self.gpu_baseline_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._nvml_handles = []
        self._nvml_getter = None

    def _initialize_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self._nvml_getter = getattr(
                pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None
            ) or getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None)
        except Exception:
            self._pynvml = None
            self._nvml_handles = []
            self._nvml_getter = None

    def _gpu_process_bytes(self) -> int | None:
        if self._nvml_getter is None:
            return None
        observed = 0
        queried = False
        try:
            for handle in self._nvml_handles:
                for process in self._nvml_getter(handle):
                    queried = True
                    if process.pid == os.getpid():
                        value = int(process.usedGpuMemory)
                        if value >= 0:
                            observed += value
        except Exception:
            return None
        return observed if queried or self._nvml_handles else None

    def _sample(self) -> None:
        try:
            import psutil

            process = psutil.Process()
            while not self._stop.is_set():
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(process.memory_info().rss))
                gpu = self._gpu_process_bytes()
                if gpu is not None:
                    self.peak_gpu_bytes = (
                        gpu if self.peak_gpu_bytes is None else max(self.peak_gpu_bytes, gpu)
                    )
                self._stop.wait(self.interval)
        except Exception:
            return

    def __enter__(self):
        self._initialize_nvml()
        self.gpu_baseline_bytes = self._gpu_process_bytes()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()

    def result(self) -> dict[str, int | None]:
        rss_scale = 1 if sys.platform == "darwin" else 1024
        fallback_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale)
        peak = max(self.peak_rss_bytes, fallback_rss)
        incremental = None
        if self.peak_gpu_bytes is not None and self.gpu_baseline_bytes is not None:
            incremental = max(0, self.peak_gpu_bytes - self.gpu_baseline_bytes)
        return {
            "peak_rss_bytes": peak,
            "gpu_baseline_bytes": self.gpu_baseline_bytes,
            "peak_gpu_process_bytes": self.peak_gpu_bytes,
            "peak_incremental_gpu_bytes": incremental,
            "sampling_interval_seconds": self.interval,
            "monitored_device_id": self.device_id,
            "gpu_process_scope": (
                "sum across all NVML devices for this PID; multi-GPU fits are rejected"
            ),
        }


def _synchronize_cuda(cupy=None, device_id: int = 0) -> None:
    if cupy is not None:
        if device_id == 0:
            cupy.cuda.runtime.deviceSynchronize()
        else:
            with cupy.cuda.Device(device_id):
                cupy.cuda.runtime.deviceSynchronize()
        return
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def fit_umap(
    X: np.memmap,
    args: argparse.Namespace,
    *,
    umap_class=None,
    cupy=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit partitioned RAPIDS UMAP directly from the read-only host memmap."""
    import_started = time.perf_counter()
    if umap_class is None or cupy is None:
        import cupy as cp
        from cuml.manifold import UMAP

        umap_class, cupy = UMAP, cp
    import_seconds = time.perf_counter() - import_started
    build_kwds = umap_build_kwds(args)
    reducer = umap_class(
        n_components=args.target_dimension,
        n_neighbors=args.neighbors,
        min_dist=args.umap_min_dist,
        spread=args.umap_spread,
        metric="euclidean",
        init=args.init,
        n_epochs=args.umap_epochs,
        negative_sample_rate=args.umap_negative_sample_rate,
        random_state=args.seed,
        build_algo="nn_descent",
        build_kwds=build_kwds,
        device_ids=args.umap_device_ids,
        force_serial_epochs=args.umap_force_serial_epochs,
        hash_input=False,
        output_type="numpy",
        verbose=args.verbose,
    )
    device_id = selected_device_id(args)
    _synchronize_cuda(cupy, device_id)
    started = time.perf_counter()
    layout = reducer.fit_transform(X)
    _synchronize_cuda(cupy, device_id)
    fit_seconds = time.perf_counter() - started
    return np.asarray(layout), {
        "backend": f"{type(reducer).__module__}.{type(reducer).__name__}",
        "backend_import_seconds": import_seconds,
        "fit_seconds": fit_seconds,
        "build_algo": "nn_descent",
        "build_kwds": build_kwds,
        "device_ids": args.umap_device_ids,
        "init": args.init,
        "metric": "euclidean",
        "min_dist": args.umap_min_dist,
        "spread": args.umap_spread,
        "n_epochs": args.umap_epochs,
        "negative_sample_rate": args.umap_negative_sample_rate,
        "force_serial_epochs": args.umap_force_serial_epochs,
        "nn_descent_exactly_deterministic": False,
        "determinism_note": (
            "RAPIDS documents NN-descent graph construction as non-deterministic even "
            "when random_state is set"
        ),
        "hash_input": False,
        "output_type": "numpy",
        "host_input_type": type(X).__name__,
        "host_input_memmap": isinstance(X, np.memmap),
        "host_input_writeable": bool(X.flags.writeable),
        "harness_full_host_normalization_copy": False,
    }


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit_dire(
    X: np.memmap,
    args: argparse.Namespace,
    *,
    create_dire=None,
    cupy=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit DiRe with the requested cuVS index and without input normalization."""
    repository = Path(args.dire_repo).expanduser().resolve() if args.dire_repo is not None else None
    if create_dire is None:
        import cupy as cp

        dire_rapids, identity = load_dire(repository)
        create_dire = dire_rapids.create_dire
        cupy = cp
    else:
        dire_rapids = sys.modules.get("dire_rapids")
        identity = {
            "source_checkout": str(repository) if repository is not None else None,
            "module_path": None,
            "version": getattr(dire_rapids, "__version__", None),
            "git_revision": git_revision(repository),
        }
    reducer = create_dire(
        backend="cuvs",
        n_components=args.target_dimension,
        n_neighbors=args.neighbors,
        init=args.init,
        max_iter_layout=args.dire_iterations,
        neg_ratio=args.dire_neg_ratio,
        metric=None,
        normalize=False,
        cuvs_index_type=args.dire_cuvs_index,
        random_state=args.seed,
        verbose=args.verbose,
    )
    resolved_index = (
        reducer._select_cuvs_index_type(X.shape[0], X.shape[1], "sqeuclidean")
        if hasattr(reducer, "_select_cuvs_index_type")
        else None
    )
    _synchronize_cuda(cupy)
    started = time.perf_counter()
    layout = reducer.fit_transform(X)
    _synchronize_cuda(cupy)
    fit_seconds = time.perf_counter() - started
    return np.asarray(layout), {
        "backend": f"{type(reducer).__module__}.{type(reducer).__name__}",
        "fit_seconds": fit_seconds,
        "dire_repo": identity["source_checkout"],
        "dire_revision": identity["git_revision"],
        "dire_version": identity["version"],
        "dire_module": identity["module_path"],
        "implementation_sha256": implementation_identity(create_dire)["sha256"],
        "knn_backend": "cuvs",
        "cuvs_index_requested": args.dire_cuvs_index,
        "cuvs_index_resolved": resolved_index,
        "iterations": args.dire_iterations,
        "negative_sample_ratio": args.dire_neg_ratio,
        "default_negative_sample_ratio": DEFAULT_DIRE_NEG_RATIO,
        "init": args.init,
        "metric": "sqeuclidean",
        "normalize": False,
        "host_input_type": type(X).__name__,
        "host_input_memmap": isinstance(X, np.memmap),
        "host_input_writeable": bool(X.flags.writeable),
        "harness_full_host_normalization_copy": False,
    }


def validate_and_summarize_layout(
    layout: np.ndarray,
    *,
    rows: int,
    target_dimension: int,
    block_rows: int,
) -> dict[str, Any]:
    """Validate every output coordinate and compute bounded-memory summaries."""
    if not isinstance(layout, np.ndarray):
        raise IntegrityFailure("backend output is not a NumPy array")
    if layout.shape != (rows, target_dimension):
        raise IntegrityFailure(
            f"layout shape {layout.shape} does not equal {(rows, target_dimension)}"
        )
    if layout.dtype != np.dtype(np.float32):
        raise IntegrityFailure(f"layout dtype {layout.dtype} is not float32")
    sums = np.zeros(target_dimension, dtype=np.float64)
    sums_squared = np.zeros(target_dimension, dtype=np.float64)
    minima = np.full(target_dimension, np.inf, dtype=np.float64)
    maxima = np.full(target_dimension, -np.inf, dtype=np.float64)
    try:
        for start in range(0, rows, block_rows):
            stop = min(start + block_rows, rows)
            block = np.asarray(layout[start:stop])
            if not np.all(np.isfinite(block)):
                raise IntegrityFailure("layout contains non-finite coordinates")
            sums += np.sum(block, axis=0, dtype=np.float64)
            sums_squared += np.sum(np.square(block, dtype=np.float64), axis=0, dtype=np.float64)
            minima = np.minimum(minima, np.min(block, axis=0))
            maxima = np.maximum(maxima, np.max(block, axis=0))
    except IntegrityFailure:
        raise
    except Exception as exc:
        raise MetricFailure(f"layout summary failed: {type(exc).__name__}: {exc}") from exc
    means = sums / rows
    variances = np.maximum(0.0, sums_squared / rows - means**2)
    return {
        "coordinate_mean": means.tolist(),
        "coordinate_std": np.sqrt(variances).tolist(),
        "coordinate_min": minima.tolist(),
        "coordinate_max": maxima.tolist(),
        "finite_coordinates": int(rows * target_dimension),
        "validation_block_rows": block_rows,
    }


def _memory_status(exc: BaseException) -> str | None:
    module = type(exc).__module__.lower()
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    memory_words = any(
        token in message
        for token in ("out of memory", "out_of_memory", "memoryallocation", "bad_alloc")
    )
    cuda_origin = any(
        token in module or token in name or token in message
        for token in ("cuda", "cupy", "torch", "rmm", "device memory")
    )
    if (cuda_origin and memory_words) or "cuda out of memory" in message:
        return "cuda_oom"
    if isinstance(exc, MemoryError):
        return "host_oom_or_sigkill"
    if (
        getattr(exc, "errno", None) == 12
        or "cannot allocate memory" in message
        or "std::bad_alloc" in message
    ):
        return "host_oom_or_sigkill"
    return None


def classify_exception(exc: BaseException, *, phase: str) -> str:
    """Map worker exceptions onto the deliberately closed status vocabulary."""
    memory_status = _memory_status(exc)
    if memory_status is not None:
        return memory_status
    message = str(exc).lower()
    if any(
        marker in message
        for marker in (
            "cudaerrornodevice",
            "no cuda-capable device",
            "requires cuda gpu",
            "cuda driver version is insufficient",
            "cuda driver initialization failed",
        )
    ):
        return "unsupported_device"
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "integrity_failed"
    if isinstance(exc, MetricFailure) or phase == "metric":
        return "metric_failed"
    if isinstance(exc, IntegrityFailure) or phase in {
        "input",
        "layout_validation",
        "layout_write",
    }:
        return "integrity_failed"
    if isinstance(exc, (ImportError, ModuleNotFoundError, NotImplementedError, OverflowError)):
        return "unsupported_scale"
    return "unsupported_scale"


def _failure_cause(exc: BaseException | str, status: str, phase: str) -> str:
    if isinstance(exc, BaseException):
        if _memory_status(exc) == "cuda_oom":
            return "cuda_memory_exhaustion"
        if _memory_status(exc) == "host_oom_or_sigkill":
            return "host_memory_exhaustion"
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            return "insufficient_disk"
        if isinstance(exc, MetricFailure):
            return "metric_computation"
        if isinstance(exc, IntegrityFailure):
            return "integrity_contract"
    return {
        "timeout": "wall_clock_timeout",
        "host_oom_or_sigkill": "host_oom_or_sigkill",
        "unsupported_device": "accelerator_unavailable",
        "unsupported_scale": "backend_or_scale_unsupported",
        "integrity_failed": f"integrity_or_io_{phase}",
        "metric_failed": "metric_computation",
    }.get(status, "unknown")


def _exception_chain(exc: BaseException) -> list[dict[str, str]]:
    chain = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return chain


def _hardware(device_id: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > device_id:
            properties = cp.cuda.runtime.getDeviceProperties(device_id)
            name = properties.get("name", properties.get(b"name", "unknown"))
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            free, total = cp.cuda.Device(device_id).mem_info
            result.update(
                {
                    "gpu_device_id": int(device_id),
                    "gpu_name": str(name),
                    "gpu_free_memory_bytes_at_report": int(free),
                    "gpu_total_memory_bytes": int(total),
                    "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
                    "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                }
            )
    except Exception:
        pass
    return result


def _versions() -> dict[str, str]:
    versions = {"numpy": np.__version__}
    for name in ("cupy", "cuml", "cuvs", "torch", "dire_rapids"):
        module = sys.modules.get(name)
        if module is not None:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _input_payload(
    info, rows: int, *, source_snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    stat = Path(info.path).stat()
    return {
        **asdict(info),
        "path": str(Path(info.path).resolve()),
        "selected_rows": rows,
        "selected_payload_bytes": int(rows * info.dimension * np.dtype("<f4").itemsize),
        "file_size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "selection": "exact leading prefix",
        "mapping_mode": "read-only host memmap",
        "complete_host_copy_by_harness": False,
        "normalization_by_harness": False,
        "payload_values_prescanned": False,
        "source_snapshot_before": source_snapshot,
        "source_mutation_check": "stat identity and header digest before and after fit",
    }


def configuration_payload(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "method": args.method,
        "rows": args.rows,
        "target_dimension": args.target_dimension,
        "seed": args.seed,
        "neighbors": args.neighbors,
        "init": args.init,
    }
    if args.method == "dire":
        common["dire"] = {
            "repo": (
                str(Path(args.dire_repo).expanduser().resolve())
                if args.dire_repo is not None
                else None
            ),
            "cuvs_index": args.dire_cuvs_index,
            "iterations": args.dire_iterations,
            "negative_sample_ratio": args.dire_neg_ratio,
            "normalize": False,
        }
    else:
        common["umap"] = {
            "build_algo": "nn_descent",
            "build_kwds": umap_build_kwds(args),
            "device_ids": args.umap_device_ids,
            "min_dist": args.umap_min_dist,
            "spread": args.umap_spread,
            "n_epochs": args.umap_epochs,
            "negative_sample_rate": args.umap_negative_sample_rate,
            "force_serial_epochs": args.umap_force_serial_epochs,
            "hash_input": False,
            "random_state": args.seed,
            "nn_descent_exactly_deterministic": False,
            "determinism_note": (
                "random_state controls supported stochastic stages, but RAPIDS "
                "NN-descent graph construction is not exactly deterministic"
            ),
        }
    return common


def _preflight_disposition(preflight: dict[str, Any] | None, attempted_fit: bool) -> str:
    if preflight is None:
        return "not_completed"
    if preflight["warning"]:
        return "warning_then_fit_attempted" if attempted_fit else "warning_without_fit_attempt"
    return "clear_then_fit_attempted" if attempted_fit else "clear_without_fit_attempt"


def _failure_payload(
    args: argparse.Namespace,
    *,
    status: str,
    error: BaseException | str,
    phase: str,
    preflight: dict[str, Any] | None,
    input_payload: dict[str, Any] | None,
    attempted_fit: bool,
    started_at: str,
) -> dict[str, Any]:
    if status not in STATUSES or status == "ok":
        raise ValueError(f"invalid failure status {status!r}")
    if isinstance(error, BaseException):
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": phase,
            "cause": _failure_cause(error, status, phase),
            "exception_chain": _exception_chain(error),
        }
    else:
        error_payload = {
            "type": None,
            "message": error,
            "phase": phase,
            "cause": _failure_cause(error, status, phase),
            "exception_chain": [],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": getattr(args, "run_id", None),
        "status": status,
        "error": error_payload,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "method": args.method,
        "rows_requested": args.rows,
        "rows": input_payload["selected_rows"] if input_payload else None,
        "features": input_payload["dimension"] if input_payload else None,
        "target_dimension": args.target_dimension,
        "seed": args.seed,
        "configuration": configuration_payload(args),
        "input": input_payload,
        "preflight": preflight,
        "preflight_advisory": bool(preflight and preflight["warning"]),
        "preflight_disposition": _preflight_disposition(preflight, attempted_fit),
        "attempted_fit": attempted_fit,
        "observed_attempt_failure": attempted_fit,
        "layout": None,
        "layout_summary": None,
        "timing": {},
        "resources": {},
        "hardware": _hardware(selected_device_id(args)),
        "versions": _versions(),
    }


def run_worker(args: argparse.Namespace) -> int:
    """Execute a single fit and write a private worker result atomically."""
    started_at = _utc_now()
    worker_started = time.perf_counter()
    result_path = _absolute_path(args.worker_result)
    heartbeat_path = _absolute_path(args.worker_heartbeat)
    layout_path = _absolute_path(args.layout_path)
    phase = "input"
    heartbeat_phase = "worker_started"
    preflight = None
    input_payload = None
    attempted_fit = False
    monitor = PeakMonitor(args.resource_sample_interval, device_id=selected_device_id(args))
    try:
        _validate_publication_paths(args)
        source_path = Path(args.data).expanduser().resolve()
        private_paths = {result_path, heartbeat_path, layout_path}
        public_control_paths = {
            _absolute_path(args.result_path),
            _absolute_path(args.log_path),
        }
        if (
            len(private_paths) != 3
            or source_path in private_paths
            or {result_path, heartbeat_path} & public_control_paths
        ):
            raise IntegrityFailure(
                "source, worker result, heartbeat, and layout paths must be distinct"
            )
        if layout_path.suffix.lower() != ".npy":
            raise IntegrityFailure("worker layout path must end in .npy")
        _write_heartbeat(args, heartbeat_phase, attempted_fit=False)
        inspect_started = time.perf_counter()
        heartbeat_phase = "input_validation"
        _write_heartbeat(args, heartbeat_phase, attempted_fit=False)
        source_snapshot_before = _source_snapshot(source_path)
        info = read_fbin_header(args.data)
        _assert_source_unchanged(source_path, source_snapshot_before)
        if args.rows > info.rows:
            raise IntegrityFailure(
                f"requested {args.rows} rows but the fbin contains only {info.rows}"
            )
        input_payload = _input_payload(info, args.rows, source_snapshot=source_snapshot_before)
        input_payload["source_manifest"] = _load_source_manifest(source_path, info)
        preflight = memory_preflight(args, rows=args.rows, features=info.dimension)
        phase = "preflight"
        _attach_disk_preflight(preflight, args, layout_path)
        X = open_fbin_memmap(args.data, limit_rows=args.rows)
        if X.shape != (args.rows, info.dimension):
            raise IntegrityFailure("fbin memmap returned the wrong shape")
        if X.dtype != np.dtype("<f4"):
            raise IntegrityFailure(f"fbin memmap dtype is {X.dtype}, expected little-endian f4")
        if X.flags.writeable:
            raise IntegrityFailure("fbin input memmap is unexpectedly writeable")
        inspect_seconds = time.perf_counter() - inspect_started
        heartbeat_phase = "preflight_complete"
        _write_heartbeat(args, heartbeat_phase, attempted_fit=False)

        with monitor:
            phase = "fit"
            attempted_fit = True
            heartbeat_phase = "fit_started"
            _write_heartbeat(args, heartbeat_phase, attempted_fit=True)
            if args.method == "dire":
                layout, details = fit_dire(X, args)
            else:
                layout, details = fit_umap(X, args)
            heartbeat_phase = "fit_returned"
            _write_heartbeat(args, heartbeat_phase, attempted_fit=True)
            _assert_source_unchanged(source_path, source_snapshot_before)

            phase = "layout_validation"
            validation_started = time.perf_counter()
            layout_summary = validate_and_summarize_layout(
                layout,
                rows=args.rows,
                target_dimension=args.target_dimension,
                block_rows=args.validation_block_rows,
            )
            validation_seconds = time.perf_counter() - validation_started
            heartbeat_phase = "layout_validated"
            _write_heartbeat(args, heartbeat_phase, attempted_fit=True)
            _assert_source_unchanged(source_path, source_snapshot_before)

            phase = "layout_write"
            write_started = time.perf_counter()
            layout_payload = _save_layout_atomic(
                layout_path, layout, block_rows=args.layout_write_block_rows
            )
            write_seconds = time.perf_counter() - write_started
            source_snapshot_after = _assert_source_unchanged(source_path, source_snapshot_before)
            heartbeat_phase = "layout_published"
            _write_heartbeat(args, heartbeat_phase, attempted_fit=True)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "status": "ok",
            "error": None,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "method": args.method,
            "rows_requested": args.rows,
            "rows": args.rows,
            "features": info.dimension,
            "target_dimension": args.target_dimension,
            "seed": args.seed,
            "configuration": configuration_payload(args),
            "input": input_payload,
            "source_snapshot_after": source_snapshot_after,
            "source_unchanged": True,
            "preflight": preflight,
            "preflight_advisory": bool(preflight["warning"]),
            "preflight_disposition": _preflight_disposition(preflight, True),
            "attempted_fit": True,
            "observed_attempt_failure": False,
            "layout": layout_payload,
            "layout_summary": layout_summary,
            "details": details,
            "timing": {
                "input_inspection_and_open_seconds": inspect_seconds,
                "backend_import_seconds": details.get("backend_import_seconds"),
                "fit_seconds": details["fit_seconds"],
                "layout_validation_seconds": validation_seconds,
                "layout_write_seconds": write_seconds,
                "worker_total_seconds": time.perf_counter() - worker_started,
            },
            "resources": monitor.result(),
            "hardware": _hardware(selected_device_id(args)),
            "versions": _versions(),
        }
        _atomic_json(result_path, payload)
        return 0
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        status = classify_exception(exc, phase=phase)
        with contextlib.suppress(Exception):
            _write_heartbeat(
                args,
                "worker_failed",
                attempted_fit=attempted_fit,
                last_phase=heartbeat_phase,
            )
        payload = _failure_payload(
            args,
            status=status,
            error=exc,
            phase=phase,
            preflight=preflight,
            input_payload=input_payload,
            attempted_fit=attempted_fit,
            started_at=started_at,
        )
        payload["timing"] = {"worker_total_seconds": time.perf_counter() - worker_started}
        payload["resources"] = monitor.result()
        payload["worker_last_phase"] = heartbeat_phase
        _atomic_json(result_path, payload)
        return 1


def _worker_command(
    args: argparse.Namespace, worker_result: Path, worker_heartbeat: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--data",
        str(Path(args.data).expanduser().resolve()),
        "--method",
        args.method,
        "--rows",
        str(args.rows),
        "--target-dimension",
        str(args.target_dimension),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--layout-path",
        str(_absolute_path(args.layout_path)),
        "--worker-result",
        str(worker_result),
        "--worker-heartbeat",
        str(worker_heartbeat),
        "--run-id",
        args.run_id,
        "--seed",
        str(args.seed),
        "--neighbors",
        str(args.neighbors),
        "--init",
        args.init,
        "--dire-cuvs-index",
        args.dire_cuvs_index,
        "--dire-iterations",
        str(args.dire_iterations),
        "--dire-neg-ratio",
        str(args.dire_neg_ratio),
        "--umap-knn-n-clusters",
        str(args.umap_knn_n_clusters),
        "--umap-knn-overlap-factor",
        str(args.umap_knn_overlap_factor),
        "--umap-nnd-graph-degree",
        str(args.umap_nnd_graph_degree),
        "--umap-nnd-intermediate-graph-degree",
        str(args.umap_nnd_intermediate_graph_degree),
        "--umap-nnd-max-iterations",
        str(args.umap_nnd_max_iterations),
        "--umap-nnd-termination-threshold",
        str(args.umap_nnd_termination_threshold),
        "--umap-device-ids",
        (
            args.umap_device_ids
            if isinstance(args.umap_device_ids, str)
            else (
                "none" if args.umap_device_ids is None else ",".join(map(str, args.umap_device_ids))
            )
        ),
        "--umap-min-dist",
        str(args.umap_min_dist),
        "--umap-spread",
        str(args.umap_spread),
        "--umap-negative-sample-rate",
        str(args.umap_negative_sample_rate),
        "--preflight-safety-factor",
        str(args.preflight_safety_factor),
        "--preflight-vram-fraction",
        str(args.preflight_vram_fraction),
        "--disk-reserve-gib",
        str(args.disk_reserve_gib),
        "--validation-block-rows",
        str(args.validation_block_rows),
        "--layout-write-block-rows",
        str(args.layout_write_block_rows),
        "--resource-sample-interval",
        str(args.resource_sample_interval),
    ]
    if args.dire_repo is not None:
        command.extend(("--dire-repo", str(args.dire_repo)))
    if args.umap_epochs is not None:
        command.extend(("--umap-epochs", str(args.umap_epochs)))
    if args.umap_force_serial_epochs is not None:
        command.append(
            "--umap-force-serial-epochs"
            if args.umap_force_serial_epochs
            else "--no-umap-force-serial-epochs"
        )
    if args.verbose:
        command.append("--verbose")
    return command


def _terminate_process_group(
    process: subprocess.Popen, process_group_id: int | None = None
) -> int | None:
    if process_group_id is None:
        with contextlib.suppress(ProcessLookupError):
            process_group_id = os.getpgid(process.pid)
    if process_group_id is None or process_group_id == os.getpgrp():
        # Never signal the harness's own process group.
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process_group_id is None or process_group_id == os.getpgrp():
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
        process.wait()
    return (
        -process.returncode if process.returncode is not None and process.returncode < 0 else None
    )


def _run_child(command: list[str], log_path: Path, timeout: int) -> ChildOutcome:
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                process_group_id = os.getpgid(process.pid)
            except ProcessLookupError:
                process_group_id = process.pid
            try:
                returncode = process.wait(timeout=None if timeout <= 0 else timeout)
                termination_signal = -returncode if returncode < 0 else None
                return ChildOutcome(
                    returncode=returncode,
                    wall_seconds=time.perf_counter() - started,
                    timed_out=False,
                    termination_signal=termination_signal,
                )
            except subprocess.TimeoutExpired:
                termination_signal = _terminate_process_group(process, process_group_id)
                log.write(f"\nworker timed out after {timeout} seconds\n")
                log.flush()
                return ChildOutcome(
                    returncode=process.returncode,
                    wall_seconds=time.perf_counter() - started,
                    timed_out=True,
                    termination_signal=termination_signal,
                )
            except BaseException:
                _terminate_process_group(process, process_group_id)
                raise
    except OSError as exc:
        return ChildOutcome(
            returncode=None,
            wall_seconds=time.perf_counter() - started,
            timed_out=False,
            launch_error=f"{type(exc).__name__}: {exc}",
        )


def _abrupt_status(outcome: ChildOutcome) -> str:
    if outcome.timed_out:
        return "timeout"
    if outcome.termination_signal == signal.SIGKILL or outcome.returncode == 128 + signal.SIGKILL:
        return "host_oom_or_sigkill"
    return "unsupported_scale"


def _validate_worker_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityFailure("worker result has the wrong schema version")
    if payload.get("status") not in STATUSES:
        raise IntegrityFailure("worker result contains an unknown status")
    expected = {
        "run_id": args.run_id,
        "method": args.method,
        "rows_requested": args.rows,
        "target_dimension": args.target_dimension,
        "seed": args.seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise IntegrityFailure(f"worker result {key!r} does not match the invocation")
    if payload["status"] == "ok":
        layout = payload.get("layout") or {}
        if layout.get("rows") != args.rows:
            raise IntegrityFailure("successful worker layout has the wrong row count")
        if layout.get("target_dimension") != args.target_dimension:
            raise IntegrityFailure("successful worker layout has the wrong dimension")
        if _absolute_path(layout.get("path", "")) != _absolute_path(args.layout_path):
            raise IntegrityFailure("successful worker reported an unexpected layout path")
        try:
            layout_path = Path(layout["path"])
            reopened = np.load(layout_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, KeyError) as exc:
            raise IntegrityFailure(f"successful worker layout cannot be opened: {exc}") from exc
        if reopened.shape != (args.rows, args.target_dimension):
            raise IntegrityFailure("published worker layout has the wrong shape")
        if reopened.dtype != np.dtype(np.float32):
            raise IntegrityFailure("published worker layout has the wrong dtype")
        if layout_path.stat().st_size != layout.get("file_bytes"):
            raise IntegrityFailure("published worker layout file size changed")


def _run_isolated_impl(args: argparse.Namespace) -> int:
    """Launch one worker, classify abrupt termination, and publish one result."""
    _validate_publication_paths(args)
    source_path = Path(args.data).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = _absolute_path(args.result_path)
    layout_pointer = _absolute_path(args.layout_path)
    log_path = _absolute_path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_preexisted = result_path.exists()
    layout_preexisted = os.path.lexists(layout_pointer)
    if not args.force and (result_preexisted or layout_preexisted):
        raise FileExistsError("result or layout already exists; pass --force to replace it")
    if result_preexisted and result_path.stat().st_size > 16 * 1024 * 1024:
        raise IntegrityFailure("refusing to replace an unexpectedly large result JSON")

    run_id = uuid.uuid4().hex
    args.run_id = run_id
    generation_path = _generation_layout_path(layout_pointer, run_id)
    args._active_generation_path = str(generation_path)
    args._generation_committed = False
    if os.path.lexists(generation_path):
        raise IntegrityFailure(f"immutable generation already exists: {generation_path}")
    worker_args = argparse.Namespace(**vars(args))
    worker_args.layout_path = str(generation_path)

    started_at = _utc_now()
    input_payload = None
    preflight = None
    source_snapshot_before = None
    try:
        source_snapshot_before = _source_snapshot(source_path)
        info = read_fbin_header(args.data)
        _assert_source_unchanged(source_path, source_snapshot_before)
        if args.rows > info.rows:
            raise IntegrityFailure(
                f"requested {args.rows} rows but the fbin contains only {info.rows}"
            )
        input_payload = _input_payload(info, args.rows, source_snapshot=source_snapshot_before)
        input_payload["source_manifest"] = _load_source_manifest(source_path, info)
        preflight = memory_preflight(args, rows=args.rows, features=info.dimension)
        _attach_disk_preflight(preflight, args, generation_path)
    except BaseException as exc:
        status = classify_exception(exc, phase="input")
        payload = _failure_payload(
            args,
            status=status,
            error=exc,
            phase="parent_input_preflight",
            preflight=preflight,
            input_payload=input_payload,
            attempted_fit=False,
            started_at=started_at,
        )
        payload["isolation"] = {
            "worker_launched": False,
            "timeout_seconds": args.timeout,
        }
        failure_path = (
            result_path.with_name(f"{result_path.stem}.{run_id}.failed{result_path.suffix}")
            if result_preexisted
            else result_path
        )
        _atomic_json(failure_path, payload)
        return 1

    worker_result = output / f".worker-result-{run_id}.json"
    worker_heartbeat = output / f".worker-heartbeat-{run_id}.json"
    private_paths = {worker_result.resolve(), worker_heartbeat.resolve()}
    public_paths = {source_path, result_path, layout_pointer, log_path, generation_path}
    if private_paths & public_paths or len(private_paths) != 2:
        raise IntegrityFailure("worker control paths collide with experiment paths")
    command = _worker_command(worker_args, worker_result, worker_heartbeat)
    try:
        outcome = _run_child(command, log_path, args.timeout)
    except BaseException:
        _quarantine_generation(generation_path, run_id)
        raise
    heartbeat = _read_heartbeat(worker_heartbeat, worker_args)
    attempted_from_heartbeat = bool(heartbeat and heartbeat.get("attempted_fit"))
    payload = None
    if worker_result.exists():
        try:
            payload = json.loads(worker_result.read_text())
            _validate_worker_payload(payload, worker_args)
            if (payload["status"] == "ok") != (outcome.returncode == 0):
                raise IntegrityFailure(
                    "worker process return code and result status are inconsistent"
                )
            if payload["status"] == "ok" and (
                heartbeat is None
                or heartbeat.get("last_phase") is not None
                or heartbeat.get("phase") != "layout_published"
            ):
                raise IntegrityFailure(
                    "successful worker did not leave a layout-published heartbeat"
                )
        except (OSError, ValueError, IntegrityFailure) as exc:
            payload = _failure_payload(
                worker_args,
                status="integrity_failed",
                error=exc,
                phase="worker_result_validation",
                preflight=preflight,
                input_payload=input_payload,
                attempted_fit=attempted_from_heartbeat,
                started_at=started_at,
            )
    if outcome.timed_out:
        payload = _failure_payload(
            worker_args,
            status="timeout",
            error=f"worker exceeded {args.timeout} seconds",
            phase=heartbeat.get("last_phase") or heartbeat.get("phase") if heartbeat else "worker",
            preflight=preflight,
            input_payload=input_payload,
            attempted_fit=attempted_from_heartbeat,
            started_at=started_at,
        )
    elif payload is None:
        payload = _failure_payload(
            worker_args,
            status=_abrupt_status(outcome),
            error=outcome.launch_error or "worker exited without a valid result",
            phase=heartbeat.get("last_phase") or heartbeat.get("phase") if heartbeat else "worker",
            preflight=preflight,
            input_payload=input_payload,
            attempted_fit=attempted_from_heartbeat,
            started_at=started_at,
        )

    try:
        source_snapshot_after = _assert_source_unchanged(source_path, source_snapshot_before)
    except BaseException as exc:
        payload = _failure_payload(
            worker_args,
            status="integrity_failed",
            error=exc,
            phase="parent_source_postcheck",
            preflight=preflight,
            input_payload=input_payload,
            attempted_fit=attempted_from_heartbeat or bool(payload.get("attempted_fit")),
            started_at=started_at,
        )
        source_snapshot_after = None

    payload["isolation"] = {
        "worker_launched": outcome.launch_error is None,
        "worker_returncode": outcome.returncode,
        "worker_termination_signal": outcome.termination_signal,
        "worker_wall_seconds": outcome.wall_seconds,
        "timeout_seconds": args.timeout,
        "timed_out": outcome.timed_out,
        "launch_error": outcome.launch_error,
        "log": str(log_path),
        "heartbeat": heartbeat,
    }
    payload["source_snapshot_after_parent"] = source_snapshot_after
    payload["source_unchanged_parent"] = source_snapshot_after is not None
    effective_preflight = payload.get("preflight")
    payload["preflight_advisory"] = bool(effective_preflight and effective_preflight["warning"])
    payload["preflight_disposition"] = _preflight_disposition(
        effective_preflight, payload["attempted_fit"]
    )
    payload["observed_attempt_failure"] = bool(
        payload["attempted_fit"] and payload["status"] != "ok"
    )
    if payload["status"] != "ok":
        payload["quarantined_layouts"] = _quarantine_generation(generation_path, run_id)
        failure_path = (
            result_path.with_name(f"{result_path.stem}.{run_id}.failed{result_path.suffix}")
            if result_preexisted
            else result_path
        )
        _atomic_json(failure_path, payload)
        for private in (worker_result, worker_heartbeat):
            with contextlib.suppress(FileNotFoundError):
                private.unlink()
        return 1

    # The immutable generation is canonical and is already validated. Publish the
    # result first, then update the convenience pointer; a failed forced rerun never
    # touches the previous result or pointer.
    try:
        _assert_source_unchanged(source_path, source_snapshot_before)
        payload["layout_publication"] = {
            "generation": str(generation_path),
            "pointer": str(layout_pointer),
            "publication_order": "immutable generation, result JSON, atomic latest symlink",
            "pointer_status": "pending",
        }
        _atomic_json(result_path, payload)
        args._generation_committed = True
    except (KeyboardInterrupt, SystemExit):
        _quarantine_generation(generation_path, run_id)
        raise
    except Exception as exc:
        quarantined = _quarantine_generation(generation_path, run_id)
        failure = _failure_payload(
            worker_args,
            status=classify_exception(exc, phase="layout_write"),
            error=exc,
            phase="result_publication",
            preflight=payload.get("preflight"),
            input_payload=input_payload,
            attempted_fit=True,
            started_at=started_at,
        )
        failure["quarantined_layouts"] = quarantined
        failure["isolation"] = payload["isolation"]
        failure_path = (
            result_path.with_name(f"{result_path.stem}.{run_id}.failed{result_path.suffix}")
            if result_preexisted
            else result_path
        )
        _atomic_json(failure_path, failure)
        for private in (worker_result, worker_heartbeat):
            with contextlib.suppress(FileNotFoundError):
                private.unlink()
        return 1
    try:
        _publish_layout_pointer(layout_pointer, generation_path)
        payload["layout_publication"]["pointer_status"] = "ok"
    except OSError as exc:  # The result still points to the canonical generation.
        payload["layout_publication"].update(
            {
                "pointer_status": "failed_nonfatal",
                "pointer_error_type": type(exc).__name__,
                "pointer_error_message": str(exc),
            }
        )
    try:
        _atomic_json(result_path, payload)
    except OSError as exc:
        print(
            f"warning: could not finalize pointer status in {result_path}: {exc}",
            file=sys.stderr,
        )
    for private in (worker_result, worker_heartbeat):
        with contextlib.suppress(FileNotFoundError):
            private.unlink()
    return 0


def run_isolated(args: argparse.Namespace) -> int:
    """Guard the isolated run so interrupted publications cannot leave live orphans."""
    try:
        return _run_isolated_impl(args)
    except BaseException:
        generation = getattr(args, "_active_generation_path", None)
        if generation and not getattr(args, "_generation_committed", False):
            _quarantine_generation(Path(generation), getattr(args, "run_id", "unknown"))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Yandex fbin base vectors")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--rows", type=int, required=True, help="exact leading row prefix")
    parser.add_argument("--target-dimension", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-path")
    parser.add_argument("--layout-path")
    parser.add_argument("--log-path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--init", choices=("random", "pca", "spectral"), default="random")
    parser.add_argument("--timeout", type=int, default=0, help="seconds; zero disables")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--dire-repo",
        default=None,
        help="Optional DiRe source checkout; installed dire-rapids is used by default",
    )
    parser.add_argument("--dire-cuvs-index", choices=CUVS_INDEX_TYPES, default="ivf_pq")
    parser.add_argument("--dire-iterations", type=int, default=128)
    parser.add_argument("--dire-neg-ratio", type=int, default=DEFAULT_DIRE_NEG_RATIO)

    parser.add_argument("--umap-knn-n-clusters", type=int, default=16)
    parser.add_argument("--umap-knn-overlap-factor", type=int, default=2)
    parser.add_argument("--umap-nnd-graph-degree", type=int, default=64)
    parser.add_argument("--umap-nnd-intermediate-graph-degree", type=int, default=128)
    parser.add_argument("--umap-nnd-max-iterations", type=int, default=20)
    parser.add_argument("--umap-nnd-termination-threshold", type=float, default=0.0001)
    parser.add_argument("--umap-device-ids", type=_parse_device_ids, default=None)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-spread", type=float, default=1.0)
    parser.add_argument("--umap-epochs", type=int)
    parser.add_argument("--umap-negative-sample-rate", type=int, default=5)
    parser.add_argument(
        "--umap-force-serial-epochs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--preflight-safety-factor", type=float, default=1.5)
    parser.add_argument("--preflight-vram-fraction", type=float, default=0.85)
    parser.add_argument("--disk-reserve-gib", type=float, default=1.0)
    parser.add_argument("--validation-block-rows", type=int, default=1_048_576)
    parser.add_argument("--layout-write-block-rows", type=int, default=1_048_576)
    parser.add_argument("--resource-sample-interval", type=float, default=0.1)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--worker-heartbeat", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rows < 2:
        parser.error("--rows must be at least two")
    if args.target_dimension < 1:
        parser.error("--target-dimension must be positive")
    if args.neighbors < 1 or args.neighbors >= args.rows:
        parser.error("--neighbors must be in [1, rows)")
    if args.timeout < 0:
        parser.error("--timeout cannot be negative")
    if min(args.dire_iterations, args.dire_neg_ratio) < 1:
        parser.error("DiRe iterations and negative ratio must be positive")
    if (
        min(
            args.umap_knn_n_clusters,
            args.umap_knn_overlap_factor,
            args.umap_nnd_graph_degree,
            args.umap_nnd_intermediate_graph_degree,
            args.umap_nnd_max_iterations,
            args.umap_negative_sample_rate,
        )
        < 1
    ):
        parser.error("UMAP graph controls must be positive")
    if args.umap_knn_n_clusters > 1 and args.umap_knn_overlap_factor >= args.umap_knn_n_clusters:
        parser.error("UMAP overlap must be smaller than knn-n-clusters")
    if args.umap_nnd_graph_degree < args.neighbors:
        parser.error("UMAP NN-descent graph degree must be at least --neighbors")
    if args.umap_nnd_intermediate_graph_degree <= args.umap_nnd_graph_degree:
        parser.error("UMAP intermediate graph degree must exceed graph degree")
    if args.umap_nnd_termination_threshold <= 0:
        parser.error("UMAP termination threshold must be positive")
    if args.umap_epochs is not None and args.umap_epochs < 1:
        parser.error("UMAP epochs must be positive")
    if args.method == "umap" and args.init == "pca":
        parser.error("RAPIDS UMAP does not accept --init=pca")
    if args.method == "umap" and (
        args.umap_device_ids == "all"
        or (isinstance(args.umap_device_ids, list) and len(args.umap_device_ids) > 1)
    ):
        parser.error(
            "multi-GPU UMAP is disabled because process-memory monitoring and "
            "capacity preflight currently cover exactly one device"
        )
    if (
        args.method == "umap"
        and args.umap_force_serial_epochs is True
        and args.target_dimension > 512
    ):
        parser.error("RAPIDS UMAP serial epochs support at most 512 target dimensions")
    if not 0 < args.umap_min_dist <= args.umap_spread:
        parser.error("UMAP min-dist must be positive and no larger than spread")
    if args.preflight_safety_factor < 1:
        parser.error("preflight safety factor must be at least one")
    if not 0 < args.preflight_vram_fraction <= 1:
        parser.error("preflight VRAM fraction must be in (0, 1]")
    if args.disk_reserve_gib < 0:
        parser.error("disk reserve cannot be negative")
    if min(args.validation_block_rows, args.layout_write_block_rows) < 1:
        parser.error("layout block sizes must be positive")
    if args.resource_sample_interval <= 0:
        parser.error("resource sample interval must be positive")

    output = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(output)
    args.result_path = str(
        _absolute_path(args.result_path) if args.result_path else output / "result.json"
    )
    args.layout_path = str(
        _absolute_path(args.layout_path) if args.layout_path else output / "layout.npy"
    )
    args.log_path = str(_absolute_path(args.log_path) if args.log_path else output / "worker.log")
    try:
        _validate_publication_paths(args)
    except IntegrityFailure as exc:
        parser.error(str(exc))
    if args.worker and not all((args.worker_result, args.worker_heartbeat, args.run_id)):
        parser.error("internal worker mode requires result, heartbeat, and run ID")
    if args.worker:
        args.worker_result = str(_absolute_path(args.worker_result))
        args.worker_heartbeat = str(_absolute_path(args.worker_heartbeat))
        all_files = [
            Path(args.data).expanduser().resolve(),
            Path(args.result_path),
            Path(args.layout_path),
            Path(args.log_path),
            Path(args.worker_result),
            Path(args.worker_heartbeat),
        ]
        existing_alias = any(
            _paths_are_existing_aliases(first, second)
            for index, first in enumerate(all_files)
            for second in all_files[index + 1 :]
        )
        if len(set(all_files)) != 6 or existing_alias:
            parser.error("source and all worker/publication files must be distinct")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_worker(args) if args.worker else run_isolated(args)


if __name__ == "__main__":
    raise SystemExit(main())
