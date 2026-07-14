#!/usr/bin/env python3
"""Exact, bounded-memory retrieval evaluation for Yandex/BigANN layouts.

The evaluator compares neighbours in a reduced-dimensional ``base`` layout with
the original-space neighbours stored in a BigANN ground-truth file.  It scans
the base layout once using positional bounded reads and processes query batches
inside each base block; the complete ``queries x base`` distance matrix is never
constructed.  Neighbours are ordered lexicographically by ``(squared Euclidean
distance, base row id)``, so ties are reproducible across base-block boundaries.

Example
-------
Evaluate the first 100 public queries against a 10-million-row layout::

    python experiments/evaluate_yandex_retrieval.py \
      --base-layout artifacts/deep-10m-layout.npy \
      --query-layout artifacts/deep-query-layout.npy \
      --ground-truth artifacts/groundtruth.deep-10M.bin \
      --query-limit 100 --k 100 \
      --output artifacts/deep-10m-retrieval.json

The search is exact in the supplied layout, not in the original 96-dimensional
space.  ``recall@k`` is the mean, over evaluated queries, of the set
intersection with the first ``k`` original-space ground-truth ids, divided by
``k``.  This is deliberately strict ID recall: unlike the official BigANN
scorer, it does not credit a different database id whose original-space
distance happens to tie the ground-truth boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import resource
import shutil
import struct
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

GROUND_TRUTH_HEADER_BYTES = 8
GROUND_TRUTH_ID_DTYPE = np.dtype("<u4")
GROUND_TRUTH_DISTANCE_DTYPE = np.dtype("<f4")
_GROUND_TRUTH_HEADER = struct.Struct("<II")
SCHEMA_VERSION = 1
PROTOCOL = "exact-layout-retrieval-v1"
_MIB = 2**20
_GIB = 2**30


@dataclass(frozen=True)
class GroundTruthInfo:
    """Strictly validated metadata for one BigANN ground-truth file."""

    path: Path
    queries: int
    neighbors: int
    expected_size: int

    @property
    def ids_offset(self) -> int:
        return GROUND_TRUTH_HEADER_BYTES

    @property
    def distances_offset(self) -> int:
        return self.ids_offset + self.queries * self.neighbors * GROUND_TRUTH_ID_DTYPE.itemsize


@dataclass(frozen=True)
class LayoutInfo:
    """Strict metadata needed for bounded positional reads of a C-order NPY."""

    path: Path
    rows: int
    dimension: int
    dtype: np.dtype
    data_offset: int
    expected_size: int


class BoundedNpyReader:
    """Read owned row blocks without retaining corpus pages in a memory map."""

    def __init__(self, info: LayoutInfo):
        self.info = info
        self.shape = (info.rows, info.dimension)
        self.dtype = info.dtype
        self.row_bytes = info.dimension * info.dtype.itemsize
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        self._fd = -1
        self._fd = os.open(info.path, flags)

    def close(self) -> None:
        descriptor, self._fd = self._fd, -1
        if descriptor >= 0:
            os.close(descriptor)

    def __enter__(self) -> BoundedNpyReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive exceptional cleanup.
        with contextlib.suppress(OSError):
            self.close()

    def _pread_exact(self, offset: int, length: int) -> bytearray:
        payload = bytearray(length)
        view = memoryview(payload)
        cursor = 0
        while cursor < length:
            try:
                if hasattr(os, "preadv"):
                    received = os.preadv(self._fd, [view[cursor:]], offset + cursor)
                else:  # pragma: no cover - benchmark Linux hosts expose preadv.
                    chunk = os.pread(self._fd, length - cursor, offset + cursor)
                    received = len(chunk)
                    view[cursor : cursor + received] = chunk
            except InterruptedError:
                continue
            if received <= 0:
                raise ValueError(
                    f"short positional NPY read at byte {offset + cursor}: "
                    f"expected {length - cursor} more bytes"
                )
            cursor += received
        return payload

    def read_block(self, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= self.info.rows:
            raise IndexError(f"invalid bounded NPY block [{start}, {stop})")
        payload = self._pread_exact(
            self.info.data_offset + start * self.row_bytes,
            (stop - start) * self.row_bytes,
        )
        return np.frombuffer(payload, dtype=self.info.dtype).reshape(
            stop - start, self.info.dimension
        )


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def inspect_ground_truth(path: str | os.PathLike[str]) -> GroundTruthInfo:
    """Decode a BigANN ground-truth header and require its exact file length.

    The little-endian format is ``uint32(n_queries, k)``, followed by all
    ``n_queries * k`` uint32 neighbour ids and then the same number of float32
    distances.  Truncation and trailing bytes are both rejected.
    """

    source = Path(path)
    try:
        size = source.stat().st_size
        with source.open("rb") as handle:
            header = handle.read(GROUND_TRUTH_HEADER_BYTES)
    except OSError as error:
        raise ValueError(f"could not inspect ground truth {source}: {error}") from error
    if len(header) != GROUND_TRUTH_HEADER_BYTES:
        raise ValueError(
            f"invalid ground-truth header in {source}: expected "
            f"{GROUND_TRUTH_HEADER_BYTES} bytes, found {len(header)}"
        )
    queries, neighbors = _GROUND_TRUTH_HEADER.unpack(header)
    if queries == 0 or neighbors == 0:
        raise ValueError(
            f"invalid ground-truth shape in {source}: queries={queries}, neighbors={neighbors}"
        )
    payload_values = int(queries) * int(neighbors)
    expected = GROUND_TRUTH_HEADER_BYTES + payload_values * (
        GROUND_TRUTH_ID_DTYPE.itemsize + GROUND_TRUTH_DISTANCE_DTYPE.itemsize
    )
    if size != expected:
        kind = "truncated" if size < expected else "has trailing bytes"
        raise ValueError(
            f"invalid ground-truth length in {source}: file {kind}; "
            f"expected {expected} bytes, found {size}"
        )
    return GroundTruthInfo(source, int(queries), int(neighbors), int(expected))


def open_ground_truth(
    path: str | os.PathLike[str],
) -> tuple[GroundTruthInfo, np.memmap, np.memmap]:
    """Return read-only memory maps for the strictly validated ids and distances."""

    info = inspect_ground_truth(path)
    shape = (info.queries, info.neighbors)
    ids = np.memmap(
        info.path,
        dtype=GROUND_TRUTH_ID_DTYPE,
        mode="r",
        offset=info.ids_offset,
        shape=shape,
        order="C",
    )
    distances = np.memmap(
        info.path,
        dtype=GROUND_TRUTH_DISTANCE_DTYPE,
        mode="r",
        offset=info.distances_offset,
        shape=shape,
        order="C",
    )
    ids.flags.writeable = False
    distances.flags.writeable = False
    return info, ids, distances


def _load_layout(path: Path, *, name: str) -> np.memmap:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{name} must be an uncompressed .npy file: {path}")
    try:
        layout = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"could not memory-map {name} {path}: {error}") from error
    if not isinstance(layout, np.ndarray) or layout.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if layout.shape[0] == 0 or layout.shape[1] == 0:
        raise ValueError(f"{name} must have two positive axes, found {layout.shape}")
    if layout.dtype.kind != "f" or layout.dtype.itemsize not in (4, 8):
        raise ValueError(f"{name} dtype must be float32 or float64, found {layout.dtype}")
    if not isinstance(layout, np.memmap):
        raise ValueError(f"{name} could not be represented as a memory-mapped .npy array")
    if not layout.flags.c_contiguous:
        raise ValueError(f"{name} must use C row-major storage for bounded row streaming")
    expected_size = int(layout.offset) + int(layout.nbytes)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        kind = "truncated" if actual_size < expected_size else "has trailing bytes"
        raise ValueError(
            f"invalid {name} length in {path}: file {kind}; "
            f"expected {expected_size} bytes, found {actual_size}"
        )
    layout.flags.writeable = False
    return layout


def _inspect_layout(path: Path, *, name: str) -> LayoutInfo:
    """Inspect an NPY via NumPy and close its temporary metadata mapping."""

    layout = _load_layout(path, name=name)
    info = LayoutInfo(
        path=path,
        rows=int(layout.shape[0]),
        dimension=int(layout.shape[1]),
        dtype=layout.dtype,
        data_offset=int(layout.offset),
        expected_size=int(layout.offset) + int(layout.nbytes),
    )
    mmap_handle = layout._mmap
    del layout
    mmap_handle.close()
    return info


def _resolved_identity(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def validate_distinct_paths(paths: dict[str, Path]) -> None:
    """Reject lexical, symlink, and hard-link aliases among inputs and output."""

    items = list(paths.items())
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1 :]:
            same = _resolved_identity(left_path) == _resolved_identity(right_path)
            if not same and left_path.exists() and right_path.exists():
                with contextlib.suppress(OSError):
                    same = os.path.samefile(left_path, right_path)
            if same:
                raise ValueError(
                    f"path alias is not allowed: {left_name} and {right_name} both refer to "
                    f"{_resolved_identity(left_path)}"
                )


def _source_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def conservative_memory_plan(
    *,
    base_rows: int,
    dimension: int,
    query_count: int,
    k: int,
    query_batch_size: int,
    base_block_rows: int,
    budget_bytes: int,
    base_dtype_itemsize: int = 8,
) -> dict[str, Any]:
    """Estimate explicit peak workspaces and reject an insufficient host budget.

    NumPy/BLAS process baselines and kernel page cache are not controllable by
    this program.  The estimate adds a 25% workspace margin and 64 MiB for BLAS
    internals on top of all arrays explicitly live in the inner search loop.
    """

    rows = _positive_integer(base_rows, name="base_rows")
    dim = _positive_integer(dimension, name="dimension")
    queries = _positive_integer(query_count, name="query_count")
    neighbors = _positive_integer(k, name="k")
    query_batch = min(queries, _positive_integer(query_batch_size, name="query_batch_size"))
    base_block = min(rows, _positive_integer(base_block_rows, name="base_block_rows"))
    budget = _positive_integer(budget_bytes, name="budget_bytes")
    source_itemsize = _positive_integer(base_dtype_itemsize, name="base_dtype_itemsize")
    components = {
        "distance_matrix_float64": query_batch * base_block * 8,
        "query_batch_float64": query_batch * dim * 8,
        "base_block_float64": base_block * dim * 8,
        "owned_base_source_block": base_block * dim * source_itemsize,
        "query_and_base_norms_float64": (query_batch + base_block) * 8,
        "base_finiteness_mask": base_block * dim,
        "argpartition_indices_one_query": base_block * 8,
        "tie_masks_one_query": base_block * 2,
        "all_query_best_topk": queries * neighbors * 16,
        "batch_block_and_merge_topk": query_batch * neighbors * 16 * 4,
        "ground_truth_batch_copies": query_batch * neighbors * 8,
    }
    explicit = int(sum(components.values()))
    with_margin = int(explicit * 1.25) + 64 * _MIB
    if with_margin > budget:
        raise ValueError(
            "host-memory budget is too small for the requested query/base blocks: "
            f"estimated {with_margin / _GIB:.3f} GiB > budget {budget / _GIB:.3f} GiB"
        )
    return {
        "budget_bytes": budget,
        "effective_query_batch_size": query_batch,
        "effective_base_block_rows": base_block,
        "components_bytes": components,
        "explicit_workspace_bytes": explicit,
        "conservative_peak_workspace_bytes": with_margin,
        "scope": (
            "Explicit evaluator arrays plus a 25% margin and 64 MiB BLAS allowance; "
            "excludes the process baseline, memory-mapped virtual address ranges, and OS page cache."
        ),
    }


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ValueError(f"no existing ancestor for output path {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def _disk_preflight(output: Path, *, reserve_bytes: int) -> dict[str, int]:
    if reserve_bytes < 0:
        raise ValueError("disk reserve must be nonnegative")
    ancestor = _nearest_existing_directory(output.parent)
    usage = shutil.disk_usage(ancestor)
    atomic_result_allowance = _MIB
    required = int(reserve_bytes) + atomic_result_allowance
    if usage.free < required:
        raise ValueError(
            f"insufficient free disk for atomic result: {usage.free} bytes available, "
            f"{required} bytes required including reserve"
        )
    return {
        "free_bytes_before_run": int(usage.free),
        "reserve_bytes": int(reserve_bytes),
        "atomic_result_allowance_bytes": atomic_result_allowance,
    }


def _validate_ground_truth_values(
    ids: np.ndarray,
    distances: np.ndarray,
    *,
    base_rows: int,
    k: int,
    row_block: int = 4096,
) -> None:
    """Validate all GT values in bounded query-row blocks."""

    for start in range(0, ids.shape[0], row_block):
        stop = min(ids.shape[0], start + row_block)
        id_block = np.asarray(ids[start:stop])
        distance_block = np.asarray(distances[start:stop])
        if np.any(id_block >= base_rows):
            bad = int(id_block[id_block >= base_rows][0])
            raise ValueError(f"ground-truth id {bad} is outside base layout with {base_rows} rows")
        if not np.all(np.isfinite(distance_block)):
            raise ValueError("ground-truth distances contain non-finite values")
        if np.any(distance_block < 0):
            raise ValueError("ground-truth distances contain negative values")
        if distance_block.shape[1] > 1 and np.any(np.diff(distance_block, axis=1) < 0):
            raise ValueError("ground-truth distances are not nondecreasing within each query")
        for row in id_block[:, :k]:
            if np.unique(row).size != k:
                raise ValueError(f"ground-truth first {k} ids contain duplicates")


def _validate_query_values(query_layout: np.ndarray, *, block_rows: int = 4096) -> None:
    for start in range(0, query_layout.shape[0], block_rows):
        block = np.asarray(query_layout[start : start + block_rows])
        if not np.all(np.isfinite(block)):
            raise ValueError("query layout contains non-finite values")


def _first_equal_positions(row: np.ndarray, value: float, count: int) -> np.ndarray:
    """Return the first ``count`` matching positions without a corpus-sized id list."""

    if count == 0:
        return np.empty(0, dtype=np.int64)
    pieces: list[np.ndarray] = []
    remaining = count
    scan_rows = 65_536
    for start in range(0, row.size, scan_rows):
        stop = min(row.size, start + scan_rows)
        matches = np.flatnonzero(row[start:stop] == value)
        if matches.size:
            selected = matches[:remaining].astype(np.int64, copy=False) + start
            pieces.append(selected)
            remaining -= selected.size
            if remaining == 0:
                break
    if remaining:
        raise RuntimeError("internal top-k tie selection failed")
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def _block_topk(
    distances: np.ndarray,
    *,
    base_start: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select per-row block top-k exactly by ``(distance, absolute id)``."""

    local_k = min(k, distances.shape[1])
    top_distances = np.empty((distances.shape[0], local_k), dtype=np.float64)
    top_ids = np.empty((distances.shape[0], local_k), dtype=np.uint64)
    for query_index, row in enumerate(distances):
        partition = np.argpartition(row, local_k - 1)
        provisional = partition[:local_k]
        threshold = float(np.max(row[provisional]))
        below = np.flatnonzero(row < threshold).astype(np.int64, copy=False)
        needed = local_k - below.size
        equal = _first_equal_positions(row, threshold, needed)
        selected = np.concatenate((below, equal))
        ids = selected.astype(np.uint64) + np.uint64(base_start)
        order = np.lexsort((ids, row[selected]))
        selected = selected[order]
        top_distances[query_index] = row[selected]
        top_ids[query_index] = selected.astype(np.uint64) + np.uint64(base_start)
    return top_distances, top_ids


def _merge_topk(
    current_distances: np.ndarray,
    current_ids: np.ndarray,
    block_distances: np.ndarray,
    block_ids: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two sorted candidate tables by the same deterministic ordering."""

    candidate_distances = np.concatenate((current_distances, block_distances), axis=1)
    candidate_ids = np.concatenate((current_ids, block_ids), axis=1)
    keep = min(k, candidate_distances.shape[1])
    output_distances = np.empty((candidate_distances.shape[0], keep), dtype=np.float64)
    output_ids = np.empty((candidate_ids.shape[0], keep), dtype=np.uint64)
    for row_index in range(candidate_distances.shape[0]):
        order = np.lexsort((candidate_ids[row_index], candidate_distances[row_index]))[:keep]
        output_distances[row_index] = candidate_distances[row_index, order]
        output_ids[row_index] = candidate_ids[row_index, order]
    return output_distances, output_ids


def _squared_euclidean(queries: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Compute one bounded float64 distance tile with no q*b*d temporary."""

    distances = queries @ base.T
    distances *= -2.0
    distances += np.einsum("ij,ij->i", queries, queries)[:, None]
    distances += np.einsum("ij,ij->i", base, base)[None, :]
    np.maximum(distances, 0.0, out=distances)
    if not np.all(np.isfinite(distances)):
        raise ValueError("squared Euclidean distance overflowed despite finite layout values")
    return distances


def _exact_streamed_neighbors(
    base_reader: BoundedNpyReader,
    query_layout: np.ndarray,
    *,
    query_count: int,
    k: int,
    query_batch_size: int,
    base_block_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan each base block once while updating bounded per-query top-k state."""

    best_distances = np.full((query_count, k), np.inf, dtype=np.float64)
    best_ids = np.full((query_count, k), np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, base_reader.shape[0], base_block_rows):
        stop = min(base_reader.shape[0], start + base_block_rows)
        source_block = base_reader.read_block(start, stop)
        base = np.array(source_block, dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(base)):
            raise ValueError(f"base layout rows [{start}, {stop}) contain non-finite values")
        for query_start in range(0, query_count, query_batch_size):
            query_stop = min(query_count, query_start + query_batch_size)
            queries = np.array(
                query_layout[query_start:query_stop],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            if not np.all(np.isfinite(queries)):
                raise ValueError("query layout contains non-finite values")
            distances = _squared_euclidean(queries, base)
            block_distances, block_ids = _block_topk(distances, base_start=start, k=k)
            merged_distances, merged_ids = _merge_topk(
                best_distances[query_start:query_stop],
                best_ids[query_start:query_stop],
                block_distances,
                block_ids,
                k=k,
            )
            best_distances[query_start:query_stop] = merged_distances
            best_ids[query_start:query_stop] = merged_ids
    if np.any(best_ids == np.iinfo(np.uint64).max) or not np.all(np.isfinite(best_distances)):
        raise RuntimeError("internal search ended with fewer than k finite neighbours")
    return best_distances, best_ids


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Validate inputs and run exact streamed CPU retrieval evaluation."""

    started_wall = time.perf_counter()
    started_at = _utc_now()
    base_path = Path(args.base_layout)
    query_path = Path(args.query_layout)
    ground_truth_path = Path(args.ground_truth)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"output must have a .json suffix: {output_path}")
    validate_distinct_paths(
        {
            "base_layout": base_path,
            "query_layout": query_path,
            "ground_truth": ground_truth_path,
            "output": output_path,
        }
    )
    snapshots_before = {
        "base_layout": _source_snapshot(base_path),
        "query_layout": _source_snapshot(query_path),
        "ground_truth": _source_snapshot(ground_truth_path),
    }
    base_info = _inspect_layout(base_path, name="base layout")
    query_layout = _load_layout(query_path, name="query layout")
    gt_info, gt_ids, gt_distances = open_ground_truth(ground_truth_path)
    if base_info.dimension != query_layout.shape[1]:
        raise ValueError(
            "base and query layout dimensions differ: "
            f"{base_info.dimension} != {query_layout.shape[1]}"
        )
    if query_layout.shape[0] != gt_info.queries:
        raise ValueError(
            "query layout row count must exactly match ground truth: "
            f"{query_layout.shape[0]} != {gt_info.queries}"
        )
    k = _positive_integer(args.k, name="k")
    if k > gt_info.neighbors:
        raise ValueError(
            f"requested k={k}, but ground truth contains only {gt_info.neighbors} neighbours"
        )
    if k > base_info.rows:
        raise ValueError(f"requested k={k}, but base layout has only {base_info.rows} rows")
    if args.query_limit is None:
        query_count = gt_info.queries
    else:
        query_count = _positive_integer(args.query_limit, name="query_limit")
        if query_count > gt_info.queries:
            raise ValueError(
                f"query_limit={query_count} exceeds the {gt_info.queries} available queries"
            )
    query_batch_size = _positive_integer(args.query_batch_size, name="query_batch_size")
    base_block_rows = _positive_integer(args.base_block_rows, name="base_block_rows")
    budget_bytes = int(float(args.host_memory_budget_gib) * _GIB)
    if not np.isfinite(args.host_memory_budget_gib) or budget_bytes <= 0:
        raise ValueError("host_memory_budget_gib must be finite and positive")
    reserve_bytes = int(float(args.disk_reserve_gib) * _GIB)
    if not np.isfinite(args.disk_reserve_gib) or reserve_bytes < 0:
        raise ValueError("disk_reserve_gib must be finite and nonnegative")
    memory_plan = conservative_memory_plan(
        base_rows=base_info.rows,
        dimension=base_info.dimension,
        query_count=query_count,
        k=k,
        query_batch_size=query_batch_size,
        base_block_rows=base_block_rows,
        budget_bytes=budget_bytes,
        base_dtype_itemsize=base_info.dtype.itemsize,
    )
    disk_plan = _disk_preflight(output_path, reserve_bytes=reserve_bytes)

    validation_started = time.perf_counter()
    _validate_query_values(query_layout)
    _validate_ground_truth_values(
        gt_ids,
        gt_distances,
        base_rows=base_info.rows,
        k=k,
    )
    validation_seconds = time.perf_counter() - validation_started

    search_started = time.perf_counter()
    recall_hits = 0
    digest = hashlib.sha256()
    effective_query_batch = memory_plan["effective_query_batch_size"]
    with BoundedNpyReader(base_info) as base_reader:
        all_distances, all_ids = _exact_streamed_neighbors(
            base_reader,
            query_layout,
            query_count=query_count,
            k=k,
            query_batch_size=effective_query_batch,
            base_block_rows=memory_plan["effective_base_block_rows"],
        )
    for query_start in range(0, query_count, effective_query_batch):
        query_stop = min(query_count, query_start + effective_query_batch)
        ids = all_ids[query_start:query_stop]
        digest.update(np.ascontiguousarray(ids, dtype="<u8").tobytes())
        truth_batch = np.asarray(gt_ids[query_start:query_stop, :k], dtype=np.uint64)
        for retrieved, truth in zip(ids, truth_batch, strict=True):
            recall_hits += int(np.intersect1d(retrieved, truth, assume_unique=True).size)
    maximum_squared_distance = float(np.max(all_distances))
    minimum_squared_distance = float(np.min(all_distances))
    search_seconds = time.perf_counter() - search_started

    snapshots_after = {
        "base_layout": _source_snapshot(base_path),
        "query_layout": _source_snapshot(query_path),
        "ground_truth": _source_snapshot(ground_truth_path),
    }
    if snapshots_after != snapshots_before:
        raise ValueError("an input file changed during retrieval evaluation")
    distance_evaluations = int(query_count * base_info.rows)
    total_seconds = time.perf_counter() - started_wall
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "engine": "numpy-cpu-float64-distance",
        "metric": "squared_euclidean",
        "tie_break": "ascending_base_row_id",
        "base_layout": {
            **snapshots_before["base_layout"],
            "rows": base_info.rows,
            "dimension": base_info.dimension,
            "dtype": str(base_info.dtype),
        },
        "query_layout": {
            **snapshots_before["query_layout"],
            "rows": int(query_layout.shape[0]),
            "rows_evaluated": int(query_count),
            "dimension": int(query_layout.shape[1]),
            "dtype": str(query_layout.dtype),
        },
        "ground_truth": {
            **snapshots_before["ground_truth"],
            "queries": gt_info.queries,
            "neighbors_per_query": gt_info.neighbors,
            "format": "little-endian uint32 header, uint32 ids, float32 distances",
        },
        "search": {
            "k": k,
            "query_batch_size_requested": query_batch_size,
            "query_batch_size_effective": effective_query_batch,
            "base_block_rows_requested": base_block_rows,
            "base_block_rows_effective": memory_plan["effective_base_block_rows"],
            "base_stream_passes": 1,
            "base_input_access": "positional owned row blocks; no persistent base mmap",
            "distance_evaluations": distance_evaluations,
            "full_query_by_base_matrix_materialized": False,
            "per_query_topk_state_materialized": True,
            "retrieved_ids_sha256_uint64_little_endian": digest.hexdigest(),
            "minimum_retrieved_squared_distance": minimum_squared_distance,
            "maximum_retrieved_squared_distance": maximum_squared_distance,
        },
        "quality": {
            f"recall@{k}": recall_hits / (query_count * k),
            "recall_hits": recall_hits,
            "recall_denominator": query_count * k,
            "tie_policy": (
                "strict ground-truth ID intersection; alternative original-space "
                "distance ties are not credited"
            ),
            "definition": (
                "mean per-query set intersection with the first k original-space "
                "ground-truth ids, divided by k"
            ),
        },
        "timing_seconds": {
            "input_value_validation": validation_seconds,
            "exact_layout_search": search_seconds,
            "total": total_seconds,
        },
        "throughput": {
            "distance_evaluations_per_second": (
                distance_evaluations / search_seconds if search_seconds > 0 else None
            )
        },
        "resources": {
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "peak_rss_scope": (
                "ru_maxrss high-water mark for the process lifetime; invoke through the CLI "
                "in a fresh process for run-local interpretation"
            ),
            "host_memory_plan": memory_plan,
            "disk_plan": disk_plan,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "input_unchanged_during_run": True,
    }


def execute(args: argparse.Namespace) -> int:
    result = run_evaluation(args)
    _atomic_json(Path(args.output), result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-layout", type=Path, required=True)
    parser.add_argument("--query-layout", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Evaluate this prefix of queries (default: every ground-truth query).",
    )
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--base-block-rows", type=int, default=131_072)
    parser.add_argument("--host-memory-budget-gib", type=float, default=4.0)
    parser.add_argument(
        "--disk-reserve-gib",
        type=float,
        default=0.1,
        help="Free space that must remain in addition to a 1 MiB atomic-result allowance.",
    )
    return parser


def _test_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "base_layout": Path("base.npy"),
        "query_layout": Path("query.npy"),
        "ground_truth": Path("ground-truth.bin"),
        "output": Path("retrieval.json"),
        "k": 1,
        "query_limit": None,
        "query_batch_size": 2,
        "base_block_rows": 3,
        "host_memory_budget_gib": 1.0,
        "disk_reserve_gib": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GroundTruthInfo",
    "conservative_memory_plan",
    "execute",
    "inspect_ground_truth",
    "open_ground_truth",
    "run_evaluation",
]
