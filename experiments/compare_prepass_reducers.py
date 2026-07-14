#!/usr/bin/env python3
"""Compare direct and compressed-input DiRe/UMAP pipelines.

This is a deliberately small, transductive pilot harness.  It accepts one prepared
``float32`` NPY matrix, makes one deterministic row subset, and compares

``X -> DiRe/UMAP``

with raw PCA scores and with canonicalized PCA/autoencoder coordinates:

``X -> PCA scores -> DiRe/UMAP``

``X -> PCA/autoencoder -> whiten -> row-normalize -> DiRe/UMAP``.

Each intermediate is cached once per pre-pass seed and reused by every downstream
method and final width.  Every graph reducer runs in a fresh subprocess.  Quality is
always evaluated against the original selected rows, never against the compressed
intermediate.  Raw MSE-autoencoder coordinates are canonicalized because their
Euclidean metric is otherwise unidentified up to an invertible linear change of basis.

Example (arXiv 100k pilot)::

    conda run -n rapids-26.06 python experiments/compare_prepass_reducers.py \
      --data artifacts/large-arxiv-d8/n723457/prepared-l2.npy \
      --labels artifacts/large-arxiv-d8/n723457/labels.npy \
      --allow-pickled-labels --rows 100000 --prepasses direct pca_raw pca ae \
      --prepass-widths 32 64 --methods dire umap --final-dimensions 8 \
      --seeds 0 --dire-repo ../dire-rapids \
      --output-dir artifacts/prepass-arxiv-100k-d8

Object-dtype label NPY files require the explicit ``--allow-pickled-labels`` opt-in.
Never enable it for an untrusted file.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

PREPASS_KINDS = ("direct", "pca_raw", "pca", "ae")
GRAPH_METHODS = ("dire", "umap")
SCHEMA_VERSION = 2

PREPASS_VARIANTS = {
    "direct": "direct_original_input",
    "pca_raw": "pca_raw_scores",
    # Preserve the original CLI spellings while making serialized labels explicit.
    "pca": "pca_whitened_row_l2",
    "ae": "ae_whitened_row_l2",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
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


def _atomic_save_array(path: str | Path, array: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=array.dtype.hasobject)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _sha256_file(path: str | Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _logical_float32_sha256(array: np.ndarray, block_rows: int = 16_384) -> str:
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], block_rows):
        block = np.ascontiguousarray(array[start : start + block_rows], dtype="<f4")
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def configuration_identity(payload: dict[str, Any]) -> str:
    """Return a stable identity for cache-relevant JSON configuration."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_snapshot(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    with source.open("rb") as handle:
        header = handle.read(4096)
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def deterministic_indices(total_rows: int, selected_rows: int, seed: int) -> np.ndarray:
    if total_rows < 2 or not 2 <= selected_rows <= total_rows:
        raise ValueError("selected rows must be in [2, total rows]")
    if selected_rows == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=selected_rows, replace=False)).astype(np.int64)


def monotone_hidden_dimensions(input_dimension: int, bottleneck_dimension: int) -> tuple[int, ...]:
    """Choose a strictly decreasing encoder which never dips below the bottleneck."""
    if not 0 < bottleneck_dimension < input_dimension:
        raise ValueError("bottleneck dimension must be positive and below input dimension")
    candidates = (1024, 512, 256, 128, 64, 32, 16)
    hidden = tuple(width for width in candidates if bottleneck_dimension < width < input_dimension)
    chain = (input_dimension, *hidden, bottleneck_dimension)
    if any(left <= right for left, right in zip(chain, chain[1:])):
        raise RuntimeError("autoencoder architecture is not strictly narrowing")
    return hidden


def canonicalize_array(
    codes: np.ndarray,
    *,
    relative_eigenvalue_floor: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Center, regularized-whiten, and row-L2-normalize an intermediate array."""
    values = np.asarray(codes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("codes must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("codes contain non-finite values")
    if not 0 < relative_eigenvalue_floor < 1:
        raise ValueError("relative eigenvalue floor must be in (0, 1)")
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(max(np.max(eigenvalues), np.finfo(np.float64).eps))
    floor = maximum * relative_eigenvalue_floor
    clipped = np.maximum(eigenvalues, floor)
    whitened = (centered @ eigenvectors) / np.sqrt(clipped)
    norms = np.linalg.norm(whitened, axis=1)
    zero_rows = norms <= np.finfo(np.float64).tiny
    norms[zero_rows] = 1.0
    canonical = np.ascontiguousarray(whitened / norms[:, None], dtype=np.float32)
    if not np.all(np.isfinite(canonical)):
        raise RuntimeError("canonicalization produced non-finite values")
    return canonical, {
        "relative_eigenvalue_floor": float(relative_eigenvalue_floor),
        "absolute_eigenvalue_floor": floor,
        "effective_rank": int(np.count_nonzero(eigenvalues > floor)),
        "input_width": int(values.shape[1]),
        "zero_rows_after_whitening": int(np.count_nonzero(zero_rows)),
        "eigenvalues_ascending": eigenvalues.tolist(),
        "transform": "center -> regularized covariance whitening -> row L2 normalization",
    }


def canonicalize_npy(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    relative_eigenvalue_floor: float,
    block_rows: int = 16_384,
) -> dict[str, Any]:
    """Bounded-row implementation of :func:`canonicalize_array` for a cached NPY."""
    started = time.perf_counter()
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.dtype != np.float32 or source.shape[0] < 2:
        raise ValueError("raw prepass codes must be a 2-D float32 NPY")
    rows, width = source.shape
    sums = np.zeros(width, dtype=np.float64)
    for start in range(0, rows, block_rows):
        block = np.asarray(source[start : start + block_rows], dtype=np.float64)
        if not np.all(np.isfinite(block)):
            raise ValueError("raw prepass codes contain non-finite values")
        sums += block.sum(axis=0)
    mean = sums / rows
    covariance_sum = np.zeros((width, width), dtype=np.float64)
    for start in range(0, rows, block_rows):
        centered = np.asarray(source[start : start + block_rows], dtype=np.float64) - mean
        covariance_sum += centered.T @ centered
    covariance = covariance_sum / max(1, rows - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(max(np.max(eigenvalues), np.finfo(np.float64).eps))
    floor = maximum * relative_eigenvalue_floor
    scale = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    zero_rows = 0
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(rows, width))
    try:
        for start in range(0, rows, block_rows):
            stop = min(rows, start + block_rows)
            centered = np.asarray(source[start:stop], dtype=np.float64) - mean
            whitened = (centered @ eigenvectors) * scale
            norms = np.linalg.norm(whitened, axis=1)
            zeros = norms <= np.finfo(np.float64).tiny
            norms[zeros] = 1.0
            block = np.ascontiguousarray(whitened / norms[:, None], dtype="<f4")
            if not np.all(np.isfinite(block)):
                raise RuntimeError("canonicalization produced non-finite values")
            output[start:stop] = block
            digest.update(block.tobytes(order="C"))
            zero_rows += int(np.count_nonzero(zeros))
        output.flush()
        completed_output = output
        output = None
        del completed_output
        os.replace(temporary, destination)
    except BaseException:
        if output is not None:
            with contextlib.suppress(Exception):
                del output
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    return {
        "seconds": time.perf_counter() - started,
        "path": str(destination.resolve()),
        "rows": int(rows),
        "width": int(width),
        "logical_sha256": digest.hexdigest(),
        "relative_eigenvalue_floor": float(relative_eigenvalue_floor),
        "absolute_eigenvalue_floor": floor,
        "effective_rank": int(np.count_nonzero(eigenvalues > floor)),
        "zero_rows_after_whitening": zero_rows,
        "eigenvalues_ascending": eigenvalues.tolist(),
        "transform": "center -> regularized covariance whitening -> row L2 normalization",
    }


def timing_accounting(
    prepass_compute_seconds: float,
    canonicalization_seconds: float,
    graph_fit_seconds: float,
    *,
    prepass_diagnostic_seconds: float = 0.0,
    prepass_worker_wall_seconds: float | None = None,
    graph_worker_wall_seconds: float | None = None,
) -> dict[str, float | None]:
    """Keep production-core, diagnostic-inclusive, and graph-only scopes distinct."""
    core_total = prepass_compute_seconds + canonicalization_seconds + graph_fit_seconds
    diagnostic_total = core_total + prepass_diagnostic_seconds
    process_total = None
    if prepass_worker_wall_seconds is not None and graph_worker_wall_seconds is not None:
        process_total = (
            prepass_worker_wall_seconds + canonicalization_seconds + graph_worker_wall_seconds
        )
    return {
        "prepass_core_compute_seconds": float(prepass_compute_seconds),
        # Backward-compatible spelling; this now explicitly means production core.
        "prepass_compute_seconds": float(prepass_compute_seconds),
        "prepass_diagnostic_seconds": float(prepass_diagnostic_seconds),
        "canonicalization_seconds": float(canonicalization_seconds),
        "graph_only_fit_seconds": float(graph_fit_seconds),
        "cold_end_to_end_core_compute_seconds": float(core_total),
        "cold_end_to_end_diagnostic_inclusive_compute_seconds": float(diagnostic_total),
        # Backward-compatible spelling; diagnostics are intentionally excluded.
        "cold_end_to_end_compute_seconds": float(core_total),
        "prepass_worker_wall_seconds": prepass_worker_wall_seconds,
        "graph_worker_wall_seconds": graph_worker_wall_seconds,
        "cold_end_to_end_process_seconds": process_total,
        "scope_note": (
            "core compute excludes reconstruction diagnostics and interpreter/import startup; "
            "process includes fresh-worker walls (and therefore diagnostics) plus "
            "canonicalization"
        ),
    }


def _neighbors_without_self(values: np.ndarray, neighbors: int) -> np.ndarray:
    raw = (
        NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=1)
        .fit(values)
        .kneighbors(values, return_distance=False)
    )
    result = np.empty((values.shape[0], neighbors), dtype=np.int64)
    for row in range(values.shape[0]):
        candidates = raw[row][raw[row] != row]
        if candidates.size < neighbors:
            raise RuntimeError("could not exclude self from neighbor result")
        result[row] = candidates[:neighbors]
    return result


def _distance_metrics(reference: np.ndarray, embedded: np.ndarray) -> tuple[float, float]:
    correlation = float(spearmanr(reference, embedded).statistic)
    denominator = float(np.dot(embedded, embedded))
    reference_norm = float(np.dot(reference, reference))
    if not np.isfinite(correlation) or denominator <= 0 or reference_norm <= 0:
        raise ValueError("sampled pair distances are degenerate")
    scale = float(np.dot(reference, embedded) / denominator)
    stress = float(np.sqrt(np.sum((reference - scale * embedded) ** 2) / reference_norm))
    return correlation, stress


def _paired_distances(
    values: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    block_rows: int = 4096,
) -> np.ndarray:
    """Compute declared pair distances without a pair-count-by-width temporary."""
    result = np.empty(first.size, dtype=np.float64)
    for start in range(0, first.size, block_rows):
        stop = min(first.size, start + block_rows)
        left = np.asarray(values[first[start:stop]], dtype=np.float64)
        right = np.asarray(values[second[start:stop]], dtype=np.float64)
        result[start:stop] = np.linalg.norm(left - right, axis=1)
    return result


def evaluate_against_original(
    original: np.ndarray,
    layout: np.ndarray,
    *,
    metric_rows: int,
    neighbors: int,
    distance_pairs: int,
    seed: int,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate ``layout`` with the declared reference ``original`` only."""
    original = np.asarray(original)
    layout = np.asarray(layout)
    if (
        original.ndim != 2
        or layout.ndim != 2
        or original.shape[0] != layout.shape[0]
        or not np.all(np.isfinite(original))
        or not np.all(np.isfinite(layout))
    ):
        raise ValueError("original and layout must be finite aligned matrices")
    if labels is not None and np.asarray(labels).shape != (original.shape[0],):
        raise ValueError("labels must align with original rows")
    count = min(metric_rows, original.shape[0])
    if count < 4:
        raise ValueError("metric sample must contain at least four rows")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(original.shape[0], size=count, replace=False))
    X = np.asarray(original[indices], dtype=np.float32)
    Y = np.asarray(layout[indices], dtype=np.float32)
    k = min(neighbors, count // 2 - 1)
    if k < 1:
        raise ValueError("metric sample is too small for neighbors")
    original_neighbors = _neighbors_without_self(X, k)
    layout_neighbors = _neighbors_without_self(Y, k)
    overlap = float(
        np.mean(
            [
                np.intersect1d(original_neighbors[row], layout_neighbors[row]).size / k
                for row in range(count)
            ]
        )
    )
    first = rng.integers(0, original.shape[0], size=distance_pairs)
    second = rng.integers(0, original.shape[0] - 1, size=distance_pairs)
    second += second >= first
    original_distance = _paired_distances(original, first, second)
    layout_distance = _paired_distances(layout, first, second)
    correlation, stress = _distance_metrics(original_distance, layout_distance)
    payload: dict[str, Any] = {
        "reference_space": "original selected input rows",
        "metric_sample_rows": int(count),
        "metric_sample_index_sha256": hashlib.sha256(
            np.ascontiguousarray(indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "neighbors": int(k),
        "trustworthiness": float(trustworthiness(X, Y, n_neighbors=k)),
        "continuity": float(trustworthiness(Y, X, n_neighbors=k)),
        "neighbor_overlap": overlap,
        "distance_pair_count": int(distance_pairs),
        "distance_spearman": correlation,
        "normalized_stress": stress,
        "category_accuracy": None,
        "category_balanced_accuracy": None,
        "category_rows": 0,
        "category_classes": 0,
    }
    if labels is not None:
        sampled_labels = np.asarray(labels)[indices]
        unique, counts = np.unique(sampled_labels, return_counts=True)
        eligible = unique[counts >= 2]
        keep = np.isin(sampled_labels, eligible) & (sampled_labels != "")
        Ys, ys = Y[keep], sampled_labels[keep]
        if np.unique(ys).size >= 2 and ys.size >= 10:
            minimum = int(np.min(np.unique(ys, return_counts=True)[1]))
            stratify = ys if minimum >= 2 else None
            train_X, test_X, train_y, test_y = train_test_split(
                Ys, ys, test_size=0.2, random_state=seed, stratify=stratify
            )
            classifier = KNeighborsClassifier(
                n_neighbors=min(k, train_X.shape[0]), weights="distance", n_jobs=1
            ).fit(train_X, train_y)
            prediction = classifier.predict(test_X)
            payload.update(
                {
                    "category_accuracy": float(accuracy_score(test_y, prediction)),
                    "category_balanced_accuracy": float(
                        balanced_accuracy_score(test_y, prediction)
                    ),
                    "category_rows": int(ys.size),
                    "category_classes": int(np.unique(ys).size),
                }
            )
    return payload


def _reconstruction_metrics(original: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    difference = reconstruction - original
    mse = float(np.mean(difference**2))
    baseline = float(np.mean((original - original.mean(axis=0)) ** 2))
    dot = np.sum(reconstruction * original, axis=1)
    denominator = np.linalg.norm(reconstruction, axis=1) * np.linalg.norm(original, axis=1)
    cosine = float(np.mean(dot / np.maximum(denominator, np.finfo(np.float64).eps)))
    return {
        "reconstruction_mse": mse,
        "reconstruction_nmse": mse / max(baseline, np.finfo(np.float64).eps),
        "reconstruction_cosine": cosine,
    }


class PeakMonitor:
    """Best-effort RSS/NVML process sampler; values are explicitly scoped."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss = 0
        self.gpu_baseline = None
        self.gpu_peak = None
        self._stop = threading.Event()
        self._thread = None
        self._pynvml = None
        self._handle = None
        self._getter = None

    def _gpu_bytes(self):
        if self._getter is None:
            return None
        try:
            for process in self._getter(self._handle):
                if process.pid == os.getpid():
                    return int(process.usedGpuMemory)
        except Exception:
            return None
        return 0

    def _sample(self):
        try:
            import psutil

            process = psutil.Process()
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, int(process.memory_info().rss))
                gpu = self._gpu_bytes()
                if gpu is not None:
                    self.gpu_peak = gpu if self.gpu_peak is None else max(self.gpu_peak, gpu)
                self._stop.wait(self.interval)
        except Exception:
            return

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._getter = getattr(
                pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None
            ) or getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None)
        except Exception:
            self._pynvml = None
        self.gpu_baseline = self._gpu_bytes()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pynvml is not None:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()

    def result(self) -> dict[str, Any]:
        fallback = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        incremental = None
        if self.gpu_baseline is not None and self.gpu_peak is not None:
            incremental = max(0, self.gpu_peak - self.gpu_baseline)
        return {
            "peak_rss_bytes": max(self.peak_rss, fallback),
            "gpu_baseline_process_bytes": self.gpu_baseline,
            "peak_gpu_process_bytes": self.gpu_peak,
            "peak_incremental_gpu_process_bytes": incremental,
            "sample_interval_seconds": self.interval,
        }


def _prime_rapids() -> tuple[dict[str, Any], Any | None]:
    """Import RAPIDS before Torch and separate startup cost from measured fits."""
    started = time.perf_counter()
    versions: dict[str, Any] = {}
    cupy = None
    error = None
    try:
        import cuml
        import cupy as cp

        cupy = cp
        versions.update({"cupy": cp.__version__, "cuml": cuml.__version__})
        versions["cuda_devices"] = int(cp.cuda.runtime.getDeviceCount())
        if versions["cuda_devices"]:
            cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "seconds": time.perf_counter() - started,
        "versions": versions,
        "error": error,
        "policy": "RAPIDS/CuPy imported and initialized before Torch",
    }, cupy


def _build_autoencoder(torch, input_dimension: int, width: int):
    hidden = monotone_hidden_dimensions(input_dimension, width)

    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder = []
            previous = input_dimension
            for dimension in hidden:
                encoder.extend((torch.nn.Linear(previous, dimension), torch.nn.SiLU()))
                previous = dimension
            encoder.append(torch.nn.Linear(previous, width))
            self.encoder = torch.nn.Sequential(*encoder)
            decoder = []
            previous = width
            for dimension in reversed(hidden):
                decoder.extend((torch.nn.Linear(previous, dimension), torch.nn.SiLU()))
                previous = dimension
            decoder.append(torch.nn.Linear(previous, input_dimension))
            self.decoder = torch.nn.Sequential(*decoder)

        def forward(self, batch):
            return self.decoder(self.encoder(batch))

    return Autoencoder(), hidden


def _train_epochs(
    torch,
    model,
    data,
    indices,
    *,
    validation_indices,
    epochs,
    patience,
    min_delta,
    batch_size,
    learning_rate,
    weight_decay,
    seed,
    fixed_epochs=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    best_value = math.inf
    best_epoch = epochs if fixed_epochs is None else fixed_epochs
    stale = 0
    limit = fixed_epochs or epochs
    for epoch in range(1, limit + 1):
        model.train()
        order = indices[torch.randperm(indices.numel(), generator=generator, device=data.device)]
        for start in range(0, order.numel(), batch_size):
            batch = data[order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch) - batch) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("autoencoder produced a non-finite training loss")
            loss.backward()
            optimizer.step()
        if validation_indices is not None:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for start in range(0, validation_indices.numel(), batch_size):
                    batch = data[validation_indices[start : start + batch_size]]
                    total += float(torch.sum((model(batch) - batch) ** 2).item())
                    count += batch.numel()
            value = total / count
            if value < best_value * (1.0 - min_delta):
                best_value, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return int(best_epoch), float(best_value)


def _run_ae_prepass(config: dict[str, Any], cupy) -> dict[str, Any]:
    backend_import_started = time.perf_counter()
    import torch

    backend_import_seconds = time.perf_counter() - backend_import_started
    core_started = time.perf_counter()
    source = np.load(config["input"], mmap_mode="r", allow_pickle=False)
    requested_device = config["device"]
    resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "auto"
        else requested_device
    )
    device = torch.device(resolved_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for AE but is unavailable")
    # Own the host buffer: an NPY read-only mapping otherwise produces a non-writable
    # Torch tensor warning (and would be unsafe if a future training path mutated it).
    host = torch.from_numpy(np.array(source, dtype=np.float32, order="C", copy=True))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    staging_started = time.perf_counter()
    data = host.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    staging_seconds = time.perf_counter() - staging_started

    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(source.shape[0])
    validation_size = min(
        source.shape[0] - 2,
        max(1, int(round(config["ae_validation"] * source.shape[0]))),
    )
    validation = torch.as_tensor(order[:validation_size], dtype=torch.long, device=device)
    training = torch.as_tensor(order[validation_size:], dtype=torch.long, device=device)
    all_indices = torch.arange(source.shape[0], dtype=torch.long, device=device)

    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    tuning, hidden = _build_autoencoder(torch, source.shape[1], config["width"])
    tuning = tuning.to(device)
    selection_started = time.perf_counter()
    selected_epoch, best_validation = _train_epochs(
        torch,
        tuning,
        data,
        training,
        validation_indices=validation,
        epochs=config["ae_epochs"],
        patience=config["ae_patience"],
        min_delta=config["ae_min_delta"],
        batch_size=config["ae_batch_size"],
        learning_rate=config["ae_learning_rate"],
        weight_decay=config["ae_weight_decay"],
        seed=config["seed"],
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    selection_seconds = time.perf_counter() - selection_started
    del tuning

    torch.manual_seed(config["seed"] + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"] + 1)
    model, _ = _build_autoencoder(torch, source.shape[1], config["width"])
    model = model.to(device)
    refit_started = time.perf_counter()
    _train_epochs(
        torch,
        model,
        data,
        all_indices,
        validation_indices=None,
        epochs=config["ae_epochs"],
        patience=config["ae_patience"],
        min_delta=config["ae_min_delta"],
        batch_size=config["ae_batch_size"],
        learning_rate=config["ae_learning_rate"],
        weight_decay=config["ae_weight_decay"],
        seed=config["seed"] + 1,
        fixed_epochs=selected_epoch,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    refit_seconds = time.perf_counter() - refit_started

    output = Path(config["raw_output"])
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    codes = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(source.shape[0], config["width"])
    )
    model.eval()
    encode_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, source.shape[0], config["ae_batch_size"]):
            batch = data[start : start + config["ae_batch_size"]]
            encoded = model.encoder(batch)
            length = batch.shape[0]
            codes[start : start + length] = encoded.float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_started
    codes.flush()
    del codes
    os.replace(temporary, output)
    prepass_core_compute_seconds = time.perf_counter() - core_started

    # Reconstruction quality is useful benchmark evidence, but decoding and these
    # reductions are not part of a production prepass.  Keep them out of the core
    # time and run them in a deliberately separate diagnostic pass.
    diagnostic_started = time.perf_counter()
    reconstruction_sse = 0.0
    reconstruction_cosine = 0.0
    elements = rows = 0
    with torch.no_grad():
        for start in range(0, source.shape[0], config["ae_batch_size"]):
            batch = data[start : start + config["ae_batch_size"]]
            reconstructed = model(batch).float()
            target = batch.float()
            reconstruction_sse += float(torch.sum((reconstructed - target) ** 2).item())
            dot = torch.sum(reconstructed * target, dim=1)
            denominator = (
                torch.linalg.vector_norm(reconstructed, dim=1)
                * torch.linalg.vector_norm(target, dim=1)
            ).clamp_min(1e-12)
            reconstruction_cosine += float(torch.sum(dot / denominator).item())
            elements += batch.numel()
            rows += batch.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    mean = np.zeros(source.shape[1], dtype=np.float64)
    squared = 0.0
    for start in range(0, source.shape[0], config["ae_batch_size"]):
        block = np.asarray(source[start : start + config["ae_batch_size"]], dtype=np.float64)
        mean += block.sum(axis=0)
        squared += float(np.sum(block**2))
    mean /= source.shape[0]
    baseline = max(
        np.finfo(np.float64).eps,
        (squared / source.shape[0] - float(np.dot(mean, mean))) / source.shape[1],
    )
    mse = reconstruction_sse / elements
    reconstruction_diagnostic_seconds = time.perf_counter() - diagnostic_started
    measured_core_stage_seconds = (
        staging_seconds + selection_seconds + refit_seconds + encode_seconds
    )
    return {
        "backend": "torch",
        "backend_import_seconds": backend_import_seconds,
        "device": str(device),
        "hidden_dimensions": list(hidden),
        "architecture_chain": [source.shape[1], *hidden, config["width"]],
        "selected_epochs": selected_epoch,
        "best_validation_mse": best_validation,
        "staging_seconds": staging_seconds,
        "selection_seconds": selection_seconds,
        "refit_seconds": refit_seconds,
        "encoding_seconds": encode_seconds,
        "reconstruction_diagnostic_seconds": reconstruction_diagnostic_seconds,
        "measured_core_stage_seconds": measured_core_stage_seconds,
        "core_unattributed_seconds": max(
            0.0, prepass_core_compute_seconds - measured_core_stage_seconds
        ),
        "prepass_core_compute_seconds": prepass_core_compute_seconds,
        "prepass_compute_seconds": prepass_core_compute_seconds,
        "prepass_diagnostic_inclusive_seconds": (
            prepass_core_compute_seconds + reconstruction_diagnostic_seconds
        ),
        "reconstruction_mse": mse,
        "reconstruction_nmse": mse / baseline,
        "reconstruction_cosine": reconstruction_cosine / rows,
        "torch_version": torch.__version__,
    }


def _run_pca_prepass(config: dict[str, Any], cupy) -> dict[str, Any]:
    source = np.load(config["input"], mmap_mode="r", allow_pickle=False)
    backend_import_started = time.perf_counter()
    use_cuml = cupy is not None and cupy.cuda.runtime.getDeviceCount() >= 1
    if use_cuml:
        from cuml.decomposition import PCA

        reducer = PCA(n_components=config["width"], output_type="numpy")
        backend = "cuml.decomposition.PCA"
    else:
        from sklearn.decomposition import PCA

        reducer = PCA(
            n_components=config["width"],
            svd_solver="randomized",
            random_state=config["seed"],
        )
        backend = "sklearn.decomposition.PCA"
    backend_import_seconds = time.perf_counter() - backend_import_started
    if use_cuml:
        cupy.cuda.runtime.deviceSynchronize()
    fit_started = time.perf_counter()
    codes = np.asarray(reducer.fit_transform(source), dtype=np.float32)
    if use_cuml:
        cupy.cuda.runtime.deviceSynchronize()
    fit_seconds = time.perf_counter() - fit_started
    if use_cuml:
        components = cupy.asnumpy(cupy.asarray(reducer.components_))
        mean = cupy.asnumpy(cupy.asarray(reducer.mean_))
    else:
        components = np.asarray(reducer.components_)
        mean = np.asarray(reducer.mean_)
    _atomic_save_array(config["raw_output"], codes)
    metric_started = time.perf_counter()
    sums = {"sse": 0.0, "cosine": 0.0, "elements": 0, "rows": 0}
    block_rows = config["metric_block_rows"]
    for start in range(0, source.shape[0], block_rows):
        stop = min(source.shape[0], start + block_rows)
        original = np.asarray(source[start:stop], dtype=np.float64)
        reconstructed = np.asarray(codes[start:stop], dtype=np.float64) @ components + mean
        sums["sse"] += float(np.sum((reconstructed - original) ** 2))
        dot = np.sum(reconstructed * original, axis=1)
        denominator = np.linalg.norm(reconstructed, axis=1) * np.linalg.norm(original, axis=1)
        sums["cosine"] += float(np.sum(dot / np.maximum(denominator, np.finfo(np.float64).eps)))
        sums["elements"] += original.size
        sums["rows"] += original.shape[0]
    mean_original = np.zeros(source.shape[1], dtype=np.float64)
    for start in range(0, source.shape[0], block_rows):
        mean_original += np.sum(
            np.asarray(source[start : start + block_rows], dtype=np.float64), axis=0
        )
    mean_original /= source.shape[0]
    baseline = 0.0
    for start in range(0, source.shape[0], block_rows):
        block = np.asarray(source[start : start + block_rows], dtype=np.float64)
        baseline += float(np.sum((block - mean_original) ** 2))
    baseline /= source.size
    metric_seconds = time.perf_counter() - metric_started
    mse = sums["sse"] / sums["elements"]
    return {
        "backend": backend,
        "backend_import_seconds": backend_import_seconds,
        "fit_seconds": fit_seconds,
        "reconstruction_metric_seconds": metric_seconds,
        "prepass_core_compute_seconds": fit_seconds,
        "prepass_compute_seconds": fit_seconds,
        "reconstruction_diagnostic_seconds": metric_seconds,
        "prepass_diagnostic_inclusive_seconds": fit_seconds + metric_seconds,
        "reconstruction_mse": mse,
        "reconstruction_nmse": mse / max(baseline, np.finfo(np.float64).eps),
        "reconstruction_cosine": sums["cosine"] / sums["rows"],
    }


def _run_graph(config: dict[str, Any], cupy) -> dict[str, Any]:
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("DiRe/UMAP graph worker requires a visible CUDA device")
    X = np.load(config["input"], mmap_mode="r", allow_pickle=False)
    import_started = time.perf_counter()
    if config["method"] == "umap":
        from cuml.manifold import UMAP

        reducer = UMAP(
            n_components=config["final_dimension"],
            n_neighbors=config["neighbors"],
            min_dist=config["umap_min_dist"],
            spread=config["umap_spread"],
            metric="euclidean",
            init="random",
            random_state=config["seed"],
            build_algo="nn_descent",
            force_serial_epochs=True,
            output_type="numpy",
            verbose=False,
        )
        backend = "cuml.manifold.UMAP"
        details = {
            "build_algo": "nn_descent",
            "init": "random",
            "metric": "euclidean",
            "nn_descent_exactly_deterministic": False,
        }
    else:
        repository = Path(config["dire_repo"]).expanduser().resolve()
        if not repository.is_dir():
            raise FileNotFoundError(f"DiRe repository does not exist: {repository}")
        sys.path.insert(0, str(repository))
        import dire_rapids

        if not Path(dire_rapids.__file__).resolve().is_relative_to(repository):
            raise RuntimeError("imported DiRe outside requested repository")
        reducer = dire_rapids.create_dire(
            backend="cuvs",
            n_components=config["final_dimension"],
            n_neighbors=config["neighbors"],
            init="random",
            max_iter_layout=config["dire_iterations"],
            metric=None,
            normalize=False,
            cuvs_index_type="ivf_flat",
            random_state=config["seed"],
            verbose=False,
        )
        backend = f"{type(reducer).__module__}.{type(reducer).__name__}"
        implementation = Path(inspect.getfile(dire_rapids.create_dire)).resolve()
        details = {
            "dire_repo": str(repository),
            "dire_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None,
            "implementation_sha256": _sha256_file(implementation),
            "cuvs_index": "ivf_flat",
            "init": "random",
            "init_note": (
                "fixed seed and common init mode; DiRe implements random init as an "
                "input-dependent random projection, so coordinates are not identical to "
                "UMAP's input-independent random initialization"
            ),
            "metric": "sqeuclidean",
            "normalize": False,
        }
    backend_import_seconds = time.perf_counter() - import_started
    cupy.cuda.runtime.deviceSynchronize()
    fit_started = time.perf_counter()
    layout = np.asarray(reducer.fit_transform(X), dtype=np.float32)
    cupy.cuda.runtime.deviceSynchronize()
    fit_seconds = time.perf_counter() - fit_started
    if layout.shape != (X.shape[0], config["final_dimension"]) or not np.all(np.isfinite(layout)):
        raise RuntimeError("graph reducer returned an invalid layout")
    _atomic_save_array(config["layout_output"], layout)
    return {
        "backend": backend,
        "backend_import_seconds": backend_import_seconds,
        "fit_seconds": fit_seconds,
        "details": details,
    }


def run_worker(config_path: str | Path, result_path: str | Path) -> int:
    started_at = _utc_now()
    worker_started = time.perf_counter()
    config = json.loads(Path(config_path).read_text())
    startup, cupy = _prime_rapids()
    monitor = PeakMonitor(config.get("resource_sample_interval", 0.05))
    try:
        with monitor:
            if config["stage"] == "prepass":
                details = (
                    _run_ae_prepass(config, cupy)
                    if config["kind"] == "ae"
                    else _run_pca_prepass(config, cupy)
                )
                artifact = config["raw_output"]
            elif config["stage"] == "graph":
                details = _run_graph(config, cupy)
                artifact = config["layout_output"]
            else:
                raise ValueError(f"unknown worker stage {config['stage']!r}")
        array = np.load(artifact, mmap_mode="r", allow_pickle=False)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "error": None,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "config_identity": configuration_identity(config),
            "stage": config["stage"],
            "startup": startup,
            "details": details,
            "artifact": {
                "path": str(Path(artifact).resolve()),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "logical_sha256": _logical_float32_sha256(array),
                "file_sha256": _sha256_file(artifact),
            },
            "resources": monitor.result(),
            "worker_total_seconds": time.perf_counter() - worker_started,
            "python": platform.python_version(),
        }
        _atomic_json(result_path, payload)
        return 0
    except BaseException as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        _atomic_json(
            result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "config_identity": configuration_identity(config),
                "stage": config.get("stage"),
                "startup": startup,
                "resources": monitor.result(),
                "worker_total_seconds": time.perf_counter() - worker_started,
            },
        )
        return 1


def _launch_worker(
    config: dict[str, Any],
    *,
    config_path: Path,
    result_path: Path,
    log_path: Path,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    _atomic_json(config_path, config)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        str(config_path),
        "--worker-result",
        str(result_path),
    ]
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=None if timeout <= 0 else timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"worker exceeded {timeout} seconds; see {log_path}") from exc
    wall = time.perf_counter() - started
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"worker produced no valid result; see {log_path}") from exc
    if completed.returncode != 0 or payload.get("status") != "ok":
        raise RuntimeError(f"worker failed: {payload.get('error')}; see {log_path}")
    if payload.get("config_identity") != configuration_identity(config):
        raise RuntimeError("worker result configuration identity mismatch")
    return payload, wall


@dataclass(frozen=True)
class PreparedSubset:
    data_path: str
    labels_path: str | None
    rows: int
    features: int
    data_logical_sha256: str
    data_file_sha256: str
    indices_sha256: str
    source_snapshot: dict[str, Any]
    labels_source_snapshot: dict[str, Any] | None
    labels_file_sha256: str | None


def dire_repository_identity(path: str | Path) -> dict[str, Any]:
    """Capture the current DiRe source identity without importing Torch."""
    repository = Path(path).expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"DiRe repository does not exist: {repository}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_files = [
        repository / "dire_rapids" / "dire_pytorch.py",
        repository / "dire_rapids" / "dire_cuvs.py",
        repository / "dire_rapids" / "__init__.py",
    ]
    missing = [str(source) for source in source_files if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"DiRe source files are missing: {missing}")
    return {
        "path": str(repository),
        "git_revision": revision or None,
        "source_sha256": {
            str(source.relative_to(repository)): _sha256_file(source) for source in source_files
        },
    }


def _load_labels(path: str | Path, *, allow_pickled: bool, expected_rows: int) -> np.ndarray:
    try:
        labels = np.load(path, allow_pickle=allow_pickled)
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                "object-label NPY requires --allow-pickled-labels; only enable it for a trusted file"
            ) from exc
        raise
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != expected_rows:
        raise ValueError("label NPY must be one-dimensional and aligned with source rows")
    if labels.dtype.hasobject and not allow_pickled:
        raise ValueError("object labels require explicit pickle opt-in")
    return labels


def prepare_subset(args: argparse.Namespace) -> PreparedSubset:
    source_path = Path(args.data).expanduser().resolve()
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.dtype != np.float32 or source.shape[0] < 2:
        raise ValueError("--data must be a two-dimensional float32 NPY")
    rows = source.shape[0] if args.rows is None else args.rows
    indices = deterministic_indices(source.shape[0], rows, args.subset_seed)
    indices_sha = hashlib.sha256(np.ascontiguousarray(indices, dtype="<i8").tobytes()).hexdigest()
    snapshot = source_snapshot(source_path)
    subset_dir = Path(args.output_dir) / "subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    data_path = subset_dir / "original.npy"
    labels_path = subset_dir / "labels.npy" if args.labels else None
    manifest_path = subset_dir / "manifest.json"
    labels_snapshot = source_snapshot(args.labels) if args.labels else None
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot": snapshot,
        "rows": rows,
        "subset_seed": args.subset_seed,
        "indices_sha256": indices_sha,
        "labels_snapshot": labels_snapshot,
        "allow_pickled_labels": bool(args.allow_pickled_labels),
    }
    identity = configuration_identity(identity_payload)
    if manifest_path.exists() and data_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise RuntimeError(
                "cached subset identity differs; use a new output directory or --force"
            )
        prepared = PreparedSubset(**manifest["prepared"])
        if prepared.data_file_sha256 != _sha256_file(data_path):
            raise RuntimeError("cached selected-data file hash differs; pass --force to rebuild")
        if prepared.labels_path:
            if not Path(prepared.labels_path).is_file():
                raise RuntimeError("cached selected-label file is missing; pass --force to rebuild")
            if prepared.labels_file_sha256 != _sha256_file(prepared.labels_path):
                raise RuntimeError(
                    "cached selected-label file hash differs; pass --force to rebuild"
                )
        return prepared

    output = np.lib.format.open_memmap(
        data_path.with_suffix(".npy.tmp"),
        mode="w+",
        dtype=np.float32,
        shape=(rows, source.shape[1]),
    )
    digest = hashlib.sha256()
    temporary = Path(f"{data_path}.tmp")
    try:
        for start in range(0, rows, args.block_rows):
            block = np.ascontiguousarray(
                source[indices[start : start + args.block_rows]], dtype="<f4"
            )
            if not np.all(np.isfinite(block)):
                raise ValueError("selected source rows contain non-finite values")
            output[start : start + block.shape[0]] = block
            digest.update(block.tobytes(order="C"))
        output.flush()
        completed_output = output
        output = None
        del completed_output
        os.replace(temporary, data_path)
    except BaseException:
        if output is not None:
            with contextlib.suppress(Exception):
                del output
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    if args.labels:
        labels = _load_labels(
            args.labels,
            allow_pickled=args.allow_pickled_labels,
            expected_rows=source.shape[0],
        )
        _atomic_save_array(labels_path, labels[indices])
    prepared = PreparedSubset(
        data_path=str(data_path.resolve()),
        labels_path=str(labels_path.resolve()) if labels_path else None,
        rows=int(rows),
        features=int(source.shape[1]),
        data_logical_sha256=digest.hexdigest(),
        data_file_sha256=_sha256_file(data_path),
        indices_sha256=indices_sha,
        source_snapshot=snapshot,
        labels_source_snapshot=labels_snapshot,
        labels_file_sha256=_sha256_file(labels_path) if labels_path else None,
    )
    _atomic_json(manifest_path, {"identity": identity, "prepared": prepared.__dict__})
    return prepared


def _cache_ok(result_path: Path, config: dict[str, Any], artifact_path: Path) -> dict | None:
    if not result_path.exists() or not artifact_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text())
        artifact = payload["artifact"]
        array = np.load(artifact_path, mmap_mode="r", allow_pickle=False)
        if (
            payload.get("status") != "ok"
            or payload.get("config_identity") != configuration_identity(config)
            or artifact.get("shape") != list(array.shape)
            or artifact.get("dtype") != str(array.dtype)
            or artifact.get("file_sha256") != _sha256_file(artifact_path)
        ):
            return None
        return payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _prepass_specs(args: argparse.Namespace, seed: int):
    for kind in args.prepasses:
        if kind == "direct":
            yield kind, None
        else:
            for width in args.prepass_widths:
                yield kind, width


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        record = {}
        for key, value in row.items():
            record[key] = (
                json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
            )
        flattened.append(record)
    fields = sorted({key for row in flattened for key in row})
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    os.replace(temporary, path)


def orchestrate(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_subset(args)
    original = np.load(prepared.data_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(prepared.labels_path, allow_pickle=True) if prepared.labels_path else None
    harness_sha256 = _sha256_file(Path(__file__).resolve())
    dire_identity = dire_repository_identity(args.dire_repo) if "dire" in args.methods else None
    if any(width >= prepared.features for width in args.prepass_widths):
        raise ValueError("every prepass width must be below the input width")
    if any(final >= min(args.prepass_widths) for final in args.final_dimensions) and any(
        kind != "direct" for kind in args.prepasses
    ):
        raise ValueError("final dimensions must be below every compressed prepass width")

    rows: list[dict[str, Any]] = []
    prepass_records: list[dict[str, Any]] = []
    computed_prepasses: dict[tuple[str, int, int], tuple[dict[str, Any], float]] = {}
    for seed in args.seeds:
        for kind, width in _prepass_specs(args, seed):
            variant = PREPASS_VARIANTS[kind]
            name = f"{variant}-s{seed}" if kind == "direct" else f"{variant}-d{width}-s{seed}"
            directory = output / "prepasses" / name
            directory.mkdir(parents=True, exist_ok=True)
            if kind == "direct":
                intermediate_path = Path(prepared.data_path)
                prepass = {
                    "status": "ok",
                    "kind": variant,
                    "cli_kind": kind,
                    "width": prepared.features,
                    "seed": seed,
                    "artifact": {
                        "path": prepared.data_path,
                        "logical_sha256": prepared.data_logical_sha256,
                    },
                    "details": {
                        "prepass_core_compute_seconds": 0.0,
                        "prepass_compute_seconds": 0.0,
                        "reconstruction_diagnostic_seconds": 0.0,
                        "prepass_diagnostic_inclusive_seconds": 0.0,
                        "reconstruction_mse": 0.0,
                        "reconstruction_nmse": 0.0,
                        "reconstruction_cosine": 1.0,
                    },
                    "canonicalization": {"seconds": 0.0, "transform": "none"},
                    "worker_total_seconds": 0.0,
                    "worker_launch_wall_seconds": 0.0,
                }
            else:
                base_kind = "pca" if kind == "pca_raw" else kind
                # Raw PCA scores and canonical PCA coordinates deliberately share
                # one fitted PCA artifact.  This both avoids duplicate work and
                # removes a possible fit-to-fit confound in the metric comparison.
                compute_name = f"{base_kind}-d{width}-s{seed}"
                compute_directory = output / "prepass-compute" / compute_name
                compute_directory.mkdir(parents=True, exist_ok=True)
                raw_path = compute_directory / "raw.npy"
                worker_result = compute_directory / "worker-result.json"
                config_path = compute_directory / "worker-config.json"
                log_path = compute_directory / "worker.log"
                config = {
                    "schema_version": SCHEMA_VERSION,
                    "harness_sha256": harness_sha256,
                    "stage": "prepass",
                    "kind": base_kind,
                    "input": prepared.data_path,
                    "input_logical_sha256": prepared.data_logical_sha256,
                    "width": width,
                    "seed": seed,
                    "raw_output": str(raw_path.resolve()),
                    "device": args.device,
                    "ae_epochs": args.ae_epochs,
                    "ae_patience": args.ae_patience,
                    "ae_validation": args.ae_validation,
                    "ae_min_delta": args.ae_min_delta,
                    "ae_batch_size": args.ae_batch_size,
                    "ae_learning_rate": args.ae_learning_rate,
                    "ae_weight_decay": args.ae_weight_decay,
                    "metric_block_rows": args.block_rows,
                    "resource_sample_interval": args.resource_sample_interval,
                }
                compute_key = (base_kind, int(width), seed)
                if compute_key in computed_prepasses:
                    worker, launch_wall = computed_prepasses[compute_key]
                else:
                    worker = None if args.force else _cache_ok(worker_result, config, raw_path)
                    if worker is None:
                        worker, launch_wall = _launch_worker(
                            config,
                            config_path=config_path,
                            result_path=worker_result,
                            log_path=log_path,
                            timeout=args.timeout,
                        )
                    else:
                        launch_wall = worker["worker_total_seconds"]
                    computed_prepasses[compute_key] = (worker, launch_wall)
                if kind == "pca_raw":
                    intermediate_path = raw_path
                    canonical = {
                        "seconds": 0.0,
                        "transform": "none; raw centered PCA scores",
                    }
                    intermediate_artifact = {
                        "path": str(raw_path.resolve()),
                        "logical_sha256": worker["artifact"]["logical_sha256"],
                        "file_sha256": worker["artifact"]["file_sha256"],
                    }
                else:
                    canonical_path = directory / "canonical.npy"
                    canonical_manifest_path = directory / "canonical.json"
                    canonical_identity = configuration_identity(
                        {
                            "raw_logical_sha256": worker["artifact"]["logical_sha256"],
                            "harness_sha256": harness_sha256,
                            "relative_eigenvalue_floor": args.whitening_floor,
                            "block_rows": args.block_rows,
                        }
                    )
                    canonical = None
                    if (
                        canonical_manifest_path.exists()
                        and canonical_path.exists()
                        and not args.force
                    ):
                        candidate = json.loads(canonical_manifest_path.read_text())
                        if candidate.get("identity") == canonical_identity and candidate.get(
                            "file_sha256"
                        ) == _sha256_file(canonical_path):
                            canonical = candidate
                    if canonical is None:
                        canonical = canonicalize_npy(
                            raw_path,
                            canonical_path,
                            relative_eigenvalue_floor=args.whitening_floor,
                            block_rows=args.block_rows,
                        )
                        canonical.update(
                            {
                                "identity": canonical_identity,
                                "file_sha256": _sha256_file(canonical_path),
                            }
                        )
                        _atomic_json(canonical_manifest_path, canonical)
                    intermediate_path = canonical_path
                    intermediate_artifact = {
                        "path": str(canonical_path.resolve()),
                        "logical_sha256": canonical["logical_sha256"],
                        "file_sha256": canonical["file_sha256"],
                    }
                prepass = {
                    **worker,
                    "kind": variant,
                    "cli_kind": kind,
                    "width": width,
                    "seed": seed,
                    "canonicalization": canonical,
                    "intermediate_artifact": intermediate_artifact,
                    "worker_launch_wall_seconds": launch_wall,
                }
            intermediate = np.load(intermediate_path, mmap_mode="r", allow_pickle=False)
            prepass["intermediate_fidelity"] = evaluate_against_original(
                original,
                intermediate,
                metric_rows=args.metric_rows,
                neighbors=args.neighbors,
                distance_pairs=args.distance_pairs,
                seed=args.evaluation_seed,
                labels=labels,
            )
            _atomic_json(directory / "result.json", prepass)
            prepass_records.append(prepass)

            for final_dimension in args.final_dimensions:
                for method in args.methods:
                    cell_name = f"{name}-{method}-d{final_dimension}"
                    cell_dir = output / "cells" / cell_name
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    layout_path = cell_dir / "layout.npy"
                    result_path = cell_dir / "worker-result.json"
                    config_path = cell_dir / "worker-config.json"
                    log_path = cell_dir / "worker.log"
                    graph_config = {
                        "schema_version": SCHEMA_VERSION,
                        "harness_sha256": harness_sha256,
                        "stage": "graph",
                        "method": method,
                        "input": str(intermediate_path.resolve()),
                        "input_logical_sha256": (
                            prepared.data_logical_sha256
                            if kind == "direct"
                            else prepass["intermediate_artifact"]["logical_sha256"]
                        ),
                        "layout_output": str(layout_path.resolve()),
                        "final_dimension": final_dimension,
                        "seed": seed,
                        "neighbors": args.neighbors,
                        "dire_repo": str(Path(args.dire_repo).expanduser().resolve()),
                        "dire_repository_identity": dire_identity,
                        "dire_iterations": args.dire_iterations,
                        "umap_min_dist": args.umap_min_dist,
                        "umap_spread": args.umap_spread,
                        "resource_sample_interval": args.resource_sample_interval,
                    }
                    graph = (
                        None if args.force else _cache_ok(result_path, graph_config, layout_path)
                    )
                    if graph is None:
                        graph, graph_launch_wall = _launch_worker(
                            graph_config,
                            config_path=config_path,
                            result_path=result_path,
                            log_path=log_path,
                            timeout=args.timeout,
                        )
                    else:
                        graph_launch_wall = graph["worker_total_seconds"]
                    layout = np.load(layout_path, mmap_mode="r", allow_pickle=False)
                    metrics = evaluate_against_original(
                        original,
                        layout,
                        metric_rows=args.metric_rows,
                        neighbors=args.neighbors,
                        distance_pairs=args.distance_pairs,
                        seed=args.evaluation_seed,
                        labels=labels,
                    )
                    prepass_compute = float(prepass["details"]["prepass_compute_seconds"])
                    prepass_diagnostic = float(
                        prepass["details"].get("reconstruction_diagnostic_seconds", 0.0)
                    )
                    canonical_seconds = float(prepass["canonicalization"]["seconds"])
                    timing = timing_accounting(
                        prepass_compute,
                        canonical_seconds,
                        float(graph["details"]["fit_seconds"]),
                        prepass_diagnostic_seconds=prepass_diagnostic,
                        prepass_worker_wall_seconds=float(
                            prepass.get("worker_launch_wall_seconds", 0.0)
                        ),
                        graph_worker_wall_seconds=float(graph_launch_wall),
                    )
                    row = {
                        "dataset": args.dataset_name,
                        "rows": prepared.rows,
                        "features": prepared.features,
                        "seed": seed,
                        "prepass_kind": variant,
                        "prepass_cli_kind": kind,
                        "prepass_transform": prepass["canonicalization"]["transform"],
                        "prepass_width": None if kind == "direct" else width,
                        "method": method,
                        "final_dimension": final_dimension,
                        "status": "ok",
                        "timing": timing,
                        "metrics": metrics,
                        "intermediate_fidelity": prepass["intermediate_fidelity"],
                        "reconstruction": {
                            key: prepass["details"][key]
                            for key in (
                                "reconstruction_mse",
                                "reconstruction_nmse",
                                "reconstruction_cosine",
                            )
                        },
                        "prepass_result": str((directory / "result.json").resolve()),
                        "graph_result": str(result_path.resolve()),
                        "layout": str(layout_path.resolve()),
                    }
                    _atomic_json(cell_dir / "result.json", row)
                    rows.append(row)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "protocol": (
            "cached transductive prepass; fresh downstream workers; final quality always "
            "against original selected input"
        ),
        "prepared": prepared.__dict__,
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key not in {"worker_config", "worker_result", "force"}
        },
        "prepasses": prepass_records,
        "rows": rows,
    }
    _atomic_json(output / "results.json", payload)
    _write_csv(output / "results.csv", rows)
    print(f"wrote {output / 'results.json'} ({len(rows)} successful cells)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-name", default="prepared-embeddings")
    parser.add_argument("--data")
    parser.add_argument("--labels")
    parser.add_argument("--allow-pickled-labels", action="store_true")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--subset-seed", type=int, default=1729)
    parser.add_argument("--evaluation-seed", type=int, default=2718)
    parser.add_argument(
        "--prepasses", nargs="+", choices=PREPASS_KINDS, default=list(PREPASS_KINDS)
    )
    parser.add_argument("--prepass-widths", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--methods", nargs="+", choices=GRAPH_METHODS, default=list(GRAPH_METHODS))
    parser.add_argument("--final-dimensions", nargs="+", type=int, default=[8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--metric-rows", type=int, default=5000)
    parser.add_argument("--distance-pairs", type=int, default=200_000)
    parser.add_argument("--whitening-floor", type=float, default=1e-6)
    parser.add_argument("--block-rows", type=int, default=16_384)
    parser.add_argument("--output-dir", default="artifacts/prepass-reducer-comparison")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resource-sample-interval", type=float, default=0.05)
    parser.add_argument("--dire-repo", default="../dire-rapids")
    parser.add_argument("--dire-iterations", type=int, default=128)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-spread", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--ae-patience", type=int, default=15)
    parser.add_argument("--ae-validation", type=float, default=0.05)
    parser.add_argument("--ae-min-delta", type=float, default=1e-5)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.worker_config or args.worker_result:
        if not args.worker_config or not args.worker_result:
            parser.error("internal worker mode requires config and result")
        return args
    if not args.data:
        parser.error("--data is required")
    positive = [
        *args.prepass_widths,
        *args.final_dimensions,
        args.neighbors,
        args.metric_rows,
        args.distance_pairs,
        args.block_rows,
        args.dire_iterations,
        args.ae_epochs,
        args.ae_patience,
        args.ae_batch_size,
    ]
    if any(value < 1 for value in positive):
        parser.error("dimensions, counts, and iteration controls must be positive")
    if args.rows is not None and args.rows < 2:
        parser.error("--rows must be at least two")
    if len(set(args.prepasses)) != len(args.prepasses):
        parser.error("prepass kinds must be unique")
    if len(set(args.prepass_widths)) != len(args.prepass_widths):
        parser.error("prepass widths must be unique")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    if not 0 < args.whitening_floor < 1:
        parser.error("whitening floor must be in (0,1)")
    if not 0 < args.ae_validation < 0.5:
        parser.error("AE validation fraction must be in (0,0.5)")
    if args.timeout < 0 or args.resource_sample_interval <= 0:
        parser.error("timeout must be nonnegative and sample interval positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        return run_worker(args.worker_config, args.worker_result)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
