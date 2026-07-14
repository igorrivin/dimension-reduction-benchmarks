#!/usr/bin/env python3
"""Scale-safe AE/DiRe/cuML UMAP/PCA comparison on embedding corpora.

The primary protocol is transductive: each reducer fits and embeds the same complete
matrix at a fixed target dimension.  The autoencoder is trained from scratch for every
dataset/size/dimension/seed.  A 5% validation split selects its epoch count, after which a
fresh model is refit on all rows.  Thus its headline layout, like DiRe and UMAP, has seen
the complete dataset.  Held-out encoder generalization is deliberately not claimed.

Unlike :mod:`experiments.compare_fixed_dimension`, this harness avoids full-corpus
quadratic metrics and native inverse transforms.  Quality metrics are computed on fixed,
auditable samples; their names and metadata explicitly describe that conditioning.

Examples
--------
Local arXiv corpus (the metadata labels are optional)::

    conda run -n rapids-26.06 python experiments/compare_large_embeddings.py \
      --dataset-name arxiv \
      --data ../dire-rapids-arxiv/data/embeddings.npy \
      --labels ../dire-rapids-arxiv/data/meta.parquet \
      --label-column primary_category --sizes 50000 200000 723457 \
      --target-dimensions 2 8 --methods autoencoder dire umap pca \
      --dire-repo /tmp/dire-rapids-origin-f7d5736 \
      --dire-revision f7d57367af8e10f6013e64026e2dbb0af6781947 \
      --seeds 0 1 2 --output-dir artifacts/large-arxiv

Cached Cohere Wikipedia shards::

    conda run -n rapids-26.06 python experiments/compare_large_embeddings.py \
      --dataset-name wikipedia --data /tmp/cohere_en_emb --shard-glob 'emb_*.npy' \
      --labels /path/to/wiki-metadata --label-shard-glob 'meta-*.parquet' \
      --label-column url --label-policy group \
      --sizes 100000 250000 1000000 --target-dimensions 2 8 \
      --methods autoencoder dire umap pca --output-dir artifacts/large-wikipedia

Every reducer runs in a fresh subprocess.  Dataset loading/preparation is timed
separately from fitting.  Partial worker results are written atomically and successful
runs are resumable unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

METHODS = ("autoencoder", "dire", "umap", "pca")


@dataclass(frozen=True)
class SourceInfo:
    files: tuple[str, ...]
    rows: int
    features: int
    dtype: str
    fingerprint: str


@dataclass(frozen=True)
class PreparedInfo:
    path: str
    rows: int
    features: int
    source_index_sha256: str
    normalized_sha256: str
    source_fingerprint: str | None
    zero_rows: int
    preparation_seconds: float


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    indices = np.ascontiguousarray(indices, dtype="<i8")
    return _sha256_bytes(indices.tobytes())


def file_fingerprint(paths: list[Path]) -> str:
    """Hash stable file metadata without rereading multi-gigabyte source arrays."""
    records = []
    for path in paths:
        stat = path.stat()
        records.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return _sha256_bytes(json.dumps(records, sort_keys=True).encode())


def discover_sources(
    data: str | Path, shard_glob: str = "emb_*.npy"
) -> tuple[list[Path], SourceInfo]:
    data = Path(data).expanduser().resolve()
    if data.is_file():
        paths = [data]
    elif data.is_dir():
        paths = sorted(data.glob(shard_glob))
    else:
        paths = sorted(Path().glob(str(data))) if not data.is_absolute() else []
    if not paths:
        raise FileNotFoundError(f"no NPY inputs found for {data} ({shard_glob!r})")

    rows = 0
    features = None
    dtype = None
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"{path} is not a two-dimensional array")
        if features is None:
            features = int(array.shape[1])
            dtype = array.dtype
        if array.shape[1] != features or array.dtype != dtype:
            raise ValueError("all NPY shards must have identical feature widths and dtypes")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("embedding arrays must have a floating dtype")
        rows += int(array.shape[0])
    info = SourceInfo(
        files=tuple(str(path) for path in paths),
        rows=rows,
        features=int(features),
        dtype=str(dtype),
        fingerprint=file_fingerprint(paths),
    )
    return paths, info


def deterministic_nested_indices(total: int, sizes, seed: int) -> dict[int, np.ndarray]:
    """Return sorted samples nested inside one deterministic random permutation."""
    sizes = tuple(sorted(set(int(size) for size in sizes)))
    if total < 1 or not sizes or sizes[0] < 2 or sizes[-1] > total:
        raise ValueError("sizes must be unique integers in [2, total]")
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    return {size: np.sort(order[:size]) for size in sizes}


def row_l2_normalize(block, *, return_zero_count: bool = False):
    """L2-normalize float rows without promoting a complete corpus to float64."""
    block = np.asarray(block, dtype=np.float32)
    if block.ndim != 2:
        raise ValueError("embedding block must be two-dimensional")
    if not np.all(np.isfinite(block)):
        raise ValueError("embedding block contains non-finite values")
    # The float64 accumulation avoids norm overflow while only allocating O(rows).
    norms = np.sqrt(np.sum(block.astype(np.float64) ** 2, axis=1))
    zeros = norms <= np.finfo(np.float32).tiny
    safe = norms.copy()
    safe[zeros] = 1.0
    normalized = np.ascontiguousarray(block / safe[:, None], dtype=np.float32)
    if return_zero_count:
        return normalized, int(np.count_nonzero(zeros))
    return normalized


def _shard_offsets(paths: list[Path]) -> np.ndarray:
    counts = [np.load(path, mmap_mode="r", allow_pickle=False).shape[0] for path in paths]
    return np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))


def _read_selected(paths: list[Path], offsets: np.ndarray, selected: np.ndarray, start, stop):
    wanted = selected[start:stop]
    width = np.load(paths[0], mmap_mode="r", allow_pickle=False).shape[1]
    block = np.empty((wanted.size, width), dtype=np.float32)
    shard_ids = np.searchsorted(offsets[1:], wanted, side="right")
    for shard_id in np.unique(shard_ids):
        positions = np.flatnonzero(shard_ids == shard_id)
        local = wanted[positions] - offsets[shard_id]
        shard = np.load(paths[shard_id], mmap_mode="r", allow_pickle=False)
        block[positions] = shard[local]
    return block


def prepare_normalized_data(
    paths: list[Path],
    selected,
    destination: str | Path,
    *,
    block_rows: int = 16_384,
    source_fingerprint: str | None = None,
) -> PreparedInfo:
    """Select and normalize rows into a reusable float32 NPY, block by block."""
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 2 or np.any(np.diff(selected) <= 0):
        raise ValueError("selected indices must be a sorted unique one-dimensional array")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    offsets = _shard_offsets(paths)
    if selected[0] < 0 or selected[-1] >= offsets[-1]:
        raise IndexError("selected source index is out of bounds")
    width = np.load(paths[0], mmap_mode="r", allow_pickle=False).shape[1]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(selected.size, width)
    )
    digest = hashlib.sha256()
    zero_rows = 0
    started = time.perf_counter()
    try:
        for start in range(0, selected.size, block_rows):
            stop = min(start + block_rows, selected.size)
            source = _read_selected(paths, offsets, selected, start, stop)
            normalized, zeros = row_l2_normalize(source, return_zero_count=True)
            output[start:stop] = normalized
            digest.update(normalized.tobytes(order="C"))
            zero_rows += zeros
        output.flush()
        del output
        os.replace(temporary, destination)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    return PreparedInfo(
        path=str(destination.resolve()),
        rows=int(selected.size),
        features=int(width),
        source_index_sha256=index_sha256(selected),
        normalized_sha256=digest.hexdigest(),
        source_fingerprint=source_fingerprint,
        zero_rows=zero_rows,
        preparation_seconds=time.perf_counter() - started,
    )


def discover_label_sources(path: str | Path, shard_glob: str = "*.parquet") -> list[Path]:
    path = Path(path).expanduser().resolve()
    paths = [path] if path.is_file() else sorted(path.glob(shard_glob)) if path.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no label inputs found for {path} ({shard_glob!r})")
    return paths


def load_labels(
    path: str | Path,
    column: str | None,
    expected_rows: int,
    shard_glob: str = "*.parquet",
):
    paths = discover_label_sources(path, shard_glob)
    arrays = []
    for source in paths:
        if source.suffix == ".npy":
            values = np.load(source, allow_pickle=True)
        elif source.suffix in {".parquet", ".pq"}:
            if not column:
                raise ValueError("--label-column is required for Parquet labels")
            import pandas as pd

            values = pd.read_parquet(source, columns=[column])[column].fillna("").to_numpy()
        else:
            raise ValueError("labels must be NPY or Parquet files")
        arrays.append(np.asarray(values))
    labels = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
    if labels.ndim != 1 or labels.shape[0] != expected_rows:
        raise ValueError("labels must be one-dimensional and aligned with source rows")
    return labels


def label_provenance(path: str | Path, column: str | None, shard_glob: str):
    paths = discover_label_sources(path, shard_glob)
    if any(source.suffix in {".parquet", ".pq"} for source in paths):
        if not column:
            raise ValueError("--label-column is required for Parquet labels")
    return {
        "files": [str(source) for source in paths],
        "column": column,
        "fingerprint": file_fingerprint(paths),
    }


def _cuda_synchronize(torch=None, cupy=None, device=None) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif cupy is not None:
        cupy.cuda.runtime.deviceSynchronize()


class PeakMonitor:
    """Best-effort process RSS and NVML GPU-memory sampler."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_rss_bytes = 0
        self.peak_gpu_bytes = None
        self.gpu_baseline_bytes = None
        self._stop = threading.Event()
        self._thread = None
        self._pynvml = None
        self._nvml_handle = None
        self._nvml_getter = None

    def _initialize_nvml(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_getter = getattr(
                pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None
            ) or getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None)
        except Exception:
            self._pynvml = None
            self._nvml_handle = None
            self._nvml_getter = None

    def _gpu_process_bytes(self):
        if self._nvml_getter is None:
            return None
        try:
            for process in self._nvml_getter(self._nvml_handle):
                if process.pid == os.getpid():
                    value = int(process.usedGpuMemory)
                    return None if value < 0 else value
        except Exception:
            return None
        return 0

    def _sample(self):
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

    def result(self):
        fallback_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        peak = max(self.peak_rss_bytes, fallback_rss)
        incremental = None
        if self.peak_gpu_bytes is not None and self.gpu_baseline_bytes is not None:
            incremental = max(0, self.peak_gpu_bytes - self.gpu_baseline_bytes)
        return {
            "peak_rss_bytes": peak,
            "gpu_baseline_bytes": self.gpu_baseline_bytes,
            "peak_gpu_process_bytes": self.peak_gpu_bytes,
            "peak_incremental_gpu_bytes": incremental,
        }


def _save_layout_atomic(path: Path, layout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(layout, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _worker_hardware():
    info = {"machine": platform.machine(), "platform": platform.platform()}
    cupy = sys.modules.get("cupy")
    try:
        if cupy is not None and cupy.cuda.runtime.getDeviceCount():
            properties = cupy.cuda.runtime.getDeviceProperties(0)
            name = properties.get("name", properties.get(b"name", "unknown"))
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            _, total = cupy.cuda.Device(0).mem_info
            info.update(
                {
                    "gpu_name": str(name),
                    "gpu_total_memory_bytes": int(total),
                    "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
                    "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
                }
            )
            return info
    except Exception:
        pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "gpu_name": properties.name,
                    "gpu_total_memory_bytes": int(properties.total_memory),
                    "cuda_compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    except Exception:
        pass
    return info


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_revision(actual: str | None, expected: str | None) -> None:
    # A `git archive` export has no .git directory.  There the expected revision is
    # recorded as provenance and the imported implementation hash is the exact guard.
    if expected and actual is not None and not actual.startswith(expected):
        raise RuntimeError(f"DiRe revision mismatch: expected {expected}, found {actual}")


def fit_dire_large(X, args):
    repo = Path(args.dire_repo).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"DiRe repository does not exist: {repo}")
    actual_revision = _git_revision(repo)
    _check_revision(actual_revision, args.dire_revision)
    sys.path.insert(0, str(repo))
    import cupy as cp
    import dire_rapids

    module_path = Path(dire_rapids.__file__).resolve()
    if not module_path.is_relative_to(repo):
        raise RuntimeError(f"imported DiRe outside pinned repository: {module_path}")
    implementation = Path(inspect.getfile(dire_rapids.create_dire)).resolve()
    reducer = dire_rapids.create_dire(
        backend="cuvs",
        n_components=args.worker_target_dimension,
        n_neighbors=args.neighbors,
        init="pca",
        max_iter_layout=args.dire_iterations,
        metric=None,
        normalize=False,
        knn_backend="cuvs",
        cuvs_index_type=args.dire_cuvs_index,
        random_state=args.worker_seed,
        verbose=False,
    )
    resolved_cuvs_index = (
        reducer._select_cuvs_index_type(X.shape[0], X.shape[1], "sqeuclidean")
        if hasattr(reducer, "_select_cuvs_index_type")
        else None
    )
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    layout = reducer.fit_transform(X)
    cp.cuda.runtime.deviceSynchronize()
    return np.asarray(layout), {
        "fit_seconds": time.perf_counter() - started,
        "backend": type(reducer).__name__,
        "dire_repo": str(repo),
        "dire_revision": actual_revision,
        "dire_revision_expected": args.dire_revision,
        "dire_version": getattr(dire_rapids, "__version__", "unknown"),
        "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "knn_backend_requested": "cuvs",
        "knn_backend_used": getattr(reducer, "_last_knn_backend", None),
        "cuvs_index_requested": args.dire_cuvs_index,
        "cuvs_index_resolved": resolved_cuvs_index,
        "iterations": args.dire_iterations,
    }


def fit_umap_large(X, args, *, umap_class=None, cupy=None):
    """Fit scalable cuML UMAP.  Native inverse is intentionally absent."""
    if umap_class is None or cupy is None:
        import cupy as cp
        from cuml.manifold import UMAP

        umap_class, cupy = UMAP, cp
    reducer = umap_class(
        n_components=args.worker_target_dimension,
        n_neighbors=args.neighbors,
        min_dist=args.umap_min_dist,
        spread=args.umap_spread,
        metric="euclidean",
        init="spectral",
        random_state=args.worker_seed,
        build_algo=args.umap_build_algo,
        force_serial_epochs=True,
        output_type="numpy",
        verbose=False,
    )
    cupy.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    layout = reducer.fit_transform(X)
    cupy.cuda.runtime.deviceSynchronize()
    return np.asarray(layout), {
        "fit_seconds": time.perf_counter() - started,
        "backend": f"{type(reducer).__module__}.{type(reducer).__name__}",
        "build_algo_requested": args.umap_build_algo,
        "native_inverse": "disabled by large-data protocol",
    }


def fit_pca_large(X, args):
    try:
        import cupy as cp
        from cuml.decomposition import PCA

        reducer = PCA(
            n_components=args.worker_target_dimension,
            output_type="numpy",
        )
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        layout = reducer.fit_transform(X)
        cp.cuda.runtime.deviceSynchronize()
        backend = "cuml.decomposition.PCA"
    except (ImportError, ModuleNotFoundError):
        from sklearn.decomposition import PCA

        reducer = PCA(
            n_components=args.worker_target_dimension,
            svd_solver="randomized",
            random_state=args.worker_seed,
        )
        started = time.perf_counter()
        layout = reducer.fit_transform(X)
        backend = "sklearn.decomposition.PCA"
    return np.asarray(layout), {
        "fit_seconds": time.perf_counter() - started,
        "backend": backend,
        "native_inverse": "disabled by large-data protocol",
    }


def _auto_hidden_dimensions(n_features: int) -> tuple[int, ...]:
    return (256, 128, 32) if n_features <= 512 else (512, 256, 64)


def _build_autoencoder(torch, n_features, target_dimension, hidden_dimensions):
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder = []
            previous = n_features
            for width in hidden_dimensions:
                encoder.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            encoder.append(torch.nn.Linear(previous, target_dimension))
            self.encoder = torch.nn.Sequential(*encoder)
            decoder = []
            previous = target_dimension
            for width in reversed(hidden_dimensions):
                decoder.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            decoder.append(torch.nn.Linear(previous, n_features))
            self.decoder = torch.nn.Sequential(*decoder)

        def forward(self, batch):
            return self.decoder(self.encoder(batch))

    return Autoencoder()


def _torch_loader(torch, tensor, indices, batch_size, seed, shuffle):
    subset = torch.utils.data.Subset(torch.utils.data.TensorDataset(tensor), indices)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=min(batch_size, len(subset)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _torch_mse(torch, model, loader, device, amp):
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                prediction = model(batch)
                loss = torch.sum((prediction.float() - batch.float()) ** 2)
            total += float(loss.item())
            count += batch.numel()
    return total / max(1, count)


def _train_auto_epochs(
    torch,
    model,
    tensor,
    train_indices,
    *,
    validation_indices,
    args,
    seed,
    device,
    fixed_epochs=None,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.ae_learning_rate, weight_decay=args.ae_weight_decay
    )
    train_loader = _torch_loader(
        torch, tensor, train_indices, args.ae_batch_size, seed, shuffle=True
    )
    validation_loader = None
    if validation_indices is not None:
        validation_loader = _torch_loader(
            torch, tensor, validation_indices, args.ae_batch_size, seed, shuffle=False
        )
    epochs = fixed_epochs or args.ae_epochs
    best_epoch, best_validation, stale = epochs, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.ae_amp):
                loss = torch.mean((model(batch).float() - batch.float()) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite autoencoder loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        if validation_loader is not None:
            value = _torch_mse(torch, model, validation_loader, device, args.ae_amp)
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite autoencoder validation loss at epoch {epoch}")
            if value < best_validation * (1.0 - args.ae_min_delta):
                best_validation, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
                if stale >= args.ae_patience:
                    break
    return best_epoch, best_validation


def _device_mse(torch, model, data, indices, batch_size, device, amp):
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            batch = data[indices[start : start + batch_size]]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                prediction = model(batch)
                loss = torch.sum((prediction.float() - batch.float()) ** 2)
            total += float(loss.item())
            count += batch.numel()
    return total / max(1, count)


def _train_auto_epochs_device(
    torch,
    model,
    data,
    train_indices,
    *,
    validation_indices,
    args,
    seed,
    device,
    fixed_epochs=None,
):
    """Train from one resident device tensor with device-side shuffling/slicing."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.ae_learning_rate, weight_decay=args.ae_weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    epochs = fixed_epochs or args.ae_epochs
    best_epoch, best_validation, stale = epochs, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_indices.numel(), generator=generator, device=device)
        for start in range(0, train_indices.numel(), args.ae_batch_size):
            batch_indices = train_indices[permutation[start : start + args.ae_batch_size]]
            batch = data[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.ae_amp):
                loss = torch.mean((model(batch).float() - batch.float()) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite autoencoder loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        if validation_indices is not None:
            value = _device_mse(
                torch,
                model,
                data,
                validation_indices,
                args.ae_batch_size,
                device,
                args.ae_amp,
            )
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite autoencoder validation loss at epoch {epoch}")
            if value < best_validation * (1.0 - args.ae_min_delta):
                best_validation, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
                if stale >= args.ae_patience:
                    break
    return best_epoch, best_validation


def fit_autoencoder_large(X, args, layout_path: Path):
    import torch

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA was requested for the autoencoder but is unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    hidden = (
        tuple(args.ae_hidden_dimensions)
        if args.ae_hidden_dimensions
        else _auto_hidden_dimensions(X.shape[1])
    )
    rng = np.random.default_rng(args.worker_seed)
    order = rng.permutation(X.shape[0])
    validation_size = min(X.shape[0] - 2, max(1, int(round(args.ae_validation * X.shape[0]))))
    validation_indices = order[:validation_size]
    training_indices = order[validation_size:]
    all_indices = np.arange(X.shape[0], dtype=np.int64)

    host_tensor = torch.from_numpy(X)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        staging_started = time.perf_counter()
        tensor = host_tensor.to(device)
        validation_indices = torch.as_tensor(validation_indices, dtype=torch.long, device=device)
        training_indices = torch.as_tensor(training_indices, dtype=torch.long, device=device)
        all_indices = torch.arange(X.shape[0], dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        staging_seconds = time.perf_counter() - staging_started
        training_pipeline = "cuda-resident device-side randperm and batch indexing"
        train_epochs = _train_auto_epochs_device
    else:
        tensor = host_tensor
        staging_seconds = 0.0
        training_pipeline = "CPU DataLoader fallback"
        train_epochs = _train_auto_epochs

    torch.manual_seed(args.worker_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.worker_seed)
        torch.cuda.synchronize(device)
    selection_started = time.perf_counter()
    tuning_model = _build_autoencoder(torch, X.shape[1], args.worker_target_dimension, hidden).to(
        device
    )
    selected_epoch, best_validation = train_epochs(
        torch,
        tuning_model,
        tensor,
        training_indices,
        validation_indices=validation_indices,
        args=args,
        seed=args.worker_seed,
        device=device,
    )
    _cuda_synchronize(torch=torch, device=device)
    selection_seconds = time.perf_counter() - selection_started
    del tuning_model

    torch.manual_seed(args.worker_seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.worker_seed + 1)
        torch.cuda.synchronize(device)
    refit_started = time.perf_counter()
    model = _build_autoencoder(torch, X.shape[1], args.worker_target_dimension, hidden).to(device)
    train_epochs(
        torch,
        model,
        tensor,
        all_indices,
        validation_indices=None,
        args=args,
        seed=args.worker_seed + 1,
        device=device,
        fixed_epochs=selected_epoch,
    )
    _cuda_synchronize(torch=torch, device=device)
    refit_seconds = time.perf_counter() - refit_started

    temporary = layout_path.with_suffix(layout_path.suffix + ".tmp")
    layout = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(X.shape[0], args.worker_target_dimension),
    )
    if device.type == "cuda":
        evaluation_batches = (
            tensor[start : start + args.ae_batch_size]
            for start in range(0, X.shape[0], args.ae_batch_size)
        )
    else:
        evaluation_batches = (
            batch
            for (batch,) in _torch_loader(
                torch,
                tensor,
                all_indices,
                args.ae_batch_size,
                args.worker_seed,
                shuffle=False,
            )
        )
    reconstruction_sse = 0.0
    reconstruction_cosine = 0.0
    elements = 0
    rows = 0
    cursor = 0
    model.eval()
    encode_started = time.perf_counter()
    with torch.no_grad():
        for batch in evaluation_batches:
            if device.type != "cuda":
                batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.ae_amp):
                encoded = model.encoder(batch)
                reconstructed = model.decoder(encoded).float()
            length = batch.shape[0]
            layout[cursor : cursor + length] = encoded.float().cpu().numpy()
            cursor += length
            target = batch.float()
            reconstruction_sse += float(torch.sum((reconstructed - target) ** 2).item())
            dot = torch.sum(reconstructed * target, dim=1)
            denominator = torch.linalg.vector_norm(reconstructed, dim=1).clamp_min(1e-12)
            reconstruction_cosine += float(torch.sum(dot / denominator).item())
            elements += batch.numel()
            rows += length
    _cuda_synchronize(torch=torch, device=device)
    encode_seconds = time.perf_counter() - encode_started
    layout.flush()
    del layout
    os.replace(temporary, layout_path)

    # Compute the mean-prediction baseline blockwise.  Most rows have unit norm,
    # while explicitly supported all-zero rows contribute zero rather than one.
    finalize_started = time.perf_counter()
    mean = np.zeros(X.shape[1], dtype=np.float64)
    sum_squared_norm = 0.0
    for start in range(0, X.shape[0], args.ae_batch_size):
        block = X[start : start + args.ae_batch_size]
        mean += np.sum(block, axis=0, dtype=np.float64)
        sum_squared_norm += float(np.sum(block.astype(np.float64) ** 2))
    mean /= X.shape[0]
    mean_squared_norm = sum_squared_norm / X.shape[0]
    baseline_mse = max(
        np.finfo(np.float64).eps,
        (mean_squared_norm - float(np.dot(mean, mean))) / X.shape[1],
    )
    native_mse = reconstruction_sse / elements
    metric_finalize_seconds = time.perf_counter() - finalize_started
    return None, {
        "fit_seconds": (
            staging_seconds
            + selection_seconds
            + refit_seconds
            + encode_seconds
            + metric_finalize_seconds
        ),
        "staging_seconds": staging_seconds,
        "staged_input_bytes": int(X.nbytes) if device.type == "cuda" else 0,
        "training_pipeline": training_pipeline,
        "selection_seconds": selection_seconds,
        "refit_seconds": refit_seconds,
        "encode_reconstruction_seconds": encode_seconds,
        "native_metric_finalize_seconds": metric_finalize_seconds,
        "selected_epochs": int(selected_epoch),
        "best_validation_mse": float(best_validation),
        "validation_fraction": args.ae_validation,
        "hidden_dimensions": list(hidden),
        "batch_size": args.ae_batch_size,
        "device": str(device),
        "amp_bfloat16": bool(args.ae_amp),
        "native_reconstruction_mse": native_mse,
        "native_reconstruction_nmse": native_mse / baseline_mse,
        "native_reconstruction_cosine": reconstruction_cosine / rows,
        "protocol": "validation epoch selection followed by fresh full-data refit",
    }


def run_worker(args) -> int:
    result_path = Path(args.worker_result)
    layout_path = Path(args.worker_layout)
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": args.dataset_name,
        "method": args.worker_method,
        "seed": args.worker_seed,
        "target_dimension": args.worker_target_dimension,
        "prepared_data": str(Path(args.worker_data).resolve()),
    }
    try:
        X = np.load(args.worker_data, mmap_mode="r+", allow_pickle=False)
        if X.ndim != 2 or X.dtype != np.float32:
            raise ValueError("prepared worker data must be a two-dimensional float32 NPY")
        with PeakMonitor() as monitor:
            if args.worker_method == "autoencoder":
                layout, details = fit_autoencoder_large(X, args, layout_path)
            elif args.worker_method == "dire":
                layout, details = fit_dire_large(X, args)
            elif args.worker_method == "umap":
                layout, details = fit_umap_large(X, args)
            elif args.worker_method == "pca":
                layout, details = fit_pca_large(X, args)
            else:
                raise ValueError(f"unknown method {args.worker_method!r}")
            if layout is not None:
                if layout.shape != (X.shape[0], args.worker_target_dimension):
                    raise RuntimeError("reducer returned a misaligned layout")
                if not np.all(np.isfinite(layout)):
                    raise RuntimeError("reducer returned non-finite coordinates")
                _save_layout_atomic(layout_path, layout)
        payload = {
            **common,
            "status": "ok",
            "error": None,
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "layout": str(layout_path.resolve()),
            "details": details,
            "memory": monitor.result(),
            "hardware": _worker_hardware(),
            "versions": {
                name: getattr(sys.modules[name], "__version__", "unknown")
                for name in ("numpy", "torch", "cupy", "cuml", "dire_rapids")
                if name in sys.modules
            },
        }
        _atomic_json(result_path, payload)
        return 0
    except Exception as exc:
        _atomic_json(
            result_path,
            {
                **common,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "layout": None,
                "details": {},
            },
        )
        raise


def deterministic_sample(n_rows: int, count: int, seed: int, labels=None) -> np.ndarray:
    count = min(int(count), n_rows)
    if count < 2:
        raise ValueError("sample count must be at least two")
    rng = np.random.default_rng(seed)
    if labels is None or count == n_rows:
        return np.sort(rng.choice(n_rows, size=count, replace=False))
    labels = np.asarray(labels)
    selected = []
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size > count:
        # It is impossible to represent every class.  Fall back to a documented,
        # deterministic uniform sample instead of manufacturing invalid quotas.
        return np.sort(rng.choice(n_rows, size=count, replace=False))
    quotas = np.floor(count * counts / counts.sum()).astype(int)
    quotas[(quotas == 0) & (counts > 0)] = 1
    while quotas.sum() > count:
        eligible = np.flatnonzero(quotas > 1)
        quotas[eligible[np.argmax(quotas[eligible])]] -= 1
    while quotas.sum() < count:
        room = counts - quotas
        quotas[np.argmax(room)] += 1
    for label, quota in zip(unique, quotas, strict=True):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, size=quota, replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def sample_pairs(n_rows: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n_rows, size=count, dtype=np.int64)
    second = rng.integers(0, n_rows - 1, size=count, dtype=np.int64)
    second += second >= first
    return first, second


def sampled_pair_distances(X, layout, pairs, block_rows=4096):
    first, second = pairs
    reference = np.empty(first.size, dtype=np.float64)
    embedded = np.empty(first.size, dtype=np.float64)
    for start in range(0, first.size, block_rows):
        stop = min(start + block_rows, first.size)
        a, b = first[start:stop], second[start:stop]
        left, right = X[a], X[b]
        dot = np.sum(left * right, axis=1, dtype=np.float64)
        left_norm = np.sum(left * left, axis=1, dtype=np.float64)
        right_norm = np.sum(right * right, axis=1, dtype=np.float64)
        reference[start:stop] = np.sqrt(np.maximum(0.0, left_norm + right_norm - 2.0 * dot))
        embedded[start:stop] = np.linalg.norm(layout[a] - layout[b], axis=1)
    return reference, embedded


def distance_fidelity(reference, embedded):
    denominator = float(np.dot(embedded, embedded))
    reference_norm = float(np.dot(reference, reference))
    if denominator <= 0 or reference_norm <= 0:
        raise ValueError("sampled distances are degenerate")
    correlation = float(spearmanr(reference, embedded).statistic)
    scale = float(np.dot(reference, embedded) / denominator)
    stress = float(np.sqrt(np.sum((reference - scale * embedded) ** 2) / reference_norm))
    if not np.isfinite(correlation) or not np.isfinite(stress):
        raise ValueError("sampled distance metric is non-finite")
    return correlation, stress


def neighbors_without_self(X, n_neighbors, *, n_jobs=1):
    raw = (
        NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        .fit(X)
        .kneighbors(X, return_distance=False)
    )
    result = np.empty((X.shape[0], n_neighbors), dtype=np.int64)
    for row in range(X.shape[0]):
        candidates = raw[row][raw[row] != row]
        if candidates.size < n_neighbors:
            raise ValueError("could not exclude self from neighborhood query")
        result[row] = candidates[:n_neighbors]
    return result


def sample_conditional_metrics(X, layout, indices, n_neighbors, *, n_jobs=1):
    """Quadratic metrics on a declared sample, never presented as full-corpus metrics."""
    Xs = np.asarray(X[indices], dtype=np.float32)
    Ys = np.asarray(layout[indices], dtype=np.float32)
    k = min(n_neighbors, Xs.shape[0] // 2 - 1)
    if k < 1:
        raise ValueError("metric sample is too small")
    original = neighbors_without_self(Xs, k, n_jobs=n_jobs)
    embedded = neighbors_without_self(Ys, k, n_jobs=n_jobs)
    overlap = np.mean(
        [np.intersect1d(original[row], embedded[row]).size / k for row in range(Xs.shape[0])]
    )
    return {
        "sample_conditional_trustworthiness": float(trustworthiness(Xs, Ys, n_neighbors=k)),
        "sample_conditional_continuity": float(trustworthiness(Ys, Xs, n_neighbors=k)),
        "sample_conditional_neighbor_overlap": float(overlap),
        "metric_sample_rows": int(Xs.shape[0]),
        "metric_sample_index_sha256": index_sha256(indices),
    }


def local_decodability_nmse(X, layout, indices, n_neighbors, seed, *, n_jobs=1):
    Xs = np.asarray(X[indices], dtype=np.float32)
    Ys = np.asarray(layout[indices], dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices.size)
    split = min(indices.size - 1, max(2, int(round(0.8 * indices.size))))
    train, test = order[:split], order[split:]
    mean = Ys[train].mean(axis=0)
    centered = Ys[train] - mean
    covariance = centered.T @ centered / max(1, train.size - 1)
    values, vectors = np.linalg.eigh(covariance)
    threshold = max(float(np.max(values)) * 1e-8, np.finfo(np.float64).eps)
    keep = values > threshold
    if not np.any(keep):
        raise ValueError("sampled layout is degenerate")
    whitening = vectors[:, keep] / np.sqrt(values[keep])
    train_layout = centered @ whitening
    test_layout = (Ys[test] - mean) @ whitening
    decoder = KNeighborsRegressor(
        n_neighbors=min(n_neighbors, train.size), weights="distance", n_jobs=n_jobs
    ).fit(train_layout, Xs[train])
    prediction = decoder.predict(test_layout)
    mse = float(np.mean((prediction - Xs[test]) ** 2))
    baseline = float(np.mean((Xs[test] - Xs[train].mean(axis=0)) ** 2))
    return mse / max(baseline, np.finfo(np.float64).eps)


def category_knn_metrics(layout, labels, indices, n_neighbors, seed, min_count=5, *, n_jobs=1):
    labels = np.asarray(labels)[indices]
    embedding = np.asarray(layout[indices], dtype=np.float32)
    unique, counts = np.unique(labels, return_counts=True)
    eligible = unique[counts >= min_count]
    keep = np.isin(labels, eligible) & (labels != "")
    labels, embedding = labels[keep], embedding[keep]
    if np.unique(labels).size < 2:
        return {"category_knn_accuracy": None, "category_knn_balanced_accuracy": None}
    train_X, test_X, train_y, test_y = train_test_split(
        embedding,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    classifier = KNeighborsClassifier(
        n_neighbors=min(n_neighbors, train_X.shape[0]), weights="distance", n_jobs=n_jobs
    ).fit(train_X, train_y)
    prediction = classifier.predict(test_X)
    return {
        "category_knn_accuracy": float(accuracy_score(test_y, prediction)),
        "category_knn_balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "category_rows": int(labels.size),
        "category_classes": int(np.unique(labels).size),
        "category_minimum_sample_count": int(min_count),
    }


def group_neighbor_metrics(layout, labels, indices, n_neighbors, *, n_jobs=1):
    """Same-group retrieval for keys such as Wikipedia article URL.

    ``indices`` are fixed query anchors, but the search index is the *complete* low-D
    layout. This is intentionally different from the sample-conditional quadratic
    metrics: article mates would otherwise disappear when Wikipedia is subsampled.
    Singleton groups in the full corpus are excluded rather than counted as failures.
    """
    labels = np.asarray(labels)
    anchors = np.asarray(indices, dtype=np.int64)
    embedding = np.asarray(layout, dtype=np.float32)
    k = min(n_neighbors, embedding.shape[0] - 1)
    raw = (
        NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs)
        .fit(embedding)
        .kneighbors(embedding[anchors], return_distance=False)
    )
    neighbors = np.empty((anchors.size, k), dtype=np.int64)
    for row, global_index in enumerate(anchors):
        candidates = raw[row][raw[row] != global_index]
        if candidates.size < k:
            raise ValueError("could not exclude global anchor from group-neighbor query")
        neighbors[row] = candidates[:k]
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    anchor_groups = inverse[anchors]
    eligible = (counts[anchor_groups] >= 2) & (labels[anchors] != "")
    if not np.any(eligible):
        return {
            "group_neighbor_purity": None,
            "group_neighbor_hit_rate": None,
            "group_neighbor_recall": None,
            "group_eligible_rows": 0,
        }
    matches = labels[neighbors] == labels[anchors, None]
    match_count = np.sum(matches, axis=1)
    possible = np.minimum(k, counts[anchor_groups] - 1)
    return {
        "group_neighbor_purity": float(np.mean(match_count[eligible] / k)),
        "group_neighbor_hit_rate": float(np.mean(match_count[eligible] > 0)),
        "group_neighbor_recall": float(np.mean(match_count[eligible] / possible[eligible])),
        "group_eligible_rows": int(np.count_nonzero(eligible)),
        "group_query_rows": int(anchors.size),
        "group_reference_rows": int(labels.size),
    }


def evaluate_layout(X, layout, args, labels=None):
    if layout.ndim != 2 or layout.shape[0] != X.shape[0] or not np.all(np.isfinite(layout)):
        raise ValueError("layout must be finite and aligned with prepared data")
    indices = deterministic_sample(
        X.shape[0], args.metric_sample, args.evaluation_seed, labels=labels
    )
    metrics = sample_conditional_metrics(
        X, layout, indices, args.neighbors, n_jobs=args.metric_jobs
    )
    decoder_indices = deterministic_sample(
        X.shape[0], args.decoder_sample, args.evaluation_seed + 1, labels=labels
    )
    metrics.update(
        {
            "sampled_local_decodability_nmse": float(
                local_decodability_nmse(
                    X,
                    layout,
                    decoder_indices,
                    args.neighbors,
                    args.evaluation_seed,
                    n_jobs=args.metric_jobs,
                )
            ),
            "decoder_sample_rows": int(decoder_indices.size),
            "decoder_sample_index_sha256": index_sha256(decoder_indices),
        }
    )
    pairs = sample_pairs(X.shape[0], args.distance_pairs, args.evaluation_seed + 2)
    reference, embedded = sampled_pair_distances(X, layout, pairs)
    correlation, stress = distance_fidelity(reference, embedded)
    metrics.update(
        {
            "sampled_pair_distance_spearman": correlation,
            "sampled_pair_normalized_stress": stress,
            "distance_pair_count": int(args.distance_pairs),
            "distance_pair_sha256": _sha256_bytes(
                np.ascontiguousarray(pairs[0], dtype="<i8").tobytes(),
                np.ascontiguousarray(pairs[1], dtype="<i8").tobytes(),
            ),
        }
    )
    policy = args.label_policy
    if policy == "auto":
        policy = (
            "group" if (args.label_column or "").lower() in {"url", "article_url"} else "category"
        )
    metrics["label_evaluation_policy"] = policy if labels is not None else "none"
    if labels is not None and policy == "category":
        metrics.update(
            category_knn_metrics(
                layout,
                labels,
                indices,
                args.neighbors,
                args.evaluation_seed,
                min_count=args.category_min_count,
                n_jobs=args.metric_jobs,
            )
        )
        metrics.update(
            {
                "group_neighbor_purity": None,
                "group_neighbor_hit_rate": None,
                "group_neighbor_recall": None,
            }
        )
    elif labels is not None and policy == "group":
        metrics.update(
            group_neighbor_metrics(layout, labels, indices, args.neighbors, n_jobs=args.metric_jobs)
        )
        metrics.update({"category_knn_accuracy": None, "category_knn_balanced_accuracy": None})
    else:
        metrics.update(
            {
                "category_knn_accuracy": None,
                "category_knn_balanced_accuracy": None,
                "group_neighbor_purity": None,
                "group_neighbor_hit_rate": None,
                "group_neighbor_recall": None,
            }
        )
    return metrics


def _worker_command(args, prepared: Path, method, seed, target, layout, result):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset-name",
        args.dataset_name,
        "--worker-data",
        str(prepared),
        "--worker-method",
        method,
        "--worker-seed",
        str(seed),
        "--worker-target-dimension",
        str(target),
        "--worker-layout",
        str(layout),
        "--worker-result",
        str(result),
        "--neighbors",
        str(args.neighbors),
        "--dire-repo",
        str(args.dire_repo),
        "--dire-iterations",
        str(args.dire_iterations),
        "--dire-cuvs-index",
        args.dire_cuvs_index,
        "--umap-build-algo",
        args.umap_build_algo,
        "--umap-min-dist",
        str(args.umap_min_dist),
        "--umap-spread",
        str(args.umap_spread),
        "--device",
        args.device,
        "--ae-epochs",
        str(args.ae_epochs),
        "--ae-patience",
        str(args.ae_patience),
        "--ae-validation",
        str(args.ae_validation),
        "--ae-min-delta",
        str(args.ae_min_delta),
        "--ae-batch-size",
        str(args.ae_batch_size),
        "--ae-learning-rate",
        str(args.ae_learning_rate),
        "--ae-weight-decay",
        str(args.ae_weight_decay),
    ]
    if args.dire_revision:
        command.extend(("--dire-revision", args.dire_revision))
    if args.ae_hidden_dimensions:
        command.extend(("--ae-hidden-dimensions", ",".join(map(str, args.ae_hidden_dimensions))))
    if args.ae_amp:
        command.append("--ae-amp")
    return command


def _run_subprocess(command, log_path: Path, timeout: int):
    started = time.perf_counter()
    try:
        # Write directly to a file instead of capturing a pipe. Some CUDA/RAPIDS
        # libraries start helper processes; inherited capture-pipe descriptors can keep
        # ``communicate`` waiting after the reducer itself exits.
        with log_path.open("w") as log_handle:
            completed = subprocess.run(
                command,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=None if timeout <= 0 else timeout,
            )
        return completed.returncode, time.perf_counter() - started, None
    except subprocess.TimeoutExpired:
        with log_path.open("a") as log_handle:
            log_handle.write(f"\nworker timed out after {timeout}s\n")
        return None, time.perf_counter() - started, f"worker timed out after {timeout}s"


def _write_csv(path: Path, rows: list[dict]):
    flat = []
    for row in rows:
        record = {}
        for key, value in row.items():
            record[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        flat.append(record)
    fields = sorted({key for row in flat for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def orchestrate(args) -> int:
    paths, source = discover_sources(args.data, args.shard_glob)
    if max(args.target_dimensions) > source.features:
        raise ValueError("target dimensions cannot exceed the source feature width")
    sizes = args.sizes or [source.rows]
    nested = deterministic_nested_indices(source.rows, sizes, args.sample_seed)
    labels = (
        load_labels(args.labels, args.label_column, source.rows, args.label_shard_glob)
        if args.labels
        else None
    )
    labels_metadata = (
        label_provenance(args.labels, args.label_column, args.label_shard_glob)
        if args.labels
        else None
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    prepared_records = []

    for size in sorted(nested):
        selected = nested[size]
        size_dir = output / f"n{size}"
        prepared_path = size_dir / "prepared-l2.npy"
        prepared_manifest = size_dir / "prepared.json"
        if prepared_path.exists() and prepared_manifest.exists() and not args.force:
            record = json.loads(prepared_manifest.read_text())
            if record.get("source_index_sha256") != index_sha256(selected):
                raise RuntimeError("prepared data index hash does not match requested sample")
            if record.get("source_fingerprint") != source.fingerprint:
                raise RuntimeError("prepared data source fingerprint is stale")
            prepared = PreparedInfo(**record)
        else:
            prepared = prepare_normalized_data(
                paths,
                selected,
                prepared_path,
                block_rows=args.prepare_block_rows,
                source_fingerprint=source.fingerprint,
            )
            _atomic_json(prepared_manifest, asdict(prepared))
        prepared_records.append(asdict(prepared))
        selected_labels = labels[selected] if labels is not None else None
        if selected_labels is not None:
            with (size_dir / "labels.npy").open("wb") as handle:
                np.save(handle, selected_labels, allow_pickle=True)

        X = np.load(prepared_path, mmap_mode="r", allow_pickle=False)
        for target in args.target_dimensions:
            for seed in args.seeds:
                for method in args.methods:
                    stem = f"{method}-d{target}-s{seed}"
                    layout_path = size_dir / f"{stem}-layout.npy"
                    result_path = size_dir / f"{stem}.json"
                    log_path = size_dir / f"{stem}.log"
                    reusable = False
                    if result_path.exists() and layout_path.exists() and not args.force:
                        prior = json.loads(result_path.read_text())
                        reusable = prior.get("status") == "ok"
                    if not reusable:
                        command = _worker_command(
                            args,
                            prepared_path,
                            method,
                            seed,
                            target,
                            layout_path,
                            result_path,
                        )
                        returncode, wall, timeout_error = _run_subprocess(
                            command, log_path, args.timeout
                        )
                        if timeout_error:
                            _atomic_json(
                                result_path,
                                {
                                    "dataset": args.dataset_name,
                                    "method": method,
                                    "seed": seed,
                                    "target_dimension": target,
                                    "status": "timeout",
                                    "error": timeout_error,
                                    "layout": None,
                                    "worker_wall_seconds": wall,
                                },
                            )
                        elif returncode and not result_path.exists():
                            _atomic_json(
                                result_path,
                                {
                                    "dataset": args.dataset_name,
                                    "method": method,
                                    "seed": seed,
                                    "target_dimension": target,
                                    "status": "failed",
                                    "error": f"worker exited with code {returncode}",
                                    "layout": None,
                                    "worker_wall_seconds": wall,
                                },
                            )
                    row = json.loads(result_path.read_text())
                    row.update(
                        {
                            "n_samples": size,
                            "source_index_sha256": prepared.source_index_sha256,
                            "prepared_sha256": prepared.normalized_sha256,
                        }
                    )
                    if row.get("status") == "ok":
                        try:
                            layout = np.load(layout_path, mmap_mode="r", allow_pickle=False)
                            row["metrics"] = evaluate_layout(
                                X, layout, args, labels=selected_labels
                            )
                        except Exception as exc:
                            row["metric_error"] = f"{type(exc).__name__}: {exc}"
                    all_rows.append(row)
                    _atomic_json(result_path, row)

    payload = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "protocol": "transductive fixed-target large-embedding comparison",
            "metric_scope": "sample-conditional except sampled random-pair distances",
            "dataset": args.dataset_name,
            "source": asdict(source),
            "labels": labels_metadata,
            "label_policy": args.label_policy,
            "prepared": prepared_records,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("worker_") and key not in {"worker", "force"}
            },
        },
        "rows": all_rows,
    }
    _atomic_json(output / "results.json", payload)
    _write_csv(output / "results.csv", all_rows)
    failures = [row for row in all_rows if row.get("status") != "ok"]
    print(f"wrote {output / 'results.json'} ({len(all_rows)} runs, {len(failures)} failures)")
    return 1 if args.require_all and failures else 0


def parse_int_tuple(value):
    if value in (None, ""):
        return ()
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-name", default="embeddings")
    parser.add_argument("--data")
    parser.add_argument("--shard-glob", default="emb_*.npy")
    parser.add_argument("--labels")
    parser.add_argument("--label-column")
    parser.add_argument("--label-shard-glob", default="*.parquet")
    parser.add_argument(
        "--label-policy",
        choices=("auto", "category", "group", "none"),
        default="auto",
        help="category classification or same-group neighbor retrieval",
    )
    parser.add_argument("--category-min-count", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--target-dimensions", nargs="+", type=int, default=[2])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--sample-seed", type=int, default=1729)
    parser.add_argument("--evaluation-seed", type=int, default=2718)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--metric-sample", type=int, default=5000)
    parser.add_argument("--decoder-sample", type=int, default=10_000)
    parser.add_argument("--distance-pairs", type=int, default=200_000)
    parser.add_argument(
        "--metric-jobs",
        type=int,
        default=1,
        help="CPU jobs for sampled neighbor metrics; one avoids joblib worker trees",
    )
    parser.add_argument("--prepare-block-rows", type=int, default=16_384)
    parser.add_argument("--output-dir", default="artifacts/large-embedding-comparison")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")

    parser.add_argument("--dire-repo", default="../dire-rapids")
    parser.add_argument("--dire-revision")
    parser.add_argument("--dire-iterations", type=int, default=128)
    parser.add_argument(
        "--dire-cuvs-index",
        choices=("auto", "flat", "ivf_flat", "ivf_pq", "cagra"),
        default="auto",
    )
    parser.add_argument(
        "--umap-build-algo",
        choices=("auto", "nn_descent", "brute_force_knn"),
        default="auto",
    )
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-spread", type=float, default=1.0)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ae-hidden-dimensions", type=parse_int_tuple, default=())
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--ae-patience", type=int, default=15)
    parser.add_argument("--ae-validation", type=float, default=0.05)
    parser.add_argument("--ae-min-delta", type=float, default=1e-5)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ae-amp", action="store_true")

    parser.add_argument("--worker-data", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-target-dimension", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-layout", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        required = (
            args.worker_data,
            args.worker_method,
            args.worker_seed,
            args.worker_target_dimension,
            args.worker_layout,
            args.worker_result,
        )
        if any(value is None for value in required):
            parser.error("worker mode requires all internal worker arguments")
    elif args.data is None:
        parser.error("--data is required")
    if set(args.methods) - set(METHODS):
        parser.error(f"--methods must be drawn from {METHODS}")
    if any(value < 1 for value in args.target_dimensions):
        parser.error("target dimensions must be positive")
    if args.neighbors < 1 or min(args.metric_sample, args.decoder_sample) < 3:
        parser.error("neighbor and metric sample sizes are too small")
    if args.distance_pairs < 1 or args.prepare_block_rows < 1 or args.metric_jobs < 1:
        parser.error("distance-pairs, prepare-block-rows, and metric-jobs must be positive")
    if args.category_min_count < 2:
        parser.error("category-min-count must be at least two")
    if not 0 < args.ae_validation < 0.5:
        parser.error("ae-validation must be in (0, 0.5)")
    if min(args.ae_epochs, args.ae_patience, args.ae_batch_size) < 1:
        parser.error("AE epochs, patience, and batch size must be positive")
    if args.umap_build_algo == "brute_force_knn" and not args.worker:
        parser.error("large-data orchestration forbids UMAP brute-force kNN")
    return args


def main(argv=None):
    args = parse_args(argv)
    return run_worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
