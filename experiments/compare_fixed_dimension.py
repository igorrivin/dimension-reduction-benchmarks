#!/usr/bin/env python3
"""Compare fixed-width Autoencoder, DiRe, RAPIDS UMAP, and PCA embeddings.

The headline comparison is deliberately transductive: every reducer embeds the same
complete matrix because DiRe does not currently expose an out-of-sample ``transform``.
Only metrics shared by every embedding are ranked. Native reconstruction is a separate
diagnostic, and a common kNN decoder probe supplies the apples-to-apples reconstruction
comparison.

Examples
--------
Quick CPU-capable run (RAPIDS UMAP is recorded as skipped when no CUDA device is visible)::

    conda run -n rapids-26.06 python experiments/compare_fixed_dimension.py \
      --datasets linear,swiss-roll,digits --target-dimension 2 --seeds 0

Repeated CUDA run::

    conda run -n rapids-26.06 python experiments/compare_fixed_dimension.py \
      --datasets linear,swiss-roll,digits --target-dimension 2 --seeds 0,1,2 \
      --device cuda --require-all --output-dir artifacts/fixed-dimension-cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    labels: np.ndarray | None
    latent: np.ndarray | None
    known_dimension: int | None
    intrinsic_metric: str | None


@dataclass
class BackendContext:
    cuml_umap: object | None = None
    cupy: object | None = None
    dire_class: object | None = None
    torch: object | None = None
    cuml_error: str | None = None
    dire_error: str | None = None
    torch_error: str | None = None
    cuda_devices: int = 0
    versions: dict | None = None
    hardware: dict | None = None


def _git_revision(repo):
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_backends(methods, dire_repo):
    """Load cuML before PyTorch to avoid CUDA shared-library shadowing."""
    context = BackendContext(versions={}, hardware={})
    if "umap" in methods or "dire" in methods:
        try:
            import cuml
            import cupy as cp
            from cuml.manifold import UMAP

            context.cuml_umap = UMAP
            context.cupy = cp
            context.versions["cuml"] = cuml.__version__
            context.versions["cupy"] = cp.__version__
            try:
                context.cuda_devices = int(cp.cuda.runtime.getDeviceCount())
                if context.cuda_devices:
                    properties = cp.cuda.runtime.getDeviceProperties(0)
                    name = properties.get("name", properties.get(b"name", "unknown"))
                    if isinstance(name, bytes):
                        name = name.decode(errors="replace")
                    _, total_memory = cp.cuda.Device(0).mem_info
                    context.hardware.update(
                        {
                            "gpu_name": str(name),
                            "gpu_total_memory_bytes": int(total_memory),
                            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
                            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                        }
                    )
            except Exception as exc:  # CUDA runtime reports the exact unavailable state
                context.cuml_error = f"CUDA device discovery failed: {type(exc).__name__}: {exc}"
        except Exception as exc:
            context.cuml_error = f"cuML import failed: {type(exc).__name__}: {exc}"

    if "dire" in methods:
        try:
            dire_repo = Path(dire_repo).expanduser().resolve()
            if str(dire_repo) not in sys.path:
                sys.path.insert(0, str(dire_repo))
            import dire_rapids

            context.dire_class = dire_rapids.DiRePyTorch
            context.versions["dire_rapids"] = getattr(dire_rapids, "__version__", "unknown")
            context.versions["dire_rapids_git"] = _git_revision(dire_repo)
            context.versions["dire_rapids_module"] = str(Path(dire_rapids.__file__).resolve())
            implementation = Path(inspect.getfile(dire_rapids.DiRePyTorch)).resolve()
            context.versions["dire_pytorch_sha256"] = hashlib.sha256(
                implementation.read_bytes()
            ).hexdigest()
        except Exception as exc:
            context.dire_error = f"DiRe import failed: {type(exc).__name__}: {exc}"

    if "autoencoder" in methods:
        try:
            import torch

            context.torch = torch
            context.versions["torch"] = torch.__version__
            if not context.cuda_devices and torch.cuda.is_available():
                context.cuda_devices = torch.cuda.device_count()
                properties = torch.cuda.get_device_properties(0)
                context.hardware.update(
                    {
                        "gpu_name": properties.name,
                        "gpu_total_memory_bytes": int(properties.total_memory),
                        "cuda_compute_capability": f"{properties.major}.{properties.minor}",
                    }
                )
        except Exception as exc:
            context.torch_error = f"PyTorch import failed: {type(exc).__name__}: {exc}"
    return context


def _orthogonal_embed(X, ambient_dimension, rng):
    if ambient_dimension < X.shape[1]:
        raise ValueError("ambient_dimension must not be smaller than source dimension")
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dimension, X.shape[1])))
    return X @ basis.T


def _stratified_subset(X, labels, n_samples, seed):
    if n_samples is None or n_samples >= X.shape[0]:
        return X, labels
    rng = np.random.default_rng(seed)
    selected = []
    classes, counts = np.unique(labels, return_counts=True)
    quotas = np.floor(n_samples * counts / counts.sum()).astype(int)
    quotas = np.maximum(quotas, 1)
    while quotas.sum() > n_samples:
        quotas[np.argmax(quotas)] -= 1
    while quotas.sum() < n_samples:
        room = counts - quotas
        quotas[np.argmax(room)] += 1
    for label, quota in zip(classes, quotas, strict=True):
        indices = np.flatnonzero(labels == label)
        selected.extend(rng.choice(indices, size=quota, replace=False))
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return X[selected], labels[selected]


def make_dataset(name, *, n_samples, ambient_dimension, target_dimension, seed, noise):
    rng = np.random.default_rng(seed)
    if name == "linear":
        latent = rng.normal(size=(n_samples, target_dimension))
        X = _orthogonal_embed(latent, ambient_dimension, rng)
        if noise:
            X += noise * rng.normal(size=X.shape)
        return Dataset(name, X, None, latent, target_dimension, "euclidean")

    if name == "swiss-roll":
        source, parameter = make_swiss_roll(
            n_samples=n_samples,
            noise=noise,
            random_state=seed,
        )
        arc_length = 0.5 * (parameter * np.sqrt(1.0 + parameter**2) + np.arcsinh(parameter))
        latent = np.column_stack((arc_length, source[:, 1]))
        X = _orthogonal_embed(source, ambient_dimension, rng)
        return Dataset(name, X, None, latent, 2, "euclidean")

    if name == "sphere":
        source = rng.normal(size=(n_samples, 3))
        source /= np.linalg.norm(source, axis=1, keepdims=True)
        X = _orthogonal_embed(source, ambient_dimension, rng)
        if noise:
            X += noise * rng.normal(size=X.shape)
        return Dataset(name, X, None, source, 2, "spherical")

    if name == "torus":
        u = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        v = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        source = np.column_stack((np.cos(u), np.sin(u), np.cos(v), np.sin(v)))
        X = _orthogonal_embed(source, ambient_dimension, rng)
        if noise:
            X += noise * rng.normal(size=X.shape)
        return Dataset(name, X, None, np.column_stack((u, v)), 2, "torus")

    if name == "digits":
        digits = load_digits()
        X, labels = _stratified_subset(
            digits.data.astype(np.float64),
            digits.target.astype(np.int64),
            n_samples,
            seed,
        )
        return Dataset(name, X, labels, None, None, None)

    raise ValueError(f"unknown dataset {name!r}")


def preprocess(X):
    """Center and apply one global RMS scale, preserving relative feature geometry."""
    X = np.asarray(X, dtype=np.float64)
    center = X.mean(axis=0)
    centered = X - center
    scale = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError("dataset has no variation after centering")
    return np.ascontiguousarray(centered / scale, dtype=np.float32)


def _neighbors_without_self(X, n_neighbors):
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
    raw = model.kneighbors(X, return_distance=False)
    result = np.empty((X.shape[0], n_neighbors), dtype=np.int64)
    for row in range(X.shape[0]):
        candidates = raw[row][raw[row] != row]
        result[row] = candidates[:n_neighbors]
    return result


def _sample_pairs(n_samples, count, seed):
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n_samples, size=count)
    second = rng.integers(0, n_samples - 1, size=count)
    second += second >= first
    return first, second


def _pair_distances(X, pairs):
    first, second = pairs
    return np.linalg.norm(X[first] - X[second], axis=1)


def _intrinsic_pair_distances(coordinates, pairs, metric):
    first, second = pairs
    if metric == "euclidean":
        return np.linalg.norm(coordinates[first] - coordinates[second], axis=1)
    if metric == "spherical":
        cosine = np.sum(coordinates[first] * coordinates[second], axis=1)
        return np.arccos(np.clip(cosine, -1.0, 1.0))
    if metric == "torus":
        delta = np.abs(coordinates[first] - coordinates[second])
        delta = np.minimum(delta, 2.0 * np.pi - delta)
        return np.linalg.norm(delta, axis=1)
    raise ValueError(f"unknown intrinsic metric {metric!r}")


def _distance_fidelity(reference_distance, embedded_distance):
    denominator = float(np.dot(embedded_distance, embedded_distance))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("embedding pairwise distances are degenerate")
    correlation = float(spearmanr(reference_distance, embedded_distance).statistic)
    if not np.isfinite(correlation):
        raise ValueError("embedding pairwise distances are degenerate")
    scale = float(np.dot(reference_distance, embedded_distance) / denominator)
    reference_norm = float(np.dot(reference_distance, reference_distance))
    stress = float(
        np.sqrt(np.sum((reference_distance - scale * embedded_distance) ** 2) / reference_norm)
    )
    if not np.isfinite(stress):
        raise ValueError("embedding produced non-finite stress")
    return correlation, stress


def _decoder_probe_nmse(X, embedding, seed, n_neighbors):
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    split = max(2, int(round(0.8 * X.shape[0])))
    split = min(split, X.shape[0] - 1)
    train, test = order[:split], order[split:]
    neighbors = min(n_neighbors, train.size)
    # Autoencoder latents are non-identifiable under any invertible linear coordinate
    # change because the decoder can absorb its inverse. Train-fold whitening makes the
    # common kNN probe invariant to that nuisance reparameterization, up to rotation.
    embedding_mean = embedding[train].mean(axis=0)
    centered_train = embedding[train] - embedding_mean
    centered_test = embedding[test] - embedding_mean
    covariance = centered_train.T @ centered_train / max(1, train.size - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    threshold = max(float(np.max(eigenvalues)) * 1e-8, np.finfo(np.float64).eps)
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise ValueError("embedding has no non-degenerate dimensions for decoder whitening")
    whitening = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    train_embedding = centered_train @ whitening
    test_embedding = centered_test @ whitening
    decoder = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
    decoder.fit(train_embedding, X[train])
    prediction = decoder.predict(test_embedding)
    mse = float(np.mean((prediction - X[test]) ** 2))
    baseline = float(np.mean((X[test] - X[train].mean(axis=0)) ** 2))
    return mse / baseline


def evaluate_embedding(
    X,
    embedding,
    *,
    labels,
    latent,
    intrinsic_metric,
    n_neighbors,
    pairs,
    seed,
):
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[0] != X.shape[0]:
        raise ValueError("embedding must be a two-dimensional array aligned with X")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("embedding contains non-finite values")
    n_neighbors = min(n_neighbors, X.shape[0] // 2 - 1)
    if n_neighbors < 1:
        raise ValueError("not enough samples for neighborhood metrics")

    original_neighbors = _neighbors_without_self(X, n_neighbors)
    embedded_neighbors = _neighbors_without_self(embedding, n_neighbors)
    overlap = np.mean(
        [
            np.intersect1d(original_neighbors[row], embedded_neighbors[row]).size / n_neighbors
            for row in range(X.shape[0])
        ]
    )

    original_distance = _pair_distances(X, pairs)
    embedded_distance = _pair_distances(embedding, pairs)
    distance_spearman, normalized_stress = _distance_fidelity(original_distance, embedded_distance)
    intrinsic_spearman = None
    intrinsic_stress = None
    if latent is not None and intrinsic_metric is not None:
        intrinsic_distance = _intrinsic_pair_distances(latent, pairs, intrinsic_metric)
        intrinsic_spearman, intrinsic_stress = _distance_fidelity(
            intrinsic_distance, embedded_distance
        )

    metrics = {
        "trustworthiness": float(trustworthiness(X, embedding, n_neighbors=n_neighbors)),
        "continuity": float(trustworthiness(embedding, X, n_neighbors=n_neighbors)),
        "neighbor_overlap": float(overlap),
        "distance_spearman": distance_spearman,
        "normalized_stress": normalized_stress,
        "intrinsic_distance_spearman": intrinsic_spearman,
        "intrinsic_normalized_stress": intrinsic_stress,
        "decoder_probe_nmse": float(
            _decoder_probe_nmse(X, embedding, seed=seed, n_neighbors=n_neighbors)
        ),
        "knn_accuracy": None,
        "balanced_accuracy": None,
        "latent_procrustes_disparity": None,
    }

    if labels is not None:
        labels = np.asarray(labels)
        minimum_class = int(np.min(np.unique(labels, return_counts=True)[1]))
        folds = min(5, minimum_class)
        if folds >= 2:
            classifier = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0] - 1))
            predictions = cross_val_predict(
                classifier,
                embedding,
                labels,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
            )
            metrics["knn_accuracy"] = float(np.mean(predictions == labels))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))

    if (
        latent is not None
        and intrinsic_metric == "euclidean"
        and latent.shape[1] == embedding.shape[1]
    ):
        _, _, disparity = procrustes(np.asarray(latent, dtype=np.float64), embedding)
        metrics["latent_procrustes_disparity"] = float(disparity)
    return metrics


def _native_nmse(X, reconstruction):
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    mse = float(np.mean((reconstruction - X) ** 2))
    baseline = float(np.mean((X - X.mean(axis=0)) ** 2))
    return mse / baseline


def fit_pca(X, target_dimension, seed):
    started = time.perf_counter()
    reducer = PCA(n_components=target_dimension, random_state=seed)
    embedding = reducer.fit_transform(X)
    elapsed = time.perf_counter() - started
    return embedding, elapsed, _native_nmse(X, reducer.inverse_transform(embedding)), {}


def _build_autoencoder(torch, n_features, target_dimension, hidden_dimensions):
    class FixedAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder_layers = []
            previous = n_features
            for width in hidden_dimensions:
                encoder_layers.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            encoder_layers.append(torch.nn.Linear(previous, target_dimension))
            self.encoder = torch.nn.Sequential(*encoder_layers)

            decoder_layers = []
            previous = target_dimension
            for width in reversed(hidden_dimensions):
                decoder_layers.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            decoder_layers.append(torch.nn.Linear(previous, n_features))
            self.decoder = torch.nn.Sequential(*decoder_layers)

        def forward(self, batch):
            return self.decoder(self.encoder(batch))

    return FixedAutoencoder()


def _torch_mse(torch, model, data, batch_size, device):
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = data[start : start + batch_size].to(device)
            total += float(torch.sum((model(batch) - batch) ** 2).item())
            count += batch.numel()
    return total / count


def _train_torch_epochs(
    torch,
    model,
    data,
    *,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    seed,
    device,
    validation=None,
    patience=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(batch_size, data.shape[0]),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    best_epoch = epochs
    best_validation = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch) - batch) ** 2)
            loss.backward()
            optimizer.step()
        if validation is not None:
            value = _torch_mse(torch, model, validation, batch_size, device)
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite autoencoder validation loss at epoch {epoch + 1}")
            if value < best_validation * (1.0 - 1e-5):
                best_validation = value
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return best_epoch


def fit_autoencoder(
    X,
    target_dimension,
    seed,
    *,
    torch,
    device,
    hidden_dimensions,
    epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
):
    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the autoencoder but is unavailable")

    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    started = time.perf_counter()
    tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    validation_size = max(1, int(round(0.15 * X.shape[0])))
    validation_size = min(validation_size, X.shape[0] - 2)
    validation = tensor[order[:validation_size]]
    training = tensor[order[validation_size:]]

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            tuning_model = _build_autoencoder(
                torch, X.shape[1], target_dimension, hidden_dimensions
            ).to(resolved)
            best_epoch = _train_torch_epochs(
                torch,
                tuning_model,
                training,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed,
                device=resolved,
                validation=validation,
                patience=patience,
            )

            torch.default_generator.manual_seed(seed + 1)
            model = _build_autoencoder(torch, X.shape[1], target_dimension, hidden_dimensions).to(
                resolved
            )
            _train_torch_epochs(
                torch,
                model,
                tensor,
                epochs=best_epoch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=seed + 1,
                device=resolved,
            )
            model.eval()
            with torch.no_grad():
                device_tensor = tensor.to(resolved)
                embedding_tensor = model.encoder(device_tensor)
                reconstruction_tensor = model.decoder(embedding_tensor)
                embedding = embedding_tensor.detach().cpu().numpy()
                reconstruction = reconstruction_tensor.detach().cpu().numpy()
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)
        elapsed = time.perf_counter() - started
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn)

    details = {"device": str(resolved), "selected_epochs": int(best_epoch)}
    return embedding, elapsed, _native_nmse(X, reconstruction), details


def fit_dire(
    X,
    target_dimension,
    seed,
    *,
    context,
    n_neighbors,
    iterations,
    knn_backend,
):
    if context.dire_class is None:
        raise RuntimeError(context.dire_error or "DiRe is unavailable")
    started = time.perf_counter()
    kwargs = dict(
        n_components=target_dimension,
        n_neighbors=n_neighbors,
        init="pca",
        max_iter_layout=iterations,
        metric=None,
        normalize=False,
        random_state=seed,
        verbose=False,
    )
    if "knn_backend" in inspect.signature(context.dire_class).parameters:
        kwargs["knn_backend"] = knn_backend
    reducer = context.dire_class(**kwargs)
    embedding = reducer.fit_transform(X)
    elapsed = time.perf_counter() - started
    details = {
        "backend": type(reducer).__name__,
        "device": str(getattr(reducer, "device", "unknown")),
        "iterations": iterations,
        "knn_backend_requested": knn_backend,
        "knn_backend_used": getattr(reducer, "_last_knn_backend", "legacy-auto"),
    }
    return np.asarray(embedding), elapsed, None, details


def fit_umap(X, target_dimension, seed, *, context, n_neighbors):
    if context.cuml_umap is None:
        raise RuntimeError(context.cuml_error or "cuML UMAP is unavailable")
    if context.cuda_devices < 1:
        raise RuntimeError(context.cuml_error or "cuML UMAP requires a visible CUDA device")
    cp = context.cupy
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    reducer = context.cuml_umap(
        n_components=target_dimension,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        spread=1.0,
        metric="euclidean",
        random_state=seed,
        build_algo="brute_force_knn",
        force_serial_epochs=True,
        output_type="numpy",
        verbose=False,
    )
    embedding = np.asarray(reducer.fit_transform(X))
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    native_nmse = None
    details = {"backend": "cuml.manifold.UMAP", "native_inverse": "not attempted"}
    # cuML's native inverse uses a CPU Delaunay phase. It rejects d=1 and becomes
    # combinatorially expensive/unreliable at d>=8; neither case belongs in the shared
    # comparison, which already includes the common decoder probe.
    if 2 <= target_dimension < 8 and hasattr(reducer, "inverse_transform"):
        try:
            reconstruction = reducer.inverse_transform(embedding)
            cp.cuda.runtime.deviceSynchronize()
            native_nmse = _native_nmse(X, np.asarray(reconstruction))
            details["native_inverse"] = "ok"
        except Exception as exc:
            details["native_inverse"] = f"failed: {type(exc).__name__}: {exc}"
            native_nmse = None
    elif target_dimension < 2:
        details["native_inverse"] = "unsupported below two dimensions"
    else:
        details["native_inverse"] = (
            "skipped at target dimension >= 8: cuML warns quality may be poor or fail, "
            "and the CPU Delaunay stage can be prohibitively expensive"
        )
    return embedding, elapsed, native_nmse, details


def _method_result(method, dataset, X, pairs, seed, args, context):
    common = {
        "dataset": dataset.name,
        "seed": seed,
        "known_dimension": dataset.known_dimension,
        "target_dimension": args.target_dimension,
        "method": method,
    }
    try:
        if method == "pca":
            embedding, elapsed, native_nmse, details = fit_pca(X, args.target_dimension, seed)
        elif method == "autoencoder":
            embedding, elapsed, native_nmse, details = fit_autoencoder(
                X,
                args.target_dimension,
                seed,
                torch=context.torch,
                device=args.device,
                hidden_dimensions=args.ae_hidden_dimensions,
                epochs=args.ae_epochs,
                patience=args.ae_patience,
                batch_size=args.ae_batch_size,
                learning_rate=args.ae_learning_rate,
                weight_decay=args.ae_weight_decay,
            )
        elif method == "dire":
            embedding, elapsed, native_nmse, details = fit_dire(
                X,
                args.target_dimension,
                seed,
                context=context,
                n_neighbors=args.neighbors,
                iterations=args.dire_iterations,
                knn_backend=args.dire_knn_backend,
            )
        elif method == "umap":
            embedding, elapsed, native_nmse, details = fit_umap(
                X,
                args.target_dimension,
                seed,
                context=context,
                n_neighbors=args.neighbors,
            )
        else:
            raise ValueError(f"unknown method {method!r}")

        metrics = evaluate_embedding(
            X,
            embedding,
            labels=dataset.labels,
            latent=dataset.latent,
            intrinsic_metric=dataset.intrinsic_metric,
            n_neighbors=args.neighbors,
            pairs=pairs,
            seed=seed,
        )
        return {
            **common,
            "status": "ok",
            "error": None,
            "fit_seconds": elapsed,
            "native_reconstruction_nmse": native_nmse,
            **metrics,
            "details": details,
        }
    except Exception as exc:
        status = "skipped" if method == "umap" and context.cuda_devices < 1 else "failed"
        return {
            **common,
            "status": status,
            "error": f"{type(exc).__name__}: {exc}",
            "fit_seconds": None,
            "native_reconstruction_nmse": None,
            "trustworthiness": None,
            "continuity": None,
            "neighbor_overlap": None,
            "distance_spearman": None,
            "normalized_stress": None,
            "intrinsic_distance_spearman": None,
            "intrinsic_normalized_stress": None,
            "decoder_probe_nmse": None,
            "knn_accuracy": None,
            "balanced_accuracy": None,
            "latent_procrustes_disparity": None,
            "details": {},
        }


def _write_results(output_dir, payload):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = payload["rows"]
    fieldnames = sorted({key for row in rows for key in row if key != "details"}) + ["details"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["details"] = json.dumps(
                serialized.get("details", {}), sort_keys=True, allow_nan=False
            )
            writer.writerow(serialized)
    summary = payload["summary"]
    if summary:
        summary_fields = sorted({key for row in summary for key in row})
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary)
    return json_path, csv_path, summary_path


def summarize_rows(rows):
    metric_names = (
        "fit_seconds",
        "trustworthiness",
        "continuity",
        "neighbor_overlap",
        "distance_spearman",
        "normalized_stress",
        "intrinsic_distance_spearman",
        "intrinsic_normalized_stress",
        "decoder_probe_nmse",
        "knn_accuracy",
        "balanced_accuracy",
        "latent_procrustes_disparity",
        "native_reconstruction_nmse",
    )
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    summary = []
    for (dataset, method), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "ok"]
        result = {
            "dataset": dataset,
            "method": method,
            "runs": len(group),
            "successful_runs": len(successful),
        }
        for metric in metric_names:
            values = [row[metric] for row in successful if row.get(metric) is not None]
            if values:
                values = np.asarray(values, dtype=np.float64)
                result[f"{metric}_mean"] = float(np.mean(values))
                result[f"{metric}_std"] = float(np.std(values))
                result[f"{metric}_median"] = float(np.median(values))
            else:
                result[f"{metric}_mean"] = None
                result[f"{metric}_std"] = None
                result[f"{metric}_median"] = None
        summary.append(result)
    return summary


def _format(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(rows):
    print(
        f"{'dataset':<12} {'method':<12} {'status':<8} {'time':>8} {'trust':>8} "
        f"{'cont':>8} {'overlap':>8} {'rho':>8} {'stress':>8} {'decode':>8} {'knn':>8}"
    )
    print("-" * 116)
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['method']:<12} {row['status']:<8} "
            f"{_format(row['fit_seconds']):>8} {_format(row['trustworthiness']):>8} "
            f"{_format(row['continuity']):>8} {_format(row['neighbor_overlap']):>8} "
            f"{_format(row['distance_spearman']):>8} {_format(row['normalized_stress']):>8} "
            f"{_format(row['decoder_probe_nmse']):>8} {_format(row['knn_accuracy']):>8}"
        )
        if row["error"]:
            print(f"  {row['error']}")


def print_aggregate_summary(summary):
    print("\nPaired-seed aggregate (metric means; time is median)")
    print(
        f"{'dataset':<12} {'method':<12} {'runs':>5} {'time':>8} {'trust':>8} "
        f"{'cont':>8} {'overlap':>8} {'rho':>8} {'stress':>8} {'decode':>8} {'knn':>8}"
    )
    print("-" * 116)
    for row in summary:
        print(
            f"{row['dataset']:<12} {row['method']:<12} {row['successful_runs']:>5} "
            f"{_format(row['fit_seconds_median']):>8} "
            f"{_format(row['trustworthiness_mean']):>8} "
            f"{_format(row['continuity_mean']):>8} "
            f"{_format(row['neighbor_overlap_mean']):>8} "
            f"{_format(row['distance_spearman_mean']):>8} "
            f"{_format(row['normalized_stress_mean']):>8} "
            f"{_format(row['decoder_probe_nmse_mean']):>8} "
            f"{_format(row['knn_accuracy_mean']):>8}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="linear,swiss-roll,digits",
        help="Comma-separated: linear, swiss-roll, sphere, torus, digits",
    )
    parser.add_argument(
        "--methods", default="pca,autoencoder,dire,umap", help="Comma-separated methods"
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated integer seeds")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--ambient-dimension", type=int, default=32)
    parser.add_argument("--target-dimension", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--distance-pairs", type=int, default=50_000)
    parser.add_argument("--device", default="auto", help="Autoencoder torch device")
    parser.add_argument("--ae-hidden-dimensions", default="128,64")
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--ae-patience", type=int, default=20)
    parser.add_argument("--ae-batch-size", type=int, default=256)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    parser.add_argument("--dire-iterations", type=int, default=128)
    parser.add_argument(
        "--dire-knn-backend",
        choices=("pytorch", "pykeops", "cuvs", "auto"),
        default="pytorch",
    )
    parser.add_argument(
        "--dire-revision",
        default=None,
        help="Expected DiRe git revision recorded for provenance (for example f7d5736)",
    )
    parser.add_argument("--dire-repo", default="../dire-rapids")
    parser.add_argument("--output-dir", default="artifacts/fixed-dimension")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    args.methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    args.seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    args.ae_hidden_dimensions = tuple(
        int(item) for item in args.ae_hidden_dimensions.split(",") if item.strip()
    )
    valid_methods = {"pca", "autoencoder", "dire", "umap"}
    if not args.methods or set(args.methods) - valid_methods:
        raise ValueError(f"methods must be drawn from {sorted(valid_methods)}")
    if not args.datasets or not args.seeds:
        raise ValueError("at least one dataset and seed are required")
    if args.target_dimension < 1 or args.ambient_dimension < args.target_dimension:
        raise ValueError("ambient dimension must be at least the positive target dimension")
    return args


def main():
    args = parse_args()
    context = load_backends(args.methods, args.dire_repo)
    rows = []
    for seed in args.seeds:
        for dataset_name in args.datasets:
            dataset = make_dataset(
                dataset_name,
                n_samples=args.n_samples,
                ambient_dimension=args.ambient_dimension,
                target_dimension=args.target_dimension,
                seed=seed,
                noise=args.noise,
            )
            X = preprocess(dataset.X)
            pair_count = min(args.distance_pairs, max(1, X.shape[0] * (X.shape[0] - 1)))
            pairs = _sample_pairs(X.shape[0], pair_count, seed)
            for method in args.methods:
                print(f"running dataset={dataset_name} seed={seed} method={method}", flush=True)
                rows.append(_method_result(method, dataset, X, pairs, seed, args, context))

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "backend_versions": context.versions,
        "hardware": context.hardware,
        "harness_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "cuda_devices": context.cuda_devices,
        "cuml_error": context.cuml_error,
        "dire_error": context.dire_error,
        "torch_error": context.torch_error,
        "protocol": "transductive fixed-target-dimension embedding comparison",
        "configuration": {
            key: value for key, value in vars(args).items() if key not in {"require_all"}
        },
    }
    summary = summarize_rows(rows)
    payload = {"metadata": metadata, "rows": rows, "summary": summary}
    json_path, csv_path, summary_path = _write_results(Path(args.output_dir), payload)
    print_summary(rows)
    print_aggregate_summary(summary)
    print(f"\nJSON:    {json_path}\nCSV:     {csv_path}\nSummary: {summary_path}")
    failures = [row for row in rows if row["status"] != "ok"]
    if args.require_all and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
