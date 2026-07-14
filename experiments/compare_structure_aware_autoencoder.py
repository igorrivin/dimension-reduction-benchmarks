#!/usr/bin/env python3
"""Pilot sparse structure-aware autoencoder losses on the small benchmark datasets.

This harness deliberately leaves :mod:`experiments.compare_fixed_dimension` unchanged.
It uses the same datasets, preprocessing, architecture, and shared metrics, but replaces
the reconstruction-MSE early-stopping protocol with a paired fixed-epoch comparison.
Every arm therefore starts from the same weights and sees the same reconstruction-batch
order for a given dataset and seed.

The optional ranking objective is built from a fixed-size input-space reservoir.  It
encodes sparse ``(anchor, positive, near-negative, uniform-negative)`` tuples; it never
forms a dense pairwise tensor during training.  Labels are passed only to the unchanged
post-fit evaluator and are never available to model fitting or structure preparation.

Example
-------
Run the default MSE baseline and three ranking weights on a GPU::

    conda run -n rapids-26.06 python \
      experiments/compare_structure_aware_autoencoder.py \
      --datasets digits,swiss-roll,linear --seeds 0,1,2 --device cuda

Add two sparse normalized-stress arms::

    conda run -n rapids-26.06 python \
      experiments/compare_structure_aware_autoencoder.py \
      --stress-weights 0.03,0.1 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.neighbors import NearestNeighbors

_HELPERS = importlib.import_module(
    f"{__package__}.compare_fixed_dimension" if __package__ else "compare_fixed_dimension"
)
_build_autoencoder = _HELPERS._build_autoencoder
_native_nmse = _HELPERS._native_nmse
_sample_pairs = _HELPERS._sample_pairs
evaluate_embedding = _HELPERS.evaluate_embedding
make_dataset = _HELPERS.make_dataset
preprocess = _HELPERS.preprocess

PROTOCOL_COMPARABILITY = (
    "Within-pilot paired comparisons only: this single-fit fixed-epoch MSE baseline is "
    "not directly comparable to authoritative early-stopping plus fresh-refit artifacts."
)


@dataclass(frozen=True)
class StructureGraph:
    """Bounded input-space supervision for sparse auxiliary losses.

    All indices are global row indices into the prepared matrix. ``positive_indices``
    contains the exact reservoir-conditional top-k neighbors for provenance. Each row
    of ``triplets`` is actually a four-tuple: anchor, positive, near negative, and
    uniformly sampled non-neighbor. ``stress_pairs`` is a sparse list, not a matrix.
    """

    reservoir_indices: np.ndarray
    positive_indices: np.ndarray
    triplets: np.ndarray
    stress_pairs: np.ndarray
    stress_target_distances: np.ndarray
    reservoir_sha256: str
    positive_sha256: str
    triplet_sha256: str
    pair_sha256: str
    target_distance_sha256: str
    preparation_seconds: float
    tie_filtered_candidates: int
    anchors_with_triplets: int


@dataclass(frozen=True)
class Arm:
    name: str
    loss_kind: str
    rank_weight: float = 0.0
    stress_weight: float = 0.0


def _canonical_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.kind in "iu":
        dtype = (
            np.dtype(f"<i{array.dtype.itemsize}")
            if array.dtype.kind == "i"
            else np.dtype(f"<u{array.dtype.itemsize}")
        )
    elif array.dtype.kind == "f":
        dtype = np.dtype(f"<f{array.dtype.itemsize}")
    else:
        raise TypeError(f"unsupported hash dtype {array.dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


def array_sha256(array: np.ndarray) -> str:
    """Hash an array with explicit shape and canonical little-endian bytes."""
    canonical = _canonical_array(array)
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def pair_sha256(pairs: tuple[np.ndarray, np.ndarray] | np.ndarray) -> str:
    if isinstance(pairs, tuple):
        array = np.column_stack(pairs)
    else:
        array = np.asarray(pairs)
    return array_sha256(np.asarray(array, dtype=np.int64))


def _strictly_farther(candidate: float, positive: float, tie_rtol: float) -> bool:
    scale = max(abs(candidate), abs(positive), np.finfo(np.float64).tiny)
    return candidate > positive + tie_rtol * scale


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


def build_structure_graph(
    X: np.ndarray,
    *,
    reservoir_size: int = 256,
    n_neighbors: int = 15,
    stress_pair_count: int = 2_048,
    seed: int = 0,
    tie_rtol: float = 1e-7,
) -> StructureGraph:
    """Build a deterministic, bounded sparse graph using input features only.

    A top-k positive is never compared against a negative at an indistinguishable
    input-space distance. Such candidates are filtered instead of imposing an arbitrary
    ordering at a kNN tie. One tuple is attempted for every positive of every anchor.
    """
    started = time.perf_counter()
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 4 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite two-dimensional matrix with at least four rows")
    if not 4 <= reservoir_size <= X.shape[0]:
        raise ValueError("reservoir_size must be between four and the number of rows")
    if not 1 <= n_neighbors < reservoir_size - 2:
        raise ValueError("n_neighbors must leave room for two distinct negatives")
    if stress_pair_count < 1:
        raise ValueError("stress_pair_count must be positive")
    if not np.isfinite(tie_rtol) or tie_rtol < 0:
        raise ValueError("tie_rtol must be finite and non-negative")

    reservoir_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    reservoir = np.sort(
        reservoir_rng.choice(X.shape[0], size=reservoir_size, replace=False).astype(np.int64)
    )
    sample = np.ascontiguousarray(X[reservoir], dtype=np.float64)

    # This is the only dense neighbor operation, and both axes are capped by the
    # declared reservoir size rather than the corpus or reconstruction batch size.
    distances, indices = (
        NearestNeighbors(n_neighbors=reservoir_size, algorithm="brute", metric="euclidean")
        .fit(sample)
        .kneighbors(sample, return_distance=True)
    )

    ordered_local = np.empty((reservoir_size, reservoir_size - 1), dtype=np.int64)
    ordered_distance = np.empty((reservoir_size, reservoir_size - 1), dtype=np.float64)
    for local_anchor in range(reservoir_size):
        keep = indices[local_anchor] != local_anchor
        local_indices = indices[local_anchor, keep]
        local_distances = distances[local_anchor, keep]
        if local_indices.size != reservoir_size - 1:
            raise RuntimeError("could not exclude the reservoir anchor from its neighbors")
        # NearestNeighbors is distance-sorted; lexsort makes exact ties platform-stable.
        order = np.lexsort((reservoir[local_indices], local_distances))
        ordered_local[local_anchor] = local_indices[order]
        ordered_distance[local_anchor] = local_distances[order]

    positives_local = ordered_local[:, :n_neighbors]
    positives = reservoir[positives_local]
    triplet_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    tuples: list[tuple[int, int, int, int]] = []
    tie_filtered = 0
    anchors_with_triplets = 0
    near_stop = min(4 * n_neighbors, reservoir_size - 1)
    for local_anchor in range(reservoir_size):
        anchor_tuples = 0
        positive_set = set(int(value) for value in positives_local[local_anchor])
        ranked = ordered_local[local_anchor]
        ranked_distance = ordered_distance[local_anchor]
        for positive_rank in range(n_neighbors):
            positive_local = int(ranked[positive_rank])
            positive_distance = float(ranked_distance[positive_rank])
            near_candidates = []
            for rank in range(n_neighbors, near_stop):
                candidate_local = int(ranked[rank])
                candidate_distance = float(ranked_distance[rank])
                if _strictly_farther(candidate_distance, positive_distance, tie_rtol):
                    near_candidates.append(candidate_local)
                else:
                    tie_filtered += 1
            if not near_candidates:
                continue
            near_local = int(near_candidates[triplet_rng.integers(len(near_candidates))])

            uniform_candidates = []
            for rank in range(n_neighbors, reservoir_size - 1):
                candidate_local = int(ranked[rank])
                if candidate_local == near_local or candidate_local in positive_set:
                    continue
                candidate_distance = float(ranked_distance[rank])
                if _strictly_farther(candidate_distance, positive_distance, tie_rtol):
                    uniform_candidates.append(candidate_local)
                else:
                    tie_filtered += 1
            if not uniform_candidates:
                continue
            uniform_local = int(uniform_candidates[triplet_rng.integers(len(uniform_candidates))])
            tuples.append(
                (
                    int(reservoir[local_anchor]),
                    int(reservoir[positive_local]),
                    int(reservoir[near_local]),
                    int(reservoir[uniform_local]),
                )
            )
            anchor_tuples += 1
        anchors_with_triplets += anchor_tuples > 0

    if not tuples:
        raise ValueError("the reservoir contains no strict positive/negative distance ordering")
    triplets = np.asarray(tuples, dtype=np.int64)

    local_pairs = _sample_pairs(
        reservoir_size,
        min(stress_pair_count, reservoir_size * (reservoir_size - 1)),
        seed=seed + 2,
    )
    stress_pairs = np.column_stack((reservoir[local_pairs[0]], reservoir[local_pairs[1]])).astype(
        np.int64, copy=False
    )
    stress_target = np.linalg.norm(
        np.asarray(X[stress_pairs[:, 0]], dtype=np.float64)
        - np.asarray(X[stress_pairs[:, 1]], dtype=np.float64),
        axis=1,
    )
    if not np.any(stress_target > np.finfo(np.float64).eps):
        raise ValueError("all sparse stress pairs have zero input-space distance")

    reservoir = _readonly(reservoir.astype(np.int64, copy=False))
    positives = _readonly(positives.astype(np.int64, copy=False))
    triplets = _readonly(triplets)
    stress_pairs = _readonly(stress_pairs)
    stress_target = _readonly(stress_target.astype(np.float32))
    return StructureGraph(
        reservoir_indices=reservoir,
        positive_indices=positives,
        triplets=triplets,
        stress_pairs=stress_pairs,
        stress_target_distances=stress_target,
        reservoir_sha256=array_sha256(reservoir),
        positive_sha256=array_sha256(positives),
        triplet_sha256=array_sha256(triplets),
        pair_sha256=array_sha256(stress_pairs),
        target_distance_sha256=array_sha256(stress_target),
        preparation_seconds=time.perf_counter() - started,
        tie_filtered_candidates=int(tie_filtered),
        anchors_with_triplets=int(anchors_with_triplets),
    )


def scale_invariant_rank_loss(
    torch,
    anchors,
    positives,
    negatives,
    *,
    temperature: float = 0.2,
):
    """InfoNCE ranking loss normalized by detached negative squared distance."""
    if anchors.ndim != 2 or positives.shape != anchors.shape:
        raise ValueError("anchors and positives must be aligned two-dimensional tensors")
    if negatives.ndim != 3 or negatives.shape[:1] != anchors.shape[:1]:
        raise ValueError("negatives must have shape (anchors, negatives, dimensions)")
    if negatives.shape[2] != anchors.shape[1] or negatives.shape[1] < 1:
        raise ValueError("negative dimensions must align and at least one is required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    positive_distance = torch.sum((anchors - positives) ** 2, dim=1, keepdim=True)
    negative_distance = torch.sum((anchors[:, None, :] - negatives) ** 2, dim=2)
    floor = torch.finfo(negative_distance.dtype).tiny
    distance_scale = negative_distance.mean().detach().clamp_min(floor)
    logits = -torch.cat((positive_distance, negative_distance), dim=1)
    logits = logits / (temperature * distance_scale)
    target = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
    return torch.nn.functional.cross_entropy(logits, target)


def scale_invariant_stress_loss(torch, first, second, target_distance):
    """Squared normalized stress with the optimal global latent-distance scale."""
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("stress endpoints must be aligned two-dimensional tensors")
    if target_distance.ndim != 1 or target_distance.shape[0] != first.shape[0]:
        raise ValueError("target distances must align with stress endpoints")
    latent_distance = torch.linalg.vector_norm(first - second, dim=1)
    target_distance = target_distance.to(dtype=latent_distance.dtype)
    floor = torch.finfo(latent_distance.dtype).tiny
    denominator = torch.sum(latent_distance * latent_distance).clamp_min(floor)
    scale = torch.sum(target_distance * latent_distance) / denominator
    reference_norm = torch.sum(target_distance * target_distance).clamp_min(floor)
    return torch.sum((target_distance - scale * latent_distance) ** 2) / reference_norm


def _resolve_device(torch, requested: str):
    if requested == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the autoencoder but is unavailable")
    return resolved


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warm_torch_runtime(torch, requested_device: str) -> tuple[str, float]:
    """Pay one-time optimizer/runtime initialization outside every measured arm."""
    device = _resolve_device(torch, requested_device)
    started = time.perf_counter()
    parameter = torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.sum((parameter @ parameter) ** 2)
    loss.backward()
    optimizer.step()
    _synchronize(torch, device)
    return str(device), time.perf_counter() - started


def fit_fixed_epoch_autoencoder(
    X: np.ndarray,
    target_dimension: int,
    seed: int,
    *,
    torch,
    device: str = "auto",
    hidden_dimensions: tuple[int, ...] = (128, 64),
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    structure_graph: StructureGraph | None = None,
    rank_weight: float = 0.0,
    stress_weight: float = 0.0,
    structure_interval: int = 4,
    structure_batch_size: int = 64,
    stress_batch_size: int = 128,
    rank_temperature: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one paired fixed-epoch arm and return layout, reconstruction, and audit data."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite two-dimensional matrix")
    if min(target_dimension, epochs, batch_size, structure_interval, structure_batch_size) < 1:
        raise ValueError("dimensions, epochs, batches, and structure interval must be positive")
    if stress_batch_size < 1:
        raise ValueError("stress_batch_size must be positive")
    if min(rank_weight, stress_weight) < 0 or not all(
        math.isfinite(value) for value in (rank_weight, stress_weight)
    ):
        raise ValueError("auxiliary weights must be finite and non-negative")
    auxiliary_enabled = rank_weight > 0 or stress_weight > 0
    if auxiliary_enabled and structure_graph is None:
        raise ValueError("a structure graph is required for a positive auxiliary weight")

    resolved = _resolve_device(torch, device)
    tensor = torch.from_numpy(X)
    total_started = time.perf_counter()
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with torch.random.fork_rng(devices=[]):
            # Model initialization is on CPU, as in the authoritative small harness.
            torch.default_generator.manual_seed(seed)
            model = _build_autoencoder(torch, X.shape[1], target_dimension, hidden_dimensions).to(
                resolved
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            batch_generator = torch.Generator().manual_seed(seed + 1)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tensor),
                batch_size=min(batch_size, X.shape[0]),
                shuffle=True,
                generator=batch_generator,
                pin_memory=resolved.type == "cuda",
                num_workers=0,
            )
            structure_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            reconstruction_sse = 0.0
            reconstruction_elements = 0
            objective_sum = 0.0
            rank_sum = 0.0
            stress_sum = 0.0
            rank_events = 0
            stress_events = 0
            maximum_auxiliary_rows = 0
            updates = 0

            _synchronize(torch, resolved)
            training_started = time.perf_counter()
            for _epoch in range(epochs):
                model.train()
                for (host_batch,) in loader:
                    batch = host_batch.to(resolved, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    reconstruction = model(batch)
                    reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
                    objective = reconstruction_loss
                    auxiliary_rows = 0
                    active_update = auxiliary_enabled and (updates + 1) % structure_interval == 0
                    if active_update and rank_weight > 0:
                        assert structure_graph is not None
                        selected = structure_rng.integers(
                            0, structure_graph.triplets.shape[0], size=structure_batch_size
                        )
                        tuple_indices = np.asarray(
                            structure_graph.triplets[selected], dtype=np.int64
                        )
                        flat_indices = torch.from_numpy(tuple_indices.reshape(-1))
                        structure_rows = tensor[flat_indices].to(resolved, non_blocking=True)
                        encoded = (
                            model.encoder(structure_rows)
                            .float()
                            .reshape(structure_batch_size, 4, target_dimension)
                        )
                        rank_loss = scale_invariant_rank_loss(
                            torch,
                            encoded[:, 0],
                            encoded[:, 1],
                            encoded[:, 2:],
                            temperature=rank_temperature,
                        )
                        if not bool(torch.isfinite(rank_loss).item()):
                            raise RuntimeError("the sparse rank loss became non-finite")
                        objective = objective + rank_weight * rank_loss
                        rank_sum += float(rank_loss.detach().item())
                        rank_events += 1
                        auxiliary_rows += int(structure_rows.shape[0])

                    if active_update and stress_weight > 0:
                        assert structure_graph is not None
                        selected = structure_rng.integers(
                            0,
                            structure_graph.stress_pairs.shape[0],
                            size=stress_batch_size,
                        )
                        selected_pairs = np.asarray(
                            structure_graph.stress_pairs[selected], dtype=np.int64
                        )
                        flat_indices = torch.from_numpy(selected_pairs.reshape(-1))
                        structure_rows = tensor[flat_indices].to(resolved, non_blocking=True)
                        encoded = (
                            model.encoder(structure_rows)
                            .float()
                            .reshape(stress_batch_size, 2, target_dimension)
                        )
                        target = torch.from_numpy(
                            np.asarray(
                                structure_graph.stress_target_distances[selected],
                                dtype=np.float32,
                            )
                        ).to(resolved)
                        stress_loss = scale_invariant_stress_loss(
                            torch, encoded[:, 0], encoded[:, 1], target
                        )
                        if not bool(torch.isfinite(stress_loss).item()):
                            raise RuntimeError("the sparse stress loss became non-finite")
                        objective = objective + stress_weight * stress_loss
                        stress_sum += float(stress_loss.detach().item())
                        stress_events += 1
                        auxiliary_rows += int(structure_rows.shape[0])

                    if not bool(torch.isfinite(objective).item()):
                        raise RuntimeError("the autoencoder objective became non-finite")
                    objective.backward()
                    optimizer.step()

                    reconstruction_sse += float(reconstruction_loss.detach().item()) * batch.numel()
                    reconstruction_elements += batch.numel()
                    objective_sum += float(objective.detach().item())
                    maximum_auxiliary_rows = max(maximum_auxiliary_rows, auxiliary_rows)
                    updates += 1
            _synchronize(torch, resolved)
            training_seconds = time.perf_counter() - training_started

            model.eval()
            layouts = []
            reconstructions = []
            _synchronize(torch, resolved)
            encoding_started = time.perf_counter()
            with torch.no_grad():
                for start in range(0, tensor.shape[0], batch_size):
                    batch = tensor[start : start + batch_size].to(resolved, non_blocking=True)
                    encoded = model.encoder(batch)
                    reconstructed = model.decoder(encoded)
                    layouts.append(encoded.detach().cpu().numpy())
                    reconstructions.append(reconstructed.detach().cpu().numpy())
            _synchronize(torch, resolved)
            encoding_seconds = time.perf_counter() - encoding_started
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn)

    if auxiliary_enabled and rank_weight > 0 and rank_events < 1:
        raise RuntimeError("the fixed budget produced no scheduled rank-loss update")
    if auxiliary_enabled and stress_weight > 0 and stress_events < 1:
        raise RuntimeError("the fixed budget produced no scheduled stress-loss update")
    embedding = np.ascontiguousarray(np.concatenate(layouts, axis=0), dtype=np.float32)
    reconstructed = np.ascontiguousarray(np.concatenate(reconstructions, axis=0), dtype=np.float32)
    details: dict[str, Any] = {
        "device": str(resolved),
        "fixed_epochs": int(epochs),
        "optimizer_updates": int(updates),
        "training_seconds": float(training_seconds),
        "encoding_seconds": float(encoding_seconds),
        "fit_seconds_excluding_structure_preparation": float(time.perf_counter() - total_started),
        "mean_reconstruction_training_mse": float(reconstruction_sse / reconstruction_elements),
        "mean_training_objective_per_update": float(objective_sum / updates),
        "mean_rank_loss_on_active_updates": (
            float(rank_sum / rank_events) if rank_events else None
        ),
        "mean_stress_loss_on_active_updates": (
            float(stress_sum / stress_events) if stress_events else None
        ),
        "rank_events": int(rank_events),
        "stress_events": int(stress_events),
        "maximum_auxiliary_rows_per_update": int(maximum_auxiliary_rows),
        "structure_interval": int(structure_interval),
        "structure_batch_size": int(structure_batch_size),
        "stress_batch_size": int(stress_batch_size),
        "rank_temperature": float(rank_temperature),
        "rank_weight": float(rank_weight),
        "stress_weight": float(stress_weight),
    }
    return embedding, reconstructed, details


def latent_diagnostics(embedding: np.ndarray) -> dict[str, float | int]:
    embedding = np.asarray(embedding, dtype=np.float64)
    centered = embedding - embedding.mean(axis=0)
    rms = float(np.sqrt(np.mean(centered * centered)))
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= np.finfo(np.float64).eps:
        return {
            "latent_rms": rms,
            "latent_effective_rank": 0.0,
            "latent_numerical_rank": 0,
        }
    probability = eigenvalues[eigenvalues > 0] / total
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    numerical_rank = int(np.count_nonzero(eigenvalues > eigenvalues[-1] * 1e-8))
    return {
        "latent_rms": rms,
        "latent_effective_rank": effective_rank,
        "latent_numerical_rank": numerical_rank,
    }


def evaluate_full_and_complement(
    X,
    embedding,
    dataset,
    graph: StructureGraph,
    *,
    n_neighbors: int,
    distance_pair_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate unchanged common metrics on all rows and reservoir-complement rows."""
    full_count = min(distance_pair_count, X.shape[0] * (X.shape[0] - 1))
    full_pairs = _sample_pairs(X.shape[0], full_count, seed=seed)
    full = evaluate_embedding(
        X,
        embedding,
        labels=dataset.labels,
        latent=dataset.latent,
        intrinsic_metric=dataset.intrinsic_metric,
        n_neighbors=n_neighbors,
        pairs=full_pairs,
        seed=seed,
    )

    keep = np.ones(X.shape[0], dtype=bool)
    keep[graph.reservoir_indices] = False
    complement_indices = np.flatnonzero(keep)
    if complement_indices.size < 4:
        raise ValueError("the reservoir complement must contain at least four rows")
    complement_X = np.asarray(X[complement_indices])
    complement_embedding = np.asarray(embedding[complement_indices])
    complement_labels = (
        None if dataset.labels is None else np.asarray(dataset.labels)[complement_indices]
    )
    complement_latent = (
        None if dataset.latent is None else np.asarray(dataset.latent)[complement_indices]
    )
    complement_count = min(
        distance_pair_count, complement_indices.size * (complement_indices.size - 1)
    )
    complement_pairs = _sample_pairs(complement_indices.size, complement_count, seed=seed + 1)
    complement = evaluate_embedding(
        complement_X,
        complement_embedding,
        labels=complement_labels,
        latent=complement_latent,
        intrinsic_metric=dataset.intrinsic_metric,
        n_neighbors=n_neighbors,
        pairs=complement_pairs,
        seed=seed,
    )
    complement_global_pairs = (
        complement_indices[complement_pairs[0]],
        complement_indices[complement_pairs[1]],
    )
    provenance = {
        "full_metric_pair_count": int(full_count),
        "full_metric_pair_sha256": pair_sha256(full_pairs),
        "complement_rows": int(complement_indices.size),
        "complement_index_sha256": array_sha256(complement_indices.astype(np.int64)),
        "complement_metric_pair_count": int(complement_count),
        "complement_metric_pair_sha256": pair_sha256(complement_global_pairs),
    }
    prefixed = {f"complement_{key}": value for key, value in complement.items()}
    return {**full, **prefixed}, provenance


def _weight_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_arms(
    rank_weights: tuple[float, ...], stress_weights: tuple[float, ...]
) -> tuple[Arm, ...]:
    arms = [Arm("autoencoder_mse", "mse")]
    arms.extend(
        Arm(f"autoencoder_rank_l{_weight_name(weight)}", "rank", rank_weight=weight)
        for weight in rank_weights
    )
    arms.extend(
        Arm(f"autoencoder_stress_l{_weight_name(weight)}", "stress", stress_weight=weight)
        for weight in stress_weights
    )
    return tuple(arms)


SUMMARY_METRICS = (
    "fit_seconds",
    "training_seconds",
    "native_reconstruction_nmse",
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
    "complement_trustworthiness",
    "complement_continuity",
    "complement_neighbor_overlap",
    "complement_distance_spearman",
    "complement_normalized_stress",
    "complement_intrinsic_distance_spearman",
    "complement_intrinsic_normalized_stress",
    "complement_decoder_probe_nmse",
    "complement_knn_accuracy",
    "complement_balanced_accuracy",
    "latent_rms",
    "latent_effective_rank",
)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    summary = []
    for (dataset, method), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [row.get(metric) for row in group if row.get(metric) is not None]
            if values:
                numeric = np.asarray(values, dtype=np.float64)
                record[f"{metric}_mean"] = float(np.mean(numeric))
                record[f"{metric}_std"] = float(np.std(numeric))
                record[f"{metric}_median"] = float(np.median(numeric))
            else:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None
                record[f"{metric}_median"] = None
        summary.append(record)
    return summary


def _write_payload(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = payload["rows"]
    fields = sorted({key for row in rows for key in row if key != "details"}) + ["details"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["details"] = json.dumps(
                serialized.get("details", {}), sort_keys=True, allow_nan=False
            )
            writer.writerow(serialized)
    summary = payload["summary"]
    if summary:
        fields = sorted({key for row in summary for key in row})
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
    return json_path, csv_path, summary_path


def _graph_record(dataset: str, seed: int, graph: StructureGraph) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": seed,
        "reservoir_rows": int(graph.reservoir_indices.size),
        "reservoir_sha256": graph.reservoir_sha256,
        "positive_sha256": graph.positive_sha256,
        "triplets": int(graph.triplets.shape[0]),
        "triplet_sha256": graph.triplet_sha256,
        "stress_pairs": int(graph.stress_pairs.shape[0]),
        "stress_pair_sha256": graph.pair_sha256,
        "stress_target_distance_sha256": graph.target_distance_sha256,
        "preparation_seconds": graph.preparation_seconds,
        "tie_filtered_candidates": graph.tie_filtered_candidates,
        "anchors_with_triplets": graph.anchors_with_triplets,
    }


def run(args) -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError("this pilot requires PyTorch") from exc

    warmup_device, warmup_seconds = _warm_torch_runtime(torch, args.device)
    rows: list[dict[str, Any]] = []
    graph_records = []
    arms = make_arms(args.rank_weights, args.stress_weights)
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
            if args.reservoir_size >= X.shape[0] - 3:
                raise ValueError(
                    "reservoir_size must leave at least four rows for complement metrics"
                )
            graph = build_structure_graph(
                X,
                reservoir_size=args.reservoir_size,
                n_neighbors=args.neighbors,
                stress_pair_count=args.stress_pairs,
                seed=seed + args.structure_seed_offset,
                tie_rtol=args.tie_rtol,
            )
            graph_records.append(_graph_record(dataset_name, seed, graph))
            for arm in arms:
                print(f"running dataset={dataset_name} seed={seed} arm={arm.name}", flush=True)
                embedding, reconstruction, details = fit_fixed_epoch_autoencoder(
                    X,
                    args.target_dimension,
                    seed,
                    torch=torch,
                    device=args.device,
                    hidden_dimensions=args.ae_hidden_dimensions,
                    epochs=args.ae_epochs,
                    batch_size=args.ae_batch_size,
                    learning_rate=args.ae_learning_rate,
                    weight_decay=args.ae_weight_decay,
                    structure_graph=graph,
                    rank_weight=arm.rank_weight,
                    stress_weight=arm.stress_weight,
                    structure_interval=args.structure_interval,
                    structure_batch_size=args.structure_batch_size,
                    stress_batch_size=args.stress_batch_size,
                    rank_temperature=args.rank_temperature,
                )
                metrics, evaluation_provenance = evaluate_full_and_complement(
                    X,
                    embedding,
                    dataset,
                    graph,
                    n_neighbors=args.neighbors,
                    distance_pair_count=args.distance_pairs,
                    seed=seed + args.evaluation_seed_offset,
                )
                structure_used = arm.rank_weight > 0 or arm.stress_weight > 0
                fit_without_preparation = details["fit_seconds_excluding_structure_preparation"]
                details.update(
                    {
                        "loss_kind": arm.loss_kind,
                        "labels_used_for_training": False,
                        "protocol_comparability": PROTOCOL_COMPARABILITY,
                        "structure_used": structure_used,
                        "shared_structure_preparation_seconds": graph.preparation_seconds,
                        "reservoir_sha256": graph.reservoir_sha256,
                        "positive_sha256": graph.positive_sha256,
                        "triplet_sha256": graph.triplet_sha256,
                        "stress_pair_sha256": graph.pair_sha256,
                        **evaluation_provenance,
                    }
                )
                diagnostic = latent_diagnostics(embedding)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": arm.name,
                        "loss_kind": arm.loss_kind,
                        "target_dimension": args.target_dimension,
                        "known_dimension": dataset.known_dimension,
                        "rank_weight": arm.rank_weight,
                        "stress_weight": arm.stress_weight,
                        "fit_seconds": float(
                            fit_without_preparation
                            + (graph.preparation_seconds if structure_used else 0.0)
                        ),
                        "training_seconds": details["training_seconds"],
                        "encoding_seconds": details["encoding_seconds"],
                        "structure_preparation_seconds": (
                            graph.preparation_seconds if structure_used else 0.0
                        ),
                        "native_reconstruction_nmse": _native_nmse(X, reconstruction),
                        **diagnostic,
                        **metrics,
                        "details": details,
                    }
                )

    summary = summarize_rows(rows)
    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": "paired fixed-epoch sparse structure-aware autoencoder pilot",
        "protocol_comparability": PROTOCOL_COMPARABILITY,
        "labels_used_for_training": False,
        "structure_source": "prepared input features only",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "torch": torch.__version__,
        "runtime_warmup_device": warmup_device,
        "runtime_warmup_seconds_excluded_from_arms": warmup_seconds,
        "harness_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "configuration": vars(args),
        "arms": [arm.__dict__ for arm in arms],
        "structure_graphs": graph_records,
    }
    payload = {"metadata": metadata, "rows": rows, "summary": summary}
    paths = _write_payload(Path(args.output_dir), payload)
    print("\nPaired structure-aware AE summary (means across seeds)")
    print(
        f"{'dataset':<12} {'arm':<28} {'time':>8} {'NMSE':>8} "
        f"{'overlap':>9} {'comp-ov':>9} {'trust':>8} {'knn':>8}"
    )
    for record in summary:

        def fmt(key):
            value = record.get(key)
            return "-" if value is None else f"{value:.3f}"

        print(
            f"{record['dataset']:<12} {record['method']:<28} "
            f"{fmt('fit_seconds_median'):>8} "
            f"{fmt('native_reconstruction_nmse_mean'):>8} "
            f"{fmt('neighbor_overlap_mean'):>9} "
            f"{fmt('complement_neighbor_overlap_mean'):>9} "
            f"{fmt('trustworthiness_mean'):>8} "
            f"{fmt('knn_accuracy_mean'):>8}"
        )
    print(f"\nWrote {paths[0]}, {paths[1]}, and {paths[2]}")
    return payload


def _parse_weights(value: str, name: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    weights = tuple(float(item) for item in value.split(",") if item.strip())
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError(f"{name} must contain finite positive weights")
    if len(set(weights)) != len(weights):
        raise ValueError(f"{name} must not contain duplicate weights")
    return weights


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="digits,swiss-roll,linear")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-samples", type=int, default=1_000)
    parser.add_argument("--ambient-dimension", type=int, default=32)
    parser.add_argument("--target-dimension", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--distance-pairs", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ae-hidden-dimensions", default="128,64")
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--ae-batch-size", type=int, default=256)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    parser.add_argument("--rank-weights", default="0.03,0.1,0.3")
    parser.add_argument(
        "--stress-weights",
        default="",
        help="Optional comma-separated normalized-stress arms (for example 0.03,0.1)",
    )
    parser.add_argument("--reservoir-size", type=int, default=256)
    parser.add_argument("--structure-batch-size", type=int, default=64)
    parser.add_argument("--stress-batch-size", type=int, default=128)
    parser.add_argument("--structure-interval", type=int, default=4)
    parser.add_argument("--stress-pairs", type=int, default=2_048)
    parser.add_argument("--rank-temperature", type=float, default=0.2)
    parser.add_argument("--tie-rtol", type=float, default=1e-7)
    parser.add_argument("--structure-seed-offset", type=int, default=10_000)
    parser.add_argument("--evaluation-seed-offset", type=int, default=20_000)
    parser.add_argument("--output-dir", default="artifacts/structure-aware-autoencoder-pilot")
    args = parser.parse_args(argv)
    args.datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    args.seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    args.ae_hidden_dimensions = tuple(
        int(item) for item in args.ae_hidden_dimensions.split(",") if item.strip()
    )
    args.rank_weights = _parse_weights(args.rank_weights, "rank_weights")
    args.stress_weights = _parse_weights(args.stress_weights, "stress_weights")
    valid_datasets = {"digits", "linear", "sphere", "swiss-roll", "torus"}
    if not args.datasets or set(args.datasets) - valid_datasets:
        raise ValueError(f"datasets must be drawn from {sorted(valid_datasets)}")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    positive = (
        args.n_samples,
        args.ambient_dimension,
        args.target_dimension,
        args.neighbors,
        args.distance_pairs,
        args.ae_epochs,
        args.ae_batch_size,
        args.reservoir_size,
        args.structure_batch_size,
        args.stress_batch_size,
        args.structure_interval,
        args.stress_pairs,
    )
    if min(positive) < 1:
        raise ValueError("count and dimension arguments must be positive")
    if args.ambient_dimension < args.target_dimension:
        raise ValueError("ambient_dimension must be at least target_dimension")
    if not math.isfinite(args.rank_temperature) or args.rank_temperature <= 0:
        raise ValueError("rank_temperature must be finite and positive")
    if not math.isfinite(args.tie_rtol) or args.tie_rtol < 0:
        raise ValueError("tie_rtol must be finite and non-negative")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
