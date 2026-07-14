#!/usr/bin/env python3
"""Evaluate topology preservation for large, precomputed embeddings.

The full reference matrix and layouts are opened as memory maps.  Only the common
sample for a seed is materialized, making the evaluator suitable for million-row
corpora.  The reference input is expected to have already been row-L2 normalized.

Example
-------
::

    conda run -n rapids-26.06 python experiments/evaluate_large_topology.py \
      --input artifacts/arxiv/prepared.npy \
      --layout dire=artifacts/arxiv/dire_d2.npy \
      --layout umap=artifacts/arxiv/umap_d2.npy \
      --layout autoencoder=artifacts/arxiv/autoencoder_d2.npy \
      --labels /home/igor/devel/dire-rapids-arxiv/data/meta.parquet \
      --label-column primary_category \
      --source-indices artifacts/arxiv/source_indices.npy \
      --output-dir artifacts/arxiv/topology

Outputs are ``raw.json`` (including Betti curves and complete provenance),
``raw.csv`` (one metric row per seed and method), ``aggregate.csv``, and the exact
sample-index arrays used for every seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import cdist, pdist

DEFAULT_DIRE_REPO = Path(__file__).resolve().parents[2] / "dire-rapids"
RAW_COLUMNS = (
    "seed",
    "method",
    "sample_size",
    "sample_index_sha256",
    "source_sample_index_sha256",
    "dtw_beta0",
    "dtw_beta1",
    "reference_diameter_before_scaling",
    "layout_diameter_before_scaling",
    "reference_longest_h0",
    "reference_top5_median_h0",
    "reference_gini_h0",
    "reference_finite_h0_bars",
    "layout_longest_h0",
    "layout_top5_median_h0",
    "layout_gini_h0",
    "layout_finite_h0_bars",
)
AGGREGATE_METRICS = (
    "dtw_beta0",
    "dtw_beta1",
    "reference_longest_h0",
    "reference_top5_median_h0",
    "reference_gini_h0",
    "layout_longest_h0",
    "layout_top5_median_h0",
    "layout_gini_h0",
)


@dataclass(frozen=True)
class TopologyBackend:
    """Lazy topology functions plus their exact implementation provenance."""

    curve: Callable
    persistence: Callable
    dtw: Callable
    provenance: dict


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    """Hash indices in a canonical, platform-independent integer representation."""
    values = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _git_revision(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_topology_backend(
    dire_repo: str | Path,
    *,
    expected_dire_revision: str | None = None,
    expected_betti_sha256: str | None = None,
) -> TopologyBackend:
    """Load the pinned DiRe Betti implementation, ripser, and fastdtw lazily."""
    repo = Path(dire_repo).expanduser().resolve()
    revision = _git_revision(repo)
    if expected_dire_revision and (
        revision is None or not revision.startswith(expected_dire_revision)
    ):
        raise RuntimeError(
            "DiRe revision mismatch: "
            f"expected {expected_dire_revision!r}, found {revision!r} in {repo}"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import dire_rapids
    from dire_rapids import betti_curve
    from fastdtw import fastdtw
    from ripser import ripser

    implementation = Path(inspect.getfile(betti_curve.compute_betti_curve_ripser)).resolve()
    implementation_sha256 = sha256_file(implementation)
    if expected_betti_sha256 and implementation_sha256 != expected_betti_sha256.lower():
        raise RuntimeError(
            "Betti implementation hash mismatch: "
            f"expected {expected_betti_sha256}, found {implementation_sha256}"
        )

    return TopologyBackend(
        curve=betti_curve.compute_betti_curve_ripser,
        persistence=ripser,
        dtw=fastdtw,
        provenance={
            "dire_repo": str(repo),
            "dire_revision": revision,
            "dire_rapids_version": getattr(dire_rapids, "__version__", "unknown"),
            "betti_implementation": str(implementation),
            "betti_implementation_sha256": implementation_sha256,
            "ripser_version": _package_version("ripser"),
            "fastdtw_version": _package_version("fastdtw"),
        },
    )


def _canonical_labels(labels) -> np.ndarray:
    """Convert possibly mixed labels into stable strings for grouping."""
    values = np.asarray(labels).reshape(-1)
    result = np.empty(values.shape[0], dtype=object)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            result[i] = "<missing>"
        elif isinstance(value, bytes):
            result[i] = value.decode(errors="replace")
        else:
            result[i] = str(value)
    return result.astype(str)


def sample_indices(
    n_rows: int,
    sample_size: int,
    seed: int,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Return an exact-size deterministic uniform or stratified sample.

    When the requested sample can cover every observed class, one point per class is
    reserved before the remaining quota is allocated proportionally.  Largest-remainder
    allocation keeps the final sample size exact, unlike rounded per-class quotas.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(n_rows, size=sample_size, replace=False).astype(np.int64))

    labels = _canonical_labels(labels)
    if labels.shape[0] != n_rows:
        raise ValueError(f"labels have {labels.shape[0]} rows, expected {n_rows}")
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    quotas = np.zeros(classes.shape[0], dtype=np.int64)
    if sample_size >= classes.shape[0]:
        quotas[:] = 1
    remaining = sample_size - int(quotas.sum())
    capacity = counts - quotas
    if remaining:
        expected = remaining * capacity / capacity.sum()
        addition = np.floor(expected).astype(np.int64)
        addition = np.minimum(addition, capacity)
        quotas += addition
        remainder = sample_size - int(quotas.sum())
        fractions = expected - addition
        order = np.argsort(-fractions, kind="stable")
        while remainder:
            progressed = False
            for class_index in order:
                if quotas[class_index] < counts[class_index]:
                    quotas[class_index] += 1
                    remainder -= 1
                    progressed = True
                    if remainder == 0:
                        break
            if not progressed:  # defensive: capacity must equal n_rows - base quota
                raise RuntimeError("could not allocate the requested stratified sample")

    selected = []
    for class_index, quota in enumerate(quotas):
        if quota:
            pool = np.flatnonzero(inverse == class_index)
            selected.append(rng.choice(pool, size=int(quota), replace=False))
    indices = np.concatenate(selected).astype(np.int64, copy=False)
    if indices.shape[0] != sample_size:
        raise RuntimeError("stratified sampler returned the wrong number of rows")
    return np.sort(indices)


def max_pairwise_distance(points: np.ndarray, block_size: int = 512) -> float:
    """Compute exact Euclidean diameter with bounded temporary memory."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty 2-D array")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite values")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if points.shape[0] == 1:
        return 0.0

    diameter = 0.0
    for start in range(0, points.shape[0], block_size):
        left = points[start : start + block_size]
        if left.shape[0] > 1:
            diameter = max(diameter, float(pdist(left, metric="euclidean").max()))
        for other_start in range(start + block_size, points.shape[0], block_size):
            right = points[other_start : other_start + block_size]
            diameter = max(
                diameter,
                float(cdist(left, right, metric="euclidean").max()),
            )
    return diameter


def rescale_unit_diameter(points: np.ndarray, block_size: int = 512) -> tuple[np.ndarray, float]:
    """Copy a sampled cloud to float32 and rescale its Euclidean diameter to one."""
    points = np.asarray(points)
    diameter = max_pairwise_distance(points, block_size=block_size)
    if not np.isfinite(diameter) or diameter <= np.finfo(np.float64).eps:
        raise ValueError("sampled cloud has zero or invalid diameter")
    return np.ascontiguousarray(points / diameter, dtype=np.float32), diameter


def finite_h0_persistences(diagram: np.ndarray) -> np.ndarray:
    """Extract positive finite persistence lengths from an H0 diagram."""
    diagram = np.asarray(diagram, dtype=np.float64)
    if diagram.size == 0:
        return np.empty(0, dtype=np.float64)
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError("persistence diagram must have shape (n_bars, 2)")
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    persistence = finite[:, 1] - finite[:, 0]
    return persistence[persistence > 0]


def gini(values: np.ndarray) -> float:
    """Return the Gini coefficient of non-negative values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Gini input must be one-dimensional")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Gini input must contain finite non-negative values")
    if values.size == 0 or values.sum() == 0:
        return 0.0
    ordered = np.sort(values)
    cumulative = np.cumsum(ordered)
    return float(
        1.0 - 2.0 * (cumulative.sum() - cumulative[-1] / 2.0) / (ordered.size * cumulative[-1])
    )


def islandness(diagram: np.ndarray) -> dict:
    """Compute longest-bar, mean-top-five/median, and Gini H0 diagnostics."""
    bars = finite_h0_persistences(diagram)
    if bars.size == 0:
        return {
            "longest_h0": 0.0,
            "top5_median_h0": 0.0,
            "gini_h0": 0.0,
            "finite_h0_bars": 0,
        }
    descending = np.sort(bars)[::-1]
    top = descending[:5]
    median = float(np.median(descending))
    return {
        "longest_h0": float(descending[0]),
        "top5_median_h0": float(top.mean() / median) if median > 0 else 0.0,
        "gini_h0": gini(descending),
        "finite_h0_bars": int(descending.size),
    }


def _curve_arrays(result: dict) -> tuple[np.ndarray, np.ndarray]:
    try:
        beta0 = np.asarray(result["beta_0"], dtype=np.float64).reshape(-1)
        beta1 = np.asarray(result["beta_1"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError) as exc:
        raise ValueError("Betti backend did not return beta_0 and beta_1") from exc
    if beta0.size == 0 or beta1.size == 0:
        raise ValueError("Betti backend returned an empty curve")
    if beta0.size != beta1.size:
        raise ValueError("Betti backend returned curves with different lengths")
    if not np.all(np.isfinite(beta0)) or not np.all(np.isfinite(beta1)):
        raise ValueError("Betti backend returned non-finite curve values")
    return beta0, beta1


def _filtration_array(result: dict, curve_size: int) -> np.ndarray:
    values = np.asarray(
        result.get("filtration_values", np.arange(curve_size)), dtype=np.float64
    ).reshape(-1)
    if values.size != curve_size or not np.all(np.isfinite(values)):
        raise ValueError("Betti backend returned an invalid filtration axis")
    return values


def _persistence_diagram(result) -> np.ndarray:
    diagrams = result["dgms"] if isinstance(result, dict) else result
    if not diagrams:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(diagrams[0], dtype=np.float64)


def _dtw_distance(function: Callable, left: np.ndarray, right: np.ndarray) -> float:
    result = function(left.astype(float), right.astype(float))
    distance = result[0] if isinstance(result, tuple) else result
    distance = float(distance)
    if not np.isfinite(distance):
        raise ValueError("fastdtw returned a non-finite distance")
    return distance


def evaluate_seed(
    reference,
    layouts: dict[str, np.ndarray],
    indices: np.ndarray,
    seed: int,
    backend: TopologyBackend,
    *,
    n_steps: int = 50,
    diameter_block_size: int = 512,
    source_indices: np.ndarray | None = None,
) -> list[dict]:
    """Evaluate every layout on one shared sampled point cloud."""
    indices = np.asarray(indices, dtype=np.int64)
    reference_sample, reference_diameter = rescale_unit_diameter(
        np.asarray(reference[indices]), block_size=diameter_block_size
    )
    reference_curve = backend.curve(reference_sample, n_steps=n_steps, maxdim=1)
    reference_beta0, reference_beta1 = _curve_arrays(reference_curve)
    reference_filtration = _filtration_array(reference_curve, reference_beta0.size)
    reference_diagram = _persistence_diagram(backend.persistence(reference_sample, maxdim=1))
    reference_islandness = islandness(reference_diagram)
    source_sample = source_indices[indices] if source_indices is not None else indices

    records = []
    for method, layout in sorted(layouts.items()):
        layout_sample, layout_diameter = rescale_unit_diameter(
            np.asarray(layout[indices]), block_size=diameter_block_size
        )
        layout_curve = backend.curve(layout_sample, n_steps=n_steps, maxdim=1)
        layout_beta0, layout_beta1 = _curve_arrays(layout_curve)
        layout_filtration = _filtration_array(layout_curve, layout_beta0.size)
        layout_diagram = _persistence_diagram(backend.persistence(layout_sample, maxdim=1))
        layout_islandness = islandness(layout_diagram)
        record = {
            "seed": int(seed),
            "method": method,
            "sample_size": int(indices.size),
            "sample_index_sha256": index_sha256(indices),
            "source_sample_index_sha256": index_sha256(source_sample),
            "dtw_beta0": _dtw_distance(backend.dtw, reference_beta0, layout_beta0),
            "dtw_beta1": _dtw_distance(backend.dtw, reference_beta1, layout_beta1),
            "reference_diameter_before_scaling": float(reference_diameter),
            "layout_diameter_before_scaling": float(layout_diameter),
            **{f"reference_{key}": value for key, value in reference_islandness.items()},
            **{f"layout_{key}": value for key, value in layout_islandness.items()},
            "reference_beta0": reference_beta0.tolist(),
            "reference_beta1": reference_beta1.tolist(),
            "reference_filtration": reference_filtration.tolist(),
            "layout_beta0": layout_beta0.tolist(),
            "layout_beta1": layout_beta1.tolist(),
            "layout_filtration": layout_filtration.tolist(),
        }
        records.append(record)
    return records


def aggregate_records(records: list[dict]) -> list[dict]:
    """Aggregate per-seed records by method using sample mean and standard deviation."""
    methods = sorted({record["method"] for record in records})
    aggregate = []
    for method in methods:
        selected = [record for record in records if record["method"] == method]
        row = {"method": method, "n_seeds": len(selected)}
        for metric in AGGREGATE_METRICS:
            values = np.asarray([record[metric] for record in selected], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        aggregate.append(row)
    return aggregate


def _array_provenance(path: Path, array: np.ndarray) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _load_source_indices(
    path: str | Path | None, n_rows: int
) -> tuple[np.ndarray | None, dict | None]:
    if path is None:
        return None, None
    path = Path(path).expanduser().resolve()
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 1 or values.shape[0] != n_rows:
        raise ValueError(f"source indices must have shape ({n_rows},), found {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("source indices must have integer dtype")
    if values.size and np.min(values) < 0:
        raise ValueError("source indices must be non-negative")
    return values, _array_provenance(path, values)


def _load_label_values(path: Path, column: str | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        if column:
            raise ValueError("--label-column is only valid for parquet labels")
        # Label arrays commonly use object strings.  This is the only input for which
        # pickle is allowed; reference, layout, and index arrays remain pickle-disabled.
        values = np.load(path, allow_pickle=True)
        if values.ndim != 1:
            raise ValueError("NPY labels must be one-dimensional")
        return values
    if path.suffix.lower() in {".parquet", ".pq"}:
        if not column:
            raise ValueError("--label-column is required for parquet labels")
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "pandas with parquet support is required for parquet labels"
            ) from exc
        frame = pd.read_parquet(path, columns=[column])
        return frame[column].to_numpy()
    raise ValueError("labels must be a .npy, .parquet, or .pq file")


def load_labels(
    path: str | Path | None,
    *,
    column: str | None,
    n_rows: int,
    source_indices: np.ndarray | None,
    alignment: str = "auto",
) -> tuple[np.ndarray | None, dict | None]:
    """Load labels aligned to prepared rows, optionally through source indices."""
    if path is None:
        if column:
            raise ValueError("--label-column requires --labels")
        return None, None
    path = Path(path).expanduser().resolve()
    values = _load_label_values(path, column)
    if alignment not in {"auto", "prepared", "source"}:
        raise ValueError("labels alignment must be auto, prepared, or source")
    resolved_alignment = alignment
    if alignment == "auto":
        resolved_alignment = "prepared" if values.shape[0] == n_rows else "source"

    if resolved_alignment == "prepared":
        if values.shape[0] != n_rows:
            raise ValueError(
                f"prepared-aligned labels have {values.shape[0]} rows, expected {n_rows}"
            )
        aligned = values
    else:
        if source_indices is None:
            raise ValueError("source-aligned labels require --source-indices")
        if source_indices.size and int(np.max(source_indices)) >= values.shape[0]:
            raise ValueError("source index falls outside the label array")
        aligned = np.asarray(values[source_indices])

    provenance = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "column": column,
        "source_rows": int(values.shape[0]),
        "alignment": resolved_alignment,
    }
    return _canonical_labels(aligned), provenance


def _parse_layouts(specifications: list[str]) -> dict[str, Path]:
    layouts = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"layout must be METHOD=PATH, found {specification!r}")
        method, raw_path = specification.split("=", 1)
        method = method.strip()
        if not method or method in layouts:
            raise ValueError(f"layout method must be non-empty and unique: {method!r}")
        layouts[method] = Path(raw_path).expanduser().resolve()
    return layouts


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_reference_l2(reference, indices: np.ndarray, atol: float) -> float:
    if atol < 0:
        raise ValueError("L2 tolerance must be non-negative")
    sample = np.asarray(reference[indices], dtype=np.float64)
    if not np.all(np.isfinite(sample)):
        raise ValueError("reference sample contains non-finite values")
    errors = np.abs(np.linalg.norm(sample, axis=1) - 1.0)
    maximum = float(errors.max(initial=0.0))
    if maximum > atol:
        raise ValueError(
            "reference input is not row-L2 normalized on the evaluation sample: "
            f"maximum norm error {maximum:.6g} exceeds {atol:.6g}"
        )
    return maximum


def run(args, backend: TopologyBackend | None = None) -> dict:
    """Run the evaluator and return the JSON payload written to disk."""
    input_path = Path(args.input).expanduser().resolve()
    reference = np.load(input_path, mmap_mode="r", allow_pickle=False)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("reference input must be a 2-D array with at least two rows")
    if not np.issubdtype(reference.dtype, np.floating):
        raise ValueError("reference input must have floating dtype")

    layout_paths = _parse_layouts(args.layout)
    layouts = {}
    layout_provenance = {}
    target_dimension = None
    for method, path in layout_paths.items():
        layout = np.load(path, mmap_mode="r", allow_pickle=False)
        if layout.ndim != 2 or layout.shape[0] != reference.shape[0]:
            raise ValueError(
                f"layout {method!r} has shape {layout.shape}; expected "
                f"({reference.shape[0]}, target_dimension)"
            )
        if not np.issubdtype(layout.dtype, np.floating):
            raise ValueError(f"layout {method!r} must have floating dtype")
        if target_dimension is None:
            target_dimension = int(layout.shape[1])
            if target_dimension < 1:
                raise ValueError("layouts must have at least one target dimension")
        elif layout.shape[1] != target_dimension:
            raise ValueError(
                f"layout {method!r} has target dimension {layout.shape[1]}, "
                f"expected {target_dimension}"
            )
        layouts[method] = layout
        layout_provenance[method] = _array_provenance(path, layout)

    source_indices, source_provenance = _load_source_indices(
        args.source_indices, reference.shape[0]
    )
    labels, label_provenance = load_labels(
        args.labels,
        column=args.label_column,
        n_rows=reference.shape[0],
        source_indices=source_indices,
        alignment=args.labels_alignment,
    )
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    samples = {
        seed: sample_indices(reference.shape[0], args.sample_size, seed, labels) for seed in seeds
    }
    checked_indices = np.unique(np.concatenate(list(samples.values())))
    l2_error = _validate_reference_l2(reference, checked_indices, args.l2_atol)

    if backend is None:
        backend = load_topology_backend(
            args.dire_repo,
            expected_dire_revision=args.expected_dire_revision,
            expected_betti_sha256=args.expected_betti_sha256,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_provenance = {}
    for seed, indices in samples.items():
        sample_path = output_dir / f"sample_indices_seed{seed}.npy"
        np.save(sample_path, indices)
        item = {
            "path": str(sample_path),
            "sha256": sha256_file(sample_path),
            "index_sha256": index_sha256(indices),
            "size": int(indices.size),
            "indices": indices.tolist(),
        }
        if source_indices is not None:
            selected_source = np.asarray(source_indices[indices], dtype=np.int64)
            source_path = output_dir / f"sample_source_indices_seed{seed}.npy"
            np.save(source_path, selected_source)
            item.update(
                {
                    "source_path": str(source_path),
                    "source_sha256": sha256_file(source_path),
                    "source_index_sha256": index_sha256(selected_source),
                    "source_indices": selected_source.tolist(),
                }
            )
        sample_provenance[str(seed)] = item

    records = []
    for seed in seeds:
        records.extend(
            evaluate_seed(
                reference,
                layouts,
                samples[seed],
                seed,
                backend,
                n_steps=args.n_steps,
                diameter_block_size=args.diameter_block_size,
                source_indices=source_indices,
            )
        )
    aggregate = aggregate_records(records)
    aggregate_columns = ["method", "n_seeds"] + [
        f"{metric}_{suffix}" for metric in AGGREGATE_METRICS for suffix in ("mean", "std")
    ]

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "sample_size": int(args.sample_size),
            "seeds": list(seeds),
            "sampling": "stratified" if labels is not None else "uniform",
            "target_dimension": target_dimension,
            "n_steps": int(args.n_steps),
            "diameter_block_size": int(args.diameter_block_size),
            "l2_atol": float(args.l2_atol),
        },
        "provenance": {
            "harness": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "reference": _array_provenance(input_path, reference),
            "layouts": layout_provenance,
            "labels": label_provenance,
            "source_indices": source_provenance,
            "samples": sample_provenance,
            "row_l2_maximum_sample_error": l2_error,
            "backend": backend.provenance,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "records": records,
        "aggregate": aggregate,
    }
    (output_dir / "raw.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "raw.csv", records, list(RAW_COLUMNS))
    _write_csv(output_dir / "aggregate.csv", aggregate, aggregate_columns)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="prepared row-L2-normalized NPY")
    parser.add_argument(
        "--layout",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="aligned layout NPY; repeat once per method",
    )
    parser.add_argument("--labels", help="optional aligned/source .npy or parquet labels")
    parser.add_argument("--label-column", help="column to read from parquet labels")
    parser.add_argument(
        "--labels-alignment",
        choices=("auto", "prepared", "source"),
        default="auto",
        help="whether labels align to prepared rows or original source rows",
    )
    parser.add_argument(
        "--source-indices",
        help="optional NPY mapping each prepared row to its original source row",
    )
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--diameter-block-size", type=int, default=512)
    parser.add_argument("--l2-atol", type=float, default=5e-4)
    parser.add_argument("--dire-repo", default=str(DEFAULT_DIRE_REPO))
    parser.add_argument("--expected-dire-revision")
    parser.add_argument("--expected-betti-sha256")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(
        f"wrote {len(payload['records'])} raw rows and "
        f"{len(payload['aggregate'])} aggregate rows to {Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
