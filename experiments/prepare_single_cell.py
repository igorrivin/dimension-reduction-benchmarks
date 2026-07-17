#!/usr/bin/env python3
"""Prepare reproducible single-cell inputs for reducer benchmarks.

The module intentionally imports Census, AnnData, Scanpy, pandas, and PyArrow
only inside the functions which need them.  This keeps the ordinary benchmark
environment importable without the separate single-cell preparation stack.

Three sources are supported:

* ``tabula-sapiens`` queries a pinned CELLxGENE Census release, resolves the
  collection's dataset IDs, and downloads only the deterministically selected
  observations from the raw-count layer; and
* ``local-h5ad`` reads an already-local AnnData file and never fetches data.
* ``tenx-brain-1m3`` reads a local 10x HDF5 matrix or atomically caches the
  official remote matrix before reading it.

The default scientific pipeline is raw counts -> batch-aware Seurat-v3 HVGs ->
library-size normalization -> log1p -> optional feature scaling -> non-whitened
PCA.  Labels may be retained for evaluation or used to stratify sampling, but
cell type and tissue labels never enter HVG selection.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import platform
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA_VERSION = 1
RAW_CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_CENSUS_VERSION = "2025-11-08"
TABULA_SAPIENS_COLLECTION_ID = "e5f58829-1a66-40b5-a624-9046778e74f5"
TABULA_SAPIENS_DATASET_TITLE = "Tabula Sapiens - All Cells"
TENX_BRAIN_DATASET_ID = "10x-1.3m-mouse-brain"
TENX_BRAIN_ASSAY = "10x Universal 3' (chemistry unresolved)"
TENX_BRAIN_MATRIX_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/"
    "1M_neurons_filtered_gene_bc_matrices_h5.h5"
)
CENSUS_PILOT_CELLS = 100_000
DEFAULT_SAMPLE_SEED = 20_260_715
DEFAULT_N_HVGS = 2_000
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_TARGET_SUM = 10_000.0
DEFAULT_SCALE_CLIP = 10.0
DEFAULT_HVG_SPAN = 0.5
DEFAULT_MAX_DENSE_BYTES = 32 * 1024**3
DEFAULT_METADATA_FIELDS = (
    "cell_type",
    "cell_type_ontology_term_id",
    "donor_id",
    "tissue",
    "tissue_general",
    "assay",
    "dataset_id",
    "soma_joinid",
)
DEFAULT_CENSUS_STRATIFY_FIELDS = ("donor_id", "cell_type")
DEFAULT_HVG_BATCH_FIELDS = ("donor_id", "assay")
FORBIDDEN_HVG_BATCH_FIELDS = frozenset(
    {"cell_type", "cell_type_ontology_term_id", "tissue", "tissue_general"}
)
QC_COLUMNS = ("total_counts", "n_genes_by_counts", "pct_counts_mt")
PINNED_RELEASE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SAFE_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RAW_H5AD_GENERATION_PATTERN = re.compile(r"^selected_raw\.[0-9a-f]{64}\.h5ad$")


@dataclass
class LoadedCells:
    """A selected raw-count AnnData object and its source-level provenance."""

    adata: Any
    source: dict[str, Any]
    sampling: dict[str, Any]
    qc: dict[str, Any]


@dataclass
class PreparedRepresentations:
    """In-memory products ready for atomic artifact serialization."""

    pca: np.ndarray
    log_hvg: np.ndarray | None
    metadata: Any
    genes: Any
    pipeline: dict[str, Any]


class HVGLoessDiagnosticsError(RuntimeError):
    """A batch-specific Seurat-v3 LOESS preflight failure."""

    def __init__(self, message: str, diagnostics: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.diagnostics = tuple(dict(item) for item in diagnostics)


def _require_module(module_name: str, purpose: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"{module_name!r} is required to {purpose}; install "
            "requirements/single-cell-prep.txt in the preparation environment"
        ) from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert NumPy/pandas-adjacent values into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        with contextlib.suppress(TypeError, ValueError):
            return value.isoformat()
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_int64(values: Sequence[int] | np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_strings(values: Sequence[Any] | np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _finish_stage(
    timings: dict[str, float | None] | None,
    statuses: dict[str, str] | None,
    name: str,
    started: float,
    *,
    status: str = "executed",
) -> None:
    if timings is not None:
        timings[name] = round(time.perf_counter() - started, 6)
    if statuses is not None:
        statuses[name] = status


def _set_skipped_stage(
    timings: dict[str, float | None] | None,
    statuses: dict[str, str] | None,
    name: str,
    status: str,
) -> None:
    if timings is not None:
        timings[name] = None
    if statuses is not None:
        statuses[name] = status


def _update_digest_array(
    digest: Any,
    values: Any,
    *,
    dtype: str,
    chunk_elements: int = 1_000_000,
) -> None:
    flat = np.asarray(values).reshape(-1)
    digest.update(str(flat.size).encode("ascii"))
    for start in range(0, flat.size, chunk_elements):
        normalized = np.ascontiguousarray(flat[start : start + chunk_elements], dtype=dtype)
        digest.update(memoryview(normalized).cast("B"))


def _canonicalize_raw_matrix(adata: Any) -> str:
    sparse = _require_module("scipy.sparse", "canonicalize a raw checkpoint matrix")
    if not sparse.issparse(adata.X):
        if getattr(adata.X, "format", None) == "csr":
            return "csr"
        return "dense"
    matrix = adata.X.tocsr(copy=False)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    adata.X = matrix
    return "csr"


def _matrix_semantic_sha256(matrix: Any) -> str:
    """Hash matrix values independent of sparse index and numeric storage widths."""

    sparse = _require_module("scipy.sparse", "hash a raw count matrix")
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes({"shape": list(map(int, matrix.shape))}))
    if sparse.issparse(matrix) or getattr(matrix, "format", None) == "csr":
        digest.update(b"csr-row-blocks-v1")
        block_rows = 4_096
        for start in range(0, int(matrix.shape[0]), block_rows):
            stop = min(start + block_rows, int(matrix.shape[0]))
            block = matrix[start:stop]
            if not sparse.issparse(block):
                raise TypeError("CSR-like matrix row slice did not return a sparse matrix")
            already_canonical = (
                sparse.isspmatrix_csr(block)
                and block.has_canonical_format
                and block.has_sorted_indices
                and not np.any(block.data == 0)
            )
            if already_canonical:
                canonical = block
            else:
                canonical = block.tocsr(copy=True)
                canonical.sum_duplicates()
                canonical.sort_indices()
                canonical.eliminate_zeros()
            digest.update(_canonical_json_bytes({"row_start": start, "row_stop": stop}))
            _update_digest_array(digest, canonical.indptr, dtype="<i8")
            _update_digest_array(digest, canonical.indices, dtype="<i8")
            _update_digest_array(digest, canonical.data, dtype="<f8")
    else:
        digest.update(b"dense")
        _update_digest_array(digest, np.asarray(matrix), dtype="<f8")
    return digest.hexdigest()


def _logical_values_sha256(values: Sequence[Any] | np.ndarray) -> str:
    return _sha256_strings([_canonical_stratum_value(value) for value in values])


def _stable_feature_ids(adata: Any) -> np.ndarray:
    if "feature_id" in adata.var.columns:
        return np.asarray(adata.var["feature_id"].astype(str), dtype=str)
    return np.asarray(adata.var_names.astype(str), dtype=str)


def _selected_raw_semantic_record(
    adata: Any,
    *,
    required_obs_columns: Sequence[str],
) -> dict[str, Any]:
    missing = [column for column in required_obs_columns if column not in adata.obs.columns]
    if missing:
        raise ValueError(f"selected raw checkpoint is missing observation columns: {missing}")
    _validate_unique_observation_ids(adata)
    feature_ids = _stable_feature_ids(adata)
    if np.unique(feature_ids).size != feature_ids.size:
        raise ValueError("selected raw checkpoint feature identifiers must be unique")
    storage = _canonicalize_raw_matrix(adata)
    if storage != "csr":
        raise ValueError("selected raw checkpoint matrix must be sparse CSR")
    matrix_sha256 = _matrix_semantic_sha256(adata.X)
    feature_names = (
        np.asarray(adata.var["feature_name"].astype(str), dtype=str)
        if "feature_name" in adata.var.columns
        else np.asarray(adata.var_names.astype(str), dtype=str)
    )
    record = {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "storage": storage,
        "matrix_sha256": matrix_sha256,
        "obs_names_sha256": _sha256_strings(adata.obs_names.astype(str)),
        "var_names_sha256": _sha256_strings(adata.var_names.astype(str)),
        "feature_ids_sha256": _sha256_strings(feature_ids),
        "feature_names_sha256": _sha256_strings(feature_names),
        "required_obs_columns": list(required_obs_columns),
        "obs_columns_sha256": {
            column: _logical_values_sha256(np.asarray(adata.obs[column], dtype=object))
            for column in required_obs_columns
        },
    }
    if "soma_joinid" in adata.obs.columns:
        record["soma_joinids_sha256"] = _sha256_int64(
            np.asarray(adata.obs["soma_joinid"], dtype=np.int64)
        )
    record["semantic_sha256"] = _sha256_json(record)
    return record


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
    encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False).encode(
        "utf-8"
    )
    _atomic_bytes(path, encoded + b"\n")


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_parquet(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def validate_pinned_census_version(version: str) -> str:
    """Reject moving Census aliases and malformed release identifiers."""

    normalized = str(version).strip()
    if normalized.lower() in {"latest", "stable"} or not PINNED_RELEASE_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "Census version must be an exact YYYY-MM-DD release, not 'stable' or 'latest'"
        )
    return normalized


def _canonical_stratum_value(value: Any) -> str:
    if value is None:
        return "<NA>"
    with contextlib.suppress(TypeError, ValueError):
        if bool(np.asarray(value).ndim == 0 and value != value):
            return "<NA>"
    return str(value)


def composite_strata(frame: Any, fields: Sequence[str], separator: str = "|") -> np.ndarray:
    """Return unambiguous, stable composite stratum labels from metadata."""

    fields = tuple(fields)
    if not fields:
        return np.full(len(frame), "__all__", dtype=object)
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ValueError(f"metadata is missing required stratum columns: {missing}")
    columns: list[np.ndarray] = []
    for field in fields:
        values = np.asarray(frame[field], dtype=object)
        canonical = np.asarray([_canonical_stratum_value(value) for value in values], dtype=object)
        if any(separator in value for value in canonical):
            raise ValueError(
                f"metadata column {field!r} contains reserved stratum separator {separator!r}"
            )
        columns.append(canonical)
    result = columns[0].copy()
    for column in columns[1:]:
        result = np.char.add(np.char.add(result.astype(str), separator), column.astype(str))
    return np.asarray(result, dtype=object)


def _largest_remainder_allocation(counts: np.ndarray, sample_size: int) -> np.ndarray:
    """Allocate an exact stratified sample with representation when possible."""

    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts < 1):
        raise ValueError("stratum counts must be a non-empty vector of positive integers")
    population = int(counts.sum())
    if sample_size < 1 or sample_size > population:
        raise ValueError("sample size must lie between one and the population size")
    if sample_size == population:
        return counts.copy()

    ideals = sample_size * counts.astype(np.float64) / population
    allocation = np.floor(ideals).astype(np.int64)
    if sample_size >= counts.size:
        allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)

    while int(allocation.sum()) > sample_size:
        lower_bound = 1 if sample_size >= counts.size else 0
        removable = np.flatnonzero(allocation > lower_bound)
        if removable.size == 0:
            raise RuntimeError("unable to reduce stratified allocation to requested size")
        excess = allocation[removable] - ideals[removable]
        # Stable tie break: the lexicographically later group loses first.
        chosen = removable[np.lexsort((-removable, -excess))[0]]
        allocation[chosen] -= 1

    while int(allocation.sum()) < sample_size:
        eligible = np.flatnonzero(allocation < counts)
        if eligible.size == 0:
            raise RuntimeError("unable to fill stratified allocation to requested size")
        residual = ideals[eligible] - allocation[eligible]
        # Stable tie break: the lexicographically earlier group gains first.
        chosen = eligible[np.lexsort((eligible, -residual))[0]]
        allocation[chosen] += 1
    return allocation


def deterministic_stratified_subsample(
    strata: Sequence[Any] | np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select exact, reproducible row indices and report every stratum allocation.

    When the sample can represent every stratum, largest-remainder allocation
    gives each non-empty stratum at least one row.  If there are more strata than
    requested rows, a seeded, population-weighted choice determines which strata
    are represented.  Rows are sampled independently within each chosen stratum
    and returned in source order.
    """

    labels = np.asarray([_canonical_stratum_value(value) for value in strata], dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("strata must be a non-empty one-dimensional sequence")
    if sample_size < 1 or sample_size > labels.size:
        raise ValueError("sample size must lie between one and the number of rows")
    unique, inverse, counts = np.unique(labels.astype(str), return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)

    if sample_size < unique.size:
        chosen_groups = rng.choice(
            unique.size,
            size=sample_size,
            replace=False,
            p=counts.astype(np.float64) / counts.sum(),
        )
        allocation = np.zeros(unique.size, dtype=np.int64)
        allocation[chosen_groups] = 1
    else:
        allocation = _largest_remainder_allocation(counts, sample_size)

    selected_parts: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for group_index, (label, count, take) in enumerate(
        zip(unique, counts, allocation, strict=True)
    ):
        candidates = np.flatnonzero(inverse == group_index)
        if take:
            chosen = rng.choice(candidates, size=int(take), replace=False)
            selected_parts.append(np.asarray(chosen, dtype=np.int64))
        details.append(
            {
                "stratum": str(label),
                "candidate_cells": int(count),
                "selected_cells": int(take),
                "represented": bool(take),
            }
        )
    selected = np.sort(np.concatenate(selected_parts)).astype(np.int64, copy=False)
    if selected.size != sample_size or np.unique(selected).size != sample_size:
        raise RuntimeError("stratified sampler did not produce the requested unique rows")
    return selected, details


def validate_raw_counts(matrix: Any, integer_tolerance: float = 1e-6) -> dict[str, Any]:
    """Validate the count domain without forcing a sparse matrix to dense."""

    sparse = _require_module("scipy.sparse", "validate sparse count matrices")
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("raw count matrix must be two-dimensional")
    chunk_values = 1_000_000
    if sparse.issparse(matrix):
        value_arrays = (np.asarray(matrix.data).reshape(-1),)
        storage = "sparse"
    elif getattr(matrix, "format", None) == "csr":
        value_arrays = (
            np.asarray(matrix[start : min(start + 4_096, matrix.shape[0])].data).reshape(-1)
            for start in range(0, int(matrix.shape[0]), 4_096)
        )
        storage = "sparse"
    else:
        value_arrays = (np.asarray(matrix).reshape(-1),)
        storage = "dense"
    for values in value_arrays:
        for start in range(0, values.size, chunk_values):
            chunk = values[start : start + chunk_values]
            if not np.all(np.isfinite(chunk)):
                raise ValueError("raw count matrix contains non-finite values")
            if np.any(chunk < 0):
                raise ValueError("raw count matrix contains negative values")
            if np.issubdtype(chunk.dtype, np.integer):
                continue
            numeric = chunk.astype(np.float64, copy=False)
            if np.any(np.abs(numeric - np.rint(numeric)) > integer_tolerance):
                raise ValueError("raw count matrix contains non-integer-like values")
    return {
        "shape": [int(size) for size in matrix.shape],
        "storage": storage,
        "dtype": str(matrix.dtype),
        "finite": True,
        "nonnegative": True,
        "integer_like": True,
        "integer_tolerance": float(integer_tolerance),
    }


def compute_count_qc(matrix: Any, gene_names: Sequence[Any]) -> dict[str, np.ndarray]:
    """Compute per-cell QC covariates from the untouched raw selected counts."""

    sparse = _require_module("scipy.sparse", "compute sparse count QC")
    names = np.asarray([str(value) for value in gene_names], dtype=str)
    if matrix.ndim != 2 or matrix.shape[1] != names.size:
        raise ValueError("gene names must match the count matrix columns")
    total = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float64)
    if sparse.issparse(matrix):
        detected = np.asarray((matrix > 0).sum(axis=1)).reshape(-1).astype(np.int64)
    else:
        detected = np.count_nonzero(np.asarray(matrix) > 0, axis=1).astype(np.int64)
    mitochondrial = np.char.startswith(np.char.upper(names), "MT-")
    if np.any(mitochondrial):
        mt_counts = np.asarray(matrix[:, mitochondrial].sum(axis=1)).reshape(-1).astype(np.float64)
    else:
        mt_counts = np.zeros(matrix.shape[0], dtype=np.float64)
    pct_mt = np.divide(
        100.0 * mt_counts,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )
    return {
        "total_counts": total,
        "n_genes_by_counts": detected,
        "pct_counts_mt": pct_mt,
    }


def _gene_names(adata: Any) -> np.ndarray:
    if "feature_name" in adata.var.columns:
        return np.asarray(adata.var["feature_name"].astype(str), dtype=str)
    return np.asarray(adata.var_names.astype(str), dtype=str)


def _attach_qc(adata: Any, qc: dict[str, np.ndarray]) -> None:
    for column in QC_COLUMNS:
        adata.obs[column] = np.asarray(qc[column])


def _qc_summary(qc: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in QC_COLUMNS:
        values = np.asarray(qc[column], dtype=np.float64)
        summary[column] = {
            "min": float(np.min(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
        }
    return summary


def _detection_counts(matrix: Any) -> np.ndarray:
    sparse = _require_module("scipy.sparse", "filter genes by detected-cell count")
    if sparse.issparse(matrix):
        return np.asarray((matrix > 0).sum(axis=0)).reshape(-1).astype(np.int64)
    return np.count_nonzero(np.asarray(matrix) > 0, axis=0).astype(np.int64)


def _sampling_record(
    *,
    candidate_rows: int,
    selected_indices: np.ndarray,
    fields: Sequence[str],
    allocations: list[dict[str, Any]],
    seed: int,
    selected_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    record = {
        "method": "deterministic_stratified_largest_remainder",
        "seed": int(seed),
        "stratify_fields": list(fields),
        "candidate_cells": int(candidate_rows),
        "selected_cells": int(selected_indices.size),
        "selected_source_indices_sha256": _sha256_int64(selected_indices),
        "strata": allocations,
        "represented_strata": int(sum(row["represented"] for row in allocations)),
        "unrepresented_strata": int(sum(not row["represented"] for row in allocations)),
    }
    if selected_ids is not None:
        record["selected_soma_joinids_sha256"] = _sha256_int64(selected_ids)
    return record


def _subset_rows(adata: Any, indices: np.ndarray) -> Any:
    return adata[np.asarray(indices, dtype=np.int64), :].copy()


def _validate_unique_observation_ids(adata: Any) -> None:
    values = np.asarray(adata.obs_names.astype(str), dtype=str)
    if np.unique(values).size != values.size:
        raise ValueError("observation identifiers must be unique")


def load_local_h5ad(
    path: Path,
    *,
    counts_layer: str | None = None,
    max_cells: int | None = None,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    stratify_fields: Sequence[str] = (),
    min_genes: int | None = None,
    max_pct_mt: float | None = None,
    min_cells_per_gene: int | None = 3,
) -> LoadedCells:
    """Load, optionally QC-filter, and deterministically sample a local H5AD."""

    anndata = _require_module("anndata", "read a local H5AD source")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = anndata.read_h5ad(path)
    _validate_unique_observation_ids(adata)
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise ValueError(f"count layer {counts_layer!r} is absent from {path}")
        adata.X = adata.layers[counts_layer].copy()
    count_validation = validate_raw_counts(adata.X)
    original_cells, original_genes = map(int, adata.shape)
    qc_all = compute_count_qc(adata.X, _gene_names(adata))

    keep_cells = np.ones(original_cells, dtype=bool)
    if min_genes is not None:
        if min_genes < 0:
            raise ValueError("min_genes must be nonnegative")
        keep_cells &= qc_all["n_genes_by_counts"] >= min_genes
    if max_pct_mt is not None:
        if not 0 <= max_pct_mt <= 100:
            raise ValueError("max_pct_mt must lie between zero and 100")
        keep_cells &= qc_all["pct_counts_mt"] <= max_pct_mt
    post_cell_qc_indices = np.flatnonzero(keep_cells)
    if not post_cell_qc_indices.size:
        raise ValueError("local cell QC removed every observation")
    adata = _subset_rows(adata, post_cell_qc_indices)

    requested_cells = int(adata.n_obs) if max_cells is None else int(max_cells)
    if requested_cells < 1 or requested_cells > adata.n_obs:
        raise ValueError("max_cells must be positive and no larger than the post-QC cohort")
    strata = composite_strata(adata.obs, stratify_fields)
    selected_indices, allocations = deterministic_stratified_subsample(
        strata, requested_cells, sample_seed
    )
    adata = _subset_rows(adata, selected_indices)

    # These covariates describe the complete raw selected matrix, before any
    # optional gene filtering or transformation.
    selected_qc = compute_count_qc(adata.X, _gene_names(adata))
    _attach_qc(adata, selected_qc)
    adata.obs["sampling_stratum"] = composite_strata(adata.obs, stratify_fields)

    genes_before_filter = int(adata.n_vars)
    if min_cells_per_gene is not None:
        if min_cells_per_gene < 1:
            raise ValueError("min_cells_per_gene must be positive")
        keep_genes = _detection_counts(adata.X) >= min_cells_per_gene
        if not np.any(keep_genes):
            raise ValueError("local gene QC removed every feature")
        adata = adata[:, keep_genes].copy()

    sampling = _sampling_record(
        candidate_rows=int(post_cell_qc_indices.size),
        selected_indices=selected_indices,
        fields=stratify_fields,
        allocations=allocations,
        seed=sample_seed,
    )
    sampling["selected_cell_ids_sha256"] = _sha256_strings(adata.obs_names.astype(str))
    source = {
        "mode": "local_h5ad",
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "counts_layer": counts_layer or "X",
        "original_shape": [original_cells, original_genes],
        "post_cell_qc_cells": int(post_cell_qc_indices.size),
        "selected_shape": [int(adata.n_obs), int(adata.n_vars)],
        "genes_before_detection_filter": genes_before_filter,
        "genes_after_detection_filter": int(adata.n_vars),
        "raw_count_validation": count_validation,
    }
    qc_record = {
        "computed_from": "raw selected counts before gene filtering or transformations",
        "cell_filters": {
            "enabled": min_genes is not None or max_pct_mt is not None,
            "min_genes": min_genes,
            "max_pct_mt": max_pct_mt,
        },
        "gene_filter": {
            "enabled": min_cells_per_gene is not None,
            "min_cells_per_gene": min_cells_per_gene,
            "genes_before": genes_before_filter,
            "genes_after": int(adata.n_vars),
        },
        "summary": _qc_summary(selected_qc),
    }
    return LoadedCells(adata=adata, source=source, sampling=sampling, qc=qc_record)


def cache_remote_file(
    url: str,
    cache_dir: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Atomically cache a URL and return response/file provenance.

    Existing cache entries are always re-hashed.  A URL digest prefixes the
    remote basename, preventing two different URLs with the same filename from
    silently sharing a cache entry.
    """

    import urllib.parse
    import urllib.request

    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("matrix URL must use HTTP or HTTPS")
    basename = Path(urllib.parse.unquote(parsed.path)).name
    if not basename:
        raise ValueError("matrix URL has no filename")
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_sha256 = hashlib.sha256(url.encode("utf-8")).hexdigest()
    target = cache_dir / f"{url_sha256[:16]}-{basename}"
    if target.is_file():
        actual_sha256 = _sha256_file(target)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"cached file {target} does not match --expected-sha256")
        return target, {
            "url": url,
            "url_sha256": url_sha256,
            "cache_hit": True,
            "resolved_path": str(target),
            "sha256": actual_sha256,
            "bytes": int(target.stat().st_size),
            "expected_sha256": expected_sha256,
        }

    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    response_record: dict[str, Any] = {}
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "dimension-reduction-benchmarks/0.1"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as handle,
        ):
            response_record = {
                "resolved_url": response.geturl(),
                "status": getattr(response, "status", None),
                "content_length": response.headers.get("Content-Length"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
            while block := response.read(8 * 1024 * 1024):
                handle.write(block)
                digest.update(block)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size == 0:
            raise RuntimeError("matrix download produced an empty file")
        content_length = response_record.get("content_length")
        if content_length is not None and temporary.stat().st_size != int(content_length):
            raise RuntimeError(
                "matrix download byte count does not match the response Content-Length"
            )
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError("downloaded file does not match --expected-sha256")
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return target, {
        "url": url,
        "url_sha256": url_sha256,
        "cache_hit": False,
        "resolved_path": str(target),
        "sha256": actual_sha256,
        "bytes": int(target.stat().st_size),
        "expected_sha256": expected_sha256,
        "response": response_record,
    }


def load_tenx_brain_h5(
    path: Path,
    *,
    download_provenance: dict[str, Any] | None = None,
    max_cells: int | None = None,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    stratify_fields: Sequence[str] = (),
    min_genes: int | None = None,
    max_pct_mt: float | None = None,
    min_cells_per_gene: int | None = 3,
    scanpy_module: Any | None = None,
) -> LoadedCells:
    """Read the official filtered 10x brain matrix without assuming its shape."""

    scanpy = scanpy_module or _require_module("scanpy", "read a 10x HDF5 matrix")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = scanpy.read_10x_h5(str(path), gex_only=True)
    _validate_unique_observation_ids(adata)
    original_feature_names = np.asarray(adata.var_names.astype(str), dtype=str)
    if hasattr(adata, "var_names_make_unique"):
        adata.var_names_make_unique()
    if "feature_id" not in adata.var.columns:
        if "gene_ids" in adata.var.columns:
            adata.var["feature_id"] = np.asarray(adata.var["gene_ids"].astype(str), dtype=str)
        else:
            adata.var["feature_id"] = np.asarray(adata.var_names.astype(str), dtype=str)
    if "feature_name" not in adata.var.columns:
        adata.var["feature_name"] = original_feature_names

    # The public H5 does not expose per-cell mouse/library assignments.  These
    # constants make that limitation explicit and give HVG selection one global
    # donor-by-assay batch instead of inventing biological batches.
    adata.obs["dataset_id"] = TENX_BRAIN_DATASET_ID
    adata.obs["donor_id"] = "pooled-two-mice-unresolved"
    adata.obs["assay"] = TENX_BRAIN_ASSAY
    count_validation = validate_raw_counts(adata.X)
    original_cells, original_genes = map(int, adata.shape)
    qc_all = compute_count_qc(adata.X, _gene_names(adata))
    keep_cells = np.ones(original_cells, dtype=bool)
    if min_genes is not None:
        if min_genes < 0:
            raise ValueError("min_genes must be nonnegative")
        keep_cells &= qc_all["n_genes_by_counts"] >= min_genes
    if max_pct_mt is not None:
        if not 0 <= max_pct_mt <= 100:
            raise ValueError("max_pct_mt must lie between zero and 100")
        keep_cells &= qc_all["pct_counts_mt"] <= max_pct_mt
    post_cell_qc_indices = np.flatnonzero(keep_cells)
    if not post_cell_qc_indices.size:
        raise ValueError("10x cell QC removed every observation")
    adata = _subset_rows(adata, post_cell_qc_indices)

    requested_cells = int(adata.n_obs) if max_cells is None else int(max_cells)
    if requested_cells < 1 or requested_cells > adata.n_obs:
        raise ValueError("max_cells must be positive and no larger than the post-QC cohort")
    strata = composite_strata(adata.obs, stratify_fields)
    selected_indices, allocations = deterministic_stratified_subsample(
        strata, requested_cells, sample_seed
    )
    adata = _subset_rows(adata, selected_indices)
    selected_qc = compute_count_qc(adata.X, _gene_names(adata))
    _attach_qc(adata, selected_qc)
    adata.obs["sampling_stratum"] = composite_strata(adata.obs, stratify_fields)

    genes_before_filter = int(adata.n_vars)
    if min_cells_per_gene is not None:
        if min_cells_per_gene < 1:
            raise ValueError("min_cells_per_gene must be positive")
        keep_genes = _detection_counts(adata.X) >= min_cells_per_gene
        if not np.any(keep_genes):
            raise ValueError("10x gene QC removed every feature")
        adata = adata[:, keep_genes].copy()
    sampling = _sampling_record(
        candidate_rows=int(post_cell_qc_indices.size),
        selected_indices=selected_indices,
        fields=stratify_fields,
        allocations=allocations,
        seed=sample_seed,
    )
    sampling["selected_cell_ids_sha256"] = _sha256_strings(adata.obs_names.astype(str))
    path_sha256 = (
        str(download_provenance["sha256"])
        if download_provenance is not None
        and Path(download_provenance.get("resolved_path", "")).resolve() == path
        and download_provenance.get("sha256")
        else _sha256_file(path)
    )
    source = {
        "mode": "tenx_h5",
        "dataset_id": TENX_BRAIN_DATASET_ID,
        "path": str(path),
        "sha256": path_sha256,
        "bytes": int(path.stat().st_size),
        "download": download_provenance,
        "reader": "scanpy.read_10x_h5(gex_only=True)",
        "original_shape": [original_cells, original_genes],
        "post_cell_qc_cells": int(post_cell_qc_indices.size),
        "selected_shape": [int(adata.n_obs), int(adata.n_vars)],
        "genes_before_detection_filter": genes_before_filter,
        "genes_after_detection_filter": int(adata.n_vars),
        "observation_constants": {
            "dataset_id": TENX_BRAIN_DATASET_ID,
            "donor_id": "pooled-two-mice-unresolved",
            "assay": TENX_BRAIN_ASSAY,
        },
        "metadata_limitations": [
            "per-cell mouse assignment is unavailable",
            "per-cell library assignment is unavailable",
            "10x chemistry generation is not established by the source page",
        ],
        "raw_count_validation": count_validation,
    }
    qc_record = {
        "computed_from": "raw selected counts before gene filtering or transformations",
        "cell_filters": {
            "enabled": min_genes is not None or max_pct_mt is not None,
            "min_genes": min_genes,
            "max_pct_mt": max_pct_mt,
        },
        "gene_filter": {
            "enabled": min_cells_per_gene is not None,
            "min_cells_per_gene": min_cells_per_gene,
            "genes_before": genes_before_filter,
            "genes_after": int(adata.n_vars),
        },
        "summary": _qc_summary(selected_qc),
    }
    return LoadedCells(adata=adata, source=source, sampling=sampling, qc=qc_record)


def build_census_value_filter(dataset_ids: Sequence[str]) -> str:
    """Build a safe TileDB-SOMA filter which always enforces primary data."""

    normalized = sorted({str(value) for value in dataset_ids})
    if not normalized:
        raise ValueError("at least one resolved Census dataset ID is required")
    unsafe = [value for value in normalized if not SAFE_DATASET_ID_PATTERN.fullmatch(value)]
    if unsafe:
        raise ValueError(f"unsafe Census dataset IDs cannot enter a value filter: {unsafe}")
    quoted = ", ".join(repr(value) for value in normalized)
    return f"is_primary_data == True and dataset_id in [{quoted}]"


def _resolved_census_datasets(
    census: Any,
    *,
    collection_id: str,
    dataset_title: str | None,
) -> tuple[list[str], list[dict[str, Any]], int]:
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    if "collection_id" not in datasets.columns or "dataset_id" not in datasets.columns:
        raise RuntimeError("Census dataset table lacks collection_id or dataset_id")
    collection_rows = datasets.loc[datasets["collection_id"].astype(str) == collection_id].copy()
    if collection_rows.empty:
        raise ValueError(f"collection {collection_id!r} is absent from this Census release")
    if dataset_title is not None:
        if "dataset_title" not in collection_rows.columns:
            raise RuntimeError("Census dataset table lacks dataset_title")
        resolved = collection_rows.loc[
            collection_rows["dataset_title"].astype(str) == dataset_title
        ].copy()
        if resolved.empty:
            available = sorted(collection_rows["dataset_title"].astype(str).unique().tolist())
            raise ValueError(
                f"dataset title {dataset_title!r} is absent from collection; available={available}"
            )
    else:
        resolved = collection_rows
    ids = sorted(resolved["dataset_id"].astype(str).unique().tolist())
    provenance_columns = [
        column
        for column in (
            "collection_id",
            "collection_name",
            "dataset_id",
            "dataset_title",
            "dataset_version_id",
            "dataset_total_cell_count",
            "dataset_h5ad_path",
        )
        if column in resolved.columns
    ]
    rows = resolved[provenance_columns].sort_values("dataset_id").to_dict(orient="records")
    return ids, _jsonable(rows), int(collection_rows.shape[0])


def _primary_mask(values: Any) -> np.ndarray:
    result = []
    for value in values:
        if value is True or isinstance(value, np.bool_) and bool(value):
            result.append(True)
        elif isinstance(value, (int, np.integer)) and int(value) == 1:
            result.append(True)
        else:
            result.append(str(value).lower() == "true")
    return np.asarray(result, dtype=bool)


def _align_census_rows(adata: Any, selected_ids: np.ndarray) -> Any:
    if "soma_joinid" not in adata.obs.columns:
        raise RuntimeError("Census AnnData response omitted soma_joinid")
    returned = np.asarray(adata.obs["soma_joinid"], dtype=np.int64)
    if returned.size != selected_ids.size or np.unique(returned).size != returned.size:
        raise RuntimeError("Census AnnData response has a wrong or duplicate row count")
    positions = {int(value): index for index, value in enumerate(returned)}
    try:
        order = np.asarray([positions[int(value)] for value in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("Census AnnData response omitted a selected soma_joinid") from error
    return _subset_rows(adata, order)


def _census_obs_columns(
    metadata_fields: Sequence[str],
    stratify_fields: Sequence[str],
    hvg_batch_fields: Sequence[str] = DEFAULT_HVG_BATCH_FIELDS,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *metadata_fields,
                *stratify_fields,
                *hvg_batch_fields,
                "soma_joinid",
                "is_primary_data",
            ]
        )
    )


def census_raw_request_identity(
    *,
    census_version: str,
    collection_id: str,
    dataset_title: str | None,
    max_cells: int | None,
    sample_seed: int,
    stratify_fields: Sequence[str],
    metadata_fields: Sequence[str],
    hvg_batch_fields: Sequence[str] = DEFAULT_HVG_BATCH_FIELDS,
    organism: str = "Homo sapiens",
    min_cells_per_gene: int | None = 3,
) -> dict[str, Any]:
    """Return the exact acquisition/selection identity for a raw checkpoint."""

    return {
        "checkpoint_schema_version": RAW_CHECKPOINT_SCHEMA_VERSION,
        "source_mode": "cellxgene_census",
        "census_version": validate_pinned_census_version(census_version),
        "organism": str(organism),
        "collection_id": str(collection_id),
        "dataset_title": dataset_title,
        "X_name": "raw",
        "is_primary_data": True,
        "max_cells": max_cells,
        "sample_seed": int(sample_seed),
        "stratify_fields": list(stratify_fields),
        "metadata_fields": list(metadata_fields),
        "hvg_batch_fields": list(hvg_batch_fields),
        "observation_columns": list(
            _census_obs_columns(metadata_fields, stratify_fields, hvg_batch_fields)
        ),
        "min_cells_per_gene": min_cells_per_gene,
    }


def _raw_checkpoint_paths(
    cache_dir: Path, request_identity: Mapping[str, Any]
) -> tuple[str, Path, Path]:
    identity_sha256 = _sha256_json(request_identity)
    checkpoint_dir = Path(cache_dir).expanduser().resolve() / (
        f"tabula-sapiens-{identity_sha256[:16]}"
    )
    return identity_sha256, checkpoint_dir, checkpoint_dir / "checkpoint.json"


@contextlib.contextmanager
def _checkpoint_lock(checkpoint_dir: Path):
    fcntl = _require_module("fcntl", "lock a selected-raw checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_dir / ".checkpoint.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_public_record(checkpoint: Mapping[str, Any], *, cache_hit: bool) -> dict[str, Any]:
    selected = checkpoint["selected_raw_h5ad"]
    semantic = checkpoint["semantic"]
    return {
        "schema_version": int(checkpoint["schema_version"]),
        "cache_key": str(checkpoint["identity_sha256"]),
        "cache_hit": bool(cache_hit),
        "file_sha256": str(selected["sha256"]),
        "semantic_sha256": str(semantic["semantic_sha256"]),
        "shape": list(semantic["shape"]),
    }


def _cleanup_checkpoint_files(checkpoint_dir: Path, *, keep_generation: Path) -> None:
    """Remove files not referenced by the committed manifest while holding its lock."""

    for candidate in checkpoint_dir.glob("selected_raw.*.h5ad"):
        if candidate != keep_generation:
            with contextlib.suppress(OSError):
                candidate.unlink()
    for candidate in checkpoint_dir.glob(".selected_raw.*.tmp.h5ad"):
        with contextlib.suppress(OSError):
            candidate.unlink()


def write_selected_raw_checkpoint(
    cache_dir: Path,
    *,
    loaded: LoadedCells,
    request_identity: Mapping[str, Any],
    required_obs_columns: Sequence[str],
) -> dict[str, Any]:
    """Atomically commit the selected raw sparse object and its provenance."""

    anndata = _require_module("anndata", "write a selected-raw checkpoint")
    identity = _jsonable(dict(request_identity))
    identity_sha256, checkpoint_dir, manifest_path = _raw_checkpoint_paths(cache_dir, identity)
    with _checkpoint_lock(checkpoint_dir):
        semantic = _selected_raw_semantic_record(
            loaded.adata, required_obs_columns=required_obs_columns
        )
        validate_raw_counts(loaded.adata.X)
        temporary = checkpoint_dir / f".selected_raw.{uuid.uuid4().hex}.tmp.h5ad"
        generation_path: Path | None = None
        generation_created = False
        try:
            loaded.adata.write_h5ad(temporary, compression="lzf")
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            roundtrip = anndata.read_h5ad(temporary, backed="r")
            try:
                roundtrip_semantic = _selected_raw_semantic_record(
                    roundtrip, required_obs_columns=required_obs_columns
                )
                validate_raw_counts(roundtrip.X)
            finally:
                roundtrip_file = getattr(roundtrip, "file", None)
                if roundtrip_file is not None:
                    roundtrip_file.close()
            if roundtrip_semantic != semantic:
                raise RuntimeError("selected-raw H5AD round trip changed semantic content")
            file_sha256 = _sha256_file(temporary)
            generation_path = checkpoint_dir / f"selected_raw.{file_sha256}.h5ad"
            if generation_path.exists():
                if _sha256_file(generation_path) == file_sha256:
                    temporary.unlink()
                else:
                    os.replace(temporary, generation_path)
            else:
                os.replace(temporary, generation_path)
                generation_created = True
            _fsync_directory(checkpoint_dir)
            loaded_payload = {
                "source": _jsonable(loaded.source),
                "sampling": _jsonable(loaded.sampling),
                "qc": _jsonable(loaded.qc),
            }
            checkpoint = {
                "schema_version": RAW_CHECKPOINT_SCHEMA_VERSION,
                "request_identity": identity,
                "identity_sha256": identity_sha256,
                "selected_raw_h5ad": {
                    "path": generation_path.name,
                    "sha256": file_sha256,
                    "bytes": int(generation_path.stat().st_size),
                },
                "semantic": semantic,
                "loaded": {
                    **loaded_payload,
                    "source_sha256": _sha256_json(loaded_payload["source"]),
                    "sampling_sha256": _sha256_json(loaded_payload["sampling"]),
                    "qc_sha256": _sha256_json(loaded_payload["qc"]),
                },
            }
            _atomic_json(manifest_path, checkpoint)
        except Exception:
            if generation_created and generation_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    generation_path.unlink()
            raise
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        _fsync_directory(checkpoint_dir)
        _cleanup_checkpoint_files(checkpoint_dir, keep_generation=generation_path)
        _fsync_directory(checkpoint_dir)
    return _checkpoint_public_record(checkpoint, cache_hit=False)


def _invalid_checkpoint(message: str) -> ValueError:
    return ValueError(
        f"invalid selected-raw checkpoint: {message}; pass --refresh-raw-cache "
        "to replace it explicitly"
    )


def load_selected_raw_checkpoint(
    cache_dir: Path,
    *,
    request_identity: Mapping[str, Any],
    required_obs_columns: Sequence[str],
) -> tuple[LoadedCells, dict[str, Any]] | None:
    """Load and fully validate a committed selected-raw checkpoint."""

    anndata = _require_module("anndata", "read a selected-raw checkpoint")
    identity = _jsonable(dict(request_identity))
    identity_sha256, checkpoint_dir, manifest_path = _raw_checkpoint_paths(cache_dir, identity)
    with _checkpoint_lock(checkpoint_dir):
        if not manifest_path.exists():
            return None
        try:
            checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise _invalid_checkpoint(f"cannot read {manifest_path.name}: {error}") from error
        if not isinstance(checkpoint, dict):
            raise _invalid_checkpoint("checkpoint.json must contain a JSON object")
        if checkpoint.get("schema_version") != RAW_CHECKPOINT_SCHEMA_VERSION:
            raise _invalid_checkpoint("schema version mismatch")
        if checkpoint.get("request_identity") != identity:
            raise _invalid_checkpoint("request identity mismatch")
        if checkpoint.get("identity_sha256") != identity_sha256:
            raise _invalid_checkpoint("request identity hash mismatch")
        selected = checkpoint.get("selected_raw_h5ad", {})
        selected_name = selected.get("path") if isinstance(selected, dict) else None
        if not isinstance(selected_name, str) or not RAW_H5AD_GENERATION_PATTERN.fullmatch(
            selected_name
        ):
            raise _invalid_checkpoint("selected raw H5AD has an unsafe generation path")
        selected_sha256 = selected.get("sha256")
        if selected_name != f"selected_raw.{selected_sha256}.h5ad":
            raise _invalid_checkpoint("H5AD generation path does not match its SHA-256")
        h5ad_path = checkpoint_dir / selected_name
        if not h5ad_path.is_file():
            raise _invalid_checkpoint("selected raw H5AD generation is absent")
        if int(selected.get("bytes", -1)) != h5ad_path.stat().st_size:
            raise _invalid_checkpoint("H5AD byte count mismatch")
        if selected_sha256 != _sha256_file(h5ad_path):
            raise _invalid_checkpoint("H5AD SHA-256 mismatch")

        try:
            adata = anndata.read_h5ad(h5ad_path)
            validate_raw_counts(adata.X)
            semantic = _selected_raw_semantic_record(
                adata, required_obs_columns=required_obs_columns
            )
        except MemoryError:
            raise
        except Exception as error:
            raise _invalid_checkpoint(f"cannot validate selected_raw.h5ad: {error}") from error
        if checkpoint.get("semantic") != semantic:
            raise _invalid_checkpoint("semantic content mismatch")
        loaded_record = checkpoint.get("loaded", {})
        for field in ("source", "sampling", "qc"):
            value = loaded_record.get(field)
            if not isinstance(value, dict):
                raise _invalid_checkpoint(f"loaded.{field} provenance is absent")
            if loaded_record.get(f"{field}_sha256") != _sha256_json(value):
                raise _invalid_checkpoint(f"loaded.{field} provenance hash mismatch")
        _cleanup_checkpoint_files(checkpoint_dir, keep_generation=h5ad_path)

    public = _checkpoint_public_record(checkpoint, cache_hit=True)
    source = dict(loaded_record["source"])
    source["selected_raw_checkpoint"] = public
    loaded = LoadedCells(
        adata=adata,
        source=source,
        sampling=dict(loaded_record["sampling"]),
        qc=dict(loaded_record["qc"]),
    )
    return loaded, public


def load_tabula_sapiens_census(
    *,
    census_version: str = DEFAULT_CENSUS_VERSION,
    collection_id: str = TABULA_SAPIENS_COLLECTION_ID,
    dataset_title: str | None = TABULA_SAPIENS_DATASET_TITLE,
    max_cells: int | None = CENSUS_PILOT_CELLS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    stratify_fields: Sequence[str] = DEFAULT_CENSUS_STRATIFY_FIELDS,
    metadata_fields: Sequence[str] = DEFAULT_METADATA_FIELDS,
    hvg_batch_fields: Sequence[str] = DEFAULT_HVG_BATCH_FIELDS,
    organism: str = "Homo sapiens",
    min_cells_per_gene: int | None = 3,
    stage_timings: dict[str, float | None] | None = None,
    stage_status: dict[str, str] | None = None,
) -> LoadedCells:
    """Load Tabula Sapiens raw counts from a pinned Census query.

    This function first obtains observation metadata, samples its soma_joinids,
    and only then asks Census for the selected raw matrix.  It never downloads a
    full expression matrix merely to discard rows locally.
    """

    version = validate_pinned_census_version(census_version)
    census_api = _require_module("cellxgene_census", "query CELLxGENE Census")
    obs_columns = list(_census_obs_columns(metadata_fields, stratify_fields, hvg_batch_fields))
    calls: list[str] = []
    started = time.perf_counter()
    with census_api.open_soma(census_version=version) as census:
        dataset_ids, dataset_rows, collection_dataset_count = _resolved_census_datasets(
            census, collection_id=collection_id, dataset_title=dataset_title
        )
        _finish_stage(stage_timings, stage_status, "census_dataset_resolution", started)
        value_filter = build_census_value_filter(dataset_ids)
        calls.append("get_obs")
        started = time.perf_counter()
        candidates = census_api.get_obs(
            census,
            organism,
            value_filter=value_filter,
            column_names=obs_columns,
        )
        _finish_stage(stage_timings, stage_status, "census_candidate_metadata", started)
        missing = [column for column in obs_columns if column not in candidates.columns]
        if missing:
            raise RuntimeError(f"Census observation response omitted columns: {missing}")
        if candidates.empty:
            raise ValueError("pinned Census query returned no primary observations")
        if not np.all(_primary_mask(candidates["is_primary_data"])):
            raise RuntimeError("Census metadata response contains non-primary observations")
        candidates = candidates.sort_values("soma_joinid", kind="mergesort").reset_index(drop=True)
        soma_ids = np.asarray(candidates["soma_joinid"], dtype=np.int64)
        if np.unique(soma_ids).size != soma_ids.size:
            raise RuntimeError("Census metadata response contains duplicate soma_joinids")

        started = time.perf_counter()
        requested_cells = int(candidates.shape[0]) if max_cells is None else int(max_cells)
        if requested_cells < 1 or requested_cells > candidates.shape[0]:
            raise ValueError(
                "requested Census sample must be positive and no larger than the eligible cohort"
            )
        strata = composite_strata(candidates, stratify_fields)
        selected_indices, allocations = deterministic_stratified_subsample(
            strata, requested_cells, sample_seed
        )
        selected_soma_ids = soma_ids[selected_indices]
        _finish_stage(stage_timings, stage_status, "census_sampling", started)

        calls.append("get_anndata")
        started = time.perf_counter()
        adata = census_api.get_anndata(
            census,
            organism,
            X_name="raw",
            obs_value_filter=value_filter,
            obs_coords=np.asarray(selected_soma_ids, dtype=np.int64),
            obs_column_names=obs_columns,
            var_column_names=["feature_id", "feature_name"],
        )
        _finish_stage(stage_timings, stage_status, "census_raw_matrix_fetch", started)
    if calls != ["get_obs", "get_anndata"]:
        raise RuntimeError("Census query order violated metadata-first loading")
    started = time.perf_counter()
    adata = _align_census_rows(adata, selected_soma_ids)
    if int(adata.n_obs) != requested_cells:
        raise RuntimeError("Census raw response did not contain the exact selected cell count")
    if not np.all(_primary_mask(adata.obs["is_primary_data"])):
        raise RuntimeError("Census raw response contains non-primary observations")

    count_validation = validate_raw_counts(adata.X)
    selected_qc = compute_count_qc(adata.X, _gene_names(adata))
    _attach_qc(adata, selected_qc)
    adata.obs["sampling_stratum"] = composite_strata(adata.obs, stratify_fields)
    genes_before_filter = int(adata.n_vars)
    if min_cells_per_gene is not None:
        if min_cells_per_gene < 1:
            raise ValueError("min_cells_per_gene must be positive")
        keep_genes = _detection_counts(adata.X) >= min_cells_per_gene
        if not np.any(keep_genes):
            raise ValueError("Census gene QC removed every feature")
        adata = adata[:, keep_genes].copy()
    sampling = _sampling_record(
        candidate_rows=int(candidates.shape[0]),
        selected_indices=selected_indices,
        fields=stratify_fields,
        allocations=allocations,
        seed=sample_seed,
        selected_ids=selected_soma_ids,
    )
    sampling["selected_cell_ids_sha256"] = _sha256_strings(adata.obs_names.astype(str))
    query_record = {
        "census_version": version,
        "organism": organism,
        "collection_id": collection_id,
        "dataset_title": dataset_title,
        "resolved_dataset_ids": dataset_ids,
        "value_filter": value_filter,
        "X_name": "raw",
        "is_primary_data": True,
    }
    source = {
        "mode": "cellxgene_census",
        **query_record,
        "query_sha256": _sha256_json(query_record),
        "collection_dataset_rows": collection_dataset_count,
        "resolved_datasets": dataset_rows,
        "resolved_datasets_sha256": _sha256_json(dataset_rows),
        "candidate_cells": int(candidates.shape[0]),
        "selected_shape": [int(adata.n_obs), int(adata.n_vars)],
        "genes_before_detection_filter": genes_before_filter,
        "genes_after_detection_filter": int(adata.n_vars),
        "raw_count_validation": count_validation,
    }
    qc_record = {
        "computed_from": "raw selected counts before transformations",
        "cell_filters": {"enabled": False, "reason": "Census source is already curated"},
        "gene_filter": {
            "enabled": min_cells_per_gene is not None,
            "min_cells_per_gene": min_cells_per_gene,
            "genes_before": genes_before_filter,
            "genes_after": int(adata.n_vars),
        },
        "summary": _qc_summary(selected_qc),
    }
    _finish_stage(stage_timings, stage_status, "census_alignment_qc_gene_filter", started)
    return LoadedCells(adata=adata, source=source, sampling=sampling, qc=qc_record)


def _dense_float32(matrix: Any, *, max_bytes: int, description: str) -> np.ndarray:
    rows, columns = map(int, matrix.shape)
    predicted = rows * columns * np.dtype(np.float32).itemsize
    if predicted > max_bytes:
        raise MemoryError(
            f"{description} would require at least {predicted:,} bytes, exceeding "
            f"--max-dense-bytes={max_bytes:,}"
        )
    sparse = _require_module("scipy.sparse", "materialize a sparse prepared matrix")
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{description} contains non-finite values")
    return result


def _metadata_table(adata: Any, metadata_fields: Sequence[str]) -> Any:
    pd = _require_module("pandas", "write prepared metadata")
    requested = list(
        dict.fromkeys([*metadata_fields, "sampling_stratum", "hvg_batch", *QC_COLUMNS])
    )
    available = [column for column in requested if column in adata.obs.columns]
    frame = adata.obs[available].copy()
    if "soma_joinid" not in frame.columns:
        frame.insert(0, "cell_id", np.asarray(adata.obs_names.astype(str), dtype=str))
    elif "cell_id" not in frame.columns:
        frame.insert(0, "cell_id", np.asarray(adata.obs_names.astype(str), dtype=str))
    return pd.DataFrame(frame).reset_index(drop=True)


def _gene_table(work: Any) -> Any:
    pd = _require_module("pandas", "write prepared gene metadata")
    frame = work.var.copy()
    stable_ids = (
        np.asarray(frame["feature_id"].astype(str), dtype=str)
        if "feature_id" in frame.columns
        else np.asarray(work.var_names.astype(str), dtype=str)
    )
    if np.unique(stable_ids).size != stable_ids.size:
        raise ValueError("selected stable feature IDs are not unique")
    frame["gene_id"] = stable_ids
    if "feature_id" not in frame.columns:
        frame["feature_id"] = np.asarray(work.var_names.astype(str), dtype=str)
    if "feature_name" not in frame.columns:
        frame["feature_name"] = np.asarray(work.var_names.astype(str), dtype=str)
    leading = ["gene_id", "feature_id", "feature_name"]
    frame = frame[leading + [column for column in frame.columns if column not in leading]]
    loadings = work.varm.get("PCs") if hasattr(work, "varm") else None
    if loadings is not None:
        loadings = np.asarray(loadings, dtype=np.float32)
        for index in range(loadings.shape[1]):
            frame[f"pca_loading_{index + 1}"] = loadings[:, index]
    return pd.DataFrame(frame).reset_index(drop=True)


def diagnose_seurat_v3_batches(
    adata: Any,
    *,
    batch_key: str,
    span: float,
    loess_factory: Any | None = None,
    aggregate_func: Any | None = None,
) -> list[dict[str, Any]]:
    """Preflight the exact fixed-span LOESS geometry for every HVG batch."""

    if not 0 < span <= 1:
        raise ValueError("HVG LOESS span must lie in (0, 1]")
    if batch_key not in adata.obs.columns:
        raise ValueError(f"HVG batch key {batch_key!r} is absent from observation metadata")
    if loess_factory is None:
        loess_factory = _require_module("skmisc.loess", "diagnose Seurat-v3 HVG batches").loess
    if aggregate_func is None:
        aggregate_func = _require_module(
            "scanpy", "match Seurat-v3 batch mean/variance aggregation"
        ).get.aggregate

    aggregated = aggregate_func(adata, by=batch_key, func=["mean", "var"])
    if batch_key not in aggregated.obs.columns:
        raise RuntimeError("Scanpy aggregate omitted the requested HVG batch key")
    if "mean" not in aggregated.layers or "var" not in aggregated.layers:
        raise RuntimeError("Scanpy aggregate omitted mean or variance layers")
    sparse = _require_module("scipy.sparse", "read Seurat-v3 batch diagnostics")
    means = aggregated.layers["mean"]
    variances = aggregated.layers["var"]
    means = means.toarray() if sparse.issparse(means) else np.asarray(means)
    variances = variances.toarray() if sparse.issparse(variances) else np.asarray(variances)
    labels = np.asarray(aggregated.obs[batch_key], dtype=str)
    if "n_obs_aggregated" in aggregated.obs.columns:
        cell_counts = np.asarray(aggregated.obs["n_obs_aggregated"], dtype=np.int64)
    else:
        source_labels = np.asarray(adata.obs[batch_key], dtype=str)
        cell_counts = np.asarray(
            [np.count_nonzero(source_labels == label) for label in labels], dtype=np.int64
        )
    if means.shape != variances.shape or means.shape[0] != labels.size:
        raise RuntimeError("Scanpy aggregate returned inconsistent HVG diagnostic shapes")

    diagnostics: list[dict[str, Any]] = []
    failed: list[str] = []
    for index in np.argsort(labels, kind="stable"):
        batch_label = str(labels[index])
        cell_count = int(cell_counts[index])
        error: Exception | None = None
        mean = np.asarray(means[index], dtype=np.float64).reshape(-1)
        variance = np.asarray(variances[index], dtype=np.float64).reshape(-1)
        nonconstant_mask = variance > 0
        predictors = np.log10(mean[nonconstant_mask])
        response = np.log10(variance[nonconstant_mask])
        if not np.all(np.isfinite(predictors)) or not np.all(np.isfinite(response)):
            error = ValueError("non-finite log10 mean or variance entered the LOESS fit")
        elif predictors.size:
            try:
                model = loess_factory(predictors, response, span=float(span), degree=2)
                model.fit()
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as caught:
                error = caught

        nonconstant = int(predictors.size)
        distinct = int(np.unique(predictors).size)
        diagnostic = {
            "batch_label": str(batch_label),
            "cell_count": cell_count,
            "nonconstant_genes": nonconstant,
            "distinct_log10_means": distinct,
            "duplicate_log10_mean_fraction": (
                float(1.0 - distinct / nonconstant) if nonconstant else None
            ),
            "log10_mean_min": float(np.min(predictors)) if nonconstant else None,
            "log10_mean_max": float(np.max(predictors)) if nonconstant else None,
            "span": float(span),
            "degree": 2,
            "loess_neighborhood_points": int(math.ceil(span * nonconstant)),
            "loess_applied": bool(nonconstant),
            "status": "failed" if error is not None else "ok",
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
        }
        diagnostics.append(diagnostic)
        if error is not None:
            failed.append(batch_label)

    if failed:
        joined = ", ".join(repr(label) for label in failed)
        raise HVGLoessDiagnosticsError(
            f"Seurat-v3 LOESS preflight failed for hvg_batch values: {joined}; "
            f"the globally fixed span was {span:g} and no fallback was attempted",
            diagnostics,
        )
    return diagnostics


def validate_hvg_batch_fields(fields: Sequence[str]) -> None:
    """Keep biological labels out of the technical-batch HVG correction."""

    forbidden = sorted(set(map(str, fields)) & FORBIDDEN_HVG_BATCH_FIELDS)
    if forbidden:
        joined = ", ".join(forbidden)
        raise ValueError(
            "HVG batch fields may describe only technical batches; biological "
            f"cell-type/tissue fields are forbidden: {joined}"
        )


def prepare_representations(
    loaded: LoadedCells,
    *,
    n_hvgs: int = DEFAULT_N_HVGS,
    n_pcs: int = DEFAULT_PCA_COMPONENTS,
    target_sum: float = DEFAULT_TARGET_SUM,
    scale: bool = True,
    scale_clip: float = DEFAULT_SCALE_CLIP,
    write_log_hvg: bool = False,
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    hvg_batch_fields: Sequence[str] = DEFAULT_HVG_BATCH_FIELDS,
    hvg_span: float = DEFAULT_HVG_SPAN,
    metadata_fields: Sequence[str] = DEFAULT_METADATA_FIELDS,
    random_seed: int = DEFAULT_SAMPLE_SEED,
    scanpy_module: Any | None = None,
    loess_factory: Any | None = None,
    aggregate_func: Any | None = None,
    stage_timings: dict[str, float | None] | None = None,
    stage_status: dict[str, str] | None = None,
) -> PreparedRepresentations:
    """Apply the frozen raw-count preprocessing contract to selected cells."""

    if n_hvgs < 1 or n_hvgs > loaded.adata.n_vars:
        raise ValueError("n_hvgs must be positive and no larger than the feature count")
    if n_pcs < 1 or n_pcs > min(loaded.adata.n_obs, n_hvgs):
        raise ValueError("n_pcs must be positive and no larger than cells or HVGs")
    if target_sum <= 0 or scale_clip <= 0 or max_dense_bytes < 1:
        raise ValueError("target_sum, scale_clip, and max_dense_bytes must be positive")
    if not 0 < hvg_span <= 1:
        raise ValueError("HVG LOESS span must lie in (0, 1]")
    validate_hvg_batch_fields(hvg_batch_fields)
    scanpy = scanpy_module or _require_module("scanpy", "run single-cell preprocessing")
    adata = loaded.adata
    started = time.perf_counter()
    _canonicalize_raw_matrix(adata)
    validation = validate_raw_counts(adata.X)
    raw_matrix_sha256 = _matrix_semantic_sha256(adata.X)
    _finish_stage(stage_timings, stage_status, "raw_count_validation", started)

    # donor|assay controls donor/library effects without treating tissue-joined
    # dataset IDs or biological cell-type labels as HVG batches.
    pd = _require_module("pandas", "record ordered HVG batch assignments")
    hvg_batches = np.asarray(composite_strata(adata.obs, hvg_batch_fields), dtype=str)
    batch_categories = sorted(np.unique(hvg_batches).tolist())
    adata.obs["hvg_batch"] = pd.Categorical(hvg_batches, categories=batch_categories, ordered=True)
    hvg_batch_key = "hvg_batch"
    started = time.perf_counter()
    try:
        diagnostics = diagnose_seurat_v3_batches(
            adata,
            batch_key=hvg_batch_key,
            span=float(hvg_span),
            loess_factory=loess_factory,
            aggregate_func=aggregate_func,
        )
    except Exception:
        _finish_stage(
            stage_timings,
            stage_status,
            "hvg_batch_diagnostics",
            started,
            status="failed",
        )
        raise
    else:
        _finish_stage(stage_timings, stage_status, "hvg_batch_diagnostics", started)

    started = time.perf_counter()
    try:
        scanpy.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=int(n_hvgs),
            batch_key=hvg_batch_key,
            span=float(hvg_span),
            subset=False,
            inplace=True,
        )
        if _matrix_semantic_sha256(adata.X) != raw_matrix_sha256:
            raise RuntimeError("Scanpy HVG selection modified the selected raw count matrix")
        if "highly_variable" not in adata.var.columns:
            raise HVGLoessDiagnosticsError(
                "Scanpy did not annotate highly_variable genes after the fixed-span "
                "Seurat-v3 fit; no fallback was attempted",
                diagnostics,
            )
        hvg_mask = np.asarray(adata.var["highly_variable"], dtype=bool)
        actual_hvgs = int(hvg_mask.sum())
        if actual_hvgs != n_hvgs:
            raise HVGLoessDiagnosticsError(
                f"Scanpy selected {actual_hvgs} HVGs; expected exactly {n_hvgs}; "
                "no fallback was attempted",
                diagnostics,
            )
    except HVGLoessDiagnosticsError:
        _finish_stage(stage_timings, stage_status, "hvg_selection", started, status="failed")
        raise
    except MemoryError:
        _finish_stage(stage_timings, stage_status, "hvg_selection", started, status="failed")
        raise
    except Exception as error:
        _finish_stage(stage_timings, stage_status, "hvg_selection", started, status="failed")
        raise HVGLoessDiagnosticsError(
            "Scanpy Seurat-v3 HVG selection failed after every batch passed the "
            f"fixed-span {hvg_span:g} LOESS preflight; no fallback was attempted: {error}",
            diagnostics,
        ) from error
    else:
        _finish_stage(stage_timings, stage_status, "hvg_selection", started)
    started = time.perf_counter()
    work = adata[:, hvg_mask].copy()
    _finish_stage(stage_timings, stage_status, "hvg_slice", started)

    started = time.perf_counter()
    scanpy.pp.normalize_total(work, target_sum=float(target_sum), inplace=True)
    _finish_stage(stage_timings, stage_status, "normalize_total", started)
    started = time.perf_counter()
    scanpy.pp.log1p(work)
    _finish_stage(stage_timings, stage_status, "log1p", started)
    predicted_scale_bytes: int | None = None
    if scale:
        # Scaling with zero centering is dense.  Check the known lower bound
        # before asking pinned Scanpy to materialize its float64 sparse-input result.
        predicted_scale_bytes = int(work.n_obs) * int(work.n_vars) * np.dtype(np.float64).itemsize
        if predicted_scale_bytes > max_dense_bytes:
            raise MemoryError(
                "scaled HVG matrix would require at least "
                f"{predicted_scale_bytes:,} bytes, exceeding "
                f"--max-dense-bytes={max_dense_bytes:,}"
            )
        started = time.perf_counter()
        scanpy.pp.scale(work, zero_center=True, max_value=float(scale_clip))
        _finish_stage(stage_timings, stage_status, "scale", started)
    else:
        _set_skipped_stage(stage_timings, stage_status, "scale", "disabled")

    log_hvg = None
    if write_log_hvg:
        log_hvg = _dense_float32(
            work.X, max_bytes=max_dense_bytes, description="log-HVG sensitivity matrix"
        )

    started = time.perf_counter()
    scanpy.pp.pca(
        work,
        n_comps=int(n_pcs),
        zero_center=True,
        svd_solver="randomized",
        random_state=int(random_seed),
        dtype="float32",
    )
    _finish_stage(stage_timings, stage_status, "pca", started)
    if "X_pca" not in work.obsm:
        raise RuntimeError("Scanpy PCA did not produce obsm['X_pca']")
    pca = np.ascontiguousarray(work.obsm["X_pca"], dtype=np.float32)
    if pca.shape != (work.n_obs, n_pcs) or not np.all(np.isfinite(pca)):
        raise RuntimeError("PCA result has a wrong shape or non-finite values")

    started = time.perf_counter()
    metadata = _metadata_table(loaded.adata, metadata_fields)
    genes = _gene_table(work)
    _finish_stage(stage_timings, stage_status, "output_tables", started)
    pipeline = {
        "operation_order": [
            "raw_counts",
            "seurat_v3_hvg",
            "normalize_total",
            "log1p",
            *(["scale"] if scale else []),
            "non_whitened_pca",
        ],
        "raw_count_validation": {
            **validation,
            "semantic_sha256": raw_matrix_sha256,
            "unchanged_after_hvg_selection": True,
        },
        "hvg": {
            "flavor": "seurat_v3",
            "span": float(hvg_span),
            "degree": 2,
            "span_policy": "fixed_global",
            "fallback": "none",
            "requested_genes": int(n_hvgs),
            "selected_genes": actual_hvgs,
            "batch_key": hvg_batch_key,
            "batch_fields": list(hvg_batch_fields),
            "batch_definition": "|".join(hvg_batch_fields) if hvg_batch_fields else "__all__",
            "batch_count": int(np.unique(hvg_batches).size),
            "batch_assignments_sha256": _sha256_strings(hvg_batches),
            "batch_diagnostics": diagnostics,
            "cell_type_or_tissue_labels_used": False,
        },
        "normalize_total": {"target_sum": float(target_sum)},
        "log1p": True,
        "scaling": {
            "enabled": bool(scale),
            "zero_center": bool(scale),
            "unit_variance": bool(scale),
            "max_value": float(scale_clip) if scale else None,
            "dense_guard_dtype": "float64" if scale else None,
            "predicted_dense_bytes": predicted_scale_bytes,
        },
        "pca": {
            "components": int(n_pcs),
            "whiten": False,
            "zero_center": True,
            "svd_solver": "randomized",
            "random_seed": int(random_seed),
            "row_l2_normalized": False,
        },
        "dense_sensitivity_written": bool(write_log_hvg),
        "max_dense_bytes": int(max_dense_bytes),
    }
    pipeline["configuration_sha256"] = _sha256_json(pipeline)
    return PreparedRepresentations(
        pca=pca, log_hvg=log_hvg, metadata=metadata, genes=genes, pipeline=pipeline
    )


def _package_versions() -> dict[str, str | None]:
    distributions = (
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "anndata",
        "scanpy",
        "scikit-misc",
        "cellxgene-census",
        "tiledbsoma",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _npy_artifact(path: Path, values: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
    }


def _parquet_artifact(path: Path, frame: Any) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "dtype": "parquet",
        "columns": [str(column) for column in frame.columns],
    }


def write_prepared_outputs(
    output_dir: Path,
    *,
    loaded: LoadedCells,
    prepared: PreparedRepresentations,
    stage_timings: dict[str, float | None] | None = None,
    stage_status: dict[str, str] | None = None,
    run_started_at: float | None = None,
) -> dict[str, Any]:
    """Atomically write standard artifacts and a hash-bearing manifest."""

    if prepared.pca.ndim != 2 or prepared.pca.dtype != np.float32:
        raise ValueError("prepared PCA must be a two-dimensional float32 array")
    rows = int(prepared.pca.shape[0])
    if int(prepared.metadata.shape[0]) != rows:
        raise ValueError("metadata row count does not match PCA row count")
    if prepared.log_hvg is not None:
        if prepared.log_hvg.ndim != 2 or prepared.log_hvg.dtype != np.float32:
            raise ValueError("prepared log-HVG matrix must be a two-dimensional float32 array")
        if int(prepared.log_hvg.shape[0]) != rows:
            raise ValueError("log-HVG row count does not match PCA row count")
        if int(prepared.genes.shape[0]) != int(prepared.log_hvg.shape[1]):
            raise ValueError("gene table row count does not match log-HVG columns")
    selected_hvgs = prepared.pipeline.get("hvg", {}).get("selected_genes")
    if selected_hvgs is not None and int(prepared.genes.shape[0]) != int(selected_hvgs):
        raise ValueError("gene table row count does not match selected HVGs")

    timings = {} if stage_timings is None else stage_timings
    statuses = {} if stage_status is None else stage_status
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_name = f"pca{prepared.pca.shape[1]}"
    pca_path = output_dir / f"{pca_name}.npy"
    metadata_path = output_dir / "metadata.parquet"
    genes_path = output_dir / "genes.parquet"
    log_name = f"log_hvg{prepared.log_hvg.shape[1]}" if prepared.log_hvg is not None else None
    log_path = output_dir / f"{log_name}.npy" if log_name is not None else None
    output_started = time.perf_counter()
    _atomic_npy(pca_path, prepared.pca)
    _atomic_parquet(metadata_path, prepared.metadata)
    _atomic_parquet(genes_path, prepared.genes)
    artifacts = {
        pca_name: _npy_artifact(pca_path, prepared.pca),
        "metadata": _parquet_artifact(metadata_path, prepared.metadata),
        "genes": _parquet_artifact(genes_path, prepared.genes),
    }
    if prepared.log_hvg is not None and log_path is not None and log_name is not None:
        _atomic_npy(log_path, prepared.log_hvg)
        artifacts[log_name] = _npy_artifact(log_path, prepared.log_hvg)
    _finish_stage(timings, statuses, "output_write", output_started)
    if run_started_at is not None:
        _finish_stage(timings, statuses, "total_before_manifest", run_started_at)
    if set(timings) != set(statuses):
        raise RuntimeError("stage timing and status keys are inconsistent")

    selection = dict(loaded.sampling)
    selection["selected_cells"] = rows
    selection["qc"] = loaded.qc
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": loaded.source,
        "selection": selection,
        "preprocessing": prepared.pipeline,
        "artifacts": artifacts,
        "software": {
            "created_at_utc": _utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "timing_clock": "time.perf_counter",
            "stage_timings_seconds": dict(sorted(timings.items())),
            "stage_status": dict(sorted(statuses.items())),
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return _jsonable(manifest)


def _csv_fields(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--n-hvg", type=int, default=DEFAULT_N_HVGS)
    parser.add_argument("--n-pcs", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--target-sum", type=float, default=DEFAULT_TARGET_SUM)
    parser.add_argument(
        "--scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="center, scale, and clip log-HVGs before PCA (default: enabled)",
    )
    parser.add_argument("--scale-clip", type=float, default=DEFAULT_SCALE_CLIP)
    parser.add_argument("--write-log-hvg", action="store_true")
    parser.add_argument("--max-dense-bytes", type=int, default=DEFAULT_MAX_DENSE_BYTES)
    parser.add_argument(
        "--hvg-batch-fields",
        default=",".join(DEFAULT_HVG_BATCH_FIELDS),
        help="comma-separated technical fields; default is donor_id,assay",
    )
    parser.add_argument(
        "--hvg-span",
        type=float,
        default=DEFAULT_HVG_SPAN,
        help="globally fixed Seurat-v3 LOESS span (default: 0.5; no fallback)",
    )
    parser.add_argument(
        "--metadata-fields",
        default=",".join(DEFAULT_METADATA_FIELDS),
        help="comma-separated observation columns to export when available",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    census = subparsers.add_parser(
        "tabula-sapiens", aliases=["census"], help="query a pinned Census release"
    )
    census.set_defaults(source_mode="census")
    _add_common_arguments(census)
    census.add_argument("--census-version", default=DEFAULT_CENSUS_VERSION)
    census.add_argument("--collection-id", default=TABULA_SAPIENS_COLLECTION_ID)
    census.add_argument("--dataset-title", default=TABULA_SAPIENS_DATASET_TITLE)
    census.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="exactly 100000 for the pilot; omit to prepare the complete eligible cohort",
    )
    census.add_argument("--stratify-by", default=",".join(DEFAULT_CENSUS_STRATIFY_FIELDS))
    census.add_argument("--min-cells-per-gene", type=int, default=3)
    census.add_argument(
        "--raw-cache-dir",
        type=Path,
        help="identity-keyed selected-raw H5AD checkpoint root",
    )
    census.add_argument(
        "--refresh-raw-cache",
        action="store_true",
        help="explicitly replace the matching selected-raw checkpoint",
    )

    local = subparsers.add_parser(
        "local-h5ad", aliases=["h5ad", "local"], help="prepare an existing local H5AD"
    )
    local.set_defaults(source_mode="local")
    _add_common_arguments(local)
    local.add_argument("--input-h5ad", type=Path, required=True)
    local.add_argument("--counts-layer")
    local.add_argument("--max-cells", type=int)
    local.add_argument("--stratify-by", default="")
    local.add_argument("--min-genes", type=int)
    local.add_argument("--max-pct-mt", type=float)
    local.add_argument("--min-cells-per-gene", type=int, default=3)

    tenx = subparsers.add_parser(
        "tenx-brain-1m3", aliases=["tenx"], help="prepare the filtered 10x 1.3M brain matrix"
    )
    tenx.set_defaults(source_mode="tenx")
    _add_common_arguments(tenx)
    source = tenx.add_mutually_exclusive_group()
    source.add_argument("--matrix-h5", type=Path)
    source.add_argument("--matrix-url")
    tenx.add_argument(
        "--cache-dir",
        type=Path,
        help="download cache (default: a downloads directory beside --output)",
    )
    tenx.add_argument("--expected-sha256")
    tenx.add_argument("--max-cells", type=int)
    tenx.add_argument("--stratify-by", default="")
    tenx.add_argument("--min-genes", type=int)
    tenx.add_argument("--max-pct-mt", type=float)
    tenx.add_argument("--min-cells-per-gene", type=int, default=3)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_started_at = time.perf_counter()
    stage_timings: dict[str, float | None] = {}
    stage_status: dict[str, str] = {}
    metadata_fields = _csv_fields(args.metadata_fields)
    hvg_batch_fields = _csv_fields(args.hvg_batch_fields)
    stratify_fields = _csv_fields(args.stratify_by)
    if not 0 < args.hvg_span <= 1:
        raise ValueError("HVG LOESS span must lie in (0, 1]")
    validate_hvg_batch_fields(hvg_batch_fields)
    if args.source_mode == "census":
        if args.max_cells is not None and args.max_cells != CENSUS_PILOT_CELLS:
            raise ValueError(
                f"the Census pilot is fixed at exactly {CENSUS_PILOT_CELLS} cells; "
                "omit --max-cells for the complete cohort"
            )
        if args.refresh_raw_cache and args.raw_cache_dir is None:
            raise ValueError("--refresh-raw-cache requires --raw-cache-dir")
        request_identity = census_raw_request_identity(
            census_version=args.census_version,
            collection_id=args.collection_id,
            dataset_title=args.dataset_title,
            max_cells=args.max_cells,
            sample_seed=args.sample_seed,
            stratify_fields=stratify_fields,
            metadata_fields=metadata_fields,
            hvg_batch_fields=hvg_batch_fields,
            min_cells_per_gene=args.min_cells_per_gene,
        )
        required_obs_columns = tuple(
            dict.fromkeys(
                [
                    *_census_obs_columns(metadata_fields, stratify_fields, hvg_batch_fields),
                    *QC_COLUMNS,
                    "sampling_stratum",
                ]
            )
        )
        cached: tuple[LoadedCells, dict[str, Any]] | None = None
        if args.raw_cache_dir is not None and not args.refresh_raw_cache:
            cache_started = time.perf_counter()
            cached = load_selected_raw_checkpoint(
                args.raw_cache_dir,
                request_identity=request_identity,
                required_obs_columns=required_obs_columns,
            )
            if cached is not None:
                loaded, _ = cached
                _finish_stage(
                    stage_timings,
                    stage_status,
                    "raw_cache",
                    cache_started,
                    status="cache_hit",
                )
                stage_timings["source_load"] = stage_timings["raw_cache"]
                stage_status["source_load"] = "cache_hit"
        if cached is None:
            source_started = time.perf_counter()
            loaded = load_tabula_sapiens_census(
                census_version=args.census_version,
                collection_id=args.collection_id,
                dataset_title=args.dataset_title,
                max_cells=args.max_cells,
                sample_seed=args.sample_seed,
                stratify_fields=stratify_fields,
                metadata_fields=metadata_fields,
                hvg_batch_fields=hvg_batch_fields,
                min_cells_per_gene=args.min_cells_per_gene,
                stage_timings=stage_timings,
                stage_status=stage_status,
            )
            _finish_stage(stage_timings, stage_status, "source_load", source_started)
            if args.max_cells == CENSUS_PILOT_CELLS and loaded.adata.n_obs != CENSUS_PILOT_CELLS:
                raise RuntimeError(
                    f"Census pilot did not return exactly {CENSUS_PILOT_CELLS} cells"
                )
            if args.raw_cache_dir is not None:
                cache_started = time.perf_counter()
                checkpoint = write_selected_raw_checkpoint(
                    args.raw_cache_dir,
                    loaded=loaded,
                    request_identity=request_identity,
                    required_obs_columns=required_obs_columns,
                )
                checkpoint["cache_hit"] = False
                loaded.source["selected_raw_checkpoint"] = checkpoint
                _finish_stage(
                    stage_timings,
                    stage_status,
                    "raw_cache",
                    cache_started,
                    status="refreshed" if args.refresh_raw_cache else "written",
                )
            else:
                _set_skipped_stage(stage_timings, stage_status, "raw_cache", "disabled")
        if args.max_cells == CENSUS_PILOT_CELLS and loaded.adata.n_obs != CENSUS_PILOT_CELLS:
            raise RuntimeError(f"Census pilot did not return exactly {CENSUS_PILOT_CELLS} cells")
    elif args.source_mode == "local":
        source_started = time.perf_counter()
        loaded = load_local_h5ad(
            args.input_h5ad,
            counts_layer=args.counts_layer,
            max_cells=args.max_cells,
            sample_seed=args.sample_seed,
            stratify_fields=stratify_fields,
            min_genes=args.min_genes,
            max_pct_mt=args.max_pct_mt,
            min_cells_per_gene=args.min_cells_per_gene,
        )
        _finish_stage(stage_timings, stage_status, "source_load", source_started)
        _set_skipped_stage(stage_timings, stage_status, "raw_cache", "not_applicable")
    else:
        source_started = time.perf_counter()
        if args.matrix_h5 is not None:
            matrix_path = args.matrix_h5
            download = None
        else:
            matrix_url = args.matrix_url or TENX_BRAIN_MATRIX_URL
            cache_dir = args.cache_dir or (
                Path(args.output).expanduser().resolve().parent / "downloads"
            )
            matrix_path, download = cache_remote_file(
                matrix_url,
                cache_dir,
                expected_sha256=args.expected_sha256,
            )
        loaded = load_tenx_brain_h5(
            matrix_path,
            download_provenance=download,
            max_cells=args.max_cells,
            sample_seed=args.sample_seed,
            stratify_fields=stratify_fields,
            min_genes=args.min_genes,
            max_pct_mt=args.max_pct_mt,
            min_cells_per_gene=args.min_cells_per_gene,
        )
        _finish_stage(stage_timings, stage_status, "source_load", source_started)
        _set_skipped_stage(stage_timings, stage_status, "raw_cache", "not_applicable")

    preprocessing_started = time.perf_counter()
    try:
        prepared = prepare_representations(
            loaded,
            n_hvgs=args.n_hvg,
            n_pcs=args.n_pcs,
            target_sum=args.target_sum,
            scale=args.scale,
            scale_clip=args.scale_clip,
            write_log_hvg=args.write_log_hvg,
            max_dense_bytes=args.max_dense_bytes,
            hvg_batch_fields=hvg_batch_fields,
            hvg_span=args.hvg_span,
            metadata_fields=metadata_fields,
            random_seed=args.sample_seed,
            stage_timings=stage_timings,
            stage_status=stage_status,
        )
    except HVGLoessDiagnosticsError as error:
        _finish_stage(
            stage_timings,
            stage_status,
            "preprocessing",
            preprocessing_started,
            status="failed",
        )
        failure_path = Path(args.output).expanduser().resolve().parent / (
            f"{Path(args.output).name}.hvg-loess-failure.json"
        )
        _atomic_json(
            failure_path,
            {
                "created_at_utc": _utc_now(),
                "error": str(error),
                "hvg_span": float(args.hvg_span),
                "hvg_batch_fields": list(hvg_batch_fields),
                "fallback": "none",
                "diagnostics": list(error.diagnostics),
                "selected_raw_checkpoint": loaded.source.get("selected_raw_checkpoint"),
                "stage_timings_seconds": stage_timings,
                "stage_status": stage_status,
            },
        )
        raise
    _finish_stage(stage_timings, stage_status, "preprocessing", preprocessing_started)
    failure_path = Path(args.output).expanduser().resolve().parent / (
        f"{Path(args.output).name}.hvg-loess-failure.json"
    )
    with contextlib.suppress(FileNotFoundError):
        failure_path.unlink()
    return write_prepared_outputs(
        args.output,
        loaded=loaded,
        prepared=prepared,
        stage_timings=stage_timings,
        stage_status=stage_status,
        run_started_at=run_started_at,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = execute(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "rows": manifest["selection"]["selected_cells"],
                "manifest": str((Path(args.output) / "manifest.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
