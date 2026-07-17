#!/usr/bin/env python3
"""Benchmark PCA, autoencoder, DiRe, and RAPIDS UMAP on prepared scRNA-seq data.

This harness deliberately consumes *prepared* dense float32 arrays.  It never
normalizes, whitens, centers, scales, or otherwise rewrites them: ``pca50.npy`` and
``log_hvg2000.npy`` therefore define two distinct, auditable estimands.  Reducers run
in fresh subprocesses, while quality metrics are evaluated on deterministic samples.

The primary Tabula Sapiens pilot command is::

    conda run -n rapids-26.06 python experiments/compare_single_cell_reducers.py \
      --data artifacts/single-cell/tabula-sapiens-pilot/pca50.npy \
      --metadata artifacts/single-cell/tabula-sapiens-pilot/metadata.parquet \
      --manifest artifacts/single-cell/tabula-sapiens-pilot/manifest.json \
      --dataset-name tabula-sapiens-pilot-100k --input-representation pca50 \
      --methods pca autoencoder dire umap --target-dimensions 2 8 \
      --seeds 0 1 2 --neighbors 30 --ae-amp \
      --dire-repo ../dire-rapids \
      --output-dir artifacts/single-cell/tabula-sapiens-pilot/benchmark-pca50

Append ``--validate-input`` to validate only, or ``--dry-run`` to print every
isolated worker command without launching a reducer.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

try:  # Package import in tests.
    from .compare_large_embeddings import (
        AE_EVALUATION_PRECISIONS,
        METHODS,
        PeakMonitor,
        _ae_precision_configuration,
        _atomic_json,
        _auto_hidden_dimensions,
        _run_subprocess,
        _save_layout_atomic,
        _sha256_bytes,
        _sha256_file,
        _worker_hardware,
        _write_csv,
        deterministic_sample,
        distance_fidelity,
        fit_autoencoder_large,
        fit_dire_large,
        fit_pca_large,
        fit_umap_large,
        index_sha256,
        sample_conditional_metrics,
        sample_pairs,
        sampled_pair_distances,
    )
except ImportError:  # Direct ``python experiments/...py`` execution.
    from compare_large_embeddings import (  # type: ignore[no-redef]
        AE_EVALUATION_PRECISIONS,
        METHODS,
        PeakMonitor,
        _ae_precision_configuration,
        _atomic_json,
        _auto_hidden_dimensions,
        _run_subprocess,
        _save_layout_atomic,
        _sha256_bytes,
        _sha256_file,
        _worker_hardware,
        _write_csv,
        deterministic_sample,
        distance_fidelity,
        fit_autoencoder_large,
        fit_dire_large,
        fit_pca_large,
        fit_umap_large,
        index_sha256,
        sample_conditional_metrics,
        sample_pairs,
        sampled_pair_distances,
    )


DEFAULT_TECHNICAL_COLUMNS = ("total_counts", "n_genes_by_counts", "pct_counts_mt")
INPUT_PROTOCOL = "prepared values used verbatim; no row normalization, scaling, or whitening"


def _empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    text = str(value).strip()
    return text in {"", "<NA>", "nan", "None"}


def _strings(values) -> np.ndarray:
    return np.asarray(["" if _empty(value) else str(value) for value in values], dtype=str)


def load_metadata(path: str | Path) -> dict[str, np.ndarray]:
    """Load aligned metadata, importing pandas only for tabular file formats."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    if path.suffix in {".parquet", ".pq", ".csv"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required to read Parquet or CSV metadata") from exc
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        return {str(name): frame[name].to_numpy() for name in frame.columns}
    raise ValueError("metadata must be Parquet, CSV, or NPZ")


def _validate_metadata(metadata: dict[str, np.ndarray], rows: int) -> None:
    if not metadata:
        raise ValueError("metadata contains no columns")
    for name, values in metadata.items():
        if np.asarray(values).ndim != 1 or len(values) != rows:
            raise ValueError(
                f"metadata column {name!r} must be one-dimensional and aligned with {rows} rows"
            )


def _check_manifest_artifact(
    manifest: dict,
    manifest_path: Path,
    *,
    artifact_name: str,
    actual_path: Path,
    actual_sha256: str,
    actual_rows: int,
    actual_shape: tuple[int, ...] | None = None,
    actual_dtype: str | None = None,
    require_shape: bool = False,
    require_dtype: bool = False,
    require_row_dimension: bool = False,
) -> dict:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(artifact_name), dict):
        raise ValueError(f"manifest has no artifacts.{artifact_name} record")
    record = artifacts[artifact_name]
    required = ("sha256",)
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"manifest artifacts.{artifact_name} is missing keys: {missing}")
    path_field = "path" if "path" in record else "filename" if "filename" in record else None
    if path_field is None:
        raise ValueError(f"manifest artifacts.{artifact_name} requires path (or filename)")
    declared_path = Path(record[path_field])
    if not declared_path.is_absolute():
        declared_path = manifest_path.parent / declared_path
    if declared_path.resolve() != actual_path.resolve():
        raise ValueError(
            f"manifest artifacts.{artifact_name}.{path_field} does not identify {actual_path}"
        )
    if record["sha256"] != actual_sha256:
        raise ValueError(f"manifest artifacts.{artifact_name}.sha256 does not match the file")
    if "rows" in record:
        if int(record["rows"]) != actual_rows:
            raise ValueError(f"manifest artifacts.{artifact_name}.rows does not match the file")
    if "shape" in record:
        declared_shape = tuple(map(int, record["shape"]))
        if not declared_shape or declared_shape[0] != actual_rows:
            raise ValueError(f"manifest artifacts.{artifact_name}.shape has the wrong row count")
        if actual_shape is not None and declared_shape != actual_shape:
            raise ValueError(f"manifest artifacts.{artifact_name}.shape does not match the array")
    elif require_shape:
        raise ValueError(f"manifest artifacts.{artifact_name}.shape is required")
    if require_row_dimension and "rows" not in record and "shape" not in record:
        raise ValueError(f"manifest artifacts.{artifact_name} requires shape (or rows)")
    if "dtype" in record:
        if actual_dtype is not None and str(record["dtype"]) != actual_dtype:
            raise ValueError(f"manifest artifacts.{artifact_name}.dtype does not match the array")
    elif require_dtype:
        raise ValueError(f"manifest artifacts.{artifact_name}.dtype is required")
    if "bytes" in record and int(record["bytes"]) != actual_path.stat().st_size:
        raise ValueError(f"manifest artifacts.{artifact_name}.bytes does not match the file")
    verified = [path_field, "sha256"]
    verified.extend(key for key in ("rows", "bytes") if key in record)
    if "shape" in record:
        verified.append("shape")
    if "dtype" in record:
        verified.append("dtype")
    return {"artifact": artifact_name, "verified_fields": verified}


def validate_prepared_input(
    data: str | Path,
    *,
    metadata_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    input_representation: str | None = None,
    required_columns=(),
    technical_columns=DEFAULT_TECHNICAL_COLUMNS,
    block_rows: int = 16_384,
) -> tuple[dict, dict[str, np.ndarray] | None, dict | None]:
    """Validate a prepared matrix without changing it and return auditable provenance."""
    data = Path(data).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"prepared array does not exist: {data}")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    X = np.load(data, mmap_mode="r", allow_pickle=False)
    if X.ndim != 2:
        raise ValueError("prepared data must be a two-dimensional NPY array")
    if X.dtype != np.float32:
        raise ValueError("prepared data must have dtype float32")
    if min(X.shape) < 2:
        raise ValueError("prepared data must contain at least two rows and two features")

    minimum, maximum = float("inf"), float("-inf")
    zero_rows = 0
    for start in range(0, X.shape[0], block_rows):
        block = np.asarray(X[start : start + block_rows])
        if not np.all(np.isfinite(block)):
            raise ValueError("prepared data contains non-finite values")
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
        zero_rows += int(np.count_nonzero(np.all(block == 0, axis=1)))

    data_fingerprint = _sha256_file(data)
    metadata = load_metadata(metadata_path) if metadata_path is not None else None
    missing_required = []
    available_technical = []
    missing_technical = list(technical_columns)
    metadata_fingerprint = None
    if metadata is not None:
        _validate_metadata(metadata, X.shape[0])
        missing_required = [name for name in required_columns if name not in metadata]
        if missing_required:
            raise ValueError(f"metadata is missing required columns: {missing_required}")
        available_technical = [name for name in technical_columns if name in metadata]
        missing_technical = [name for name in technical_columns if name not in metadata]
        metadata_fingerprint = _sha256_file(metadata_path)

    manifest = None
    manifest_fingerprint = None
    manifest_checks = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest must contain a JSON object")
        manifest_fingerprint = _sha256_file(manifest_path)
        selection = manifest.get("selection")
        if isinstance(selection, dict) and "selected_cells" in selection:
            if int(selection["selected_cells"]) != X.shape[0]:
                raise ValueError("manifest selection.selected_cells does not match the array")
            row_count_field = "selection.selected_cells"
        elif "row_count" in manifest:  # Compatibility with the provisional schema.
            if int(manifest["row_count"]) != X.shape[0]:
                raise ValueError("manifest row_count does not match the prepared array")
            row_count_field = "row_count"
        else:
            raise ValueError("manifest requires selection.selected_cells (or row_count)")
        manifest_checks = {"authoritative_row_count": row_count_field}
        if input_representation is not None:
            manifest_checks["representation"] = _check_manifest_artifact(
                manifest,
                manifest_path,
                artifact_name=input_representation,
                actual_path=data,
                actual_sha256=data_fingerprint,
                actual_rows=int(X.shape[0]),
                actual_shape=tuple(map(int, X.shape)),
                actual_dtype=str(X.dtype),
                require_shape=True,
                require_dtype=True,
            )
        if metadata_path is not None:
            metadata_suffix = Path(metadata_path).suffix.lower()
            metadata_dtype = {
                ".parquet": "parquet",
                ".pq": "parquet",
                ".csv": "csv",
                ".npz": "npz",
            }.get(metadata_suffix)
            manifest_checks["metadata"] = _check_manifest_artifact(
                manifest,
                manifest_path,
                artifact_name="metadata",
                actual_path=Path(metadata_path).expanduser().resolve(),
                actual_sha256=metadata_fingerprint,
                actual_rows=int(X.shape[0]),
                actual_shape=(int(X.shape[0]), len(metadata)),
                actual_dtype=metadata_dtype,
                require_row_dimension=True,
            )

    report = {
        "data": str(data),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "dtype": str(X.dtype),
        "file_bytes": int(data.stat().st_size),
        "data_sha256": data_fingerprint,
        "finite": True,
        "minimum": minimum,
        "maximum": maximum,
        "all_zero_rows": zero_rows,
        "input_transform": "none",
        "input_protocol": INPUT_PROTOCOL,
        "metadata": (
            str(Path(metadata_path).expanduser().resolve()) if metadata_path is not None else None
        ),
        "metadata_sha256": metadata_fingerprint,
        "metadata_columns": sorted(metadata) if metadata is not None else [],
        "available_technical_columns": available_technical,
        "missing_technical_columns": missing_technical,
        "manifest": (
            str(Path(manifest_path).expanduser().resolve()) if manifest_path is not None else None
        ),
        "manifest_sha256": manifest_fingerprint,
        "manifest_semantic_checks": manifest_checks,
    }
    return report, metadata, manifest


def _combined_strata(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.asarray([f"{label}\x1f{group}" for label, group in zip(labels, groups, strict=True)])


def deterministic_label_sample(labels, groups, count: int, seed: int) -> np.ndarray:
    """Sample cell-type/donor strata deterministically, excluding missing annotations."""
    labels, groups = _strings(labels), _strings(groups)
    eligible = np.flatnonzero((labels != "") & (groups != ""))
    if eligible.size < 2:
        return eligible
    local = deterministic_sample(
        eligible.size,
        min(int(count), eligible.size),
        seed,
        labels=_combined_strata(labels[eligible], groups[eligible]),
    )
    return eligible[local]


def donor_heldout_cell_type_metrics(
    layout,
    cell_types,
    donors,
    *,
    sample_size: int,
    seed: int,
    n_neighbors: int,
    min_count: int,
    min_donors: int,
    n_jobs: int = 1,
) -> dict:
    """Run a transductive cell-type probe with strictly donor-disjoint folds."""
    cell_types, donors = _strings(cell_types), _strings(donors)
    valid = (cell_types != "") & (donors != "")
    eligible_labels = []
    for label in np.unique(cell_types[valid]):
        label_rows = valid & (cell_types == label)
        if (
            np.count_nonzero(label_rows) >= min_count
            and np.unique(donors[label_rows]).size >= min_donors
        ):
            eligible_labels.append(label)
    eligible_rows = np.flatnonzero(valid & np.isin(cell_types, eligible_labels))
    if eligible_rows.size >= 2:
        local = deterministic_label_sample(
            cell_types[eligible_rows], donors[eligible_rows], sample_size, seed
        )
        indices = eligible_rows[local]
    else:
        indices = eligible_rows
    if indices.size < 4:
        return {
            "donor_heldout_macro_f1": None,
            "donor_heldout_balanced_accuracy": None,
            "donor_heldout_accuracy": None,
            "donor_heldout_per_cell_type_f1": None,
            "donor_heldout_evaluated_rows": 0,
            "donor_heldout_folds": 0,
            "label_probe_status": (
                "insufficient cell types meeting global count and donor coverage thresholds"
            ),
            "cell_type_minimum_sample_count": int(min_count),
            "cell_type_minimum_donors": int(min_donors),
        }
    labels = cell_types[indices]
    groups = donors[indices]
    coordinates = np.asarray(layout[indices], dtype=np.float32)
    if np.unique(labels).size < 2 or np.unique(groups).size < 2:
        return {
            "donor_heldout_macro_f1": None,
            "donor_heldout_balanced_accuracy": None,
            "donor_heldout_accuracy": None,
            "donor_heldout_per_cell_type_f1": None,
            "donor_heldout_evaluated_rows": 0,
            "donor_heldout_folds": 0,
            "label_probe_status": "fewer than two eligible cell types or donors",
            "label_sample_rows": int(indices.size),
            "label_sample_index_sha256": index_sha256(indices),
            "cell_type_minimum_sample_count": int(min_count),
            "cell_type_minimum_donors": int(min_donors),
        }

    truths, predictions = [], []
    fold_rows = []
    excluded_unseen = 0
    for donor in np.unique(groups):
        test_mask = groups == donor
        train_mask = ~test_mask
        train_labels = labels[train_mask]
        visible = np.isin(labels[test_mask], np.unique(train_labels))
        excluded_unseen += int(np.count_nonzero(~visible))
        test_positions = np.flatnonzero(test_mask)[visible]
        train_positions = np.flatnonzero(train_mask)
        if test_positions.size == 0 or train_positions.size < 2:
            continue
        k = min(n_neighbors, train_positions.size)
        classifier = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=n_jobs)
        classifier.fit(coordinates[train_positions], labels[train_positions])
        prediction = classifier.predict(coordinates[test_positions])
        truths.extend(labels[test_positions].tolist())
        predictions.extend(prediction.tolist())
        fold_rows.append({"donor": str(donor), "test_rows": int(test_positions.size)})

    if not truths or np.unique(truths).size < 2:
        return {
            "donor_heldout_macro_f1": None,
            "donor_heldout_balanced_accuracy": None,
            "donor_heldout_accuracy": None,
            "donor_heldout_per_cell_type_f1": None,
            "donor_heldout_evaluated_rows": len(truths),
            "donor_heldout_folds": len(fold_rows),
            "donor_heldout_excluded_unseen_cell_type_rows": excluded_unseen,
            "label_probe_status": "fewer than two shared cell types across held-out folds",
            "label_sample_rows": int(indices.size),
            "label_sample_index_sha256": index_sha256(indices),
            "cell_type_minimum_sample_count": int(min_count),
            "cell_type_minimum_donors": int(min_donors),
        }
    truths, predictions = np.asarray(truths), np.asarray(predictions)
    evaluated_cell_types = np.unique(truths)
    per_cell_type_f1 = f1_score(
        truths,
        predictions,
        labels=evaluated_cell_types,
        average=None,
        zero_division=0,
    )
    return {
        "donor_heldout_macro_f1": float(f1_score(truths, predictions, average="macro")),
        "donor_heldout_balanced_accuracy": float(balanced_accuracy_score(truths, predictions)),
        "donor_heldout_accuracy": float(accuracy_score(truths, predictions)),
        "donor_heldout_per_cell_type_f1": {
            str(label): float(score)
            for label, score in zip(evaluated_cell_types, per_cell_type_f1, strict=True)
        },
        "donor_heldout_per_cell_type_support": {
            str(label): int(np.count_nonzero(truths == label)) for label in evaluated_cell_types
        },
        "donor_heldout_evaluated_rows": int(truths.size),
        "donor_heldout_folds": len(fold_rows),
        "donor_heldout_fold_rows": fold_rows,
        "donor_heldout_excluded_unseen_cell_type_rows": excluded_unseen,
        "label_sample_rows": int(indices.size),
        "label_sample_index_sha256": index_sha256(indices),
        "eligible_cell_types": int(np.unique(truths).size),
        "cell_type_minimum_sample_count": int(min_count),
        "cell_type_minimum_donors": int(min_donors),
        "label_probe_status": "ok",
        "label_probe_protocol": "leave-one-donor-out kNN on a transductively fitted layout",
    }


def sampled_cell_type_silhouette(
    layout,
    cell_types,
    donors,
    *,
    sample_size: int,
    seed: int,
    min_donors: int,
) -> dict:
    """Compute cell-type average silhouette width on a bounded stratified sample."""
    cell_types, donors = _strings(cell_types), _strings(donors)
    valid = (cell_types != "") & (donors != "")
    eligible_labels = []
    for label in np.unique(cell_types[valid]):
        rows = valid & (cell_types == label)
        if np.count_nonzero(rows) >= 2 and np.unique(donors[rows]).size >= min_donors:
            eligible_labels.append(label)
    eligible = np.flatnonzero(valid & np.isin(cell_types, eligible_labels))
    if eligible.size < 3 or len(eligible_labels) < 2:
        return {
            "sampled_cell_type_silhouette": None,
            "cell_type_asw_status": "fewer than two eligible cell types",
            "cell_type_asw_minimum_donors": int(min_donors),
        }
    local = deterministic_label_sample(
        cell_types[eligible], donors[eligible], min(sample_size, eligible.size), seed
    )
    indices = eligible[local]
    labels = cell_types[indices]
    if indices.size <= np.unique(labels).size:
        return {
            "sampled_cell_type_silhouette": None,
            "cell_type_asw_status": "too few sampled rows per cell type",
            "cell_type_asw_minimum_donors": int(min_donors),
        }
    value = silhouette_score(np.asarray(layout[indices], dtype=np.float32), labels)
    return {
        "sampled_cell_type_silhouette": float(value),
        "cell_type_average_silhouette_width": float(value),
        "cell_type_asw_status": "ok",
        "cell_type_asw_sample_rows": int(indices.size),
        "cell_type_asw_sample_index_sha256": index_sha256(indices),
        "cell_type_asw_classes": int(np.unique(labels).size),
        "cell_type_asw_minimum_donors": int(min_donors),
        "cell_type_asw_protocol": "Euclidean silhouette on a cell-type/donor-stratified sample",
    }


def technical_covariate_leakage(
    layout,
    cell_types,
    donors,
    covariates: dict[str, np.ndarray],
    *,
    sample_size: int,
    seed: int,
    n_neighbors: int,
    min_donors: int,
    n_jobs: int = 1,
) -> dict[str, dict]:
    """Measure cross-donor technical decodability conditional on cell type."""
    cell_types, donors = _strings(cell_types), _strings(donors)
    annotated = np.flatnonzero((cell_types != "") & (donors != ""))
    if annotated.size < 4:
        return {
            name: {
                "status": "insufficient cell-type/donor annotations",
                "r2": None,
                "nmse": None,
            }
            for name in covariates
        }
    local = deterministic_label_sample(
        cell_types[annotated],
        donors[annotated],
        min(int(sample_size), annotated.size),
        seed,
    )
    indices = annotated[local]
    coordinates = np.asarray(layout[indices], dtype=np.float32)
    sampled_cell_types = cell_types[indices]
    sampled_donors = donors[indices]
    output = {}
    for name, raw_values in covariates.items():
        try:
            values = np.asarray(raw_values, dtype=np.float64)[indices]
        except (TypeError, ValueError):
            output[name] = {"status": "not numeric", "r2": None, "nmse": None}
            continue
        finite = np.isfinite(values)
        truths, predictions, baselines, evaluated_types = [], [], [], []
        folds = 0
        eligible_rows = 0
        eligible_types = []
        for cell_type in np.unique(sampled_cell_types):
            type_rows = sampled_cell_types == cell_type
            type_donors = np.unique(sampled_donors[type_rows & finite])
            if type_donors.size < min_donors:
                continue
            eligible_types.append(str(cell_type))
            eligible_rows += int(np.count_nonzero(type_rows & finite))
            for donor in type_donors:
                test = type_rows & (sampled_donors == donor) & finite
                train = type_rows & (sampled_donors != donor) & finite
                if np.count_nonzero(test) == 0 or np.count_nonzero(train) < 2:
                    continue
                k = min(n_neighbors, int(np.count_nonzero(train)))
                regressor = KNeighborsRegressor(
                    n_neighbors=k,
                    weights="distance",
                    n_jobs=n_jobs,
                )
                regressor.fit(coordinates[train], values[train])
                prediction = regressor.predict(coordinates[test])
                truths.extend(values[test].tolist())
                predictions.extend(prediction.tolist())
                baselines.extend(np.repeat(np.mean(values[train]), np.count_nonzero(test)).tolist())
                evaluated_types.extend(np.repeat(cell_type, np.count_nonzero(test)).tolist())
                folds += 1
        if len(truths) < 3:
            output[name] = {
                "status": "insufficient within-cell-type cross-donor coverage",
                "r2": None,
                "nmse": None,
                "evaluated_rows": len(truths),
                "folds": folds,
                "eligible_rows": eligible_rows,
                "coverage_fraction": (len(truths) / eligible_rows if eligible_rows else 0.0),
                "eligible_cell_types": eligible_types,
            }
            continue
        truths = np.asarray(truths, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        baselines = np.asarray(baselines, dtype=np.float64)
        evaluated_types = np.asarray(evaluated_types)
        model_sse = float(np.sum((truths - predictions) ** 2))
        baseline_sse = float(np.sum((truths - baselines) ** 2))
        if baseline_sse <= np.finfo(np.float64).eps:
            output[name] = {
                "status": "insufficient within-cell-type numeric variation",
                "r2": None,
                "nmse": None,
                "evaluated_rows": int(truths.size),
                "folds": folds,
                "eligible_rows": eligible_rows,
                "coverage_fraction": truths.size / eligible_rows if eligible_rows else 0.0,
                "eligible_cell_types": eligible_types,
            }
            continue
        truth_residuals = np.empty_like(truths)
        prediction_residuals = np.empty_like(predictions)
        for cell_type in np.unique(evaluated_types):
            type_rows = evaluated_types == cell_type
            truth_residuals[type_rows] = truths[type_rows] - np.mean(truths[type_rows])
            prediction_residuals[type_rows] = predictions[type_rows] - np.mean(
                predictions[type_rows]
            )
        rho = float(spearmanr(truth_residuals, prediction_residuals).statistic)
        conditional_nmse = model_sse / baseline_sse
        conditional_r2 = 1.0 - conditional_nmse
        output[name] = {
            "status": "ok",
            "r2": conditional_r2,
            "nmse": conditional_nmse,
            "cell_type_conditioned_r2": conditional_r2,
            "cell_type_conditioned_nmse": conditional_nmse,
            "spearman": rho if np.isfinite(rho) else None,
            "cell_type_residual_spearman": rho if np.isfinite(rho) else None,
            "evaluated_rows": int(truths.size),
            "folds": folds,
            "eligible_rows": eligible_rows,
            "coverage_fraction": truths.size / eligible_rows if eligible_rows else 0.0,
            "eligible_cell_types": eligible_types,
            "evaluated_cell_types": int(np.unique(evaluated_types).size),
            "minimum_donors_per_cell_type": int(min_donors),
            "sample_rows": int(indices.size),
            "sample_index_sha256": index_sha256(indices),
            "protocol": (
                "within-cell-type leave-one-donor-out kNN regression; baseline is the "
                "same-cell-type training-donor mean; higher conditioned R2 means more leakage"
            ),
        }
    return output


def evaluate_single_cell_layout(X, layout, args, metadata=None) -> dict:
    """Evaluate geometry, donor-held-out cell labels, and technical leakage."""
    if layout.shape != (X.shape[0], args.current_target_dimension):
        raise ValueError("layout shape is not aligned with the prepared input")
    if not np.all(np.isfinite(layout)):
        raise ValueError("layout contains non-finite values")
    indices = deterministic_sample(X.shape[0], args.metric_sample, args.evaluation_seed)
    metrics = sample_conditional_metrics(
        X, layout, indices, args.neighbors, n_jobs=args.metric_jobs
    )
    pairs = sample_pairs(X.shape[0], args.distance_pairs, args.evaluation_seed + 1)
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
            "geometry_reference": args.input_representation,
            "geometry_scope": "uniform deterministic sample and random pairs",
        }
    )
    if metadata is None:
        metrics.update(
            {
                "label_metrics": None,
                "technical_covariate_leakage": None,
                "annotation_status": "metadata not supplied",
            }
        )
        return metrics

    label_metrics = donor_heldout_cell_type_metrics(
        layout,
        metadata[args.cell_type_column],
        metadata[args.donor_column],
        sample_size=args.label_sample,
        seed=args.evaluation_seed + 2,
        n_neighbors=args.neighbors,
        min_count=args.cell_type_min_count,
        min_donors=args.cell_type_min_donors,
        n_jobs=args.metric_jobs,
    )
    label_metrics.update(
        sampled_cell_type_silhouette(
            layout,
            metadata[args.cell_type_column],
            metadata[args.donor_column],
            sample_size=args.silhouette_sample,
            seed=args.evaluation_seed + 4,
            min_donors=args.cell_type_min_donors,
        )
    )
    covariates = {name: metadata[name] for name in args.technical_columns if name in metadata}
    leakage = technical_covariate_leakage(
        layout,
        metadata[args.cell_type_column],
        metadata[args.donor_column],
        covariates,
        sample_size=args.technical_sample,
        seed=args.evaluation_seed + 3,
        n_neighbors=args.neighbors,
        min_donors=args.cell_type_min_donors,
        n_jobs=args.metric_jobs,
    )
    metrics.update(
        {
            "label_metrics": label_metrics,
            "technical_covariate_leakage": leakage,
            "missing_technical_columns": [
                name for name in args.technical_columns if name not in metadata
            ],
            "annotation_status": "ok",
        }
    )
    return metrics


def _fit_worker_method(X, args, layout_path: Path):
    if args.worker_method == "autoencoder":
        layout, details = fit_autoencoder_large(X, args, layout_path)
        # The reusable large-corpus implementation assumes unit input rows only for
        # its optional cosine diagnostic. MSE/NMSE and every timing remain valid here.
        details["native_reconstruction_cosine"] = None
        details["native_reconstruction_cosine_status"] = (
            "not reported because single-cell prepared rows are intentionally unnormalized"
        )
        details["timing_scope"] = "validation selection + full refit + encoding/reconstruction"
        return layout, details
    if args.worker_method == "dire":
        return fit_dire_large(X, args)
    if args.worker_method == "umap":
        return fit_umap_large(X, args)
    if args.worker_method == "pca":
        return fit_pca_large(X, args)
    raise ValueError(f"unknown method {args.worker_method!r}")


def run_worker(args) -> int:
    result_path = Path(args.worker_result)
    layout_path = Path(args.worker_layout)
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": args.dataset_name,
        "input_representation": args.input_representation,
        "method": args.worker_method,
        "seed": args.worker_seed,
        "target_dimension": args.worker_target_dimension,
        "prepared_data": str(Path(args.worker_data).resolve()),
        "input_protocol": INPUT_PROTOCOL,
        "launch_nonce": args.worker_launch_nonce,
    }
    try:
        X = np.load(args.worker_data, mmap_mode="r", allow_pickle=False)
        if X.ndim != 2 or X.dtype != np.float32:
            raise ValueError("worker data must be a two-dimensional float32 NPY")
        with PeakMonitor() as monitor:
            layout, details = _fit_worker_method(X, args, layout_path)
            if layout is not None:
                if layout.shape != (X.shape[0], args.worker_target_dimension):
                    raise RuntimeError("reducer returned a misaligned layout")
                if not np.all(np.isfinite(layout)):
                    raise RuntimeError("reducer returned non-finite coordinates")
                _save_layout_atomic(layout_path, layout)
        _atomic_json(
            result_path,
            {
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
            },
        )
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


def _worker_command(args, method, seed, target, layout, result, launch_nonce) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset-name",
        args.dataset_name,
        "--input-representation",
        args.input_representation,
        "--worker-data",
        str(Path(args.data).expanduser().resolve()),
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
        "--worker-launch-nonce",
        launch_nonce,
        "--neighbors",
        str(args.neighbors),
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
        "--ae-evaluation-precision",
        args.ae_evaluation_precision,
    ]
    if args.dire_repo:
        command.extend(("--dire-repo", str(args.dire_repo)))
    if args.dire_revision:
        command.extend(("--dire-revision", args.dire_revision))
    if args.ae_hidden_dimensions:
        command.extend(("--ae-hidden-dimensions", ",".join(map(str, args.ae_hidden_dimensions))))
    if args.ae_amp:
        command.append("--ae-amp")
    return command


def _method_configuration(args, method, n_features) -> dict:
    if method == "autoencoder":
        training, evaluation = _ae_precision_configuration(args)
        return {
            "hidden_dimensions": list(
                args.ae_hidden_dimensions or _auto_hidden_dimensions(n_features)
            ),
            "epochs": args.ae_epochs,
            "patience": args.ae_patience,
            "validation": args.ae_validation,
            "batch_size": args.ae_batch_size,
            "learning_rate": args.ae_learning_rate,
            "weight_decay": args.ae_weight_decay,
            "training_precision": training,
            "evaluation_precision": evaluation,
        }
    if method == "dire":
        return {
            "neighbors": args.neighbors,
            "iterations": args.dire_iterations,
            "cuvs_index": args.dire_cuvs_index,
            "normalize": False,
            "dire_repo": str(Path(args.dire_repo).resolve()) if args.dire_repo else None,
            "dire_revision_expected": args.dire_revision,
        }
    if method == "umap":
        return {
            "neighbors": args.neighbors,
            "build_algo": args.umap_build_algo,
            "min_dist": args.umap_min_dist,
            "spread": args.umap_spread,
            "metric": "euclidean",
        }
    return {"backend_preference": "cuml_then_sklearn"}


def run_identity(args, validation: dict, method: str, seed: int, target: int) -> tuple[dict, str]:
    identity = {
        "schema_version": 1,
        "harness_sha256": _sha256_file(Path(__file__).resolve()),
        "reducer_harness_sha256": _sha256_file(
            Path(__file__).with_name("compare_large_embeddings.py")
        ),
        "dataset": args.dataset_name,
        "input_representation": args.input_representation,
        "input_protocol": INPUT_PROTOCOL,
        "data_sha256": validation["data_sha256"],
        "manifest_sha256": validation["manifest_sha256"],
        "rows": validation["rows"],
        "features": validation["features"],
        "method": method,
        "seed": int(seed),
        "target_dimension": int(target),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "configuration": _method_configuration(args, method, validation["features"]),
    }
    digest = _sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    return identity, digest


def planned_runs(args, validation: dict) -> list[dict]:
    output = Path(args.output_dir).expanduser().resolve()
    runs = []
    for target in args.target_dimensions:
        for seed in args.seeds:
            for method in args.methods:
                stem = f"{method}-d{target}-s{seed}"
                layout = output / f"{stem}-layout.npy"
                result = output / f"{stem}.json"
                identity, digest = run_identity(args, validation, method, seed, target)
                command = _worker_command(
                    args, method, seed, target, layout, result, "<launch-nonce>"
                )
                runs.append(
                    {
                        "method": method,
                        "target_dimension": int(target),
                        "seed": int(seed),
                        "layout": str(layout),
                        "result": str(result),
                        "fit_run_identity": identity,
                        "fit_run_identity_sha256": digest,
                        "command": shlex.join(command),
                    }
                )
    return runs


def _reusable(result_path: Path, layout_path: Path, digest: str, rows: int, target: int):
    try:
        result = json.loads(result_path.read_text())
        layout = np.load(layout_path, mmap_mode="r", allow_pickle=False)
        return (
            result
            if result.get("status") == "ok"
            and result.get("fit_run_identity_sha256") == digest
            and layout.shape == (rows, target)
            and layout.dtype == np.float32
            and np.all(np.isfinite(layout))
            and result.get("layout_sha256") == _sha256_file(layout_path)
            else None
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def orchestrate(args) -> int:
    required = (args.cell_type_column, args.donor_column) if args.metadata else ()
    validation, metadata, manifest = validate_prepared_input(
        args.data,
        metadata_path=args.metadata,
        manifest_path=args.manifest,
        input_representation=args.input_representation,
        required_columns=required,
        technical_columns=args.technical_columns,
        block_rows=args.validation_block_rows,
    )
    if max(args.target_dimensions) > validation["features"]:
        raise ValueError("target dimensions cannot exceed prepared input feature width")
    if args.validate_input and not args.dry_run:
        print(
            json.dumps({"validation": validation, "manifest": manifest}, indent=2, sort_keys=True)
        )
        return 0
    runs = planned_runs(args, validation)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "validation": validation,
                    "input_protocol": INPUT_PROTOCOL,
                    "planned_run_count": len(runs),
                    "runs": runs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    X = np.load(validation["data"], mmap_mode="r", allow_pickle=False)
    rows = []
    for plan in runs:
        method = plan["method"]
        target = plan["target_dimension"]
        seed = plan["seed"]
        layout_path = Path(plan["layout"])
        result_path = Path(plan["result"])
        log_path = result_path.with_suffix(".log")
        prior = (
            None
            if args.force
            else _reusable(
                result_path,
                layout_path,
                plan["fit_run_identity_sha256"],
                validation["rows"],
                target,
            )
        )
        if prior is not None:
            row = prior
        else:
            nonce = uuid.uuid4().hex
            _atomic_json(
                result_path,
                {
                    "dataset": args.dataset_name,
                    "method": method,
                    "seed": seed,
                    "target_dimension": target,
                    "status": "running",
                    "launch_nonce": nonce,
                },
            )
            command = _worker_command(args, method, seed, target, layout_path, result_path, nonce)
            returncode, wall, timeout_error = _run_subprocess(command, log_path, args.timeout)
            try:
                attempt = json.loads(result_path.read_text())
            except (OSError, ValueError, TypeError):
                attempt = None
            matching = isinstance(attempt, dict) and attempt.get("launch_nonce") == nonce
            if timeout_error:
                row = {
                    "status": "timeout",
                    "error": timeout_error,
                    "layout": None,
                    "launch_nonce": nonce,
                }
            elif returncode or not matching:
                row = (
                    attempt
                    if matching
                    else {
                        "status": "failed",
                        "error": f"worker exited with code {returncode} without a matching result",
                        "layout": None,
                        "launch_nonce": nonce,
                    }
                )
            else:
                row = attempt
            row["worker_wall_seconds"] = wall
            row["worker_returncode"] = returncode
        row.update(
            {
                "dataset": args.dataset_name,
                "input_representation": args.input_representation,
                "input_protocol": INPUT_PROTOCOL,
                "method": method,
                "seed": seed,
                "target_dimension": target,
                "fit_run_identity": plan["fit_run_identity"],
                "fit_run_identity_sha256": plan["fit_run_identity_sha256"],
            }
        )
        if row.get("status") == "ok":
            metric_started = None
            try:
                layout = np.load(layout_path, mmap_mode="r", allow_pickle=False)
                args.current_target_dimension = target
                metric_started = time.perf_counter()
                row["metrics"] = evaluate_single_cell_layout(X, layout, args, metadata)
                row["metric_seconds"] = time.perf_counter() - metric_started
                metric_started = None
                row["layout_sha256"] = _sha256_file(layout_path)
                row["layout_shape"] = list(layout.shape)
                row["layout_dtype"] = str(layout.dtype)
            except Exception as exc:
                if metric_started is not None:
                    row["metric_seconds"] = time.perf_counter() - metric_started
                row["status"] = "metric_failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["metric_seconds"] = None
        worker_wall = row.get("worker_wall_seconds")
        metric_seconds = row.get("metric_seconds")
        row["benchmark_scoped_total_seconds"] = (
            float(worker_wall) + float(metric_seconds)
            if isinstance(worker_wall, (int, float)) and isinstance(metric_seconds, (int, float))
            else None
        )
        row["benchmark_scoped_total_protocol"] = (
            "worker subprocess wall time + parent metric evaluation; excludes data download, "
            "single-cell preparation, input validation, and artifact hashing"
        )
        rows.append(row)
        _atomic_json(result_path, row)

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "fixed-target single-cell comparison on preserved prepared values",
            "timing_scope": (
                "benchmark_scoped_total_seconds = worker_wall_seconds + metric_seconds; "
                "preparation/download/input validation are excluded"
            ),
            "input_validation": validation,
            "source_manifest": manifest,
            "dataset": args.dataset_name,
            "input_representation": args.input_representation,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "configuration": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("worker_") and key not in {"current_target_dimension"}
            },
        },
        "rows": rows,
    }
    _atomic_json(output / "results.json", payload)
    _write_csv(output / "results.csv", rows)
    failures = [row for row in rows if row.get("status") != "ok"]
    print(f"wrote {output / 'results.json'} ({len(rows)} runs, {len(failures)} failures)")
    return 1 if args.require_all and failures else 0


def parse_int_tuple(value):
    if value in (None, ""):
        return ()
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data")
    parser.add_argument("--metadata")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-name", default="single-cell")
    parser.add_argument("--input-representation", default="pca50")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--target-dimensions", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--metric-sample", type=int, default=5_000)
    parser.add_argument("--distance-pairs", type=int, default=200_000)
    parser.add_argument("--label-sample", type=int, default=20_000)
    parser.add_argument("--silhouette-sample", type=int, default=5_000)
    parser.add_argument("--technical-sample", type=int, default=20_000)
    parser.add_argument("--evaluation-seed", type=int, default=2718)
    parser.add_argument("--metric-jobs", type=int, default=1)
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--technical-columns", nargs="*", default=list(DEFAULT_TECHNICAL_COLUMNS))
    parser.add_argument("--cell-type-min-count", type=int, default=20)
    parser.add_argument("--cell-type-min-donors", type=int, default=3)
    parser.add_argument("--validation-block-rows", type=int, default=16_384)
    parser.add_argument("--output-dir", default="artifacts/single-cell/benchmark")
    parser.add_argument("--validate-input", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")

    parser.add_argument("--dire-repo")
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
        default="nn_descent",
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
    parser.add_argument(
        "--ae-evaluation-precision",
        choices=AE_EVALUATION_PRECISIONS,
        default="fp32",
    )

    parser.add_argument("--worker-data", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-target-dimension", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-layout", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--worker-launch-nonce", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        required = (
            args.worker_data,
            args.worker_method,
            args.worker_seed,
            args.worker_target_dimension,
            args.worker_layout,
            args.worker_result,
            args.worker_launch_nonce,
        )
        if any(value is None for value in required):
            parser.error("worker mode requires all internal worker arguments")
    elif args.data is None:
        parser.error("--data is required")
    if set(args.methods) - set(METHODS):
        parser.error(f"--methods must be drawn from {METHODS}")
    if any(value < 1 for value in args.target_dimensions):
        parser.error("target dimensions must be positive")
    if min(args.neighbors, args.metric_jobs, args.validation_block_rows) < 1:
        parser.error("neighbors, metric-jobs, and validation-block-rows must be positive")
    if (
        min(
            args.metric_sample,
            args.label_sample,
            args.silhouette_sample,
            args.technical_sample,
        )
        < 3
    ):
        parser.error("metric and annotation samples must contain at least three rows")
    if args.distance_pairs < 1 or args.cell_type_min_count < 2 or args.cell_type_min_donors < 2:
        parser.error("distance-pairs must be positive; cell-type thresholds must be at least two")
    if not 0 < args.ae_validation < 0.5:
        parser.error("ae-validation must be in (0, 0.5)")
    if min(args.ae_epochs, args.ae_patience, args.ae_batch_size) < 1:
        parser.error("AE epochs, patience, and batch size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    return run_worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
