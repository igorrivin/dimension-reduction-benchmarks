#!/usr/bin/env python3
"""Audit DiRe NN-descent graph quality and its effect on single-cell layouts.

The experiment compares two or more NN-descent working-graph configurations at
fixed final ``k``.  Each approximate graph is audited against exact neighbors for
a deterministic set of queries, then reused for paired 2D and 8D DiRe layouts so
that target dimension does not introduce a second graph realization.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .compare_large_embeddings import (
        PeakMonitor,
        _atomic_json,
        _check_revision,
        _save_layout_atomic,
        _sha256_bytes,
        _sha256_file,
        _worker_hardware,
        _write_csv,
        deterministic_sample,
        implementation_identity,
        index_sha256,
        load_dire,
    )
    from .compare_single_cell_reducers import (
        DEFAULT_TECHNICAL_COLUMNS,
        INPUT_PROTOCOL,
        evaluate_single_cell_layout,
        validate_prepared_input,
    )
except ImportError:
    from compare_large_embeddings import (  # type: ignore[no-redef]
        PeakMonitor,
        _atomic_json,
        _check_revision,
        _save_layout_atomic,
        _sha256_bytes,
        _sha256_file,
        _worker_hardware,
        _write_csv,
        deterministic_sample,
        implementation_identity,
        index_sha256,
        load_dire,
    )
    from compare_single_cell_reducers import (  # type: ignore[no-redef]
        DEFAULT_TECHNICAL_COLUMNS,
        INPUT_PROTOCOL,
        evaluate_single_cell_layout,
        validate_prepared_input,
    )


@dataclass(frozen=True)
class GraphConfiguration:
    """NN-descent working-graph configuration at fixed final neighborhood width."""

    name: str
    graph_degree: int
    intermediate_graph_degree: int
    max_iterations: int = 20
    termination_threshold: float = 1e-4


def parse_graph_configuration(value: str) -> GraphConfiguration:
    """Parse ``NAME:GRAPH_DEGREE:INTERMEDIATE_DEGREE``."""
    fields = value.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "graph configuration must be NAME:GRAPH_DEGREE:INTERMEDIATE_DEGREE"
        )
    name, graph_degree, intermediate_degree = fields
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise argparse.ArgumentTypeError("graph configuration name is not filesystem-safe")
    try:
        graph_degree = int(graph_degree)
        intermediate_degree = int(intermediate_degree)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("graph degrees must be integers") from exc
    if graph_degree < 1 or intermediate_degree < graph_degree:
        raise argparse.ArgumentTypeError(
            "graph degree must be positive and intermediate degree must be at least graph degree"
        )
    return GraphConfiguration(name, graph_degree, intermediate_degree)


def remove_query_self(
    neighbors: np.ndarray,
    query_indices: np.ndarray,
    k: int,
    distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove each query's global row id from exact-search results."""
    neighbors = np.asarray(neighbors)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if neighbors.ndim != 2 or neighbors.shape[0] != query_indices.size:
        raise ValueError("neighbors and query indices are not aligned")
    if neighbors.shape[1] < k:
        raise ValueError("neighbor result is narrower than requested k")
    if distances is not None and np.asarray(distances).shape != neighbors.shape:
        raise ValueError("distances and neighbors are not aligned")
    clean_neighbors = np.empty((neighbors.shape[0], k), dtype=np.int64)
    clean_distances = (
        np.empty((neighbors.shape[0], k), dtype=np.float32) if distances is not None else None
    )
    for row, query_index in enumerate(query_indices):
        keep = neighbors[row] != query_index
        selected = np.flatnonzero(keep)[:k]
        if selected.size != k:
            raise ValueError("exact search returned an underfilled row after removing self")
        clean_neighbors[row] = neighbors[row, selected]
        if clean_distances is not None:
            clean_distances[row] = np.asarray(distances)[row, selected]
    return clean_neighbors, clean_distances


def neighbor_recall_summary(
    exact_neighbors: np.ndarray,
    approximate_neighbors: np.ndarray,
    ks: tuple[int, ...] = (1, 10, 30),
) -> dict:
    """Compute per-row set recall summaries for nested neighborhood widths."""
    exact = np.asarray(exact_neighbors, dtype=np.int64)
    approximate = np.asarray(approximate_neighbors, dtype=np.int64)
    if exact.ndim != 2 or approximate.ndim != 2 or exact.shape[0] != approximate.shape[0]:
        raise ValueError("exact and approximate neighbors must be aligned matrices")
    if not ks or min(ks) < 1 or max(ks) > min(exact.shape[1], approximate.shape[1]):
        raise ValueError("recall widths are incompatible with neighbor matrices")
    result: dict[str, float | int | dict] = {}
    widest = max(ks)
    widest_recall = None
    for k in ks:
        row_recall = np.fromiter(
            (
                len(set(exact[row, :k]).intersection(approximate[row, :k])) / k
                for row in range(exact.shape[0])
            ),
            dtype=np.float64,
            count=exact.shape[0],
        )
        result[f"mean_recall_at_{k}"] = float(row_recall.mean())
        result[f"exact_match_fraction_at_{k}"] = float(np.mean(row_recall == 1.0))
        if k == widest:
            widest_recall = row_recall
    assert widest_recall is not None
    result["row_recall_quantiles_at_max_k"] = {
        str(quantile): float(np.quantile(widest_recall, quantile))
        for quantile in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }
    result["query_rows"] = int(exact.shape[0])
    return result


def graph_contract_summary(indices: np.ndarray, n_rows: int) -> dict:
    """Report invalid, duplicate, or self-containing approximate graph rows."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != n_rows:
        raise ValueError("approximate graph has the wrong row count")
    row_ids = np.arange(n_rows, dtype=np.int64)[:, None]
    self_rows = np.any(indices == row_ids, axis=1)
    out_of_range_rows = np.any((indices < 0) | (indices >= n_rows), axis=1)
    duplicate_rows = np.fromiter(
        (np.unique(row).size != row.size for row in indices),
        dtype=bool,
        count=n_rows,
    )
    return {
        "rows": int(n_rows),
        "neighbors": int(indices.shape[1]),
        "self_rows": int(self_rows.sum()),
        "out_of_range_rows": int(out_of_range_rows.sum()),
        "duplicate_rows": int(duplicate_rows.sum()),
    }


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    os.replace(temporary, path)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _sync_cuda() -> None:
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


def _new_reducer(dire_rapids, args, config, *, target_dimension: int, seed: int):
    return dire_rapids.create_dire(
        backend="cuvs",
        n_components=target_dimension,
        n_neighbors=args.neighbors,
        init="pca",
        max_iter_layout=args.dire_iterations,
        metric=None,
        normalize=False,
        knn_backend="cuvs",
        cuvs_index_type="auto",
        cuvs_knn_method="all_neighbors",
        all_neighbors_algo="nn_descent",
        all_neighbors_n_clusters=1,
        all_neighbors_algo_params={
            "graph_degree": config.graph_degree,
            "intermediate_graph_degree": config.intermediate_graph_degree,
            "max_iterations": config.max_iterations,
            "termination_threshold": config.termination_threshold,
            "return_distances": True,
        },
        random_state=seed,
        verbose=False,
    )


def build_exact_reference(X, query_indices, neighbors):
    """Search fixed queries exactly against the complete input on the GPU."""
    import cupy as cp
    from cuvs.neighbors import brute_force

    _sync_cuda()
    total_started = time.perf_counter()
    stage_started = time.perf_counter()
    dataset = cp.asarray(X, dtype=cp.float32, order="C")
    queries = cp.asarray(np.asarray(X[query_indices]), dtype=cp.float32, order="C")
    _sync_cuda()
    staging_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    index = brute_force.build(dataset, metric="sqeuclidean")
    _sync_cuda()
    index_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    distances, indices = brute_force.search(index, queries, neighbors + 1)
    _sync_cuda()
    search_seconds = time.perf_counter() - stage_started
    indices_host = cp.asnumpy(cp.asarray(indices))
    distances_host = cp.asnumpy(cp.asarray(distances))
    clean_indices, clean_distances = remove_query_self(
        indices_host, query_indices, neighbors, distances_host
    )
    total_seconds = time.perf_counter() - total_started
    del dataset, queries, index, distances, indices
    cp.get_default_memory_pool().free_all_blocks()
    return (
        clean_indices,
        clean_distances,
        {
            "staging_seconds": staging_seconds,
            "index_seconds": index_seconds,
            "search_seconds": search_seconds,
            "total_seconds": total_seconds,
            "protocol": "exact cuVS brute-force queries against every prepared input row",
        },
    )


def build_approximate_graph(X, dire_rapids, args, config, seed):
    """Build one DiRe all-neighbors graph and retain its host graph contract."""
    builder = _new_reducer(
        dire_rapids,
        args,
        config,
        target_dimension=min(args.target_dimensions),
        seed=seed,
    )
    with PeakMonitor() as monitor:
        _sync_cuda()
        started = time.perf_counter()
        builder._compute_knn_all_neighbors(X)
        _sync_cuda()
        graph_seconds = time.perf_counter() - started
    indices = np.ascontiguousarray(builder._knn_indices, dtype=np.int64)
    distances = np.ascontiguousarray(builder._knn_distances, dtype=np.float32)
    if indices.shape != (X.shape[0], args.neighbors):
        raise RuntimeError("DiRe graph has an unexpected shape")
    if distances.shape != indices.shape or not np.all(np.isfinite(distances)):
        raise RuntimeError("DiRe graph distances violate the graph contract")
    return (
        indices,
        distances,
        {
            "graph_seconds": graph_seconds,
            "cuvs_knn_method": builder._last_cuvs_knn_method,
            "all_neighbors_algo": builder.all_neighbors_algo,
            "all_neighbors_n_clusters": builder.all_neighbors_n_clusters,
            "configuration": asdict(config),
            "memory": monitor.result(),
        },
    )


def fit_layout_from_graph(
    X, graph_indices, graph_distances, dire_rapids, args, config, target, seed
):
    """Run DiRe initialization and force layout using one already-audited graph."""
    reducer = _new_reducer(
        dire_rapids,
        args,
        config,
        target_dimension=target,
        seed=seed,
    )
    phase_seconds: dict[str, float] = {}

    def inject_graph(_data, chunk_size=None, use_fp16=None):
        del chunk_size, use_fp16
        reducer._knn_indices = graph_indices
        reducer._knn_distances = graph_distances
        reducer._knn_indices_torch = None
        reducer._last_knn_backend = "cuvs"
        reducer._last_cuvs_knn_method = "precomputed-audited-all-neighbors"

    original_initialize = reducer._initialize_embedding
    original_optimize = reducer._optimize_layout

    def timed_initialize(data):
        _sync_cuda()
        started = time.perf_counter()
        value = original_initialize(data)
        _sync_cuda()
        phase_seconds["initialization_seconds"] = time.perf_counter() - started
        return value

    def timed_optimize(initial_positions):
        _sync_cuda()
        started = time.perf_counter()
        value = original_optimize(initial_positions)
        _sync_cuda()
        phase_seconds["layout_seconds"] = time.perf_counter() - started
        return value

    reducer._compute_knn = inject_graph
    reducer._initialize_embedding = timed_initialize
    reducer._optimize_layout = timed_optimize
    with PeakMonitor() as monitor:
        _sync_cuda()
        started = time.perf_counter()
        layout = reducer.fit_transform(X)
        _sync_cuda()
        fit_seconds = time.perf_counter() - started
    phase_seconds["fit_from_precomputed_graph_seconds"] = fit_seconds
    phase_seconds["other_fit_seconds"] = max(
        0.0,
        fit_seconds
        - phase_seconds.get("initialization_seconds", 0.0)
        - phase_seconds.get("layout_seconds", 0.0),
    )
    return np.asarray(layout, dtype=np.float32), phase_seconds, monitor.result()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-name", default="single-cell")
    parser.add_argument("--input-representation", default="pca50")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dire-repo", required=True)
    parser.add_argument("--dire-revision")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--dire-iterations", type=int, default=128)
    parser.add_argument("--target-dimensions", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--graph-config",
        action="append",
        type=parse_graph_configuration,
        dest="graph_configurations",
    )
    parser.add_argument("--exact-query-sample", type=int, default=5_000)
    parser.add_argument("--evaluation-seed", type=int, default=2718)
    parser.add_argument("--metric-sample", type=int, default=5_000)
    parser.add_argument("--distance-pairs", type=int, default=200_000)
    parser.add_argument("--label-sample", type=int, default=20_000)
    parser.add_argument("--silhouette-sample", type=int, default=5_000)
    parser.add_argument("--technical-sample", type=int, default=20_000)
    parser.add_argument("--metric-jobs", type=int, default=1)
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument("--donor-column", default="donor_id")
    parser.add_argument("--technical-columns", nargs="*", default=list(DEFAULT_TECHNICAL_COLUMNS))
    parser.add_argument("--cell-type-min-count", type=int, default=20)
    parser.add_argument("--cell-type-min-donors", type=int, default=3)
    parser.add_argument("--validation-block-rows", type=int, default=16_384)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.graph_configurations is None:
        args.graph_configurations = [
            GraphConfiguration("minimum", 31, 62),
            GraphConfiguration("umap-matched", 64, 128),
        ]
    if len({config.name for config in args.graph_configurations}) != len(args.graph_configurations):
        parser.error("graph configuration names must be unique")
    if min(args.neighbors, args.dire_iterations, args.exact_query_sample, args.metric_jobs) < 1:
        parser.error("neighbors, iterations, query sample, and metric jobs must be positive")
    if any(config.graph_degree < args.neighbors + 1 for config in args.graph_configurations):
        parser.error("every graph degree must be at least neighbors + 1 to retain self safely")
    if any(target < 1 for target in args.target_dimensions) or len(set(args.seeds)) != len(
        args.seeds
    ):
        parser.error("target dimensions must be positive and seeds must be unique")
    return args


def _evaluation_args(args, target):
    return SimpleNamespace(
        current_target_dimension=target,
        metric_sample=args.metric_sample,
        evaluation_seed=args.evaluation_seed,
        neighbors=args.neighbors,
        metric_jobs=args.metric_jobs,
        distance_pairs=args.distance_pairs,
        input_representation=args.input_representation,
        cell_type_column=args.cell_type_column,
        donor_column=args.donor_column,
        label_sample=args.label_sample,
        silhouette_sample=args.silhouette_sample,
        cell_type_min_count=args.cell_type_min_count,
        cell_type_min_donors=args.cell_type_min_donors,
        technical_columns=args.technical_columns,
        technical_sample=args.technical_sample,
    )


def main(argv=None):
    args = parse_args(argv)
    validation, metadata, manifest = validate_prepared_input(
        args.data,
        metadata_path=args.metadata,
        manifest_path=args.manifest,
        input_representation=args.input_representation,
        required_columns=(args.cell_type_column, args.donor_column),
        technical_columns=args.technical_columns,
        block_rows=args.validation_block_rows,
    )
    if args.exact_query_sample > validation["rows"]:
        raise ValueError("exact query sample exceeds the prepared row count")
    if max(args.target_dimensions) > validation["features"]:
        raise ValueError("target dimension exceeds prepared feature width")

    plan = {
        "dataset": args.dataset_name,
        "rows": validation["rows"],
        "features": validation["features"],
        "neighbors": args.neighbors,
        "exact_query_sample": args.exact_query_sample,
        "graph_configurations": [asdict(config) for config in args.graph_configurations],
        "seeds": args.seeds,
        "target_dimensions": args.target_dimensions,
        "layout_runs": len(args.graph_configurations)
        * len(args.seeds)
        * len(args.target_dimensions),
        "input_protocol": INPUT_PROTOCOL,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    }
    if args.dry_run:
        print(json.dumps({"validation": validation, "plan": plan}, indent=2, sort_keys=True))
        return 0

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix a fresh audit with existing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    X = np.load(validation["data"], mmap_mode="r", allow_pickle=False)
    dire_rapids, dire_identity = load_dire(args.dire_repo)
    _check_revision(dire_identity["git_revision"], args.dire_revision)
    dire_identity["implementation_sha256"] = implementation_identity(dire_rapids.create_dire)[
        "sha256"
    ]

    experiment_started = time.perf_counter()
    query_indices = deterministic_sample(
        validation["rows"], args.exact_query_sample, args.evaluation_seed
    )
    print(f"building exact reference for {query_indices.size} queries", flush=True)
    exact_neighbors, exact_distances, exact_timing = build_exact_reference(
        X, query_indices, args.neighbors
    )
    exact_path = output / "exact-reference.npz"
    _save_npz_atomic(
        exact_path,
        query_indices=np.asarray(query_indices, dtype=np.int64),
        neighbors=exact_neighbors,
        distances=exact_distances,
    )
    exact_record = {
        "path": str(exact_path),
        "sha256": _sha256_file(exact_path),
        "query_index_sha256": index_sha256(query_indices),
        "neighbor_index_sha256": index_sha256(exact_neighbors),
        "shape": list(exact_neighbors.shape),
        "timing": exact_timing,
    }
    _atomic_json(output / "exact-reference.json", exact_record)
    print(f"exact reference complete in {exact_timing['total_seconds']:.3f}s", flush=True)

    graph_rows = []
    layout_rows = []
    for seed_position, seed in enumerate(args.seeds):
        configurations = list(args.graph_configurations)
        if seed_position % 2:
            configurations.reverse()
        for config in configurations:
            stem = f"graph-{config.name}-s{seed}"
            print(f"building {stem}", flush=True)
            graph_indices, graph_distances, graph_details = build_approximate_graph(
                X, dire_rapids, args, config, seed
            )
            graph_path = output / f"{stem}-indices.npy"
            _save_npy_atomic(graph_path, graph_indices)
            contract = graph_contract_summary(graph_indices, validation["rows"])
            if any(contract[key] for key in ("self_rows", "out_of_range_rows", "duplicate_rows")):
                raise RuntimeError(f"{stem} violates the graph contract: {contract}")
            recall = neighbor_recall_summary(
                exact_neighbors,
                graph_indices[query_indices],
                ks=tuple(sorted({1, min(10, args.neighbors), args.neighbors})),
            )
            graph_record = {
                "status": "ok",
                "dataset": args.dataset_name,
                "seed": seed,
                "configuration": asdict(config),
                "indices": str(graph_path),
                "indices_sha256": _sha256_file(graph_path),
                "indices_shape": list(graph_indices.shape),
                "indices_dtype": str(graph_indices.dtype),
                "distances_sha256": _sha256_bytes(
                    np.ascontiguousarray(graph_distances, dtype="<f4").tobytes()
                ),
                "contract": contract,
                "recall": recall,
                "details": graph_details,
                "exact_reference_sha256": exact_record["sha256"],
            }
            graph_rows.append(graph_record)
            _atomic_json(output / f"{stem}.json", graph_record)
            print(
                f"{stem}: recall@{args.neighbors}="
                f"{recall[f'mean_recall_at_{args.neighbors}']:.6f}, "
                f"graph={graph_details['graph_seconds']:.3f}s",
                flush=True,
            )

            for target in args.target_dimensions:
                run_stem = f"dire-{config.name}-d{target}-s{seed}"
                print(f"fitting {run_stem} from audited graph", flush=True)
                layout, phase_seconds, memory = fit_layout_from_graph(
                    X,
                    graph_indices,
                    graph_distances,
                    dire_rapids,
                    args,
                    config,
                    target,
                    seed,
                )
                layout_path = output / f"{run_stem}-layout.npy"
                _save_layout_atomic(layout_path, layout)
                metric_started = time.perf_counter()
                metrics = evaluate_single_cell_layout(
                    X, layout, _evaluation_args(args, target), metadata
                )
                metric_seconds = time.perf_counter() - metric_started
                result = {
                    "status": "ok",
                    "error": None,
                    "dataset": args.dataset_name,
                    "input_representation": args.input_representation,
                    "input_protocol": INPUT_PROTOCOL,
                    "method": "dire",
                    "variant": config.name,
                    "seed": seed,
                    "target_dimension": target,
                    "rows": validation["rows"],
                    "features": validation["features"],
                    "layout": str(layout_path),
                    "layout_sha256": _sha256_file(layout_path),
                    "layout_shape": list(layout.shape),
                    "layout_dtype": str(layout.dtype),
                    "graph_indices": str(graph_path),
                    "graph_indices_sha256": graph_record["indices_sha256"],
                    "graph_recall": recall,
                    "graph_configuration": asdict(config),
                    "graph_seconds": graph_details["graph_seconds"],
                    "phase_seconds": phase_seconds,
                    "end_to_end_unamortized_seconds": (
                        graph_details["graph_seconds"]
                        + phase_seconds["fit_from_precomputed_graph_seconds"]
                    ),
                    "metric_seconds": metric_seconds,
                    "metrics": metrics,
                    "memory": memory,
                    "hardware": _worker_hardware(),
                    "dire": dire_identity,
                }
                layout_rows.append(result)
                _atomic_json(output / f"{run_stem}.json", result)
                print(
                    f"{run_stem}: layout={phase_seconds['layout_seconds']:.3f}s, "
                    f"metrics={metric_seconds:.3f}s",
                    flush=True,
                )
                gc.collect()
            del graph_indices, graph_distances
            gc.collect()

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "fixed-query exact graph recall followed by paired DiRe layouts from one "
                "audited graph per configuration and seed"
            ),
            "plan": plan,
            "input_validation": validation,
            "source_manifest": manifest,
            "dire": dire_identity,
            "exact_reference": exact_record,
            "experiment_wall_seconds": time.perf_counter() - experiment_started,
        },
        "graphs": graph_rows,
        "rows": layout_rows,
    }
    _atomic_json(output / "results.json", payload)
    _write_csv(output / "results.csv", layout_rows)
    print(
        f"wrote {output / 'results.json'} "
        f"({len(graph_rows)} graphs, {len(layout_rows)} layouts, 0 failures)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
