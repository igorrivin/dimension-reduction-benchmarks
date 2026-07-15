#!/usr/bin/env python3
r"""Evaluate several recall cutoffs from one exact streamed search per AE arm.

The generic Yandex evaluator accepts one ``k`` at a time.  This focused wrapper
performs a single exact top-``max(cutoffs)`` scan for every selected
training-pool-size checkpoint and computes strict ID recall from prefixes of
that one ordered neighbour table.  Thus the default Recall@10 and Recall@100
share exactly the same search, query prefix, and numerical results.

One arm can be supplied with the explicit layout options::

    python experiments/evaluate_ae_training_size_retrieval.py \
      --base-layout base.npy --query-layout query.npy \
      --encoding-provenance encoding.json --label pool-65536 \
      --ground-truth groundtruth.deep-10M.bin \
      --output-json retrieval.json --output-csv retrieval.csv

For a paired comparison, repeat a four-value arm specification::

    --arm pool-65536 base-64k.npy query-64k.npy encoding-64k.json \
    --arm pool-1m base-1m.npy query-1m.npy encoding-1m.json

Every layout is checked against its encoding provenance, including SHA-256,
shape, dtype, selected initialization seed and pool size.  Distances are
computed by the existing bounded-memory float64 exact-search implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from experiments.encode_ae_training_size_checkpoint import (
        PROTOCOL as ENCODING_PROTOCOL,
    )
    from experiments.evaluate_yandex_retrieval import (
        BoundedNpyReader,
        _atomic_json,
        _disk_preflight,
        _exact_streamed_neighbors,
        _inspect_layout,
        _load_layout,
        _peak_rss_bytes,
        _source_snapshot,
        _utc_now,
        _validate_ground_truth_values,
        _validate_query_values,
        conservative_memory_plan,
        open_ground_truth,
        validate_distinct_paths,
    )
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from encode_ae_training_size_checkpoint import (  # type: ignore[no-redef]
        PROTOCOL as ENCODING_PROTOCOL,
    )
    from evaluate_yandex_retrieval import (  # type: ignore[no-redef]
        BoundedNpyReader,
        _atomic_json,
        _disk_preflight,
        _exact_streamed_neighbors,
        _inspect_layout,
        _load_layout,
        _peak_rss_bytes,
        _source_snapshot,
        _utc_now,
        _validate_ground_truth_values,
        _validate_query_values,
        conservative_memory_plan,
        open_ground_truth,
        validate_distinct_paths,
    )


PROTOCOL = "deep1b-ae-training-size-multicutoff-retrieval-v1"
SCHEMA_VERSION = 1
DEFAULT_CUTOFFS = (10, 100)
_GIB = 2**30
_MAX_PROVENANCE_BYTES = 16 * 2**20


@dataclass(frozen=True)
class ArmSpec:
    """One encoded checkpoint layout and its provenance record."""

    label: str
    base_layout: Path
    query_layout: Path
    encoding_provenance: Path


def _sha256_file(path: Path, block_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal string")
    return value


def _object(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"encoding provenance field {key!r} must be an object")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _load_provenance(path: Path) -> tuple[dict[str, Any], str]:
    size = path.stat().st_size
    if size > _MAX_PROVENANCE_BYTES:
        raise ValueError(f"encoding provenance is unexpectedly large: {size} bytes")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid encoding provenance JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("encoding provenance must be a JSON object")
    if payload.get("protocol") != ENCODING_PROTOCOL or payload.get("status") != "ok":
        raise ValueError("input is not a successful training-size encoding provenance record")
    return payload, hashlib.sha256(raw).hexdigest()


def _verify_layout_record(
    record: Mapping[str, Any],
    *,
    kind: str,
    path: Path,
    rows: int,
    dimension: int,
    dtype: np.dtype,
) -> dict[str, Any]:
    expected_sha256 = _sha256_text(record.get("layout_sha256"), name=f"{kind}.layout_sha256")
    observed_sha256 = _sha256_file(path)
    checks = {
        "sha256": observed_sha256 == expected_sha256,
        "rows": record.get("layout_rows") == rows,
        "dimension": record.get("layout_dimension") == dimension,
        "dtype": record.get("layout_dtype") == str(dtype),
        "file_bytes": record.get("layout_file_bytes") == path.stat().st_size,
    }
    if not all(checks.values()):
        failures = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"{kind} layout disagrees with encoding provenance: {failures}")
    return {
        "verified": True,
        "checks": checks,
        "sha256": observed_sha256,
        "recorded_path": record.get("layout_path"),
        "recorded_path_matches": record.get("layout_path") == str(path.resolve()),
    }


def _verify_encoding_provenance(
    spec: ArmSpec,
    *,
    base_rows: int,
    base_dimension: int,
    base_dtype: np.dtype,
    query_rows: int,
    query_dimension: int,
    query_dtype: np.dtype,
) -> dict[str, Any]:
    payload, provenance_sha256 = _load_provenance(spec.encoding_provenance)
    selection = _object(payload, "selection")
    model = _object(payload, "model")
    base = _object(payload, "base")
    query = _object(payload, "query")
    training_result = _object(payload, "training_result")
    split = _object(training_result, "split")
    checkpoint = _object(payload, "checkpoint")
    init_seed = _nonnegative_integer(selection.get("init_seed"), name="selection.init_seed")
    pool_size = _positive_integer(selection.get("pool_size"), name="selection.pool_size")
    selected_update = _positive_integer(
        selection.get("selected_update"), name="selection.selected_update"
    )
    if split.get("verified") is not True or split.get("selected_pool_size") != pool_size:
        raise ValueError("encoding provenance does not contain a verified matching training split")
    if model.get("target_dimension") != base_dimension or query_dimension != base_dimension:
        raise ValueError("layout dimensions disagree with the encoded model target dimension")
    if checkpoint.get("init_seed") != init_seed or checkpoint.get("pool_size") != pool_size:
        raise ValueError("checkpoint identity disagrees with encoding selection")
    _sha256_text(checkpoint.get("state_dict_sha256"), name="checkpoint.state_dict_sha256")
    if query.get("all_rows_encoded") is not True:
        raise ValueError("encoding provenance does not assert that every query was encoded")
    if base.get("source_rows") != base_rows or query.get("source_rows") != query_rows:
        raise ValueError("encoded source row counts disagree with the supplied layouts")
    base_verification = _verify_layout_record(
        base,
        kind="base",
        path=spec.base_layout,
        rows=base_rows,
        dimension=base_dimension,
        dtype=base_dtype,
    )
    query_verification = _verify_layout_record(
        query,
        kind="query",
        path=spec.query_layout,
        rows=query_rows,
        dimension=query_dimension,
        dtype=query_dtype,
    )
    return {
        "verified": True,
        "encoding_provenance_path": str(spec.encoding_provenance.resolve()),
        "encoding_provenance_sha256": provenance_sha256,
        "training_result_sha256": _sha256_text(
            training_result.get("sha256"), name="training_result.sha256"
        ),
        "checkpoint_state_dict_sha256": checkpoint["state_dict_sha256"],
        "base_layout": base_verification,
        "query_layout": query_verification,
        "selection": {
            "init_seed": init_seed,
            "pool_size": pool_size,
            "selected_update": selected_update,
        },
    }


def _normalize_cutoffs(values: Sequence[int]) -> tuple[int, ...]:
    cutoffs = tuple(_positive_integer(value, name="cutoff") for value in values)
    if not cutoffs:
        raise ValueError("at least one cutoff is required")
    if tuple(sorted(set(cutoffs))) != cutoffs:
        raise ValueError("cutoffs must be unique and strictly increasing")
    return cutoffs


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        return _normalize_cutoffs(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _resolve_arms(args) -> tuple[ArmSpec, ...]:
    repeated = tuple(
        ArmSpec(label, Path(base), Path(query), Path(provenance))
        for label, base, query, provenance in (args.arm or ())
    )
    single_values = (args.base_layout, args.query_layout, args.encoding_provenance)
    has_single = any(value is not None for value in single_values)
    if repeated and (has_single or args.label is not None):
        raise ValueError(
            "use repeated --arm specifications or the single-layout/label options, not both"
        )
    if repeated:
        arms = repeated
    else:
        if not all(value is not None for value in single_values):
            raise ValueError(
                "supply --arm, or all of --base-layout, --query-layout, and --encoding-provenance"
            )
        arms = (
            ArmSpec(
                args.label or "single",
                Path(args.base_layout),
                Path(args.query_layout),
                Path(args.encoding_provenance),
            ),
        )
    labels = [arm.label for arm in arms]
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("arm labels must be non-empty and unique")
    return tuple(
        ArmSpec(
            arm.label,
            arm.base_layout.resolve(),
            arm.query_layout.resolve(),
            arm.encoding_provenance.resolve(),
        )
        for arm in arms
    )


def _quality_at_cutoffs(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    query_count = retrieved_ids.shape[0]
    hits: dict[str, int] = {}
    denominators: dict[str, int] = {}
    recalls: dict[str, float] = {}
    for cutoff in cutoffs:
        count = 0
        truth = ground_truth_ids[:query_count, :cutoff]
        for retrieved_row, truth_row in zip(retrieved_ids[:, :cutoff], truth, strict=True):
            count += int(np.intersect1d(retrieved_row, truth_row, assume_unique=True).size)
        key = str(cutoff)
        denominator = query_count * cutoff
        hits[key] = count
        denominators[key] = denominator
        recalls[key] = count / denominator
    return {
        "recall_at": recalls,
        "recall_hits": hits,
        "recall_denominator": denominators,
        "tie_policy": (
            "strict ground-truth ID intersection at each matching cutoff; alternative "
            "original-space distance ties are not credited"
        ),
        "definition": (
            "for each cutoff k, mean per-query set intersection between the first k "
            "retrieved and first k ground-truth IDs, divided by k"
        ),
    }


def _close_mapping(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _run_arm(
    spec: ArmSpec,
    *,
    gt_info,
    gt_ids: np.ndarray,
    gt_distances: np.ndarray,
    cutoffs: tuple[int, ...],
    args,
) -> dict[str, Any]:
    started = time.perf_counter()
    maximum_k = max(cutoffs)
    snapshots_before = {
        "base_layout": _source_snapshot(spec.base_layout),
        "query_layout": _source_snapshot(spec.query_layout),
        "encoding_provenance": _source_snapshot(spec.encoding_provenance),
    }
    base_info = _inspect_layout(spec.base_layout, name=f"{spec.label} base layout")
    query_layout = _load_layout(spec.query_layout, name=f"{spec.label} query layout")
    try:
        if base_info.dimension != query_layout.shape[1]:
            raise ValueError(f"arm {spec.label!r} base and query dimensions differ")
        if query_layout.shape[0] != gt_info.queries:
            raise ValueError(
                f"arm {spec.label!r} query rows must exactly match ground truth: "
                f"{query_layout.shape[0]} != {gt_info.queries}"
            )
        if maximum_k > min(base_info.rows, gt_info.neighbors):
            raise ValueError(
                f"maximum cutoff {maximum_k} exceeds the base or ground-truth neighbour count"
            )
        if args.query_limit is None:
            query_count = gt_info.queries
        else:
            query_count = _positive_integer(args.query_limit, name="query_limit")
            if query_count > gt_info.queries:
                raise ValueError("query_limit exceeds the available public queries")
        verification = _verify_encoding_provenance(
            spec,
            base_rows=base_info.rows,
            base_dimension=base_info.dimension,
            base_dtype=base_info.dtype,
            query_rows=int(query_layout.shape[0]),
            query_dimension=int(query_layout.shape[1]),
            query_dtype=query_layout.dtype,
        )
        query_batch_size = _positive_integer(args.query_batch_size, name="query_batch_size")
        base_block_rows = _positive_integer(args.base_block_rows, name="base_block_rows")
        budget_bytes = int(float(args.host_memory_budget_gib) * _GIB)
        if not np.isfinite(args.host_memory_budget_gib) or budget_bytes <= 0:
            raise ValueError("host_memory_budget_gib must be finite and positive")
        memory_plan = conservative_memory_plan(
            base_rows=base_info.rows,
            dimension=base_info.dimension,
            query_count=query_count,
            k=maximum_k,
            query_batch_size=query_batch_size,
            base_block_rows=base_block_rows,
            budget_bytes=budget_bytes,
            base_dtype_itemsize=base_info.dtype.itemsize,
        )

        validation_started = time.perf_counter()
        _validate_query_values(query_layout)
        _validate_ground_truth_values(
            gt_ids,
            gt_distances,
            base_rows=base_info.rows,
            k=maximum_k,
        )
        validation_seconds = time.perf_counter() - validation_started

        search_started = time.perf_counter()
        with BoundedNpyReader(base_info) as base_reader:
            all_distances, all_ids = _exact_streamed_neighbors(
                base_reader,
                query_layout,
                query_count=query_count,
                k=maximum_k,
                query_batch_size=memory_plan["effective_query_batch_size"],
                base_block_rows=memory_plan["effective_base_block_rows"],
            )
        search_seconds = time.perf_counter() - search_started
        quality = _quality_at_cutoffs(all_ids, gt_ids, cutoffs)
        digest = hashlib.sha256(np.ascontiguousarray(all_ids, dtype="<u8").tobytes()).hexdigest()
        snapshots_after = {
            "base_layout": _source_snapshot(spec.base_layout),
            "query_layout": _source_snapshot(spec.query_layout),
            "encoding_provenance": _source_snapshot(spec.encoding_provenance),
        }
        if snapshots_after != snapshots_before:
            raise ValueError(f"an input for arm {spec.label!r} changed during evaluation")
        distance_evaluations = query_count * base_info.rows
        selection = verification["selection"]
        return {
            "label": spec.label,
            "selection": selection,
            "verification": verification,
            "base_layout": {
                **snapshots_before["base_layout"],
                "rows": base_info.rows,
                "dimension": base_info.dimension,
                "dtype": str(base_info.dtype),
            },
            "query_layout": {
                **snapshots_before["query_layout"],
                "rows": int(query_layout.shape[0]),
                "rows_evaluated": query_count,
                "dimension": int(query_layout.shape[1]),
                "dtype": str(query_layout.dtype),
            },
            "search": {
                "maximum_k": maximum_k,
                "cutoffs_derived_from_same_topk_table": list(cutoffs),
                "exact_search_invocations": 1,
                "base_stream_passes": 1,
                "distance_evaluations": distance_evaluations,
                "query_batch_size_requested": query_batch_size,
                "query_batch_size_effective": memory_plan["effective_query_batch_size"],
                "base_block_rows_requested": base_block_rows,
                "base_block_rows_effective": memory_plan["effective_base_block_rows"],
                "retrieved_topk_ids_sha256_uint64_little_endian": digest,
                "minimum_retrieved_squared_distance": float(np.min(all_distances)),
                "maximum_retrieved_squared_distance": float(np.max(all_distances)),
                "full_query_by_base_matrix_materialized": False,
            },
            "quality": quality,
            "timing_seconds": {
                "input_value_validation": validation_seconds,
                "exact_layout_search": search_seconds,
                "arm_total": time.perf_counter() - started,
            },
            "throughput": {
                "distance_evaluations_per_second": (
                    distance_evaluations / search_seconds if search_seconds > 0 else None
                )
            },
            "resources": {
                "process_peak_rss_bytes": _peak_rss_bytes(),
                "host_memory_plan": memory_plan,
            },
            "input_unchanged_during_run": True,
        }
    finally:
        _close_mapping(query_layout)


def _summary_rows(
    arms: Sequence[Mapping[str, Any]], cutoffs: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        selection = arm["selection"]
        quality = arm["quality"]
        row = {
            "label": arm["label"],
            "init_seed": selection["init_seed"],
            "pool_size": selection["pool_size"],
            "selected_update": selection["selected_update"],
            "query_count": arm["query_layout"]["rows_evaluated"],
            "maximum_k": arm["search"]["maximum_k"],
            "exact_search_invocations": arm["search"]["exact_search_invocations"],
            "exact_search_seconds": arm["timing_seconds"]["exact_layout_search"],
        }
        for cutoff in cutoffs:
            key = str(cutoff)
            row[f"recall_at_{cutoff}"] = quality["recall_at"][key]
            row[f"recall_hits_at_{cutoff}"] = quality["recall_hits"][key]
            row[f"recall_denominator_at_{cutoff}"] = quality["recall_denominator"][key]
        rows.append(row)
    return rows


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty retrieval summary")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def run_evaluation(args) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _utc_now()
    cutoffs = _normalize_cutoffs(args.cutoffs)
    arms = _resolve_arms(args)
    ground_truth_path = Path(args.ground_truth).resolve()
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    if output_json.suffix.lower() != ".json" or output_csv.suffix.lower() != ".csv":
        raise ValueError("output_json and output_csv must have .json and .csv suffixes")
    paths = {
        "ground_truth": ground_truth_path,
        "output_json": output_json,
        "output_csv": output_csv,
    }
    for index, arm in enumerate(arms):
        paths[f"arm_{index}_base"] = arm.base_layout
        paths[f"arm_{index}_query"] = arm.query_layout
        paths[f"arm_{index}_provenance"] = arm.encoding_provenance
    validate_distinct_paths(paths)
    if not args.force and (output_json.exists() or output_csv.exists()):
        raise FileExistsError("retrieval output exists; pass --force to replace it")
    reserve_bytes = int(float(args.disk_reserve_gib) * _GIB)
    if not np.isfinite(args.disk_reserve_gib) or reserve_bytes < 0:
        raise ValueError("disk_reserve_gib must be finite and nonnegative")
    disk_plan = _disk_preflight(output_json, reserve_bytes=reserve_bytes)
    ground_truth_snapshot = _source_snapshot(ground_truth_path)
    gt_info, gt_ids, gt_distances = open_ground_truth(ground_truth_path)
    try:
        arm_results = [
            _run_arm(
                arm,
                gt_info=gt_info,
                gt_ids=gt_ids,
                gt_distances=gt_distances,
                cutoffs=cutoffs,
                args=args,
            )
            for arm in arms
        ]
        if _source_snapshot(ground_truth_path) != ground_truth_snapshot:
            raise ValueError("ground truth changed during retrieval evaluation")
    finally:
        _close_mapping(gt_ids)
        _close_mapping(gt_distances)
    query_counts = {arm["query_layout"]["rows_evaluated"] for arm in arm_results}
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "cutoffs": list(cutoffs),
        "maximum_k": max(cutoffs),
        "query_count": next(iter(query_counts)) if len(query_counts) == 1 else None,
        "query_selection": "leading public-query prefix",
        "engine": "numpy-cpu-float64-distance",
        "metric": "squared_euclidean",
        "tie_break": "ascending_base_row_id",
        "ground_truth": {
            **ground_truth_snapshot,
            "queries": gt_info.queries,
            "neighbors_per_query": gt_info.neighbors,
        },
        "arms": arm_results,
        "verification": {
            "all_encoding_provenance_verified": all(
                arm["verification"]["verified"] for arm in arm_results
            ),
            "one_exact_max_k_search_per_arm": all(
                arm["search"]["exact_search_invocations"] == 1 for arm in arm_results
            ),
            "ground_truth_unchanged_during_run": True,
        },
        "resources": {"disk_plan": disk_plan},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "timing_seconds": {"total_before_output_publication": time.perf_counter() - started},
    }
    return result


def execute(args) -> dict[str, Any]:
    result = run_evaluation(args)
    rows = _summary_rows(result["arms"], result["cutoffs"])
    _atomic_json(Path(args.output_json), result)
    _atomic_csv(Path(args.output_csv), rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        action="append",
        nargs=4,
        metavar=("LABEL", "BASE_LAYOUT", "QUERY_LAYOUT", "ENCODING_PROVENANCE"),
        help="Repeat once per checkpoint arm.",
    )
    parser.add_argument("--base-layout", type=Path)
    parser.add_argument("--query-layout", type=Path)
    parser.add_argument("--encoding-provenance", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cutoffs", type=_parse_cutoffs, default=DEFAULT_CUTOFFS)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--base-block-rows", type=int, default=131_072)
    parser.add_argument("--host-memory-budget-gib", type=float, default=4.0)
    parser.add_argument("--disk-reserve-gib", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser


def _test_args(**overrides):
    defaults = {
        "arm": None,
        "base_layout": None,
        "query_layout": None,
        "encoding_provenance": None,
        "label": None,
        "ground_truth": Path("ground-truth.bin"),
        "output_json": Path("retrieval.json"),
        "output_csv": Path("retrieval.csv"),
        "cutoffs": DEFAULT_CUTOFFS,
        "query_limit": None,
        "query_batch_size": 2,
        "base_block_rows": 64,
        "host_memory_budget_gib": 1.0,
        "disk_reserve_gib": 0.0,
        "force": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(f"evaluated {len(result['arms'])} arm(s) at cutoffs {result['cutoffs']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
