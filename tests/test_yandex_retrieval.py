"""Tests for strict, bounded-memory Yandex layout retrieval evaluation."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from experiments.evaluate_yandex_retrieval import (
    _test_args,
    conservative_memory_plan,
    execute,
    inspect_ground_truth,
    open_ground_truth,
    run_evaluation,
)


def _write_ground_truth(
    path: Path,
    ids: np.ndarray,
    distances: np.ndarray | None = None,
) -> None:
    ids = np.ascontiguousarray(ids, dtype="<u4")
    if distances is None:
        distances = np.broadcast_to(np.arange(ids.shape[1], dtype=np.float32), ids.shape).copy()
    distances = np.ascontiguousarray(distances, dtype="<f4")
    assert ids.shape == distances.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *ids.shape))
        handle.write(ids.tobytes())
        handle.write(distances.tobytes())


def _known_case(tmp_path: Path) -> tuple[Path, Path, Path]:
    # Query 0 retrieves GT id 0; query 1 retrieves id 3 instead of GT id 2.
    base = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    queries = np.array([[0.1, 0.0], [29.9, 0.0]])
    base_path = tmp_path / "base.npy"
    query_path = tmp_path / "query.npy"
    gt_path = tmp_path / "gt.bin"
    np.save(base_path, base.astype(np.float32))
    np.save(query_path, queries.astype(np.float32))
    _write_ground_truth(gt_path, np.array([[0], [2]], dtype=np.uint32))
    return base_path, query_path, gt_path


def test_ground_truth_reader_maps_ids_and_distances_strictly(tmp_path):
    path = tmp_path / "gt.bin"
    ids = np.array([[3, 1], [4, 2]], dtype=np.uint32)
    distances = np.array([[0.25, 1.5], [2.0, 7.0]], dtype=np.float32)
    _write_ground_truth(path, ids, distances)

    info, mapped_ids, mapped_distances = open_ground_truth(path)

    assert (info.queries, info.neighbors) == (2, 2)
    assert info.expected_size == 8 + 2 * 2 * 8
    assert np.array_equal(mapped_ids, ids)
    assert np.array_equal(mapped_distances, distances)
    assert not mapped_ids.flags.writeable
    assert not mapped_distances.flags.writeable


@pytest.mark.parametrize("mutation", ["truncate", "append", "zero_axis", "short_header"])
def test_ground_truth_reader_rejects_malformed_lengths_and_shapes(tmp_path, mutation):
    path = tmp_path / "gt.bin"
    _write_ground_truth(path, np.array([[0, 1]], dtype=np.uint32))
    payload = path.read_bytes()
    if mutation == "truncate":
        path.write_bytes(payload[:-1])
    elif mutation == "append":
        path.write_bytes(payload + b"x")
    elif mutation == "zero_axis":
        path.write_bytes(struct.pack("<II", 0, 2))
    else:
        path.write_bytes(b"123")

    with pytest.raises(ValueError, match="ground-truth"):
        inspect_ground_truth(path)


def test_known_recall_and_atomic_json_output(tmp_path):
    base, queries, gt = _known_case(tmp_path)
    output = tmp_path / "result.json"
    args = _test_args(
        base_layout=base,
        query_layout=queries,
        ground_truth=gt,
        output=output,
        base_block_rows=2,
        query_batch_size=1,
    )

    assert execute(args) == 0
    result = json.loads(output.read_text())

    assert result["status"] == "ok"
    assert result["metric"] == "squared_euclidean"
    assert result["quality"]["recall@1"] == pytest.approx(0.5)
    assert result["quality"]["recall_hits"] == 1
    assert result["quality"]["recall_denominator"] == 2
    assert result["search"]["distance_evaluations"] == 8
    assert result["search"]["full_query_by_base_matrix_materialized"] is False
    assert len(result["search"]["retrieved_ids_sha256_uint64_little_endian"]) == 64
    assert not list(tmp_path.glob(".*.tmp"))


def test_neighbors_and_recall_are_invariant_to_block_and_query_batch_size(tmp_path):
    base, queries, gt = _known_case(tmp_path)
    first = run_evaluation(
        _test_args(
            base_layout=base,
            query_layout=queries,
            ground_truth=gt,
            output=tmp_path / "first.json",
            base_block_rows=1,
            query_batch_size=1,
        )
    )
    second = run_evaluation(
        _test_args(
            base_layout=base,
            query_layout=queries,
            ground_truth=gt,
            output=tmp_path / "second.json",
            base_block_rows=4,
            query_batch_size=2,
        )
    )

    assert first["quality"] == second["quality"]
    assert (
        first["search"]["retrieved_ids_sha256_uint64_little_endian"]
        == second["search"]["retrieved_ids_sha256_uint64_little_endian"]
    )


def test_query_limit_evaluates_a_deterministic_prefix(tmp_path):
    base, queries, gt = _known_case(tmp_path)
    result = run_evaluation(
        _test_args(
            base_layout=base,
            query_layout=queries,
            ground_truth=gt,
            output=tmp_path / "result.json",
            query_limit=1,
        )
    )

    assert result["query_layout"]["rows_evaluated"] == 1
    assert result["quality"]["recall@1"] == pytest.approx(1.0)
    assert result["search"]["distance_evaluations"] == 4


def test_ties_are_broken_by_ascending_base_id_across_blocks(tmp_path):
    base_path = tmp_path / "base.npy"
    query_path = tmp_path / "query.npy"
    gt_path = tmp_path / "gt.bin"
    np.save(base_path, np.zeros((6, 2), dtype=np.float32))
    np.save(query_path, np.zeros((1, 2), dtype=np.float32))
    _write_ground_truth(gt_path, np.array([[0, 1, 2]], dtype=np.uint32))

    one = run_evaluation(
        _test_args(
            base_layout=base_path,
            query_layout=query_path,
            ground_truth=gt_path,
            output=tmp_path / "one.json",
            k=3,
            base_block_rows=2,
        )
    )
    two = run_evaluation(
        _test_args(
            base_layout=base_path,
            query_layout=query_path,
            ground_truth=gt_path,
            output=tmp_path / "two.json",
            k=3,
            base_block_rows=5,
        )
    )

    assert one["quality"]["recall@3"] == 1.0
    assert (
        one["search"]["retrieved_ids_sha256_uint64_little_endian"]
        == two["search"]["retrieved_ids_sha256_uint64_little_endian"]
    )


def test_rejects_nonfinite_layout_bad_dtype_and_path_alias(tmp_path):
    base, queries, gt = _known_case(tmp_path)
    bad_values = tmp_path / "bad-values.npy"
    bad_dtype = tmp_path / "bad-dtype.npy"
    np.save(
        bad_values,
        np.array([[np.nan, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
    )
    np.save(bad_dtype, np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64))

    with pytest.raises(ValueError, match="non-finite"):
        run_evaluation(
            _test_args(
                base_layout=bad_values,
                query_layout=queries,
                ground_truth=gt,
                output=tmp_path / "values.json",
            )
        )
    with pytest.raises(ValueError, match="dtype"):
        run_evaluation(
            _test_args(
                base_layout=bad_dtype,
                query_layout=queries,
                ground_truth=gt,
                output=tmp_path / "dtype.json",
            )
        )
    aliased_output = tmp_path / "aliased-output.json"
    os.link(base, aliased_output)
    with pytest.raises(ValueError, match="path alias"):
        run_evaluation(
            _test_args(
                base_layout=base,
                query_layout=queries,
                ground_truth=gt,
                output=aliased_output,
            )
        )


def test_hard_link_alias_is_rejected(tmp_path):
    base, queries, gt = _known_case(tmp_path)
    alias = tmp_path / "alias.npy"
    os.link(base, alias)

    with pytest.raises(ValueError, match="path alias"):
        run_evaluation(
            _test_args(
                base_layout=base,
                query_layout=alias,
                ground_truth=gt,
                output=tmp_path / "result.json",
            )
        )


def test_layout_with_trailing_bytes_is_rejected(tmp_path):
    base, queries, gt = _known_case(tmp_path)
    with base.open("ab") as handle:
        handle.write(b"unexpected")

    with pytest.raises(ValueError, match="trailing bytes"):
        run_evaluation(
            _test_args(
                base_layout=base,
                query_layout=queries,
                ground_truth=gt,
                output=tmp_path / "result.json",
            )
        )


def test_conservative_memory_plan_rejects_oversized_tiles():
    with pytest.raises(ValueError, match="host-memory budget"):
        conservative_memory_plan(
            base_rows=1_000_000_000,
            dimension=8,
            query_count=10_000,
            k=100,
            query_batch_size=10_000,
            base_block_rows=1_000_000,
            budget_bytes=1 * 2**30,
        )

    plan = conservative_memory_plan(
        base_rows=10_000,
        dimension=8,
        query_count=10,
        k=5,
        query_batch_size=4,
        base_block_rows=100,
        budget_bytes=1 * 2**30,
    )
    assert plan["effective_query_batch_size"] == 4
    assert plan["effective_base_block_rows"] == 100
    assert plan["components_bytes"]["distance_matrix_float64"] == 4 * 100 * 8


def test_rejects_ground_truth_values_that_do_not_match_base_contract(tmp_path):
    base_path = tmp_path / "base.npy"
    query_path = tmp_path / "query.npy"
    gt_path = tmp_path / "gt.bin"
    np.save(base_path, np.zeros((2, 2), dtype=np.float32))
    np.save(query_path, np.zeros((1, 2), dtype=np.float32))
    _write_ground_truth(gt_path, np.array([[7]], dtype=np.uint32))

    with pytest.raises(ValueError, match="outside base layout"):
        run_evaluation(
            _test_args(
                base_layout=base_path,
                query_layout=query_path,
                ground_truth=gt_path,
                output=tmp_path / "result.json",
            )
        )
