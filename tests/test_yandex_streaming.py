"""CPU tests for the genuinely bounded-memory Yandex AE scaling protocol."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import experiments.compare_yandex_streaming as streaming
import numpy as np
import pytest
from experiments.compare_yandex_streaming import (
    BoundedFbinReader,
    _test_args,
    bounded_validation_indices,
    conservative_host_memory_plan,
    deterministic_block_order,
    execute,
    iter_training_blocks,
    run_experiment,
)


def _write_fbin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *values.shape))
        handle.write(values.tobytes())


def _small_fbin(tmp_path: Path, rows: int = 41, dimension: int = 6) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(19)
    values = rng.normal(size=(rows, dimension)).astype(np.float32)
    path = tmp_path / "base.fbin"
    _write_fbin(path, values)
    return path, values


def test_bounded_validation_sample_is_deterministic_unique_and_small():
    first = bounded_validation_indices(1_000_000_000, 127, seed=5)
    second = bounded_validation_indices(1_000_000_000, 127, seed=5)

    assert np.array_equal(first, second)
    assert first.shape == (127,)
    assert np.all(np.diff(first) > 0)
    assert first[0] >= 0
    assert first[-1] < 1_000_000_000
    with pytest.raises(ValueError, match="validation_rows"):
        bounded_validation_indices(10, 10, seed=5)


def test_affine_block_order_is_a_deterministic_permutation():
    first = list(deterministic_block_order(103, 10, seed=7, pass_index=0))
    again = list(deterministic_block_order(103, 10, seed=7, pass_index=0))
    other_pass = list(deterministic_block_order(103, 10, seed=7, pass_index=1))

    assert first == again
    assert sorted(first) == list(range(11))
    assert sorted(other_pass) == list(range(11))
    assert first != other_pass


def test_training_blocks_are_bounded_deterministic_and_exclude_validation():
    validation = np.array([0, 3, 17, 18, 49, 63], dtype=np.int64)
    first = list(iter_training_blocks(64, 9, validation, seed=11, pass_index=0))
    second = list(iter_training_blocks(64, 9, validation, seed=11, pass_index=0))
    visited = np.concatenate([block.indices for block in first])

    assert all(block.indices.size <= 9 for block in first)
    assert all(
        np.array_equal(left.indices, right.indices)
        for left, right in zip(first, second, strict=True)
    )
    assert not set(visited).intersection(validation)
    assert sorted(visited.tolist()) == sorted(set(range(64)) - set(validation))


def test_fbin_reader_uses_owned_bounded_reads_without_a_corpus_mapping(tmp_path):
    data_path, values = _small_fbin(tmp_path, rows=23, dimension=5)
    info = streaming.read_fbin_header(data_path)
    with BoundedFbinReader(data_path, info, rows=19) as reader:
        block = reader.read_block(4, 9)
        selected = reader.read_rows(np.array([0, 2, 3, 11, 18], dtype=np.int64))

        assert reader.mapped_bytes == 0
        assert not isinstance(block, np.memmap)
        assert block.flags.writeable
        np.testing.assert_array_equal(block, values[4:9])
        np.testing.assert_array_equal(selected, values[[0, 2, 3, 11, 18]])


def test_conservative_host_memory_budget_rejects_corpus_sized_knobs():
    with pytest.raises(ValueError, match="host-memory-budget"):
        conservative_host_memory_plan(
            rows=1_000_000_000,
            input_dimension=96,
            target_dimension=8,
            validation_rows=900_000_000,
            block_rows=1_000_000_000,
            batch_size=1_000_000_000,
            budget_bytes=2 * 2**30,
        )

    plan = conservative_host_memory_plan(
        rows=1_000,
        input_dimension=96,
        target_dimension=8,
        validation_rows=100,
        block_rows=200,
        batch_size=50,
        budget_bytes=2 * 2**30,
    )
    components = plan["components_bytes"]
    assert components["heldout_simultaneous_float64_workspaces"] == 100 * 96 * 16
    assert components["floyd_sampler_python_set_upper_bound"] == 100 * 80


def test_cpu_streaming_run_has_exact_counts_finite_atomic_layout_and_metrics(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    data_path, _ = _small_fbin(tmp_path, rows=41, dimension=6)
    source_bytes = data_path.read_bytes()
    Path(f"{data_path}.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"url": "test://deep"},
                "prefix": {
                    "rows": 41,
                    "dimension": 6,
                    "file_bytes": len(source_bytes),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                },
            }
        )
    )
    layout_path = tmp_path / "layout.npy"
    args = _test_args(
        data=data_path,
        rows=41,
        layout_path=layout_path,
        result_path=tmp_path / "result.json",
        validation_rows=5,
        block_rows=7,
        batch_size=4,
        passes=2,
        hidden_dimensions=(10, 4),
    )

    def forbid_mapping(*_, **__):
        raise AssertionError("streaming harness attempted a corpus-sized NumPy mapping")

    with monkeypatch.context() as patch:
        patch.setattr(streaming.np, "memmap", forbid_mapping)
        patch.setattr(streaming.np.lib.format, "open_memmap", forbid_mapping)
        assert execute(args) == 0
    result = json.loads(args.result_path.read_text())
    layout = np.load(layout_path, allow_pickle=False)

    assert result["status"] == "ok"
    assert result["source"]["declared_rows"] == 41
    assert result["source"]["rows_used"] == 41
    assert result["source"]["download_manifest"]["content_sha256_verified"] is True
    assert result["budget"]["eligible_rows_per_pass"] == 36
    assert result["budget"]["training_examples"] == 72
    assert result["budget"]["training_elements"] == 72 * 6
    assert result["budget"]["training_payload_bytes"] == 72 * 6 * 4
    assert result["budget"]["completed_passes"] == 2
    assert result["budget"]["equivalent_passes"] == pytest.approx(2.0)
    assert result["validation"]["excluded_from_all_training_passes"] is True
    assert all(
        np.isfinite(result["validation"][name])
        for name in (
            "heldout_reconstruction_mse",
            "heldout_reconstruction_nmse",
            "heldout_reconstruction_cosine",
        )
    )
    assert result["precision"] == {
        "training": "float32",
        "loss": "float32",
        "parameters": ["float32"],
        "optimizer_state": ["float32"],
        "validation": "float32",
        "encoding": "float32",
        "bf16_requested": False,
    }
    assert result["streaming"]["corpus_sized_permutation"] is False
    assert result["streaming"]["full_corpus_device_copy"] is False
    assert result["streaming"]["persistent_input_mapping_bytes"] == 0
    assert result["streaming"]["persistent_layout_mapping_bytes"] == 0
    assert result["streaming"]["host_memory_preflight"]["persistent_corpus_mapping_bytes"] == 0
    assert layout.shape == (41, 2)
    assert layout.dtype == np.float32
    assert np.all(np.isfinite(layout))
    assert layout_path.is_symlink()
    assert result["run_id"] in layout_path.readlink().name
    assert not list(tmp_path.glob(".*.tmp"))


def test_update_cap_reports_exact_partial_budget(tmp_path):
    pytest.importorskip("torch")
    data_path, _ = _small_fbin(tmp_path, rows=29, dimension=5)
    args = _test_args(
        data=data_path,
        rows=29,
        layout_path=tmp_path / "layout.npy",
        result_path=tmp_path / "result.json",
        validation_rows=3,
        block_rows=8,
        batch_size=3,
        passes=4,
        max_updates=2,
        hidden_dimensions=(7,),
    )
    validation = bounded_validation_indices(29, 3, args.seed + args.validation_seed_offset)
    first_block = next(iter_training_blocks(29, 8, validation, seed=args.seed, pass_index=0))
    expected_examples = min(first_block.indices.size, 2 * args.batch_size)

    result = run_experiment(args)

    assert result["budget"]["optimizer_updates"] == 2
    assert result["budget"]["training_examples"] == expected_examples
    assert result["budget"]["completed_passes"] == 0
    assert result["budget"]["equivalent_passes"] == pytest.approx(expected_examples / 26)
    assert result["budget"]["stopped_by_update_cap"] is True
    assert result["budget"]["source_rows_materialized_for_training"] <= 8


def test_bf16_training_metadata_keeps_evaluation_fp32_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "autocast"):
        pytest.skip("torch does not expose autocast")
    data_path, _ = _small_fbin(tmp_path, rows=17, dimension=4)
    args = _test_args(
        data=data_path,
        rows=17,
        layout_path=tmp_path / "layout.npy",
        result_path=tmp_path / "result.json",
        validation_rows=3,
        block_rows=8,
        batch_size=4,
        hidden_dimensions=(6,),
        bf16=True,
    )

    result = run_experiment(args)

    assert result["precision"]["training"] == "bfloat16_autocast"
    assert result["precision"]["loss"] == "float32"
    assert result["precision"]["parameters"] == ["float32"]
    assert result["precision"]["optimizer_state"] == ["float32"]
    assert result["precision"]["validation"] == "float32"
    assert result["precision"]["encoding"] == "float32"


def test_execute_writes_explicit_failure_without_partial_layout(tmp_path):
    pytest.importorskip("torch")
    data_path, _ = _small_fbin(tmp_path, rows=10, dimension=3)
    result_path = tmp_path / "result.json"
    layout_path = tmp_path / "layout.npy"
    args = _test_args(
        data=data_path,
        rows=11,
        result_path=result_path,
        layout_path=layout_path,
        validation_rows=2,
    )

    assert execute(args) == 1
    payload = json.loads(result_path.read_text())
    assert payload["status"] == "failed"
    assert payload["failure"]["category"] == "invalid_input_or_integrity"
    assert "exceeds" in payload["failure"]["message"]
    assert not layout_path.exists()


def test_failed_force_rerun_preserves_previous_result_and_layout_pointer(tmp_path):
    pytest.importorskip("torch")
    data_path, _ = _small_fbin(tmp_path, rows=10, dimension=3)
    result_path = tmp_path / "result.json"
    layout_path = tmp_path / "layout.npy"
    np.save(layout_path, np.ones((2, 2), dtype=np.float32))
    previous_layout = layout_path.read_bytes()
    result_path.write_text('{"status": "ok", "old": true}\n')
    previous_result = result_path.read_text()
    args = _test_args(
        data=data_path,
        rows=11,
        result_path=result_path,
        layout_path=layout_path,
        validation_rows=2,
        force=True,
    )

    assert execute(args) == 1
    assert layout_path.read_bytes() == previous_layout
    assert result_path.read_text() == previous_result
    failures = list(tmp_path.glob("result.*.failed.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["status"] == "failed"


def test_pointer_publication_failure_is_nonfatal_and_keeps_generation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    data_path, _ = _small_fbin(tmp_path, rows=12, dimension=4)
    result_path = tmp_path / "result.json"
    layout_path = tmp_path / "layout.npy"
    args = _test_args(
        data=data_path,
        rows=12,
        result_path=result_path,
        layout_path=layout_path,
        validation_rows=2,
        hidden_dimensions=(6,),
    )

    def fail_pointer(*_):
        raise OSError("simulated pointer failure")

    monkeypatch.setattr(streaming, "_publish_layout_pointer", fail_pointer)
    assert execute(args) == 0
    result = json.loads(result_path.read_text())
    generation = Path(result["streaming"]["layout_generation_path"])
    assert result["status"] == "ok"
    assert result["streaming"]["layout_pointer_publication"]["status"] == "failed_nonfatal"
    assert generation.exists()
    assert np.all(np.isfinite(np.load(generation, allow_pickle=False)))
    assert not layout_path.exists()


def test_optional_query_fbin_is_strictly_hashed_and_encoded_in_fp32(tmp_path):
    pytest.importorskip("torch")
    data_path, _ = _small_fbin(tmp_path, rows=18, dimension=6)
    rng = np.random.default_rng(71)
    query_values = rng.normal(size=(7, 6)).astype(np.float32)
    query_path = tmp_path / "query.fbin"
    _write_fbin(query_path, query_values)
    result_path = tmp_path / "result.json"
    layout_path = tmp_path / "layout.npy"
    query_layout_path = tmp_path / "query-layout.npy"
    args = _test_args(
        data=data_path,
        query_data=query_path,
        rows=18,
        result_path=result_path,
        layout_path=layout_path,
        query_layout_path=query_layout_path,
        validation_rows=3,
        hidden_dimensions=(8, 4),
    )

    assert execute(args) == 0
    result = json.loads(result_path.read_text())
    query = result["query"]
    query_layout = np.load(query_layout_path, allow_pickle=False)
    assert query["source"]["rows"] == 7
    assert query["source"]["dimension"] == 6
    assert query["source"]["content_sha256"] == hashlib.sha256(query_path.read_bytes()).hexdigest()
    assert query["encoding_precision"] == "float32"
    assert query["layout_pointer_publication"]["status"] == "ok"
    assert query_layout_path.is_symlink()
    assert query_layout.shape == (7, 2)
    assert query_layout.dtype == np.float32
    assert np.all(np.isfinite(query_layout))
