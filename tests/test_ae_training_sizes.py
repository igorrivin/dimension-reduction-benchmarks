"""Focused CPU tests for the controlled AE training-pool-size protocol."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from experiments.compare_ae_training_sizes import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INIT_SEEDS,
    DEFAULT_POOL_SIZES,
    DEFAULT_UPDATE_BUDGET,
    PUBLISHED_TEST_INDICES_SHA256,
    PUBLISHED_TEST_SEED,
    _sha256_indices,
    _test_args,
    execute,
    inspect_source,
    make_experiment_indices,
    parse_args,
    run_experiment,
)


def _write_fbin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *values.shape))
        handle.write(values.tobytes())


def _data(rows: int = 64, dimension: int = 6) -> np.ndarray:
    rng = np.random.default_rng(91)
    latent = rng.normal(size=(rows, 2)).astype(np.float32)
    projection = rng.normal(size=(2, dimension)).astype(np.float32)
    return np.ascontiguousarray(latent @ projection, dtype=np.float32)


def test_published_10m_test_identity_is_pinned():
    indices = make_experiment_indices(
        universe_rows=10_000_000,
        test_rows=65_536,
        tune_rows=8,
        maximum_pool_size=16,
        test_seed=PUBLISHED_TEST_SEED,
        tune_seed=7,
        pool_seed=11,
    )

    assert _sha256_indices(indices.test) == PUBLISHED_TEST_INDICES_SHA256


def test_split_is_disjoint_and_training_pools_are_nested_uniform_prefixes():
    indices = make_experiment_indices(
        universe_rows=101,
        test_rows=11,
        tune_rows=13,
        maximum_pool_size=64,
        test_seed=3,
        tune_seed=5,
        pool_seed=7,
    )

    assert not np.intersect1d(indices.test, indices.tune).size
    assert not np.intersect1d(indices.test, indices.training_order).size
    assert not np.intersect1d(indices.tune, indices.training_order).size
    assert np.unique(indices.training_order).size == 64
    assert set(indices.training_order[:8]).issubset(indices.training_order[:32])
    assert set(indices.training_order[:32]).issubset(indices.training_order[:64])
    assert indices.unused.size == 13


def test_strict_npy_and_fbin_sources(tmp_path):
    values = _data(rows=17, dimension=5)
    npy = tmp_path / "source.npy"
    fbin = tmp_path / "source.fbin"
    np.save(npy, values)
    _write_fbin(fbin, values)

    assert inspect_source(npy).dimension == 5
    assert inspect_source(fbin).rows == 17

    bad_dtype = tmp_path / "float64.npy"
    np.save(bad_dtype, values.astype(np.float64))
    with pytest.raises(ValueError, match="native float32"):
        inspect_source(bad_dtype)
    bad_shape = tmp_path / "vector.npy"
    np.save(bad_shape, values[0])
    with pytest.raises(ValueError, match="two-dimensional"):
        inspect_source(bad_shape)


def test_cpu_protocol_has_paired_initialization_exact_budgets_and_no_test_leakage(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "source.npy"
    np.save(path, _data())
    args = _test_args(data=path, init_seeds=(101, 202), pool_sizes=(8, 16, 48))

    result = run_experiment(args)

    assert len(result["arms"]) == 6
    assert result["split"]["disjointness"] == {
        "test_tune_intersection": 0,
        "test_training_intersection": 0,
        "tune_training_intersection": 0,
        "combined_rows": 64,
        "exact_unique_rows": 64,
        "test_used_for_training_or_selection": False,
    }
    assert result["protocol_configuration"]["full_layouts_written"] is False
    for seed in args.init_seeds:
        paired = [arm for arm in result["arms"] if arm["init_seed"] == seed]
        assert len({arm["initial_state_sha256"] for arm in paired}) == 1
        for arm in paired:
            assert arm["first_pass"]["presentations"] == arm["pool_size"]
            assert arm["fixed_budget"]["update"] == 12
            assert arm["fixed_budget"]["presentations"] == 48
            assert arm["fixed_budget"]["unique_training_rows"] == arm["pool_size"]
            assert arm["fixed_budget"]["equivalent_passes"] == pytest.approx(48 / arm["pool_size"])
            assert arm["test"]["evaluated_once"] is True
            assert arm["test"]["checkpoint_selected_without_test"] is True
            assert np.isfinite(arm["test"]["reconstruction_nmse"])
            assert np.isfinite(arm["test"]["reconstruction_cosine"])
            assert arm["checkpoint_path"] is None
    assert not list(tmp_path.rglob("*.pt"))


def test_execute_writes_json_csv_and_checkpoints_only_on_request(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "source.fbin"
    _write_fbin(path, _data())
    output = tmp_path / "results"
    args = _test_args(
        data=path,
        output_dir=output,
        init_seeds=(101,),
        pool_sizes=(48,),
        checkpoint_dir=tmp_path / "checkpoints",
    )

    result = execute(args)
    loaded = json.loads((output / "results.json").read_text())
    with (output / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert loaded["protocol"] == result["protocol"]
    assert len(rows) == 1
    assert float(rows[0]["test_nmse"]) == pytest.approx(
        result["arms"][0]["test"]["reconstruction_nmse"]
    )
    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1
    assert loaded["arms"][0]["checkpoint_path"] == str(checkpoints[0].resolve())


def test_cli_defaults_are_the_exact_full_batch_design(tmp_path):
    data = tmp_path / "source.npy"
    np.save(data, _data())
    args = parse_args(["--data", str(data)])

    assert args.pool_sizes == DEFAULT_POOL_SIZES
    assert args.batch_size == DEFAULT_BATCH_SIZE
    assert args.update_budget == DEFAULT_UPDATE_BUDGET
    assert args.init_seeds == DEFAULT_INIT_SEEDS
    assert len(args.init_seeds) >= 3
    assert all(size % args.batch_size == 0 for size in args.pool_sizes)
    assert args.update_budget * args.batch_size == max(args.pool_sizes)
