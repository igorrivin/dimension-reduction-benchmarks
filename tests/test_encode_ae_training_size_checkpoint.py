"""Focused CPU tests for verified training-size-checkpoint layout encoding."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from experiments.compare_ae_training_sizes import _test_args as training_args
from experiments.compare_ae_training_sizes import execute as train
from experiments.compare_yandex_streaming import _build_autoencoder
from experiments.encode_ae_training_size_checkpoint import (
    _test_args,
    execute,
    parse_args,
)


def _write_fbin(path: Path, values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *values.shape))
        handle.write(values.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def trained_fixture(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("training-size-encoder")
    rng = np.random.default_rng(317)
    projection = rng.normal(size=(4, 96)).astype(np.float32)
    base_values = np.ascontiguousarray(
        rng.normal(size=(32, 4)).astype(np.float32) @ projection, dtype=np.float32
    )
    query_values = np.ascontiguousarray(
        rng.normal(size=(9, 4)).astype(np.float32) @ projection, dtype=np.float32
    )
    base = root / "base.fbin"
    query = root / "query.fbin"
    base_sha256 = _write_fbin(base, base_values)
    query_sha256 = _write_fbin(query, query_values)
    for path, values, sha256 in (
        (base, base_values, base_sha256),
        (query, query_values, query_sha256),
    ):
        Path(f"{path}.manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": {"url": f"test://{path.name}"},
                    "prefix": {
                        "rows": values.shape[0],
                        "dimension": values.shape[1],
                        "file_bytes": path.stat().st_size,
                        "sha256": sha256,
                    },
                }
            )
        )

    output = root / "training"
    checkpoints = root / "checkpoints"
    train(
        training_args(
            data=base,
            output_dir=output,
            checkpoint_dir=checkpoints,
            universe_rows=32,
            test_rows=4,
            tune_rows=4,
            pool_sizes=(8, 24),
            init_seeds=(101,),
            batch_size=4,
            evaluation_batch_size=4,
            update_budget=6,
            tune_interval_updates=2,
            target_dimension=8,
            hidden_dimensions=(256, 128, 32),
            load_block_rows=8,
        )
    )
    return {
        "base": base,
        "query": query,
        "base_values": base_values,
        "query_values": query_values,
        "results": output / "results.json",
        "checkpoint": checkpoints / "best-seed-101-pool-24.pt",
    }


def _local_args(tmp_path: Path, fixture, **overrides):
    defaults = {
        "data": fixture["base"],
        "query_data": fixture["query"],
        "training_results": fixture["results"],
        "checkpoint": fixture["checkpoint"],
        "init_seed": 101,
        "pool_size": 24,
        "base_layout": tmp_path / "base-layout.npy",
        "query_layout": tmp_path / "query-layout.npy",
        "provenance": tmp_path / "provenance.json",
        "block_rows": 7,
        "batch_size": 3,
    }
    defaults.update(overrides)
    return _test_args(**defaults)


def test_encoder_reproduces_full_base_and_all_queries_with_audited_provenance(
    tmp_path, trained_fixture
):
    torch = pytest.importorskip("torch")
    args = _local_args(tmp_path, trained_fixture)

    provenance = execute(args)
    base_layout = np.load(args.base_layout, allow_pickle=False)
    query_layout = np.load(args.query_layout, allow_pickle=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = _build_autoencoder(torch, 96, 8, (256, 128, 32))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        expected_base = model.encoder(torch.from_numpy(trained_fixture["base_values"])).numpy()
        expected_query = model.encoder(torch.from_numpy(trained_fixture["query_values"])).numpy()

    # CPU BLAS may choose a different, last-bit-equivalent GEMM kernel for the
    # direct all-row batch than for the encoder's bounded three-row batches.
    np.testing.assert_allclose(base_layout, expected_base, rtol=0, atol=2e-7)
    np.testing.assert_allclose(query_layout, expected_query, rtol=0, atol=2e-7)
    assert base_layout.shape == (32, 8)
    assert query_layout.shape == (9, 8)
    assert provenance["training_result"]["split"]["verified"] is True
    assert provenance["training_result"]["split"]["selected_pool_size"] == 24
    recorded = json.loads(trained_fixture["results"].read_text())
    selected_arm = next(arm for arm in recorded["arms"] if arm["pool_size"] == 24)
    assert provenance["checkpoint"]["selected_update"] == selected_arm["selection"]["best_update"]
    assert provenance["model"]["architecture"] == "96-256-128-32-8-symmetric"
    assert (
        provenance["base"]["source_content_sha256"]
        == provenance["base"]["expected_training_source_sha256"]
    )
    assert provenance["base"]["download_manifest"]["content_sha256_verified"] is True
    assert provenance["query"]["download_manifest"]["content_sha256_verified"] is True
    assert provenance["query"]["all_rows_encoded"] is True
    assert json.loads(args.provenance.read_text())["status"] == "ok"
    assert not list(tmp_path.glob(".*.staging"))


def test_tampered_split_hash_is_rejected_before_encoding(tmp_path, trained_fixture):
    result = json.loads(trained_fixture["results"].read_text())
    result["split"]["training"]["pool_indices_sha256"]["24"] = "0" * 64
    tampered = tmp_path / "results.json"
    tampered.write_text(json.dumps(result))
    args = _local_args(tmp_path, trained_fixture, training_results=tampered)

    with pytest.raises(ValueError, match="pool hash does not reproduce"):
        execute(args)
    assert not args.base_layout.exists()
    assert not args.query_layout.exists()
    assert not args.provenance.exists()


def test_checkpoint_metadata_mismatch_is_rejected(tmp_path, trained_fixture):
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(trained_fixture["checkpoint"], map_location="cpu", weights_only=True)
    checkpoint["selected_update"] += 1
    tampered = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, tampered)
    args = _local_args(tmp_path, trained_fixture, checkpoint=tampered)

    with pytest.raises(ValueError, match="checkpoint metadata disagrees"):
        execute(args)
    assert not args.base_layout.exists()
    assert not args.query_layout.exists()


def test_source_hash_mismatch_leaves_no_layouts_or_staging_files(tmp_path, trained_fixture):
    result = json.loads(trained_fixture["results"].read_text())
    result["source"]["file_sha256"] = "0" * 64
    tampered = tmp_path / "results.json"
    tampered.write_text(json.dumps(result))
    args = _local_args(tmp_path, trained_fixture, training_results=tampered)

    with pytest.raises(ValueError, match="base fbin content SHA-256"):
        execute(args)
    assert not args.base_layout.exists()
    assert not args.query_layout.exists()
    assert not args.provenance.exists()
    assert not list(tmp_path.glob(".*.staging"))


def test_cli_requires_explicit_selection_and_layout_paths(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--data", str(tmp_path / "base.fbin")])


def test_moved_checkpoint_is_allowed_but_disclosed(tmp_path, trained_fixture):
    moved = tmp_path / "transported-checkpoint.pt"
    shutil.copy2(trained_fixture["checkpoint"], moved)
    args = _local_args(tmp_path, trained_fixture, checkpoint=moved)

    provenance = execute(args)

    assert provenance["checkpoint"]["recorded_path_matches"] is False
    assert provenance["checkpoint"]["file_sha256"] == hashlib.sha256(moved.read_bytes()).hexdigest()
