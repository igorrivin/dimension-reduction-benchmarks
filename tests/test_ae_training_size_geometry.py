"""Focused CPU tests for post-hoc AE training-size geometry evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from experiments.compare_ae_training_sizes import (
    PROTOCOL as TRAINING_PROTOCOL,
)
from experiments.compare_ae_training_sizes import (
    _sha256_indices,
    _state_sha256,
)
from experiments.compare_yandex_streaming import (
    _build_autoencoder,
    bounded_validation_indices,
)
from experiments.evaluate_ae_training_size_geometry import (
    DEFAULT_DECODER_SAMPLE,
    DEFAULT_DISTANCE_PAIRS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_METRIC_SAMPLE,
    DEFAULT_NEIGHBORS,
    _test_args,
    _validate_published_configuration,
    execute,
    parse_args,
    reconstruct_test_indices,
)


def _write_fbin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", *values.shape))
        handle.write(values.tobytes())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_completed_run(tmp_path: Path):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(91)
    latent = rng.normal(size=(64, 2)).astype(np.float32)
    projection = rng.normal(size=(2, 6)).astype(np.float32)
    values = np.ascontiguousarray(latent @ projection, dtype=np.float32)
    data = tmp_path / "deep-test.fbin"
    _write_fbin(data, values)

    test_seed = 7
    test_indices = bounded_validation_indices(64, 32, test_seed)
    test_hash = _sha256_indices(test_indices)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    torch.manual_seed(101)
    model = _build_autoencoder(torch, 6, 2, (10, 4))
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_hash = _state_sha256(torch, state)
    arms = []
    for pool_size, update in ((8, 3), (16, 4)):
        checkpoint = checkpoint_dir / f"best-seed-101-pool-{pool_size}.pt"
        torch.save(
            {
                "protocol": TRAINING_PROTOCOL,
                "init_seed": 101,
                "pool_size": pool_size,
                "selected_update": update,
                "model_state_dict": state,
            },
            checkpoint,
        )
        arms.append(
            {
                "checkpoint_path": str(checkpoint),
                "init_seed": 101,
                "pool_size": pool_size,
                "pool_indices_sha256": _sha256_indices(np.arange(pool_size)),
                "initial_state_sha256": initial_hash,
                "selection": {"best_update": update},
                "test": {
                    "reconstruction_nmse": 0.5,
                    "reconstruction_cosine": 0.75,
                    "checkpoint_selected_without_test": True,
                },
            }
        )

    result = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "status": "ok",
        "started_at_utc": "2026-07-14T00:00:00+00:00",
        "finished_at_utc": "2026-07-14T00:01:00+00:00",
        "source": {
            "path": str(data),
            "format": "fbin",
            "rows": 64,
            "dimension": 6,
            "dtype": "little-endian float32",
            "file_bytes": data.stat().st_size,
            "file_sha256": _sha256_file(data),
            "universe_rows": 64,
            "input_transformation": "none",
        },
        "split": {
            "test": {
                "rows": 32,
                "seed": test_seed,
                "indices_sha256": test_hash,
            }
        },
        "protocol_configuration": {
            "target_dimension": 2,
            "hidden_dimensions": [10, 4],
        },
        "arms": arms,
    }
    results = tmp_path / "results.json"
    results.write_text(json.dumps(result))
    return data, results, checkpoint_dir, test_indices


def test_cpu_evaluator_writes_paired_geometry_without_layouts(tmp_path):
    data, results, checkpoint_dir, test_indices = _synthetic_completed_run(tmp_path)
    output = tmp_path / "geometry"
    args = _test_args(
        data=data,
        results=results,
        checkpoint_dir=checkpoint_dir,
        output_dir=output,
    )

    result = execute(args)
    loaded = json.loads((output / "geometry-results.json").read_text())
    with (output / "geometry-results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert loaded["protocol"] == result["protocol"]
    assert result["published_protocol"]["verified"] is False
    assert result["published_protocol"]["only_common_test_rows_encoded"] is True
    assert result["published_protocol"]["full_layouts_written"] is False
    assert result["test_data"]["global_indices_sha256"] == _sha256_indices(test_indices)
    assert result["metric_protocol"]["geometry_sample"]["rows"] == 12
    assert result["metric_protocol"]["decoder_sample"]["rows"] == 16
    assert result["metric_protocol"]["distance_pairs"]["count"] == 100
    assert len(result["arms"]) == len(rows) == 2
    assert result["pairing"]["groups"][0]["common_initialization_verified"] is True
    assert result["pairing"]["groups"][0]["complete_pool_size_block_verified"] is True
    assert all(arm["encoding"]["rows"] == 32 for arm in result["arms"])
    assert (
        result["arms"][0]["encoding"]["layout_sha256"]
        == result["arms"][1]["encoding"]["layout_sha256"]
    )
    assert all(
        delta == pytest.approx(0.0)
        for delta in result["arms"][0]["paired_comparison"][
            "metric_delta_arm_minus_reference"
        ].values()
    )
    assert result["artifacts"]["retained_layout_paths"] == []
    assert not list(output.glob("*.npy"))


def test_recomputed_test_hash_must_match_training_record(tmp_path):
    _, results, _, _ = _synthetic_completed_run(tmp_path)
    payload = json.loads(results.read_text())
    payload["split"]["test"]["indices_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="recomputed test indices"):
        reconstruct_test_indices(payload, require_published=False)


def test_published_cli_defaults_pin_paper_metric_protocol(tmp_path):
    args = parse_args(
        [
            "--results",
            str(tmp_path / "results.json"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--data",
            str(tmp_path / "data.fbin"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.evaluation_seed == DEFAULT_EVALUATION_SEED
    assert args.neighbors == DEFAULT_NEIGHBORS
    assert args.metric_sample == DEFAULT_METRIC_SAMPLE
    assert args.decoder_sample == DEFAULT_DECODER_SAMPLE
    assert args.distance_pairs == DEFAULT_DISTANCE_PAIRS
    _validate_published_configuration(args)

    args.metric_sample -= 1
    with pytest.raises(ValueError, match="published mode fixes"):
        _validate_published_configuration(args)
