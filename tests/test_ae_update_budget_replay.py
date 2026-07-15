"""Tests for the public-safe 1,204-update replay verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.verify_ae_update_budget_replay import (
    ANCHOR_UPDATE,
    DISCOVERY_PROTOCOL,
    EXPECTED_POOLS,
    EXPECTED_SEEDS,
    PRIOR_PROTOCOL,
    PROTOCOL,
    _state_sha256,
    execute,
    parse_args,
    verify,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _configuration() -> dict:
    return {
        "target_dimension": 8,
        "hidden_dimensions": [256, 128, 32],
        "architecture": "96-256-128-32-8-symmetric",
        "parameter_count": 124_328,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "loss": "elementwise reconstruction MSE",
        "batch_size": 8_192,
    }


def _source(path: str) -> dict:
    return {
        "path": path,
        "format": "fbin",
        "rows": 10_000_000,
        "dimension": 96,
        "dtype": "little-endian float32",
        "file_bytes": 3_840_000_008,
        "file_sha256": _hash("source"),
        "universe_policy": "fixed leading prefix",
        "universe_rows": 10_000_000,
        "input_transformation": "none",
    }


def _tune(pool: int, seed: int) -> dict:
    offset = (pool // 65_536 + seed) * 1e-8
    return {
        "rows": 65_536,
        "reconstruction_mse": 0.2 + offset,
        "reconstruction_nmse": 0.45 + offset,
        "reconstruction_cosine": 0.75 - offset,
        "mean_predictor_mse": 0.44 + offset,
    }


def _fixture(tmp_path: Path):
    torch = pytest.importorskip("torch")
    checkpoint_dir = tmp_path / "private-checkpoints"
    checkpoint_dir.mkdir()
    pool_hashes = {str(pool): _hash(f"pool-{pool}") for pool in EXPECTED_POOLS}
    test_hash = _hash("test")
    tune_hash = _hash("tune")
    prior_arms = []
    trajectories = []
    for pool in EXPECTED_POOLS:
        for seed in EXPECTED_SEEDS:
            torch.manual_seed(seed + pool)
            state = {
                "encoder.weight": torch.randn(3, 2),
                "decoder.bias": torch.randn(2),
            }
            state_hash = _state_sha256(state)
            tune = _tune(pool, seed)
            checkpoint = checkpoint_dir / f"best-seed-{seed}-pool-{pool}.pt"
            torch.save(
                {
                    "protocol": PRIOR_PROTOCOL,
                    "init_seed": seed,
                    "pool_size": pool,
                    "selected_update": ANCHOR_UPDATE,
                    "model_state_dict": state,
                },
                checkpoint,
            )
            prior_arms.append(
                {
                    "init_seed": seed,
                    "pool_size": pool,
                    "pool_indices_sha256": pool_hashes[str(pool)],
                    "initial_state_sha256": _hash(f"initial-{seed}"),
                    "fixed_budget": {"update": ANCHOR_UPDATE, "tune": tune},
                    "selection": {"best_update": ANCHOR_UPDATE},
                }
            )
            trajectories.append(
                {
                    "init_seed": seed,
                    "pool_size": pool,
                    "pool_indices_sha256": pool_hashes[str(pool)],
                    "initial_state_sha256": _hash(f"initial-{seed}"),
                    "checkpoints": [
                        {
                            "update": ANCHOR_UPDATE,
                            "model_state_sha256": state_hash,
                            "tune": tune,
                            "fixed_legacy_evaluation": None,
                            # Intentionally malformed: success demonstrates the verifier
                            # neither interprets nor compares geometry.
                            "geometry": "must-not-be-accessed",
                        }
                    ],
                }
            )

    configuration = _configuration()
    prior = {
        "protocol": PRIOR_PROTOCOL,
        "status": "ok",
        "arguments": {"shuffle_seed_offset": 10_000_019},
        "protocol_configuration": configuration,
        "source": _source("/private/prior/source.fbin"),
        "split": {
            "test": {"rows": 65_536, "seed": 1_001_732, "indices_sha256": test_hash},
            "tune": {"rows": 65_536, "seed": 2_718_281, "indices_sha256": tune_hash},
            "training": {
                "pool_seed": 3_141_593,
                "pool_indices_sha256": pool_hashes,
            },
        },
        "arms": prior_arms,
    }
    discovery = {
        "protocol": DISCOVERY_PROTOCOL,
        "status": "ok",
        "phase": "discovery",
        "protocol_document": {
            "path": "/private/frozen/protocol.md",
            "file_sha256": _hash("protocol"),
        },
        "arguments": {"shuffle_seed_offset": 10_000_019},
        "protocol_configuration": configuration,
        "source": _source("/private/discovery/source.fbin"),
        "split": {
            "fixed_legacy_evaluation": {
                "rows": 65_536,
                "seed": 1_001_732,
                "indices_sha256": test_hash,
                "materialized": False,
                "reconstruction_updates": [],
            },
            "tune": {"rows": 65_536, "seed": 2_718_281, "indices_sha256": tune_hash},
            "training_pool": {
                "pool_seed": 3_141_593,
                "prefix_indices_sha256": pool_hashes,
            },
        },
        "trajectories": trajectories,
    }
    discovery_path = tmp_path / "private-discovery.json"
    prior_path = tmp_path / "private-prior.json"
    discovery_path.write_text(json.dumps(discovery, indent=2) + "\n")
    prior_path.write_text(json.dumps(prior, indent=2) + "\n")
    output = tmp_path / "public" / "replay-verification.json"
    args = parse_args(
        [
            "--discovery-results",
            str(discovery_path),
            "--prior-results",
            str(prior_path),
            "--prior-checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(output),
        ]
    )
    return args, discovery, prior, discovery_path, prior_path, checkpoint_dir


def test_exact_replay_is_verified_and_public_output_has_no_paths(tmp_path):
    args, _, _, _, _, _ = _fixture(tmp_path)

    result = execute(args)
    public_text = args.output.read_text()

    assert result["protocol"] == PROTOCOL
    assert result["status"] == "ok"
    assert result["replay_verified"] is True
    assert result["external_prior_result_checked"] is True
    assert result["all_required_identities_match"] is True
    assert result["protocol_document_sha256"] == _hash("protocol")
    assert result["scope"] == {
        "pool_sizes": list(EXPECTED_POOLS),
        "init_seeds": list(EXPECTED_SEEDS),
        "trajectory_count": 6,
        "source_dataset_opened": False,
        "legacy_cohort_accessed": False,
        "geometry_accessed": False,
    }
    assert len(result["trajectories"]) == 6
    assert all(item["replay_verified"] for item in result["trajectories"])
    assert result["inputs"]["prior_checkpoints"]["count"] == 6
    assert str(tmp_path) not in public_text
    assert "/private/" not in public_text
    assert "must-not-be-accessed" not in public_text
    with pytest.raises(FileExistsError, match="pass --force"):
        execute(args)


@pytest.mark.parametrize("mismatch", ["tune", "model", "source", "split"])
def test_exact_identity_mismatch_produces_failed_public_verification(tmp_path, mismatch):
    (
        args,
        discovery,
        _,
        discovery_path,
        _,
        _,
    ) = _fixture(tmp_path)
    if mismatch == "tune":
        discovery["trajectories"][0]["checkpoints"][0]["tune"]["reconstruction_nmse"] += 1e-15
    elif mismatch == "model":
        discovery["trajectories"][0]["checkpoints"][0]["model_state_sha256"] = "0" * 64
    elif mismatch == "source":
        discovery["source"]["file_sha256"] = _hash("different source")
    else:
        discovery["split"]["tune"]["indices_sha256"] = _hash("different tune")
    discovery_path.write_text(json.dumps(discovery, indent=2) + "\n")

    result = execute(args)

    assert result["status"] == "failed"
    assert result["replay_verified"] is False
    assert result["all_required_identities_match"] is False
    assert str(tmp_path) not in args.output.read_text()


def test_discovery_legacy_access_is_rejected_without_reading_metrics(tmp_path):
    args, discovery, _, discovery_path, _, _ = _fixture(tmp_path)
    discovery["split"]["fixed_legacy_evaluation"]["materialized"] = True
    discovery_path.write_text(json.dumps(discovery, indent=2) + "\n")

    with pytest.raises(ValueError, match="legacy-cohort access"):
        verify(args)


def test_missing_checkpoint_is_a_failed_check_not_a_path_leak(tmp_path):
    args, _, _, _, _, checkpoint_dir = _fixture(tmp_path)
    missing = checkpoint_dir / f"best-seed-{EXPECTED_SEEDS[0]}-pool-{EXPECTED_POOLS[0]}.pt"
    missing.unlink()

    result = execute(args)

    failed = next(
        item
        for item in result["trajectories"]
        if item["pool_size"] == EXPECTED_POOLS[0] and item["init_seed"] == EXPECTED_SEEDS[0]
    )
    assert result["replay_verified"] is False
    assert failed["checks"]["checkpoint_file_exists"] is False
    assert str(checkpoint_dir) not in args.output.read_text()
