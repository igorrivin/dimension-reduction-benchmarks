#!/usr/bin/env python3
"""Verify the 1,204-update AE replay without touching legacy data or geometry.

The extended update-budget discovery run deliberately includes update 1,204
as a replay anchor.  This verifier compares that anchor with the earlier
training-size experiment for the two shared pools and three shared seeds.  It
checks protocol, source, split, pool, initialization, model-state, and tune
reconstruction identities.  It reads only result metadata and the six small
prior checkpoint files; it never opens a source dataset, encoded layout,
legacy-cohort artifact, or geometry artifact.

The output is suitable for a public result bundle: it contains hashes, compact
checks, and tune reconstruction values, but no input or checkpoint paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DISCOVERY_PROTOCOL = "deep1b-ae-update-budget-sensitivity-v1"
PRIOR_PROTOCOL = "deep1b-ae-training-pool-sensitivity-v1"
PROTOCOL = "deep1b-ae-update-budget-replay-verification-v1"
SCHEMA_VERSION = 1
ANCHOR_UPDATE = 1_204
EXPECTED_POOLS = (65_536, 1_048_576)
EXPECTED_SEEDS = (1_729, 1_730, 1_731)
TUNE_KEYS = (
    "rows",
    "reconstruction_mse",
    "reconstruction_nmse",
    "reconstruction_cosine",
    "mean_predictor_mse",
)
CONFIGURATION_KEYS = (
    "target_dimension",
    "hidden_dimensions",
    "architecture",
    "parameter_count",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "loss",
    "batch_size",
)
SOURCE_KEYS = (
    "format",
    "rows",
    "dimension",
    "dtype",
    "file_bytes",
    "file_sha256",
    "universe_policy",
    "universe_rows",
    "input_transformation",
)
_MAX_JSON_BYTES = 128 * 2**20


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, block_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_sha256(state: Mapping[str, Any]) -> str:
    """Hash a state dict exactly as the two training harnesses do."""

    import torch

    if not state or any(not isinstance(name, str) for name in state):
        raise ValueError("checkpoint model_state_dict must be a non-empty string-keyed object")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise ValueError("checkpoint model_state_dict values must be tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input is unexpectedly large: {path.name}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain one object: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _object(payload: Mapping[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _sha256_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        adjective = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {adjective} integer")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tune_metrics(value: Any, *, context: str) -> dict[str, float | int]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} tune metrics must be an object")
    result: dict[str, float | int] = {}
    for key in TUNE_KEYS:
        if key == "rows":
            result[key] = _integer(value.get(key), name=f"{context}.tune.{key}", positive=True)
        else:
            result[key] = _finite(value.get(key), name=f"{context}.tune.{key}")
    return result


def _configuration_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    configuration = _object(payload, "protocol_configuration", context="result")
    return {key: configuration.get(key) for key in CONFIGURATION_KEYS}


def _source_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _object(payload, "source", context="result")
    identity = {key: source.get(key) for key in SOURCE_KEYS}
    _sha256_text(identity["file_sha256"], name="source.file_sha256")
    return identity


def _discovery_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    split = _object(payload, "split", context="discovery")
    legacy = _object(split, "fixed_legacy_evaluation", context="discovery.split")
    tune = _object(split, "tune", context="discovery.split")
    training = _object(split, "training_pool", context="discovery.split")
    hashes = _object(training, "prefix_indices_sha256", context="discovery.split.training_pool")
    if legacy.get("materialized") is not False or legacy.get("reconstruction_updates") != []:
        raise ValueError("discovery result indicates legacy-cohort access")
    return {
        "test_rows": legacy.get("rows"),
        "test_seed": legacy.get("seed"),
        "test_indices_sha256": _sha256_text(
            legacy.get("indices_sha256"), name="discovery legacy indices"
        ),
        "tune_rows": tune.get("rows"),
        "tune_seed": tune.get("seed"),
        "tune_indices_sha256": _sha256_text(
            tune.get("indices_sha256"), name="discovery tune indices"
        ),
        "pool_seed": training.get("pool_seed"),
        "pool_indices_sha256": {
            str(pool): _sha256_text(hashes.get(str(pool)), name=f"discovery pool {pool} indices")
            for pool in EXPECTED_POOLS
        },
    }


def _prior_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    split = _object(payload, "split", context="prior")
    test = _object(split, "test", context="prior.split")
    tune = _object(split, "tune", context="prior.split")
    training = _object(split, "training", context="prior.split")
    hashes = _object(training, "pool_indices_sha256", context="prior.split.training")
    return {
        "test_rows": test.get("rows"),
        "test_seed": test.get("seed"),
        "test_indices_sha256": _sha256_text(test.get("indices_sha256"), name="prior test indices"),
        "tune_rows": tune.get("rows"),
        "tune_seed": tune.get("seed"),
        "tune_indices_sha256": _sha256_text(tune.get("indices_sha256"), name="prior tune indices"),
        "pool_seed": training.get("pool_seed"),
        "pool_indices_sha256": {
            str(pool): _sha256_text(hashes.get(str(pool)), name=f"prior pool {pool} indices")
            for pool in EXPECTED_POOLS
        },
    }


def _trajectory_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    raw = payload.get("trajectories")
    if not isinstance(raw, list):
        raise ValueError("discovery.trajectories must be a list")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for trajectory in raw:
        if not isinstance(trajectory, dict):
            raise ValueError("every discovery trajectory must be an object")
        seed = _integer(trajectory.get("init_seed"), name="trajectory seed")
        pool = _integer(trajectory.get("pool_size"), name="trajectory pool", positive=True)
        identity = (pool, seed)
        if identity in result:
            raise ValueError(f"duplicate discovery trajectory {identity}")
        result[identity] = trajectory
    return result


def _prior_arm_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    raw = payload.get("arms")
    if not isinstance(raw, list):
        raise ValueError("prior.arms must be a list")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for arm in raw:
        if not isinstance(arm, dict):
            raise ValueError("every prior arm must be an object")
        seed = _integer(arm.get("init_seed"), name="prior arm seed")
        pool = _integer(arm.get("pool_size"), name="prior arm pool", positive=True)
        identity = (pool, seed)
        if identity in result:
            raise ValueError(f"duplicate prior arm {identity}")
        result[identity] = arm
    return result


def _anchor_checkpoint(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    raw = trajectory.get("checkpoints")
    if not isinstance(raw, list):
        raise ValueError("discovery trajectory checkpoints must be a list")
    matches = [
        checkpoint
        for checkpoint in raw
        if isinstance(checkpoint, dict) and checkpoint.get("update") == ANCHOR_UPDATE
    ]
    if len(matches) != 1:
        raise ValueError("discovery trajectory must contain exactly one 1204-update checkpoint")
    checkpoint = matches[0]
    if checkpoint.get("fixed_legacy_evaluation") is not None:
        raise ValueError("discovery anchor contains legacy-cohort metrics")
    return checkpoint


def _prior_anchor(arm: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float | int]]:
    fixed = _object(arm, "fixed_budget", context="prior arm")
    selection = _object(arm, "selection", context="prior arm")
    if fixed.get("update") != ANCHOR_UPDATE or selection.get("best_update") != ANCHOR_UPDATE:
        raise ValueError("prior arm checkpoint is not the 1204-update state")
    return fixed, _tune_metrics(fixed.get("tune"), context="prior arm fixed budget")


def _checkpoint_state(
    torch, path: Path, *, pool: int, seed: int
) -> tuple[str, str, dict[str, bool]]:
    checks = {"file_exists": path.is_file()}
    if not checks["file_exists"]:
        return "", "", checks
    file_sha256 = _sha256_file(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checks["safe_load"] = False
        return file_sha256, "", checks
    checks["safe_load"] = True
    if not isinstance(payload, dict):
        checks["metadata"] = False
        return file_sha256, "", checks
    expected = {
        "protocol": PRIOR_PROTOCOL,
        "init_seed": seed,
        "pool_size": pool,
        "selected_update": ANCHOR_UPDATE,
    }
    checks["metadata"] = {key: payload.get(key) for key in expected} == expected
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        checks["state_dict"] = False
        return file_sha256, "", checks
    checks["state_dict"] = True
    try:
        state_sha256 = _state_sha256(state)
    except ValueError:
        checks["state_hash"] = False
        return file_sha256, "", checks
    checks["state_hash"] = True
    return file_sha256, state_sha256, checks


def verify(args) -> dict[str, Any]:
    import torch

    discovery, discovery_file_sha256 = _load_json(args.discovery_results)
    prior, prior_file_sha256 = _load_json(args.prior_results)
    protocol_document = discovery.get("protocol_document")
    protocol_document_sha256 = None
    if isinstance(protocol_document, dict) and protocol_document.get("file_sha256") is not None:
        protocol_document_sha256 = _sha256_text(
            protocol_document.get("file_sha256"), name="protocol_document.file_sha256"
        )
    protocol_checks = {
        "discovery_protocol": discovery.get("protocol") == DISCOVERY_PROTOCOL,
        "discovery_status": discovery.get("status") == "ok",
        "discovery_phase": discovery.get("phase") == "discovery",
        "prior_protocol": prior.get("protocol") == PRIOR_PROTOCOL,
        "prior_status": prior.get("status") == "ok",
    }

    discovery_configuration = _configuration_identity(discovery)
    prior_configuration = _configuration_identity(prior)
    configuration_match = discovery_configuration == prior_configuration
    discovery_arguments = _object(discovery, "arguments", context="discovery")
    prior_arguments = _object(prior, "arguments", context="prior")
    shuffle_identity = discovery_arguments.get("shuffle_seed_offset") == prior_arguments.get(
        "shuffle_seed_offset"
    )

    discovery_source = _source_identity(discovery)
    prior_source = _source_identity(prior)
    source_match = discovery_source == prior_source
    discovery_split = _discovery_split(discovery)
    prior_split = _prior_split(prior)
    split_match = discovery_split == prior_split

    trajectories = _trajectory_map(discovery)
    arms = _prior_arm_map(prior)
    expected = {(pool, seed) for pool in EXPECTED_POOLS for seed in EXPECTED_SEEDS}
    expected_discovery_present = expected.issubset(trajectories)
    expected_prior_present = expected.issubset(arms)

    trajectory_results: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    for pool in EXPECTED_POOLS:
        for seed in EXPECTED_SEEDS:
            identity = (pool, seed)
            trajectory = trajectories.get(identity)
            arm = arms.get(identity)
            if trajectory is None or arm is None:
                trajectory_results.append(
                    {
                        "pool_size": pool,
                        "init_seed": seed,
                        "replay_verified": False,
                        "checks": {
                            "discovery_trajectory_present": trajectory is not None,
                            "prior_arm_present": arm is not None,
                        },
                    }
                )
                continue

            checkpoint = _anchor_checkpoint(trajectory)
            _, prior_tune = _prior_anchor(arm)
            discovery_tune = _tune_metrics(checkpoint.get("tune"), context="discovery anchor")
            pool_hash = _sha256_text(
                trajectory.get("pool_indices_sha256"), name="discovery trajectory pool hash"
            )
            prior_pool_hash = _sha256_text(
                arm.get("pool_indices_sha256"), name="prior arm pool hash"
            )
            initial_hash = _sha256_text(
                trajectory.get("initial_state_sha256"), name="discovery initial-state hash"
            )
            prior_initial_hash = _sha256_text(
                arm.get("initial_state_sha256"), name="prior initial-state hash"
            )
            discovery_state_hash = _sha256_text(
                checkpoint.get("model_state_sha256"), name="discovery model-state hash"
            )
            checkpoint_name = f"best-seed-{seed}-pool-{pool}.pt"
            checkpoint_path = args.prior_checkpoint_dir / checkpoint_name
            file_hash, prior_state_hash, checkpoint_checks = _checkpoint_state(
                torch, checkpoint_path, pool=pool, seed=seed
            )
            checks = {
                "pool_identity": pool_hash == prior_pool_hash,
                "pool_matches_split": (
                    pool_hash
                    == discovery_split["pool_indices_sha256"][str(pool)]
                    == prior_split["pool_indices_sha256"][str(pool)]
                ),
                "initialization_identity": initial_hash == prior_initial_hash,
                "checkpoint_file_exists": checkpoint_checks.get("file_exists") is True,
                "checkpoint_safe_load": checkpoint_checks.get("safe_load") is True,
                "checkpoint_metadata": checkpoint_checks.get("metadata") is True,
                "checkpoint_state_hashable": checkpoint_checks.get("state_hash") is True,
                "model_state_identity": bool(prior_state_hash)
                and discovery_state_hash == prior_state_hash,
                "tune_reconstruction_exact": discovery_tune == prior_tune,
            }
            verified = all(checks.values())
            checkpoint_manifest.append(
                {
                    "filename": checkpoint_name,
                    "file_sha256": file_hash or None,
                }
            )
            trajectory_results.append(
                {
                    "pool_size": pool,
                    "init_seed": seed,
                    "anchor_update": ANCHOR_UPDATE,
                    "replay_verified": verified,
                    "checks": checks,
                    "hashes": {
                        "pool_indices_sha256": pool_hash,
                        "initial_state_sha256": initial_hash,
                        "discovery_model_state_sha256": discovery_state_hash,
                        "prior_model_state_sha256": prior_state_hash or None,
                        "prior_checkpoint_file_sha256": file_hash or None,
                        "tune_reconstruction_sha256": _sha256_json(discovery_tune),
                    },
                    "tune_reconstruction": discovery_tune,
                }
            )

    checks = {
        **protocol_checks,
        "protocol_document_bound": protocol_document_sha256 is not None,
        "configuration_identity": configuration_match,
        "shuffle_seed_offset_identity": shuffle_identity,
        "source_identity": source_match,
        "split_identity": split_match,
        "expected_discovery_trajectories_present": expected_discovery_present,
        "expected_prior_arms_present": expected_prior_present,
        "all_trajectory_replays_verified": bool(trajectory_results)
        and all(item["replay_verified"] for item in trajectory_results),
    }
    replay_verified = all(checks.values())
    checkpoint_manifest_sha256 = _sha256_json(checkpoint_manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok" if replay_verified else "failed",
        "generated_at_utc": _utc_now(),
        "replay_verified": replay_verified,
        "discovery_results_sha256": discovery_file_sha256,
        "protocol_document_sha256": protocol_document_sha256,
        "external_prior_result_checked": True,
        "all_required_identities_match": replay_verified,
        "anchor_update": ANCHOR_UPDATE,
        "scope": {
            "pool_sizes": list(EXPECTED_POOLS),
            "init_seeds": list(EXPECTED_SEEDS),
            "trajectory_count": len(EXPECTED_POOLS) * len(EXPECTED_SEEDS),
            "source_dataset_opened": False,
            "legacy_cohort_accessed": False,
            "geometry_accessed": False,
        },
        "inputs": {
            "discovery_results": {
                "protocol": DISCOVERY_PROTOCOL,
                "file_sha256": discovery_file_sha256,
            },
            "prior_results": {
                "protocol": PRIOR_PROTOCOL,
                "file_sha256": prior_file_sha256,
            },
            "prior_checkpoints": {
                "count": len(checkpoint_manifest),
                "manifest_sha256": checkpoint_manifest_sha256,
            },
        },
        "checks": checks,
        "identities": {
            "source": {key: value for key, value in discovery_source.items() if key != "path"},
            "split_sha256": _sha256_json(discovery_split),
            "configuration_sha256": _sha256_json(discovery_configuration),
        },
        "trajectories": trajectory_results,
        "limitations": [
            "This verifies exact replay at update 1204 only.",
            "It does not inspect or compare legacy-cohort or geometry metrics.",
            "It does not establish that update 1204 or any later checkpoint is optimal.",
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute(args) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError("replay output exists; pass --force to replace it")
    result = verify(args)
    _atomic_json(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--discovery-results", required=True, type=Path)
    parser.add_argument("--prior-results", required=True, type=Path)
    parser.add_argument("--prior-checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/ae-update-budget-sensitivity/replay-verification.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for name in ("discovery_results", "prior_results", "prior_checkpoint_dir", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute(args)
    print(
        f"wrote replay verification ({'verified' if result['replay_verified'] else 'failed'}) "
        f"to {args.output}"
    )
    return 0 if result["replay_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
