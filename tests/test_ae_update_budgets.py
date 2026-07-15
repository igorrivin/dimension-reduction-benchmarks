"""Focused CPU tests for the AE optimizer-update non-inferiority protocol."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from experiments.compare_ae_training_sizes import inspect_source
from experiments.compare_ae_update_budgets import (
    DEFAULT_CHECKPOINT_UPDATES,
    DEFAULT_POOL_SIZES,
    GEOMETRY_METRIC_KEYS,
    NONINFERIORITY_MARGINS,
    _test_args,
    execute,
    parse_args,
    run_experiment,
    signed_degradation,
    validate_args,
)


def _data(rows: int = 80, dimension: int = 6) -> np.ndarray:
    rng = np.random.default_rng(91)
    latent = rng.normal(size=(rows, 2)).astype(np.float32)
    projection = rng.normal(size=(2, dimension)).astype(np.float32)
    noise = 0.01 * rng.normal(size=(rows, dimension)).astype(np.float32)
    return np.ascontiguousarray(latent @ projection + noise, dtype=np.float32)


def _metric_signature(result: dict) -> list[tuple]:
    signature = []
    for run in result["trajectories"]:
        for checkpoint in run["checkpoints"]:
            signature.append(
                (
                    run["init_seed"],
                    run["pool_size"],
                    checkpoint["update"],
                    checkpoint["model_state_sha256"],
                    checkpoint["encoding"]["layout_sha256"],
                    checkpoint["tune"]["reconstruction_nmse"],
                    *(checkpoint["geometry"][key] for key in GEOMETRY_METRIC_KEYS),
                )
            )
    return signature


def test_discovery_trains_one_trajectory_per_seed_pool_and_never_reads_legacy(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "source.npy"
    np.save(path, _data())
    args = _test_args(data=path)

    result = run_experiment(args)

    configuration = result["protocol_configuration"]
    assert result["phase"] == "discovery"
    assert configuration["training_trajectories"] == 4
    assert configuration["one_trajectory_per_seed_and_pool"] is True
    assert configuration["legacy_used_for_discovery_selection_or_stopping"] is False
    assert result["split"]["fixed_legacy_evaluation"]["materialized"] is False
    assert result["split"]["fixed_legacy_evaluation"]["matrix_sha256"] is None
    assert result["split"]["fixed_legacy_evaluation"]["reconstruction_updates"] == []
    assert result["geometry_protocol"]["cohort_role"] == "disjoint tune cohort"
    assert (
        result["geometry_protocol"]["cohort_global_indices_sha256"]
        == result["split"]["tune"]["indices_sha256"]
    )
    assert result["split"]["disjointness"] == {
        "legacy_tune_intersection": 0,
        "legacy_training_intersection": 0,
        "tune_training_intersection": 0,
        "combined_rows": 72,
        "exact_unique_rows": 72,
        "legacy_used_for_training": False,
    }
    for seed in args.init_seeds:
        paired = [run for run in result["trajectories"] if run["init_seed"] == seed]
        assert len(paired) == 2
        assert len({run["initial_state_sha256"] for run in paired}) == 1
    for run in result["trajectories"]:
        assert run["optimizer_steps_executed"] == 8
        assert run["single_uninterrupted_training_trajectory"] is True
        assert run["model_reinitialized_between_checkpoints"] is False
        assert run["checkpoint_states_retained"] is False
        assert [item["update"] for item in run["checkpoints"]] == [0, 1, 2, 4, 8]
        assert run["checkpoints"][0]["model_state_sha256"] == run["initial_state_sha256"]
        for checkpoint in run["checkpoints"]:
            assert checkpoint["encoding"]["retained"] is False
            assert checkpoint["encoding"]["scope"] == "disjoint tune cohort"
            assert checkpoint["fixed_legacy_evaluation"] is None
            assert np.isfinite(checkpoint["tune"]["reconstruction_nmse"])
            assert all(np.isfinite(checkpoint["geometry"][key]) for key in GEOMETRY_METRIC_KEYS)
            assert checkpoint["comparison_to_reference"]["positive_means_worse"] is True
    assert result["artifacts"]["checkpoint_paths"] == []
    assert result["artifacts"]["retained_layout_paths"] == []
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*layout*.npy"))


def test_intermediate_evaluations_do_not_change_final_training_state(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "source.npy"
    np.save(path, _data())
    dense_args = _test_args(data=path, init_seeds=(101,), pool_sizes=(32,))
    terminal_args = _test_args(
        data=path,
        init_seeds=(101,),
        pool_sizes=(32,),
        checkpoint_updates=(8,),
        reference_update=8,
        geometry_updates=(8,),
    )

    dense = run_experiment(dense_args)["trajectories"][0]["checkpoints"][-1]
    terminal = run_experiment(terminal_args)["trajectories"][0]["checkpoints"][0]

    assert terminal["update"] == dense["update"] == 8
    assert terminal["model_state_sha256"] == dense["model_state_sha256"]
    assert terminal["tune"] == dense["tune"]
    assert terminal["geometry"] == dense["geometry"]


def test_cpu_sweep_is_metric_and_state_deterministic(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "source.npy"
    np.save(path, _data())
    args = _test_args(
        data=path,
        init_seeds=(101,),
        pool_sizes=(32,),
        checkpoint_updates=(0, 2, 8),
        geometry_updates=(0, 2, 8),
    )

    first = run_experiment(args)
    second = run_experiment(args)

    assert _metric_signature(first) == _metric_signature(second)
    assert first["split"] == second["split"]
    assert first["geometry_protocol"] == second["geometry_protocol"]


def test_execute_flattens_results_and_retains_only_requested_small_states(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "source.npy"
    np.save(path, _data())
    output = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoints"
    protocol_document = tmp_path / "protocol.md"
    protocol_document.write_text("# Frozen protocol\n")
    args = _test_args(
        data=path,
        output_dir=output,
        init_seeds=(101,),
        pool_sizes=(32,),
        checkpoint_updates=(0, 4, 8),
        geometry_updates=(0, 4, 8),
        retain_updates=(4, 8),
        checkpoint_dir=checkpoint_dir,
        protocol_document=protocol_document,
    )

    result = execute(args)
    loaded = json.loads((output / "update-budget-results.json").read_text())
    with (output / "update-budget-results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert loaded["protocol"] == result["protocol"]
    assert result["protocol_document"]["file_sha256"]
    assert result["protocol_document"]["binding"] == "frozen before the run"
    assert [int(row["update"]) for row in rows] == [0, 4, 8]
    assert float(rows[-1]["tune_nmse"]) == pytest.approx(
        result["trajectories"][0]["checkpoints"][-1]["tune"]["reconstruction_nmse"]
    )
    assert len(result["artifacts"]["checkpoint_paths"]) == 2
    for checkpoint in result["trajectories"][0]["retained_checkpoints"]:
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        assert payload["protocol"] == result["protocol"]
        assert payload["model_state_sha256"] == checkpoint["model_state_sha256"]
        assert payload["update"] in (4, 8)
    assert loaded["artifacts"]["csv_sha256"]
    assert not list(tmp_path.rglob("*layout*.npy"))
    with pytest.raises(FileExistsError, match="pass --force"):
        execute(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"checkpoint_updates": (0, 2, 2)}, "strictly increasing"),
        ({"checkpoint_updates": (0,), "reference_update": 0}, "trained checkpoint"),
        ({"pool_sizes": (16, 31)}, "divisible"),
        ({"metric_sample": 17}, "must fit"),
        ({"legacy_updates": (8,)}, "forbids"),
        ({"reference_update": 7}, "reference_update"),
        ({"retain_updates": (8,)}, "checkpoint_dir"),
    ],
)
def test_validation_rejects_ambiguous_or_leaky_discovery_designs(tmp_path, overrides, message):
    path = tmp_path / "source.npy"
    np.save(path, _data())
    args = _test_args(data=path, **overrides)

    with pytest.raises(ValueError, match=message):
        validate_args(args, inspect_source(path))


def test_confirmation_grid_is_frozen_before_legacy_access(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, _data())
    locked = (3, 4, 5, 8, 1_204)
    args = _test_args(
        data=path,
        phase="confirmation",
        checkpoint_updates=locked,
        reference_update=8,
        legacy_updates=locked,
        geometry_updates=(4, 8),
        locked_candidate_update=4,
        locked_step_updates=1,
        locked_reference_update=8,
    )

    validate_args(args, inspect_source(path))
    args.checkpoint_updates = (4, 8, 1_204)
    args.legacy_updates = args.checkpoint_updates
    with pytest.raises(ValueError, match="exactly B-step"):
        validate_args(args, inspect_source(path))


def test_signed_degradation_uses_one_positive_harm_orientation():
    assert signed_degradation("reconstruction_nmse", 1.01, 1.0) == pytest.approx(0.01)
    assert signed_degradation("reconstruction_cosine", 0.895, 0.9) == pytest.approx(0.005)
    assert signed_degradation("sample_conditional_trustworthiness", 0.898, 0.9) == pytest.approx(
        0.002
    )
    assert signed_degradation("sampled_pair_normalized_stress", 0.205, 0.2) == pytest.approx(0.005)
    assert NONINFERIORITY_MARGINS["sampled_local_decodability_nmse"]["scale"] == "relative"


def test_cli_defaults_are_tune_only_two_pool_extended_discovery(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, _data())

    args = parse_args(["--data", str(path)])

    assert args.phase == "discovery"
    assert args.pool_sizes == DEFAULT_POOL_SIZES == (65_536, 1_048_576)
    assert args.checkpoint_updates == DEFAULT_CHECKPOINT_UPDATES
    assert args.checkpoint_updates[0] == 0
    assert args.checkpoint_updates[-1] == args.reference_update == 4_816
    assert 1_204 in args.checkpoint_updates
    assert args.legacy_updates == ()
    assert args.geometry_updates == args.checkpoint_updates
    assert args.retain_updates == ()
    assert args.init_seeds == (1729, 1730, 1731)
