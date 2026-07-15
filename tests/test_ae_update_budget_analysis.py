"""Tests for the public-safe AE optimizer-update analyzer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.analyze_ae_update_budgets import (
    FROZEN_COARSE_GRID,
    FROZEN_DISCOVERY_SEEDS,
    FROZEN_PROTOCOL_SHA256,
    METRICS,
    PROTOCOL,
    TRAINING_PROTOCOL,
    analyze,
    parse_args,
    signed_degradation,
)

REFERENCE_VALUES = {
    "reconstruction_nmse": 0.45,
    "reconstruction_cosine": 0.76,
    "sample_conditional_trustworthiness": 0.962,
    "sample_conditional_continuity": 0.971,
    "sample_conditional_neighbor_overlap": 0.312,
    "sampled_pair_distance_spearman": 0.51,
    "sampled_pair_normalized_stress": 0.27,
    "sampled_local_decodability_nmse": 0.52,
}


def _raw_margins():
    return {
        metric: {
            "margin": spec.margin,
            "scale": spec.margin_scale,
            "better": spec.better,
        }
        for metric, spec in METRICS.items()
    }


def _candidate_value(metric: str, reference: float, degradation: float) -> float:
    spec = METRICS[metric]
    signed_raw = degradation if spec.better == "lower" else -degradation
    if spec.margin_scale == "relative":
        return reference * (1.0 + signed_raw)
    return reference + signed_raw


def _checkpoint(
    *,
    phase: str,
    update: int,
    seed_offset: float,
    degradation_fraction: float,
):
    values = {}
    for metric, spec in METRICS.items():
        reference = REFERENCE_VALUES[metric] * (1.0 + seed_offset)
        values[metric] = _candidate_value(metric, reference, degradation_fraction * spec.margin)
    reconstruction = {
        "reconstruction_nmse": values["reconstruction_nmse"],
        "reconstruction_cosine": values["reconstruction_cosine"],
    }
    geometry = {
        metric: value for metric, value in values.items() if METRICS[metric].group == "geometry"
    }
    return {
        "update": update,
        "presentations": update * 8_192,
        "model_state_sha256": f"{update:064x}"[-64:],
        "training": {"cumulative_mse": 0.1},
        "tune": reconstruction if phase == "discovery" else None,
        "fixed_legacy_evaluation": reconstruction if phase == "confirmation" else None,
        "geometry": geometry,
        "geometry_cohort_role": (
            "fixed tune cohort" if phase == "discovery" else "fixed legacy evaluation cohort"
        ),
        "encoding": {"retained": False},
        "retained_checkpoint": None,
        "timing_seconds": {"optimizer_cumulative": update / 100.0},
    }


def _payload(
    *,
    phase: str,
    seeds: list[int],
    candidate_fraction: float = 0.25,
):
    if phase == "discovery":
        updates = [50, 100, 200, 300, 400, 1_204, 2_408, 3_612, 4_816]
    else:
        updates = [168, 200, 232, 1_204, 4_816]
    fractions = {
        50: 1.5,
        100: 1.2,
        168: 1.1,
        200: candidate_fraction,
        232: 0.20,
        300: 0.22,
        400: 0.20,
        1_204: 0.15,
        2_408: 0.10,
        3_612: 0.05,
        4_816: 0.0,
    }
    pools = [65_536, 1_048_576]
    trajectories = []
    for pool in pools:
        for seed_index, seed in enumerate(seeds):
            seed_offset = (seed_index - (len(seeds) - 1) / 2) * 0.0002
            checkpoints = [
                _checkpoint(
                    phase=phase,
                    update=update,
                    seed_offset=seed_offset,
                    degradation_fraction=(
                        fractions[update] + (seed_index - (len(seeds) - 1) / 2) * 0.005
                    ),
                )
                for update in updates
            ]
            trajectories.append(
                {
                    "trajectory_id": f"{phase}-{pool}-{seed}",
                    "init_seed": seed,
                    "pool_size": pool,
                    "initial_state_sha256": f"{seed:064x}"[-64:],
                    "pool_indices_sha256": f"{pool:064x}"[-64:],
                    "maximum_update": updates[-1],
                    "optimizer_steps_executed": updates[-1],
                    "checkpoints": checkpoints,
                }
            )
    configuration = {
        "pool_sizes": pools,
        "checkpoint_updates_predeclared": updates,
        "reference_update": 4_816,
        "replay_anchor_update": 1_204,
        "init_seeds": seeds,
    }
    if phase == "confirmation":
        configuration["locked_candidate_update"] = 200
    return {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "status": "ok",
        "phase": phase,
        "protocol_configuration": configuration,
        "decision_protocol": {"noninferiority_margins": _raw_margins()},
        "trajectories": trajectories,
    }


def _strict_discovery_payload():
    seeds = list(FROZEN_DISCOVERY_SEEDS)
    pools = [65_536, 1_048_576]
    updates = list(FROZEN_COARSE_GRID)
    trajectories = []
    replay_entries = []
    for pool in pools:
        for seed_index, seed in enumerate(seeds):
            seed_offset = (seed_index - 1) * 0.0002
            checkpoints = []
            for update in updates:
                if update == 4_816:
                    fraction = 0.0
                elif update == 3_612:
                    fraction = 0.05
                elif update == 2_408:
                    fraction = 0.10
                elif update >= 512:
                    fraction = 0.20
                else:
                    fraction = 1.20
                checkpoints.append(
                    _checkpoint(
                        phase="discovery",
                        update=update,
                        seed_offset=seed_offset,
                        degradation_fraction=fraction,
                    )
                )
            initial_hash = f"{seed:064x}"[-64:]
            pool_hash = f"{pool:064x}"[-64:]
            trajectories.append(
                {
                    "trajectory_id": f"strict-{pool}-{seed}",
                    "init_seed": seed,
                    "pool_size": pool,
                    "initial_state_sha256": initial_hash,
                    "pool_indices_sha256": pool_hash,
                    "maximum_update": 4_816,
                    "optimizer_steps_executed": 4_816,
                    "checkpoints": checkpoints,
                }
            )
            anchor = next(item for item in checkpoints if item["update"] == 1_204)
            replay_entries.append(
                {
                    "init_seed": seed,
                    "pool_size": pool,
                    "initial_state_sha256": initial_hash,
                    "pool_indices_sha256": pool_hash,
                    "anchor_present": True,
                    "model_state_sha256": anchor["model_state_sha256"],
                    "tune": anchor["tune"],
                }
            )
    return {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "status": "ok",
        "phase": "discovery",
        "protocol_document": {"file_sha256": FROZEN_PROTOCOL_SHA256},
        "protocol_configuration": {
            "pool_sizes": pools,
            "checkpoint_updates_predeclared": updates,
            "reference_update": 4_816,
            "replay_anchor_update": 1_204,
            "init_seeds": seeds,
        },
        "decision_protocol": {
            "noninferiority_margins": _raw_margins(),
            "discovery_rule": {"seeds": seeds},
            "confirmation_rule": {"fresh_seed_default": list(range(1_732, 1_744))},
        },
        "replay_audit": {
            "anchor_update": 1_204,
            "current_run_identity_exposed": True,
            "external_prior_result_checked": False,
            "entries": replay_entries,
        },
        "trajectories": trajectories,
    }


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _args(tmp_path: Path, *, confirmation: bool = False):
    discovery = tmp_path / "private-discovery.json"
    _write(discovery, _payload(phase="discovery", seeds=[1, 2, 3]))
    arguments = [
        "--discovery-results",
        str(discovery),
        "--output-dir",
        str(tmp_path / "public"),
        "--allow-nonprotocol-input",
    ]
    confirmation_path = None
    if confirmation:
        confirmation_path = tmp_path / "private-confirmation.json"
        _write(
            confirmation_path,
            _payload(phase="confirmation", seeds=list(range(10, 22))),
        )
        arguments.extend(["--confirmation-results", str(confirmation_path)])
    return parse_args(arguments), discovery, confirmation_path


def _csv_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_signed_degradation_obeys_direction_and_scale():
    assert signed_degradation("reconstruction_nmse", 0.4545, 0.45) == pytest.approx(0.01)
    assert signed_degradation("reconstruction_cosine", 0.755, 0.76) == pytest.approx(0.005)
    assert signed_degradation("sampled_pair_normalized_stress", 0.275, 0.27) == pytest.approx(0.005)
    assert signed_degradation("sample_conditional_trustworthiness", 0.960, 0.962) == pytest.approx(
        0.002
    )


def test_discovery_only_selects_first_sustained_candidate_and_is_path_free(tmp_path):
    args, _, _ = _args(tmp_path)
    summary = analyze(args)

    assert summary["protocol"] == PROTOCOL
    assert summary["discovery"]["candidate_update_by_pool"] == {
        "65536": 200,
        "1048576": 200,
    }
    assert summary["discovery"]["earliest_common_sustained_candidate_update"] == 200
    assert summary["confirmation"] is None
    assert summary["recommendation_gate"]["recommended_update"] is None
    output = args.output_dir
    for name in (
        "aggregate.csv",
        "paired-degradation.csv",
        "candidate-decisions.csv",
        "terminal-slope.csv",
        "replay-anchor.csv",
        "update-budget-curves.png",
        "update-budget-curves.pdf",
        "summary.json",
    ):
        assert (output / name).stat().st_size > 0
    public_text = "\n".join(
        path.read_text(errors="ignore")
        for path in output.iterdir()
        if path.suffix in {".json", ".csv"}
    )
    assert str(tmp_path) not in public_text
    assert "private-discovery.json" not in public_text


def test_sustained_rule_rejects_isolated_early_pass(tmp_path):
    args, discovery_path, _ = _args(tmp_path)
    payload = json.loads(discovery_path.read_text())
    for trajectory in payload["trajectories"]:
        # Update 100 passes, but update 200 fails; the first sustained window begins at 300.
        for checkpoint in trajectory["checkpoints"]:
            if checkpoint["update"] == 100:
                replacement = _checkpoint(
                    phase="discovery",
                    update=100,
                    seed_offset=0.0,
                    degradation_fraction=0.2,
                )
                checkpoint.update(replacement)
            elif checkpoint["update"] == 200:
                replacement = _checkpoint(
                    phase="discovery",
                    update=200,
                    seed_offset=0.0,
                    degradation_fraction=1.2,
                )
                checkpoint.update(replacement)
    _write(discovery_path, payload)
    summary = analyze(args)
    assert summary["discovery"]["earliest_common_sustained_candidate_update"] == 300


def test_terminal_report_uses_relative_reconstruction_and_absolute_geometry(tmp_path):
    args, _, _ = _args(tmp_path)
    summary = analyze(args)
    rows = _csv_rows(args.output_dir / "terminal-slope.csv")
    nmse = next(
        row
        for row in rows
        if row["pool_size"] == "65536"
        and row["from_update"] == "2408"
        and row["metric"] == "reconstruction_nmse"
    )
    trust = next(
        row
        for row in rows
        if row["pool_size"] == "65536"
        and row["from_update"] == "2408"
        and row["metric"] == "sample_conditional_trustworthiness"
    )
    decoder = next(
        row
        for row in rows
        if row["pool_size"] == "65536"
        and row["from_update"] == "2408"
        and row["metric"] == "sampled_local_decodability_nmse"
    )
    assert nmse["change_scale"] == "relative"
    assert trust["change_scale"] == "absolute"
    assert decoder["change_scale"] == "relative"
    assert float(nmse["signed_degradation_mean"]) < 0
    assert summary["terminal_slope"]["plateau_claimed"] is False


def test_confirmation_uses_fresh_paired_t_upper_bounds(tmp_path):
    args, _, _ = _args(tmp_path, confirmation=True)
    summary = analyze(args)
    assert summary["confirmation"]["fresh_seeds"] == list(range(10, 22))
    assert summary["confirmation"]["overall_all_metric_noninferiority_pass"] is True
    rows = _csv_rows(args.output_dir / "paired-degradation.csv")
    candidate = [row for row in rows if row["phase"] == "confirmation" and row["update"] == "200"]
    assert len(candidate) == 2 * len(METRICS)
    assert all(row["one_sided_confidence"] == "0.95" for row in candidate)
    assert all(row["confidence_bound_within_margin"] == "True" for row in candidate)


def test_confirmation_failure_is_reported_without_changing_candidate(tmp_path):
    args, _, confirmation_path = _args(tmp_path, confirmation=True)
    assert confirmation_path is not None
    payload = _payload(phase="confirmation", seeds=list(range(10, 22)), candidate_fraction=1.2)
    _write(confirmation_path, payload)
    summary = analyze(args)
    assert summary["discovery"]["earliest_common_sustained_candidate_update"] == 200
    assert summary["confirmation"]["overall_all_metric_noninferiority_pass"] is False
    assert all(summary["confirmation"]["pools"][index]["failed_metrics"] for index in (0, 1))


def test_margin_mismatch_is_rejected(tmp_path):
    args, discovery_path, _ = _args(tmp_path)
    payload = json.loads(discovery_path.read_text())
    payload["decision_protocol"]["noninferiority_margins"]["reconstruction_nmse"]["margin"] = 0.02
    _write(discovery_path, payload)
    with pytest.raises(ValueError, match="frozen reconstruction_nmse margin"):
        analyze(args)


def test_discovery_rejects_legacy_cohort_leakage(tmp_path):
    args, discovery_path, _ = _args(tmp_path)
    payload = json.loads(discovery_path.read_text())
    payload["trajectories"][0]["checkpoints"][0]["fixed_legacy_evaluation"] = {
        "reconstruction_nmse": 0.4,
        "reconstruction_cosine": 0.8,
    }
    _write(discovery_path, payload)
    with pytest.raises(ValueError, match="leaked"):
        analyze(args)


def test_confirmation_rejects_reused_seeds_and_unlocked_candidate(tmp_path):
    args, _, confirmation_path = _args(tmp_path, confirmation=True)
    assert confirmation_path is not None
    reused = _payload(phase="confirmation", seeds=[1, 2, 3])
    _write(confirmation_path, reused)
    with pytest.raises(ValueError, match="fresh"):
        analyze(args)

    unlocked = _payload(phase="confirmation", seeds=list(range(10, 22)))
    unlocked["protocol_configuration"]["locked_candidate_update"] = 300
    _write(confirmation_path, unlocked)
    with pytest.raises(ValueError, match="common sustained"):
        analyze(args)


def test_common_candidate_is_revalidated_in_every_pool(tmp_path):
    args, discovery_path, _ = _args(tmp_path)
    payload = json.loads(discovery_path.read_text())
    for trajectory in payload["trajectories"]:
        pool = trajectory["pool_size"]
        for checkpoint in trajectory["checkpoints"]:
            update = checkpoint["update"]
            fraction = None
            if pool == 65_536 and update in {100, 200, 300}:
                fraction = 0.2
            elif pool == 65_536 and update == 400:
                fraction = 1.2
            elif pool == 1_048_576 and update in {100, 200, 300}:
                fraction = 1.2
            elif pool == 1_048_576 and update == 400:
                fraction = 0.2
            if fraction is not None:
                checkpoint.update(
                    _checkpoint(
                        phase="discovery",
                        update=update,
                        seed_offset=0.0,
                        degradation_fraction=fraction,
                    )
                )
    _write(discovery_path, payload)
    summary = analyze(args)
    assert summary["discovery"]["candidate_update_by_pool"] == {
        "65536": 100,
        "1048576": 400,
    }
    # max(per-pool candidate)=400 is invalid for the small pool; 1204 is the
    # first update with a sustained pass in both pools.
    assert summary["discovery"]["earliest_common_sustained_candidate_update"] == 1_204


def test_public_mode_requires_the_exact_frozen_protocol_binding(tmp_path):
    args, _, _ = _args(tmp_path)
    args.allow_nonprotocol_input = False
    with pytest.raises(ValueError, match="frozen public protocol SHA-256"):
        analyze(args)


def test_strict_public_mode_validates_protocol_but_suppresses_premature_recommendation(
    tmp_path,
):
    discovery_path = tmp_path / "strict-discovery.json"
    _write(discovery_path, _strict_discovery_payload())
    args = parse_args(
        [
            "--discovery-results",
            str(discovery_path),
            "--output-dir",
            str(tmp_path / "strict-public"),
        ]
    )
    summary = analyze(args)
    assert summary["decision_protocol"]["strict_protocol_validation"] is True
    assert summary["discovery"]["earliest_common_sustained_candidate_update"] == 512
    assert summary["discovery"]["fine_grid"]["all_pools_complete"] is False
    assert summary["terminal_slope"]["all_pools_terminal_flatness_pass"] is True
    assert summary["external_replay_verification"]["verified"] is False
    assert summary["recommendation_gate"]["recommended_update"] is None


def test_external_replay_verification_is_hash_bound_and_public_safe(tmp_path):
    discovery_path = tmp_path / "strict-discovery.json"
    _write(discovery_path, _strict_discovery_payload())
    discovery_sha256 = hashlib.sha256(discovery_path.read_bytes()).hexdigest()
    replay_path = tmp_path / "replay-verification.json"
    _write(
        replay_path,
        {
            "protocol": "deep1b-ae-update-budget-replay-verification-v1",
            "status": "ok",
            "discovery_results_sha256": discovery_sha256,
            "protocol_document_sha256": FROZEN_PROTOCOL_SHA256,
            "external_prior_result_checked": True,
            "all_required_identities_match": True,
            "replay_verified": True,
        },
    )
    args = parse_args(
        [
            "--discovery-results",
            str(discovery_path),
            "--replay-verification",
            str(replay_path),
            "--output-dir",
            str(tmp_path / "public-with-replay"),
        ]
    )
    summary = analyze(args)
    assert summary["external_replay_verification"]["verified"] is True
    assert str(tmp_path) not in json.dumps(summary)


def test_analyzer_consumes_the_real_cpu_harness_schema(tmp_path):
    from experiments.compare_ae_update_budgets import _test_args, execute

    rng = np.random.default_rng(8)
    latent = rng.normal(size=(80, 2)).astype(np.float32)
    projection = rng.normal(size=(2, 6)).astype(np.float32)
    data = np.ascontiguousarray(latent @ projection, dtype=np.float32)
    data_path = tmp_path / "source.npy"
    np.save(data_path, data)
    raw_dir = tmp_path / "raw"
    harness_args = _test_args(
        data=data_path,
        output_dir=raw_dir,
        pool_sizes=(16, 32),
        checkpoint_updates=(0, 1, 2, 4, 8),
        reference_update=8,
        legacy_updates=(),
        geometry_updates=(0, 1, 2, 4, 8),
        retain_updates=(),
    )
    execute(harness_args)

    analysis_args = parse_args(
        [
            "--discovery-results",
            str(raw_dir / "update-budget-results.json"),
            "--output-dir",
            str(tmp_path / "public-real-schema"),
            "--allow-nonprotocol-input",
        ]
    )
    summary = analyze(analysis_args)
    assert summary["discovery"]["long_reference_update"] == 8
    assert summary["discovery"]["pool_sizes"] == [16, 32]
    assert summary["terminal_slope"]["complete_by_pool"] == {"16": False, "32": False}
