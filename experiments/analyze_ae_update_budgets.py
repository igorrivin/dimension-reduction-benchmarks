#!/usr/bin/env python3
"""Analyze Deep1B autoencoder optimizer-update learning curves.

The analyzer is deliberately independent of the GPU harness.  It consumes the
discovery JSON and, optionally, a locked fresh-seed confirmation JSON emitted by
``compare_ae_update_budgets.py``.  It recomputes every paired degradation from
the raw checkpoint metrics instead of trusting decisions embedded by the
runner.

Discovery selects the earliest update whose point estimates satisfy all frozen
practical margins, for which no replicate exceeds twice its metric margin, and
for which the next two evaluated checkpoints also satisfy those conditions.
Confirmation is a one-sided paired non-inferiority analysis against the locked
long-budget reference.  The fixed 1,204-update replay anchor and the post-1,204
terminal changes are reported separately; neither is described automatically
as a plateau.

Outputs are a path-free JSON summary, compact CSV tables, and matching PNG/PDF
figures.  Checkpoint paths, source paths, and layout paths are never copied from
the raw inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

TRAINING_PROTOCOL = "deep1b-ae-update-budget-sensitivity-v1"
PROTOCOL = "deep1b-ae-update-budget-public-analysis-v1"
REPLAY_VERIFICATION_PROTOCOL = "deep1b-ae-update-budget-replay-verification-v1"
SCHEMA_VERSION = 1
DEFAULT_REPLAY_ANCHOR_UPDATE = 1_204
FROZEN_PROTOCOL_SHA256 = "41db717aade5062028efd9a7e3d8d57ca238a1f06c781aaf54b06b87deea89a0"
FROZEN_POOL_SIZES = (65_536, 1_048_576)
FROZEN_DISCOVERY_SEEDS = (1_729, 1_730, 1_731)
FROZEN_CONFIRMATION_SEEDS = tuple(range(1_732, 1_744))
FROZEN_COARSE_GRID = (
    0,
    8,
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1_024,
    1_204,
    1_808,
    2_408,
    3_612,
    4_816,
)
TERMINAL_UPDATES_BY_REFERENCE = {
    4_816: (2_408, 3_612, 4_816),
    9_632: (4_816, 7_224, 9_632),
}
SLOPE_UPDATES_BY_REFERENCE = {
    4_816: (1_204, 1_808, 2_408, 3_612, 4_816),
    9_632: (1_204, 1_808, 2_408, 3_612, 4_816, 7_224, 9_632),
}
ONE_SIDED_CONFIDENCE = 0.95
_MAX_JSON_BYTES = 128 * 2**20


@dataclass(frozen=True)
class MetricSpec:
    """Frozen direction, scale, and practical margin for one response."""

    metric: str
    display_name: str
    group: str
    better: str
    margin_scale: str
    margin: float


METRICS = {
    spec.metric: spec
    for spec in (
        MetricSpec(
            "reconstruction_nmse",
            "Reconstruction NMSE",
            "reconstruction",
            "lower",
            "relative",
            0.01,
        ),
        MetricSpec(
            "reconstruction_cosine",
            "Reconstruction cosine",
            "reconstruction",
            "higher",
            "absolute",
            0.005,
        ),
        MetricSpec(
            "sample_conditional_trustworthiness",
            "Trustworthiness",
            "geometry",
            "higher",
            "absolute",
            0.002,
        ),
        MetricSpec(
            "sample_conditional_continuity",
            "Continuity",
            "geometry",
            "higher",
            "absolute",
            0.002,
        ),
        MetricSpec(
            "sample_conditional_neighbor_overlap",
            "Neighbor overlap at 16",
            "geometry",
            "higher",
            "absolute",
            0.005,
        ),
        MetricSpec(
            "sampled_pair_distance_spearman",
            "Pair-distance Spearman",
            "geometry",
            "higher",
            "absolute",
            0.010,
        ),
        MetricSpec(
            "sampled_pair_normalized_stress",
            "Normalized stress",
            "geometry",
            "lower",
            "absolute",
            0.005,
        ),
        MetricSpec(
            "sampled_local_decodability_nmse",
            "Local-decoder NMSE",
            "geometry",
            "lower",
            "relative",
            0.01,
        ),
    )
}


@dataclass
class ParsedPhase:
    """Validated, path-free view of one raw harness result."""

    phase: str
    input_sha256: str
    design_identity_sha256: str
    pool_sizes: list[int]
    seeds: list[int]
    checkpoint_updates: list[int]
    reference_update: int
    replay_anchor_update: int | None
    locked_candidate_update: int | None
    records: dict[tuple[int, int, int], dict[str, float]]
    optimizer_seconds: dict[tuple[int, int, int], float]
    replay_audit: dict[str, Any] | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, block_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
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


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _integer_list(value: Any, *, name: str, positive: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    check = _positive_integer if positive else _nonnegative_integer
    result = [check(item, name=f"{name} item") for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be unique and strictly increasing")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fieldnames = list(rows[0]) if rows else ["status"]
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            if rows:
                writer.writerows(rows)
            else:
                writer.writerow({"status": "not_available"})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def signed_degradation(
    metric: str,
    candidate: float,
    reference: float,
    *,
    scale: str | None = None,
) -> float:
    """Return candidate-versus-reference degradation with positive meaning worse."""

    spec = METRICS[metric]
    active_scale = spec.margin_scale if scale is None else scale
    if active_scale == "relative":
        if reference == 0:
            raise ValueError(f"cannot compute relative change from zero {metric} reference")
        raw = candidate / reference - 1.0
    elif active_scale == "absolute":
        raw = candidate - reference
    else:
        raise ValueError(f"unknown comparison scale {active_scale!r}")
    return float(raw if spec.better == "lower" else -raw)


def _validate_margins(payload: Mapping[str, Any]) -> None:
    decision = _object(payload, "decision_protocol", context="result")
    raw = _object(decision, "noninferiority_margins", context="decision_protocol")
    if set(raw) != set(METRICS):
        raise ValueError("input non-inferiority metric set differs from the frozen analyzer set")
    for metric, spec in METRICS.items():
        rule = _object(raw, metric, context="decision_protocol.noninferiority_margins")
        margin = _finite(rule.get("margin"), name=f"{metric} margin")
        if not math.isclose(margin, spec.margin, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"input margin differs from frozen {metric} margin")
        if rule.get("scale") != spec.margin_scale or rule.get("better") != spec.better:
            raise ValueError(f"input direction or scale differs for {metric}")


def _design_identity_sha256(payload: Mapping[str, Any]) -> str:
    """Hash phase-invariant design fields without retaining any local paths."""

    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    legacy = (
        split.get("fixed_legacy_evaluation")
        if isinstance(split.get("fixed_legacy_evaluation"), dict)
        else {}
    )
    tune = split.get("tune") if isinstance(split.get("tune"), dict) else {}
    training = split.get("training_pool") if isinstance(split.get("training_pool"), dict) else {}
    configuration = (
        payload.get("protocol_configuration")
        if isinstance(payload.get("protocol_configuration"), dict)
        else {}
    )
    protocol_document = (
        payload.get("protocol_document")
        if isinstance(payload.get("protocol_document"), dict)
        else {}
    )
    identity = {
        "source": {
            key: source.get(key)
            for key in (
                "file_sha256",
                "rows",
                "dimension",
                "dtype",
                "universe_rows",
                "input_transformation",
            )
        },
        "split": {
            "legacy_indices_sha256": legacy.get("indices_sha256"),
            "tune_indices_sha256": tune.get("indices_sha256"),
            "training_pool_seed": training.get("pool_seed"),
            "training_prefix_hashes": training.get("prefix_indices_sha256"),
        },
        "model_and_optimizer": {
            key: configuration.get(key)
            for key in (
                "target_dimension",
                "hidden_dimensions",
                "architecture",
                "parameter_count",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "loss",
                "batch_size",
                "pool_sizes",
                "precision",
            )
        },
        "protocol_document_sha256": protocol_document.get("file_sha256"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _configuration_reference(configuration: Mapping[str, Any]) -> int:
    for key in ("reference_update", "locked_reference_update", "maximum_updates"):
        if configuration.get(key) is not None:
            return _positive_integer(configuration[key], name=f"configuration.{key}")
    updates = _integer_list(
        configuration.get("checkpoint_updates_predeclared"),
        name="configuration.checkpoint_updates_predeclared",
    )
    return updates[-1]


def _checkpoint_metrics(checkpoint: Mapping[str, Any], phase: str) -> dict[str, float]:
    if phase == "discovery":
        reconstruction = checkpoint.get("tune")
    else:
        reconstruction = checkpoint.get("fixed_legacy_evaluation")
    geometry = checkpoint.get("geometry")
    if reconstruction is None and geometry is None:
        return {}
    result: dict[str, float] = {}
    if reconstruction is not None:
        if not isinstance(reconstruction, dict):
            raise ValueError("checkpoint reconstruction record must be an object or null")
        for metric in ("reconstruction_nmse", "reconstruction_cosine"):
            result[metric] = _finite(
                reconstruction.get(metric), name=f"checkpoint {phase} {metric}"
            )
    if geometry is not None:
        if not isinstance(geometry, dict):
            raise ValueError("checkpoint geometry record must be an object or null")
        for metric, spec in METRICS.items():
            if spec.group == "geometry":
                result[metric] = _finite(geometry.get(metric), name=f"checkpoint {metric}")
    return result


def _parse_phase(
    payload: dict[str, Any],
    input_sha256: str,
    expected_phase: str,
    *,
    strict_protocol: bool,
) -> ParsedPhase:
    if payload.get("protocol") != TRAINING_PROTOCOL or payload.get("status") != "ok":
        raise ValueError("input is not a successful AE update-budget result")
    phase = payload.get("phase")
    if phase != expected_phase:
        raise ValueError(f"expected {expected_phase!r} input, found {phase!r}")
    _validate_margins(payload)
    configuration = _object(payload, "protocol_configuration", context="result")
    updates = _integer_list(
        configuration.get("checkpoint_updates_predeclared"),
        name="configuration.checkpoint_updates_predeclared",
    )
    pool_sizes = _integer_list(
        configuration.get("pool_sizes"), name="configuration.pool_sizes", positive=True
    )
    seeds = _integer_list(configuration.get("init_seeds"), name="configuration.init_seeds")
    decision = _object(payload, "decision_protocol", context="result")
    rule_name = "discovery_rule" if phase == "discovery" else "confirmation_rule"
    phase_rule = decision.get(rule_name)
    if strict_protocol and isinstance(phase_rule, dict):
        seed_field = "seeds" if phase == "discovery" else "fresh_seed_default"
        if phase_rule.get(seed_field) is not None:
            declared_seeds = _integer_list(
                phase_rule[seed_field], name=f"decision_protocol.{rule_name}.{seed_field}"
            )
            if seeds != declared_seeds:
                raise ValueError(f"{phase} seeds differ from the frozen decision protocol")
    reference_update = _configuration_reference(configuration)
    if reference_update not in updates:
        raise ValueError("configured reference update is absent from the checkpoint grid")
    anchor_raw = configuration.get("replay_anchor_update", DEFAULT_REPLAY_ANCHOR_UPDATE)
    replay_anchor = (
        _nonnegative_integer(anchor_raw, name="configuration.replay_anchor_update")
        if anchor_raw is not None
        else None
    )
    locked_raw = configuration.get("locked_candidate_update")
    if locked_raw is None:
        confirmation_rule = decision.get("confirmation_rule")
        if isinstance(confirmation_rule, dict):
            locked_raw = confirmation_rule.get("locked_candidate_update")
    locked_candidate = (
        _nonnegative_integer(locked_raw, name="configuration.locked_candidate_update")
        if locked_raw is not None
        else None
    )
    if phase == "confirmation" and locked_candidate is None:
        raise ValueError("confirmation input must identify its locked candidate update")
    confirmation_rule = decision.get("confirmation_rule")
    locked_step = None
    locked_reference = None
    if isinstance(confirmation_rule, dict):
        if confirmation_rule.get("locked_step_updates") is not None:
            locked_step = _positive_integer(
                confirmation_rule["locked_step_updates"],
                name="decision_protocol.confirmation_rule.locked_step_updates",
            )
        if confirmation_rule.get("locked_reference_update") is not None:
            locked_reference = _positive_integer(
                confirmation_rule["locked_reference_update"],
                name="decision_protocol.confirmation_rule.locked_reference_update",
            )

    protocol_document = payload.get("protocol_document")
    protocol_sha256 = None
    if isinstance(protocol_document, dict):
        protocol_sha256 = _sha256(
            protocol_document.get("file_sha256"), name="protocol_document.file_sha256"
        )
    if strict_protocol:
        if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
            raise ValueError("result is not bound to the frozen public protocol SHA-256")
        if pool_sizes != list(FROZEN_POOL_SIZES):
            raise ValueError("pool sizes differ from the frozen public protocol")
        expected_seeds = (
            list(FROZEN_DISCOVERY_SEEDS)
            if phase == "discovery"
            else list(FROZEN_CONFIRMATION_SEEDS)
        )
        if seeds != expected_seeds:
            raise ValueError(f"{phase} seeds differ from the frozen public protocol")
        if replay_anchor != DEFAULT_REPLAY_ANCHOR_UPDATE:
            raise ValueError("replay anchor differs from the frozen public protocol")
        if phase == "discovery":
            if reference_update not in TERMINAL_UPDATES_BY_REFERENCE:
                raise ValueError("discovery reference is not a frozen terminal update")
            required_grid = set(FROZEN_COARSE_GRID)
            if reference_update == 9_632:
                required_grid.update((7_224, 9_632))
            if not required_grid.issubset(updates):
                raise ValueError("discovery grid omits frozen coarse or terminal checkpoints")
        else:
            if locked_step is None or locked_reference is None:
                raise ValueError("confirmation input omits locked step or reference")
            if locked_reference != reference_update:
                raise ValueError("locked confirmation reference differs from configured reference")
            expected_grid = sorted(
                {
                    max(0, int(locked_candidate) - locked_step),
                    int(locked_candidate),
                    int(locked_candidate) + locked_step,
                    DEFAULT_REPLAY_ANCHOR_UPDATE,
                    locked_reference,
                }
            )
            if updates != expected_grid:
                raise ValueError("confirmation grid differs from the frozen locked grid")

    raw_trajectories = payload.get("trajectories")
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise ValueError("result.trajectories must be a non-empty list")
    records: dict[tuple[int, int, int], dict[str, float]] = {}
    optimizer_seconds: dict[tuple[int, int, int], float] = {}
    trajectory_ids: set[tuple[int, int]] = set()
    initial_hashes: dict[int, set[str]] = {seed: set() for seed in seeds}
    trajectory_identity: dict[tuple[int, int], dict[str, Any]] = {}
    for trajectory in raw_trajectories:
        if not isinstance(trajectory, dict):
            raise ValueError("every trajectory must be an object")
        seed = _nonnegative_integer(trajectory.get("init_seed"), name="trajectory init_seed")
        pool = _positive_integer(trajectory.get("pool_size"), name="trajectory pool_size")
        identity = (pool, seed)
        if identity in trajectory_ids or pool not in pool_sizes or seed not in seeds:
            raise ValueError(f"duplicate or unknown trajectory {identity}")
        trajectory_ids.add(identity)
        initial_hash = _sha256(
            trajectory.get("initial_state_sha256"), name=f"trajectory {identity} initial state"
        )
        pool_hash = _sha256(
            trajectory.get("pool_indices_sha256"), name=f"trajectory {identity} pool indices"
        )
        initial_hashes[seed].add(initial_hash)
        raw_checkpoints = trajectory.get("checkpoints")
        if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
            raise ValueError(f"trajectory {identity} has no checkpoints")
        observed_updates: list[int] = []
        checkpoint_states: dict[int, str] = {}
        for checkpoint in raw_checkpoints:
            if not isinstance(checkpoint, dict):
                raise ValueError("every checkpoint must be an object")
            update = _nonnegative_integer(checkpoint.get("update"), name="checkpoint update")
            observed_updates.append(update)
            if update not in updates:
                raise ValueError(f"trajectory {identity} contains undeclared update {update}")
            checkpoint_states[update] = _sha256(
                checkpoint.get("model_state_sha256"),
                name=f"trajectory {identity} update {update} model state",
            )
            if phase == "discovery":
                if checkpoint.get("fixed_legacy_evaluation") is not None:
                    raise ValueError("discovery input leaked the fixed legacy evaluation cohort")
            elif checkpoint.get("tune") is not None:
                raise ValueError("confirmation input contains tune-cohort checkpoint metrics")
            metrics = _checkpoint_metrics(checkpoint, phase)
            if metrics:
                role = checkpoint.get("geometry_cohort_role")
                role_text = role.lower() if isinstance(role, str) else ""
                if (
                    checkpoint.get("geometry") is not None
                    and phase == "discovery"
                    and "tune" not in role_text
                ):
                    raise ValueError("discovery geometry must use the tune cohort")
                if (
                    checkpoint.get("geometry") is not None
                    and phase == "confirmation"
                    and not any(token in role_text for token in ("legacy", "fixed"))
                ):
                    raise ValueError("confirmation geometry must use the fixed legacy cohort")
                records[(pool, seed, update)] = metrics
            timing = checkpoint.get("timing_seconds")
            if isinstance(timing, dict) and timing.get("optimizer_cumulative") is not None:
                optimizer_seconds[(pool, seed, update)] = _finite(
                    timing["optimizer_cumulative"], name="optimizer cumulative seconds"
                )
        if observed_updates != sorted(set(observed_updates)):
            raise ValueError(f"trajectory {identity} checkpoints are not strictly increasing")
        if observed_updates != updates:
            raise ValueError(f"trajectory {identity} does not cover the full declared grid")
        if trajectory.get("maximum_update") != updates[-1]:
            raise ValueError(f"trajectory {identity} maximum update is inconsistent")
        if trajectory.get("optimizer_steps_executed") != updates[-1]:
            raise ValueError(f"trajectory {identity} optimizer-step count is inconsistent")
        trajectory_identity[identity] = {
            "initial_state_sha256": initial_hash,
            "pool_indices_sha256": pool_hash,
            "checkpoint_states": checkpoint_states,
        }

    expected = {(pool, seed) for pool in pool_sizes for seed in seeds}
    if trajectory_ids != expected:
        raise ValueError("input is not a complete pool-by-seed trajectory block")
    for seed, hashes in initial_hashes.items():
        if len(hashes) > 1:
            raise ValueError(f"pool trajectories for seed {seed} do not share initialization")
    replay_audit = None
    raw_replay = payload.get("replay_audit")
    if isinstance(raw_replay, dict):
        entries = raw_replay.get("entries")
        if not isinstance(entries, list):
            raise ValueError("replay_audit.entries must be a list")
        if any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("every replay_audit entry must be an object")
        if len(entries) != len(pool_sizes) * len(seeds):
            raise ValueError("replay audit does not cover every seed-by-pool trajectory")
        audit_anchor = _nonnegative_integer(
            raw_replay.get("anchor_update"), name="replay_audit.anchor_update"
        )
        if replay_anchor is not None and audit_anchor != replay_anchor:
            raise ValueError("replay audit and configuration identify different anchors")
        seen_replay: set[tuple[int, int]] = set()
        for entry in entries:
            replay_seed = _nonnegative_integer(
                entry.get("init_seed"), name="replay entry init_seed"
            )
            replay_pool = _positive_integer(entry.get("pool_size"), name="replay entry pool_size")
            identity = (replay_pool, replay_seed)
            if identity in seen_replay or identity not in trajectory_identity:
                raise ValueError(f"duplicate or unknown replay entry {identity}")
            seen_replay.add(identity)
            trajectory_record = trajectory_identity[identity]
            if (
                _sha256(entry.get("initial_state_sha256"), name="replay entry initial state")
                != trajectory_record["initial_state_sha256"]
            ):
                raise ValueError(f"replay entry {identity} initial state disagrees")
            if (
                _sha256(entry.get("pool_indices_sha256"), name="replay entry pool indices")
                != trajectory_record["pool_indices_sha256"]
            ):
                raise ValueError(f"replay entry {identity} pool indices disagree")
            anchor_present = audit_anchor in trajectory_record["checkpoint_states"]
            if entry.get("anchor_present") is not anchor_present:
                raise ValueError(f"replay entry {identity} anchor-presence flag disagrees")
            if (
                anchor_present
                and _sha256(entry.get("model_state_sha256"), name="replay entry model state")
                != trajectory_record["checkpoint_states"][audit_anchor]
            ):
                raise ValueError(f"replay entry {identity} model state disagrees")
            if anchor_present and phase == "discovery":
                replay_tune = _object(entry, "tune", context=f"replay entry {identity}")
                checkpoint_metrics = records[(replay_pool, replay_seed, audit_anchor)]
                for metric in ("reconstruction_nmse", "reconstruction_cosine"):
                    if (
                        _finite(replay_tune.get(metric), name=f"replay entry {identity} {metric}")
                        != checkpoint_metrics[metric]
                    ):
                        raise ValueError(f"replay entry {identity} tune metric disagrees")
        replay_audit = {
            "anchor_update": audit_anchor,
            "entry_count": len(entries),
            "all_entries_report_anchor_present": bool(entries)
            and all(entry.get("anchor_present") is True for entry in entries),
            "current_run_identity_exposed": raw_replay.get("current_run_identity_exposed") is True,
            "external_prior_result_checked": raw_replay.get("external_prior_result_checked")
            is True,
        }
    if strict_protocol and replay_audit is None:
        raise ValueError("frozen public analysis requires the harness replay audit")
    return ParsedPhase(
        phase=phase,
        input_sha256=input_sha256,
        design_identity_sha256=_design_identity_sha256(payload),
        pool_sizes=pool_sizes,
        seeds=seeds,
        checkpoint_updates=updates,
        reference_update=reference_update,
        replay_anchor_update=replay_anchor,
        locked_candidate_update=locked_candidate,
        records=records,
        optimizer_seconds=optimizer_seconds,
        replay_audit=replay_audit,
    )


def _descriptive(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else None,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _upper_one_sided_t(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    if sample_sd == 0:
        return float(mean)
    critical = float(student_t.ppf(ONE_SIDED_CONFIDENCE, len(values) - 1))
    return float(mean + critical * sample_sd / math.sqrt(len(values)))


def _complete_metric_updates(phase: ParsedPhase, pool: int) -> list[int]:
    return [
        update
        for update in phase.checkpoint_updates
        if all(
            (pool, seed, update) in phase.records
            and set(phase.records[(pool, seed, update)]) == set(METRICS)
            for seed in phase.seeds
        )
    ]


def _aggregate_rows(phase: ParsedPhase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pool in phase.pool_sizes:
        for update in phase.checkpoint_updates:
            for metric, spec in METRICS.items():
                if not all(
                    (pool, seed, update) in phase.records
                    and metric in phase.records[(pool, seed, update)]
                    for seed in phase.seeds
                ):
                    continue
                values = [phase.records[(pool, seed, update)][metric] for seed in phase.seeds]
                summary = _descriptive(values)
                rows.append(
                    {
                        "phase": phase.phase,
                        "pool_size": pool,
                        "update": update,
                        "metric": metric,
                        "metric_group": spec.group,
                        "display_name": spec.display_name,
                        "better": spec.better,
                        **summary,
                    }
                )
    return rows


def _degradation_rows(
    phase: ParsedPhase,
    *,
    include_confidence_bounds: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pool in phase.pool_sizes:
        for metric in METRICS:
            if not all(
                (pool, seed, phase.reference_update) in phase.records
                and metric in phase.records[(pool, seed, phase.reference_update)]
                for seed in phase.seeds
            ):
                raise ValueError(f"pool {pool} lacks reference metric {metric}")
        for update in phase.checkpoint_updates:
            for metric, spec in METRICS.items():
                if not all(
                    (pool, seed, update) in phase.records
                    and metric in phase.records[(pool, seed, update)]
                    for seed in phase.seeds
                ):
                    continue
                values = [
                    signed_degradation(
                        metric,
                        phase.records[(pool, seed, update)][metric],
                        phase.records[(pool, seed, phase.reference_update)][metric],
                    )
                    for seed in phase.seeds
                ]
                summary = _descriptive(values)
                upper = _upper_one_sided_t(values) if include_confidence_bounds else None
                rows.append(
                    {
                        "phase": phase.phase,
                        "pool_size": pool,
                        "update": update,
                        "reference_update": phase.reference_update,
                        "metric": metric,
                        "metric_group": spec.group,
                        "comparison_scale": spec.margin_scale,
                        "positive_means_worse": True,
                        "margin": spec.margin,
                        "n": summary["n"],
                        "mean_signed_degradation": summary["mean"],
                        "sample_sd": summary["sample_sd"],
                        "maximum_seed_degradation": summary["maximum"],
                        "mean_within_margin": bool(float(summary["mean"]) < spec.margin),
                        "no_seed_exceeds_twice_margin": bool(
                            float(summary["maximum"]) < 2.0 * spec.margin
                        ),
                        "one_sided_confidence": (
                            ONE_SIDED_CONFIDENCE if include_confidence_bounds else None
                        ),
                        "one_sided_upper_bound": upper,
                        "confidence_bound_within_margin": (
                            bool(upper < spec.margin) if upper is not None else None
                        ),
                    }
                )
    return rows


def _discovery_decisions(
    phase: ParsedPhase, degradation_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, int | None]]:
    indexed = {
        (int(row["pool_size"]), int(row["update"]), str(row["metric"])): row
        for row in degradation_rows
        if row["phase"] == "discovery"
    }
    decisions: list[dict[str, Any]] = []
    selected: dict[int, int | None] = {}
    for pool in phase.pool_sizes:
        updates = _complete_metric_updates(phase, pool)
        point_pass: dict[int, bool] = {}
        for update in updates:
            point_pass[update] = all(
                bool(indexed[(pool, update, metric)]["mean_within_margin"])
                and bool(indexed[(pool, update, metric)]["no_seed_exceeds_twice_margin"])
                for metric in METRICS
            )
        candidate = None
        for index, update in enumerate(updates):
            sustained_updates = updates[index : index + 3]
            sustained = (
                update > 0
                and update < phase.reference_update
                and len(sustained_updates) == 3
                and all(point_pass[item] for item in sustained_updates)
            )
            timing_values = [
                phase.optimizer_seconds[(pool, seed, update)]
                for seed in phase.seeds
                if (pool, seed, update) in phase.optimizer_seconds
            ]
            reference_timing = [
                phase.optimizer_seconds[(pool, seed, phase.reference_update)]
                for seed in phase.seeds
                if (pool, seed, phase.reference_update) in phase.optimizer_seconds
            ]
            mean_timing = statistics.fmean(timing_values) if timing_values else None
            mean_reference_timing = statistics.fmean(reference_timing) if reference_timing else None
            decisions.append(
                {
                    "pool_size": pool,
                    "update": update,
                    "reference_update": phase.reference_update,
                    "all_metric_point_pass": point_pass[update],
                    "sustained_window_updates": "|".join(map(str, sustained_updates)),
                    "three_checkpoint_sustained_pass": sustained,
                    "selected_discovery_candidate": False,
                    "update_reduction_fraction": 1.0 - update / phase.reference_update,
                    "idealized_optimizer_step_speedup": (
                        phase.reference_update / update if update > 0 else None
                    ),
                    "mean_optimizer_cumulative_seconds": mean_timing,
                    "mean_reference_optimizer_seconds": mean_reference_timing,
                    "measured_optimizer_speedup": (
                        mean_reference_timing / mean_timing
                        if mean_reference_timing is not None
                        and mean_timing is not None
                        and mean_timing > 0
                        else None
                    ),
                }
            )
            if candidate is None and sustained:
                candidate = update
        if candidate is not None:
            for row in decisions:
                if row["pool_size"] == pool and row["update"] == candidate:
                    row["selected_discovery_candidate"] = True
                    break
        selected[pool] = candidate
    return decisions, selected


def _common_sustained_candidate(
    phase: ParsedPhase, decisions: Sequence[Mapping[str, Any]]
) -> int | None:
    """Return the earliest update sustained for every pool, never max-per-pool."""

    sustained = {
        (int(row["pool_size"]), int(row["update"])): bool(row["three_checkpoint_sustained_pass"])
        for row in decisions
    }
    common_updates = set(phase.checkpoint_updates)
    for pool in phase.pool_sizes:
        common_updates.intersection_update(
            update for update in phase.checkpoint_updates if sustained.get((pool, update), False)
        )
    return min(common_updates) if common_updates else None


def _fine_grid_status(phase: ParsedPhase, decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check whether every coarse last-fail/first-pass bracket was filled."""

    point_pass = {
        (int(row["pool_size"]), int(row["update"])): bool(row["all_metric_point_pass"])
        for row in decisions
    }
    initial_coarse = set(FROZEN_COARSE_GRID)
    if phase.reference_update == 9_632:
        initial_coarse.update((7_224, 9_632))
    coarse = [update for update in phase.checkpoint_updates if update in initial_coarse]
    by_pool = {}
    for pool in phase.pool_sizes:
        coarse_sustained = None
        for index, update in enumerate(coarse):
            window = coarse[index : index + 3]
            if (
                update > 0
                and len(window) == 3
                and all(point_pass.get((pool, item), False) for item in window)
            ):
                coarse_sustained = update
                break
        if coarse_sustained is None:
            by_pool[str(pool)] = {
                "complete": False,
                "reason": "no sustained coarse pass",
                "lower_failing_update": None,
                "upper_sustained_update": None,
                "fine_step_updates": None,
                "missing_updates": [],
            }
            continue
        earlier_failures = [
            update
            for update in coarse
            if update < coarse_sustained and not point_pass.get((pool, update), False)
        ]
        lower = max(earlier_failures) if earlier_failures else 0
        step = 16 if coarse_sustained <= 256 else 32
        required = list(range(lower + step, coarse_sustained, step))
        missing = [update for update in required if update not in phase.checkpoint_updates]
        by_pool[str(pool)] = {
            "complete": not missing,
            "reason": None if not missing else "last-fail/first-pass bracket is not filled",
            "lower_failing_update": lower,
            "upper_sustained_update": coarse_sustained,
            "fine_step_updates": step,
            "required_interior_updates": required,
            "missing_updates": missing,
        }
    return {
        "by_pool": by_pool,
        "all_pools_complete": all(item["complete"] for item in by_pool.values()),
    }


def _terminal_rows(phase: ParsedPhase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completeness: dict[str, bool] = {}
    terminal_updates = TERMINAL_UPDATES_BY_REFERENCE.get(phase.reference_update)
    if terminal_updates is None:
        return rows, {
            "requested_updates": [],
            "complete_by_pool": {str(pool): False for pool in phase.pool_sizes},
            "terminal_flatness_pass_by_pool": {str(pool): False for pool in phase.pool_sizes},
            "all_pools_terminal_flatness_pass": False,
            "extension_required": False,
            "plateau_claimed": False,
            "interpretation": "No frozen terminal schedule exists for this reference.",
        }
    flatness_by_pool: dict[str, bool] = {}
    slope_updates = SLOPE_UPDATES_BY_REFERENCE[phase.reference_update]
    gate_intervals = set(zip(terminal_updates, terminal_updates[1:]))
    for pool in phase.pool_sizes:
        available = set(_complete_metric_updates(phase, pool))
        complete = set(terminal_updates).issubset(available)
        completeness[str(pool)] = complete
        if not complete:
            flatness_by_pool[str(pool)] = False
            continue
        pool_passes = []
        for start, stop in zip(slope_updates, slope_updates[1:]):
            if start not in available or stop not in available:
                continue
            flatness_gate_interval = (start, stop) in gate_intervals
            for metric, spec in METRICS.items():
                scale = spec.margin_scale
                raw_changes = []
                signed_changes = []
                for seed in phase.seeds:
                    before = phase.records[(pool, seed, start)][metric]
                    after = phase.records[(pool, seed, stop)][metric]
                    if scale == "relative":
                        if before == 0:
                            raise ValueError(f"zero {metric} at terminal update {start}")
                        raw = after / before - 1.0
                    else:
                        raw = after - before
                    raw_changes.append(float(raw))
                    signed_changes.append(float(raw if spec.better == "lower" else -raw))
                raw_summary = _descriptive(raw_changes)
                signed_summary = _descriptive(signed_changes)
                mean_absolute_change = float(statistics.fmean(abs(value) for value in raw_changes))
                flatness_threshold = 0.0025 if metric == "reconstruction_nmse" else spec.margin
                flatness_pass = mean_absolute_change <= flatness_threshold
                if flatness_gate_interval:
                    pool_passes.append(flatness_pass)
                rows.append(
                    {
                        "pool_size": pool,
                        "from_update": start,
                        "to_update": stop,
                        "metric": metric,
                        "metric_group": spec.group,
                        "change_scale": scale,
                        "raw_change_mean": raw_summary["mean"],
                        "raw_change_sample_sd": raw_summary["sample_sd"],
                        "mean_absolute_raw_change": mean_absolute_change,
                        "terminal_flatness_threshold": flatness_threshold,
                        "terminal_flatness_gate_interval": flatness_gate_interval,
                        "terminal_flatness_pass": (
                            flatness_pass if flatness_gate_interval else None
                        ),
                        "signed_degradation_mean": signed_summary["mean"],
                        "signed_degradation_sample_sd": signed_summary["sample_sd"],
                        "n": signed_summary["n"],
                        "positive_signed_change_means_worse": True,
                    }
                )
        flatness_by_pool[str(pool)] = all(pool_passes)
    overall_flat = all(flatness_by_pool.values()) and len(flatness_by_pool) == len(phase.pool_sizes)
    return rows, {
        "requested_updates": list(terminal_updates),
        "reported_post_anchor_slope_updates": list(slope_updates),
        "complete_by_pool": completeness,
        "terminal_flatness_pass_by_pool": flatness_by_pool,
        "all_pools_terminal_flatness_pass": overall_flat,
        "extension_required": phase.reference_update == 4_816 and not overall_flat,
        "extended_terminal_still_not_flat": phase.reference_update == 9_632 and not overall_flat,
        "plateau_claimed": False,
        "interpretation": (
            "Frozen terminal-flatness diagnostic: reconstruction NMSE and local-decoder NMSE "
            "use relative changes; all other metrics use absolute changes. Passing supports "
            "the protocol's bounded terminal-flatness statement, not a universal plateau."
        ),
    }


def _replay_anchor_rows(phase: ParsedPhase) -> list[dict[str, Any]]:
    anchor = phase.replay_anchor_update
    if anchor is None or anchor == phase.reference_update:
        return []
    rows: list[dict[str, Any]] = []
    for pool in phase.pool_sizes:
        for metric, spec in METRICS.items():
            if not all(
                (pool, seed, anchor) in phase.records
                and metric in phase.records[(pool, seed, anchor)]
                and (pool, seed, phase.reference_update) in phase.records
                and metric in phase.records[(pool, seed, phase.reference_update)]
                for seed in phase.seeds
            ):
                continue
            values = [
                signed_degradation(
                    metric,
                    phase.records[(pool, seed, anchor)][metric],
                    phase.records[(pool, seed, phase.reference_update)][metric],
                )
                for seed in phase.seeds
            ]
            summary = _descriptive(values)
            rows.append(
                {
                    "phase": phase.phase,
                    "pool_size": pool,
                    "replay_anchor_update": anchor,
                    "long_reference_update": phase.reference_update,
                    "metric": metric,
                    "comparison_scale": spec.margin_scale,
                    "n": summary["n"],
                    "mean_signed_degradation": summary["mean"],
                    "sample_sd": summary["sample_sd"],
                    "positive_means_anchor_is_worse": True,
                }
            )
    return rows


def _replay_verification(path: Path | None, *, discovery_sha256: str) -> dict[str, Any]:
    """Validate an optional external comparison to the prior 1,204-update run."""

    if path is None:
        return {
            "provided": False,
            "verified": False,
            "file_sha256": None,
            "reason": "external prior-result replay verification was not supplied",
        }
    payload, file_sha256 = _load_json(path)
    if payload.get("protocol") != REPLAY_VERIFICATION_PROTOCOL or payload.get("status") != "ok":
        raise ValueError("replay verification is not a successful recognized result")
    if (
        _sha256(
            payload.get("discovery_results_sha256"),
            name="replay verification discovery-result hash",
        )
        != discovery_sha256
    ):
        raise ValueError("replay verification identifies another discovery result")
    if (
        _sha256(
            payload.get("protocol_document_sha256"),
            name="replay verification protocol-document hash",
        )
        != FROZEN_PROTOCOL_SHA256
    ):
        raise ValueError("replay verification identifies another frozen protocol")
    checked = payload.get("external_prior_result_checked") is True
    matched = payload.get("all_required_identities_match") is True
    return {
        "provided": True,
        "verified": checked and matched,
        "file_sha256": file_sha256,
        "external_prior_result_checked": checked,
        "all_required_identities_match": matched,
        "reason": None if checked and matched else "external replay comparison did not pass",
    }


def _confirmation_decision(
    discovery: ParsedPhase,
    confirmation: ParsedPhase,
    common_discovery_candidate: int | None,
    confirmation_degradations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(discovery.seeds).intersection(confirmation.seeds):
        raise ValueError("confirmation seeds must be fresh relative to discovery")
    if discovery.pool_sizes != confirmation.pool_sizes:
        raise ValueError("confirmation pool sizes differ from discovery")
    if discovery.design_identity_sha256 != confirmation.design_identity_sha256:
        raise ValueError("confirmation design identity differs from discovery")
    locked = confirmation.locked_candidate_update
    if common_discovery_candidate is None:
        raise ValueError("cannot confirm because discovery has no common sustained candidate")
    if locked != common_discovery_candidate:
        raise ValueError(
            "confirmation locked update differs from the common sustained discovery candidate"
        )
    if confirmation.reference_update != discovery.reference_update:
        raise ValueError("confirmation and discovery use different long-budget references")
    metric_rows = [
        row
        for row in confirmation_degradations
        if int(row["update"]) == locked and str(row["metric"]) in METRICS
    ]
    expected = len(confirmation.pool_sizes) * len(METRICS)
    if len(metric_rows) != expected:
        raise ValueError("confirmation lacks a complete candidate-by-pool metric block")
    pools = []
    for pool in confirmation.pool_sizes:
        pool_rows = [row for row in metric_rows if int(row["pool_size"]) == pool]
        passed = all(bool(row["confidence_bound_within_margin"]) for row in pool_rows)
        pools.append(
            {
                "pool_size": pool,
                "locked_candidate_update": locked,
                "reference_update": confirmation.reference_update,
                "all_metric_noninferiority_pass": passed,
                "failed_metrics": [
                    str(row["metric"])
                    for row in pool_rows
                    if not bool(row["confidence_bound_within_margin"])
                ],
            }
        )
    return {
        "locked_candidate_update": locked,
        "reference_update": confirmation.reference_update,
        "fresh_seed_count": len(confirmation.seeds),
        "fresh_seeds": confirmation.seeds,
        "confidence_method": "one-sided paired Student-t upper confidence bound",
        "confidence_level": ONE_SIDED_CONFIDENCE,
        "decision_type": "all-metric non-inferiority; not symmetric equivalence",
        "pools": pools,
        "overall_all_metric_noninferiority_pass": all(
            pool["all_metric_noninferiority_pass"] for pool in pools
        ),
    }


def _plot(
    path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[int, int | None],
    reference_update: int,
    replay_anchor_update: int | None,
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_config = Path(os.environ.get("TMPDIR", "/tmp")) / "drb-matplotlib-cache"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_metrics = (
        "reconstruction_nmse",
        "sample_conditional_trustworthiness",
        "sample_conditional_neighbor_overlap",
        "sampled_pair_normalized_stress",
    )
    lookup: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    pools = sorted({int(row["pool_size"]) for row in aggregate_rows})
    for pool in pools:
        for metric in selected_metrics:
            lookup[(pool, metric)] = sorted(
                (
                    row
                    for row in aggregate_rows
                    if int(row["pool_size"]) == pool and row["metric"] == metric
                ),
                key=lambda row: int(row["update"]),
            )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for axis, metric in zip(axes.ravel(), selected_metrics, strict=True):
        spec = METRICS[metric]
        for pool in pools:
            rows = lookup[(pool, metric)]
            if not rows:
                continue
            x = [int(row["update"]) for row in rows]
            y = [float(row["mean"]) for row in rows]
            sd = [float(row["sample_sd"] or 0.0) for row in rows]
            line = axis.plot(x, y, marker="o", markersize=3, label=f"{pool:,} rows")[0]
            axis.fill_between(
                x,
                [mean - spread for mean, spread in zip(y, sd, strict=True)],
                [mean + spread for mean, spread in zip(y, sd, strict=True)],
                color=line.get_color(),
                alpha=0.12,
            )
            candidate = candidates.get(pool)
            if candidate is not None and candidate in x:
                index = x.index(candidate)
                axis.scatter(
                    [candidate], [y[index]], s=54, facecolors="none", edgecolors=line.get_color()
                )
        axis.axvline(
            reference_update,
            color="black",
            linestyle=":",
            linewidth=1,
            label=f"{reference_update:,}-update reference",
        )
        if replay_anchor_update is not None and replay_anchor_update != reference_update:
            axis.axvline(
                replay_anchor_update,
                color="0.45",
                linestyle="--",
                linewidth=1,
                label=f"{replay_anchor_update:,}-update replay anchor",
            )
        observed_updates = [int(row["update"]) for pool in pools for row in lookup[(pool, metric)]]
        if observed_updates and min(observed_updates) <= 0:
            axis.set_xscale("symlog", linthresh=1)
            axis.set_xlim(left=-0.5, right=max(observed_updates) * 1.12)
            axis.set_xlabel("Optimizer updates (symmetric log scale)")
        else:
            axis.set_xscale("log")
            axis.set_xlabel("Optimizer updates (log scale)")
        axis.set_ylabel(spec.display_name)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "AE update-budget discovery curves\n"
        "circles: means; bands: sample SD; open markers: sustained-pass candidates",
        fontsize=11,
    )
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        figure.savefig(
            temporary,
            dpi=180,
            format=path.suffix.lstrip("."),
            metadata={},
            facecolor="white",
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def analyze(args) -> dict[str, Any]:
    discovery_payload, discovery_sha256 = _load_json(args.discovery_results)
    strict_protocol = not args.allow_nonprotocol_input
    discovery = _parse_phase(
        discovery_payload,
        discovery_sha256,
        "discovery",
        strict_protocol=strict_protocol,
    )
    discovery_aggregate = _aggregate_rows(discovery)
    discovery_degradations = _degradation_rows(discovery, include_confidence_bounds=False)
    candidate_rows, candidates = _discovery_decisions(discovery, discovery_degradations)
    common_candidate = _common_sustained_candidate(discovery, candidate_rows)
    fine_grid = _fine_grid_status(discovery, candidate_rows)
    terminal_rows, terminal_summary = _terminal_rows(discovery)
    replay_rows = _replay_anchor_rows(discovery)
    replay_verification = _replay_verification(
        args.replay_verification, discovery_sha256=discovery.input_sha256
    )

    candidate_compute = {}
    for pool, candidate in candidates.items():
        candidate_compute[str(pool)] = (
            {
                "candidate_update": candidate,
                "reference_update": discovery.reference_update,
                "updates_saved": discovery.reference_update - candidate,
                "update_reduction_fraction": 1.0 - candidate / discovery.reference_update,
                "idealized_optimizer_step_speedup": discovery.reference_update / candidate,
            }
            if candidate is not None and candidate > 0
            else None
        )
    confirmation = None
    confirmation_degradations: list[dict[str, Any]] = []
    confirmation_aggregate: list[dict[str, Any]] = []
    confirmation_summary = None
    confirmation_sha256 = None
    if args.confirmation_results is not None:
        confirmation_payload, confirmation_sha256 = _load_json(args.confirmation_results)
        confirmation = _parse_phase(
            confirmation_payload,
            confirmation_sha256,
            "confirmation",
            strict_protocol=strict_protocol,
        )
        confirmation_aggregate = _aggregate_rows(confirmation)
        confirmation_degradations = _degradation_rows(confirmation, include_confidence_bounds=True)
        confirmation_summary = _confirmation_decision(
            discovery, confirmation, common_candidate, confirmation_degradations
        )
        replay_rows.extend(_replay_anchor_rows(confirmation))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = discovery_aggregate + confirmation_aggregate
    degradation_rows = discovery_degradations + confirmation_degradations
    _atomic_csv(output_dir / "aggregate.csv", aggregate_rows)
    _atomic_csv(output_dir / "paired-degradation.csv", degradation_rows)
    _atomic_csv(output_dir / "candidate-decisions.csv", candidate_rows)
    _atomic_csv(output_dir / "terminal-slope.csv", terminal_rows)
    _atomic_csv(output_dir / "replay-anchor.csv", replay_rows)
    _plot(
        output_dir / "update-budget-curves.png",
        discovery_aggregate,
        candidates,
        discovery.reference_update,
        discovery.replay_anchor_update,
    )
    _plot(
        output_dir / "update-budget-curves.pdf",
        discovery_aggregate,
        candidates,
        discovery.reference_update,
        discovery.replay_anchor_update,
    )

    prerequisites = {
        "frozen_protocol_sha256_verified": strict_protocol,
        "fine_grid_complete": fine_grid["all_pools_complete"],
        "terminal_flatness_pass": terminal_summary["all_pools_terminal_flatness_pass"],
        "external_prior_replay_verified": replay_verification["verified"],
        "fresh_seed_confirmation_pass": bool(
            confirmation_summary and confirmation_summary["overall_all_metric_noninferiority_pass"]
        ),
    }
    all_prerequisites_pass = all(prerequisites.values())
    recommendation = common_candidate if all_prerequisites_pass else None
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "ok",
        "generated_at_utc": _utc_now(),
        "inputs": {
            "discovery": {
                "protocol": TRAINING_PROTOCOL,
                "file_sha256": discovery.input_sha256,
                "design_identity_sha256": discovery.design_identity_sha256,
            },
            "confirmation": (
                {
                    "protocol": TRAINING_PROTOCOL,
                    "file_sha256": confirmation_sha256,
                    "design_identity_sha256": confirmation.design_identity_sha256,
                }
                if confirmation is not None
                else None
            ),
        },
        "decision_protocol": {
            "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "strict_protocol_validation": strict_protocol,
            "positive_signed_degradation_means_worse": True,
            "margins": {metric: asdict(spec) for metric, spec in METRICS.items()},
            "discovery_rule": (
                "Earliest positive update whose all-metric point estimates are inside the "
                "frozen margins, no seed exceeds twice a margin, and that checkpoint plus "
                "the next two evaluated checkpoints all pass."
            ),
            "confirmation_rule": (
                "Every metric's one-sided 95% paired-t upper harm bound must be strictly "
                "below its frozen margin for every pool."
            ),
            "multiplicity": (
                "The all-metric declaration is an intersection-union gate; every component "
                "must pass, so no additional multiplicity correction is applied."
            ),
        },
        "discovery": {
            "pool_sizes": discovery.pool_sizes,
            "seeds": discovery.seeds,
            "checkpoint_updates": discovery.checkpoint_updates,
            "long_reference_update": discovery.reference_update,
            "replay_anchor_update": discovery.replay_anchor_update,
            "candidate_update_by_pool": {
                str(pool): candidate for pool, candidate in candidates.items()
            },
            "earliest_common_sustained_candidate_update": common_candidate,
            "fine_grid": fine_grid,
            "recommendation_scope": "earliest common sustained-pass checkpoint on the tested grid",
            "candidate_compute_by_pool": candidate_compute,
            "replay_audit": discovery.replay_audit,
        },
        "terminal_slope": terminal_summary,
        "external_replay_verification": replay_verification,
        "confirmation": confirmation_summary,
        "recommendation_gate": {
            "prerequisites": prerequisites,
            "all_prerequisites_pass": all_prerequisites_pass,
            "recommended_update": recommendation,
            "suppressed_reason": (
                None
                if all_prerequisites_pass
                else "No recommendation is issued until every frozen prerequisite passes."
            ),
        },
        "artifacts": [
            "aggregate.csv",
            "paired-degradation.csv",
            "candidate-decisions.csv",
            "terminal-slope.csv",
            "replay-anchor.csv",
            "update-budget-curves.png",
            "update-budget-curves.pdf",
        ],
        "limitations": [
            "The practical margins are frozen engineering tolerances, not universal constants.",
            "The long-budget checkpoint is a tested reference, not a proven global optimum.",
            "Terminal changes are descriptive and never trigger an automatic plateau claim.",
            "replay-anchor.csv compares update 1204 with the long reference; only the separate "
            "external replay-verification input can establish agreement with the prior run.",
            "Seeds, rather than geometry rows or sampled pairs, are the replicate units.",
        ],
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-results", required=True, type=Path)
    parser.add_argument("--confirmation-results", type=Path)
    parser.add_argument("--replay-verification", type=Path)
    parser.add_argument(
        "--allow-nonprotocol-input",
        action="store_true",
        help="allow synthetic smoke-test grids that are not bound to the frozen protocol",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/results/ae-update-budget-sensitivity")
    )
    args = parser.parse_args(argv)
    args.discovery_results = args.discovery_results.expanduser().resolve()
    if args.confirmation_results is not None:
        args.confirmation_results = args.confirmation_results.expanduser().resolve()
    if args.replay_verification is not None:
        args.replay_verification = args.replay_verification.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args)
    print(
        "wrote update-budget analysis with common discovery candidate "
        f"{summary['discovery']['earliest_common_sustained_candidate_update']} "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
