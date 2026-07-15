"""Tests for public-safe AE training-pool-size aggregation and plotting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from experiments.analyze_ae_training_size_results import (
    GEOMETRY_PROTOCOL,
    RETRIEVAL_PROTOCOL,
    TRAINING_PROTOCOL,
    analyze,
    parse_args,
)


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path):
    pools = [10, 100, 1000]
    seeds = [0, 1, 2]
    arms = []
    for seed in seeds:
        for pool_index, pool in enumerate(pools):
            arms.append(
                {
                    "init_seed": seed,
                    "pool_size": pool,
                    "initial_state_sha256": f"{seed + 1:064x}",
                    "first_pass": {
                        "equivalent_passes": 1.0,
                        "tune": {
                            "reconstruction_nmse": 0.90 - 0.10 * pool_index + 0.01 * seed,
                            "reconstruction_cosine": 0.40 + 0.10 * pool_index + 0.01 * seed,
                        },
                    },
                    "fixed_budget": {
                        "update": 22,
                        "tune": {
                            "reconstruction_nmse": 0.60 - 0.02 * pool_index + 0.01 * seed,
                            "reconstruction_cosine": 0.65 + 0.02 * pool_index + 0.01 * seed,
                        },
                    },
                    "selection": {"best_update": 20 + pool_index},
                    "test": {
                        "reconstruction_nmse": 0.50 - 0.01 * pool_index + 0.01 * seed,
                        "reconstruction_cosine": 0.70 + 0.01 * pool_index + 0.01 * seed,
                        "evaluated_once": True,
                        "checkpoint_selected_without_test": True,
                    },
                }
            )
    training = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "status": "ok",
        "source": {
            "path": "/private/source.fbin",
            "rows": 10_000,
            "dimension": 96,
            "file_sha256": "a" * 64,
            "input_transformation": "none",
        },
        "split": {
            "test": {"rows": 64, "indices_sha256": "b" * 64},
            "training": {"pool_sizes": pools},
        },
        "protocol_configuration": {
            "architecture": "96-20-8-symmetric",
            "target_dimension": 8,
            "update_budget": 22,
        },
        "arms": arms,
        "artifacts": {"checkpoint_paths": ["/private/checkpoint.pt"]},
    }
    training_path = tmp_path / "training.json"
    training_sha256 = _write_json(training_path, training)

    geometry_arms = []
    for arm in arms:
        pool_index = pools.index(arm["pool_size"])
        seed = arm["init_seed"]
        geometry_arms.append(
            {
                "init_seed": seed,
                "pool_size": arm["pool_size"],
                "training_result": {"best_update": arm["selection"]["best_update"]},
                "metrics": {
                    "sample_conditional_trustworthiness": 0.90 + 0.01 * pool_index + 0.001 * seed,
                    "sample_conditional_continuity": 0.91 + 0.01 * pool_index + 0.001 * seed,
                    "sample_conditional_neighbor_overlap": 0.20 + 0.02 * pool_index + 0.001 * seed,
                    "sampled_pair_distance_spearman": 0.30 + 0.03 * pool_index + 0.001 * seed,
                    "sampled_pair_normalized_stress": 0.40 - 0.02 * pool_index + 0.001 * seed,
                    "sampled_local_decodability_nmse": 0.50 - 0.01 * pool_index + 0.001 * seed,
                },
            }
        )
    geometry = {
        "schema_version": 1,
        "protocol": GEOMETRY_PROTOCOL,
        "status": "ok",
        "training_result": {"file_sha256": training_sha256},
        "published_protocol": {
            "only_common_test_rows_encoded": True,
            "test_indices_sha256": "b" * 64,
        },
        "arms": geometry_arms,
    }
    geometry_path = tmp_path / "geometry.json"
    _write_json(geometry_path, geometry)

    retrieval_arms = []
    for arm in arms:
        pool_index = pools.index(arm["pool_size"])
        quality = {"recall_at": {}, "recall_hits": {}, "recall_denominator": {}}
        for cutoff in (10, 100):
            denominator = 10 * cutoff
            hits = 5 + 2 * pool_index + arm["init_seed"]
            quality["recall_at"][str(cutoff)] = hits / denominator
            quality["recall_hits"][str(cutoff)] = hits
            quality["recall_denominator"][str(cutoff)] = denominator
        retrieval_arms.append(
            {
                "label": f"seed {arm['init_seed']} pool {arm['pool_size']}",
                "selection": {
                    "init_seed": arm["init_seed"],
                    "pool_size": arm["pool_size"],
                    "selected_update": arm["selection"]["best_update"],
                },
                "verification": {
                    "training_result_sha256": training_sha256,
                    "checkpoint_state_dict_sha256": "c" * 64,
                    "encoding_provenance_sha256": "d" * 64,
                },
                "quality": quality,
            }
        )
    retrieval = {
        "schema_version": 1,
        "protocol": RETRIEVAL_PROTOCOL,
        "status": "ok",
        "cutoffs": [10, 100],
        "query_count": 10,
        "verification": {"all_encoding_provenance_verified": True},
        "arms": retrieval_arms,
    }
    retrieval_path = tmp_path / "retrieval.json"
    _write_json(retrieval_path, retrieval)
    return training_path, geometry_path, retrieval_path


def _args(tmp_path: Path, *, optional: bool = True):
    training, geometry, retrieval = _inputs(tmp_path)
    arguments = [
        "--training-results",
        str(training),
        "--output-dir",
        str(tmp_path / "public"),
    ]
    if optional:
        arguments.extend(
            [
                "--geometry-results",
                str(geometry),
                "--retrieval-results",
                str(retrieval),
            ]
        )
    return parse_args(arguments), geometry, retrieval


def _row(rows: list[dict[str, str]], pool: int, metric: str) -> dict[str, str]:
    return next(row for row in rows if int(row["pool_size"]) == pool and row["metric"] == metric)


def test_full_analysis_aggregates_pairs_and_omits_private_paths(tmp_path):
    pytest.importorskip("matplotlib")
    args, _, _ = _args(tmp_path)

    summary = analyze(args)
    output = args.output_dir
    with (output / "aggregate.csv").open(newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    with (output / "paired-contrasts.csv").open(newline="") as handle:
        contrasts = list(csv.DictReader(handle))

    one_pass = _row(aggregate, 10, "one_pass_tune_reconstruction_nmse")
    fixed = _row(aggregate, 10, "fixed_budget_tune_reconstruction_nmse")
    selected = _row(aggregate, 10, "selected_checkpoint_legacy_eval_reconstruction_nmse")
    assert float(one_pass["mean"]) == pytest.approx(0.91)
    assert float(one_pass["sample_sd"]) == pytest.approx(0.01)
    assert float(fixed["mean"]) == pytest.approx(0.61)
    assert float(selected["mean"]) == pytest.approx(0.51)
    assert {one_pass["estimand"], fixed["estimand"], selected["estimand"]} == {
        "one_pass_tune",
        "fixed_budget_tune",
        "selected_checkpoint_legacy_eval",
    }
    assert (
        _row(aggregate, 1000, "sample_conditional_neighbor_overlap")["metric_group"] == "geometry"
    )
    assert _row(aggregate, 1000, "recall_at_100")["metric_group"] == "retrieval"

    contrast = next(
        row
        for row in contrasts
        if int(row["pool_size"]) == 10
        and row["metric"] == "selected_checkpoint_legacy_eval_reconstruction_nmse"
    )
    assert int(contrast["reference_pool_size"]) == 1000
    assert int(contrast["n_pairs"]) == 3
    assert float(contrast["mean_delta_pool_minus_reference"]) == pytest.approx(0.02)
    assert float(contrast["mean_favorable_delta"]) == pytest.approx(-0.02)

    serialized = (output / "summary.json").read_text()
    assert str(tmp_path) not in serialized
    assert "/private/source.fbin" not in serialized
    assert "/private/checkpoint.pt" not in serialized
    assert summary["public_safety"]["absolute_paths_omitted"] is True
    assert (output / "learning-curves.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "learning-curves.pdf").read_bytes().startswith(b"%PDF")
    for artifact in summary["artifacts"].values():
        assert "/" not in artifact["filename"]
        assert len(artifact["sha256"]) == 64


def test_training_only_analysis_still_writes_compact_learning_curve(tmp_path):
    pytest.importorskip("matplotlib")
    args, _, _ = _args(tmp_path, optional=False)

    summary = analyze(args)

    assert summary["inputs"]["geometry"] is None
    assert summary["inputs"]["retrieval"] == []
    assert len(summary["aggregates"]) == 3 * 6
    assert len(summary["paired_contrasts"]) == 2 * 6
    assert {row["metric_group"] for row in summary["aggregates"]} == {"reconstruction"}
    assert (args.output_dir / "learning-curves.png").is_file()
    assert (args.output_dir / "learning-curves.pdf").is_file()


def test_geometry_must_link_to_exact_training_result(tmp_path):
    args, geometry_path, _ = _args(tmp_path)
    geometry = json.loads(geometry_path.read_text())
    geometry["training_result"]["file_sha256"] = "0" * 64
    _write_json(geometry_path, geometry)

    with pytest.raises(ValueError, match="supplied training result hash"):
        analyze(args)


def test_retrieval_selection_and_provenance_are_verified(tmp_path):
    args, _, retrieval_path = _args(tmp_path)
    retrieval = json.loads(retrieval_path.read_text())
    retrieval["arms"][0]["selection"]["selected_update"] += 1
    _write_json(retrieval_path, retrieval)
    with pytest.raises(ValueError, match="selected update disagrees"):
        analyze(args)

    retrieval["arms"][0]["selection"]["selected_update"] -= 1
    retrieval["verification"]["all_encoding_provenance_verified"] = False
    _write_json(retrieval_path, retrieval)
    with pytest.raises(ValueError, match="did not verify every encoding provenance"):
        analyze(args)


def test_absolute_title_is_rejected_before_publication(tmp_path):
    training, _, _ = _inputs(tmp_path)
    with pytest.raises(ValueError, match="absolute path"):
        parse_args(
            [
                "--training-results",
                str(training),
                "--output-dir",
                str(tmp_path / "public"),
                "--title",
                "/private/title",
            ]
        )
