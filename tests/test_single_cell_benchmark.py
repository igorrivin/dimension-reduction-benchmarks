"""CPU tests for the prepared single-cell reducer benchmark."""

import hashlib
import json
import time
from types import SimpleNamespace

import experiments.compare_single_cell_reducers as benchmark
import numpy as np
import pytest
from experiments.compare_single_cell_reducers import (
    INPUT_PROTOCOL,
    deterministic_label_sample,
    donor_heldout_cell_type_metrics,
    evaluate_single_cell_layout,
    orchestrate,
    parse_args,
    planned_runs,
    run_worker,
    technical_covariate_leakage,
    validate_prepared_input,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(path, rows):
    donors = np.asarray([f"d{index % 3}" for index in range(rows)])
    cell_types = np.asarray(["T" if index % 2 else "B" for index in range(rows)])
    np.savez(
        path,
        donor_id=donors,
        cell_type=cell_types,
        total_counts=np.arange(rows, dtype=np.float64) + 100,
    )


def test_validation_preserves_prepared_values_and_reports_optional_columns(tmp_path):
    rng = np.random.default_rng(3)
    data = tmp_path / "pca50.npy"
    metadata = tmp_path / "metadata.npz"
    manifest = tmp_path / "manifest.json"
    X = rng.normal(size=(30, 6)).astype(np.float32)
    np.save(data, X)
    _metadata(metadata, len(X))
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": {"selected_cells": len(X)},
                "artifacts": {
                    "pca50": {
                        "path": data.name,
                        "sha256": _sha256(data),
                        "bytes": data.stat().st_size,
                        "shape": list(X.shape),
                        "dtype": "float32",
                    },
                    "metadata": {
                        "path": metadata.name,
                        "sha256": _sha256(metadata),
                        "bytes": metadata.stat().st_size,
                        "shape": [len(X), 3],
                        "dtype": "npz",
                        "columns": ["cell_type", "donor_id", "total_counts"],
                    },
                },
            }
        )
    )
    before = _sha256(data)

    report, loaded, source_manifest = validate_prepared_input(
        data,
        metadata_path=metadata,
        manifest_path=manifest,
        input_representation="pca50",
        required_columns=("cell_type", "donor_id"),
        technical_columns=("total_counts", "pct_counts_mt"),
        block_rows=7,
    )

    assert _sha256(data) == before
    assert np.array_equal(np.load(data), X)
    assert report["input_transform"] == "none"
    assert report["input_protocol"] == INPUT_PROTOCOL
    assert report["available_technical_columns"] == ["total_counts"]
    assert report["missing_technical_columns"] == ["pct_counts_mt"]
    assert loaded["cell_type"].shape == (30,)
    assert source_manifest["selection"]["selected_cells"] == 30
    assert (
        report["manifest_semantic_checks"]["authoritative_row_count"] == "selection.selected_cells"
    )
    assert set(report["manifest_semantic_checks"]["representation"]["verified_fields"]) >= {
        "path",
        "sha256",
        "shape",
        "dtype",
    }


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.ones((8, 3), dtype=np.float64), "float32"),
        (np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32), "non-finite"),
        (np.ones(8, dtype=np.float32), "two-dimensional"),
    ],
)
def test_validation_rejects_invalid_prepared_arrays(tmp_path, array, message):
    data = tmp_path / "bad.npy"
    np.save(data, array)
    with pytest.raises(ValueError, match=message):
        validate_prepared_input(data)


def test_validation_rejects_misaligned_or_missing_metadata(tmp_path):
    data = tmp_path / "pca50.npy"
    metadata = tmp_path / "metadata.npz"
    np.save(data, np.ones((8, 3), dtype=np.float32))
    np.savez(metadata, cell_type=np.repeat("T", 7), donor_id=np.repeat("d", 7))
    with pytest.raises(ValueError, match="aligned"):
        validate_prepared_input(
            data,
            metadata_path=metadata,
            required_columns=("cell_type", "donor_id"),
        )

    np.savez(metadata, cell_type=np.repeat("T", 8))
    with pytest.raises(ValueError, match="missing required"):
        validate_prepared_input(
            data,
            metadata_path=metadata,
            required_columns=("cell_type", "donor_id"),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("selection.selected_cells", 99, "selection.selected_cells"),
        ("pca50.sha256", "0" * 64, "pca50.sha256"),
        ("pca50.shape", [8, 99], "pca50.shape"),
        ("pca50.dtype", "float64", "pca50.dtype"),
        ("pca50.path", "other.npy", "pca50.path"),
        ("metadata.sha256", "0" * 64, "metadata.sha256"),
        ("metadata.shape", [99, 3], "metadata.shape"),
    ],
)
def test_manifest_semantics_are_cross_checked(tmp_path, field, bad_value, message):
    data = tmp_path / "pca50.npy"
    metadata = tmp_path / "metadata.npz"
    manifest_path = tmp_path / "manifest.json"
    X = np.ones((8, 3), dtype=np.float32)
    np.save(data, X)
    _metadata(metadata, len(X))
    manifest = {
        "schema_version": 1,
        "selection": {"selected_cells": len(X)},
        "artifacts": {
            "pca50": {
                "path": data.name,
                "sha256": _sha256(data),
                "bytes": data.stat().st_size,
                "shape": list(X.shape),
                "dtype": "float32",
            },
            "metadata": {
                "path": metadata.name,
                "sha256": _sha256(metadata),
                "bytes": metadata.stat().st_size,
                "shape": [len(X), 3],
                "dtype": "npz",
                "columns": ["cell_type", "donor_id", "total_counts"],
            },
        },
    }
    if field.startswith("selection."):
        manifest["selection"][field.split(".", 1)[1]] = bad_value
    elif "." in field:
        artifact, key = field.split(".")
        manifest["artifacts"][artifact][key] = bad_value
    else:
        manifest[field] = bad_value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        validate_prepared_input(
            data,
            metadata_path=metadata,
            manifest_path=manifest_path,
            input_representation="pca50",
            required_columns=("cell_type", "donor_id"),
        )


def test_provisional_manifest_field_names_remain_compatible(tmp_path):
    data = tmp_path / "pca50.npy"
    metadata = tmp_path / "metadata.npz"
    manifest_path = tmp_path / "manifest.json"
    X = np.ones((8, 3), dtype=np.float32)
    np.save(data, X)
    _metadata(metadata, len(X))
    manifest_path.write_text(
        json.dumps(
            {
                "row_count": len(X),
                "artifacts": {
                    "pca50": {
                        "filename": data.name,
                        "sha256": _sha256(data),
                        "shape": list(X.shape),
                        "dtype": "float32",
                    },
                    "metadata": {
                        "filename": metadata.name,
                        "sha256": _sha256(metadata),
                        "rows": len(X),
                    },
                },
            }
        )
    )

    report, _, _ = validate_prepared_input(
        data,
        metadata_path=metadata,
        manifest_path=manifest_path,
        input_representation="pca50",
        required_columns=("cell_type", "donor_id"),
    )

    assert report["manifest_semantic_checks"]["authoritative_row_count"] == "row_count"
    assert "filename" in report["manifest_semantic_checks"]["representation"]["verified_fields"]


def test_deterministic_label_sample_represents_cell_type_donor_strata():
    labels = np.tile(np.repeat(["A", "B"], 5), 3)
    donors = np.repeat(["d1", "d2", "d3"], 10)
    first = deterministic_label_sample(labels, donors, 18, 42)
    second = deterministic_label_sample(labels, donors, 18, 42)
    represented = {(labels[index], donors[index]) for index in first}

    assert np.array_equal(first, second)
    assert represented == {(label, donor) for label in ("A", "B") for donor in donors[::10]}


def test_donor_heldout_probe_is_perfect_for_donor_invariant_clusters():
    donors = np.repeat(["d1", "d2", "d3"], 20)
    labels = np.tile(np.repeat(["A", "B"], 10), 3)
    within = np.tile(np.arange(10), 6) * 1e-3
    layout = np.column_stack(((labels == "B").astype(float) * 10 + within, within)).astype(
        np.float32
    )

    metrics = donor_heldout_cell_type_metrics(
        layout,
        labels,
        donors,
        sample_size=60,
        seed=4,
        n_neighbors=3,
        min_count=3,
        min_donors=3,
    )

    assert metrics["label_probe_status"] == "ok"
    assert metrics["donor_heldout_folds"] == 3
    assert metrics["donor_heldout_macro_f1"] == pytest.approx(1.0)
    assert metrics["donor_heldout_balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["donor_heldout_accuracy"] == pytest.approx(1.0)
    assert metrics["donor_heldout_per_cell_type_f1"] == {"A": 1.0, "B": 1.0}
    assert metrics["cell_type_minimum_donors"] == 3


def test_donor_heldout_probe_does_not_score_donor_private_cell_types():
    layout = np.arange(24, dtype=np.float32)[:, None]
    donors = np.repeat(["d1", "d2"], 12)
    labels = np.repeat(["private-a", "private-b"], 12)

    metrics = donor_heldout_cell_type_metrics(
        layout,
        labels,
        donors,
        sample_size=24,
        seed=1,
        n_neighbors=3,
        min_count=2,
        min_donors=2,
    )

    assert metrics["donor_heldout_macro_f1"] is None
    assert metrics["donor_heldout_evaluated_rows"] == 0
    assert "donor coverage thresholds" in metrics["label_probe_status"]


def test_cell_types_must_span_requested_minimum_number_of_donors():
    donors = np.repeat(["d1", "d2"], 20)
    labels = np.tile(np.repeat(["A", "B"], 10), 2)
    layout = np.column_stack(((labels == "B").astype(float), np.arange(40))).astype(np.float32)

    metrics = donor_heldout_cell_type_metrics(
        layout,
        labels,
        donors,
        sample_size=40,
        seed=4,
        n_neighbors=3,
        min_count=3,
        min_donors=3,
    )

    assert metrics["donor_heldout_macro_f1"] is None
    assert metrics["cell_type_minimum_donors"] == 3
    assert "donor coverage thresholds" in metrics["label_probe_status"]


def test_technical_leakage_uses_cross_donor_predictions():
    grid = np.linspace(-2, 2, 20)
    donors = np.repeat(["d1", "d2", "d3"], grid.size)
    cell_types = np.repeat("T", donors.size)
    values = np.tile(grid, 3)
    layout = np.column_stack((values, values**2)).astype(np.float32)

    metrics = technical_covariate_leakage(
        layout,
        cell_types,
        donors,
        {"total_counts": values},
        sample_size=60,
        seed=2,
        n_neighbors=1,
        min_donors=3,
    )["total_counts"]

    assert metrics["status"] == "ok"
    assert metrics["folds"] == 3
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["nmse"] == pytest.approx(0.0)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["coverage_fraction"] == pytest.approx(1.0)
    assert metrics["protocol"].startswith("within-cell-type leave-one-donor-out")


def test_technical_leakage_does_not_reward_cell_type_mean_differences():
    # The covariate and layout encode cell type perfectly but have no within-cell-type
    # variation. An unconditional probe would incorrectly call this technical leakage.
    donors = np.tile(np.repeat(["d1", "d2", "d3"], 10), 2)
    cell_types = np.repeat(["A", "B"], 30)
    values = (cell_types == "B").astype(float) * 100
    layout = np.column_stack(((cell_types == "B").astype(float), np.zeros(60))).astype(np.float32)

    metrics = technical_covariate_leakage(
        layout,
        cell_types,
        donors,
        {"total_counts": values},
        sample_size=60,
        seed=2,
        n_neighbors=3,
        min_donors=3,
    )["total_counts"]

    assert metrics["r2"] is None
    assert metrics["nmse"] is None
    assert metrics["status"] == "insufficient within-cell-type numeric variation"


def test_complete_evaluation_reports_geometry_labels_and_leakage():
    rng = np.random.default_rng(8)
    layout = rng.normal(size=(60, 2)).astype(np.float32)
    X = layout.copy()
    donors = np.repeat(["d1", "d2", "d3"], 20)
    labels = np.tile(np.repeat(["A", "B"], 10), 3)
    metadata = {
        "donor_id": donors,
        "cell_type": labels,
        "total_counts": np.tile(np.arange(20), 3),
    }
    args = SimpleNamespace(
        current_target_dimension=2,
        metric_sample=60,
        evaluation_seed=5,
        neighbors=4,
        metric_jobs=1,
        distance_pairs=200,
        input_representation="pca50",
        cell_type_column="cell_type",
        donor_column="donor_id",
        label_sample=60,
        silhouette_sample=60,
        cell_type_min_count=3,
        cell_type_min_donors=3,
        technical_columns=["total_counts", "pct_counts_mt"],
        technical_sample=60,
    )

    metrics = evaluate_single_cell_layout(X, layout, args, metadata)

    assert metrics["sample_conditional_trustworthiness"] == pytest.approx(1.0)
    assert metrics["sample_conditional_continuity"] == pytest.approx(1.0)
    assert metrics["sample_conditional_neighbor_overlap"] == pytest.approx(1.0)
    assert metrics["sampled_pair_distance_spearman"] == pytest.approx(1.0)
    assert metrics["sampled_pair_normalized_stress"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["geometry_reference"] == "pca50"
    assert metrics["label_metrics"]["label_probe_protocol"].startswith("leave-one-donor")
    assert np.isfinite(metrics["label_metrics"]["sampled_cell_type_silhouette"])
    assert metrics["label_metrics"]["cell_type_asw_sample_rows"] == 60
    assert set(metrics["technical_covariate_leakage"]) == {"total_counts"}
    assert metrics["missing_technical_columns"] == ["pct_counts_mt"]


def test_dry_run_honors_method_dimension_and_seed_subsets_without_writes(tmp_path, capsys):
    data = tmp_path / "pca50.npy"
    metadata = tmp_path / "metadata.npz"
    output = tmp_path / "results"
    np.save(data, np.arange(240, dtype=np.float32).reshape(40, 6))
    _metadata(metadata, 40)
    args = parse_args(
        [
            "--data",
            str(data),
            "--metadata",
            str(metadata),
            "--methods",
            "pca",
            "autoencoder",
            "--target-dimensions",
            "2",
            "--seeds",
            "7",
            "9",
            "--output-dir",
            str(output),
            "--cell-type-min-count",
            "2",
            "--dry-run",
        ]
    )

    assert orchestrate(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_run_count"] == 4
    assert {(row["method"], row["seed"]) for row in payload["runs"]} == {
        ("pca", 7),
        ("pca", 9),
        ("autoencoder", 7),
        ("autoencoder", 9),
    }
    assert all("row-l2" not in row["command"].lower() for row in payload["runs"])
    assert not output.exists()


def test_planned_run_identity_changes_with_representation(tmp_path):
    data = tmp_path / "prepared.npy"
    np.save(data, np.ones((12, 5), dtype=np.float32))
    validation, _, _ = validate_prepared_input(data)
    args = parse_args(
        [
            "--data",
            str(data),
            "--methods",
            "pca",
            "--target-dimensions",
            "2",
            "--seeds",
            "0",
            "--dry-run",
        ]
    )
    pca50 = planned_runs(args, validation)[0]
    args.input_representation = "log_hvg2000"
    hvg = planned_runs(args, validation)[0]

    assert pca50["fit_run_identity_sha256"] != hvg["fit_run_identity_sha256"]


def test_pca_worker_writes_isolated_result_and_layout(tmp_path, monkeypatch):
    data = tmp_path / "pca50.npy"
    layout = tmp_path / "pca-layout.npy"
    result = tmp_path / "pca.json"
    np.save(data, np.random.default_rng(9).normal(size=(30, 5)).astype(np.float32))

    def cpu_pca(X, args):
        from sklearn.decomposition import PCA

        started = time.perf_counter()
        embedded = PCA(n_components=args.worker_target_dimension).fit_transform(X)
        return embedded, {
            "fit_seconds": time.perf_counter() - started,
            "backend": "sklearn.decomposition.PCA",
        }

    monkeypatch.setattr(benchmark, "fit_pca_large", cpu_pca)
    args = parse_args(
        [
            "--worker",
            "--worker-data",
            str(data),
            "--worker-method",
            "pca",
            "--worker-seed",
            "4",
            "--worker-target-dimension",
            "2",
            "--worker-layout",
            str(layout),
            "--worker-result",
            str(result),
            "--worker-launch-nonce",
            "fixture",
        ]
    )

    assert run_worker(args) == 0
    payload = json.loads(result.read_text())
    assert payload["status"] == "ok"
    assert payload["launch_nonce"] == "fixture"
    assert payload["input_protocol"] == INPUT_PROTOCOL
    assert np.load(layout).shape == (30, 2)
    assert payload["details"]["fit_seconds"] > 0
    assert payload["memory"]["peak_rss_bytes"] > 0
