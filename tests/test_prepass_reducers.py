"""CPU tests for the cached AE/PCA pre-pass comparison harness."""

from types import SimpleNamespace

import numpy as np
import pytest
from experiments.compare_prepass_reducers import (
    PREPASS_VARIANTS,
    _load_labels,
    _run_pca_prepass,
    canonicalize_array,
    canonicalize_npy,
    configuration_identity,
    deterministic_indices,
    evaluate_against_original,
    monotone_hidden_dimensions,
    parse_args,
    prepare_subset,
    timing_accounting,
)


def test_whitening_is_finite_and_invariant_to_invertible_linear_latent_change():
    rng = np.random.default_rng(11)
    codes = rng.normal(size=(800, 8))
    change = rng.normal(size=(8, 8))
    change += 3.0 * np.eye(8)

    first, first_info = canonicalize_array(codes)
    second, second_info = canonicalize_array(codes @ change)

    assert first.dtype == second.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))
    assert np.linalg.norm(first, axis=1) == pytest.approx(np.ones(800), abs=2e-6)
    # Whitened versions differ by an orthogonal rotation; row-wise Gram matrices agree.
    selected = np.arange(100)
    assert first[selected] @ first[selected].T == pytest.approx(
        second[selected] @ second[selected].T, abs=2e-5
    )
    assert first_info["effective_rank"] == second_info["effective_rank"] == 8


def test_whitening_regularizes_rank_deficiency_and_atomic_npy_path(tmp_path):
    rng = np.random.default_rng(4)
    base = rng.normal(size=(100, 3)).astype(np.float32)
    codes = np.column_stack((base, base[:, 0], np.zeros(100))).astype(np.float32)
    source = tmp_path / "raw.npy"
    output = tmp_path / "canonical.npy"
    np.save(source, codes)

    info = canonicalize_npy(source, output, relative_eigenvalue_floor=1e-6, block_rows=17)
    canonical = np.load(output)

    assert canonical.shape == codes.shape
    assert canonical.dtype == np.float32
    assert np.all(np.isfinite(canonical))
    assert info["effective_rank"] == 3
    assert len(info["logical_sha256"]) == 64


@pytest.mark.parametrize(
    ("input_dimension", "width", "expected"),
    [
        (384, 64, (256, 128)),
        (384, 32, (256, 128, 64)),
        (1024, 64, (512, 256, 128)),
        (96, 64, ()),
    ],
)
def test_autoencoder_architecture_strictly_narrows_without_hidden_dip(
    input_dimension, width, expected
):
    hidden = monotone_hidden_dimensions(input_dimension, width)
    assert hidden == expected
    chain = (input_dimension, *hidden, width)
    assert all(left > right for left, right in zip(chain, chain[1:]))
    assert all(value > width for value in hidden)


def test_autoencoder_architecture_rejects_non_compression():
    with pytest.raises(ValueError, match="below input"):
        monotone_hidden_dimensions(32, 32)
    with pytest.raises(ValueError, match="below input"):
        monotone_hidden_dimensions(32, 64)


def test_timing_accounting_keeps_graph_only_and_cold_scopes_distinct():
    timing = timing_accounting(
        2.0,
        0.5,
        7.0,
        prepass_diagnostic_seconds=0.75,
        prepass_worker_wall_seconds=3.0,
        graph_worker_wall_seconds=9.0,
    )

    assert timing["prepass_core_compute_seconds"] == 2.0
    assert timing["prepass_diagnostic_seconds"] == 0.75
    assert timing["graph_only_fit_seconds"] == 7.0
    assert timing["cold_end_to_end_core_compute_seconds"] == 9.5
    assert timing["cold_end_to_end_diagnostic_inclusive_compute_seconds"] == 10.25
    assert timing["cold_end_to_end_compute_seconds"] == 9.5
    assert timing["cold_end_to_end_process_seconds"] == 12.5


def test_raw_and_canonical_pca_have_unambiguous_variant_labels_and_cli(tmp_path):
    args = parse_args(
        [
            "--data",
            str(tmp_path / "input.npy"),
            "--prepasses",
            "direct",
            "pca_raw",
            "pca",
            "ae",
        ]
    )

    assert args.prepasses == ["direct", "pca_raw", "pca", "ae"]
    assert PREPASS_VARIANTS["pca_raw"] == "pca_raw_scores"
    assert PREPASS_VARIANTS["pca"] == "pca_whitened_row_l2"
    assert PREPASS_VARIANTS["ae"] == "ae_whitened_row_l2"


def test_pca_core_timing_is_fit_transform_and_diagnostics_are_separate(tmp_path):
    rng = np.random.default_rng(21)
    source = rng.normal(size=(80, 7)).astype(np.float32)
    source_path = tmp_path / "source.npy"
    raw_path = tmp_path / "scores.npy"
    np.save(source_path, source)

    details = _run_pca_prepass(
        {
            "input": str(source_path),
            "raw_output": str(raw_path),
            "width": 3,
            "seed": 5,
            "metric_block_rows": 17,
        },
        cupy=None,
    )

    assert np.load(raw_path).shape == (80, 3)
    assert details["prepass_core_compute_seconds"] == details["fit_seconds"]
    assert details["prepass_compute_seconds"] == details["fit_seconds"]
    assert details["reconstruction_diagnostic_seconds"] == details["reconstruction_metric_seconds"]
    assert details["prepass_diagnostic_inclusive_seconds"] == pytest.approx(
        details["fit_seconds"] + details["reconstruction_metric_seconds"]
    )


def test_configuration_identity_is_order_independent_and_change_sensitive():
    first = {"method": "dire", "nested": {"width": 32, "seed": 0}}
    reordered = {"nested": {"seed": 0, "width": 32}, "method": "dire"}
    changed = {"nested": {"seed": 1, "width": 32}, "method": "dire"}

    assert configuration_identity(first) == configuration_identity(reordered)
    assert configuration_identity(first) != configuration_identity(changed)


def test_evaluator_uses_declared_original_space_not_intermediate_space():
    angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    original = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    layout = original.copy()
    permutation = np.random.default_rng(8).permutation(original.shape[0])
    unrelated_reference = original[permutation]

    correct = evaluate_against_original(
        original,
        layout,
        metric_rows=200,
        neighbors=8,
        distance_pairs=1000,
        seed=5,
    )
    wrong = evaluate_against_original(
        unrelated_reference,
        layout,
        metric_rows=200,
        neighbors=8,
        distance_pairs=1000,
        seed=5,
    )

    assert correct["reference_space"] == "original selected input rows"
    assert correct["neighbor_overlap"] == pytest.approx(1.0)
    assert correct["trustworthiness"] > 0.999
    assert wrong["neighbor_overlap"] < 0.15
    assert wrong["trustworthiness"] < 0.65


def test_category_metric_is_reported_on_aligned_labels():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 40)
    original = np.column_stack((labels * 10.0, rng.normal(scale=0.1, size=labels.size))).astype(
        np.float32
    )
    metrics = evaluate_against_original(
        original,
        original,
        metric_rows=120,
        neighbors=5,
        distance_pairs=500,
        seed=9,
        labels=labels,
    )

    assert metrics["category_classes"] == 3
    assert metrics["category_accuracy"] == pytest.approx(1.0)
    assert metrics["category_balanced_accuracy"] == pytest.approx(1.0)


def test_object_label_npy_requires_explicit_trusted_pickle_opt_in(tmp_path):
    path = tmp_path / "labels.npy"
    labels = np.array(["a", "b", "c"], dtype=object)
    np.save(path, labels, allow_pickle=True)

    with pytest.raises(ValueError, match="allow-pickled-labels"):
        _load_labels(path, allow_pickled=False, expected_rows=3)
    loaded = _load_labels(path, allow_pickled=True, expected_rows=3)
    assert np.array_equal(loaded, labels)


def test_deterministic_subset_materialization_and_identity(tmp_path):
    data = np.arange(400, dtype=np.float32).reshape(100, 4)
    labels = np.array([f"label-{index % 4}" for index in range(100)], dtype="U8")
    data_path = tmp_path / "data.npy"
    label_path = tmp_path / "labels.npy"
    np.save(data_path, data)
    np.save(label_path, labels)
    args = SimpleNamespace(
        data=data_path,
        labels=label_path,
        rows=40,
        subset_seed=13,
        output_dir=tmp_path / "output",
        allow_pickled_labels=False,
        force=False,
        block_rows=7,
    )

    prepared = prepare_subset(args)
    selected = deterministic_indices(100, 40, 13)
    assert np.array_equal(np.load(prepared.data_path), data[selected])
    assert np.array_equal(np.load(prepared.labels_path), labels[selected])
    assert prepared.data_logical_sha256 == prepare_subset(args).data_logical_sha256
