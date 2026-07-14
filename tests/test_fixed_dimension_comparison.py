"""Unit tests for the fixed-target reducer comparison harness."""

import numpy as np
import pytest
from experiments.compare_fixed_dimension import (
    _decoder_probe_nmse,
    _sample_pairs,
    evaluate_embedding,
    fit_autoencoder,
    make_dataset,
    preprocess,
    summarize_rows,
)
from sklearn.manifold import trustworthiness


def test_preprocess_preserves_global_geometry_up_to_translation_and_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 7)) * np.arange(1, 8)
    transformed = preprocess(X)

    assert transformed.dtype == np.float32
    assert np.max(np.abs(transformed.mean(axis=0))) < 1e-6
    assert np.sqrt(np.mean(transformed.astype(np.float64) ** 2)) == pytest.approx(1.0)
    before = np.linalg.norm(X[0] - X[1]) / np.linalg.norm(X[2] - X[3])
    after = np.linalg.norm(transformed[0] - transformed[1]) / np.linalg.norm(
        transformed[2] - transformed[3]
    )
    assert after == pytest.approx(before, rel=1e-6)


def test_identity_embedding_has_perfect_shared_metrics():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 5))
    pairs = _sample_pairs(X.shape[0], 2000, seed=3)
    metrics = evaluate_embedding(
        X,
        X.copy(),
        labels=None,
        latent=X.copy(),
        intrinsic_metric="euclidean",
        n_neighbors=10,
        pairs=pairs,
        seed=3,
    )

    assert metrics["trustworthiness"] == pytest.approx(1.0)
    assert metrics["continuity"] == pytest.approx(1.0)
    assert metrics["neighbor_overlap"] == pytest.approx(1.0)
    assert metrics["distance_spearman"] == pytest.approx(1.0)
    assert metrics["normalized_stress"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["latent_procrustes_disparity"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["intrinsic_distance_spearman"] == pytest.approx(1.0)
    assert metrics["intrinsic_normalized_stress"] == pytest.approx(0.0, abs=1e-12)


def test_continuity_direction_and_label_probe_are_rotation_invariant():
    rng = np.random.default_rng(9)
    embedding = rng.normal(size=(180, 3))
    X = np.column_stack(
        (
            embedding,
            embedding[:, 0] ** 2,
            np.sin(embedding[:, 1]),
        )
    )
    labels = (embedding[:, 0] + 0.4 * embedding[:, 1] > 0).astype(int)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    pairs = _sample_pairs(X.shape[0], 3000, seed=9)
    original = evaluate_embedding(
        X,
        embedding,
        labels=labels,
        latent=None,
        intrinsic_metric=None,
        n_neighbors=10,
        pairs=pairs,
        seed=9,
    )
    rotated = evaluate_embedding(
        X,
        embedding @ rotation,
        labels=labels,
        latent=None,
        intrinsic_metric=None,
        n_neighbors=10,
        pairs=pairs,
        seed=9,
    )

    assert original["continuity"] == pytest.approx(trustworthiness(embedding, X, n_neighbors=10))
    assert rotated["continuity"] == pytest.approx(original["continuity"])
    assert rotated["knn_accuracy"] == pytest.approx(original["knn_accuracy"])


def test_whitened_decoder_probe_is_invariant_to_linear_latent_coordinates():
    rng = np.random.default_rng(10)
    embedding = rng.normal(size=(200, 3))
    X = np.column_stack((embedding, embedding[:, 0] * embedding[:, 1]))
    transform = np.array([[3.0, 1.0, 0.2], [0.0, 0.5, 0.3], [0.0, 0.0, 2.0]])

    reference = _decoder_probe_nmse(X, embedding, seed=10, n_neighbors=12)
    transformed = _decoder_probe_nmse(X, embedding @ transform, seed=10, n_neighbors=12)
    assert transformed == pytest.approx(reference, rel=1e-10, abs=1e-12)


def test_degenerate_embedding_is_rejected():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 4))
    with pytest.raises(ValueError, match="degenerate"):
        evaluate_embedding(
            X,
            np.zeros((80, 2)),
            labels=None,
            latent=None,
            intrinsic_metric=None,
            n_neighbors=8,
            pairs=_sample_pairs(80, 1000, seed=11),
            seed=11,
        )


@pytest.mark.parametrize("name", ["linear", "swiss-roll", "sphere", "torus", "digits"])
def test_dataset_generators_are_aligned(name):
    dataset = make_dataset(
        name,
        n_samples=120,
        ambient_dimension=12,
        target_dimension=2,
        seed=4,
        noise=0.01,
    )

    assert dataset.X.shape == (120, 12 if name != "digits" else 64)
    if dataset.labels is not None:
        assert dataset.labels.shape == (120,)
    if dataset.latent is not None:
        assert dataset.latent.shape[0] == 120


def test_fixed_autoencoder_returns_embedding_and_restores_rng():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(100, 2))
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    X = preprocess(latent @ basis.T)
    torch.default_generator.manual_seed(123)
    caller_state = torch.default_generator.get_state().clone()

    embedding, elapsed, native_nmse, details = fit_autoencoder(
        X,
        target_dimension=2,
        seed=5,
        torch=torch,
        device="cpu",
        hidden_dimensions=(12, 6),
        epochs=20,
        patience=5,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )

    assert embedding.shape == (100, 2)
    assert np.all(np.isfinite(embedding))
    assert elapsed > 0
    assert np.isfinite(native_nmse)
    assert details["selected_epochs"] >= 1
    assert torch.equal(torch.default_generator.get_state(), caller_state)


def test_summary_uses_metric_means_and_runtime_median():
    rows = []
    for seed, fit_seconds, trust in [(0, 10.0, 0.8), (1, 1.0, 1.0), (2, 2.0, 0.9)]:
        row = {
            "dataset": "demo",
            "method": "method",
            "seed": seed,
            "status": "ok",
            "fit_seconds": fit_seconds,
            "trustworthiness": trust,
        }
        for metric in (
            "continuity",
            "neighbor_overlap",
            "distance_spearman",
            "normalized_stress",
            "decoder_probe_nmse",
            "knn_accuracy",
            "balanced_accuracy",
            "latent_procrustes_disparity",
            "native_reconstruction_nmse",
        ):
            row[metric] = None
        rows.append(row)

    summary = summarize_rows(rows)[0]
    assert summary["fit_seconds_median"] == pytest.approx(2.0)
    assert summary["trustworthiness_mean"] == pytest.approx(0.9)
    assert summary["successful_runs"] == 3
