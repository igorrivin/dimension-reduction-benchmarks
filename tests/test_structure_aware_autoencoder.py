"""Focused tests for the bounded structure-aware autoencoder pilot."""

import numpy as np
import pytest
from experiments.compare_fixed_dimension import make_dataset, preprocess
from experiments.compare_structure_aware_autoencoder import (
    PROTOCOL_COMPARABILITY,
    build_structure_graph,
    evaluate_full_and_complement,
    fit_fixed_epoch_autoencoder,
    scale_invariant_rank_loss,
    scale_invariant_stress_loss,
    summarize_rows,
)


def _distance(X, first, second):
    return float(np.linalg.norm(np.asarray(X[first]) - np.asarray(X[second])))


def test_structure_graph_is_deterministic_and_excludes_ties_and_positives():
    # Interior points have two exactly tied nearest candidates. With k=1, the unused
    # member of that tie must not be treated as a negative for the selected positive.
    X = np.column_stack((np.arange(24, dtype=np.float32), np.zeros(24, dtype=np.float32)))
    first = build_structure_graph(
        X, reservoir_size=24, n_neighbors=1, stress_pair_count=80, seed=11
    )
    second = build_structure_graph(
        X, reservoir_size=24, n_neighbors=1, stress_pair_count=80, seed=11
    )

    assert np.array_equal(first.reservoir_indices, second.reservoir_indices)
    assert np.array_equal(first.positive_indices, second.positive_indices)
    assert np.array_equal(first.triplets, second.triplets)
    assert np.array_equal(first.stress_pairs, second.stress_pairs)
    assert first.reservoir_sha256 == second.reservoir_sha256
    assert first.triplet_sha256 == second.triplet_sha256
    assert first.pair_sha256 == second.pair_sha256
    assert first.tie_filtered_candidates > 0

    positive_by_anchor = {
        int(anchor): set(map(int, positives))
        for anchor, positives in zip(first.reservoir_indices, first.positive_indices, strict=True)
    }
    for anchor, positive, near, uniform in first.triplets:
        assert len({int(anchor), int(positive), int(near), int(uniform)}) == 4
        assert int(positive) in positive_by_anchor[int(anchor)]
        assert int(near) not in positive_by_anchor[int(anchor)]
        assert int(uniform) not in positive_by_anchor[int(anchor)]
        positive_distance = _distance(X, anchor, positive)
        assert _distance(X, anchor, near) > positive_distance
        assert _distance(X, anchor, uniform) > positive_distance


def test_structure_graph_is_invariant_to_similarity_transforms():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    transformed = (3.0 * X @ rotation + np.arange(5)).astype(np.float32)

    reference = build_structure_graph(
        X, reservoir_size=48, n_neighbors=5, stress_pair_count=100, seed=17
    )
    changed = build_structure_graph(
        transformed, reservoir_size=48, n_neighbors=5, stress_pair_count=100, seed=17
    )

    assert np.array_equal(reference.reservoir_indices, changed.reservoir_indices)
    assert np.array_equal(reference.positive_indices, changed.positive_indices)
    assert np.array_equal(reference.triplets, changed.triplets)
    assert np.array_equal(reference.stress_pairs, changed.stress_pairs)
    assert changed.stress_target_distances == pytest.approx(
        3.0 * reference.stress_target_distances, rel=2e-6, abs=2e-6
    )


def test_sparse_losses_have_finite_gradients_and_similarity_invariance():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(9)
    points = torch.randn(12, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    rank = scale_invariant_rank_loss(
        torch, points[:, 0], points[:, 1], points[:, 2:], temperature=0.2
    )
    rank.backward()
    assert torch.isfinite(rank)
    assert points.grad is not None
    assert torch.all(torch.isfinite(points.grad))
    assert torch.count_nonzero(points.grad) > 0

    with torch.no_grad():
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
        transformed = 7.0 * points.detach() @ rotation + torch.tensor(
            [2.0, -3.0, 5.0], dtype=torch.float64
        )
        changed = scale_invariant_rank_loss(
            torch,
            transformed[:, 0],
            transformed[:, 1],
            transformed[:, 2:],
            temperature=0.2,
        )
    assert changed.item() == pytest.approx(rank.item(), rel=1e-12, abs=1e-12)

    first = points.detach()[:, 0].clone().requires_grad_(True)
    second = points.detach()[:, 1]
    target = torch.linalg.vector_norm(points.detach()[:, 2] - points.detach()[:, 3], dim=1)
    stress = scale_invariant_stress_loss(torch, first, second, target)
    stress.backward()
    assert torch.isfinite(stress)
    assert first.grad is not None and torch.all(torch.isfinite(first.grad))
    changed_stress = scale_invariant_stress_loss(torch, 5.0 * first, 5.0 * second, target)
    assert changed_stress.item() == pytest.approx(stress.item(), rel=1e-12, abs=1e-12)


def test_zero_weight_structure_graph_exactly_matches_plain_baseline():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    X = preprocess(rng.normal(size=(72, 6)))
    graph = build_structure_graph(X, reservoir_size=24, n_neighbors=3, stress_pair_count=50, seed=8)
    kwargs = dict(
        torch=torch,
        device="cpu",
        hidden_dimensions=(10, 5),
        epochs=3,
        batch_size=18,
        learning_rate=1e-3,
        weight_decay=1e-5,
        structure_interval=2,
        structure_batch_size=4,
        stress_batch_size=6,
    )
    plain_embedding, plain_reconstruction, plain = fit_fixed_epoch_autoencoder(
        X, 2, 23, structure_graph=None, **kwargs
    )
    zero_embedding, zero_reconstruction, zero = fit_fixed_epoch_autoencoder(
        X,
        2,
        23,
        structure_graph=graph,
        rank_weight=0.0,
        stress_weight=0.0,
        **kwargs,
    )

    assert np.array_equal(zero_embedding, plain_embedding)
    assert np.array_equal(zero_reconstruction, plain_reconstruction)
    assert zero["mean_reconstruction_training_mse"] == pytest.approx(
        plain["mean_reconstruction_training_mse"], rel=0, abs=0
    )
    assert zero["rank_events"] == zero["stress_events"] == 0
    assert zero["maximum_auxiliary_rows_per_update"] == 0


def test_rank_training_is_sparse_and_never_calls_torch_cdist(monkeypatch):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    X = preprocess(rng.normal(size=(96, 8)))
    graph = build_structure_graph(
        X, reservoir_size=32, n_neighbors=3, stress_pair_count=60, seed=12
    )

    def forbidden_cdist(*args, **kwargs):
        raise AssertionError("training must not form a dense pairwise tensor")

    monkeypatch.setattr(torch, "cdist", forbidden_cdist)
    embedding, reconstruction, details = fit_fixed_epoch_autoencoder(
        X,
        2,
        31,
        torch=torch,
        device="cpu",
        hidden_dimensions=(12, 6),
        epochs=2,
        batch_size=24,
        learning_rate=1e-3,
        weight_decay=1e-5,
        structure_graph=graph,
        rank_weight=0.1,
        structure_interval=2,
        structure_batch_size=5,
        stress_batch_size=7,
    )

    assert embedding.shape == (96, 2)
    assert reconstruction.shape == X.shape
    assert details["rank_events"] == 4
    assert details["maximum_auxiliary_rows_per_update"] == 4 * 5
    assert np.isfinite(details["mean_rank_loss_on_active_updates"])
    # The graph stores only sparse bounded arrays; it does not retain its temporary
    # reservoir-distance result or anything with a corpus-by-corpus shape.
    assert graph.triplets.shape[1] == 4
    assert graph.positive_indices.shape == (32, 3)


def test_common_and_complement_metrics_are_both_reported():
    dataset = make_dataset(
        "linear",
        n_samples=100,
        ambient_dimension=6,
        target_dimension=2,
        seed=5,
        noise=0.01,
    )
    X = preprocess(dataset.X)
    # This is a deterministic non-degenerate layout; model fitting is irrelevant to
    # testing the evaluation split and provenance here.
    embedding = np.ascontiguousarray(dataset.latent, dtype=np.float32)
    graph = build_structure_graph(X, reservoir_size=24, n_neighbors=4, stress_pair_count=40, seed=5)
    metrics, provenance = evaluate_full_and_complement(
        X,
        embedding,
        dataset,
        graph,
        n_neighbors=4,
        distance_pair_count=300,
        seed=5,
    )

    assert "neighbor_overlap" in metrics
    assert "complement_neighbor_overlap" in metrics
    assert "trustworthiness" in metrics
    assert "complement_trustworthiness" in metrics
    assert provenance["complement_rows"] == 76
    assert len(provenance["full_metric_pair_sha256"]) == 64
    assert len(provenance["complement_metric_pair_sha256"]) == 64

    row = {
        "dataset": "linear",
        "method": "autoencoder_mse",
        **metrics,
    }
    summary = summarize_rows([row])[0]
    assert summary["intrinsic_distance_spearman_mean"] == pytest.approx(
        metrics["intrinsic_distance_spearman"]
    )
    assert summary["intrinsic_normalized_stress_mean"] == pytest.approx(
        metrics["intrinsic_normalized_stress"]
    )
    assert summary["complement_intrinsic_distance_spearman_mean"] == pytest.approx(
        metrics["complement_intrinsic_distance_spearman"]
    )
    assert summary["complement_intrinsic_normalized_stress_mean"] == pytest.approx(
        metrics["complement_intrinsic_normalized_stress"]
    )
    assert "not directly comparable" in PROTOCOL_COMPARABILITY
