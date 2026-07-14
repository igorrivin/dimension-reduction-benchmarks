"""CPU-only tests for the scale-safe large-topology evaluator."""

from argparse import Namespace

import numpy as np
import pytest
from experiments.evaluate_large_topology import (
    TopologyBackend,
    aggregate_records,
    evaluate_seed,
    finite_h0_persistences,
    gini,
    index_sha256,
    islandness,
    max_pairwise_distance,
    rescale_unit_diameter,
    run,
    sample_indices,
    sha256_file,
)


def _mock_backend(calls=None):
    calls = [] if calls is None else calls

    def curve(points, *, n_steps, maxdim):
        calls.append(("curve", np.asarray(points).copy(), n_steps, maxdim))
        n = points.shape[0]
        return {
            "beta_0": np.array([n, max(1, n // 2), 1], dtype=float),
            "beta_1": np.array([0, 2, 0], dtype=float),
        }

    def persistence(points, *, maxdim):
        calls.append(("persistence", np.asarray(points).copy(), None, maxdim))
        diagram = np.array(
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.4], [0.0, np.inf]],
            dtype=float,
        )
        return {"dgms": [diagram, np.empty((0, 2))]}

    def dtw(left, right):
        return float(np.abs(np.asarray(left) - np.asarray(right)).sum()), []

    return TopologyBackend(
        curve=curve,
        persistence=persistence,
        dtw=dtw,
        provenance={"backend": "mock"},
    )


def test_uniform_and_stratified_samples_are_exact_and_reproducible():
    uniform = sample_indices(100, 23, seed=4)
    assert uniform.shape == (23,)
    assert np.array_equal(uniform, sample_indices(100, 23, seed=4))
    assert not np.array_equal(uniform, sample_indices(100, 23, seed=5))

    labels = np.array(["rare"] * 2 + ["medium"] * 18 + ["common"] * 80)
    stratified = sample_indices(100, 20, seed=7, labels=labels)
    sampled_labels = labels[stratified]
    assert stratified.shape == (20,)
    assert set(sampled_labels) == {"rare", "medium", "common"}
    assert np.unique(stratified).size == stratified.size
    assert index_sha256(stratified) == index_sha256(stratified.astype(np.int32))


def test_unit_diameter_scaling_is_exact_across_blocks():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [10.0, 0.0]])
    expected = float(pdist_for_test(points).max())

    assert max_pairwise_distance(points, block_size=2) == pytest.approx(expected)
    scaled, diameter = rescale_unit_diameter(points, block_size=2)
    assert diameter == pytest.approx(expected)
    assert scaled.dtype == np.float32
    assert max_pairwise_distance(scaled, block_size=2) == pytest.approx(1.0, rel=1e-6)


def pdist_for_test(points):
    differences = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(differences, axis=2)


def test_islandness_uses_only_positive_finite_h0_bars():
    diagram = np.array(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [0.0, 0.0],
            [0.0, np.inf],
        ]
    )
    bars = finite_h0_persistences(diagram)
    metrics = islandness(diagram)

    assert np.array_equal(bars, np.arange(1.0, 6.0))
    assert metrics["longest_h0"] == pytest.approx(5.0)
    assert metrics["top5_median_h0"] == pytest.approx(1.0)
    assert metrics["gini_h0"] == pytest.approx(gini(np.arange(1.0, 6.0)))
    assert metrics["finite_h0_bars"] == 5


def test_evaluate_seed_reuses_one_scaled_cloud_per_source_and_method():
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(30, 4))
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    indices = sample_indices(30, 15, seed=2)
    calls = []
    backend = _mock_backend(calls)

    records = evaluate_seed(
        reference,
        {"identity": reference.copy(), "scaled": 7.0 * reference},
        indices,
        2,
        backend,
        n_steps=17,
        diameter_block_size=4,
    )

    assert [record["method"] for record in records] == ["identity", "scaled"]
    for record in records:
        assert record["dtw_beta0"] == pytest.approx(0.0)
        assert record["dtw_beta1"] == pytest.approx(0.0)
        assert record["layout_longest_h0"] == pytest.approx(record["reference_longest_h0"])
    # The curve and persistence calls for a cloud must receive the exact same scaled points.
    for curve_call, persistence_call in zip(calls[::2], calls[1::2]):
        assert curve_call[0] == "curve"
        assert persistence_call[0] == "persistence"
        assert np.array_equal(curve_call[1], persistence_call[1])
        assert max_pairwise_distance(curve_call[1], block_size=4) == pytest.approx(1.0, rel=1e-6)


def test_aggregate_uses_sample_standard_deviation():
    records = []
    for seed, value in enumerate([1.0, 2.0, 3.0]):
        record = {"seed": seed, "method": "demo"}
        for metric in (
            "dtw_beta0",
            "dtw_beta1",
            "reference_longest_h0",
            "reference_top5_median_h0",
            "reference_gini_h0",
            "layout_longest_h0",
            "layout_top5_median_h0",
            "layout_gini_h0",
        ):
            record[metric] = value
        records.append(record)

    row = aggregate_records(records)[0]
    assert row["n_seeds"] == 3
    assert row["dtw_beta0_mean"] == pytest.approx(2.0)
    assert row["dtw_beta0_std"] == pytest.approx(1.0)


def test_run_writes_metrics_samples_and_complete_hash_provenance(tmp_path):
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(24, 5)).astype(np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    layout_a = reference[:, :2].copy()
    layout_b = reference[:, 2:4].copy()
    source_indices = np.arange(0, 48, 2, dtype=np.int64)
    source_labels = np.array([f"class-{index % 3}" for index in range(48)], dtype=object)

    input_path = tmp_path / "prepared.npy"
    layout_a_path = tmp_path / "a.npy"
    layout_b_path = tmp_path / "b.npy"
    source_path = tmp_path / "source.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(input_path, reference)
    np.save(layout_a_path, layout_a)
    np.save(layout_b_path, layout_b)
    np.save(source_path, source_indices)
    np.save(labels_path, source_labels)
    output_dir = tmp_path / "results"

    args = Namespace(
        input=str(input_path),
        layout=[f"method-a={layout_a_path}", f"method-b={layout_b_path}"],
        labels=str(labels_path),
        label_column=None,
        labels_alignment="source",
        source_indices=str(source_path),
        sample_size=12,
        seeds=[0, 1],
        n_steps=8,
        diameter_block_size=4,
        l2_atol=1e-5,
        dire_repo="unused",
        expected_dire_revision=None,
        expected_betti_sha256=None,
        output_dir=str(output_dir),
    )
    payload = run(args, backend=_mock_backend())

    assert len(payload["records"]) == 4
    assert len(payload["aggregate"]) == 2
    assert payload["configuration"]["sampling"] == "stratified"
    assert payload["provenance"]["reference"]["sha256"] == sha256_file(input_path)
    assert payload["provenance"]["labels"]["alignment"] == "source"
    for filename in (
        "raw.json",
        "raw.csv",
        "aggregate.csv",
        "sample_indices_seed0.npy",
        "sample_indices_seed1.npy",
        "sample_source_indices_seed0.npy",
        "sample_source_indices_seed1.npy",
    ):
        assert (output_dir / filename).is_file()


def test_run_rejects_reference_that_is_not_row_l2_normalized(tmp_path):
    reference = np.arange(60, dtype=np.float32).reshape(20, 3) + 1.0
    layout = reference[:, :2]
    input_path = tmp_path / "prepared.npy"
    layout_path = tmp_path / "layout.npy"
    np.save(input_path, reference)
    np.save(layout_path, layout)
    args = Namespace(
        input=str(input_path),
        layout=[f"demo={layout_path}"],
        labels=None,
        label_column=None,
        labels_alignment="auto",
        source_indices=None,
        sample_size=10,
        seeds=[0],
        n_steps=8,
        diameter_block_size=4,
        l2_atol=1e-5,
        dire_repo="unused",
        expected_dire_revision=None,
        expected_betti_sha256=None,
        output_dir=str(tmp_path / "results"),
    )

    with pytest.raises(ValueError, match="not row-L2 normalized"):
        run(args, backend=_mock_backend())
