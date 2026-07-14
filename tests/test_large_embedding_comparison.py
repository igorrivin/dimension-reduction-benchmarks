"""CPU tests for the scale-safe large-embedding comparison harness."""

import json
import sys
from types import SimpleNamespace

import experiments.compare_large_embeddings as comparison
import numpy as np
import pytest
from experiments.compare_large_embeddings import (
    PreparedInfo,
    _ae_precision_configuration,
    _build_autoencoder,
    _check_revision,
    _run_subprocess,
    _sha256_file,
    _train_auto_epochs_device,
    _validate_ae_precision_support,
    _worker_command,
    deterministic_nested_indices,
    deterministic_sample,
    discover_sources,
    distance_fidelity,
    fit_autoencoder_large,
    fit_run_identity,
    fit_umap_large,
    group_neighbor_metrics,
    index_sha256,
    load_labels,
    local_decodability_nmse,
    neighbors_without_self,
    parse_args,
    prepare_normalized_data,
    result_failed,
    reusable_layout_matches,
    row_l2_normalize,
    sample_conditional_metrics,
    sample_pairs,
    sampled_pair_distances,
)


def test_nested_samples_are_deterministic_sorted_and_nested():
    first = deterministic_nested_indices(100, [75, 10, 30], seed=4)
    second = deterministic_nested_indices(100, [10, 30, 75], seed=4)

    assert all(np.array_equal(first[size], second[size]) for size in first)
    assert all(np.all(np.diff(indices) > 0) for indices in first.values())
    assert set(first[10]).issubset(first[30])
    assert set(first[30]).issubset(first[75])
    assert index_sha256(first[30]) == index_sha256(second[30])
    assert index_sha256(first[10]) != index_sha256(first[30])


def test_nested_sample_bounds_are_validated():
    with pytest.raises(ValueError, match="sizes"):
        deterministic_nested_indices(10, [1, 5], seed=0)
    with pytest.raises(ValueError, match="sizes"):
        deterministic_nested_indices(10, [11], seed=0)


def test_row_normalization_is_float32_finite_and_zero_safe():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1e20, -1e20]], dtype=np.float32)
    normalized, zeros = row_l2_normalize(X, return_zero_count=True)

    assert normalized.dtype == np.float32
    assert zeros == 1
    assert np.linalg.norm(normalized[0]) == pytest.approx(1.0)
    assert np.array_equal(normalized[1], [0.0, 0.0])
    assert np.linalg.norm(normalized[2]) == pytest.approx(1.0, rel=1e-6)
    assert np.all(np.isfinite(normalized))


def test_prepare_data_handles_aligned_shards_and_records_hash(tmp_path):
    first = np.arange(30, dtype=np.float32).reshape(10, 3) + 1
    second = np.arange(30, 60, dtype=np.float32).reshape(10, 3) + 1
    np.save(tmp_path / "emb_0000.npy", first)
    np.save(tmp_path / "emb_0001.npy", second)
    paths, source = discover_sources(tmp_path)
    selected = np.array([0, 3, 9, 10, 14, 19])

    info = prepare_normalized_data(paths, selected, tmp_path / "prepared.npy", block_rows=2)
    prepared = np.load(info.path)
    expected = row_l2_normalize(np.vstack((first, second))[selected])

    assert source.rows == 20
    assert source.features == 3
    assert np.allclose(prepared, expected)
    assert info.source_index_sha256 == index_sha256(selected)
    assert len(info.normalized_sha256) == 64


def test_discover_sources_rejects_misaligned_shards(tmp_path):
    np.save(tmp_path / "emb_0000.npy", np.ones((3, 2), dtype=np.float32))
    np.save(tmp_path / "emb_0001.npy", np.ones((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="identical"):
        discover_sources(tmp_path)


def test_load_labels_accepts_aligned_parquet_shard_directory(tmp_path):
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"url": ["a", "a", "b"]}).to_parquet(tmp_path / "meta-00.parquet")
    pd.DataFrame({"url": ["c", "c"]}).to_parquet(tmp_path / "meta-01.parquet")

    labels = load_labels(tmp_path, "url", 5, "meta-*.parquet")
    assert labels.tolist() == ["a", "a", "b", "c", "c"]
    with pytest.raises(ValueError, match="aligned"):
        load_labels(tmp_path, "url", 6, "meta-*.parquet")


def test_sampled_pair_distances_match_dense_reference_for_unit_rows():
    rng = np.random.default_rng(2)
    X = row_l2_normalize(rng.normal(size=(80, 7)))
    layout = rng.normal(size=(80, 2)).astype(np.float32)
    pairs = sample_pairs(80, 500, seed=2)
    high, low = sampled_pair_distances(X, layout, pairs, block_rows=17)

    expected_high = np.linalg.norm(X[pairs[0]] - X[pairs[1]], axis=1)
    expected_low = np.linalg.norm(layout[pairs[0]] - layout[pairs[1]], axis=1)
    assert np.allclose(high, expected_high, atol=2e-6)
    assert np.allclose(low, expected_low)
    rho, stress = distance_fidelity(high, high.copy())
    assert rho == pytest.approx(1.0)
    assert stress == pytest.approx(0.0, abs=1e-12)

    X[0] = 0.0
    zero_distance, _ = sampled_pair_distances(
        X, layout, (np.array([0]), np.array([1])), block_rows=1
    )
    assert zero_distance[0] == pytest.approx(1.0, rel=1e-6)


def test_identity_sample_metrics_are_perfect_and_self_is_excluded():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 5)).astype(np.float32)
    indices = np.arange(100)
    neighbors = neighbors_without_self(X, 8)
    metrics = sample_conditional_metrics(X, X.copy(), indices, 8)

    assert not np.any(neighbors == np.arange(100)[:, None])
    assert metrics["sample_conditional_trustworthiness"] == pytest.approx(1.0)
    assert metrics["sample_conditional_continuity"] == pytest.approx(1.0)
    assert metrics["sample_conditional_neighbor_overlap"] == pytest.approx(1.0)


def test_local_decodability_is_invariant_to_linear_layout_coordinates():
    rng = np.random.default_rng(5)
    layout = rng.normal(size=(180, 3))
    X = np.column_stack((layout, layout[:, 0] * layout[:, 1])).astype(np.float32)
    transform = np.array([[3.0, 1.0, 0.2], [0.0, 0.5, 0.3], [0.0, 0.0, 2.0]])
    indices = np.arange(180)

    original = local_decodability_nmse(X, layout, indices, 12, seed=5)
    changed = local_decodability_nmse(X, layout @ transform, indices, 12, seed=5)
    assert changed == pytest.approx(original, rel=1e-9, abs=1e-12)


def test_stratified_metric_sample_is_deterministic():
    labels = np.repeat(np.arange(4), [50, 30, 15, 5])
    first = deterministic_sample(100, 40, seed=8, labels=labels)
    second = deterministic_sample(100, 40, seed=8, labels=labels)
    assert np.array_equal(first, second)
    assert set(labels[first]) == {0, 1, 2, 3}


def test_group_neighbor_metrics_excludes_singletons_and_measures_retrieval():
    # Each repeated group is adjacent; singleton z must not count as a failure. Only
    # one member of each pair is an anchor, proving retrieval uses the full layout.
    layout = np.array([[0.0], [0.1], [10.0], [10.1], [100.0]], dtype=np.float32)
    labels = np.array(["a", "a", "b", "b", "z"])
    metrics = group_neighbor_metrics(layout, labels, np.array([0, 2, 4]), n_neighbors=1)

    assert metrics["group_eligible_rows"] == 2
    assert metrics["group_neighbor_purity"] == pytest.approx(1.0)
    assert metrics["group_neighbor_hit_rate"] == pytest.approx(1.0)
    assert metrics["group_neighbor_recall"] == pytest.approx(1.0)


def test_autoencoder_cpu_protocol_selects_then_refits_and_batches_output(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    X = row_l2_normalize(rng.normal(size=(40, 6)))
    data_path = tmp_path / "prepared.npy"
    np.save(data_path, X)
    writable = np.load(data_path, mmap_mode="r+")
    layout_path = tmp_path / "ae-layout.npy"
    args = SimpleNamespace(
        device="cpu",
        ae_hidden_dimensions=(10, 4),
        worker_seed=12,
        worker_target_dimension=2,
        ae_validation=0.2,
        ae_epochs=3,
        ae_patience=2,
        ae_min_delta=1e-5,
        ae_batch_size=8,
        ae_learning_rate=1e-3,
        ae_weight_decay=1e-5,
        ae_amp=False,
        ae_evaluation_precision="fp32",
    )

    returned, details = fit_autoencoder_large(writable, args, layout_path)
    layout = np.load(layout_path)
    assert returned is None
    assert layout.shape == (40, 2)
    assert np.all(np.isfinite(layout))
    assert 1 <= details["selected_epochs"] <= 3
    assert details["staging_seconds"] == 0.0
    assert details["staged_input_bytes"] == 0
    assert details["training_pipeline"] == "CPU DataLoader fallback"
    assert details["selection_seconds"] > 0
    assert details["refit_seconds"] > 0
    assert details["training_precision"] == "float32"
    assert details["evaluation_precision"] == "float32"
    assert details["validation_precision"] == "float32"
    assert details["encoding_precision"] == "float32"
    charged = sum(
        details[key]
        for key in (
            "staging_seconds",
            "selection_seconds",
            "refit_seconds",
            "encode_reconstruction_seconds",
            "native_metric_finalize_seconds",
        )
    )
    assert details["fit_seconds"] == pytest.approx(charged)
    assert "fresh full-data refit" in details["protocol"]


def test_resident_tensor_trainer_uses_device_indices_without_dataloader():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    data = torch.from_numpy(row_l2_normalize(rng.normal(size=(36, 6))))
    train = torch.arange(6, 36, dtype=torch.long)
    validation = torch.arange(0, 6, dtype=torch.long)
    args = SimpleNamespace(
        ae_learning_rate=1e-3,
        ae_weight_decay=1e-5,
        ae_batch_size=9,
        ae_epochs=3,
        ae_patience=2,
        ae_min_delta=1e-5,
        ae_amp=False,
        ae_evaluation_precision="fp32",
    )
    torch.manual_seed(13)
    model = _build_autoencoder(torch, 6, 2, (8, 4))

    selected, validation_mse = _train_auto_epochs_device(
        torch,
        model,
        data,
        train,
        validation_indices=validation,
        args=args,
        seed=13,
        device=torch.device("cpu"),
    )

    assert 1 <= selected <= 3
    assert np.isfinite(validation_mse)
    assert data.device.type == train.device.type == validation.device.type == "cpu"


def test_amp_training_defaults_to_fp32_validation_and_encoding(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    observed_precisions = []

    def record_precision(precision):
        observed_precisions.append(precision)
        # Exercise the dispatch without requiring CPU BF16 kernels.
        return False

    monkeypatch.setattr(comparison, "_uses_bfloat16_autocast", record_precision)
    rng = np.random.default_rng(14)
    X = row_l2_normalize(rng.normal(size=(24, 5)))
    data_path = tmp_path / "prepared.npy"
    np.save(data_path, X)
    writable = np.load(data_path, mmap_mode="r+")
    args = SimpleNamespace(
        device="cpu",
        ae_hidden_dimensions=(8, 4),
        worker_seed=14,
        worker_target_dimension=2,
        ae_validation=0.25,
        ae_epochs=1,
        ae_patience=1,
        ae_min_delta=1e-5,
        ae_batch_size=8,
        ae_learning_rate=1e-3,
        ae_weight_decay=1e-5,
        ae_amp=True,
        ae_evaluation_precision="fp32",
    )

    _, details = fit_autoencoder_large(writable, args, tmp_path / "layout.npy")

    assert "bfloat16_autocast" in observed_precisions
    assert "float32" in observed_precisions
    assert observed_precisions[-1] == "float32"
    assert details["training_precision"] == "bfloat16_autocast"
    assert details["evaluation_precision"] == "float32"
    assert details["evaluation_precision_policy"] == "fp32"


def test_ae_precision_cli_identity_and_worker_forwarding(tmp_path):
    args = parse_args(["--data", "unused.npy", "--ae-amp"])
    assert args.ae_evaluation_precision == "fp32"
    assert _ae_precision_configuration(args) == ("bfloat16_autocast", "float32")

    prepared = PreparedInfo(
        path=str(tmp_path / "prepared.npy"),
        rows=12,
        features=5,
        source_index_sha256="source-index",
        normalized_sha256="prepared-content",
        source_fingerprint="source",
        zero_rows=0,
        preparation_seconds=0.1,
    )
    identity, digest = fit_run_identity(args, prepared, "autoencoder", 0, 2, 5)
    configuration = identity["configuration"]
    assert configuration["training_precision"] == "bfloat16_autocast"
    assert configuration["evaluation_precision"] == "float32"
    assert configuration["evaluation_precision_policy"] == "fp32"

    command = _worker_command(
        args,
        tmp_path / "prepared.npy",
        "autoencoder",
        0,
        2,
        tmp_path / "layout.npy",
        tmp_path / "result.json",
        "nonce",
    )
    precision_index = command.index("--ae-evaluation-precision")
    assert command[precision_index + 1] == "fp32"
    assert "--ae-amp" in command

    matched = parse_args(
        [
            "--data",
            "unused.npy",
            "--ae-amp",
            "--ae-evaluation-precision",
            "match-training",
        ]
    )
    assert _ae_precision_configuration(matched) == (
        "bfloat16_autocast",
        "bfloat16_autocast",
    )
    matched_identity, matched_digest = fit_run_identity(matched, prepared, "autoencoder", 0, 2, 5)
    assert matched_identity["configuration"]["evaluation_precision"] == ("bfloat16_autocast")
    assert matched_digest != digest


def test_cuda_bfloat16_support_is_validated_without_a_gpu():
    device = SimpleNamespace(type="cuda")
    unsupported_torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: False))
    with pytest.raises(RuntimeError, match="does not report native BF16 support"):
        _validate_ae_precision_support(unsupported_torch, device, "bfloat16_autocast", "float32")

    native_probe = []

    def supports_native_bfloat16(*, including_emulation=True):
        native_probe.append(including_emulation)
        return not including_emulation

    supported_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_bf16_supported=supports_native_bfloat16)
    )
    _validate_ae_precision_support(supported_torch, device, "bfloat16_autocast", "float32")
    assert native_probe == [False]

    # FP32 must not probe the CUDA BF16 capability at all.
    no_probe_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_bf16_supported=lambda: (_ for _ in ()).throw(AssertionError("unexpected probe"))
        )
    )
    _validate_ae_precision_support(no_probe_torch, device, "float32", "float32")


def test_umap_large_never_calls_inverse_transform():
    class Runtime:
        @staticmethod
        def deviceSynchronize():
            return None

    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=Runtime()))

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, X):
            return X[:, :2]

        def inverse_transform(self, _):
            raise AssertionError("large protocol must never call inverse_transform")

    args = SimpleNamespace(
        worker_target_dimension=2,
        neighbors=10,
        umap_min_dist=0.1,
        umap_spread=1.0,
        worker_seed=0,
        umap_build_algo="auto",
    )
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    layout, details = fit_umap_large(X, args, umap_class=FakeUMAP, cupy=fake_cupy)
    assert layout.shape == (20, 2)
    assert details["native_inverse"] == "disabled by large-data protocol"


def test_protocol_rejects_large_brute_force_umap_and_revision_mismatch():
    with pytest.raises(SystemExit):
        parse_args(["--data", "unused.npy", "--umap-build-algo", "brute_force_knn"])
    with pytest.raises(RuntimeError, match="revision mismatch"):
        _check_revision("abc123", "def456")
    _check_revision("abc123456", "abc123")
    _check_revision(None, "archive-revision-recorded-as-provenance")


def test_metric_jobs_are_bounded_and_worker_output_uses_a_file(tmp_path):
    args = parse_args(["--data", "unused.npy"])
    assert args.metric_jobs == 1
    with pytest.raises(SystemExit):
        parse_args(["--data", "unused.npy", "--metric-jobs", "0"])

    log = tmp_path / "worker.log"
    returncode, _, error = _run_subprocess(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr)",
        ],
        log,
        timeout=10,
    )
    assert returncode == 0
    assert error is None
    assert set(log.read_text().splitlines()) == {"stdout-ok", "stderr-ok"}


def test_layout_reuse_requires_exact_fit_identity_and_layout_hash(tmp_path):
    layout_path = tmp_path / "layout.npy"
    np.save(layout_path, np.ones((12, 2), dtype=np.float32))
    prepared = PreparedInfo(
        path=str(tmp_path / "prepared.npy"),
        rows=12,
        features=5,
        source_index_sha256="source-index",
        normalized_sha256="prepared-content",
        source_fingerprint="source",
        zero_rows=0,
        preparation_seconds=0.1,
    )
    args = parse_args(
        [
            "--data",
            "unused.npy",
            "--dataset-name",
            "identity-test",
            "--ae-epochs",
            "10",
        ]
    )
    identity, digest = fit_run_identity(args, prepared, "autoencoder", 3, 2, 5)
    prior = {
        "status": "ok",
        "fit_run_identity": identity,
        "fit_run_identity_sha256": digest,
        "layout_sha256": _sha256_file(layout_path),
    }

    assert reusable_layout_matches(prior, layout_path, digest, 12, 2)
    changed = parse_args(
        [
            "--data",
            "unused.npy",
            "--dataset-name",
            "identity-test",
            "--ae-epochs",
            "11",
        ]
    )
    _, changed_digest = fit_run_identity(changed, prepared, "autoencoder", 3, 2, 5)
    assert not reusable_layout_matches(prior, layout_path, changed_digest, 12, 2)

    np.save(layout_path, np.ones((12, 3), dtype=np.float32))
    assert not reusable_layout_matches(prior, layout_path, digest, 12, 2)


def test_metric_errors_are_failures_even_if_worker_status_is_ok():
    assert not result_failed({"status": "ok"})
    assert result_failed({"status": "ok", "metric_error": "bad metric"})
    assert result_failed({"status": "failed"})


def test_crashed_rerun_cannot_resurrect_stale_success(tmp_path, monkeypatch):
    data_path = tmp_path / "data.npy"
    np.save(data_path, np.arange(60, dtype=np.float32).reshape(12, 5) + 1)
    output = tmp_path / "results"
    size_dir = output / "n12"
    size_dir.mkdir(parents=True)

    paths, source = discover_sources(data_path)
    selected = deterministic_nested_indices(12, [12], seed=1729)[12]
    prepared = prepare_normalized_data(
        paths,
        selected,
        size_dir / "prepared-l2.npy",
        source_fingerprint=source.fingerprint,
    )
    (size_dir / "prepared.json").write_text(json.dumps(prepared.__dict__))

    layout_path = size_dir / "pca-d2-s0-layout.npy"
    result_path = size_dir / "pca-d2-s0.json"
    np.save(layout_path, np.ones((12, 2), dtype=np.float32))
    old_args = parse_args(
        [
            "--data",
            str(data_path),
            "--dataset-name",
            "old-identity",
            "--sizes",
            "12",
            "--methods",
            "pca",
            "--output-dir",
            str(output),
        ]
    )
    old_identity, old_digest = fit_run_identity(old_args, prepared, "pca", 0, 2, 5)
    result_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "fit_run_identity": old_identity,
                "fit_run_identity_sha256": old_digest,
                "layout_sha256": _sha256_file(layout_path),
            }
        )
    )

    def crash_without_writing(_command, _log_path, _timeout):
        sentinel = json.loads(result_path.read_text())
        assert sentinel["status"] == "running"
        assert sentinel["launch_nonce"]
        return 9, 0.1, None

    monkeypatch.setattr(comparison, "_run_subprocess", crash_without_writing)
    new_args = parse_args(
        [
            "--data",
            str(data_path),
            "--dataset-name",
            "new-identity",
            "--sizes",
            "12",
            "--methods",
            "pca",
            "--output-dir",
            str(output),
            "--require-all",
        ]
    )

    assert comparison.orchestrate(new_args) == 1
    row = json.loads(result_path.read_text())
    assert row["status"] == "failed"
    assert row["error"] == "worker exited with code 9"
    assert row["fit_run_identity_sha256"] != old_digest
    assert "metrics" not in row
