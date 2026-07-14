"""CPU and mock tests for the isolated Yandex graph-method harness."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import experiments.compare_yandex_graph_methods as graph
import numpy as np
import pytest


def _write_fbin(path: Path, values: np.ndarray) -> Path:
    values = np.ascontiguousarray(values, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(np.asarray(values.shape, dtype="<u4").tobytes())
        handle.write(values.tobytes())
    return path


def _argv(tmp_path: Path, method: str = "umap", rows: int = 20) -> list[str]:
    return [
        "--data",
        str(tmp_path / "base.fbin"),
        "--method",
        method,
        "--rows",
        str(rows),
        "--target-dimension",
        "2",
        "--output-dir",
        str(tmp_path / "out"),
    ]


class _FakeMonitor:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def result(self):
        return {
            "peak_rss_bytes": 123,
            "gpu_baseline_bytes": 0,
            "peak_gpu_process_bytes": 10,
            "peak_incremental_gpu_bytes": 10,
            "sampling_interval_seconds": 0.1,
        }


@pytest.fixture
def stable_runtime_metadata(monkeypatch):
    monkeypatch.setattr(graph, "gpu_memory_info", lambda *_args: None)
    monkeypatch.setattr(graph, "_hardware", lambda *_args: {"machine": "test"})
    monkeypatch.setattr(graph, "_versions", lambda: {"numpy": np.__version__})


def _worker_flags(tmp_path: Path) -> list[str]:
    return [
        "--worker",
        "--worker-result",
        str(tmp_path / "private-result.json"),
        "--worker-heartbeat",
        str(tmp_path / "heartbeat.json"),
        "--run-id",
        "test-run",
        "--disk-reserve-gib",
        "0",
    ]


def test_cli_is_exactly_one_method_size_and_dimension(tmp_path):
    args = graph.parse_args(_argv(tmp_path))

    assert args.method == "umap"
    assert args.rows == 20
    assert args.target_dimension == 2
    assert args.init == "random"
    assert args.layout_path == str((tmp_path / "out" / "layout.npy").resolve())
    assert args.result_path == str((tmp_path / "out" / "result.json").resolve())
    assert graph.configuration_payload(args)["umap"]["build_algo"] == "nn_descent"


@pytest.mark.parametrize(
    "extra",
    [
        ["--umap-knn-n-clusters", "2", "--umap-knn-overlap-factor", "2"],
        ["--umap-nnd-graph-degree", "15"],
        ["--umap-nnd-intermediate-graph-degree", "64"],
        ["--neighbors", "20"],
        ["--preflight-vram-fraction", "1.1"],
    ],
)
def test_cli_rejects_invalid_scale_controls(tmp_path, extra):
    with pytest.raises(SystemExit):
        graph.parse_args(_argv(tmp_path) + extra)


def test_device_id_parser_accepts_all_none_and_unique_csv():
    assert graph._parse_device_ids("all") == "all"
    assert graph._parse_device_ids("none") is None
    assert graph._parse_device_ids("3,1") == [3, 1]
    with pytest.raises(Exception, match="unique"):
        graph._parse_device_ids("1,1")


def test_dire_preflight_includes_named_default_negative_tensor(tmp_path):
    args = graph.parse_args(_argv(tmp_path, method="dire", rows=1000))
    result = graph.memory_preflight(
        args,
        rows=1000,
        features=96,
        gpu_memory={"free_bytes": 10**12, "total_bytes": 10**12},
    )
    named = result["named_allocations"]

    assert named["input_float32_bytes"] == 1000 * 96 * 4
    assert named["knn_graph_bytes"] == 1000 * 16 * (8 + 4)
    assert named["layout_float32_bytes"] == 1000 * 2 * 4
    assert named["default_negative_samples_per_row"] == 8 * 16
    assert named["default_negative_indices_int64_bytes"] == 1000 * 128 * 8
    assert result["fit_blocked"] is False


def test_preflight_warning_is_not_an_attempt_or_a_block(tmp_path):
    args = graph.parse_args(_argv(tmp_path, method="dire", rows=1000))
    result = graph.memory_preflight(
        args,
        rows=1000,
        features=96,
        gpu_memory={"free_bytes": 1, "total_bytes": 1},
    )

    assert result["warning"] is True
    assert result["fit_blocked"] is False
    assert "exceeds" in result["messages"][0]


def test_umap_preflight_uses_configured_partition_fraction(tmp_path):
    args = graph.parse_args(
        _argv(tmp_path, rows=1000)
        + ["--umap-knn-n-clusters", "10", "--umap-knn-overlap-factor", "2"]
    )
    result = graph.memory_preflight(
        args,
        rows=1000,
        features=96,
        gpu_memory={"free_bytes": 10**12, "total_bytes": 10**12},
    )

    assert result["named_allocations"]["estimated_partition_rows"] == 200
    assert result["named_allocations"]["partition_input_float32_bytes"] == 200 * 96 * 4


def test_umap_receives_memmap_and_all_rapids_scale_settings(tmp_path):
    values = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    raw = tmp_path / "raw.bin"
    values.tofile(raw)
    X = np.memmap(raw, mode="r", dtype=np.float32, shape=(20, 4))
    args = graph.parse_args(
        _argv(tmp_path)
        + [
            "--umap-knn-n-clusters",
            "8",
            "--umap-knn-overlap-factor",
            "3",
            "--umap-nnd-graph-degree",
            "32",
            "--umap-nnd-intermediate-graph-degree",
            "48",
            "--umap-nnd-max-iterations",
            "11",
            "--umap-nnd-termination-threshold",
            "0.002",
        ]
    )
    captured = {}

    class FakeUMAP:
        __module__ = "cuml.manifold"

        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit_transform(self, incoming):
            captured["input"] = incoming
            return np.zeros((incoming.shape[0], 2), dtype=np.float32)

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None))
    )
    layout, details = graph.fit_umap(X, args, umap_class=FakeUMAP, cupy=fake_cupy)

    assert captured["input"] is X
    assert captured["kwargs"]["build_algo"] == "nn_descent"
    assert captured["kwargs"]["build_kwds"] == {
        "knn_n_clusters": 8,
        "knn_overlap_factor": 3,
        "nnd_graph_degree": 32,
        "nnd_intermediate_graph_degree": 48,
        "nnd_max_iterations": 11,
        "nnd_termination_threshold": 0.002,
    }
    assert captured["kwargs"]["hash_input"] is False
    assert captured["kwargs"]["output_type"] == "numpy"
    assert details["harness_full_host_normalization_copy"] is False
    assert details["nn_descent_exactly_deterministic"] is False
    assert "non-deterministic" in details["determinism_note"]
    assert layout.shape == (20, 2)


def test_dire_receives_cuvs_index_and_no_normalization(tmp_path):
    values = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    raw = tmp_path / "raw.bin"
    values.tofile(raw)
    X = np.memmap(raw, mode="r", dtype=np.float32, shape=(20, 4))
    args = graph.parse_args(
        _argv(tmp_path, method="dire")
        + ["--dire-repo", str(tmp_path), "--dire-cuvs-index", "cagra"]
    )
    captured = {}

    class FakeReducer:
        __module__ = "dire_rapids"

        def _select_cuvs_index_type(self, rows, features, metric):
            captured["selection"] = (rows, features, metric)
            return "cagra"

        def fit_transform(self, incoming):
            captured["input"] = incoming
            return np.zeros((incoming.shape[0], 2), dtype=np.float32)

    def fake_create_dire(**kwargs):
        captured["kwargs"] = kwargs
        return FakeReducer()

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None))
    )
    layout, details = graph.fit_dire(X, args, create_dire=fake_create_dire, cupy=fake_cupy)

    assert captured["input"] is X
    assert captured["kwargs"]["backend"] == "cuvs"
    assert captured["kwargs"]["cuvs_index_type"] == "cagra"
    assert captured["kwargs"]["normalize"] is False
    assert captured["kwargs"]["neg_ratio"] == graph.DEFAULT_DIRE_NEG_RATIO
    assert details["cuvs_index_resolved"] == "cagra"
    assert layout.dtype == np.float32


def test_layout_validation_is_exact_and_bounded():
    layout = np.arange(60, dtype=np.float32).reshape(20, 3)
    summary = graph.validate_and_summarize_layout(layout, rows=20, target_dimension=3, block_rows=4)

    assert summary["finite_coordinates"] == 60
    assert summary["coordinate_mean"] == pytest.approx(layout.mean(axis=0))
    with pytest.raises(graph.IntegrityFailure, match="shape"):
        graph.validate_and_summarize_layout(layout, rows=19, target_dimension=3, block_rows=4)
    with pytest.raises(graph.IntegrityFailure, match="dtype"):
        graph.validate_and_summarize_layout(
            layout.astype(np.float64), rows=20, target_dimension=3, block_rows=4
        )
    layout[2, 1] = np.nan
    with pytest.raises(graph.IntegrityFailure, match="non-finite"):
        graph.validate_and_summarize_layout(layout, rows=20, target_dimension=3, block_rows=4)


def test_layout_writer_is_atomic_and_records_logical_hash(tmp_path):
    layout = np.arange(40, dtype=np.float32).reshape(20, 2)
    destination = tmp_path / "layout.npy"
    metadata = graph._save_layout_atomic(destination, layout, block_rows=3)

    assert np.array_equal(np.load(destination), layout)
    assert metadata["rows"] == 20
    assert metadata["target_dimension"] == 2
    assert metadata["logical_data_sha256"] == graph._logical_array_sha256(layout)
    assert not list(tmp_path.glob(".layout.npy.tmp-*"))


def test_atomic_json_preserves_old_destination_when_replace_fails(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    destination.write_text("old\n")

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(graph.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        graph._atomic_json(destination, {"status": "ok"})

    assert destination.read_text() == "old\n"
    assert not list(tmp_path.glob(".result.json.tmp-*"))


def test_exception_and_signal_statuses_are_closed():
    CudaOOM = type("OutOfMemoryError", (RuntimeError,), {"__module__": "cupy.cuda.memory"})

    assert graph.classify_exception(CudaOOM("out of memory"), phase="fit") == "cuda_oom"
    assert (
        graph.classify_exception(CudaOOM("out of memory"), phase="layout_validation") == "cuda_oom"
    )
    assert (
        graph.classify_exception(
            MemoryError("RMM failure: std::bad_alloc: out_of_memory"), phase="fit"
        )
        == "cuda_oom"
    )
    assert graph.classify_exception(MemoryError(), phase="fit") == "host_oom_or_sigkill"
    assert (
        graph.classify_exception(RuntimeError("oops"), phase="layout_validation")
        == "integrity_failed"
    )
    assert graph.classify_exception(graph.MetricFailure("oops"), phase="metric") == "metric_failed"
    assert (
        graph.classify_exception(graph.MetricFailure("oops"), phase="layout_validation")
        == "metric_failed"
    )
    assert graph.classify_exception(NotImplementedError(), phase="fit") == "unsupported_scale"
    assert (
        graph.classify_exception(
            RuntimeError("cudaErrorNoDevice: no CUDA-capable device is detected"),
            phase="fit",
        )
        == "unsupported_device"
    )
    assert (
        graph.classify_exception(RuntimeError("cuVS backend requires CUDA GPU"), phase="fit")
        == "unsupported_device"
    )
    assert (
        graph._abrupt_status(graph.ChildOutcome(-signal.SIGKILL, 1.0, False, signal.SIGKILL))
        == "host_oom_or_sigkill"
    )
    assert graph._abrupt_status(graph.ChildOutcome(None, 1.0, True)) == "timeout"


def test_worker_success_records_exact_rows_resources_and_atomic_layout(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    _write_fbin(
        tmp_path / "base.fbin",
        np.arange(20 * 4, dtype=np.float32).reshape(20, 4),
    )
    worker_result = tmp_path / "private-result.json"
    args = graph.parse_args(_argv(tmp_path) + _worker_flags(tmp_path))
    seen = {}

    def fake_fit(X, incoming_args):
        seen["memmap"] = isinstance(X, np.memmap)
        seen["writeable"] = X.flags.writeable
        return np.zeros((20, 2), dtype=np.float32), {
            "fit_seconds": 0.25,
            "backend_import_seconds": 0.01,
        }

    monkeypatch.setattr(graph, "fit_umap", fake_fit)
    monkeypatch.setattr(graph, "PeakMonitor", _FakeMonitor)

    assert graph.run_worker(args) == 0
    payload = json.loads(worker_result.read_text())
    layout = np.load(tmp_path / "out" / "layout.npy")

    assert seen == {"memmap": True, "writeable": False}
    assert payload["status"] == "ok"
    assert payload["rows"] == payload["layout"]["rows"] == 20
    assert payload["features"] == 4
    assert payload["attempted_fit"] is True
    assert payload["resources"]["peak_rss_bytes"] == 123
    assert layout.shape == (20, 2)


def test_worker_cuda_oom_is_attempted_despite_preflight_warning(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    worker_result = tmp_path / "private-result.json"
    args = graph.parse_args(_argv(tmp_path) + _worker_flags(tmp_path))
    CudaOOM = type("OutOfMemoryError", (RuntimeError,), {"__module__": "cupy.cuda.memory"})

    def fail_fit(_X, _args):
        raise CudaOOM("CUDA out of memory")

    monkeypatch.setattr(graph, "fit_umap", fail_fit)
    monkeypatch.setattr(graph, "PeakMonitor", _FakeMonitor)
    monkeypatch.setattr(
        graph,
        "gpu_memory_info",
        lambda *_args: {"device_id": 0, "free_bytes": 1, "total_bytes": 1},
    )

    assert graph.run_worker(args) == 1
    payload = json.loads(worker_result.read_text())
    assert payload["status"] == "cuda_oom"
    assert payload["attempted_fit"] is True
    assert payload["preflight"]["warning"] is True
    assert payload["preflight"]["fit_blocked"] is False
    assert payload["preflight_disposition"] == "warning_then_fit_attempted"
    assert payload["preflight_advisory"] is True
    assert payload["observed_attempt_failure"] is True


def test_parent_rejects_row_limit_before_launch(tmp_path, monkeypatch, stable_runtime_metadata):
    _write_fbin(tmp_path / "base.fbin", np.ones((10, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path, rows=20))
    monkeypatch.setattr(
        graph,
        "_run_child",
        lambda *_args, **_kwargs: pytest.fail("worker must not launch"),
    )

    assert graph.run_isolated(args) == 1
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["status"] == "integrity_failed"
    assert payload["attempted_fit"] is False
    assert payload["isolation"]["worker_launched"] is False


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (graph.ChildOutcome(-signal.SIGKILL, 2.0, False, signal.SIGKILL), "host_oom_or_sigkill"),
        (graph.ChildOutcome(-signal.SIGTERM, 3.0, True, signal.SIGTERM), "timeout"),
        (graph.ChildOutcome(2, 1.0, False), "unsupported_scale"),
    ],
)
def test_parent_classifies_abrupt_worker_outcomes(
    tmp_path, monkeypatch, stable_runtime_metadata, outcome, expected
):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path) + ["--timeout", "10"])
    monkeypatch.setattr(graph, "_run_child", lambda *_args, **_kwargs: outcome)

    assert graph.run_isolated(args) == 1
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["status"] == expected
    assert payload["attempted_fit"] is False
    assert payload["isolation"]["worker_returncode"] == outcome.returncode


def test_worker_command_round_trips_every_scale_setting(tmp_path):
    args = graph.parse_args(
        _argv(tmp_path)
        + [
            "--umap-knn-n-clusters",
            "8",
            "--umap-knn-overlap-factor",
            "3",
            "--umap-device-ids",
            "2",
            "--umap-epochs",
            "40",
            "--umap-force-serial-epochs",
        ]
    )
    args.run_id = "round-trip"
    command = graph._worker_command(args, tmp_path / "private.json", tmp_path / "heartbeat.json")
    worker_args = graph.parse_args(command[2:])

    assert worker_args.worker is True
    assert worker_args.method == "umap"
    assert graph.umap_build_kwds(worker_args) == graph.umap_build_kwds(args)
    assert worker_args.umap_device_ids == [2]
    assert worker_args.umap_epochs == 40
    assert worker_args.umap_force_serial_epochs is True


def test_existing_result_requires_explicit_force(tmp_path, stable_runtime_metadata):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path))
    result = tmp_path / "out" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n")

    with pytest.raises(FileExistsError, match="--force"):
        graph.run_isolated(args)


def test_umap_api_edge_cases_are_validated_per_method(tmp_path):
    default = graph.parse_args(_argv(tmp_path))
    assert default.umap_force_serial_epochs is None

    with pytest.raises(SystemExit):
        graph.parse_args(_argv(tmp_path) + ["--init", "pca"])
    dire = graph.parse_args(_argv(tmp_path, method="dire") + ["--init", "pca"])
    assert dire.init == "pca"

    single_cluster = graph.parse_args(
        _argv(tmp_path) + ["--umap-knn-n-clusters", "1", "--umap-knn-overlap-factor", "2"]
    )
    assert single_cluster.umap_knn_overlap_factor == 2

    high_dimension = _argv(tmp_path, rows=600)
    high_dimension[high_dimension.index("2")] = "513"
    with pytest.raises(SystemExit):
        graph.parse_args(high_dimension + ["--umap-force-serial-epochs"])
    parallel = graph.parse_args(high_dimension + ["--no-umap-force-serial-epochs"])
    assert parallel.umap_force_serial_epochs is False


@pytest.mark.parametrize("device_ids", ["all", "0,1"])
def test_multi_gpu_is_rejected_until_monitoring_is_complete(tmp_path, device_ids):
    with pytest.raises(SystemExit):
        graph.parse_args(_argv(tmp_path) + ["--umap-device-ids", device_ids])


def test_worker_command_preserves_force_serial_epochs_tristate(tmp_path):
    args = graph.parse_args(_argv(tmp_path))
    args.run_id = "tri-state"
    command = graph._worker_command(args, tmp_path / "private.json", tmp_path / "heartbeat.json")
    assert "--umap-force-serial-epochs" not in command
    assert "--no-umap-force-serial-epochs" not in command
    assert graph.parse_args(command[2:]).umap_force_serial_epochs is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--layout-path", "{data}"],
        ["--result-path", "{data}"],
        ["--log-path", "{data}"],
        ["--layout-path", "{out}/layout.bin"],
        ["--result-path", "{out}/same", "--log-path", "{out}/same"],
    ],
)
def test_cli_rejects_source_aliases_and_invalid_layout_suffix(tmp_path, extra):
    data = str(tmp_path / "base.fbin")
    out = str(tmp_path / "out")
    resolved = [item.format(data=data, out=out) for item in extra]
    with pytest.raises(SystemExit):
        graph.parse_args(_argv(tmp_path) + resolved)


def test_worker_cli_rejects_private_control_alias(tmp_path):
    flags = _worker_flags(tmp_path)
    flags[flags.index(str(tmp_path / "private-result.json"))] = str(tmp_path / "base.fbin")
    with pytest.raises(SystemExit):
        graph.parse_args(_argv(tmp_path) + flags)


def test_cli_rejects_existing_hardlink_alias_to_source(tmp_path):
    data = _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    hardlink = tmp_path / "result-hardlink.json"
    hardlink.hardlink_to(data)
    with pytest.raises(SystemExit):
        graph.parse_args(_argv(tmp_path) + ["--result-path", str(hardlink)])


def test_disk_preflight_failure_is_recorded_before_launch(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path) + ["--disk-reserve-gib", "0"])
    monkeypatch.setattr(
        graph.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )
    monkeypatch.setattr(
        graph,
        "_run_child",
        lambda *_args, **_kwargs: pytest.fail("worker must not launch"),
    )

    assert graph.run_isolated(args) == 1
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["status"] == "integrity_failed"
    assert payload["error"]["cause"] == "insufficient_disk"
    assert payload["preflight"]["disk_requirement"]["status"] == "failed"
    assert payload["attempted_fit"] is False


def test_manifest_provenance_is_validated_and_recorded(tmp_path):
    data = _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    info = __import__("experiments.yandex_fbin", fromlist=["read_fbin_header"]).read_fbin_header(
        data
    )
    manifest = {
        "schema_version": 1,
        "source": {"url": "https://example.invalid/deep1b.fbin"},
        "prefix": {
            "rows": 20,
            "dimension": 4,
            "file_bytes": info.expected_size,
            "sha256": "a" * 64,
        },
    }
    Path(f"{data}.manifest.json").write_text(json.dumps(manifest))

    record = graph._load_source_manifest(data, info)
    assert record["present"] is True
    assert record["expected_fbin_sha256"] == "a" * 64
    assert record["source"] == manifest["source"]

    manifest["prefix"]["rows"] = 19
    Path(f"{data}.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(graph.IntegrityFailure, match="disagrees"):
        graph._load_source_manifest(data, info)


def test_staged_layout_is_validated_before_atomic_replace(tmp_path, monkeypatch):
    layout = np.ones((20, 2), dtype=np.float32)
    destination = tmp_path / "layout.npy"
    replace_calls = []

    monkeypatch.setattr(
        graph.np,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(shape=(19, 2), dtype=np.dtype(np.float32)),
    )
    monkeypatch.setattr(
        graph.os,
        "replace",
        lambda *_args: replace_calls.append(_args),
    )

    with pytest.raises(graph.IntegrityFailure, match="staged layout"):
        graph._save_layout_atomic(destination, layout)
    assert replace_calls == []
    assert not destination.exists()


def test_force_failure_preserves_previous_result_and_layout(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    output = tmp_path / "out"
    output.mkdir()
    old_layout = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.save(output / "layout.npy", old_layout)
    previous_layout = (output / "layout.npy").read_bytes()
    (output / "result.json").write_text('{"status": "ok", "old": true}\n')
    previous_result = (output / "result.json").read_text()
    args = graph.parse_args(_argv(tmp_path) + ["--force", "--disk-reserve-gib", "0"])
    monkeypatch.setattr(
        graph,
        "_run_child",
        lambda *_args, **_kwargs: graph.ChildOutcome(2, 0.1, False),
    )

    assert graph.run_isolated(args) == 1
    assert (output / "layout.npy").read_bytes() == previous_layout
    assert (output / "result.json").read_text() == previous_result
    failures = list(output.glob("result.*.failed.json"))
    assert len(failures) == 1


def test_success_uses_immutable_generation_then_result_then_pointer(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path) + ["--disk-reserve-gib", "0"])

    def fake_fit(_X, _args):
        return np.zeros((20, 2), dtype=np.float32), {
            "fit_seconds": 0.1,
            "backend_import_seconds": 0.01,
        }

    def run_inline(command, _log, _timeout):
        code = graph.main(command[2:])
        return graph.ChildOutcome(code, 0.2, False)

    monkeypatch.setattr(graph, "fit_umap", fake_fit)
    monkeypatch.setattr(graph, "PeakMonitor", _FakeMonitor)
    monkeypatch.setattr(graph, "_run_child", run_inline)

    assert graph.run_isolated(args) == 0
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    pointer = tmp_path / "out" / "layout.npy"
    generation = Path(result["layout"]["path"])
    assert generation.exists()
    assert generation != pointer
    assert pointer.is_symlink()
    assert pointer.resolve() == generation.resolve()
    assert result["layout_publication"]["pointer_status"] == "ok"
    assert result["source_unchanged_parent"] is True
    rerun = graph.parse_args(_argv(tmp_path) + ["--force"])
    assert Path(rerun.layout_path) == pointer


def test_source_mutation_overrides_abrupt_backend_status(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    data = _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path) + ["--disk-reserve-gib", "0"])

    def mutate_then_exit(_command, _log, _timeout):
        original_mtime = data.stat().st_mtime_ns
        with data.open("r+b") as handle:
            handle.seek(-4, 2)
            handle.write(np.float32(7).tobytes())
        # Some filesystems coarsen timestamps enough for a same-tick write to retain the
        # original stat identity. Move mtime forward explicitly so this test exercises
        # the mutation-classification policy rather than timestamp resolution.
        os.utime(data, ns=(original_mtime + 1_000_000_000,) * 2)
        return graph.ChildOutcome(2, 0.1, False)

    monkeypatch.setattr(graph, "_run_child", mutate_then_exit)
    assert graph.run_isolated(args) == 1
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["status"] == "integrity_failed"
    assert payload["source_unchanged_parent"] is False
    assert "changed" in payload["error"]["message"]


def test_heartbeat_marks_attempt_and_orphan_layout_is_quarantined(
    tmp_path, monkeypatch, stable_runtime_metadata
):
    _write_fbin(tmp_path / "base.fbin", np.ones((20, 4), dtype=np.float32))
    args = graph.parse_args(_argv(tmp_path) + ["--disk-reserve-gib", "0"])

    def die_after_fit_started(command, _log, _timeout):
        worker = graph.parse_args(command[2:])
        np.save(worker.layout_path, np.zeros((20, 2), dtype=np.float32))
        graph._write_heartbeat(worker, "fit_started", attempted_fit=True)
        return graph.ChildOutcome(-signal.SIGKILL, 0.1, False, signal.SIGKILL)

    monkeypatch.setattr(graph, "_run_child", die_after_fit_started)
    assert graph.run_isolated(args) == 1
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["status"] == "host_oom_or_sigkill"
    assert payload["attempted_fit"] is True
    assert payload["isolation"]["heartbeat"]["phase"] == "fit_started"
    assert len(payload["quarantined_layouts"]) == 1
    assert Path(payload["quarantined_layouts"][0]).exists()
    assert not (tmp_path / "out" / "layout.npy").exists()


def test_run_child_terminates_process_group_on_parent_base_exception(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def wait(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(graph.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(graph.os, "getpgid", lambda _pid: 67890)
    monkeypatch.setattr(
        graph,
        "_terminate_process_group",
        lambda process, pgid=None: calls.append((process.pid, pgid)),
    )

    with pytest.raises(KeyboardInterrupt):
        graph._run_child(["fake"], tmp_path / "worker.log", 10)
    assert calls == [(12345, 67890)]


def test_process_group_termination_escalates_using_recorded_pgid(monkeypatch):
    signals = []

    class FakeProcess:
        pid = 12345
        returncode = None
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise graph.subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(graph.os, "getpgrp", lambda: 111)
    monkeypatch.setattr(
        graph.os,
        "killpg",
        lambda pgid, sent_signal: signals.append((pgid, sent_signal)),
    )

    terminated_by = graph._terminate_process_group(process, 67890)
    assert signals == [(67890, signal.SIGTERM), (67890, signal.SIGKILL)]
    assert terminated_by == signal.SIGKILL


def test_preflight_separates_host_device_and_enforced_disk_estimates(tmp_path):
    args = graph.parse_args(_argv(tmp_path) + ["--disk-reserve-gib", "0"])
    preflight = graph.memory_preflight(
        args,
        rows=20,
        features=4,
        gpu_memory={"device_id": 0, "free_bytes": 10**9, "total_bytes": 10**9},
    )
    graph._attach_disk_preflight(preflight, args, tmp_path / "layout.npy")
    assert preflight["host_advisory"]["advisory_only"] is True
    assert preflight["device_advisory"]["advisory_only"] is True
    assert preflight["disk_requirement"]["enforced"] is True
    assert preflight["disk_requirement"]["status"] == "ok"
