"""CPU tests for one-scan, multi-cutoff AE training-size retrieval."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path

import experiments.evaluate_ae_training_size_retrieval as multi
import numpy as np
import pytest
from experiments.encode_ae_training_size_checkpoint import PROTOCOL as ENCODING_PROTOCOL
from experiments.evaluate_ae_training_size_retrieval import (
    _resolve_arms,
    _test_args,
    build_parser,
    execute,
    run_evaluation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ground_truth(path: Path) -> None:
    # The exact reduced-space top 100 are ids 0..99.  Ground-truth top 10
    # agrees exactly, while its top 100 contains 20 ids outside that set.
    ids = np.asarray([list(range(10)) + list(range(30, 120))], dtype="<u4")
    distances = np.arange(100, dtype="<f4")[None, :]
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", 1, 100))
        handle.write(ids.tobytes())
        handle.write(distances.tobytes())


def _write_arm(root: Path, *, label: str, pool_size: int):
    base = root / f"{label}-base.npy"
    query = root / f"{label}-query.npy"
    provenance = root / f"{label}-encoding.json"
    np.save(
        base,
        np.column_stack((np.arange(120, dtype=np.float32), np.zeros(120, dtype=np.float32))),
    )
    np.save(query, np.zeros((1, 2), dtype=np.float32))
    base_array = np.load(base, allow_pickle=False)
    query_array = np.load(query, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "protocol": ENCODING_PROTOCOL,
        "status": "ok",
        "selection": {"init_seed": 1729, "pool_size": pool_size, "selected_update": 17},
        "model": {"target_dimension": 2},
        "training_result": {
            "sha256": "a" * 64,
            "split": {"verified": True, "selected_pool_size": pool_size},
        },
        "checkpoint": {
            "init_seed": 1729,
            "pool_size": pool_size,
            "state_dict_sha256": "b" * 64,
        },
        "base": {
            "source_rows": 120,
            "layout_path": str(base.resolve()),
            "layout_rows": 120,
            "layout_dimension": 2,
            "layout_dtype": str(base_array.dtype),
            "layout_file_bytes": base.stat().st_size,
            "layout_sha256": _sha256(base),
        },
        "query": {
            "source_rows": 1,
            "all_rows_encoded": True,
            "layout_path": str(query.resolve()),
            "layout_rows": 1,
            "layout_dimension": 2,
            "layout_dtype": str(query_array.dtype),
            "layout_file_bytes": query.stat().st_size,
            "layout_sha256": _sha256(query),
        },
    }
    provenance.write_text(json.dumps(payload))
    return base, query, provenance


def _single_args(tmp_path: Path, *, pool_size: int = 65_536):
    base, query, provenance = _write_arm(tmp_path, label="small", pool_size=pool_size)
    ground_truth = tmp_path / "ground-truth.bin"
    _write_ground_truth(ground_truth)
    return _test_args(
        base_layout=base,
        query_layout=query,
        encoding_provenance=provenance,
        label="small",
        ground_truth=ground_truth,
        output_json=tmp_path / "retrieval.json",
        output_csv=tmp_path / "retrieval.csv",
        base_block_rows=37,
        query_batch_size=1,
    )


def test_default_cutoffs_share_one_exact_top100_scan_and_write_json_csv(tmp_path, monkeypatch):
    args = _single_args(tmp_path)
    calls = []
    exact = multi._exact_streamed_neighbors

    def counted(*positional, **keywords):
        calls.append(keywords["k"])
        return exact(*positional, **keywords)

    monkeypatch.setattr(multi, "_exact_streamed_neighbors", counted)
    result = execute(args)

    assert calls == [100]
    assert result["cutoffs"] == [10, 100]
    assert result["maximum_k"] == 100
    assert result["query_count"] == 1
    arm = result["arms"][0]
    assert arm["search"]["exact_search_invocations"] == 1
    assert arm["search"]["base_stream_passes"] == 1
    assert arm["quality"]["recall_at"] == {"10": 1.0, "100": 0.8}
    assert arm["quality"]["recall_hits"] == {"10": 10, "100": 80}
    assert arm["verification"]["verified"] is True
    assert arm["verification"]["training_result_sha256"] == "a" * 64
    assert result["verification"]["one_exact_max_k_search_per_arm"] is True

    saved = json.loads(args.output_json.read_text())
    with args.output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert saved["arms"][0]["quality"]["recall_at"]["100"] == pytest.approx(0.8)
    assert len(rows) == 1
    assert float(rows[0]["recall_at_10"]) == 1.0
    assert float(rows[0]["recall_at_100"]) == pytest.approx(0.8)
    assert int(rows[0]["pool_size"]) == 65_536
    assert not list(tmp_path.glob(".*.tmp"))


def test_repeated_arm_specs_run_once_per_arm_and_preserve_identity(tmp_path, monkeypatch):
    first = _write_arm(tmp_path, label="small", pool_size=65_536)
    second = _write_arm(tmp_path, label="large", pool_size=9_863_168)
    ground_truth = tmp_path / "ground-truth.bin"
    _write_ground_truth(ground_truth)
    args = _test_args(
        arm=(
            ("small", *(str(path) for path in first)),
            ("large", *(str(path) for path in second)),
        ),
        ground_truth=ground_truth,
        output_json=tmp_path / "retrieval.json",
        output_csv=tmp_path / "retrieval.csv",
    )
    calls = 0
    exact = multi._exact_streamed_neighbors

    def counted(*positional, **keywords):
        nonlocal calls
        calls += 1
        return exact(*positional, **keywords)

    monkeypatch.setattr(multi, "_exact_streamed_neighbors", counted)
    result = run_evaluation(args)

    assert calls == 2
    assert [arm["selection"]["pool_size"] for arm in result["arms"]] == [65_536, 9_863_168]
    assert all(arm["quality"]["recall_at"]["10"] == 1.0 for arm in result["arms"])


def test_tampered_layout_hash_fails_before_exact_search(tmp_path, monkeypatch):
    args = _single_args(tmp_path)
    with args.base_layout.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("search ran before provenance verification")

    monkeypatch.setattr(multi, "_exact_streamed_neighbors", forbidden)
    with pytest.raises(ValueError, match="base layout disagrees.*sha256"):
        run_evaluation(args)


def test_parser_supports_single_or_repeated_arms_and_default_cutoffs(tmp_path):
    parser = build_parser()
    common = [
        "--ground-truth",
        str(tmp_path / "gt.bin"),
        "--output-json",
        str(tmp_path / "out.json"),
        "--output-csv",
        str(tmp_path / "out.csv"),
    ]
    single = parser.parse_args(
        [
            "--base-layout",
            str(tmp_path / "base.npy"),
            "--query-layout",
            str(tmp_path / "query.npy"),
            "--encoding-provenance",
            str(tmp_path / "encoding.json"),
            *common,
        ]
    )
    repeated = parser.parse_args(
        [
            "--arm",
            "one",
            str(tmp_path / "a.npy"),
            str(tmp_path / "q.npy"),
            str(tmp_path / "p.json"),
            *common,
        ]
    )

    assert single.cutoffs == (10, 100)
    assert _resolve_arms(single)[0].label == "single"
    assert _resolve_arms(repeated)[0].label == "one"
    with pytest.raises(ValueError, match="not both"):
        _resolve_arms(
            _test_args(
                arm=repeated.arm,
                base_layout=tmp_path / "base.npy",
                query_layout=tmp_path / "query.npy",
                encoding_provenance=tmp_path / "encoding.json",
            )
        )
