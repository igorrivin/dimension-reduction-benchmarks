"""Focused safety tests for the resumable Wikipedia embedding cache."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "experiments" / "cache_wikipedia_embeddings.py"
SPEC = importlib.util.spec_from_file_location("cache_wikipedia_embeddings", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache_wikipedia = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_wikipedia)


def _write_shard(cache: Path, index: int, *, rows: int = 2, dimension: int = 3) -> None:
    values = np.arange(rows * dimension, dtype=np.float32).reshape(rows, dimension)
    np.save(cache / f"emb_{index:04d}.npy", values, allow_pickle=False)
    pd.DataFrame(
        {
            "_id": [f"row-{index}-{row}" for row in range(rows)],
            "url": [f"https://example.test/{index}/{row}" for row in range(rows)],
            "title": [f"title {index}-{row}" for row in range(rows)],
        }
    ).to_parquet(cache / f"meta_{index:04d}.parquet", index=False)


def _args(cache: Path, **overrides) -> SimpleNamespace:
    values = {
        "dataset": cache_wikipedia.DEFAULT_DATASET,
        "config": "en",
        "split": "train",
        "revision": "main",
        "n_rows": 2,
        "shard_size": 2,
        "cache_dir": cache,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manifest(args: SimpleNamespace, records: list[dict], dimension: int = 3) -> dict:
    return {
        "schema_version": 1,
        "created_or_updated_at": "2026-07-13T00:00:00+00:00",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "requested_revision": args.revision,
        "resolved_revision": "a" * 40,
        "embedding_field": "emb",
        "metadata_fields": list(cache_wikipedia.METADATA_FIELDS),
        "embedding_dimension": dimension,
        "dtype": "float32",
        "shard_size": args.shard_size,
        "target_rows": args.n_rows,
        "cached_rows": sum(record["rows"] for record in records),
        "shards": records,
    }


def test_atomic_writers_leave_complete_readable_files(tmp_path):
    array_path = tmp_path / "array.npy"
    json_path = tmp_path / "manifest.json"
    parquet_path = tmp_path / "metadata.parquet"

    cache_wikipedia._atomic_save_npy(array_path, np.ones((2, 3), dtype=np.float32))
    cache_wikipedia._atomic_write_json(json_path, {"cached_rows": 2})
    cache_wikipedia._atomic_write_parquet(
        parquet_path,
        {"_id": ["a", "b"], "url": ["u1", "u2"], "title": ["t1", "t2"]},
    )

    np.testing.assert_array_equal(np.load(array_path), np.ones((2, 3), dtype=np.float32))
    assert json.loads(json_path.read_text()) == {"cached_rows": 2}
    assert list(pd.read_parquet(parquet_path).columns) == list(cache_wikipedia.METADATA_FIELDS)
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_cache_lock_rejects_concurrent_writer_and_cleans_up(tmp_path):
    with cache_wikipedia._cache_lock(tmp_path):
        assert (tmp_path / ".cache.lock").exists()
        with pytest.raises(RuntimeError, match="already locked"):
            with cache_wikipedia._cache_lock(tmp_path):
                pass

    assert not (tmp_path / ".cache.lock").exists()


def test_inspect_existing_returns_verified_records(tmp_path):
    _write_shard(tmp_path, 0)
    _write_shard(tmp_path, 1)

    records, rows, dimension = cache_wikipedia._inspect_existing(tmp_path, shard_size=2)

    assert rows == 4
    assert dimension == 3
    assert [record["index"] for record in records] == [0, 1]
    assert [record["rows"] for record in records] == [2, 2]
    assert all(len(record["embedding_sha256"]) == 64 for record in records)
    assert all(len(record["metadata_sha256"]) == 64 for record in records)


def test_existing_shards_reject_unpaired_files(tmp_path):
    np.save(tmp_path / "emb_0000.npy", np.ones((2, 3), dtype=np.float32))

    with pytest.raises(RuntimeError, match="indices do not match"):
        cache_wikipedia._existing_shards(tmp_path)


@pytest.mark.parametrize("indices", [(1,), (0, 2)])
def test_existing_shards_reject_noncontiguous_indices(tmp_path, indices):
    for index in indices:
        _write_shard(tmp_path, index)

    with pytest.raises(RuntimeError, match="contiguous|index"):
        cache_wikipedia._existing_shards(tmp_path)


def test_existing_shards_reject_nonnumeric_indices(tmp_path):
    values = np.ones((2, 3), dtype=np.float32)
    np.save(tmp_path / "emb_bad.npy", values)
    pd.DataFrame({"_id": ["a", "b"], "url": ["u1", "u2"], "title": ["t1", "t2"]}).to_parquet(
        tmp_path / "meta_bad.parquet", index=False
    )

    with pytest.raises(RuntimeError, match="filename|index"):
        cache_wikipedia._existing_shards(tmp_path)


def test_inspect_existing_rejects_partial_nonfinal_shard(tmp_path):
    _write_shard(tmp_path, 0, rows=1)
    _write_shard(tmp_path, 1, rows=2)

    with pytest.raises(RuntimeError, match="only the final"):
        cache_wikipedia._inspect_existing(tmp_path, shard_size=2)


def test_inspect_existing_requires_all_metadata_fields(tmp_path):
    np.save(tmp_path / "emb_0000.npy", np.ones((2, 3), dtype=np.float32))
    pd.DataFrame({"_id": ["a", "b"]}).to_parquet(tmp_path / "meta_0000.parquet", index=False)

    with pytest.raises(RuntimeError, match="metadata|url|title"):
        cache_wikipedia._inspect_existing(tmp_path, shard_size=2)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_inspect_existing_rejects_empty_embedding_axes(tmp_path, shape):
    np.save(tmp_path / "emb_0000.npy", np.empty(shape, dtype=np.float32))
    pd.DataFrame(
        {
            "_id": [f"row-{row}" for row in range(shape[0])],
            "url": [f"url-{row}" for row in range(shape[0])],
            "title": [f"title-{row}" for row in range(shape[0])],
        }
    ).to_parquet(tmp_path / "meta_0000.parquet", index=False)

    with pytest.raises(RuntimeError, match="empty|invalid"):
        cache_wikipedia._inspect_existing(tmp_path, shard_size=2)


def test_main_rejects_manifest_source_mismatch_before_complete_return(tmp_path, monkeypatch):
    _write_shard(tmp_path, 0)
    records, _, _ = cache_wikipedia._inspect_existing(tmp_path, shard_size=2)
    original_args = _args(tmp_path)
    cache_wikipedia._atomic_write_json(
        tmp_path / "manifest.json", _manifest(original_args, records)
    )
    resumed_args = _args(tmp_path, config="de")
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: resumed_args)

    with pytest.raises(RuntimeError, match="manifest|config|source"):
        cache_wikipedia.main()


def test_main_rejects_manifest_with_mutable_resolved_revision(tmp_path, monkeypatch):
    _write_shard(tmp_path, 0)
    records, _, _ = cache_wikipedia._inspect_existing(tmp_path, shard_size=2)
    args = _args(tmp_path)
    manifest = _manifest(args, records)
    manifest["resolved_revision"] = "main"
    cache_wikipedia._atomic_write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: args)

    with pytest.raises(RuntimeError, match="exact resolved revision|pinned"):
        cache_wikipedia.main()


def test_main_accepts_complete_pinned_cache_without_rewriting_it(tmp_path, monkeypatch):
    _write_shard(tmp_path, 0)
    records, _, _ = cache_wikipedia._inspect_existing(tmp_path, shard_size=2)
    args = _args(tmp_path)
    cache_wikipedia._atomic_write_json(tmp_path / "manifest.json", _manifest(args, records))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: args)
    monkeypatch.setattr(
        cache_wikipedia,
        "_resolved_revision",
        lambda *_: pytest.fail("a pinned resume must not resolve mutable main again"),
    )

    assert cache_wikipedia.main() == 0

    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


def test_main_resumes_from_manifest_revision_and_appends_next_shard(tmp_path, monkeypatch):
    _write_shard(tmp_path, 0)
    records, _, _ = cache_wikipedia._inspect_existing(tmp_path, shard_size=2)
    initial_args = _args(tmp_path)
    cache_wikipedia._atomic_write_json(tmp_path / "manifest.json", _manifest(initial_args, records))
    resumed_args = _args(tmp_path, n_rows=4)
    calls = {}

    class Stream:
        def __init__(self, rows):
            self.rows = rows

        def skip(self, count):
            calls["skip"] = count
            return Stream(self.rows[count:])

        def __iter__(self):
            return iter(self.rows)

    rows = [
        {
            "_id": f"remote-{index}",
            "url": f"https://example.test/remote/{index}",
            "title": f"remote title {index}",
            "emb": [float(index), float(index + 1), float(index + 2)],
        }
        for index in range(4)
    ]

    def load_dataset(*args, **kwargs):
        calls["revision"] = kwargs["revision"]
        return Stream(rows)

    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: resumed_args)
    monkeypatch.setattr(
        cache_wikipedia,
        "_resolved_revision",
        lambda *_: pytest.fail("resume must reuse the stored exact revision"),
    )
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))

    assert cache_wikipedia.main() == 0

    assert calls == {"revision": "a" * 40, "skip": 2}
    np.testing.assert_array_equal(
        np.load(tmp_path / "emb_0001.npy"),
        np.asarray([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cached_rows"] == 4
    assert manifest["resolved_revision"] == "a" * 40
    assert [record["index"] for record in manifest["shards"]] == [0, 1]


def test_main_rejects_existing_shards_without_manifest(tmp_path, monkeypatch):
    _write_shard(tmp_path, 0)
    args = _args(tmp_path)
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: args)

    with pytest.raises(RuntimeError, match="without a manifest|cannot be proven"):
        cache_wikipedia.main()


def test_main_rejects_files_changed_since_manifest(tmp_path, monkeypatch):
    _write_shard(tmp_path, 0)
    records, _, _ = cache_wikipedia._inspect_existing(tmp_path, shard_size=2)
    args = _args(tmp_path)
    cache_wikipedia._atomic_write_json(tmp_path / "manifest.json", _manifest(args, records))
    np.save(tmp_path / "emb_0000.npy", np.full((2, 3), 7.0, dtype=np.float32))
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: args)

    with pytest.raises(RuntimeError, match="manifest|hash|changed"):
        cache_wikipedia.main()


def test_main_refuses_unpinned_new_stream(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: args)
    monkeypatch.setattr(cache_wikipedia, "_resolved_revision", lambda *_: None)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: ()),
    )

    with pytest.raises(RuntimeError, match="resolve|revision|pin"):
        cache_wikipedia.main()


def test_main_pins_manifest_before_reading_first_row(tmp_path, monkeypatch):
    args = _args(tmp_path)
    resolved_revision = "b" * 40
    monkeypatch.setattr(cache_wikipedia, "parse_args", lambda: args)
    monkeypatch.setattr(cache_wikipedia, "_resolved_revision", lambda *_: resolved_revision)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: ()),
    )

    with pytest.raises(RuntimeError, match="dataset ended"):
        cache_wikipedia.main()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["resolved_revision"] == resolved_revision
    assert manifest["cached_rows"] == 0
    assert manifest["shards"] == []
