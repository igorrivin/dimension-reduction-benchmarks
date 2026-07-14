#!/usr/bin/env python3
"""Cache a provenance-pinned prefix of Cohere Wikipedia embeddings.

Unlike the older embeddings-only cache, this writer retains stable row identifiers,
titles, and URLs and records the resolved Hugging Face revision plus per-shard hashes.
That makes a multi-hour large-scale benchmark auditable and resumable without caching
the paragraph text itself.

Example
-------
::

    conda run -n rapids-26.06 python experiments/cache_wikipedia_embeddings.py \
      --n-rows 1000000 \
      --cache-dir artifacts/data/cohere-wikipedia-english-1m
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_DATASET = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
DEFAULT_SHARD_SIZE = 100_000
METADATA_FIELDS = ("_id", "url", "title")
MANIFEST_SCHEMA_VERSION = 1
_EXACT_REVISION = re.compile(r"[0-9a-fA-F]{40}")


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_field(row: dict) -> str:
    for candidate in ("emb", "embedding", "embeddings", "vector"):
        if candidate in row:
            return candidate
    raise KeyError(f"no embedding field in row keys={list(row)}")


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_parquet(path: Path, rows: dict[str, list]) -> None:
    import pandas as pd

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


@contextlib.contextmanager
def _cache_lock(cache: Path):
    """Prevent two writers from mutating the same cache concurrently."""
    lock_path = cache / ".cache.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"cache is already locked by another writer: {lock_path}; "
            "remove the lock only after confirming that no writer is running"
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"pid={os.getpid()} created={datetime.now(UTC).isoformat()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _indexed_paths(cache: Path, prefix: str, suffix: str) -> dict[int, Path]:
    paths = {}
    for path in cache.glob(f"{prefix}*{suffix}"):
        token = path.name.removeprefix(prefix).removesuffix(suffix)
        if not token.isdigit() or token != f"{int(token):04d}":
            raise RuntimeError(f"invalid cache shard filename: {path.name}")
        index = int(token)
        if index in paths:
            raise RuntimeError(f"duplicate cache shard index {index}")
        paths[index] = path
    return paths


def _existing_shards(cache: Path) -> list[tuple[Path, Path]]:
    embeddings = _indexed_paths(cache, "emb_", ".npy")
    metadata = _indexed_paths(cache, "meta_", ".parquet")
    if embeddings.keys() != metadata.keys():
        raise RuntimeError("embedding and metadata shard indices do not match")
    indices = sorted(embeddings)
    if indices != list(range(len(indices))):
        raise RuntimeError("cache shard indices must be contiguous and start at zero")
    return [(embeddings[index], metadata[index]) for index in indices]


def _inspect_existing(cache: Path, shard_size: int) -> tuple[list[dict], int, int | None]:
    records = []
    rows = 0
    dimension = None
    shards = _existing_shards(cache)
    for index, (embedding_path, metadata_path) in enumerate(shards):
        array = np.load(embedding_path, mmap_mode="r")
        if (
            array.ndim != 2
            or array.dtype != np.float32
            or array.shape[0] == 0
            or array.shape[1] == 0
        ):
            raise RuntimeError(
                f"invalid embedding shard {embedding_path}: {array.shape} {array.dtype}"
            )
        if dimension is None:
            dimension = int(array.shape[1])
        elif array.shape[1] != dimension:
            raise RuntimeError("embedding dimensions differ across cached shards")
        if index + 1 < len(shards) and array.shape[0] != shard_size:
            raise RuntimeError("only the final cached shard may be partial")
        import pandas as pd

        try:
            metadata = pd.read_parquet(metadata_path, columns=list(METADATA_FIELDS))
        except Exception as exc:
            raise RuntimeError(
                f"invalid metadata shard {metadata_path}; required fields are " f"{METADATA_FIELDS}"
            ) from exc
        meta_rows = len(metadata)
        if meta_rows != array.shape[0]:
            raise RuntimeError(f"row mismatch between {embedding_path} and {metadata_path}")
        records.append(
            {
                "index": index,
                "rows": int(array.shape[0]),
                "embedding_file": embedding_path.name,
                "embedding_sha256": _sha256(embedding_path),
                "metadata_file": metadata_path.name,
                "metadata_sha256": _sha256(metadata_path),
            }
        )
        rows += int(array.shape[0])
    return records, rows, dimension


def _read_manifest(cache: Path) -> dict | None:
    path = cache / "manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read cache manifest {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"cache manifest {path} must contain a JSON object")
    return payload


def _manifest_payload(
    args,
    *,
    resolved_revision: str,
    embedding_field: str | None,
    dimension: int | None,
    records: list[dict],
    cached_rows: int,
) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_or_updated_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "embedding_field": embedding_field,
        "metadata_fields": list(METADATA_FIELDS),
        "embedding_dimension": dimension,
        "dtype": "float32",
        "shard_size": args.shard_size,
        "target_rows": args.n_rows,
        "cached_rows": cached_rows,
        "shards": records,
    }


def _validate_manifest(
    manifest: dict,
    args,
    records: list[dict],
    cached_rows: int,
    dimension: int | None,
) -> str:
    """Validate cache identity and file hashes before resuming or returning success."""
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "requested_revision": args.revision,
        "metadata_fields": list(METADATA_FIELDS),
        "dtype": "float32",
        "shard_size": args.shard_size,
        "cached_rows": cached_rows,
        "shards": records,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(
                f"cache manifest mismatch for {name!r}: "
                f"recorded {manifest.get(name)!r}, observed/requested {value!r}"
            )
    if manifest.get("embedding_dimension") != dimension:
        raise RuntimeError("cache manifest embedding dimension does not match the cached shards")
    field = manifest.get("embedding_field")
    if records and (not isinstance(field, str) or not field):
        raise RuntimeError("cache manifest has no embedding field for existing shards")
    resolved_revision = manifest.get("resolved_revision")
    if not isinstance(resolved_revision, str) or not _EXACT_REVISION.fullmatch(resolved_revision):
        raise RuntimeError("cache manifest has no exact resolved revision and is not pinned")
    return resolved_revision


def _resolved_revision(dataset: str, revision: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset, revision=revision).sha
    except Exception as exc:
        print(f"[cache] could not resolve dataset revision: {type(exc).__name__}: {exc}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--n-rows", type=int, default=1_000_000)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--cache-dir", default="artifacts/data/cohere-wikipedia-english-1m", type=Path
    )
    args = parser.parse_args()
    if args.n_rows < 1 or args.shard_size < 1:
        parser.error("n-rows and shard-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    cache = args.cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with _cache_lock(cache):
        return _populate_cache(args, cache)


def _populate_cache(args, cache: Path) -> int:
    records, cached_rows, dimension = _inspect_existing(cache, args.shard_size)
    manifest = _read_manifest(cache)
    if manifest is not None:
        resolved_revision = _validate_manifest(manifest, args, records, cached_rows, dimension)
        embedding_field = manifest.get("embedding_field")
    else:
        if records:
            raise RuntimeError(
                "cached shards exist without a manifest; their source revision cannot be proven"
            )
        resolved_revision = _resolved_revision(args.dataset, args.revision)
        if resolved_revision is None:
            if _EXACT_REVISION.fullmatch(args.revision):
                resolved_revision = args.revision
            else:
                raise RuntimeError(
                    f"could not resolve revision {args.revision!r}; refusing to stream "
                    "an unpinned dataset"
                )
        if not _EXACT_REVISION.fullmatch(resolved_revision):
            raise RuntimeError(
                f"resolved revision {resolved_revision!r} is not an exact dataset commit"
            )
        embedding_field = None
        _atomic_write_json(
            cache / "manifest.json",
            _manifest_payload(
                args,
                resolved_revision=resolved_revision,
                embedding_field=None,
                dimension=None,
                records=[],
                cached_rows=0,
            ),
        )

    if cached_rows >= args.n_rows:
        print(f"[cache] already have {cached_rows:,} rows in {cache}")
        return 0
    if records and records[-1]["rows"] != args.shard_size:
        raise RuntimeError("cannot resume after a partial final shard; remove or complete it first")

    from datasets import load_dataset

    print(
        f"[cache] streaming {args.dataset}[{args.config}]@{resolved_revision} "
        f"to {cache} ({cached_rows:,}/{args.n_rows:,} rows present)",
        flush=True,
    )
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        revision=resolved_revision,
    )
    if cached_rows:
        dataset = dataset.skip(cached_rows)
    iterator = iter(dataset)
    started = time.perf_counter()
    written = 0
    shard_index = len(records)

    while cached_rows + written < args.n_rows:
        capacity = min(args.shard_size, args.n_rows - cached_rows - written)
        try:
            first = next(iterator)
        except StopIteration:
            break
        field = _embedding_field(first)
        if embedding_field is None:
            embedding_field = field
        elif field != embedding_field:
            raise ValueError(f"embedding field changed from {embedding_field!r} to {field!r}")
        first_vector = np.asarray(first[field], dtype=np.float32)
        if first_vector.ndim != 1:
            raise ValueError(f"embedding field {field!r} is not one-dimensional")
        if dimension is None:
            dimension = int(first_vector.size)
        if first_vector.size != dimension:
            raise ValueError("embedding dimension changed within the stream")

        embeddings = np.empty((capacity, dimension), dtype=np.float32)
        metadata = {name: [] for name in METADATA_FIELDS}
        count = 0
        row = first
        while count < capacity:
            vector = np.asarray(row[field], dtype=np.float32)
            if vector.shape != (dimension,):
                raise ValueError(f"row embedding has shape {vector.shape}; expected {(dimension,)}")
            embeddings[count] = vector
            for name in METADATA_FIELDS:
                value = row.get(name)
                metadata[name].append(None if value is None else str(value))
            count += 1
            if count >= capacity:
                break
            try:
                row = next(iterator)
            except StopIteration:
                break

        embeddings = embeddings[:count]
        embedding_path = cache / f"emb_{shard_index:04d}.npy"
        metadata_path = cache / f"meta_{shard_index:04d}.parquet"
        _atomic_save_npy(embedding_path, embeddings)
        _atomic_write_parquet(metadata_path, metadata)
        record = {
            "index": shard_index,
            "rows": count,
            "embedding_file": embedding_path.name,
            "embedding_sha256": _sha256(embedding_path),
            "metadata_file": metadata_path.name,
            "metadata_sha256": _sha256(metadata_path),
        }
        records.append(record)
        written += count
        shard_index += 1
        elapsed = time.perf_counter() - started
        print(
            f"[cache] wrote shard {record['index']:04d}: {count:,} rows; "
            f"total={cached_rows + written:,}, new-rate={written / elapsed:,.0f} rows/s",
            flush=True,
        )
        _atomic_write_json(
            cache / "manifest.json",
            _manifest_payload(
                args,
                resolved_revision=resolved_revision,
                embedding_field=embedding_field,
                dimension=dimension,
                records=records,
                cached_rows=cached_rows + written,
            ),
        )
        if count < capacity:
            break

    if cached_rows + written < args.n_rows:
        raise RuntimeError(
            f"dataset ended after {cached_rows + written:,} rows; requested {args.n_rows:,}"
        )
    print(f"[cache] complete in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
