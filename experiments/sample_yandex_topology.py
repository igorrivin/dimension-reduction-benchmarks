#!/usr/bin/env python3
"""Prepare a small, aligned Deep1B sample for topology evaluation.

The source ``.fbin`` and every NPY layout are validated in bounded row blocks.
One deterministic set of source rows is then copied into an immutable (no
overwrite), atomically published output directory.  The resulting files can be
passed directly to :mod:`experiments.evaluate_large_topology`.

Example
-------
::

    python experiments/sample_yandex_topology.py \
      --input-fbin artifacts/data/yandex-deep1b/deep-base-prefix-10m.fbin \
      --layout autoencoder=artifacts/yandex-deep1b/ae-10m/layout.npy \
      --layout umap=artifacts/yandex-deep1b/umap-10m/layout.npy \
      --layout dire=artifacts/yandex-deep1b/dire-10m/layout.npy \
      --sample-size 20000 --seed 20260713 \
      --output-dir artifacts/yandex-deep1b/topology-10m-d8/prepared
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from experiments.yandex_fbin import FBIN_HEADER_BYTES, read_fbin_header
except ModuleNotFoundError:  # direct ``python experiments/script.py`` execution
    from yandex_fbin import FBIN_HEADER_BYTES, read_fbin_header

_METHOD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class NpyInfo:
    """Strict metadata needed for bounded positional reads from a C-order NPY."""

    path: Path
    shape: tuple[int, int]
    dtype: np.dtype
    data_offset: int
    expected_size: int

    @property
    def row_bytes(self) -> int:
        return self.shape[1] * self.dtype.itemsize


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _requested_and_resolved(path: str | os.PathLike[str]) -> tuple[Path, Path]:
    requested = Path(path).expanduser().absolute()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"input path does not resolve to a file: {requested}: {exc}") from exc
    if not resolved.is_file():
        raise ValueError(f"input path is not a regular file: {resolved}")
    return requested, resolved


def _snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _require_unchanged(path: Path, before: dict[str, int]) -> None:
    after = _snapshot(path)
    if after != before:
        raise RuntimeError(f"input changed while it was being sampled: {path}")


def sha256_file(path: str | os.PathLike[str], block_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without materializing it."""
    block_bytes = _positive_integer(block_bytes, name="block_bytes")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def index_sha256(indices: np.ndarray) -> str:
    """Hash indices in the topology evaluator's canonical representation."""
    values = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def deterministic_indices(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    """Return sorted, unique, deterministic uniformly sampled row indices."""
    n_rows = _positive_integer(n_rows, name="n_rows")
    sample_size = _positive_integer(sample_size, name="sample_size")
    if sample_size > n_rows:
        raise ValueError(f"sample_size {sample_size} exceeds source rows {n_rows}")
    if sample_size == n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False).astype(np.int64))


def _parse_layouts(specifications: list[str]) -> dict[str, tuple[Path, Path]]:
    if not specifications:
        raise ValueError("at least one --layout METHOD=PATH is required")
    result: dict[str, tuple[Path, Path]] = {}
    resolved_paths: set[Path] = set()
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"layout must be METHOD=PATH, found {specification!r}")
        method, raw_path = specification.split("=", 1)
        method = method.strip()
        if not _METHOD_RE.fullmatch(method) or method in result:
            raise ValueError(f"layout method must be safe, non-empty, and unique: {method!r}")
        requested, resolved = _requested_and_resolved(raw_path)
        if resolved.suffix.lower() != ".npy":
            raise ValueError(f"layout {method!r} must be an NPY file: {resolved}")
        if resolved in resolved_paths:
            raise ValueError(f"layout paths must be distinct; duplicate target: {resolved}")
        resolved_paths.add(resolved)
        result[method] = (requested, resolved)
    return result


def _scan_fbin(path: Path, *, block_rows: int, l2_atol: float) -> dict:
    """Hash and validate an entire fbin with one bounded application buffer."""
    block_rows = _positive_integer(block_rows, name="validation_block_rows")
    if not np.isfinite(l2_atol) or l2_atol < 0:
        raise ValueError("l2_atol must be finite and non-negative")
    info = read_fbin_header(path)
    digest = hashlib.sha256()
    maximum_l2_error = 0.0
    minimum_norm = float("inf")
    maximum_norm = 0.0
    fd = os.open(path, os.O_RDONLY)
    try:
        header = os.pread(fd, FBIN_HEADER_BYTES, 0)
        if len(header) != FBIN_HEADER_BYTES:
            raise RuntimeError(f"short fbin header read from {path}")
        digest.update(header)
        for start in range(0, info.rows, block_rows):
            rows = min(block_rows, info.rows - start)
            expected = rows * info.row_bytes
            raw = os.pread(fd, expected, FBIN_HEADER_BYTES + start * info.row_bytes)
            if len(raw) != expected:
                raise RuntimeError(f"short fbin data read at row {start} from {path}")
            digest.update(raw)
            values = np.frombuffer(raw, dtype="<f4").reshape(rows, info.dimension)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"reference fbin contains non-finite values near row {start}")
            squared = np.einsum("ij,ij->i", values, values, dtype=np.float64)
            norms = np.sqrt(squared)
            minimum_norm = min(minimum_norm, float(norms.min()))
            maximum_norm = max(maximum_norm, float(norms.max()))
            maximum_l2_error = max(
                maximum_l2_error,
                float(np.max(np.abs(norms - 1.0))),
            )
    finally:
        os.close(fd)
    if maximum_l2_error > l2_atol:
        raise ValueError(
            "reference fbin is not row-L2 normalized: maximum norm error "
            f"{maximum_l2_error:.6g} exceeds {l2_atol:.6g}"
        )
    return {
        "sha256": digest.hexdigest(),
        "shape": [info.rows, info.dimension],
        "dtype": "float32",
        "size_bytes": info.expected_size,
        "row_l2_minimum": minimum_norm,
        "row_l2_maximum": maximum_norm,
        "row_l2_maximum_error": maximum_l2_error,
    }


def _inspect_npy(path: Path) -> NpyInfo:
    """Read a numeric C-order NPY header and require an exact physical length."""
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY format version {version}")
            data_offset = stream.tell()
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not inspect NPY layout {path}: {exc}") from exc
    dtype = np.dtype(dtype)
    if len(shape) != 2:
        raise ValueError(f"layout must be a 2-D NPY array, found shape {shape}: {path}")
    if fortran_order:
        raise ValueError(f"layout must use C row order for bounded sampling: {path}")
    if dtype.hasobject:
        raise ValueError(f"layout cannot contain object values: {path}")
    expected_size = data_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ValueError(
            f"layout NPY has an invalid physical length: expected {expected_size}, "
            f"found {observed_size}: {path}"
        )
    return NpyInfo(
        path=path,
        shape=(int(shape[0]), int(shape[1])),
        dtype=dtype,
        data_offset=data_offset,
        expected_size=expected_size,
    )


def _scan_layout(info: NpyInfo, *, block_rows: int) -> dict:
    """Hash and fully finite-check an NPY using bounded positional reads."""
    block_rows = _positive_integer(block_rows, name="validation_block_rows")
    minimum = float("inf")
    maximum = float("-inf")
    digest = hashlib.sha256()
    fd = os.open(info.path, os.O_RDONLY)
    try:
        header = os.pread(fd, info.data_offset, 0)
        if len(header) != info.data_offset:
            raise RuntimeError(f"short NPY header read from {info.path}")
        digest.update(header)
        for start in range(0, info.shape[0], block_rows):
            rows = min(block_rows, info.shape[0] - start)
            expected = rows * info.row_bytes
            raw = os.pread(fd, expected, info.data_offset + start * info.row_bytes)
            if len(raw) != expected:
                raise RuntimeError(f"short NPY data read at row {start} from {info.path}")
            digest.update(raw)
            block = np.frombuffer(raw, dtype=info.dtype).reshape(rows, info.shape[1])
            if not np.all(np.isfinite(block)):
                raise ValueError(f"layout contains non-finite values near row {start}: {info.path}")
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
    finally:
        os.close(fd)
    return {
        "sha256": digest.hexdigest(),
        "shape": list(info.shape),
        "dtype": str(info.dtype),
        "size_bytes": info.expected_size,
        "data_offset": info.data_offset,
        "minimum": minimum,
        "maximum": maximum,
    }


def _open_layouts(
    paths: dict[str, tuple[Path, Path]], n_rows: int, block_rows: int
) -> tuple[dict[str, NpyInfo], dict[str, dict], int]:
    arrays: dict[str, NpyInfo] = {}
    provenance: dict[str, dict] = {}
    target_dimension: int | None = None
    for method, (requested, resolved) in sorted(paths.items()):
        before = _snapshot(resolved)
        info = _inspect_npy(resolved)
        if info.shape[0] != n_rows or info.shape[1] < 1:
            raise ValueError(
                f"layout {method!r} has shape {info.shape}; expected "
                f"({n_rows}, target_dimension)"
            )
        if not np.issubdtype(info.dtype, np.floating):
            raise ValueError(f"layout {method!r} must have floating dtype")
        if target_dimension is None:
            target_dimension = int(info.shape[1])
        elif int(info.shape[1]) != target_dimension:
            raise ValueError(
                f"layout {method!r} has target dimension {info.shape[1]}, "
                f"expected {target_dimension}"
            )
        scanned = _scan_layout(info, block_rows=block_rows)
        _require_unchanged(resolved, before)
        provenance[method] = {
            "requested_path": str(requested),
            "resolved_path": str(resolved),
            "snapshot": before,
            **scanned,
        }
        arrays[method] = info
    assert target_dimension is not None
    return arrays, provenance, target_dimension


def _write_reference_sample(
    output: Path,
    source: Path,
    indices: np.ndarray,
    *,
    dimension: int,
) -> None:
    array = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=(indices.size, dimension),
    )
    row_bytes = dimension * np.dtype("<f4").itemsize
    fd = os.open(source, os.O_RDONLY)
    try:
        for output_row, source_row in enumerate(indices):
            raw = os.pread(
                fd,
                row_bytes,
                FBIN_HEADER_BYTES + int(source_row) * row_bytes,
            )
            if len(raw) != row_bytes:
                raise RuntimeError(f"short fbin row read at source row {int(source_row)}")
            array[output_row] = np.frombuffer(raw, dtype="<f4")
        array.flush()
    finally:
        os.close(fd)
        del array
    with output.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_layout_sample(
    output: Path,
    source: NpyInfo,
    indices: np.ndarray,
    *,
    copy_block_rows: int,
) -> None:
    copy_block_rows = _positive_integer(copy_block_rows, name="copy_block_rows")
    array = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=source.dtype,
        shape=(indices.size, source.shape[1]),
    )
    fd = os.open(source.path, os.O_RDONLY)
    try:
        for start in range(0, indices.size, copy_block_rows):
            selected = indices[start : start + copy_block_rows]
            block = np.empty((selected.size, source.shape[1]), dtype=source.dtype)
            for output_row, source_row in enumerate(selected):
                raw = os.pread(
                    fd,
                    source.row_bytes,
                    source.data_offset + int(source_row) * source.row_bytes,
                )
                if len(raw) != source.row_bytes:
                    raise RuntimeError(
                        f"short NPY row read at source row {int(source_row)}: {source.path}"
                    )
                block[output_row] = np.frombuffer(raw, dtype=source.dtype)
            array[start : start + selected.size] = block
        array.flush()
    finally:
        os.close(fd)
        del array
    with output.open("rb") as stream:
        os.fsync(stream.fileno())


def _output_record(path: Path) -> dict:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"published sample candidate contains non-finite values: {path}")
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def run(args: argparse.Namespace) -> dict:
    """Validate inputs, prepare aligned samples, and return their manifest."""
    input_requested, input_resolved = _requested_and_resolved(args.input_fbin)
    if input_resolved.suffix.lower() != ".fbin":
        raise ValueError(f"reference input must be an .fbin file: {input_resolved}")
    layouts = _parse_layouts(args.layout)
    if input_resolved in {resolved for _, resolved in layouts.values()}:
        raise ValueError("reference and layout inputs must be distinct files")

    info = read_fbin_header(input_resolved)
    sample_size = _positive_integer(args.sample_size, name="sample_size")
    validation_block_rows = _positive_integer(
        args.validation_block_rows, name="validation_block_rows"
    )
    copy_block_rows = _positive_integer(args.copy_block_rows, name="copy_block_rows")
    indices = deterministic_indices(info.rows, sample_size, args.seed)

    output = Path(args.output_dir).expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if temporary.exists():  # practically impossible, but keeps ownership explicit
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()

    input_before = _snapshot(input_resolved)
    try:
        reference_provenance = _scan_fbin(
            input_resolved,
            block_rows=validation_block_rows,
            l2_atol=float(args.l2_atol),
        )
        layout_arrays, layout_provenance, target_dimension = _open_layouts(
            layouts,
            info.rows,
            validation_block_rows,
        )

        reference_path = temporary / "reference.npy"
        source_indices_path = temporary / "source_indices.npy"
        _write_reference_sample(
            reference_path,
            input_resolved,
            indices,
            dimension=info.dimension,
        )
        np.save(source_indices_path, indices, allow_pickle=False)
        with source_indices_path.open("rb") as stream:
            os.fsync(stream.fileno())

        output_layouts: dict[str, dict] = {}
        for method, array in sorted(layout_arrays.items()):
            path = temporary / f"layout-{method}.npy"
            _write_layout_sample(
                path,
                array,
                indices,
                copy_block_rows=copy_block_rows,
            )
            output_layouts[method] = _output_record(path)
            _require_unchanged(layouts[method][1], layout_provenance[method]["snapshot"])
        _require_unchanged(input_resolved, input_before)

        reference_record = _output_record(reference_path)
        source_index_record = {
            **_output_record(source_indices_path),
            "index_sha256": index_sha256(indices),
            "minimum": int(indices.min()),
            "maximum": int(indices.max()),
            "unique": int(np.unique(indices).size),
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "sample_size": sample_size,
                "seed": int(args.seed),
                "sampling": "uniform_without_replacement_sorted",
                "validation_block_rows": validation_block_rows,
                "copy_block_rows": copy_block_rows,
                "l2_atol": float(args.l2_atol),
                "target_dimension": target_dimension,
            },
            "inputs": {
                "reference": {
                    "requested_path": str(input_requested),
                    "resolved_path": str(input_resolved),
                    "snapshot": input_before,
                    **reference_provenance,
                },
                "layouts": layout_provenance,
            },
            "outputs": {
                "reference": reference_record,
                "source_indices": source_index_record,
                "layouts": output_layouts,
            },
            "publication": "atomic_directory_rename_no_overwrite",
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.rename(output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fbin", required=True)
    parser.add_argument(
        "--layout",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="aligned full-corpus NPY layout; repeat once per method",
    )
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    parser.add_argument("--validation-block-rows", type=int, default=65_536)
    parser.add_argument("--copy-block-rows", type=int, default=4_096)
    parser.add_argument("--l2-atol", type=float, default=5e-4)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    manifest = run(build_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
