#!/usr/bin/env python3
"""Strict, bounded-memory readers for Yandex/Big-ANN ``.fbin`` files.

The format is deliberately small: an eight-byte little-endian header containing
``(row_count, dimension)`` as unsigned 32-bit integers, followed by a row-major
array of little-endian float32 values.  These helpers validate the complete file
before exposing a read-only memory map; a truncated file or an unexpected suffix
therefore cannot silently become benchmark input.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FBIN_HEADER_BYTES = 8
FBIN_DTYPE = np.dtype("<f4")
_HEADER = struct.Struct("<II")


@dataclass(frozen=True)
class FbinInfo:
    """Validated metadata for one ``.fbin`` file."""

    path: Path
    rows: int
    dimension: int
    expected_size: int

    @property
    def row_bytes(self) -> int:
        return self.dimension * FBIN_DTYPE.itemsize

    @property
    def dim(self) -> int:
        """Short alias used by benchmark harnesses."""

        return self.dimension

    @property
    def dtype(self) -> np.dtype:
        return FBIN_DTYPE

    @property
    def data_bytes(self) -> int:
        return self.expected_size - FBIN_HEADER_BYTES


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def inspect_fbin(path: str | os.PathLike[str]) -> FbinInfo:
    """Read and strictly validate an ``.fbin`` header and file length.

    The header is always decoded as little endian.  Both axes must be nonzero,
    and the physical file size must equal (not merely exceed) the size declared
    by the header.
    """

    resolved_path = Path(path)
    try:
        file_size = resolved_path.stat().st_size
        with resolved_path.open("rb") as handle:
            header = handle.read(FBIN_HEADER_BYTES)
    except OSError as exc:
        raise ValueError(f"could not inspect fbin file {resolved_path}: {exc}") from exc

    if len(header) != FBIN_HEADER_BYTES:
        raise ValueError(
            f"invalid fbin header in {resolved_path}: expected {FBIN_HEADER_BYTES} bytes, "
            f"found {len(header)}"
        )
    rows, dimension = _HEADER.unpack(header)
    if rows == 0 or dimension == 0:
        raise ValueError(
            f"invalid fbin shape in {resolved_path}: rows={rows}, dimension={dimension}"
        )
    expected_size = FBIN_HEADER_BYTES + rows * dimension * FBIN_DTYPE.itemsize
    if file_size != expected_size:
        kind = "truncated" if file_size < expected_size else "has trailing bytes"
        raise ValueError(
            f"invalid fbin length in {resolved_path}: file {kind}; "
            f"expected {expected_size} bytes, found {file_size}"
        )
    return FbinInfo(
        path=resolved_path,
        rows=int(rows),
        dimension=int(dimension),
        expected_size=int(expected_size),
    )


def _limited_rows(info: FbinInfo, rows: int | None) -> int:
    if rows is None:
        return info.rows
    limit = _positive_integer(rows, name="rows")
    if limit > info.rows:
        raise ValueError(f"requested {limit} rows from {info.path}, which contains {info.rows}")
    return limit


def open_fbin(path: str | os.PathLike[str], *, rows: int | None = None) -> np.memmap:
    """Return a zero-copy, read-only memory map of a validated ``.fbin`` file.

    ``rows`` may select a prefix without mapping or copying the remaining rows.
    The complete underlying file is still length-validated first.
    """

    info = inspect_fbin(path)
    selected_rows = _limited_rows(info, rows)
    array = np.memmap(
        info.path,
        dtype=FBIN_DTYPE,
        mode="r",
        offset=FBIN_HEADER_BYTES,
        shape=(selected_rows, info.dimension),
        order="C",
    )
    # ``mode='r'`` already enforces this; keep the invariant explicit if NumPy's
    # construction behavior ever changes.
    array.flags.writeable = False
    return array


def read_fbin_header(path: str | os.PathLike[str]) -> FbinInfo:
    """Compatibility name for strict metadata inspection.

    The returned object exposes ``rows``, ``dimension``, ``dtype``, and
    ``data_bytes``.  Despite the compact name, this validates the physical file
    length as well as decoding its header.
    """

    return inspect_fbin(path)


def open_fbin_memmap(path: str | os.PathLike[str], limit_rows: int | None = None) -> np.memmap:
    """Compatibility name for a read-only full-file or prefix memory map."""

    return open_fbin(path, rows=limit_rows)


def iter_fbin_blocks(
    path: str | os.PathLike[str],
    *,
    block_rows: int,
    rows: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield read-only, zero-copy row blocks from a validated ``.fbin`` file."""

    size = _positive_integer(block_rows, name="block_rows")
    array = open_fbin(path, rows=rows)
    for start in range(0, array.shape[0], size):
        block = array[start : start + size]
        block.flags.writeable = False
        yield block


__all__ = [
    "FBIN_DTYPE",
    "FBIN_HEADER_BYTES",
    "FbinInfo",
    "inspect_fbin",
    "iter_fbin_blocks",
    "open_fbin",
    "open_fbin_memmap",
    "read_fbin_header",
]
