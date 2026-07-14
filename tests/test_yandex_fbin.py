"""Safety and resumability tests for the Yandex/Big-ANN ``.fbin`` helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


yandex_fbin = _load_script("yandex_fbin", ROOT / "experiments" / "yandex_fbin.py")
cache_deep1b = _load_script("cache_yandex_deep1b", ROOT / "experiments" / "cache_yandex_deep1b.py")


def _fbin_bytes(values: np.ndarray, *, declared_rows: int | None = None) -> bytes:
    array = np.asarray(values, dtype="<f4", order="C")
    assert array.ndim == 2
    rows = array.shape[0] if declared_rows is None else declared_rows
    return struct.pack("<II", rows, array.shape[1]) + array.tobytes(order="C")


def _write_fbin(path: Path, values: np.ndarray) -> bytes:
    payload = _fbin_bytes(values)
    path.write_bytes(payload)
    return payload


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str], *, status: int = 206):
        super().__init__(body)
        self.headers = headers
        self.status = status


class RangeOpener:
    """In-memory HTTP Range endpoint with controllable protocol failures."""

    def __init__(
        self,
        source: bytes,
        *,
        etag: str | None = '"stable-object"',
        last_modified: str | None = "Mon, 13 Jul 2026 00:00:00 GMT",
    ):
        self.source = source
        self.etag = etag
        self.last_modified = last_modified
        self.requests: list[tuple[int, int, str | None]] = []
        self.truncate_once_at: int | None = None
        self.wrong_total: int | None = None
        self.wrong_start: int | None = None
        self.status = 206
        self.changed_etag_at: int | None = None
        self.invalid_content_length_at: int | None = None
        self.raise_at: int | None = None

    def __call__(self, request, *, timeout):
        assert timeout > 0
        assert request.get_header("Accept-encoding") == "identity"
        raw_range = request.get_header("Range")
        assert raw_range is not None and raw_range.startswith("bytes=")
        start, end = (int(value) for value in raw_range.removeprefix("bytes=").split("-"))
        if self.raise_at == start:
            raise URLError("simulated disconnect")
        self.requests.append((start, end, request.get_header("If-range")))
        body = self.source[start : end + 1]
        if self.truncate_once_at == start:
            body = body[:-1]
            self.truncate_once_at = None
        observed_start = start if self.wrong_start is None else self.wrong_start
        total = len(self.source) if self.wrong_total is None else self.wrong_total
        headers = {
            "Content-Range": f"bytes {observed_start}-{end}/{total}",
            "Content-Length": str(end - start + 1),
        }
        if self.invalid_content_length_at == start:
            headers["Content-Length"] = str(end - start)
        if self.etag is not None:
            headers["ETag"] = '"changed-object"' if self.changed_etag_at == start else self.etag
        if self.last_modified is not None:
            headers["Last-Modified"] = self.last_modified
        return FakeResponse(body, headers, status=self.status)


@pytest.fixture
def values() -> np.ndarray:
    return np.arange(30, dtype=np.float32).reshape(6, 5) / 7.0


def test_inspect_fbin_reports_strict_little_endian_metadata(tmp_path, values):
    path = tmp_path / "vectors.fbin"
    payload = _write_fbin(path, values)

    info = yandex_fbin.inspect_fbin(path)

    assert info.path == path
    assert info.rows == 6
    assert info.dimension == 5
    assert info.dtype == np.dtype("<f4")
    assert info.row_bytes == 20
    assert info.data_bytes == values.nbytes
    assert info.expected_size == len(payload)
    assert yandex_fbin.read_fbin_header(path) == info


@pytest.mark.parametrize("suffix", [b"", b"x", b"unwanted suffix"])
def test_inspect_fbin_rejects_truncation_and_trailing_bytes(tmp_path, suffix):
    path = tmp_path / "bad.fbin"
    complete = _fbin_bytes(np.ones((3, 2), dtype=np.float32))
    path.write_bytes(complete[:-1] if suffix == b"" else complete + suffix)

    with pytest.raises(ValueError, match="length|truncated|trailing"):
        yandex_fbin.inspect_fbin(path)


@pytest.mark.parametrize(
    "header", [b"", b"1234567", struct.pack("<II", 0, 3), struct.pack("<II", 2, 0)]
)
def test_inspect_fbin_rejects_short_header_and_empty_axes(tmp_path, header):
    path = tmp_path / "bad.fbin"
    path.write_bytes(header)

    with pytest.raises(ValueError, match="header|shape|length"):
        yandex_fbin.inspect_fbin(path)


def test_big_endian_header_is_not_misread_as_native_endian(tmp_path):
    path = tmp_path / "big-endian-header.fbin"
    path.write_bytes(struct.pack(">II", 2, 3) + np.ones((2, 3), dtype="<f4").tobytes())

    with pytest.raises(ValueError, match="length"):
        yandex_fbin.inspect_fbin(path)


def test_open_fbin_is_read_only_zero_copy_memmap_and_can_limit_rows(tmp_path, values):
    path = tmp_path / "vectors.fbin"
    _write_fbin(path, values)

    array = yandex_fbin.open_fbin(path, rows=4)
    compatibility_array = yandex_fbin.open_fbin_memmap(path, limit_rows=2)

    assert isinstance(array, np.memmap)
    assert array.shape == (4, 5)
    assert array.dtype == np.dtype("<f4")
    assert not array.flags.writeable
    np.testing.assert_array_equal(array, values[:4])
    np.testing.assert_array_equal(compatibility_array, values[:2])
    with pytest.raises(ValueError, match="read-only"):
        array[0, 0] = 99


@pytest.mark.parametrize("rows", [0, -1, 7, True, 1.5])
def test_open_fbin_rejects_invalid_row_limits(tmp_path, values, rows):
    path = tmp_path / "vectors.fbin"
    _write_fbin(path, values)

    with pytest.raises((TypeError, ValueError), match="rows|requested|positive|integer"):
        yandex_fbin.open_fbin(path, rows=rows)


def test_iter_fbin_blocks_yields_read_only_views_and_honors_limit(tmp_path, values):
    path = tmp_path / "vectors.fbin"
    _write_fbin(path, values)

    blocks = list(yandex_fbin.iter_fbin_blocks(path, block_rows=2, rows=5))

    assert [block.shape for block in blocks] == [(2, 5), (2, 5), (1, 5)]
    assert all(isinstance(block, np.memmap) for block in blocks)
    assert all(not block.flags.writeable for block in blocks)
    np.testing.assert_array_equal(np.concatenate(blocks), values[:5])


@pytest.mark.parametrize("block_rows", [0, -3, True, 2.5])
def test_iter_fbin_blocks_rejects_invalid_block_size(tmp_path, values, block_rows):
    path = tmp_path / "vectors.fbin"
    _write_fbin(path, values)

    with pytest.raises((TypeError, ValueError), match="block_rows|positive|integer"):
        list(yandex_fbin.iter_fbin_blocks(path, block_rows=block_rows))


def test_downloader_crops_base_1b_prefix_rewrites_header_and_records_provenance(tmp_path):
    source_values = np.arange(10, dtype=np.float32).reshape(5, 2)
    source = _fbin_bytes(source_values)
    opener = RangeOpener(source)
    output = tmp_path / "deep-prefix.fbin"

    manifest = cache_deep1b.download_fbin_prefix(
        cache_deep1b.DEFAULT_URL,
        output,
        rows=3,
        chunk_bytes=9,
        opener=opener,
    )

    expected = _fbin_bytes(source_values[:3])
    assert output.read_bytes() == expected
    assert output.read_bytes()[:8] == struct.pack("<II", 3, 2)
    np.testing.assert_array_equal(yandex_fbin.open_fbin(output), source_values[:3])
    assert manifest == json.loads(Path(f"{output}.manifest.json").read_text())
    assert manifest["source"] == {
        "url": cache_deep1b.DEFAULT_URL,
        "rows": 5,
        "dimension": 2,
        "file_bytes": len(source),
        "etag": '"stable-object"',
        "last_modified": "Mon, 13 Jul 2026 00:00:00 GMT",
        "header_sha256": hashlib.sha256(source[:8]).hexdigest(),
    }
    assert manifest["prefix"]["rows"] == 3
    assert manifest["prefix"]["payload_bytes"] == 24
    assert manifest["prefix"]["file_bytes"] == 32
    assert manifest["prefix"]["sha256"] == hashlib.sha256(expected).hexdigest()
    assert opener.requests[0][:2] == (0, 7)
    assert all(end <= len(expected) - 1 for start, end, _ in opener.requests if start >= 8)
    assert not Path(f"{output}.part").exists()
    assert not Path(f"{output}.part.json").exists()
    assert not Path(f"{output}.lock").exists()


def test_downloader_resumes_only_missing_range_after_interruption(tmp_path):
    source_values = np.arange(10, dtype=np.float32).reshape(5, 2)
    source = _fbin_bytes(source_values)
    output = tmp_path / "deep-prefix.fbin"
    interrupted = RangeOpener(source)
    interrupted.truncate_once_at = 16

    with pytest.raises(RuntimeError, match="truncated"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=3,
            chunk_bytes=8,
            opener=interrupted,
        )

    part = Path(f"{output}.part")
    assert not output.exists()
    assert part.stat().st_size == 16  # local header plus one complete eight-byte row
    assert Path(f"{output}.part.json").exists()
    resumed = RangeOpener(source)

    cache_deep1b.download_fbin_prefix(
        cache_deep1b.DEFAULT_URL,
        output,
        rows=3,
        chunk_bytes=8,
        opener=resumed,
    )

    assert resumed.requests[0][:2] == (0, 7)
    assert resumed.requests[1][:2] == (16, 23)
    assert resumed.requests[1][2] == '"stable-object"'
    assert output.read_bytes() == _fbin_bytes(source_values[:3])


def test_completed_download_is_idempotent_and_manifest_is_byte_deterministic(tmp_path):
    source = _fbin_bytes(np.arange(12, dtype=np.float32).reshape(6, 2))
    output = tmp_path / "prefix.fbin"
    cache_deep1b.download_fbin_prefix(
        cache_deep1b.DEFAULT_URL, output, rows=4, opener=RangeOpener(source)
    )
    manifest_path = Path(f"{output}.manifest.json")
    before = manifest_path.read_bytes()

    returned = cache_deep1b.download_fbin_prefix(
        cache_deep1b.DEFAULT_URL, output, rows=4, opener=RangeOpener(source)
    )

    assert manifest_path.read_bytes() == before
    assert returned == json.loads(before)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"status": 200}, "HTTP 206"),
        ({"wrong_start": 1}, "Content-Range"),
        ({"wrong_total": 999}, "length|disagree|changed"),
        ({"invalid_content_length_at": 0}, "Content-Length"),
    ],
)
def test_downloader_rejects_invalid_remote_header_range(tmp_path, mutation, message):
    opener = RangeOpener(_fbin_bytes(np.ones((5, 2), dtype=np.float32)))
    for field, value in mutation.items():
        setattr(opener, field, value)

    with pytest.raises(RuntimeError, match=message):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            tmp_path / "prefix.fbin",
            rows=2,
            opener=opener,
        )


def test_downloader_rejects_header_object_length_disagreement(tmp_path):
    # Header declares five rows, but the HTTP object's physical payload has four.
    source = _fbin_bytes(np.ones((4, 2), dtype=np.float32), declared_rows=5)

    with pytest.raises(RuntimeError, match="header and object length disagree"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            tmp_path / "prefix.fbin",
            rows=2,
            opener=RangeOpener(source),
        )


def test_downloader_rejects_identity_change_and_rolls_back_current_chunk(tmp_path):
    source = _fbin_bytes(np.ones((5, 2), dtype=np.float32))
    opener = RangeOpener(source)
    opener.changed_etag_at = 8
    output = tmp_path / "prefix.fbin"

    with pytest.raises(RuntimeError, match="identity changed|ETag"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=2,
            chunk_bytes=8,
            opener=opener,
        )

    assert not output.exists()
    assert Path(f"{output}.part").stat().st_size == 8


def test_downloader_refuses_unverifiable_cross_process_resume(tmp_path):
    source = _fbin_bytes(np.ones((5, 2), dtype=np.float32))
    output = tmp_path / "prefix.fbin"
    interrupted = RangeOpener(source, etag=None, last_modified=None)
    interrupted.truncate_once_at = 16
    with pytest.raises(RuntimeError, match="truncated"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=3,
            chunk_bytes=8,
            opener=interrupted,
        )

    with pytest.raises(RuntimeError, match="cannot safely resume"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=3,
            chunk_bytes=8,
            opener=RangeOpener(source, etag=None, last_modified=None),
        )


def test_downloader_rejects_resume_against_different_source_identity(tmp_path):
    source = _fbin_bytes(np.ones((5, 2), dtype=np.float32))
    output = tmp_path / "prefix.fbin"
    interrupted = RangeOpener(source)
    interrupted.truncate_once_at = 16
    with pytest.raises(RuntimeError, match="truncated"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=3,
            chunk_bytes=8,
            opener=interrupted,
        )

    changed = RangeOpener(source, etag='"replacement"')
    with pytest.raises(RuntimeError, match="does not match|provenance"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=3,
            chunk_bytes=8,
            opener=changed,
        )


def test_downloader_never_substitutes_standalone_base_10m(tmp_path):
    opener = RangeOpener(_fbin_bytes(np.ones((5, 2), dtype=np.float32)))

    with pytest.raises(ValueError, match="base.10M|base.1B|ground truth"):
        cache_deep1b.download_fbin_prefix(
            "https://storage.example/DEEP/base.10M.fbin",
            tmp_path / "prefix.fbin",
            rows=2,
            opener=opener,
        )

    assert opener.requests == []


def test_downloader_rejects_more_rows_than_remote_source(tmp_path):
    source = _fbin_bytes(np.ones((5, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="requested 6 rows|contains 5"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            tmp_path / "prefix.fbin",
            rows=6,
            opener=RangeOpener(source),
        )


def test_downloader_rejects_orphaned_partial_state(tmp_path):
    source = _fbin_bytes(np.ones((5, 2), dtype=np.float32))
    output = tmp_path / "prefix.fbin"
    Path(f"{output}.part").write_bytes(struct.pack("<II", 2, 2))

    with pytest.raises(RuntimeError, match="must exist together|resume metadata"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=2,
            opener=RangeOpener(source),
        )


def test_failed_http_open_never_publishes_output(tmp_path):
    source = _fbin_bytes(np.ones((5, 2), dtype=np.float32))
    opener = RangeOpener(source)
    opener.raise_at = 8
    output = tmp_path / "prefix.fbin"

    with pytest.raises(URLError, match="disconnect"):
        cache_deep1b.download_fbin_prefix(
            cache_deep1b.DEFAULT_URL,
            output,
            rows=2,
            opener=opener,
        )

    assert not output.exists()
    assert not Path(f"{output}.manifest.json").exists()
    assert Path(f"{output}.part").exists()


def test_default_url_is_official_yandex_base_1b_object():
    assert cache_deep1b.DEFAULT_URL == (
        "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
    )
