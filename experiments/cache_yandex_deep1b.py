#!/usr/bin/env python3
"""Download an exact, resumable prefix of Yandex DEEP ``base.1B.fbin``.

Only the requested byte range is transferred.  The local file receives a new
header declaring the prefix row count, so it remains a strict standalone
``.fbin``.  A partial download and its source-identity sidecar are retained after
an interrupted request; the finished file and provenance manifest are published
with atomic renames.

Yandex DEEP is released under CC BY 4.0; dataset details and attribution are at
https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search.

Example
-------
::

    python experiments/cache_yandex_deep1b.py \
      --rows 100000000 --output artifacts/data/deep-100m.fbin
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import struct
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO

try:  # Package-style import in tests and tooling.
    from .yandex_fbin import FBIN_DTYPE, FBIN_HEADER_BYTES, inspect_fbin
except ImportError:  # Direct ``python experiments/cache_yandex_deep1b.py`` execution.
    from yandex_fbin import FBIN_DTYPE, FBIN_HEADER_BYTES, inspect_fbin

DEFAULT_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1
_HEADER = struct.Struct("<II")
_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)\Z")
_BASE_10M = re.compile(r"(?:^|/)base\.10m\.fbin(?:[?#]|$)", re.IGNORECASE)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _paths(output: Path) -> tuple[Path, Path, Path, Path]:
    return (
        Path(f"{output}.part"),
        Path(f"{output}.part.json"),
        Path(f"{output}.manifest.json"),
        Path(f"{output}.lock"),
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _output_lock(lock_path: Path) -> Iterator[None]:
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another downloader holds {lock_path}; remove it only after confirming "
            "that no download is running"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    if not isinstance(status, int):
        raise RuntimeError("range response has no numeric HTTP status")
    return status


def _response_headers(response: Any) -> Mapping[str, str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        raise RuntimeError("range response has no HTTP headers")
    return headers


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        # Plain dictionaries used by callers/tests need case-insensitive lookup;
        # ``http.client.HTTPMessage.get`` already supplies it in production.
        lowered = name.lower()
        for key, candidate in headers.items():
            if key.lower() == lowered:
                value = candidate
                break
    return None if value is None else str(value)


def _validate_range_response(
    response: Any,
    *,
    start: int,
    end: int,
    expected_total: int | None,
) -> tuple[int, Mapping[str, str]]:
    status = _response_status(response)
    if status != 206:
        raise RuntimeError(
            f"server did not honor byte range {start}-{end}: expected HTTP 206, got {status}"
        )
    headers = _response_headers(response)
    encoding = _header_value(headers, "Content-Encoding")
    if encoding is not None and encoding.lower() not in {"identity", ""}:
        raise RuntimeError(f"range response used unsafe Content-Encoding {encoding!r}")

    raw_content_range = _header_value(headers, "Content-Range")
    match = _CONTENT_RANGE.fullmatch(raw_content_range or "")
    if match is None:
        raise RuntimeError(f"invalid or missing Content-Range: {raw_content_range!r}")
    observed_start, observed_end, total = (int(token) for token in match.groups())
    if (observed_start, observed_end) != (start, end):
        raise RuntimeError(
            "Content-Range does not match the request: "
            f"requested {start}-{end}, received {observed_start}-{observed_end}"
        )
    if expected_total is not None and total != expected_total:
        raise RuntimeError(
            f"remote object length changed: expected {expected_total}, received {total}"
        )

    expected_length = end - start + 1
    raw_length = _header_value(headers, "Content-Length")
    try:
        observed_length = int(raw_length) if raw_length is not None else None
    except ValueError as exc:
        raise RuntimeError(f"invalid Content-Length: {raw_length!r}") from exc
    if observed_length != expected_length:
        raise RuntimeError(
            f"invalid Content-Length for range {start}-{end}: "
            f"expected {expected_length}, received {raw_length!r}"
        )
    return total, headers


def _request(
    opener: Callable[..., Any],
    *,
    url: str,
    start: int,
    end: int,
    timeout: float,
    if_range: str | None = None,
) -> Any:
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "dimension-reduction-benchmarks-yandex-prefix/1",
    }
    if if_range is not None:
        headers["If-Range"] = if_range
    request = urllib.request.Request(url, headers=headers, method="GET")
    return opener(request, timeout=timeout)


def _read_exact(response: BinaryIO, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        block = response.read(min(1024 * 1024, remaining))
        if not block:
            raise RuntimeError(
                f"truncated HTTP range body: expected {length} bytes, "
                f"received {length - remaining}"
            )
        chunks.append(block)
        remaining -= len(block)
    if response.read(1):
        raise RuntimeError("HTTP range body contains more bytes than Content-Length")
    return b"".join(chunks)


def _remote_metadata(
    url: str, *, opener: Callable[..., Any], timeout: float
) -> tuple[dict[str, Any], bytes]:
    response = _request(
        opener,
        url=url,
        start=0,
        end=FBIN_HEADER_BYTES - 1,
        timeout=timeout,
    )
    with contextlib.closing(response):
        total, headers = _validate_range_response(
            response,
            start=0,
            end=FBIN_HEADER_BYTES - 1,
            expected_total=None,
        )
        header = _read_exact(response, FBIN_HEADER_BYTES)
    source_rows, dimension = _HEADER.unpack(header)
    if source_rows == 0 or dimension == 0:
        raise RuntimeError(
            f"remote fbin has invalid shape rows={source_rows}, dimension={dimension}"
        )
    expected_total = FBIN_HEADER_BYTES + source_rows * dimension * FBIN_DTYPE.itemsize
    if total != expected_total:
        raise RuntimeError(
            "remote fbin header and object length disagree: "
            f"header implies {expected_total} bytes, Content-Range reports {total}"
        )
    metadata = {
        "url": url,
        "rows": int(source_rows),
        "dimension": int(dimension),
        "file_bytes": int(total),
        "etag": _header_value(headers, "ETag"),
        "last_modified": _header_value(headers, "Last-Modified"),
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }
    return metadata, header


def _prefix_metadata(source: Mapping[str, Any], rows: int) -> dict[str, int]:
    dimension = int(source["dimension"])
    payload_bytes = rows * dimension * FBIN_DTYPE.itemsize
    return {
        "rows": rows,
        "dimension": dimension,
        "header_bytes": FBIN_HEADER_BYTES,
        "payload_bytes": payload_bytes,
        "file_bytes": FBIN_HEADER_BYTES + payload_bytes,
    }


def _state_payload(*, source: Mapping[str, Any], prefix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": "fbin-little-endian-float32",
        "source": dict(source),
        "prefix": dict(prefix),
    }


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} {path} must contain a JSON object")
    return payload


def _identity_matches(recorded: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    invariant_fields = ("url", "rows", "dimension", "file_bytes", "header_sha256")
    if any(recorded.get(field) != observed.get(field) for field in invariant_fields):
        return False
    for field in ("etag", "last_modified"):
        old = recorded.get(field)
        new = observed.get(field)
        if old is not None and new is not None and old != new:
            return False
    return True


def _resume_validator(source: Mapping[str, Any]) -> str | None:
    etag = source.get("etag")
    if isinstance(etag, str) and etag and not etag.startswith("W/"):
        return etag
    modified = source.get("last_modified")
    if isinstance(modified, str) and modified:
        return modified
    return None


def _verify_response_identity(headers: Mapping[str, str], source: Mapping[str, Any]) -> None:
    for field, header_name in (("etag", "ETag"), ("last_modified", "Last-Modified")):
        expected = source.get(field)
        observed = _header_value(headers, header_name)
        if expected is not None and observed is not None and observed != expected:
            raise RuntimeError(
                f"remote source identity changed during download: {header_name} "
                f"was {expected!r}, is now {observed!r}"
            )


def _hash_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_payload(
    *, output: Path, source: Mapping[str, Any], prefix: Mapping[str, Any], sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": "fbin-little-endian-float32",
        "source": dict(source),
        "prefix": {
            **dict(prefix),
            "file": output.name,
            "sha256": sha256,
        },
    }


def _prepare_partial(
    part_path: Path,
    state_path: Path,
    *,
    source: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> None:
    expected_state = _state_payload(source=source, prefix=prefix)
    if part_path.exists() != state_path.exists():
        raise RuntimeError(
            f"incomplete resume metadata: {part_path} and {state_path} must exist together"
        )
    if not part_path.exists():
        with part_path.open("xb") as handle:
            handle.write(_HEADER.pack(int(prefix["rows"]), int(prefix["dimension"])))
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_write_json(state_path, expected_state)
        return

    recorded = _load_json(state_path, description="partial-download state")
    if recorded.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("partial-download state has an unsupported schema version")
    if recorded.get("format") != expected_state["format"]:
        raise RuntimeError("partial-download state has a different format")
    recorded_source = recorded.get("source")
    if not isinstance(recorded_source, dict) or not _identity_matches(recorded_source, source):
        raise RuntimeError("remote source does not match the partial-download provenance")
    if recorded.get("prefix") != dict(prefix):
        raise RuntimeError("requested prefix does not match the partial-download provenance")

    size = part_path.stat().st_size
    if size < FBIN_HEADER_BYTES or size > int(prefix["file_bytes"]):
        raise RuntimeError(
            f"partial file has invalid length {size}; expected between "
            f"{FBIN_HEADER_BYTES} and {prefix['file_bytes']}"
        )
    with part_path.open("rb") as handle:
        local_header = handle.read(FBIN_HEADER_BYTES)
    expected_header = _HEADER.pack(int(prefix["rows"]), int(prefix["dimension"]))
    if local_header != expected_header:
        raise RuntimeError("partial file header does not match the requested prefix")
    if size > FBIN_HEADER_BYTES and _resume_validator(recorded_source) is None:
        raise RuntimeError(
            "cannot safely resume: the server supplied neither a strong ETag nor Last-Modified"
        )


def _download_payload(
    part_path: Path,
    *,
    url: str,
    source: Mapping[str, Any],
    target_size: int,
    chunk_bytes: int,
    timeout: float,
    opener: Callable[..., Any],
) -> None:
    validator = _resume_validator(source)
    with part_path.open("r+b", buffering=0) as output:
        while True:
            local_start = output.seek(0, os.SEEK_END)
            if local_start == target_size:
                break
            if local_start > target_size:
                raise RuntimeError(
                    f"partial file exceeds target size: {local_start} > {target_size}"
                )
            # Both files have an eight-byte header, so local and remote byte offsets
            # agree for every payload byte despite the rewritten row count.
            start = local_start
            end = min(target_size, start + chunk_bytes) - 1
            response = _request(
                opener,
                url=url,
                start=start,
                end=end,
                timeout=timeout,
                if_range=validator,
            )
            try:
                with contextlib.closing(response):
                    _, headers = _validate_range_response(
                        response,
                        start=start,
                        end=end,
                        expected_total=int(source["file_bytes"]),
                    )
                    _verify_response_identity(headers, source)
                    remaining = end - start + 1
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise RuntimeError(
                                f"truncated HTTP range body for {start}-{end}; "
                                f"{remaining} bytes missing"
                            )
                        output.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise RuntimeError(
                            f"HTTP range body for {start}-{end} exceeds Content-Length"
                        )
                    output.flush()
                    os.fsync(output.fileno())
            except Exception:
                output.seek(local_start)
                output.truncate()
                output.flush()
                os.fsync(output.fileno())
                raise


def download_fbin_prefix(
    url: str,
    output: str | os.PathLike[str],
    *,
    rows: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout: float = 120.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Download and atomically publish the first ``rows`` of a remote ``.fbin``."""

    requested_rows = _positive_integer(rows, name="rows")
    request_bytes = _positive_integer(chunk_bytes, name="chunk_bytes")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if _BASE_10M.search(url):
        raise ValueError(
            "base.10M.fbin is a standalone debugging set; crop base.1B.fbin so row "
            "indices remain compatible with the official query ground truth"
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path, state_path, manifest_path, lock_path = _paths(output_path)
    with _output_lock(lock_path):
        source, _ = _remote_metadata(url, opener=opener, timeout=timeout)
        if requested_rows > source["rows"]:
            raise ValueError(
                f"requested {requested_rows} rows, but the remote file contains "
                f"{source['rows']}"
            )
        prefix = _prefix_metadata(source, requested_rows)

        if output_path.exists():
            if part_path.exists():
                raise RuntimeError(f"both completed and partial files exist for {output_path}")
            info = inspect_fbin(output_path)
            if (info.rows, info.dimension, info.expected_size) != (
                prefix["rows"],
                prefix["dimension"],
                prefix["file_bytes"],
            ):
                raise RuntimeError(f"existing output {output_path} is not the requested prefix")
            digest = _hash_file(output_path)
            manifest = _manifest_payload(
                output=output_path, source=source, prefix=prefix, sha256=digest
            )
            if manifest_path.exists():
                recorded = _load_json(manifest_path, description="provenance manifest")
                if recorded != manifest:
                    raise RuntimeError(f"existing provenance manifest does not match {output_path}")
            else:
                _atomic_write_json(manifest_path, manifest)
            state_path.unlink(missing_ok=True)
            return manifest

        _prepare_partial(
            part_path,
            state_path,
            source=source,
            prefix=prefix,
        )
        _download_payload(
            part_path,
            url=url,
            source=source,
            target_size=int(prefix["file_bytes"]),
            chunk_bytes=request_bytes,
            timeout=timeout,
            opener=opener,
        )

        # Validate the complete staged file before the only operation that makes it
        # visible under the requested output name.
        info = inspect_fbin(part_path)
        if (info.rows, info.dimension, info.expected_size) != (
            prefix["rows"],
            prefix["dimension"],
            prefix["file_bytes"],
        ):
            raise RuntimeError("completed partial file does not match the requested prefix")
        digest = _hash_file(part_path)
        manifest = _manifest_payload(
            output=output_path, source=source, prefix=prefix, sha256=digest
        )
        os.replace(part_path, output_path)
        _atomic_write_json(manifest_path, manifest)
        state_path.unlink(missing_ok=True)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True, help="number of base.1B prefix rows")
    parser.add_argument("--output", type=Path, required=True, help="local .fbin output path")
    parser.add_argument("--url", default=DEFAULT_URL, help="base.1B .fbin URL")
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = download_fbin_prefix(
        args.url,
        args.output,
        rows=args.rows,
        chunk_bytes=args.chunk_bytes,
        timeout=args.timeout,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
