#!/usr/bin/env python3
r"""Encode Deep1B with one selected training-pool-size AE checkpoint.

This is the layout-producing companion to
:mod:`experiments.compare_ae_training_sizes`.  The training harness deliberately
writes no corpus-sized layouts.  This program takes one of its tune-selected
checkpoints, independently reconstructs and verifies the recorded split, then
streams the *same complete training universe* and every public query through the
checkpoint encoder.  The resulting regular NPY files can be passed directly to
:mod:`experiments.evaluate_yandex_retrieval`.

For the production Deep1B experiment the recorded model is
``96-256-128-32-8-symmetric`` and the complete universe is the 10-million-row
base prefix.  The implementation derives those values from the training result
rather than silently assuming them, so small CPU fixtures can exercise the
identical protocol.

Example::

    python experiments/encode_ae_training_size_checkpoint.py \
      --data artifacts/data/yandex-deep1b/deep-base-prefix-10m.fbin \
      --query-data artifacts/data/yandex-deep1b/query.public.10K.fbin \
      --training-results artifacts/deep1b-ae-training-size/results.json \
      --checkpoint artifacts/deep1b-ae-training-size/checkpoints/best-seed-1729-pool-65536.pt \
      --init-seed 1729 --pool-size 65536 --device cuda \
      --base-layout artifacts/deep1b-ae-training-size/layouts/base-seed1729-pool65536.npy \
      --query-layout artifacts/deep1b-ae-training-size/layouts/query-seed1729-pool65536.npy \
      --provenance artifacts/deep1b-ae-training-size/layouts/seed1729-pool65536.json

No output is published until both inputs have been completely streamed and all
available content hashes agree.  Encoding is always FP32, matching the training
experiment's test evaluation precision.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from experiments.compare_ae_training_sizes import (
        PROTOCOL as TRAINING_PROTOCOL,
    )
    from experiments.compare_ae_training_sizes import (
        SCHEMA_VERSION as TRAINING_SCHEMA_VERSION,
    )
    from experiments.compare_ae_training_sizes import (
        _sha256_indices,
        _state_sha256,
        make_experiment_indices,
    )
    from experiments.compare_yandex_streaming import (
        BoundedFbinReader,
        _build_autoencoder,
        _load_source_manifest,
        _resolve_device,
        _source_snapshot,
        _stream_layout,
    )
    from experiments.yandex_fbin import FbinInfo, read_fbin_header
except ModuleNotFoundError:  # Direct execution from inside ``experiments/``.
    from compare_ae_training_sizes import (  # type: ignore[no-redef]
        PROTOCOL as TRAINING_PROTOCOL,
    )
    from compare_ae_training_sizes import (  # type: ignore[no-redef]
        SCHEMA_VERSION as TRAINING_SCHEMA_VERSION,
    )
    from compare_ae_training_sizes import (
        _sha256_indices,
        _state_sha256,
        make_experiment_indices,
    )
    from compare_yandex_streaming import (  # type: ignore[no-redef]
        BoundedFbinReader,
        _build_autoencoder,
        _load_source_manifest,
        _resolve_device,
        _source_snapshot,
        _stream_layout,
    )
    from yandex_fbin import FbinInfo, read_fbin_header  # type: ignore[no-redef]


PROTOCOL = "deep1b-ae-training-size-checkpoint-encoding-v1"
SCHEMA_VERSION = 1
_MAX_RESULTS_BYTES = 64 * 2**20


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, block_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _load_training_result(path: Path) -> tuple[dict[str, Any], str]:
    size = path.stat().st_size
    if size > _MAX_RESULTS_BYTES:
        raise ValueError(f"training result is unexpectedly large: {size} bytes")
    raw = path.read_bytes()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid training result JSON {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("training result must be a JSON object")
    if (
        result.get("schema_version") != TRAINING_SCHEMA_VERSION
        or result.get("protocol") != TRAINING_PROTOCOL
        or result.get("status") != "ok"
    ):
        raise ValueError("training result is not a successful training-pool-size experiment")
    return result, hashlib.sha256(raw).hexdigest()


def _object(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"training result field {key!r} must be an object")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _integer_list(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    values = tuple(_positive_integer(item, name=f"{name} item") for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _sha256_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal string")
    return value


def _verify_split(result: Mapping[str, Any], pool_size: int) -> dict[str, Any]:
    """Recompute the complete recorded split and require every hash to agree."""

    arguments = _object(result, "arguments")
    source = _object(result, "source")
    split = _object(result, "split")
    test_record = _object(split, "test")
    tune_record = _object(split, "tune")
    training_record = _object(split, "training")

    universe_rows = _positive_integer(source.get("universe_rows"), name="source.universe_rows")
    test_rows = _positive_integer(test_record.get("rows"), name="split.test.rows")
    tune_rows = _positive_integer(tune_record.get("rows"), name="split.tune.rows")
    pool_sizes = _integer_list(training_record.get("pool_sizes"), name="split.training.pool_sizes")
    if tuple(sorted(pool_sizes)) != pool_sizes:
        raise ValueError("recorded training pool sizes must be strictly increasing")
    if pool_size not in pool_sizes:
        raise ValueError(f"pool size {pool_size} is absent from the training result")

    argument_expectations = {
        "universe_rows": universe_rows,
        "test_rows": test_rows,
        "tune_rows": tune_rows,
        "test_seed": test_record.get("seed"),
        "tune_seed": tune_record.get("seed"),
        "pool_seed": training_record.get("pool_seed"),
        "pool_sizes": list(pool_sizes),
    }
    for key, expected in argument_expectations.items():
        if arguments.get(key) != expected:
            raise ValueError(
                f"training result arguments.{key} disagrees with its split/source record"
            )

    test_seed = _nonnegative_integer(test_record.get("seed"), name="split.test.seed")
    tune_seed = _nonnegative_integer(tune_record.get("seed"), name="split.tune.seed")
    pool_seed = _nonnegative_integer(
        training_record.get("pool_seed"), name="split.training.pool_seed"
    )
    indices = make_experiment_indices(
        universe_rows=universe_rows,
        test_rows=test_rows,
        tune_rows=tune_rows,
        maximum_pool_size=max(pool_sizes),
        test_seed=test_seed,
        tune_seed=tune_seed,
        pool_seed=pool_seed,
    )
    pool_hashes = training_record.get("pool_indices_sha256")
    if not isinstance(pool_hashes, dict):
        raise ValueError("split.training.pool_indices_sha256 must be an object")
    observed = {
        "test_indices_sha256": _sha256_indices(indices.test),
        "tune_indices_sha256": _sha256_indices(indices.tune),
        "training_order_sha256": _sha256_indices(indices.training_order),
        "unused_indices_sha256": _sha256_indices(indices.unused),
        "selected_pool_indices_sha256": _sha256_indices(indices.training_order[:pool_size]),
    }
    expected = {
        "test_indices_sha256": _sha256_text(
            test_record.get("indices_sha256"), name="split.test.indices_sha256"
        ),
        "tune_indices_sha256": _sha256_text(
            tune_record.get("indices_sha256"), name="split.tune.indices_sha256"
        ),
        "training_order_sha256": _sha256_text(
            training_record.get("nested_uniform_order_sha256"),
            name="split.training.nested_uniform_order_sha256",
        ),
        "unused_indices_sha256": _sha256_text(
            training_record.get("unused_indices_sha256"),
            name="split.training.unused_indices_sha256",
        ),
        "selected_pool_indices_sha256": _sha256_text(
            pool_hashes.get(str(pool_size)),
            name=f"split.training.pool_indices_sha256.{pool_size}",
        ),
    }
    for recorded_size in pool_sizes:
        recorded = _sha256_text(
            pool_hashes.get(str(recorded_size)),
            name=f"split.training.pool_indices_sha256.{recorded_size}",
        )
        recomputed = _sha256_indices(indices.training_order[:recorded_size])
        if recorded != recomputed:
            raise ValueError(f"recorded pool hash does not reproduce for size {recorded_size}")
    if observed != expected:
        mismatches = [name for name in observed if observed[name] != expected[name]]
        raise ValueError(f"recorded split hashes do not reproduce: {', '.join(mismatches)}")
    return {
        "verified": True,
        "universe_rows": universe_rows,
        "test_rows": test_rows,
        "tune_rows": tune_rows,
        "test_seed": test_seed,
        "tune_seed": tune_seed,
        "pool_seed": pool_seed,
        "maximum_pool_size": max(pool_sizes),
        "selected_pool_size": pool_size,
        "hashes": observed,
    }


def _selected_arm(result: Mapping[str, Any], init_seed: int, pool_size: int) -> dict[str, Any]:
    arms = result.get("arms")
    if not isinstance(arms, list):
        raise ValueError("training result arms must be a list")
    matches = [
        arm
        for arm in arms
        if isinstance(arm, dict)
        and arm.get("init_seed") == init_seed
        and arm.get("pool_size") == pool_size
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one arm for init seed {init_seed} and pool {pool_size}, "
            f"found {len(matches)}"
        )
    arm = matches[0]
    selection = _object(arm, "selection")
    _positive_integer(selection.get("best_update"), name="arm.selection.best_update")
    return arm


def _model_configuration(result: Mapping[str, Any], source_dimension: int) -> dict[str, Any]:
    configuration = _object(result, "protocol_configuration")
    target_dimension = _positive_integer(
        configuration.get("target_dimension"), name="protocol_configuration.target_dimension"
    )
    hidden = _integer_list(
        configuration.get("hidden_dimensions"),
        name="protocol_configuration.hidden_dimensions",
    )
    if not target_dimension < source_dimension:
        raise ValueError("recorded target dimension must be smaller than the source dimension")
    expected_architecture = (
        f"{source_dimension}-" + "-".join(map(str, hidden)) + f"-{target_dimension}-symmetric"
    )
    if configuration.get("architecture") != expected_architecture:
        raise ValueError("recorded architecture string disagrees with its dimensions")
    return {
        "input_dimension": source_dimension,
        "hidden_dimensions": hidden,
        "target_dimension": target_dimension,
        "architecture": expected_architecture,
        "recorded_parameter_count": _positive_integer(
            configuration.get("parameter_count"),
            name="protocol_configuration.parameter_count",
        ),
    }


def _load_checkpoint(
    torch,
    path: Path,
    *,
    arm: Mapping[str, Any],
    model_configuration: Mapping[str, Any],
    device,
) -> tuple[Any, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not safely load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be an object")
    selection = _object(arm, "selection")
    expected_metadata = {
        "protocol": TRAINING_PROTOCOL,
        "init_seed": arm.get("init_seed"),
        "pool_size": arm.get("pool_size"),
        "selected_update": selection.get("best_update"),
    }
    observed_metadata = {key: checkpoint.get(key) for key in expected_metadata}
    if observed_metadata != expected_metadata:
        raise ValueError(
            f"checkpoint metadata disagrees with selected arm: "
            f"{observed_metadata} != {expected_metadata}"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no model_state_dict")
    if any(
        not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()
    ):
        raise ValueError("checkpoint model_state_dict must map names to tensors")
    model = _build_autoencoder(
        torch,
        int(model_configuration["input_dimension"]),
        int(model_configuration["target_dimension"]),
        tuple(model_configuration["hidden_dimensions"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != model_configuration["recorded_parameter_count"]:
        raise ValueError("reconstructed parameter count disagrees with the training result")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"checkpoint state does not match the recorded architecture: {error}"
        ) from error
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("checkpoint parameters must be float32")
    state_sha256 = _state_sha256(torch, model.state_dict())
    model.to(device)
    model.eval()
    return model, {
        **observed_metadata,
        "file_sha256": _sha256_file(path),
        "state_dict_sha256": state_sha256,
        "parameter_count": parameter_count,
        "parameters": "float32",
    }


def _verify_source_metadata(result: Mapping[str, Any], info: FbinInfo) -> str:
    source = _object(result, "source")
    split_rows = _positive_integer(source.get("universe_rows"), name="source.universe_rows")
    expected = {
        "format": "fbin",
        "rows": info.rows,
        "dimension": info.dimension,
        "file_bytes": info.expected_size,
        "universe_rows": info.rows,
    }
    observed = {key: source.get(key) for key in expected}
    if observed != expected or split_rows != info.rows:
        raise ValueError(
            "base fbin is not the complete source/universe recorded by the training result: "
            f"{observed} != {expected}"
        )
    return _sha256_text(source.get("file_sha256"), name="source.file_sha256")


def _manifest_expectation(path: Path, info: FbinInfo) -> dict[str, Any]:
    manifest = _load_source_manifest(path, info)
    if not manifest["present"]:
        return {**manifest, "content_sha256_verified": False}
    return {**manifest, "content_sha256_verified": None}


def _staging_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")


def _encode_source(
    torch,
    model,
    path: Path,
    info: FbinInfo,
    staging_path: Path,
    *,
    target_dimension: int,
    block_rows: int,
    batch_size: int,
    device,
) -> dict[str, Any]:
    before = _source_snapshot(path)
    with BoundedFbinReader(path, info, rows=info.rows) as reader:
        header = reader.read_header()
        rows, seconds, source_sha256 = _stream_layout(
            torch,
            model,
            reader,
            staging_path,
            target_dimension=target_dimension,
            block_rows=block_rows,
            batch_size=batch_size,
            device=device,
            source_header=header,
        )
    if rows != info.rows:
        raise RuntimeError(f"encoder wrote {rows} rows, expected {info.rows}")
    if _source_snapshot(path) != before:
        raise RuntimeError(f"source changed during streamed encoding: {path}")
    layout = np.load(staging_path, mmap_mode="r", allow_pickle=False)
    try:
        if layout.shape != (info.rows, target_dimension) or layout.dtype != np.float32:
            raise RuntimeError(
                "streamed layout has unexpected shape or dtype: "
                f"shape={layout.shape}, dtype={layout.dtype}"
            )
        if not layout.flags.c_contiguous:
            raise RuntimeError("streamed layout is not C-contiguous")
    finally:
        mmap = layout._mmap
        del layout
        mmap.close()
    return {
        "source_identity_snapshot": before,
        "source_content_sha256": source_sha256,
        "layout_rows": rows,
        "layout_dimension": target_dimension,
        "layout_dtype": "float32",
        "layout_file_bytes": staging_path.stat().st_size,
        "layout_sha256": _sha256_file(staging_path),
        "encoding_seconds": seconds,
        "rows_per_second": rows / seconds,
        "source_bytes_per_second": info.data_bytes / seconds,
    }


def _validate_distinct_paths(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    for position, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[position + 1 :]:
            same = left_path.resolve(strict=False) == right_path.resolve(strict=False)
            if not same and left_path.exists() and right_path.exists():
                with contextlib.suppress(OSError):
                    same = os.path.samefile(left_path, right_path)
            if same:
                raise ValueError(f"{left_name} and {right_name} must be distinct paths")


def run_encoding(args) -> tuple[dict[str, Any], Path, Path]:
    """Verify one arm and stage its complete base and query layouts."""

    import torch

    started_at = _utc_now()
    total_started = time.perf_counter()
    paths = {
        "base data": Path(args.data).resolve(),
        "query data": Path(args.query_data).resolve(),
        "training results": Path(args.training_results).resolve(),
        "checkpoint": Path(args.checkpoint).resolve(),
        "base layout": Path(args.base_layout).resolve(),
        "query layout": Path(args.query_layout).resolve(),
        "provenance": Path(args.provenance).resolve(),
    }
    _validate_distinct_paths(paths)
    for name in ("base layout", "query layout", "provenance"):
        if paths[name].exists() and not args.force:
            raise FileExistsError(f"{name} exists; pass --force to replace it: {paths[name]}")
    _nonnegative_integer(args.init_seed, name="init_seed")
    _positive_integer(args.pool_size, name="pool_size")
    _positive_integer(args.block_rows, name="block_rows")
    _positive_integer(args.batch_size, name="batch_size")

    result, result_sha256 = _load_training_result(paths["training results"])
    split_verification = _verify_split(result, args.pool_size)
    arm = _selected_arm(result, args.init_seed, args.pool_size)
    if (
        arm.get("pool_indices_sha256")
        != split_verification["hashes"]["selected_pool_indices_sha256"]
    ):
        raise ValueError("selected arm pool hash disagrees with the reproduced split")

    base_info = read_fbin_header(paths["base data"])
    expected_base_sha256 = _verify_source_metadata(result, base_info)
    query_info = read_fbin_header(paths["query data"])
    if query_info.dimension != base_info.dimension:
        raise ValueError("query and base dimensions differ")
    model_configuration = _model_configuration(result, base_info.dimension)
    device = _resolve_device(torch, args.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
    model, checkpoint_verification = _load_checkpoint(
        torch,
        paths["checkpoint"],
        arm=arm,
        model_configuration=model_configuration,
        device=device,
    )
    base_manifest = _manifest_expectation(paths["base data"], base_info)
    query_manifest = _manifest_expectation(paths["query data"], query_info)

    base_staging = _staging_path(paths["base layout"])
    query_staging = _staging_path(paths["query layout"])
    try:
        base_encoding = _encode_source(
            torch,
            model,
            paths["base data"],
            base_info,
            base_staging,
            target_dimension=model_configuration["target_dimension"],
            block_rows=args.block_rows,
            batch_size=args.batch_size,
            device=device,
        )
        if base_encoding["source_content_sha256"] != expected_base_sha256:
            raise ValueError("base fbin content SHA-256 disagrees with the training result")
        if base_manifest["present"]:
            if base_encoding["source_content_sha256"] != base_manifest["expected_fbin_sha256"]:
                raise ValueError("base fbin content SHA-256 disagrees with its download manifest")
            base_manifest["content_sha256_verified"] = True

        query_encoding = _encode_source(
            torch,
            model,
            paths["query data"],
            query_info,
            query_staging,
            target_dimension=model_configuration["target_dimension"],
            block_rows=args.block_rows,
            batch_size=args.batch_size,
            device=device,
        )
        if query_manifest["present"]:
            if query_encoding["source_content_sha256"] != query_manifest["expected_fbin_sha256"]:
                raise ValueError("query fbin content SHA-256 disagrees with its download manifest")
            query_manifest["content_sha256_verified"] = True

        cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "ok",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "selection": {
                "init_seed": args.init_seed,
                "pool_size": args.pool_size,
                "selected_update": arm["selection"]["best_update"],
                "test_reconstruction_nmse": arm["test"]["reconstruction_nmse"],
                "test_reconstruction_cosine": arm["test"]["reconstruction_cosine"],
            },
            "training_result": {
                "path": str(paths["training results"]),
                "sha256": result_sha256,
                "protocol": result["protocol"],
                "split": split_verification,
            },
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "recorded_path": arm.get("checkpoint_path"),
                "recorded_path_matches": (arm.get("checkpoint_path") == str(paths["checkpoint"])),
                **checkpoint_verification,
            },
            "model": {
                **model_configuration,
                "hidden_dimensions": list(model_configuration["hidden_dimensions"]),
                "encoding_precision": "float32",
            },
            "base": {
                "source_path": str(paths["base data"]),
                "source_rows": base_info.rows,
                "source_dimension": base_info.dimension,
                "source_file_bytes": base_info.expected_size,
                "expected_training_source_sha256": expected_base_sha256,
                "download_manifest": base_manifest,
                "layout_path": str(paths["base layout"]),
                **base_encoding,
            },
            "query": {
                "source_path": str(paths["query data"]),
                "source_rows": query_info.rows,
                "source_dimension": query_info.dimension,
                "source_file_bytes": query_info.expected_size,
                "all_rows_encoded": True,
                "download_manifest": query_manifest,
                "layout_path": str(paths["query layout"]),
                **query_encoding,
            },
            "streaming": {
                "reader": "BoundedFbinReader using bounded positional reads",
                "persistent_input_mapping_bytes": 0,
                "persistent_layout_mapping_bytes": 0,
                "block_rows": args.block_rows,
                "batch_size": args.batch_size,
                "publication": "both layouts verified before atomic per-file replacement; provenance last",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "cuda_device_name": cuda_name,
                "cuda_matmul_tf32_allowed": (
                    torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
                ),
                "cudnn_tf32_allowed": (
                    torch.backends.cudnn.allow_tf32 if device.type == "cuda" else None
                ),
                "torch_deterministic_algorithms_enabled": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "harness_sha256": _sha256_file(Path(__file__).resolve()),
            },
            "timing_seconds": {"total_before_publication": time.perf_counter() - total_started},
        }
        return provenance, base_staging, query_staging
    except Exception:
        for staging in (base_staging, query_staging):
            with contextlib.suppress(FileNotFoundError):
                staging.unlink()
        raise


def execute(args) -> dict[str, Any]:
    provenance, base_staging, query_staging = run_encoding(args)
    base_layout = Path(args.base_layout).resolve()
    query_layout = Path(args.query_layout).resolve()
    provenance_path = Path(args.provenance).resolve()
    try:
        os.replace(base_staging, base_layout)
        os.replace(query_staging, query_layout)
        _atomic_json(provenance_path, provenance)
    finally:
        for staging in (base_staging, query_staging):
            with contextlib.suppress(FileNotFoundError):
                staging.unlink()
    return provenance


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--query-data", type=Path, required=True)
    parser.add_argument("--training-results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--base-layout", type=Path, required=True)
    parser.add_argument("--query-layout", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--block-rows", type=int, default=131_072)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.init_seed < 0 or min(args.pool_size, args.block_rows, args.batch_size) < 1:
        parser.error("seed must be non-negative; pool size, block rows, and batch size positive")
    return args


def _test_args(**overrides):
    defaults = {
        "data": Path("base.fbin"),
        "query_data": Path("query.fbin"),
        "training_results": Path("results.json"),
        "checkpoint": Path("checkpoint.pt"),
        "init_seed": 101,
        "pool_size": 16,
        "base_layout": Path("base-layout.npy"),
        "query_layout": Path("query-layout.npy"),
        "provenance": Path("provenance.json"),
        "device": "cpu",
        "block_rows": 8,
        "batch_size": 4,
        "force": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provenance = execute(args)
    print(
        f"encoded {provenance['base']['layout_rows']} base and "
        f"{provenance['query']['layout_rows']} query rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
