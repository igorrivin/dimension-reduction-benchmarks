"""Tests for deterministic bounded-memory Deep1B topology sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from experiments.sample_yandex_topology import deterministic_indices, run


def _write_fbin(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype="<f4", order="C")
    path.write_bytes(struct.pack("<II", array.shape[0], array.shape[1]) + array.tobytes(order="C"))


def _normalized_values(rows: int = 19, dimensions: int = 4) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(rows, dimensions)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _args(
    reference: Path,
    layouts: list[tuple[str, Path]],
    output: Path,
    *,
    sample_size: int = 8,
) -> argparse.Namespace:
    return argparse.Namespace(
        input_fbin=str(reference),
        layout=[f"{method}={path}" for method, path in layouts],
        sample_size=sample_size,
        seed=29,
        validation_block_rows=3,
        copy_block_rows=2,
        l2_atol=5e-4,
        output_dir=str(output),
    )


def test_run_writes_deterministic_aligned_atomic_sample_and_provenance(tmp_path):
    values = _normalized_values()
    reference = tmp_path / "source.fbin"
    _write_fbin(reference, values)
    layout_a = tmp_path / "a.npy"
    layout_b_generation = tmp_path / "b-generation.npy"
    layout_b = tmp_path / "b.npy"
    np.save(layout_a, values[:, :2] * 2)
    np.save(layout_b_generation, values[:, 2:] - 3)
    layout_b.symlink_to(layout_b_generation.name)
    output = tmp_path / "prepared"

    manifest = run(_args(reference, [("ae", layout_a), ("umap", layout_b)], output))

    indices = deterministic_indices(values.shape[0], 8, 29)
    np.testing.assert_array_equal(np.load(output / "source_indices.npy"), indices)
    np.testing.assert_allclose(np.load(output / "reference.npy"), values[indices])
    np.testing.assert_allclose(np.load(output / "layout-ae.npy"), values[indices, :2] * 2)
    np.testing.assert_allclose(np.load(output / "layout-umap.npy"), values[indices, 2:] - 3)
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["outputs"]["source_indices"]["index_sha256"]
    assert manifest["outputs"]["source_indices"]["unique"] == 8
    assert manifest["inputs"]["layouts"]["umap"]["requested_path"].endswith("b.npy")
    assert manifest["inputs"]["layouts"]["umap"]["resolved_path"].endswith("b-generation.npy")
    assert (
        manifest["inputs"]["reference"]["sha256"]
        == hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    assert manifest["publication"] == "atomic_directory_rename_no_overwrite"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(_args(reference, [("ae", layout_a), ("umap", layout_b)], output))


@pytest.mark.parametrize("bad_kind", ["shape", "dtype", "nonfinite"])
def test_run_rejects_invalid_layout_without_publishing_output(tmp_path, bad_kind):
    values = _normalized_values()
    reference = tmp_path / "source.fbin"
    _write_fbin(reference, values)
    layout = tmp_path / "layout.npy"
    if bad_kind == "shape":
        np.save(layout, values[:-1, :2])
    elif bad_kind == "dtype":
        np.save(layout, np.ones((values.shape[0], 2), dtype=np.int32))
    else:
        bad = values[:, :2].copy()
        bad[-2, 0] = np.nan
        np.save(layout, bad)
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="shape|floating|non-finite"):
        run(_args(reference, [("dire", layout)], output))

    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.tmp-*"))


@pytest.mark.parametrize("bad_kind", ["nonfinite", "not_normalized"])
def test_run_rejects_invalid_reference_without_publishing_output(tmp_path, bad_kind):
    values = _normalized_values()
    if bad_kind == "nonfinite":
        values[5, 2] = np.inf
    else:
        values[5] *= 2
    reference = tmp_path / "source.fbin"
    _write_fbin(reference, values)
    layout = tmp_path / "layout.npy"
    np.save(layout, np.ones((values.shape[0], 2), dtype=np.float32))
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="non-finite|not row-L2 normalized"):
        run(_args(reference, [("dire", layout)], output))

    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.tmp-*"))


def test_run_rejects_duplicate_layout_targets_and_oversized_sample(tmp_path):
    values = _normalized_values()
    reference = tmp_path / "source.fbin"
    _write_fbin(reference, values)
    layout = tmp_path / "layout.npy"
    alias = tmp_path / "alias.npy"
    np.save(layout, values[:, :2])
    alias.symlink_to(layout.name)

    with pytest.raises(ValueError, match="distinct"):
        run(_args(reference, [("ae", layout), ("dire", alias)], tmp_path / "duplicate"))
    with pytest.raises(ValueError, match="exceeds"):
        run(
            _args(
                reference,
                [("ae", layout)],
                tmp_path / "oversized",
                sample_size=values.shape[0] + 1,
            )
        )
