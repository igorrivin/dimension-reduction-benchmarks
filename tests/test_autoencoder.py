"""Tests for the optional autoencoder reconstruction-elbow diagnostic."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dimension_reduction_benchmarks.autoencoder import (
    autoencoder_intrinsic_dimension,
    reconstruction_elbow,
)


def test_reconstruction_elbow_finds_sharp_interior_knee():
    result = reconstruction_elbow(
        dimensions=np.arange(1, 8),
        errors=np.array([1.0, 0.60, 0.20, 0.11, 0.10, 0.095, 0.09]),
    )

    assert result.dimension == 3
    assert result.index == 2
    assert result.strength > 0.5
    assert result.reason is None


def test_reconstruction_elbow_smooths_noisy_curve():
    result = reconstruction_elbow(
        dimensions=[1, 2, 3, 4, 5, 6],
        errors=[1.0, 0.60, 0.65, 0.20, 0.21, 0.10],
    )

    assert np.all(np.diff(result.smoothed_errors) <= 0)
    assert result.smoothed_errors[1] == pytest.approx(0.625)
    assert result.smoothed_errors[3] == pytest.approx(0.205)


def test_reconstruction_elbow_flags_flat_curve():
    result = reconstruction_elbow(
        dimensions=[1, 2, 3, 4],
        errors=[1.0, 0.99, 1.01, 1.0],
    )

    assert result.dimension is None
    assert result.index is None
    assert result.reason == "reconstruction curve is effectively flat"


def test_reconstruction_elbow_accepts_exact_zero_error():
    result = reconstruction_elbow(
        dimensions=[1, 2, 3, 4],
        errors=[1.0, 0.0, 0.0, 0.0],
    )

    assert result.dimension == 2
    assert result.reason is None


def test_reconstruction_elbow_flags_curve_still_falling_at_grid_boundary():
    dimensions = np.arange(1, 17)
    result = reconstruction_elbow(dimensions, 1.0 / dimensions)

    assert result.dimension is None
    assert result.tail_drop_fraction > 0.01
    assert result.reason == "reconstruction curve is still falling at the largest bottlenecks"


@pytest.mark.parametrize(
    ("dimensions", "errors", "message"),
    [
        ([1, 2], [1.0, 0.5], "at least three"),
        ([1, 3, 2], [1.0, 0.5, 0.2], "strictly increasing"),
        (["one", 2, 3], [1.0, 0.5, 0.2], "integers"),
        ([1, np.nan, 3], [1.0, 0.5, 0.2], "integers"),
        ([1, 2, 3], [1.0, np.nan, 0.2], "finite"),
    ],
)
def test_reconstruction_elbow_validates_inputs(dimensions, errors, message):
    with pytest.raises(ValueError, match=message):
        reconstruction_elbow(dimensions, errors)


def test_autoencoder_validates_before_loading_torch():
    with pytest.raises(ValueError, match="smaller than the ambient"):
        autoencoder_intrinsic_dimension(np.ones((10, 4)), dimensions=[1, 2, 4], device="cpu")


def test_autoencoder_recovers_low_rank_dimension_on_cpu():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    latent = rng.uniform(-0.5, 0.5, size=(320, 2))
    basis, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    X = latent @ basis.T

    torch.default_generator.manual_seed(12345)
    caller_rng_state = torch.default_generator.get_state().clone()
    result = autoencoder_intrinsic_dimension(
        X,
        dimensions=[1, 2, 3, 4, 5],
        hidden_dims=(24, 12),
        epochs=150,
        patience=15,
        batch_size=64,
        repeats=1,
        random_state=11,
        activation="tanh",
        device="cpu",
    )

    assert result.dimension in {2, 3}
    assert result.reconstruction_errors.shape == (5,)
    assert result.repeat_errors.shape == (5, 1)
    assert result.epochs_run.shape == (5, 1)
    assert result.early_stopped.shape == (5, 1)
    assert np.all(np.isfinite(result.reconstruction_errors))
    assert result.reconstruction_errors[0] > result.reconstruction_errors[1]
    assert result.n_train + result.n_validation == X.shape[0]
    assert result.device == "cpu"
    assert torch.equal(torch.default_generator.get_state(), caller_rng_state)


def test_public_package_import_is_torch_lazy():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dimension_reduction_benchmarks; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_autoencoder_rejects_out_of_range_random_state():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="random_state"):
        autoencoder_intrinsic_dimension(
            rng.normal(size=(10, 5)),
            dimensions=[1, 2, 3],
            random_state=-1,
            device="cpu",
        )
