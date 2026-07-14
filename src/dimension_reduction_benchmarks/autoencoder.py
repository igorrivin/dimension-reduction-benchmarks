"""Autoencoder reconstruction-elbow intrinsic-dimension diagnostic.

This module provides an intentionally expensive, global intrinsic-dimension diagnostic.
It trains the same autoencoder architecture across a range of latent bottleneck widths, measures
reconstruction error on validation observations, and selects an interior elbow in the
resulting error curve.

The two approaches need not agree.  TwoNN and MLE estimate local/geodesic dimension;
an autoencoder elbow estimates the smallest latent width that the chosen network can
use to reconstruct the globally embedded data.  Topology, noise, model capacity, and
optimization can all make the latter larger or less sharply defined.

PyTorch is imported only when :func:`autoencoder_intrinsic_dimension` is called. The
base benchmark package therefore remains importable without the optional dependency.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AutoencoderDimensionResult",
    "ReconstructionElbowResult",
    "autoencoder_intrinsic_dimension",
    "reconstruction_elbow",
]


@dataclass(frozen=True)
class ReconstructionElbowResult:
    """Result of applying the elbow rule to a reconstruction-error curve."""

    dimension: int | None
    index: int | None
    smoothed_errors: np.ndarray
    scores: np.ndarray
    strength: float
    relative_drop: float
    tail_drop_fraction: float
    reason: str | None


@dataclass(frozen=True)
class AutoencoderDimensionResult:
    """Autoencoder dimension estimate plus the evidence needed to audit it.

    ``dimension`` is ``None`` when the validation reconstruction curve has no interior
    elbow meeting the configured heuristic guards. ``reconstruction_errors`` is the
    median across repeated fits, while ``repeat_errors`` retains every normalized
    validation error. ``smoothed_errors`` is the monotone curve used for selection and
    ``elbow_scores`` contains its normalized distance below the endpoint chord.

    ``epochs_run`` records the optimization budget consumed by each fit and
    ``early_stopped`` reports whether its patience guard fired; it is not a claim of
    statistical convergence. ``tail_drop_fraction`` measures how much of the total error
    reduction still occurs over the final quarter of the candidate grid.
    """

    dimension: int | None
    dimensions: np.ndarray
    reconstruction_errors: np.ndarray
    repeat_errors: np.ndarray
    smoothed_errors: np.ndarray
    elbow_scores: np.ndarray
    elbow_strength: float
    relative_drop: float
    tail_drop_fraction: float
    ambiguity_reason: str | None
    epochs_run: np.ndarray
    early_stopped: np.ndarray
    n_train: int
    n_validation: int
    baseline_mse: float
    device: str


def _as_positive_int(value, name):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _pava_decreasing(values):
    """Least-squares non-increasing fit using the pool-adjacent-violators algorithm."""
    values = np.asarray(values, dtype=np.float64)
    levels = []
    counts = []
    for value in -values:  # increasing PAVA on the negated curve
        levels.append(float(value))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total = counts[-2] + counts[-1]
            merged = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / total
            levels[-2:] = [merged]
            counts[-2:] = [total]
    return -np.repeat(np.asarray(levels, dtype=np.float64), counts)


def reconstruction_elbow(
    dimensions,
    errors,
    *,
    min_strength=0.05,
    min_relative_drop=0.05,
    max_tail_drop=0.01,
):
    """Find an interior knee in a validation reconstruction-error curve.

    The errors are first fitted by a non-increasing isotonic curve.  Dimensions and
    errors are then normalized to ``[0, 1]`` and the elbow score is the vertical gap
    below the line joining the endpoints.  Endpoints are never accepted as elbows.

    Parameters
    ----------
    dimensions : one-dimensional array-like of int
        Strictly increasing bottleneck widths.  At least three are required.
    errors : one-dimensional array-like of float
        Non-negative, finite validation reconstruction errors aligned with ``dimensions``.
    min_strength : float, default=0.05
        Minimum normalized distance below the endpoint chord.
    min_relative_drop : float, default=0.05
        Minimum fractional decrease from the first to last smoothed error.  This keeps
        numerical noise on a flat curve from manufacturing an elbow.
    max_tail_drop : float, default=0.01
        Maximum fraction of the total error reduction allowed over the final quarter of
        the dimension grid. Curves that are still materially falling at the grid boundary
        do not establish a saturated intrinsic width.

    Returns
    -------
    ReconstructionElbowResult
        ``dimension`` and ``index`` are ``None`` when no elbow meets the configured
        heuristic guards.
    """
    dimensions = np.asarray(dimensions)
    errors = np.asarray(errors, dtype=np.float64)
    if dimensions.ndim != 1 or errors.ndim != 1 or dimensions.size != errors.size:
        raise ValueError("dimensions and errors must be aligned one-dimensional arrays")
    if dimensions.size < 3:
        raise ValueError("at least three candidate dimensions are needed to find an elbow")
    if not np.issubdtype(dimensions.dtype, np.integer):
        try:
            numeric_dimensions = dimensions.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("dimensions must contain integers") from exc
        if not np.all(np.isfinite(numeric_dimensions)) or not np.all(
            np.equal(numeric_dimensions, np.floor(numeric_dimensions))
        ):
            raise ValueError("dimensions must contain integers")
        dimensions = numeric_dimensions
    dimensions = dimensions.astype(np.int64, copy=False)
    if np.any(dimensions < 1) or np.any(np.diff(dimensions) <= 0):
        raise ValueError("dimensions must be positive and strictly increasing")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError("errors must be finite and non-negative")
    if not np.isfinite(min_strength) or min_strength < 0:
        raise ValueError("min_strength must be finite and non-negative")
    if not np.isfinite(min_relative_drop) or not 0 <= min_relative_drop < 1:
        raise ValueError("min_relative_drop must be in [0, 1)")
    if not np.isfinite(max_tail_drop) or not 0 <= max_tail_drop < 1:
        raise ValueError("max_tail_drop must be in [0, 1)")

    smoothed = _pava_decreasing(errors)
    first = float(smoothed[0])
    last = float(smoothed[-1])
    scores = np.zeros_like(smoothed)
    if first <= np.finfo(np.float64).eps:
        relative_drop = 0.0
        return ReconstructionElbowResult(
            dimension=None,
            index=None,
            smoothed_errors=smoothed,
            scores=scores,
            strength=0.0,
            relative_drop=relative_drop,
            tail_drop_fraction=0.0,
            reason="reconstruction curve is effectively flat",
        )
    relative_drop = max(0.0, (first - last) / first)
    if relative_drop < min_relative_drop or np.isclose(first, last):
        return ReconstructionElbowResult(
            dimension=None,
            index=None,
            smoothed_errors=smoothed,
            scores=scores,
            strength=0.0,
            relative_drop=relative_drop,
            tail_drop_fraction=0.0,
            reason="reconstruction curve is effectively flat",
        )

    tail_start_value = dimensions[-1] - 0.25 * (dimensions[-1] - dimensions[0])
    tail_start = int(np.searchsorted(dimensions, tail_start_value, side="left"))
    tail_start = min(max(1, tail_start), dimensions.size - 2)
    tail_drop_fraction = max(0.0, (smoothed[tail_start] - last) / (first - last))

    x = (dimensions - dimensions[0]) / (dimensions[-1] - dimensions[0])
    y = (smoothed - last) / (first - last)
    scores = (1.0 - x) - y
    scores[[0, -1]] = 0.0
    index = int(np.argmax(scores[1:-1]) + 1)
    strength = float(scores[index])
    if strength < min_strength:
        return ReconstructionElbowResult(
            dimension=None,
            index=None,
            smoothed_errors=smoothed,
            scores=scores,
            strength=strength,
            relative_drop=relative_drop,
            tail_drop_fraction=tail_drop_fraction,
            reason="no sufficiently strong interior elbow",
        )

    if tail_drop_fraction > max_tail_drop:
        return ReconstructionElbowResult(
            dimension=None,
            index=None,
            smoothed_errors=smoothed,
            scores=scores,
            strength=strength,
            relative_drop=relative_drop,
            tail_drop_fraction=tail_drop_fraction,
            reason="reconstruction curve is still falling at the largest bottlenecks",
        )

    return ReconstructionElbowResult(
        dimension=int(dimensions[index]),
        index=index,
        smoothed_errors=smoothed,
        scores=scores,
        strength=strength,
        relative_drop=relative_drop,
        tail_drop_fraction=tail_drop_fraction,
        reason=None,
    )


def _candidate_dimensions(n_features, dimensions, max_dimension):
    if dimensions is None:
        max_dimension = _as_positive_int(max_dimension, "max_dimension")
        upper = min(max_dimension, n_features - 1)
        if upper < 3:
            raise ValueError(
                "the default candidate grid needs max_dimension >= 3 and at least "
                "four input features"
            )
        return np.arange(1, upper + 1, dtype=np.int64)

    try:
        raw_dimensions = list(dimensions)
    except TypeError as exc:
        raise ValueError("dimensions must be an iterable of integers") from exc
    parsed = np.asarray(
        [_as_positive_int(value, "each candidate dimension") for value in raw_dimensions],
        dtype=np.int64,
    )
    if parsed.size < 3:
        raise ValueError("at least three candidate dimensions are required")
    if np.any(np.diff(parsed) <= 0):
        raise ValueError("candidate dimensions must be strictly increasing")
    if parsed[-1] >= n_features:
        raise ValueError(
            f"candidate dimensions must be smaller than the ambient dimension ({n_features})"
        )
    return parsed


def _load_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise ImportError(
            "autoencoder intrinsic-dimension estimation needs PyTorch; install it with "
            "`python -m pip install 'dimension-reduction-benchmarks[ae]'`"
        ) from exc
    return torch


def _resolve_device(torch, requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid torch device {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("a CUDA device was requested, but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None:
        device_count = torch.cuda.device_count()
        if device.index < 0 or device.index >= device_count:
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; found {device_count} device(s)"
            )
    return device


def _activation(torch, name):
    activations = {
        "gelu": torch.nn.GELU,
        "relu": torch.nn.ReLU,
        "silu": torch.nn.SiLU,
        "tanh": torch.nn.Tanh,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError("activation must be one of 'gelu', 'relu', 'silu', or 'tanh'") from exc


def _build_autoencoder(torch, n_features, hidden_dims, bottleneck, activation):
    widths = [n_features, *hidden_dims, bottleneck, *reversed(hidden_dims), n_features]
    layers = []
    for index, (source, target) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(torch.nn.Linear(source, target))
        # Keep the latent representation linear. Activating the bottleneck would
        # confound width with a sign/range constraint, especially for ReLU or SiLU.
        if index < len(widths) - 2 and index != len(hidden_dims):
            layers.append(activation())
    return torch.nn.Sequential(*layers)


def _validation_mse(torch, model, validation, batch_size, device):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(validation),
        batch_size=batch_size,
        shuffle=False,
    )
    squared_error = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            prediction = model(batch)
            squared_error += float(torch.sum((prediction - batch) ** 2).item())
            count += batch.numel()
    return squared_error / count


def _fit_one(
    torch,
    train,
    validation,
    *,
    bottleneck,
    hidden_dims,
    activation,
    epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
    min_delta,
    seed,
    device,
):
    # The model is initialized on CPU before being moved to ``device``. Seed only the
    # CPU default generator so explicit CPU use does not initialize or perturb CUDA RNGs.
    torch.default_generator.manual_seed(seed)
    model = _build_autoencoder(
        torch,
        n_features=train.shape[1],
        hidden_dims=hidden_dims,
        bottleneck=bottleneck,
        activation=activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    best = float("inf")
    stale_epochs = 0
    epochs_run = 0
    early_stopped = False
    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.mean((prediction - batch) ** 2)
            loss.backward()
            optimizer.step()

        value = _validation_mse(torch, model, validation, batch_size, device)
        if not np.isfinite(value):
            raise RuntimeError(
                "autoencoder produced a non-finite validation loss at "
                f"bottleneck={bottleneck}, epoch={epoch + 1}"
            )
        epochs_run = epoch + 1
        threshold = min_delta * max(abs(best), np.finfo(np.float64).eps)
        if not np.isfinite(best) or best - value > threshold:
            best = value
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                early_stopped = True
                break
    return best, epochs_run, early_stopped


def autoencoder_intrinsic_dimension(
    X,
    dimensions: Sequence[int] | None = None,
    *,
    max_dimension=32,
    hidden_dims=(128, 64),
    validation_fraction=0.2,
    max_samples=10_000,
    epochs=200,
    patience=20,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
    min_delta=1e-5,
    repeats=1,
    random_state=0,
    device="auto",
    activation="silu",
    min_elbow_strength=0.05,
    min_relative_drop=0.05,
    max_tail_drop=0.01,
    deterministic=True,
):
    """Estimate a global latent dimension from an autoencoder error elbow.

    Every bottleneck width uses the same symmetric architecture and the same sampled
    train/validation split. Reconstruction errors are measured on data held out from
    gradient updates (and reused for early stopping), then
    normalized by the error of predicting the training-set mean.  Repeated fits are
    aggregated by their median before elbow detection.

    The data are centered using the training mean and divided by one global RMS scale.
    This improves optimization without changing the relative scaling of features or the
    Euclidean geometry of the input representation.

    Parameters
    ----------
    X : array (n_samples, n_features)
        Finite numeric observations.  At least five samples and four features are needed
        for the default candidate grid.
    dimensions : sequence of int, optional
        Strictly increasing bottleneck widths.  All must be smaller than ``n_features``.
        The default is every integer from 1 through
        ``min(max_dimension, n_features - 1)``.
    max_dimension : int, default=32
        Upper bound for the default bottleneck grid.
    hidden_dims : sequence of int, default=(128, 64)
        Encoder widths before the bottleneck; the decoder mirrors them.
    validation_fraction : float, default=0.2
        Fraction of sampled observations held out from optimization.
    max_samples : int or None, default=10000
        Deterministic subsample cap.  ``None`` uses every row.
    epochs, patience, batch_size : int
        Per-fit optimization budget and early-stopping controls.
    learning_rate, weight_decay : float
        AdamW optimizer settings.
    min_delta : float, default=1e-5
        Relative validation-loss improvement required to reset early stopping.
    repeats : int, default=1
        Independent fits per bottleneck.  Values of 3 or more give a more stable curve.
    random_state : int, default=0
        Controls subsampling, splitting, initialization, and batch order.
    device : str, default='auto'
        PyTorch device.  ``'auto'`` uses CUDA when available, otherwise CPU.
    activation : {'gelu','relu','silu','tanh'}, default='silu'
        Hidden-layer activation used throughout the symmetric autoencoder.
    min_elbow_strength, min_relative_drop, max_tail_drop : float
        Heuristic ambiguity guards passed to :func:`reconstruction_elbow`. The tail guard
        rejects curves that are still materially falling at the largest tested widths;
        elbow selection therefore depends on the supplied candidate grid.
    deterministic : bool, default=True
        Ask PyTorch for deterministic algorithms in best-effort warning mode during these
        fits, restoring the prior global setting before returning.

    Returns
    -------
    AutoencoderDimensionResult
        The selected integer dimension, or ``None`` when no elbow meets the configured
        heuristic guards, together with the full validation reconstruction curve and
        training diagnostics.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if X.shape[0] < 5:
        raise ValueError("X must contain at least five samples")
    if X.shape[1] < 2:
        raise ValueError("X must contain at least two features")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    candidate_dimensions = _candidate_dimensions(
        X.shape[1], dimensions=dimensions, max_dimension=max_dimension
    )
    try:
        hidden_dims = tuple(
            _as_positive_int(width, "each hidden dimension") for width in hidden_dims
        )
    except TypeError as exc:
        raise ValueError("hidden_dims must be an iterable of positive integers") from exc
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer width")
    if candidate_dimensions[-1] >= min(hidden_dims):
        raise ValueError(
            "every hidden dimension must exceed the largest candidate bottleneck width"
        )
    epochs = _as_positive_int(epochs, "epochs")
    patience = _as_positive_int(patience, "patience")
    batch_size = _as_positive_int(batch_size, "batch_size")
    repeats = _as_positive_int(repeats, "repeats")
    try:
        random_state = operator.index(random_state)
    except TypeError as exc:
        raise ValueError("random_state must be an integer") from exc
    if not 0 <= random_state or random_state + repeats > 2**63:
        raise ValueError("random_state + repeats must stay in [0, 2**63]")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if weight_decay < 0 or not np.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite and non-negative")
    if min_delta < 0 or not np.isfinite(min_delta):
        raise ValueError("min_delta must be finite and non-negative")
    if min_elbow_strength < 0 or not np.isfinite(min_elbow_strength):
        raise ValueError("min_elbow_strength must be finite and non-negative")
    if not np.isfinite(min_relative_drop) or not 0 <= min_relative_drop < 1:
        raise ValueError("min_relative_drop must be in [0, 1)")
    if not np.isfinite(max_tail_drop) or not 0 <= max_tail_drop < 1:
        raise ValueError("max_tail_drop must be in [0, 1)")
    if max_samples is not None:
        max_samples = _as_positive_int(max_samples, "max_samples")

    rng = np.random.default_rng(random_state)
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], size=max_samples, replace=False)]
    order = rng.permutation(X.shape[0])
    n_validation = max(1, int(round(validation_fraction * X.shape[0])))
    n_validation = min(n_validation, X.shape[0] - 2)
    validation_np = X[order[:n_validation]]
    train_np = X[order[n_validation:]]

    center = np.mean(train_np, axis=0, dtype=np.float64).astype(np.float32)
    centered_train = train_np - center
    scale = float(np.sqrt(np.mean(centered_train.astype(np.float64) ** 2)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        raise ValueError("X has no reconstructable variation after centering")
    train_np = np.ascontiguousarray(centered_train / scale, dtype=np.float32)
    validation_np = np.ascontiguousarray((validation_np - center) / scale, dtype=np.float32)
    baseline_mse = float(np.mean(validation_np.astype(np.float64) ** 2))
    if not np.isfinite(baseline_mse) or baseline_mse <= 0:
        raise ValueError("validation split has no variation around the training mean")

    torch = _load_torch()
    resolved_device = _resolve_device(torch, device)
    activation_class = _activation(torch, activation)
    train = torch.from_numpy(train_np)
    validation = torch.from_numpy(validation_np)
    effective_batch_size = min(batch_size, train.shape[0])

    repeat_errors = np.empty((candidate_dimensions.size, repeats), dtype=np.float64)
    epochs_run = np.empty((candidate_dimensions.size, repeats), dtype=np.int64)
    early_stopped = np.empty((candidate_dimensions.size, repeats), dtype=bool)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if deterministic:
        # Strict deterministic mode can make otherwise valid CUDA/cuBLAS operations fail
        # unless the process was configured before CUDA initialization. Warning mode keeps
        # the automatic CUDA path usable while preferring deterministic kernels.
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        # Model initialization uses PyTorch's CPU default generator. fork_rng makes the
        # estimator reproducible without changing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            for dimension_index, bottleneck in enumerate(candidate_dimensions):
                for repeat in range(repeats):
                    loss, used_epochs, did_early_stop = _fit_one(
                        torch,
                        train,
                        validation,
                        bottleneck=int(bottleneck),
                        hidden_dims=hidden_dims,
                        activation=activation_class,
                        epochs=epochs,
                        patience=patience,
                        batch_size=effective_batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        min_delta=min_delta,
                        seed=int(random_state + repeat),
                        device=resolved_device,
                    )
                    repeat_errors[dimension_index, repeat] = loss / baseline_mse
                    epochs_run[dimension_index, repeat] = used_epochs
                    early_stopped[dimension_index, repeat] = did_early_stop
    finally:
        if deterministic:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)

    reconstruction_errors = np.median(repeat_errors, axis=1)
    elbow = reconstruction_elbow(
        candidate_dimensions,
        reconstruction_errors,
        min_strength=min_elbow_strength,
        min_relative_drop=min_relative_drop,
        max_tail_drop=max_tail_drop,
    )
    return AutoencoderDimensionResult(
        dimension=elbow.dimension,
        dimensions=candidate_dimensions,
        reconstruction_errors=reconstruction_errors,
        repeat_errors=repeat_errors,
        smoothed_errors=elbow.smoothed_errors,
        elbow_scores=elbow.scores,
        elbow_strength=elbow.strength,
        relative_drop=elbow.relative_drop,
        tail_drop_fraction=elbow.tail_drop_fraction,
        ambiguity_reason=elbow.reason,
        epochs_run=epochs_run,
        early_stopped=early_stopped,
        n_train=train.shape[0],
        n_validation=validation.shape[0],
        baseline_mse=baseline_mse,
        device=str(resolved_device),
    )
