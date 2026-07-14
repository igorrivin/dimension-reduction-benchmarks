"""Compare local neighbour-based and global autoencoder dimension estimates.

Examples
--------
Run the single linear-subspace comparison::

    python examples/compare_dimension_estimators.py --dataset linear

Run all examples (the autoencoder sweep is intentionally compute-heavy)::

    python examples/compare_dimension_estimators.py --dataset all --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from dire_evoc import intrinsic_dimension

from dimension_reduction_benchmarks import autoencoder_intrinsic_dimension


def _embed(X, ambient, rng):
    if X.shape[1] > ambient:
        raise ValueError("ambient dimension must be at least the source dimension")
    basis, _ = np.linalg.qr(rng.normal(size=(ambient, X.shape[1])))
    return X @ basis.T


def _make_dataset(name, n_samples, ambient, true_dimension, seed):
    rng = np.random.default_rng(seed)
    if name == "linear":
        latent = rng.normal(size=(n_samples, true_dimension))
        X = _embed(latent, ambient, rng)
        return X, true_dimension, "low-rank linear subspace"
    if name == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        X = np.column_stack((np.cos(theta), np.sin(theta)))
        return _embed(X, ambient, rng), 1, "circle (global topology needs two coordinates)"
    if name == "swiss-roll":
        from sklearn.datasets import make_swiss_roll

        X, _ = make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=seed)
        return _embed(X, ambient, rng), 2, "Swiss roll"
    raise ValueError(f"unknown dataset {name!r}")


def _run_one(name, args):
    X, known_dimension, description = _make_dataset(
        name,
        n_samples=args.samples,
        ambient=args.ambient,
        true_dimension=args.true_dimension,
        seed=args.seed,
    )
    twonn = intrinsic_dimension(X, k=args.neighbors, method="twonn")
    mle = intrinsic_dimension(X, k=args.neighbors, method="mle")
    started = time.perf_counter()
    autoencoder = autoencoder_intrinsic_dimension(
        X,
        max_dimension=args.max_dimension,
        epochs=args.epochs,
        patience=args.patience,
        repeats=args.repeats,
        random_state=args.seed,
        device=args.device,
    )
    elapsed = time.perf_counter() - started

    print(f"\n{name}: {description}")
    print(f"  known local dimension: {known_dimension}")
    print(f"  TwoNN:                {twonn:.3f}")
    print(f"  MLE:                  {mle:.3f}")
    print(f"  autoencoder elbow:    {autoencoder.dimension}")
    if autoencoder.ambiguity_reason is not None:
        print(f"  elbow status:          {autoencoder.ambiguity_reason}")
    print(f"  elbow strength:        {autoencoder.elbow_strength:.4f}")
    print(f"  total error drop:      {autoencoder.relative_drop:.4f}")
    print(f"  final-quarter drop:    {autoencoder.tail_drop_fraction:.4f}")
    print(f"  autoencoder device:   {autoencoder.device} ({elapsed:.2f}s)")
    print("  bottleneck -> validation error | smoothed error | elbow score")
    for dimension, error, smoothed, score in zip(
        autoencoder.dimensions,
        autoencoder.reconstruction_errors,
        autoencoder.smoothed_errors,
        autoencoder.elbow_scores,
        strict=True,
    ):
        print(f"    {dimension:2d} -> {error:.6f} | {smoothed:.6f} | {score:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("linear", "circle", "swiss-roll", "all"),
        default="linear",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--ambient", type=int, default=12)
    parser.add_argument("--true-dimension", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--max-dimension", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    datasets = ("linear", "circle", "swiss-roll") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        _run_one(dataset, args)


if __name__ == "__main__":
    main()
