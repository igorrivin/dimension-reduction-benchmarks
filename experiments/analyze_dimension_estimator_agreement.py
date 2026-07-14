"""Controlled multi-seed comparison of local and reconstruction dimensions.

This experiment is deliberately separate from the large fixed-target embedding
benchmark.  It compares the TwoNN and Levina--Bickel MLE local-dimension estimators with
the autoencoder validation-reconstruction elbow on manifolds whose local dimensions are
known.  The input-space neighbours are exact CPU neighbours; this does not fit a DiRe
layout or measure the two large text corpora.

Example
-------
PYTHONPATH=src conda run -n rapids-26.06 python \
    experiments/analyze_dimension_estimator_agreement.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import dire_evoc.dimension as dire_dimension
import numpy as np
from dire_evoc import mle, twonn
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from dimension_reduction_benchmarks import autoencoder_intrinsic_dimension

DATASETS = ("linear", "circle", "swiss-roll")
METHODS = ("twonn", "mle", "autoencoder")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embed(coordinates: np.ndarray, ambient: int, rng: np.random.Generator) -> np.ndarray:
    if coordinates.shape[1] > ambient:
        raise ValueError("ambient dimension must be at least the source dimension")
    basis, _ = np.linalg.qr(rng.normal(size=(ambient, coordinates.shape[1])))
    return coordinates @ basis.T


def _make_dataset(
    name: str,
    *,
    n_samples: int,
    ambient: int,
    true_dimension: int,
    seed: int,
) -> tuple[np.ndarray, int, str]:
    """Match the generators in examples/compare_dimension_estimators.py exactly."""
    rng = np.random.default_rng(seed)
    if name == "linear":
        latent = rng.normal(size=(n_samples, true_dimension))
        return _embed(latent, ambient, rng), true_dimension, "rank-3 linear subspace"
    if name == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        coordinates = np.column_stack((np.cos(theta), np.sin(theta)))
        return _embed(coordinates, ambient, rng), 1, "circle"
    if name == "swiss-roll":
        from sklearn.datasets import make_swiss_roll

        coordinates, _ = make_swiss_roll(
            n_samples=n_samples,
            noise=0.0,
            random_state=seed,
        )
        return _embed(coordinates, ambient, rng), 2, "noise-free Swiss roll"
    raise ValueError(f"unknown dataset {name!r}")


def _as_list(values: np.ndarray) -> list:
    return values.tolist()


def _run_dataset(name: str, seed: int, args: argparse.Namespace) -> dict:
    data, local_dimension, description = _make_dataset(
        name,
        n_samples=args.samples,
        ambient=args.ambient,
        true_dimension=args.true_dimension,
        seed=seed,
    )
    neighbor_distance, _ = (
        NearestNeighbors(n_neighbors=args.neighbors, metric="euclidean").fit(data).kneighbors()
    )
    twonn_result = twonn(neighbor_distance, metric="euclidean")
    twonn_dimension = twonn_result["d"]
    mle_dimension = mle(neighbor_distance, k=args.neighbors, metric="euclidean")
    mle_n_used = int(np.sum(np.all(neighbor_distance[:, : args.neighbors] > 0, axis=1)))
    started = time.perf_counter()
    autoencoder = autoencoder_intrinsic_dimension(
        data,
        max_dimension=args.max_dimension,
        epochs=args.epochs,
        patience=args.patience,
        repeats=1,
        random_state=seed,
        device=args.device,
    )
    elapsed = time.perf_counter() - started
    result = {
        "dataset": name,
        "description": description,
        "seed": seed,
        "known_local_dimension": local_dimension,
        "twonn": float(twonn_dimension),
        "twonn_n_used": int(twonn_result["n_used"]),
        "twonn_rows_excluded_as_degenerate": int(data.shape[0] - twonn_result["n_used"]),
        "mle": float(mle_dimension),
        "mle_n_used": mle_n_used,
        "mle_rows_excluded_as_degenerate": int(data.shape[0] - mle_n_used),
        "autoencoder_dimension": autoencoder.dimension,
        "autoencoder_elapsed_seconds": elapsed,
        "autoencoder": {
            "candidate_dimensions": _as_list(autoencoder.dimensions),
            "validation_nmse": _as_list(autoencoder.reconstruction_errors),
            "repeat_validation_nmse": _as_list(autoencoder.repeat_errors),
            "smoothed_validation_nmse": _as_list(autoencoder.smoothed_errors),
            "elbow_scores": _as_list(autoencoder.elbow_scores),
            "elbow_strength": autoencoder.elbow_strength,
            "relative_drop": autoencoder.relative_drop,
            "tail_drop_fraction": autoencoder.tail_drop_fraction,
            "ambiguity_reason": autoencoder.ambiguity_reason,
            "epochs_run": _as_list(autoencoder.epochs_run),
            "early_stopped": _as_list(autoencoder.early_stopped),
            "n_train": autoencoder.n_train,
            "n_validation": autoencoder.n_validation,
            "baseline_mse": autoencoder.baseline_mse,
            "device": autoencoder.device,
        },
    }
    print(
        f"{name:10s} seed={seed}: TwoNN={twonn_dimension:.3f}, MLE={mle_dimension:.3f}, "
        f"AE={autoencoder.dimension}, AE seconds={elapsed:.2f}",
        flush=True,
    )
    return result


def _method_values(records: list[dict], dataset: str, method: str) -> np.ndarray:
    selected = [record for record in records if record["dataset"] == dataset]
    key = "autoencoder_dimension" if method == "autoencoder" else method
    values = [record[key] for record in selected if record[key] is not None]
    return np.asarray(values, dtype=np.float64)


def _summarize(records: list[dict]) -> dict:
    by_dataset = {}
    for dataset in DATASETS:
        dataset_records = [record for record in records if record["dataset"] == dataset]
        local_dimension = dataset_records[0]["known_local_dimension"]
        by_method = {}
        for method in METHODS:
            values = _method_values(records, dataset, method)
            by_method[method] = {
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean_absolute_error_from_known_local_dimension": float(
                    np.mean(np.abs(values - local_dimension))
                ),
            }
        rounded_twonn = np.rint(
            np.asarray([record["twonn"] for record in dataset_records], dtype=np.float64)
        ).astype(int)
        ae = np.asarray([record["autoencoder_dimension"] for record in dataset_records], dtype=int)
        by_dataset[dataset] = {
            "known_local_dimension": local_dimension,
            "methods": by_method,
            "rounded_twonn_exact_ae_count": int(np.sum(rounded_twonn == ae)),
            "rounded_twonn_within_one_of_ae_count": int(np.sum(np.abs(rounded_twonn - ae) <= 1)),
            "runs": len(dataset_records),
        }

    twonn = np.asarray([record["twonn"] for record in records], dtype=np.float64)
    ae = np.asarray([record["autoencoder_dimension"] for record in records], dtype=np.float64)
    rounded_twonn = np.rint(twonn).astype(int)
    rank_correlation = spearmanr(twonn, ae)
    return {
        "by_dataset": by_dataset,
        "paired_descriptive_summary": {
            "runs": len(records),
            "mean_signed_ae_minus_twonn": float(np.mean(ae - twonn)),
            "mean_absolute_ae_minus_twonn": float(np.mean(np.abs(ae - twonn))),
            "rounded_twonn_exact_ae_count": int(np.sum(rounded_twonn == ae)),
            "rounded_twonn_within_one_of_ae_count": int(np.sum(np.abs(rounded_twonn - ae) <= 1)),
            "spearman_rho": float(rank_correlation.statistic),
            "spearman_p_value_descriptive_only": float(rank_correlation.pvalue),
            "inference_warning": (
                "The nine paired values are clustered within three dataset families; "
                "the correlation and p-value are descriptive, not inferential."
            ),
        },
    }


def _plot(records: list[dict], pdf_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "twonn": "#0072B2",
        "mle": "#009E73",
        "autoencoder": "#D55E00",
    }
    labels = {
        "twonn": "TwoNN",
        "mle": "MLE (k=20)",
        "autoencoder": "AE elbow",
    }
    offsets = {"twonn": -0.23, "mle": 0.0, "autoencoder": 0.23}
    seed_jitter = np.linspace(-0.035, 0.035, len({record["seed"] for record in records}))

    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    for dataset_index, dataset in enumerate(DATASETS):
        dataset_records = [record for record in records if record["dataset"] == dataset]
        local_dimension = dataset_records[0]["known_local_dimension"]
        ax.hlines(
            local_dimension,
            dataset_index - 0.43,
            dataset_index + 0.43,
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        for method in METHODS:
            values = _method_values(records, dataset, method)
            x_center = dataset_index + offsets[method]
            ax.scatter(
                x_center + seed_jitter[: values.size],
                values,
                s=34,
                color=colors[method],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.hlines(
                np.mean(values),
                x_center - 0.075,
                x_center + 0.075,
                color=colors[method],
                linewidth=2.3,
                zorder=4,
            )

    ax.set_xticks(range(len(DATASETS)), ["rank-3 linear", "circle", "Swiss roll"])
    ax.set_ylabel("estimated dimension")
    ax.set_ylim(0.65, 3.45)
    ax.set_yticks([1, 2, 3])
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Controlled dimension-estimator cross-check (three seeds)", fontsize=11)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[method],
            markeredgecolor="white",
            markersize=7,
            label=labels[method],
        )
        for method in METHODS
    ]
    handles.append(Line2D([0], [0], color="0.25", linestyle="--", label="known local ID"))
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False
    )
    fig.text(
        0.5,
        -0.005,
        "Dots are independent data/model seeds; short bars are means. "
        "TwoNN/MLE use exact input-space neighbors.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--ambient", type=int, default=10)
    parser.add_argument("--true-dimension", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--max-dimension", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("papers/figures/dimension_estimator_agreement.json"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("papers/figures/dimension_estimator_agreement.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("papers/figures/dimension_estimator_agreement.png"),
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be distinct")

    records = [_run_dataset(dataset, seed, args) for dataset in DATASETS for seed in args.seeds]
    summary = _summarize(records)

    repository = Path(__file__).resolve().parents[1]
    source_paths = {
        "analysis_script": Path(__file__).resolve(),
        "dimension_estimator": Path(inspect.getfile(dire_dimension)).resolve(),
        "autoencoder_estimator": (repository / "src/dimension_reduction_benchmarks/autoencoder.py"),
        "original_single_seed_example": repository / "examples/compare_dimension_estimators.py",
    }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Controlled synthetic estimator cross-check; not an arXiv/Wikipedia result "
            "and not a DiRe layout fit."
        ),
        "protocol": {
            "datasets": list(DATASETS),
            "seeds": args.seeds,
            "samples_per_dataset_seed": args.samples,
            "ambient_dimension": args.ambient,
            "linear_true_dimension": args.true_dimension,
            "input_neighbor_search": "exact sklearn CPU Euclidean kNN",
            "neighbors_retrieved": args.neighbors,
            "twonn_neighbors_used": 2,
            "twonn_tail_fraction": 0.9,
            "mle_neighbors_used": args.neighbors,
            "ae_candidate_dimensions": list(range(1, args.max_dimension + 1)),
            "ae_hidden_dimensions": [128, 64],
            "ae_activation": "SiLU with linear bottleneck and output",
            "ae_loss": "validation MSE divided by validation mean-predictor MSE",
            "ae_validation_fraction": 0.2,
            "ae_epochs_cap": args.epochs,
            "ae_patience": args.patience,
            "ae_batch_size": 256,
            "ae_optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)",
            "ae_repeats_per_seed_width": 1,
            "ae_elbow_guards": {
                "min_strength": 0.05,
                "min_relative_drop": 0.05,
                "max_tail_drop": 0.01,
            },
            "device": args.device,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "scikit-learn": _version("scikit-learn"),
            "torch": _version("torch"),
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "records": records,
        "summary": summary,
        "figures": {
            "pdf": str(args.output_pdf),
            "png": str(args.output_png),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _plot(records, args.output_pdf, args.output_png)

    paired = summary["paired_descriptive_summary"]
    print(
        "paired summary: "
        f"MAE(AE, TwoNN)={paired['mean_absolute_ae_minus_twonn']:.3f}; "
        f"rounded exact={paired['rounded_twonn_exact_ae_count']}/{paired['runs']}; "
        f"rounded within one={paired['rounded_twonn_within_one_of_ae_count']}/"
        f"{paired['runs']}; Spearman rho={paired['spearman_rho']:.3f}"
    )
    print(f"wrote {args.output_json}, {args.output_pdf}, and {args.output_png}")


if __name__ == "__main__":
    main()
