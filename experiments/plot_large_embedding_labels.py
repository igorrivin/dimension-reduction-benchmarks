#!/usr/bin/env python3
"""Plot label structure in the large arXiv and Wikipedia reducer layouts.

The script consumes the saved layouts produced by ``compare_large_embeddings.py``.
Selections depend only on labels and fixed random/hash seeds, never on a reducer's
coordinates.  It writes both vector PDF and high-resolution PNG versions, together
with a JSON provenance record that makes every displayed sample auditable.

Each corpus figure has two rows.  The first shows the native two-dimensional layouts;
the second shows separate, unsupervised two-dimensional PCA views of the corresponding
eight-dimensional layouts.  The d=8 PCA views are illustrative and cannot be ranked by
visible separation across methods.  Both rows use exactly the same colored and
background point indices.

The arXiv figure uses a balanced sample from the twelve most frequent primary
categories.  The Wikipedia figure uses a deterministic hash sample among article URLs
with 40--100 paragraphs.  Native-layout category and article metrics remain the
quantitative label tests, and are shown for both target widths and all three seeds.

Example
-------
::

    conda run -n rapids-26.06 python experiments/plot_large_embedding_labels.py \
      --artifacts-dir artifacts --output-dir papers/figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

import numpy as np

# Matplotlib otherwise tries to create a cache below $HOME, which is not writable in
# some benchmark and documentation builders.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dimension-reduction-benchmarks-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dimension-reduction-benchmarks-cache")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METHODS = ("autoencoder", "dire", "umap")
METHOD_NAMES = {"autoencoder": "Autoencoder", "dire": "DiRe", "umap": "UMAP"}
METHOD_COLORS = {
    "autoencoder": "#0072B2",
    "dire": "#D55E00",
    "umap": "#009E73",
}
FIGURE_SEED = 1729
WIKIPEDIA_HASH_SALT = "dire-evoc-wikipedia-article-figure-v1"


def sha256_indices(indices: np.ndarray) -> str:
    """Hash an index vector in a platform-independent representation."""
    array = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_label_counts(labels: np.ndarray) -> list[tuple[str, int]]:
    """Return labels ordered by decreasing count, then lexicographically."""
    unique, counts = np.unique(labels.astype(str), return_counts=True)
    return sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))


def choose_balanced_indices(
    labels: np.ndarray, selected_labels: Iterable[str], per_label: int, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select at most ``per_label`` rows independently within each label."""
    labels = labels.astype(str)
    rng = np.random.default_rng(seed)
    by_label: dict[str, np.ndarray] = {}
    for label in selected_labels:
        candidates = np.flatnonzero(labels == label)
        take = min(per_label, candidates.size)
        chosen = rng.choice(candidates, size=take, replace=False)
        by_label[label] = np.sort(chosen)
    combined = np.concatenate(list(by_label.values()))
    # Shuffling avoids systematic overplotting by whichever label happens to be last.
    return rng.permutation(combined), by_label


def choose_background_indices(
    labels: np.ndarray,
    highlighted_labels: Iterable[str],
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Uniformly sample rows outside the highlighted labels."""
    eligible = np.flatnonzero(~np.isin(labels.astype(str), tuple(highlighted_labels)))
    rng = np.random.default_rng(seed)
    size = min(sample_size, eligible.size)
    return np.sort(rng.choice(eligible, size=size, replace=False))


def robust_equal_limits(points: np.ndarray, quantile: float = 0.995) -> tuple[float, float, float]:
    """Return an isotropic robust view as ``center_x, center_y, radius``."""
    center = np.median(points, axis=0)
    deviations = np.abs(points - center)
    radius = float(np.quantile(deviations, quantile))
    if not np.isfinite(radius) or radius <= 0:
        radius = float(np.max(deviations))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    return float(center[0]), float(center[1]), 1.06 * radius


def configure_plot_style() -> None:
    """Use a compact journal-style Matplotlib configuration."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    """Write PDF and PNG variants and return their paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    common = {"bbox_inches": "tight", "facecolor": "white"}
    fig.savefig(
        pdf_path,
        metadata={
            "Title": stem.name.replace("_", " "),
            "Author": "dimension-reduction-benchmarks",
            "Subject": "Deterministic large-embedding comparison figure",
        },
        **common,
    )
    fig.savefig(png_path, **common)
    plt.close(fig)
    return [str(pdf_path), str(png_path)]


def layout_path(run_dir: Path, rows: int, method: str, dimension: int, seed: int) -> Path:
    return run_dir / f"n{rows}" / f"{method}-d{dimension}-s{seed}-layout.npy"


def load_layout(path: Path, expected_rows: int, expected_dimension: int) -> np.ndarray:
    """Memory-map and validate one saved layout."""
    layout = np.load(path, mmap_mode="r", allow_pickle=False)
    if layout.shape != (expected_rows, expected_dimension):
        raise ValueError(
            f"unexpected layout shape for {path}: {layout.shape}; "
            f"expected {(expected_rows, expected_dimension)}"
        )
    if not np.issubdtype(layout.dtype, np.floating):
        raise TypeError(f"layout is not floating point: {path} ({layout.dtype})")
    return layout


def plot_layout_panel(
    ax: plt.Axes,
    layout: np.ndarray,
    background: np.ndarray,
    colored: np.ndarray,
    colors: list,
    *,
    row: int,
    column: int,
    pca_fit_indices: np.ndarray | None = None,
    background_size: float,
    colored_size: float,
    view_quantile: float,
) -> dict:
    """Draw one native-d=2 or PCA-view-d=8 panel and return its view metadata."""
    plot_indices = np.concatenate((background, colored))
    if pca_fit_indices is None:
        projected = np.asarray(layout[plot_indices], dtype=np.float64)
        components = None
        explained = None
        x_label = "layout coordinate 1"
        y_label = "native d=2\ncoordinate 2" if column == 0 else ""
    else:
        projected, components, explained = pca_projection(layout, pca_fit_indices, plot_indices)
        x_label = "per-panel PCA coordinate 1"
        y_label = "d=8 PCA view\ncoordinate 2" if column == 0 else ""
        ax.text(
            0.02,
            0.98,
            f"PCA variance: {100 * explained.sum():.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )
    projected_background = projected[: background.size]
    projected_colored = projected[background.size :]
    center_x, center_y, radius = robust_equal_limits(projected, quantile=view_quantile)
    ax.scatter(
        projected_background[:, 0],
        projected_background[:, 1],
        s=background_size,
        c="#8f8f8f",
        alpha=0.10,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        projected_colored[:, 0],
        projected_colored[:, 1],
        s=colored_size,
        c=colors,
        alpha=0.64,
        linewidths=0,
        rasterized=True,
    )
    ax.set(
        title=METHOD_NAMES[METHODS[column]] if row == 0 else "",
        xlim=(center_x - radius, center_x + radius),
        ylim=(center_y - radius, center_y + radius),
        xlabel=x_label,
        ylabel=y_label,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(length=2)
    record = {
        "view_center": [center_x, center_y],
        "view_radius": radius,
        "view_quantile": view_quantile,
    }
    if components is not None:
        record.update(
            {
                "pca_components": components.tolist(),
                "explained_variance_ratio": explained.tolist(),
            }
        )
    return record


def verify_matching_labels(labels_paths: dict[int, Path], expected_rows: int) -> np.ndarray:
    """Load labels and require exact d=2/d=8 row alignment."""
    hashes = {dimension: sha256_file(path) for dimension, path in labels_paths.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError(f"label files are not identical across target widths: {hashes}")
    labels = np.load(labels_paths[2], allow_pickle=True)
    if labels.shape != (expected_rows,):
        raise ValueError(f"unexpected labels shape: {labels.shape}; expected {(expected_rows,)}")
    return labels


def plot_arxiv_categories(
    run_dirs: dict[int, Path],
    output_dir: Path,
    reducer_seed: int,
    provenance: dict,
) -> None:
    """Plot arXiv category structure at native widths two and eight."""
    rows = 723_457
    labels_paths = {
        dimension: run_dir / f"n{rows}" / "labels.npy" for dimension, run_dir in run_dirs.items()
    }
    labels = verify_matching_labels(labels_paths, rows)
    counts = sorted_label_counts(labels)
    selected_labels = [label for label, _ in counts[:12]]
    colored, by_label = choose_balanced_indices(
        labels, selected_labels, per_label=1_500, seed=FIGURE_SEED
    )
    background = choose_background_indices(
        labels, selected_labels, sample_size=30_000, seed=FIGURE_SEED + 1
    )
    rng = np.random.default_rng(FIGURE_SEED + 2)
    pca_fit_indices = np.sort(rng.choice(rows, size=100_000, replace=False))

    palette = list(plt.get_cmap("tab20").colors[:12])
    label_to_color = dict(zip(selected_labels, palette))
    label_strings = labels.astype(str)
    colors = [label_to_color[label] for label in label_strings[colored]]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.85), constrained_layout=True)
    layout_records: dict[str, dict] = {"d2": {}, "d8": {}}
    for row, dimension in enumerate((2, 8)):
        for column, method in enumerate(METHODS):
            path = layout_path(run_dirs[dimension], rows, method, dimension, reducer_seed)
            layout = load_layout(path, rows, dimension)
            record = plot_layout_panel(
                axes[row, column],
                layout,
                background,
                colored,
                colors,
                row=row,
                column=column,
                pca_fit_indices=pca_fit_indices if dimension == 8 else None,
                background_size=0.45,
                colored_size=1.7,
                view_quantile=0.995,
            )
            record.update({"path": str(path), "sha256": sha256_file(path)})
            layout_records[f"d{dimension}"][method] = record

    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=4.0,
            markerfacecolor=label_to_color[label],
            markeredgewidth=0,
            label=label,
        )
        for label in selected_labels
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=6,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.35,
    )
    fig.suptitle(
        "arXiv primary-category structure at target widths 2 and 8\n"
        "Top: native d=2. Bottom: independent unsupervised PCA views of d=8; "
        "do not rank methods by visible PCA separation.",
        fontsize=10.5,
    )
    outputs = save_figure(fig, output_dir / "large_arxiv_category_layouts")
    provenance["arxiv_category_layouts"] = {
        "outputs": outputs,
        "output_sha256": {output: sha256_file(Path(output)) for output in outputs},
        "labels": {
            f"d{dimension}": {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for dimension, path in labels_paths.items()
        },
        "selection_rule": "12 largest primary categories by count; lexical tie-break",
        "balanced_selection_seed": FIGURE_SEED,
        "selected_label_counts": {label: dict(counts)[label] for label in selected_labels},
        "plotted_per_label": {label: int(indices.size) for label, indices in by_label.items()},
        "colored_index_sha256": sha256_indices(np.sort(colored)),
        "colored_draw_order_sha256": sha256_indices(colored),
        "selection_applies_to_target_dimensions": [2, 8],
        "background_rule": "uniform sample excluding the 12 highlighted categories",
        "background_seed": FIGURE_SEED + 1,
        "background_rows": int(background.size),
        "background_index_sha256": sha256_indices(background),
        "pca_fit_rule": "uniform 100,000-row sample, identical for all d=8 methods",
        "pca_fit_seed": FIGURE_SEED + 2,
        "pca_fit_index_sha256": sha256_indices(pca_fit_indices),
        "layout_records": layout_records,
        "interpretation_limit": (
            "the d=8 row uses an independent unsupervised PCA fit for each method; "
            "orientation and visible 2-D separation are not cross-method scores"
        ),
    }


def choose_wikipedia_articles(
    labels: np.ndarray,
    minimum_count: int = 40,
    maximum_count: int = 100,
    number: int = 8,
) -> tuple[list[str], dict[str, int]]:
    """Choose articles by a fixed hash, independent of every layout."""
    counts = dict(sorted_label_counts(labels))
    eligible = [
        label
        for label, count in counts.items()
        if label and minimum_count <= count <= maximum_count
    ]
    scored = sorted(
        eligible,
        key=lambda label: hashlib.sha256(f"{WIKIPEDIA_HASH_SALT}\0{label}".encode()).digest(),
    )
    if len(scored) < number:
        raise ValueError(f"only {len(scored)} Wikipedia articles meet the count filter")
    chosen = scored[:number]
    return chosen, {label: counts[label] for label in chosen}


def article_title(url: str) -> str:
    """Extract a readable title from a Wikipedia URL."""
    return unquote(url.rsplit("/", 1)[-1]).replace("_", " ")


def pca_projection(
    layout: np.ndarray, fit_indices: np.ndarray, plot_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a two-dimensional PCA view on fixed rows and project plotted rows."""
    fit = np.asarray(layout[fit_indices], dtype=np.float64)
    center = fit.mean(axis=0)
    covariance = np.cov(fit - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:2]].copy()
    # Eigenvectors have arbitrary signs.  Canonicalize each component so equivalent
    # LAPACK implementations do not mirror an otherwise identical published panel.
    for column in range(components.shape[1]):
        pivot = int(np.argmax(np.abs(components[:, column])))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    projected = (np.asarray(layout[plot_indices], dtype=np.float64) - center) @ components
    explained = eigenvalues[order[:2]] / np.maximum(eigenvalues.sum(), np.finfo(float).tiny)
    return projected, components, explained


def plot_wikipedia_articles(
    run_dirs: dict[int, Path],
    output_dir: Path,
    reducer_seed: int,
    provenance: dict,
) -> None:
    """Plot Wikipedia article groups at native widths two and eight."""
    rows = 1_000_000
    labels_paths = {
        dimension: run_dir / f"n{rows}" / "labels.npy" for dimension, run_dir in run_dirs.items()
    }
    labels = verify_matching_labels(labels_paths, rows)
    label_strings = labels.astype(str)
    selected_labels, selected_counts = choose_wikipedia_articles(labels)
    selected_mask = np.isin(label_strings, selected_labels)
    selected_indices = np.flatnonzero(selected_mask)

    rng = np.random.default_rng(FIGURE_SEED + 2)
    fit_indices = np.sort(rng.choice(rows, size=100_000, replace=False))
    background_candidates = np.flatnonzero(~selected_mask)
    background = np.sort(rng.choice(background_candidates, size=30_000, replace=False))

    palette = list(plt.get_cmap("Dark2").colors[:8])
    label_to_color = dict(zip(selected_labels, palette))
    colors = [label_to_color[label] for label in label_strings[selected_indices]]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.75), constrained_layout=True)
    layout_records: dict[str, dict] = {"d2": {}, "d8": {}}
    for row, dimension in enumerate((2, 8)):
        for column, method in enumerate(METHODS):
            path = layout_path(run_dirs[dimension], rows, method, dimension, reducer_seed)
            layout = load_layout(path, rows, dimension)
            record = plot_layout_panel(
                axes[row, column],
                layout,
                background,
                selected_indices,
                colors,
                row=row,
                column=column,
                pca_fit_indices=fit_indices if dimension == 8 else None,
                background_size=0.55,
                colored_size=6.0,
                view_quantile=0.997,
            )
            record.update({"path": str(path), "sha256": sha256_file(path)})
            layout_records[f"d{dimension}"][method] = record

    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=4.0,
            markerfacecolor=label_to_color[label],
            markeredgewidth=0,
            label=f"{article_title(label)} ({selected_counts[label]})",
        )
        for label in selected_labels
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.4,
    )
    fig.suptitle(
        "Wikipedia same-article groups at target widths 2 and 8\n"
        "Top: native d=2. Bottom: independent unsupervised PCA views of d=8; "
        "do not rank methods by visible PCA separation.",
        fontsize=10.5,
    )
    outputs = save_figure(fig, output_dir / "large_wikipedia_article_layouts")
    provenance["wikipedia_article_layouts"] = {
        "outputs": outputs,
        "output_sha256": {output: sha256_file(Path(output)) for output in outputs},
        "labels": {
            f"d{dimension}": {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for dimension, path in labels_paths.items()
        },
        "selection_rule": (
            "eight lowest SHA-256 scores among URLs with 40--100 rows; " "hash salt recorded below"
        ),
        "hash_salt": WIKIPEDIA_HASH_SALT,
        "selected_articles": [
            {"url": label, "title": article_title(label), "rows": selected_counts[label]}
            for label in selected_labels
        ],
        "selected_index_sha256": sha256_indices(selected_indices),
        "selection_applies_to_target_dimensions": [2, 8],
        "pca_fit_rule": "uniform 100,000-row sample, identical indices for all methods",
        "pca_and_background_rng_seed": FIGURE_SEED + 2,
        "pca_fit_index_sha256": sha256_indices(fit_indices),
        "background_rule": "uniform 30,000-row sample excluding highlighted articles",
        "background_index_sha256": sha256_indices(background),
        "layout_records": layout_records,
        "interpretation_limit": (
            "the d=8 row uses an independent unsupervised PCA fit for each method; "
            "orientation and visible 2-D separation are not cross-method scores; "
            "native-layout article retrieval remains the quantitative test"
        ),
    }


def successful_rows(results_path: Path, expected_dimension: int) -> list[dict]:
    """Load and validate a complete three-method by three-seed result set."""
    payload = json.loads(results_path.read_text())
    rows = [
        row for row in payload["rows"] if row.get("status") == "ok" and row.get("method") in METHODS
    ]
    if len(rows) != 9:
        raise ValueError(f"expected nine successful rows in {results_path}, found {len(rows)}")
    observed = {(row["method"], int(row["seed"])) for row in rows}
    expected = {(method, seed) for method in METHODS for seed in (0, 1, 2)}
    if observed != expected:
        raise ValueError(f"incomplete method/seed design in {results_path}: {sorted(observed)}")
    dimensions = {int(row["target_dimension"]) for row in rows}
    if dimensions != {expected_dimension}:
        raise ValueError(
            f"unexpected target dimensions in {results_path}: {sorted(dimensions)}; "
            f"expected {expected_dimension}"
        )
    return rows


def plot_label_metrics(
    arxiv_dirs: dict[int, Path],
    wikipedia_dirs: dict[int, Path],
    output_dir: Path,
    provenance: dict,
) -> None:
    """Plot seed-level label metrics for both corpora and target widths."""
    result_paths = {
        "arxiv": {dimension: run_dir / "results.json" for dimension, run_dir in arxiv_dirs.items()},
        "wikipedia": {
            dimension: run_dir / "results.json" for dimension, run_dir in wikipedia_dirs.items()
        },
    }
    rows_by_corpus_dimension = {
        corpus: {dimension: successful_rows(path, dimension) for dimension, path in paths.items()}
        for corpus, paths in result_paths.items()
    }
    panels = [
        (
            "arxiv",
            "category_knn_accuracy",
            "arXiv category accuracy",
            "accuracy",
        ),
        (
            "wikipedia",
            "group_neighbor_hit_rate",
            "Wikipedia article hit rate",
            "hit rate",
        ),
        (
            "wikipedia",
            "group_neighbor_purity",
            "Wikipedia article purity",
            "purity",
        ),
        (
            "wikipedia",
            "group_neighbor_recall",
            "Wikipedia truncated recall@16",
            "truncated recall@16",
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.15), constrained_layout=True)
    panel_records = {}
    x = np.arange(len(METHODS), dtype=float)
    seed_jitter = np.array([-0.038, 0.0, 0.038])
    dimension_offsets = {2: -0.13, 8: 0.13}
    dimension_markers = {2: "o", 8: "^"}
    dimension_linestyles = {2: "-", 8: "--"}
    for ax, (corpus, metric, title, ylabel) in zip(axes, panels):
        metric_record: dict[str, dict] = {"d2": {}, "d8": {}}
        maximum = 0.0
        for position, method in enumerate(METHODS):
            color = METHOD_COLORS[method]
            means = []
            for dimension in (2, 8):
                method_rows = sorted(
                    (
                        row
                        for row in rows_by_corpus_dimension[corpus][dimension]
                        if row["method"] == method
                    ),
                    key=lambda row: row["seed"],
                )
                values = np.array([row["metrics"][metric] for row in method_rows], dtype=float)
                seeds = [int(row["seed"]) for row in method_rows]
                maximum = max(maximum, float(values.max()))
                center = position + dimension_offsets[dimension]
                ax.scatter(
                    center + seed_jitter,
                    values,
                    marker=dimension_markers[dimension],
                    s=21,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                mean = float(values.mean())
                means.append((center, mean))
                ax.hlines(
                    mean,
                    center - 0.09,
                    center + 0.09,
                    color=color,
                    linestyle=dimension_linestyles[dimension],
                    lw=2.0,
                    zorder=4,
                )
                metric_record[f"d{dimension}"][method] = {
                    "seeds": seeds,
                    "values": values.tolist(),
                    "mean": mean,
                    "sample_sd": float(values.std(ddof=1)),
                }
            ax.plot(
                [item[0] for item in means],
                [item[1] for item in means],
                color=color,
                alpha=0.28,
                lw=0.8,
                zorder=1,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [METHOD_NAMES[method] for method in METHODS], rotation=25, ha="right")
        ax.set_xlim(-0.45, len(METHODS) - 0.55)
        ax.set_ylim(0, min(1.0, maximum * 1.16))
        ax.grid(axis="y", color="#dddddd", lw=0.55)
        ax.set_axisbelow(True)
        panel_records[metric] = metric_record
    width_handles = [
        Line2D(
            [],
            [],
            linestyle=dimension_linestyles[dimension],
            marker=dimension_markers[dimension],
            color="#333333",
            markersize=4.3,
            label=f"target d={dimension}",
        )
        for dimension in (2, 8)
    ]
    fig.legend(
        handles=width_handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.6,
    )
    fig.suptitle("Label/group organization at target widths 2 and 8", fontsize=11)
    outputs = save_figure(fig, output_dir / "large_label_preservation_metrics")
    provenance["label_preservation_metrics"] = {
        "outputs": outputs,
        "output_sha256": {output: sha256_file(Path(output)) for output in outputs},
        "results": {
            corpus: {
                f"d{dimension}": {"path": str(path), "sha256": sha256_file(path)}
                for dimension, path in paths.items()
            }
            for corpus, paths in result_paths.items()
        },
        "panels": panel_records,
        "points": (
            "one point per reducer/training seed; circles are d=2, triangles are d=8, "
            "and the corresponding horizontal segments are means"
        ),
        "wikipedia_recall_definition": (
            "mean same-article hits divided by min(16, article paragraph count minus one)"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="directory containing large-{arxiv,wikipedia}-d{2,8}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers/figures"),
        help="destination for PDF/PNG figures and provenance JSON",
    )
    parser.add_argument(
        "--reducer-seed",
        type=int,
        default=0,
        help="saved reducer seed used for layout illustrations (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot_style()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    arxiv_dirs = {dimension: artifacts_dir / f"large-arxiv-d{dimension}" for dimension in (2, 8)}
    wikipedia_dirs = {
        dimension: artifacts_dir / f"large-wikipedia-d{dimension}" for dimension in (2, 8)
    }
    # Fail before replacing any published figure if one arm of the 2x2 design is still
    # incomplete.  The reducer run writes results.json only after all requested workers.
    for run_dirs in (arxiv_dirs, wikipedia_dirs):
        for dimension, run_dir in run_dirs.items():
            successful_rows(run_dir / "results.json", dimension)
    provenance = {
        "schema_version": 2,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "figure_seed": FIGURE_SEED,
        "reducer_seed": args.reducer_seed,
        "versions": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    plot_arxiv_categories(arxiv_dirs, output_dir, args.reducer_seed, provenance)
    plot_wikipedia_articles(wikipedia_dirs, output_dir, args.reducer_seed, provenance)
    plot_label_metrics(arxiv_dirs, wikipedia_dirs, output_dir, provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "large_embedding_figure_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {provenance_path}")
    for section in provenance.values():
        if isinstance(section, dict) and "outputs" in section:
            for output in section["outputs"]:
                print(f"wrote {output}")


if __name__ == "__main__":
    main()
