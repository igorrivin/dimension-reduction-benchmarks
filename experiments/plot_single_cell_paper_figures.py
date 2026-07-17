#!/usr/bin/env python3
"""Create the paper figures for the Tabula Sapiens benchmark and DiRe audit.

The first figure displays the seed-0 layouts for the eight most frequent cell
types.  Native two-dimensional layouts are shown directly; eight-dimensional
layouts are shown only through an independently fitted, unsupervised PCA view.
The second figure reports exact-neighbor misses and paired downstream changes
for the controlled DiRe NN-descent working-degree audit.

Both figures are written as PDF and high-resolution PNG files.  A JSON record
captures source hashes, displayed labels, panel limits, PCA projections, and
every plotted audit value.

Example
-------
::

    conda run -n rapids-26.06 python experiments/plot_single_cell_paper_figures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Keep Matplotlib caches out of HOME in restricted documentation builders.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dimension-reduction-benchmarks-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dimension-reduction-benchmarks-cache")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METHODS = ("pca", "autoencoder", "dire", "umap")
METHOD_NAMES = {
    "pca": "PCA",
    "autoencoder": "Autoencoder",
    "dire": "DiRe",
    "umap": "UMAP",
}
TARGET_DIMENSIONS = (2, 8)
CELL_TYPE_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#882255",
    "#44AA99",
)
CELL_TYPE_SHORT_NAMES = {
    "CD4-positive, alpha-beta T cell": "CD4 T cell",
    "CD8-positive, alpha-beta T cell": "CD8 T cell",
    "stromal cell of ovary": "ovarian stromal cell",
}
AUDIT_CONFIGS = ("minimum", "umap-matched")
AUDIT_CONFIG_LABELS = {
    "minimum": "31/62",
    "umap-matched": "64/128",
}
AUDIT_CONFIG_COLORS = {"minimum": "#0072B2", "umap-matched": "#D55E00"}
DIMENSION_COLORS = {2: "#0072B2", 8: "#D55E00"}
GEOMETRY_METRICS = (
    ("Trust", ("sample_conditional_trustworthiness",), True),
    ("Continuity", ("sample_conditional_continuity",), True),
    ("Overlap", ("sample_conditional_neighbor_overlap",), True),
    (r"$\rho_S$", ("sampled_pair_distance_spearman",), True),
    ("Stress", ("sampled_pair_normalized_stress",), False),
)
BIOLOGY_METRICS = (
    ("Macro-F1", ("label_metrics", "donor_heldout_macro_f1"), True),
    ("Balanced accuracy", ("label_metrics", "donor_heldout_balanced_accuracy"), True),
    ("Cell-type ASW", ("label_metrics", "cell_type_average_silhouette_width"), True),
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def configure_plot_style() -> None:
    """Set a compact journal-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 8.3,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, subject: str) -> list[Path]:
    """Write PDF and PNG versions of a figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    common = {"bbox_inches": "tight", "facecolor": "white"}
    fig.savefig(
        outputs[0],
        metadata={
            "Title": stem.name.replace("_", " "),
            "Author": "dimension-reduction-benchmarks",
            "Subject": subject,
        },
        **common,
    )
    fig.savefig(outputs[1], **common)
    plt.close(fig)
    return outputs


def sorted_label_counts(labels: np.ndarray) -> list[tuple[str, int]]:
    """Order labels by decreasing count and then lexicographically."""
    unique, counts = np.unique(labels.astype(str), return_counts=True)
    return sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))


def robust_equal_limits(points: np.ndarray, quantile: float = 0.995) -> tuple[float, float, float]:
    """Return median-centered isotropic limits robust to a few extreme points."""
    center = np.median(points, axis=0)
    deviations = np.abs(points - center)
    radius = float(np.quantile(deviations, quantile))
    if not np.isfinite(radius) or radius <= 0:
        radius = float(np.max(deviations))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    return float(center[0]), float(center[1]), 1.06 * radius


def pca_view(layout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one layout to two dimensions by full-cohort covariance PCA."""
    values = np.asarray(layout, dtype=np.float64)
    center = values.mean(axis=0)
    centered = values - center
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order[:2]].T
    projected = centered @ components.T
    total = float(eigenvalues.sum())
    explained = eigenvalues[:2] / total if total > 0 else np.zeros(2)
    return projected, components, explained


def index_baseline_rows(payload: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Validate and index the 24 baseline benchmark rows."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("the baseline result must contain 24 runs")
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"non-successful baseline row: {row.get('method')} {row.get('status')}")
        key = (str(row["method"]), int(row["target_dimension"]), int(row["seed"]))
        if key in index:
            raise ValueError(f"duplicate baseline row {key}")
        index[key] = row
    expected = {(method, dimension, seed) for method in METHODS for dimension in TARGET_DIMENSIONS for seed in range(3)}
    if set(index) != expected:
        raise ValueError("the baseline result does not contain the complete method/dimension/seed grid")
    return index


def index_audit_rows(payload: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Validate and index the 12 paired audit layout rows."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("the audit result must contain 12 layout runs")
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"non-successful audit row: {row.get('variant')} {row.get('status')}")
        key = (str(row["variant"]), int(row["target_dimension"]), int(row["seed"]))
        if key in index:
            raise ValueError(f"duplicate audit layout row {key}")
        index[key] = row
    expected = {(variant, dimension, seed) for variant in AUDIT_CONFIGS for dimension in TARGET_DIMENSIONS for seed in range(3)}
    if set(index) != expected:
        raise ValueError("the audit result does not contain the complete variant/dimension/seed grid")
    return index


def nested_metric(row: dict[str, Any], path: tuple[str, ...]) -> float:
    """Read one nested metric from an audit row."""
    value: Any = row["metrics"]
    for key in path:
        value = value[key]
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric at {path}")
    return value


def draw_cell_layout_figure(
    artifact_root: Path,
    metadata_path: Path,
    baseline: dict[tuple[str, int, int], dict[str, Any]],
    output_dir: Path,
    top_count: int,
) -> tuple[list[Path], dict[str, Any]]:
    """Draw native d=2 layouts and descriptive PCA views of d=8 layouts."""
    metadata = pd.read_parquet(metadata_path, columns=["cell_type"])
    labels = metadata["cell_type"].astype("string").fillna("").to_numpy(dtype=str)
    counts = sorted_label_counts(labels)
    if top_count < 1 or top_count > len(CELL_TYPE_COLORS):
        raise ValueError(f"top-count must lie between 1 and {len(CELL_TYPE_COLORS)}")
    selected = counts[:top_count]
    selected_labels = [label for label, _ in selected]
    color_map = dict(zip(selected_labels, CELL_TYPE_COLORS[:top_count], strict=True))
    selected_mask = np.isin(labels, selected_labels)

    fig, axes = plt.subplots(2, len(METHODS), figsize=(7.25, 4.15), squeeze=False)
    panel_records: dict[str, Any] = {}
    layout_records: dict[str, Any] = {}
    benchmark_dir = artifact_root / "benchmark-pca50"
    for row_number, dimension in enumerate(TARGET_DIMENSIONS):
        for column, method in enumerate(METHODS):
            ax = axes[row_number, column]
            row = baseline[(method, dimension, 0)]
            layout_path = benchmark_dir / f"{method}-d{dimension}-s0-layout.npy"
            layout = np.load(layout_path, mmap_mode="r", allow_pickle=False)
            if layout.shape != (labels.size, dimension) or layout.dtype != np.float32:
                raise ValueError(f"unexpected layout contract for {layout_path}: {layout.shape}, {layout.dtype}")
            if row.get("layout_sha256") != sha256_file(layout_path):
                raise ValueError(f"layout hash mismatch for {layout_path}")

            components = None
            explained = None
            if dimension == 2:
                points = np.asarray(layout)
            else:
                points, components, explained = pca_view(layout)
            center_x, center_y, radius = robust_equal_limits(points)
            ax.scatter(
                points[~selected_mask, 0],
                points[~selected_mask, 1],
                s=0.18,
                c="#A7A7A7",
                alpha=0.11,
                linewidths=0,
                rasterized=True,
            )
            for label in selected_labels:
                mask = labels == label
                ax.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    s=0.32,
                    c=color_map[label],
                    alpha=0.58,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_xlim(center_x - radius, center_x + radius)
            ax.set_ylim(center_y - radius, center_y + radius)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_number == 0:
                ax.set_title(METHOD_NAMES[method], pad=2.5)
            if column == 0:
                row_label = "native $d=2$" if dimension == 2 else "PCA view of $d=8$"
                ax.set_ylabel(row_label, labelpad=6.0)
            if explained is not None:
                ax.text(
                    0.02,
                    0.98,
                    f"PCA variance: {100 * explained.sum():.1f}%",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.5,
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
                )
            panel_key = f"{method}-d{dimension}-s0"
            panel_records[panel_key] = {
                "view_center": [center_x, center_y],
                "view_radius": radius,
                "view_quantile": 0.995,
                "pca_components": components.tolist() if components is not None else None,
                "pca_explained_variance_ratio": explained.tolist() if explained is not None else None,
            }
            layout_records[panel_key] = {
                "path": str(layout_path),
                "sha256": row["layout_sha256"],
                "shape": list(layout.shape),
                "dtype": str(layout.dtype),
            }

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[label],
            markeredgecolor="none",
            markersize=4.2,
            label=CELL_TYPE_SHORT_NAMES.get(label, label),
        )
        for label in selected_labels
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#A7A7A7",
            markeredgecolor="none",
            markersize=4.2,
            label="other cell types",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.005),
        ncol=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.145, hspace=0.08, wspace=0.035)
    outputs = save_figure(
        fig,
        output_dir / "tabula_sapiens_cell_type_layouts",
        "Tabula Sapiens cell-type structure across dimension reducers",
    )
    record = {
        "rows": int(labels.size),
        "cell_type_count": len(counts),
        "displayed_seed": 0,
        "displayed_cell_types": [
            {
                "cell_type": label,
                "legend_label": CELL_TYPE_SHORT_NAMES.get(label, label),
                "count": count,
                "color": color_map[label],
            }
            for label, count in selected
        ],
        "background_rows": int(np.count_nonzero(~selected_mask)),
        "layouts": layout_records,
        "panels": panel_records,
        "interpretation_guard": (
            "d=8 panels are per-layout unsupervised PCA views and are not scored cross-method projections"
        ),
    }
    return outputs, record


def graph_recall_record(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return plotted recall summaries grouped by working-degree configuration."""
    record: dict[str, Any] = {}
    for variant in AUDIT_CONFIGS:
        selected = sorted(
            (graph for graph in graphs if graph["configuration"]["name"] == variant),
            key=lambda graph: int(graph["seed"]),
        )
        if len(selected) != 3 or [int(graph["seed"]) for graph in selected] != [0, 1, 2]:
            raise ValueError(f"audit graph grid is incomplete for {variant}")
        recalls: dict[str, Any] = {}
        for k in (1, 10, 30):
            values = np.asarray(
                [graph["recall"][f"mean_recall_at_{k}"] for graph in selected], dtype=float
            )
            exact_rows = np.asarray(
                [graph["recall"][f"exact_match_fraction_at_{k}"] for graph in selected],
                dtype=float,
            )
            recalls[str(k)] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "miss_percent_values": (100.0 * (1.0 - values)).tolist(),
                "exact_row_fraction_values": exact_rows.tolist(),
                "exact_row_fraction_mean": float(exact_rows.mean()),
                "exact_row_fraction_sample_sd": float(exact_rows.std(ddof=1)),
            }
        graph_seconds = np.asarray([graph["details"]["graph_seconds"] for graph in selected])
        config = selected[0]["configuration"]
        record[variant] = {
            "label": AUDIT_CONFIG_LABELS[variant],
            "graph_degree": int(config["graph_degree"]),
            "intermediate_graph_degree": int(config["intermediate_graph_degree"]),
            "recall": recalls,
            "graph_seconds_values": graph_seconds.tolist(),
            "graph_seconds_mean": float(graph_seconds.mean()),
            "graph_seconds_sample_sd": float(graph_seconds.std(ddof=1)),
        }
    return record


def paired_change_record(
    audit_rows: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Compute matched-minus-default changes, oriented so positive is improvement."""
    record: dict[str, Any] = {}
    for dimension in TARGET_DIMENSIONS:
        dimension_record: dict[str, Any] = {}
        for label, path, higher_is_better in (*GEOMETRY_METRICS, *BIOLOGY_METRICS):
            changes = []
            for seed in range(3):
                minimum = nested_metric(audit_rows[("minimum", dimension, seed)], path)
                matched = nested_metric(audit_rows[("umap-matched", dimension, seed)], path)
                change = matched - minimum if higher_is_better else minimum - matched
                changes.append(change)
            values = np.asarray(changes, dtype=float)
            dimension_record[label] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "orientation": "positive is improvement",
            }
        record[str(dimension)] = dimension_record
    return record


def draw_change_panel(
    ax: plt.Axes,
    metrics: tuple[tuple[str, tuple[str, ...], bool], ...],
    changes: dict[str, Any],
) -> None:
    """Draw all paired seed changes and their mean for one metric family."""
    y_positions = np.arange(len(metrics), dtype=float)
    seed_jitter = np.asarray((-0.045, 0.0, 0.045))
    for dimension, offset in ((2, -0.16), (8, 0.16)):
        color = DIMENSION_COLORS[dimension]
        for position, (label, _, _) in zip(y_positions, metrics, strict=True):
            values = 1000.0 * np.asarray(changes[str(dimension)][label]["values"])
            ax.scatter(
                values,
                position + offset + seed_jitter,
                s=11,
                color=color,
                alpha=0.34,
                linewidths=0,
                zorder=2,
            )
            ax.scatter(
                [values.mean()],
                [position + offset],
                s=23,
                marker="D",
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    ax.axvline(0.0, color="#555555", linewidth=0.75, zorder=1)
    ax.set_yticks(y_positions, [item[0] for item in metrics])
    ax.invert_yaxis()
    ax.set_xlabel(r"$10^3\!\times$ paired score change")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def draw_audit_figure(
    payload: dict[str, Any], output_dir: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Draw exact-graph recall and paired downstream audit effects."""
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != 6:
        raise ValueError("the audit result must contain six graph builds")
    for graph in graphs:
        if graph.get("status") != "ok":
            raise ValueError("the audit graph result contains a failed build")
        contract = graph["contract"]
        if any(contract[key] != 0 for key in ("duplicate_rows", "out_of_range_rows", "self_rows")):
            raise ValueError("the audit graph violates the neighbor-index contract")
    indexed_rows = index_audit_rows(payload)
    recall = graph_recall_record(graphs)
    changes = paired_change_record(indexed_rows)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.75),
        gridspec_kw={"width_ratios": (0.93, 1.22, 1.13)},
    )
    x = np.arange(3, dtype=float)
    for variant, offset in (("minimum", -0.035), ("umap-matched", 0.035)):
        color = AUDIT_CONFIG_COLORS[variant]
        means = []
        for position, k in enumerate((1, 10, 30)):
            values = np.asarray(recall[variant]["recall"][str(k)]["miss_percent_values"])
            axes[0].scatter(
                np.full(3, x[position] + offset),
                values,
                s=11,
                color=color,
                alpha=0.34,
                linewidths=0,
                zorder=2,
            )
            means.append(float(values.mean()))
        axes[0].plot(
            x + offset,
            means,
            marker="D",
            markersize=4.2,
            linewidth=1.15,
            color=color,
            label=AUDIT_CONFIG_LABELS[variant],
            zorder=3,
        )
    axes[0].set_xticks(x, ("1", "10", "30"))
    axes[0].set_xlabel("Recall width $k$")
    axes[0].set_ylabel(r"Missed exact neighbors (\%)")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("(a) Approximate graph")
    axes[0].grid(axis="y", color="#D8D8D8", linewidth=0.5)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(
        title="Working degrees",
        frameon=False,
        loc="upper right",
        handlelength=1.3,
        borderaxespad=0.2,
    )

    draw_change_panel(axes[1], GEOMETRY_METRICS, changes)
    axes[1].set_title("(b) Geometry")
    draw_change_panel(axes[2], BIOLOGY_METRICS, changes)
    axes[2].set_title("(c) Cell-type signal")
    dimension_handles = [
        Line2D(
            [0],
            [0],
            color=DIMENSION_COLORS[dimension],
            marker="D",
            linestyle="",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.8,
            label=f"$d={dimension}$ mean",
        )
        for dimension in TARGET_DIMENSIONS
    ]
    axes[1].legend(handles=dimension_handles, frameon=False, loc="upper right", borderaxespad=0.2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.22, wspace=0.52)
    outputs = save_figure(
        fig,
        output_dir / "dire_knn_degree_audit",
        "Exact-neighbor and downstream audit of DiRe NN-descent working degrees",
    )
    return outputs, {
        "exact_reference_query_rows": 5000,
        "final_neighbors": 30,
        "graph_builds_per_configuration": 3,
        "recall": recall,
        "paired_changes": changes,
        "change_orientation": "positive values indicate improvement under 64/128 degrees",
        "point_encoding": "small translucent points are seeds 0, 1, and 2; diamonds are means",
        "interpretation_guard": (
            "64/128 uses UMAP-like degree settings inside DiRe's cuVS all-neighbors path; "
            "it is not an extracted UMAP graph"
        ),
    }


def relative_record(paths: list[Path], repo_root: Path) -> list[dict[str, Any]]:
    """Return repository-relative hashes for generated outputs."""
    return [
        {"path": str(path.resolve().relative_to(repo_root)), "sha256": sha256_file(path)}
        for path in paths
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo_root / "artifacts/single-cell/tabula-pilot-100k",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "papers/figures")
    parser.add_argument("--top-cell-types", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate both figures and their machine-readable provenance record."""
    args = parse_args(argv)
    configure_plot_style()
    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    baseline_path = artifact_root / "benchmark-pca50/results.json"
    audit_path = artifact_root / "dire-knn-degree-audit/results.json"
    metadata_path = artifact_root / "metadata.parquet"
    manifest_path = artifact_root / "manifest.json"
    baseline_payload = load_json(baseline_path)
    audit_payload = load_json(audit_path)
    baseline = index_baseline_rows(baseline_payload)

    layout_outputs, layout_record = draw_cell_layout_figure(
        artifact_root,
        metadata_path,
        baseline,
        output_dir,
        args.top_cell_types,
    )
    audit_outputs, audit_record = draw_audit_figure(audit_payload, output_dir)
    provenance_path = output_dir / "single_cell_paper_figure_provenance.json"
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve().relative_to(repo_root)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "sources": {
            str(path.resolve().relative_to(repo_root)): sha256_file(path)
            for path in (baseline_path, audit_path, metadata_path, manifest_path)
        },
        "tabula_sapiens_layout_figure": layout_record,
        "dire_knn_degree_audit_figure": audit_record,
        "outputs": {
            "tabula_sapiens_layouts": relative_record(layout_outputs, repo_root),
            "dire_knn_degree_audit": relative_record(audit_outputs, repo_root),
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    for path in (*layout_outputs, *audit_outputs, provenance_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
