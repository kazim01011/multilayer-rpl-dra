#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.cooja import graphs_from_snapshots
from mlrpl_dra.simulator import LAYER_NAMES


LAYER_LABELS = {
    "routing": "Routing",
    "link_quality": "Link quality",
    "temporal": "Temporal",
    "trust": "Trust/anomaly",
}


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def undirected_edge_mask(adj: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mask = np.asarray(adj > eps, dtype=bool)
    np.fill_diagonal(mask, False)
    mask = np.logical_or(mask, mask.T)
    return np.triu(mask, k=1)


def edge_density(mask: np.ndarray) -> float:
    n = mask.shape[0]
    possible = n * (n - 1) / 2
    return float(mask.sum() / max(possible, 1.0))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def degree_distribution(mask: np.ndarray) -> np.ndarray:
    full = np.logical_or(mask, mask.T)
    degree = full.sum(axis=1).astype(int)
    hist = np.bincount(degree, minlength=mask.shape[0])
    return hist / max(hist.sum(), 1)


def js_dissimilarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    p = degree_distribution(mask_a)
    q = degree_distribution(mask_b)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)), 0.0))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def matrix_from_pair_rows(rows: pd.DataFrame, value_col: str, fill_diagonal: float) -> np.ndarray:
    labels = list(LAYER_NAMES)
    mat = np.full((len(labels), len(labels)), fill_diagonal, dtype=float)
    for _, row in rows.iterrows():
        i = labels.index(row["layer_i"])
        j = labels.index(row["layer_j"])
        mat[i, j] = float(row[value_col])
        mat[j, i] = float(row[value_col])
    return mat


def plot_layer_structure(layer_summary: pd.DataFrame, overlap: pd.DataFrame, dissimilarity: pd.DataFrame, out_dir: Path) -> None:
    labels = [LAYER_LABELS[layer] for layer in LAYER_NAMES]
    x = np.arange(len(labels))

    overlap_mat = matrix_from_pair_rows(overlap, "mean_jaccard", fill_diagonal=1.0)
    dissim_mat = matrix_from_pair_rows(dissimilarity, "mean_js_dissimilarity", fill_diagonal=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})

    ax = axes[0]
    ordered = layer_summary.set_index("layer").loc[list(LAYER_NAMES)]
    ax.bar(
        x - 0.18,
        ordered["mean_density"].to_numpy(),
        0.36,
        yerr=ordered["std_density"].fillna(0.0).to_numpy(),
        capsize=3,
        color="#4C78A8",
        edgecolor="#263238",
        linewidth=0.5,
        label="Density",
    )
    ax.bar(
        x + 0.18,
        ordered["mean_edge_share"].to_numpy(),
        0.36,
        yerr=ordered["std_edge_share"].fillna(0.0).to_numpy(),
        capsize=3,
        color="#F58518",
        edgecolor="#263238",
        linewidth=0.5,
        label="Edge share",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean value")
    ax.set_title("(a) Layer density and edge share", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    for ax, mat, title, cmap, vmax in [
        (axes[1], overlap_mat, "(b) Mean edge-overlap Jaccard", "Blues", 1.0),
        (axes[2], dissim_mat, "(c) Mean degree-profile dissimilarity", "Oranges", max(0.01, float(dissim_mat.max()))),
    ]:
        image = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(x)
        ax.set_yticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title, loc="left", fontweight="bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="#263238", fontsize=8.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig06_layer_structure")


def plot_aggregation_summary(aggregation: pd.DataFrame, out_dir: Path) -> None:
    means = aggregation.mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = ["Mean layer edges", "Aggregated edges", "New edges vs routing"]
    values = [
        means["mean_layer_edges"],
        means["aggregated_edges"],
        means["aggregated_edges_not_in_routing"],
    ]
    ax.bar(labels, values, color=["#4C78A8", "#9467BD", "#E45756"], edgecolor="#263238", linewidth=0.6)
    ax.set_ylabel("Undirected edges per snapshot")
    ax.set_title("Aggregation effect on graph support", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, out_dir, "fig07_aggregation_support")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze structural properties of the multilayer RPL graphs.")
    parser.add_argument(
        "--snapshots",
        default=str(PROJECT_ROOT / "data" / "cooja_clean_5seed" / "node_snapshots.csv"),
        help="Parsed node_snapshots.csv file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "cooja_clean_5seed_layer_analysis"),
        help="Output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    snapshots = pd.read_csv(args.snapshots)
    graphs = graphs_from_snapshots(snapshots)
    if not graphs:
        raise SystemExit("No graphs were built from the snapshot table.")

    layer_rows = []
    pair_rows = []
    aggregation_rows = []
    for graph in graphs:
        masks = {layer: undirected_edge_mask(graph.layers[layer]) for layer in LAYER_NAMES}
        edge_counts = {layer: int(mask.sum()) for layer, mask in masks.items()}
        total_edges = max(sum(edge_counts.values()), 1)
        for layer in LAYER_NAMES:
            layer_rows.append(
                {
                    "graph_id": graph.graph_id,
                    "attack_ratio": graph.ratio,
                    "layer": layer,
                    "edges": edge_counts[layer],
                    "density": edge_density(masks[layer]),
                    "edge_share": edge_counts[layer] / total_edges,
                }
            )

        for layer_i, layer_j in itertools.combinations(LAYER_NAMES, 2):
            pair_rows.append(
                {
                    "graph_id": graph.graph_id,
                    "attack_ratio": graph.ratio,
                    "layer_i": layer_i,
                    "layer_j": layer_j,
                    "jaccard": jaccard(masks[layer_i], masks[layer_j]),
                    "js_dissimilarity": js_dissimilarity(masks[layer_i], masks[layer_j]),
                }
            )

        aggregate = np.zeros_like(next(iter(masks.values())), dtype=bool)
        for mask in masks.values():
            aggregate = np.logical_or(aggregate, mask)
        routing = masks["routing"]
        aggregation_rows.append(
            {
                "graph_id": graph.graph_id,
                "attack_ratio": graph.ratio,
                "mean_layer_edges": float(np.mean(list(edge_counts.values()))),
                "aggregated_edges": int(aggregate.sum()),
                "aggregation_inflation": int(aggregate.sum()) / max(float(np.mean(list(edge_counts.values()))), 1.0),
                "aggregated_edges_not_in_routing": int(np.logical_and(aggregate, np.logical_not(routing)).sum()),
                "non_routing_share_in_aggregate": int(np.logical_and(aggregate, np.logical_not(routing)).sum()) / max(int(aggregate.sum()), 1),
            }
        )

    layer_detail = pd.DataFrame(layer_rows)
    pair_detail = pd.DataFrame(pair_rows)
    aggregation = pd.DataFrame(aggregation_rows)

    layer_summary = (
        layer_detail.groupby("layer")
        .agg(
            mean_edges=("edges", "mean"),
            std_edges=("edges", "std"),
            mean_density=("density", "mean"),
            std_density=("density", "std"),
            mean_edge_share=("edge_share", "mean"),
            std_edge_share=("edge_share", "std"),
        )
        .reset_index()
    )
    overlap = (
        pair_detail.groupby(["layer_i", "layer_j"])
        .agg(mean_jaccard=("jaccard", "mean"), std_jaccard=("jaccard", "std"))
        .reset_index()
    )
    dissimilarity = (
        pair_detail.groupby(["layer_i", "layer_j"])
        .agg(
            mean_js_dissimilarity=("js_dissimilarity", "mean"),
            std_js_dissimilarity=("js_dissimilarity", "std"),
        )
        .reset_index()
    )

    layer_detail.to_csv(output_dir / "layer_detail.csv", index=False)
    pair_detail.to_csv(output_dir / "layer_pair_detail.csv", index=False)
    aggregation.to_csv(output_dir / "aggregation_detail.csv", index=False)
    layer_summary.to_csv(output_dir / "layer_summary.csv", index=False)
    overlap.to_csv(output_dir / "layer_overlap.csv", index=False)
    dissimilarity.to_csv(output_dir / "layer_dissimilarity.csv", index=False)
    aggregation.describe().to_csv(output_dir / "aggregation_summary.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    plot_layer_structure(layer_summary, overlap, dissimilarity, figure_dir)
    plot_aggregation_summary(aggregation, figure_dir)

    print(layer_summary.to_string(index=False))
    print("\nAggregation summary:")
    print(aggregation[["mean_layer_edges", "aggregated_edges", "aggregation_inflation", "non_routing_share_in_aggregate"]].mean().to_string())
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
