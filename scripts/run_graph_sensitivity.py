#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.config import ExperimentConfig, SimulationConfig, SplitConfig, TrainingConfig
from mlrpl_dra.cooja import graphs_from_snapshots
from mlrpl_dra.experiment import evaluate_model


MODEL_LABELS = {
    "agg_gcn": "Agg-GCN",
    "ml_gcn": "ML-GCN",
    "attn_ml_gcn": "Attn-ML-GCN",
}


def _seed_from_graph_id(graph_id: str) -> int | None:
    match = re.search(r"_seed(\d+)", graph_id)
    return int(match.group(1)) if match else None


def split_graphs_by_seed(graphs, train_seeds: set[int], validation_seeds: set[int], test_seeds: set[int]):
    train_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in train_seeds]
    validation_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in validation_seeds]
    test_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in test_seeds]
    return train_graphs, validation_graphs, test_graphs


def apply_propagation_steps(graphs, steps: int):
    if steps == 1:
        return graphs
    out = []
    for graph in graphs:
        stepped_layers = {
            layer: np.linalg.matrix_power(adj, steps)
            for layer, adj in graph.layers.items()
        }
        out.append(replace(graph, layers=stepped_layers))
    return out


def save_sensitivity_figure(metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [model for model in ["agg_gcn", "ml_gcn", "attn_ml_gcn"] if model in set(metrics["base_model"])]
    steps = sorted(metrics["propagation_steps"].unique())
    x = np.arange(len(steps))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharex=True)
    for ax, metric, title in [
        (axes[0], "balanced_accuracy", "Balanced accuracy"),
        (axes[1], "f1", "F1 score"),
    ]:
        for idx, model in enumerate(models):
            part = metrics[metrics["base_model"] == model].sort_values("propagation_steps")
            ax.bar(
                x + (idx - (len(models) - 1) / 2) * width,
                part[metric].to_numpy(),
                width,
                color=["#9467BD", "#F58518", "#E45756"][idx],
                edgecolor="#263238",
                linewidth=0.5,
                label=MODEL_LABELS.get(model, model),
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={step}" for step in steps])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig08_propagation_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-step graph propagation sensitivity on Cooja snapshots.")
    parser.add_argument(
        "--snapshots",
        default=str(PROJECT_ROOT / "data" / "cooja_clean_5seed" / "node_snapshots.csv"),
        help="Parsed node_snapshots.csv file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "cooja_clean_5seed_graph_sensitivity"),
        help="Output directory.",
    )
    parser.add_argument("--models", nargs="+", default=["agg_gcn", "ml_gcn", "attn_ml_gcn"])
    parser.add_argument("--steps", default="1,2", help="Comma-separated propagation step values.")
    parser.add_argument("--train-seeds", default="1,2,3")
    parser.add_argument("--validation-seeds", default="4")
    parser.add_argument("--test-seeds", default="5")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--positive-class-weight", type=float, default=2.0)
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1", "precision", "recall", "accuracy"],
        default="f1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshots = pd.read_csv(args.snapshots)
    graphs = graphs_from_snapshots(snapshots)
    train_seeds = {int(seed) for seed in args.train_seeds.split(",") if seed}
    validation_seeds = {int(seed) for seed in args.validation_seeds.split(",") if seed}
    test_seeds = {int(seed) for seed in args.test_seeds.split(",") if seed}

    rows = []
    for steps in [int(item) for item in args.steps.split(",") if item]:
        stepped_graphs = apply_propagation_steps(graphs, steps)
        train_graphs, validation_graphs, test_graphs = split_graphs_by_seed(
            stepped_graphs, train_seeds, validation_seeds, test_seeds
        )
        if not train_graphs or not validation_graphs or not test_graphs:
            raise SystemExit("Seed split produced an empty train, validation, or test set.")

        cfg = ExperimentConfig(
            run_name="cooja_clean_5seed_graph_sensitivity",
            seed=args.seed,
            seeds=[args.seed],
            simulation=SimulationConfig(
                num_nodes=int(snapshots["node"].nunique()),
                root_id=0,
                malicious_ratios=sorted(snapshots["attack_ratio"].dropna().unique().tolist()),
                num_graphs_per_ratio=0,
                area_size=0.0,
                transmission_range=100.0,
                simulation_time_s=int(snapshots["bucket_end_s"].max()),
                rank_drop_min=0,
                rank_drop_max=0,
            ),
            split=SplitConfig(train=0.60, validation=0.20, test=0.20),
            training=TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                batch_size=0,
                patience=50,
            ),
            models=args.models,
        )
        for model_name in args.models:
            metrics = evaluate_model(
                model_name,
                cfg,
                train_graphs,
                validation_graphs,
                test_graphs,
                args.seed,
                threshold_metric=args.threshold_metric,
                positive_class_weight=args.positive_class_weight,
            )
            metrics["base_model"] = model_name
            metrics["model"] = f"{model_name}_k{steps}"
            metrics["propagation_steps"] = steps
            metrics["num_train_graphs"] = len(train_graphs)
            metrics["num_validation_graphs"] = len(validation_graphs)
            metrics["num_test_graphs"] = len(test_graphs)
            metrics["split_policy"] = "held_out_seed"
            metrics["train_seeds"] = args.train_seeds
            metrics["validation_seeds"] = args.validation_seeds
            metrics["test_seeds"] = args.test_seeds
            rows.append(metrics)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).sort_values(["base_model", "propagation_steps"])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    save_sensitivity_figure(metrics_df, output_dir / "figures")
    print(metrics_df[["model", "balanced_accuracy", "f1", "recall", "precision", "fpr"]].to_string(index=False))
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
