#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.config import ExperimentConfig, SimulationConfig, SplitConfig, TrainingConfig
from mlrpl_dra.cooja import graphs_from_snapshots
from mlrpl_dra.experiment import evaluate_model, split_graphs


def _seed_from_graph_id(graph_id: str) -> int | None:
    match = re.search(r"_seed(\d+)", graph_id)
    return int(match.group(1)) if match else None


def split_graphs_by_seed(graphs, train: float, validation: float):
    seeds = sorted({seed for graph in graphs if (seed := _seed_from_graph_id(graph.graph_id)) is not None})
    if len(seeds) < 3:
        return None

    n_train = max(1, int(round(train * len(seeds))))
    n_val = max(1, int(round(validation * len(seeds))))
    if n_train + n_val >= len(seeds):
        n_train = max(1, len(seeds) - 2)
        n_val = 1

    train_seeds = set(seeds[:n_train])
    val_seeds = set(seeds[n_train:n_train + n_val])
    test_seeds = set(seeds[n_train + n_val:])

    train_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in train_seeds]
    val_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in val_seeds]
    test_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in test_seeds]
    return train_graphs, val_graphs, test_graphs, train_seeds, val_seeds, test_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark models on parsed Cooja graph snapshots.")
    parser.add_argument(
        "--snapshots",
        default=str(PROJECT_ROOT / "data" / "cooja" / "node_snapshots.csv"),
        help="Parsed node_snapshots.csv file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "cooja_seed1"),
        help="Output directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dqn", "ddqn", "dueling_ddqn", "agg_gcn", "ml_gcn", "attn_ml_gcn"],
        help="Model names to evaluate.",
    )
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1", "precision", "recall", "accuracy"],
        default="balanced_accuracy",
        help="Validation metric used to select the classification threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-policy",
        choices=["seed", "random"],
        default="seed",
        help="Use held-out Cooja seeds when available; random falls back to snapshot-level split.",
    )
    args = parser.parse_args()

    snapshots = pd.read_csv(args.snapshots)
    graphs = graphs_from_snapshots(snapshots)
    if len(graphs) < 3:
        raise SystemExit("Need at least three graph snapshots for train/validation/test split.")

    cfg = ExperimentConfig(
        run_name="cooja_seed1",
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

    seed_split = split_graphs_by_seed(graphs, cfg.split.train, cfg.split.validation) if args.split_policy == "seed" else None
    split_meta = {}
    if seed_split is None:
        train_graphs, val_graphs, test_graphs = split_graphs(graphs, cfg.split.train, cfg.split.validation, args.seed)
        split_meta["split_policy"] = "random_snapshot"
    else:
        train_graphs, val_graphs, test_graphs, train_seeds, val_seeds, test_seeds = seed_split
        split_meta.update(
            {
                "split_policy": "held_out_seed",
                "train_seeds": ",".join(map(str, sorted(train_seeds))),
                "validation_seeds": ",".join(map(str, sorted(val_seeds))),
                "test_seeds": ",".join(map(str, sorted(test_seeds))),
            }
        )
    rows = []
    for model_name in args.models:
        metrics = evaluate_model(
            model_name,
            cfg,
            train_graphs,
            val_graphs,
            test_graphs,
            args.seed,
            threshold_metric=args.threshold_metric,
            positive_class_weight=args.positive_class_weight,
        )
        metrics["num_train_graphs"] = len(train_graphs)
        metrics["num_validation_graphs"] = len(val_graphs)
        metrics["num_test_graphs"] = len(test_graphs)
        metrics.update(split_meta)
        rows.append(metrics)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics_df[["model", "balanced_accuracy", "f1", "recall", "precision", "fpr"]].to_string(index=False))
    if split_meta:
        print(f"\nSplit policy: {split_meta['split_policy']}")
        if split_meta["split_policy"] == "held_out_seed":
            print(f"Train seeds: {split_meta['train_seeds']}")
            print(f"Validation seeds: {split_meta['validation_seeds']}")
            print(f"Test seeds: {split_meta['test_seeds']}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
