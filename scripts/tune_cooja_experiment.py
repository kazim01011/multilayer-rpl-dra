#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.config import ExperimentConfig, SimulationConfig, SplitConfig, TrainingConfig
from mlrpl_dra.cooja import graphs_from_snapshots
from mlrpl_dra.experiment import evaluate_model
from scripts.run_cooja_experiment import split_graphs_by_seed


def _metric_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model",
        "hidden_dim",
        "learning_rate",
        "epochs",
        "positive_class_weight",
        "threshold_metric",
        "threshold",
        "balanced_accuracy",
        "f1",
        "recall",
        "precision",
        "fpr",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    return [col for col in preferred if col in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search Cooja graph models using validation-selected thresholds.")
    parser.add_argument(
        "--snapshots",
        default=str(PROJECT_ROOT / "data" / "cooja_clean_5seed" / "node_snapshots.csv"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "cooja_clean_5seed_tuning"))
    parser.add_argument("--models", nargs="+", default=["ml_gcn", "attn_ml_gcn"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[16, 24, 32, 48])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--epochs", nargs="+", type=int, default=[250, 500])
    parser.add_argument("--positive-class-weights", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--threshold-metrics", nargs="+", default=["balanced_accuracy", "f1"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshots = pd.read_csv(args.snapshots)
    graphs = graphs_from_snapshots(snapshots)
    seed_split = split_graphs_by_seed(graphs, train=0.60, validation=0.20)
    if seed_split is None:
        raise SystemExit("Tuning requires at least three Cooja seeds for held-out-seed splitting.")
    train_graphs, val_graphs, test_graphs, train_seeds, val_seeds, test_seeds = seed_split

    rows = []
    grid = list(
        itertools.product(
            args.models,
            args.hidden_dims,
            args.learning_rates,
            args.epochs,
            args.positive_class_weights,
            args.threshold_metrics,
        )
    )
    for idx, (model, hidden_dim, lr, epochs, pos_weight, threshold_metric) in enumerate(grid, start=1):
        cfg = ExperimentConfig(
            run_name="cooja_tuning",
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
                epochs=epochs,
                learning_rate=lr,
                hidden_dim=hidden_dim,
                batch_size=0,
                patience=50,
            ),
            models=[model],
        )
        metrics = evaluate_model(
            model,
            cfg,
            train_graphs,
            val_graphs,
            test_graphs,
            args.seed,
            threshold_metric=threshold_metric,
            positive_class_weight=pos_weight,
        )
        metrics.update(
            {
                "hidden_dim": hidden_dim,
                "learning_rate": lr,
                "epochs": epochs,
                "num_train_graphs": len(train_graphs),
                "num_validation_graphs": len(val_graphs),
                "num_test_graphs": len(test_graphs),
                "train_seeds": ",".join(map(str, sorted(train_seeds))),
                "validation_seeds": ",".join(map(str, sorted(val_seeds))),
                "test_seeds": ",".join(map(str, sorted(test_seeds))),
            }
        )
        rows.append(metrics)
        print(
            f"[{idx:03d}/{len(grid)}] {model} h={hidden_dim} lr={lr:g} epochs={epochs} "
            f"w={pos_weight:g} th={threshold_metric}: "
            f"F1={metrics['f1']:.3f} BAcc={metrics['balanced_accuracy']:.3f} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} FPR={metrics['fpr']:.3f}",
            flush=True,
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["f1", "balanced_accuracy"], ascending=False)
    df.to_csv(output_dir / "tuning_results.csv", index=False)
    print("\nTop configurations by held-out test F1:")
    print(df[_metric_columns(df)].head(12).to_string(index=False))
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
