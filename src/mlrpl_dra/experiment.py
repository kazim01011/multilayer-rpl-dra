from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig, config_to_dict
from .metrics import binary_metrics, pick_threshold
from .models import ModelConfig, build_model
from .simulator import RPLGraph, generate_dataset


def split_graphs(graphs: list[RPLGraph], train: float, validation: float, seed: int) -> tuple[list[RPLGraph], list[RPLGraph], list[RPLGraph]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(graphs))
    n_train = int(round(train * len(graphs)))
    n_val = int(round(validation * len(graphs)))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx], [graphs[i] for i in test_idx]


def labels_for(graphs: list[RPLGraph]) -> np.ndarray:
    return np.concatenate([g.labels for g in graphs])


def stable_model_offset(model_name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(model_name)) % 10000


def evaluate_model(
    model_name: str,
    cfg: ExperimentConfig,
    train_graphs: list[RPLGraph],
    val_graphs: list[RPLGraph],
    test_graphs: list[RPLGraph],
    seed: int,
) -> dict[str, float | str]:
    input_dim = train_graphs[0].features.shape[1]
    model_cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=cfg.training.hidden_dim,
        learning_rate=cfg.training.learning_rate,
        seed=seed + stable_model_offset(model_name),
    )
    model = build_model(model_name, model_cfg)
    model.fit(train_graphs, val_graphs, cfg.training.epochs, cfg.training.patience)

    val_y = labels_for(val_graphs)
    val_prob = model.predict_graphs(val_graphs)
    threshold, val_bacc = pick_threshold(val_y, val_prob)

    test_y = labels_for(test_graphs)
    test_prob = model.predict_graphs(test_graphs)
    out = binary_metrics(test_y, test_prob, threshold)
    out["model"] = model_name
    out["threshold"] = threshold
    out["validation_balanced_accuracy"] = val_bacc
    out.update(model.diagnostics())
    return out


def run_experiment(cfg: ExperimentConfig, output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_root) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config_to_dict(cfg), indent=2),
        encoding="utf-8",
    )

    rows = []
    for seed in cfg.seeds:
        for ratio in cfg.simulation.malicious_ratios:
            ratio_cfg = cfg.simulation.__dict__.copy()
            ratio_cfg["malicious_ratios"] = [ratio]
            from .config import SimulationConfig

            scenario_seed = seed + int(ratio * 1000)
            graphs = generate_dataset(SimulationConfig(**ratio_cfg), scenario_seed)
            train_graphs, val_graphs, test_graphs = split_graphs(
                graphs,
                cfg.split.train,
                cfg.split.validation,
                scenario_seed,
            )
            for model_name in cfg.models:
                metrics = evaluate_model(model_name, cfg, train_graphs, val_graphs, test_graphs, seed)
                metrics["replicate_seed"] = seed
                metrics["malicious_ratio"] = ratio
                metrics["num_train_graphs"] = len(train_graphs)
                metrics["num_validation_graphs"] = len(val_graphs)
                metrics["num_test_graphs"] = len(test_graphs)
                rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "f1",
        "balanced_accuracy",
        "validation_balanced_accuracy",
    ]
    summary_df = (
        metrics_df.groupby("model", as_index=False)[metric_cols]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    summary_std_df = (
        metrics_df.groupby("model", as_index=False)[metric_cols]
        .std(ddof=0)
        .sort_values("balanced_accuracy", ascending=False)
    )
    by_ratio_df = (
        metrics_df.groupby(["malicious_ratio", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    by_ratio_df.columns = [
        "_".join([str(part) for part in col if part])
        for col in by_ratio_df.columns.to_flat_index()
    ]
    by_ratio_df = by_ratio_df.rename(
        columns={"malicious_ratio_": "malicious_ratio", "model_": "model"}
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    summary_df.to_csv(output_dir / "summary_overall_mean.csv", index=False)
    summary_std_df.to_csv(output_dir / "summary_overall_std.csv", index=False)
    by_ratio_df.to_csv(output_dir / "summary_by_ratio.csv", index=False)
    return metrics_df, summary_df
