#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.cooja import graphs_from_snapshots
from mlrpl_dra.metrics import binary_metrics, pick_threshold


def _seed_from_graph_id(graph_id: str) -> int | None:
    match = re.search(r"_seed(\d+)", graph_id)
    return int(match.group(1)) if match else None


def split_graphs_by_seed(graphs, train_seeds: set[int], validation_seeds: set[int], test_seeds: set[int]):
    train_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in train_seeds]
    validation_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in validation_seeds]
    test_graphs = [graph for graph in graphs if _seed_from_graph_id(graph.graph_id) in test_seeds]
    return train_graphs, validation_graphs, test_graphs


def stack_xy(graphs) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([graph.features for graph in graphs])
    y = np.concatenate([graph.labels for graph in graphs]).astype(int)
    return x, y


def fit_with_optional_weights(model, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
    if hasattr(model, "steps"):
        final_step = model.steps[-1][0]
        try:
            return model.fit(x, y, **{f"{final_step}__sample_weight": sample_weight})
        except TypeError:
            return model.fit(x, y)
    try:
        return model.fit(x, y, sample_weight=sample_weight)
    except TypeError:
        return model.fit(x, y)


def predict_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probabilities or decision scores.")


def param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def build_candidates(seed: int, pos_weight: float):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    class_weight = {0: 1.0, 1: pos_weight}
    candidates: dict[str, list[tuple[str, Any]]] = {
        "logistic_regression": [],
        "random_forest": [],
        "xgboost": [],
        "lightgbm": [],
        "mlp": [],
    }

    for params in param_grid({"C": [0.1, 1.0, 10.0]}):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=params["C"],
                class_weight=class_weight,
                max_iter=3000,
                solver="lbfgs",
                random_state=seed,
            ),
        )
        candidates["logistic_regression"].append((f"C={params['C']}", model))

    for params in param_grid({"max_depth": [None, 8, 16], "min_samples_leaf": [1, 3]}):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
        label = f"max_depth={params['max_depth']},min_leaf={params['min_samples_leaf']}"
        candidates["random_forest"].append((label, model))

    for params in param_grid({"max_depth": [2, 4], "learning_rate": [0.03, 0.10], "n_estimators": [150, 300]}):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=seed,
        )
        label = ",".join(f"{key}={value}" for key, value in params.items())
        candidates["xgboost"].append((label, model))

    for params in param_grid({"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.10], "n_estimators": [150, 300]}):
        model = LGBMClassifier(
            objective="binary",
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            class_weight=class_weight,
            random_state=seed,
            verbosity=-1,
        )
        label = ",".join(f"{key}={value}" for key, value in params.items())
        candidates["lightgbm"].append((label, model))

    for params in param_grid(
        {
            "hidden_layer_sizes": [(48,), (64,), (64, 32)],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001, 0.003],
        }
    ):
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
                max_iter=1200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=seed,
            ),
        )
        label = ",".join(f"{key}={value}" for key, value in params.items())
        candidates["mlp"].append((label, model))

    return candidates


def evaluate_baseline_group(
    name: str,
    candidates: list[tuple[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold_metric: str,
    positive_class_weight: float,
) -> dict[str, Any]:
    sample_weight = np.where(y_train == 1, positive_class_weight, 1.0)
    best: dict[str, Any] | None = None

    for label, model in candidates:
        fit_with_optional_weights(model, x_train, y_train, sample_weight)
        val_prob = predict_probability(model, x_val)
        threshold, val_score = pick_threshold(y_val, val_prob, metric=threshold_metric)
        val_bacc_threshold, val_bacc = pick_threshold(y_val, val_prob, metric="balanced_accuracy")
        candidate = {
            "model": name,
            "selected_params": label,
            "estimator": model,
            "threshold": threshold,
            f"validation_{threshold_metric}": val_score,
            "validation_balanced_accuracy": val_bacc,
            "validation_balanced_accuracy_threshold": val_bacc_threshold,
        }
        if best is None or val_score > best[f"validation_{threshold_metric}"]:
            best = candidate

    assert best is not None
    test_prob = predict_probability(best["estimator"], x_test)
    out = binary_metrics(y_test, test_prob, best["threshold"])
    out.update({key: value for key, value in best.items() if key != "estimator"})
    out["threshold_metric"] = threshold_metric
    out["positive_class_weight"] = positive_class_weight
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical and neural flat-feature baselines on Cooja snapshots.")
    parser.add_argument(
        "--snapshots",
        default=str(PROJECT_ROOT / "data" / "cooja_clean_5seed" / "node_snapshots.csv"),
        help="Parsed node_snapshots.csv file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "cooja_clean_5seed_flat_baselines"),
        help="Output directory.",
    )
    parser.add_argument("--train-seeds", default="1,2,3")
    parser.add_argument("--validation-seeds", default="4")
    parser.add_argument("--test-seeds", default="5")
    parser.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1", "precision", "recall", "accuracy"],
        default="f1",
    )
    parser.add_argument("--positive-class-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshots = pd.read_csv(args.snapshots)
    graphs = graphs_from_snapshots(snapshots)
    train_seeds = {int(seed) for seed in args.train_seeds.split(",") if seed}
    validation_seeds = {int(seed) for seed in args.validation_seeds.split(",") if seed}
    test_seeds = {int(seed) for seed in args.test_seeds.split(",") if seed}
    train_graphs, validation_graphs, test_graphs = split_graphs_by_seed(graphs, train_seeds, validation_seeds, test_seeds)
    if not train_graphs or not validation_graphs or not test_graphs:
        raise SystemExit("Seed split produced an empty train, validation, or test set.")

    x_train, y_train = stack_xy(train_graphs)
    x_val, y_val = stack_xy(validation_graphs)
    x_test, y_test = stack_xy(test_graphs)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    print(f"Train nodes: {len(y_train)} ({pos} DRA, {neg} benign)")
    print(f"Validation nodes: {len(y_val)}; test nodes: {len(y_test)}")

    rows = []
    candidates = build_candidates(args.seed, args.positive_class_weight)
    for name, group in candidates.items():
        print(f"Fitting {name} ({len(group)} candidates)...")
        row = evaluate_baseline_group(
            name,
            group,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            args.threshold_metric,
            args.positive_class_weight,
        )
        row["num_train_graphs"] = len(train_graphs)
        row["num_validation_graphs"] = len(validation_graphs)
        row["num_test_graphs"] = len(test_graphs)
        row["split_policy"] = "held_out_seed"
        row["train_seeds"] = args.train_seeds
        row["validation_seeds"] = args.validation_seeds
        row["test_seeds"] = args.test_seeds
        rows.append(row)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics[["model", "balanced_accuracy", "f1", "recall", "precision", "fpr", "threshold", "selected_params"]].to_string(index=False))
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
