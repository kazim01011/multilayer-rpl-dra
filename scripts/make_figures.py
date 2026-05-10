#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1 score",
    "recall": "Recall",
    "precision": "Precision",
    "fpr": "False positive rate",
}
MODEL_LABELS = {
    "dqn": "DQN",
    "ddqn": "DDQN",
    "dueling_ddqn": "Dueling DDQN",
    "ml_gcn": "ML-GCN",
    "attn_ml_gcn": "Attn-ML-GCN",
}
MODEL_ORDER = ["dqn", "ddqn", "dueling_ddqn", "ml_gcn", "attn_ml_gcn"]
MODEL_COLORS = {
    "dqn": "#7A869A",
    "ddqn": "#4C78A8",
    "dueling_ddqn": "#72B7B2",
    "ml_gcn": "#F58518",
    "attn_ml_gcn": "#E45756",
}
ATTENTION_COLS = ["attention_routing", "attention_link_quality", "attention_temporal", "attention_trust"]
ATTENTION_LABELS = {
    "attention_routing": "Routing",
    "attention_link_quality": "Link quality",
    "attention_temporal": "Temporal",
    "attention_trust": "Trust",
}


def model_sort_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_stats(metrics: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    grouped = metrics.groupby("model")[cols].agg(["mean", "std"]).reset_index()
    grouped["sort_key"] = grouped["model"].map(model_sort_key)
    return grouped.sort_values("sort_key")


def plot_overall_model_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    stats = summary_stats(metrics, ["balanced_accuracy", "f1"])
    models = stats["model"].to_list()
    labels = [MODEL_LABELS.get(model, model) for model in models]
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset, metric, color in [
        (-width / 2, "balanced_accuracy", "#3B6FB6"),
        (width / 2, "f1", "#D55E00"),
    ]:
        means = stats[(metric, "mean")].to_numpy()
        stds = stats[(metric, "std")].fillna(0.0).to_numpy()
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=color,
            edgecolor="#263238",
            linewidth=0.5,
            label=METRIC_LABELS[metric],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.82, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    clean_axis(ax)
    fig.subplots_adjust(top=0.82)
    save_figure(fig, out_dir, "fig01_overall_model_comparison")


def plot_attack_ratio_curve(metrics: pd.DataFrame, out_dir: Path) -> None:
    grouped = (
        metrics.groupby(["malicious_ratio", "model"])["balanced_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    models = sorted(grouped["model"].unique(), key=model_sort_key)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for model in models:
        part = grouped[grouped["model"] == model].sort_values("malicious_ratio")
        x = part["malicious_ratio"].to_numpy() * 100
        y = part["mean"].to_numpy()
        err = part["std"].fillna(0.0).to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=err,
            marker="o",
            linewidth=2.0,
            capsize=3,
            color=MODEL_COLORS.get(model, "#333333"),
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xticks(sorted(metrics["malicious_ratio"].unique() * 100))
    ax.set_xlabel("Malicious nodes (%)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.88, 1.0)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    clean_axis(ax)
    save_figure(fig, out_dir, "fig02_attack_ratio_balanced_accuracy")


def plot_precision_recall_fpr(metrics: pd.DataFrame, out_dir: Path) -> None:
    stats = summary_stats(metrics, ["precision", "recall", "fpr"])
    models = stats["model"].to_list()
    labels = [MODEL_LABELS.get(model, model) for model in models]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx, metric in enumerate(["precision", "recall", "fpr"]):
        means = stats[(metric, "mean")].to_numpy()
        stds = stats[(metric, "std")].fillna(0.0).to_numpy()
        ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=["#4C78A8", "#54A24B", "#E45756"][idx],
            edgecolor="#263238",
            linewidth=0.5,
            label=METRIC_LABELS[metric],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    clean_axis(ax)
    fig.subplots_adjust(top=0.82)
    save_figure(fig, out_dir, "fig03_precision_recall_fpr")


def plot_attention_weights(metrics: pd.DataFrame, out_dir: Path) -> None:
    cols = [col for col in ATTENTION_COLS if col in metrics.columns]
    attention = metrics.dropna(subset=cols, how="all")
    if attention.empty:
        return

    means = attention[cols].mean()
    stds = attention[cols].std(ddof=1).fillna(0.0)
    x = np.arange(len(cols))

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.bar(
        x,
        means.to_numpy(),
        yerr=stds.to_numpy(),
        capsize=3,
        color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"],
        edgecolor="#263238",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ATTENTION_LABELS[col] for col in cols], rotation=15, ha="right")
    ax.set_ylabel("Learned attention weight")
    ax.set_ylim(0.0, max(0.35, float((means + stds).max()) + 0.04))
    clean_axis(ax)
    save_figure(fig, out_dir, "fig04_layer_attention_weights")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper figures from experiment metrics.")
    parser.add_argument("result_dir", help="Directory containing metrics.csv.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    metrics = pd.read_csv(result_dir / "metrics.csv")
    out_dir = result_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    plot_overall_model_comparison(metrics, out_dir)
    plot_attack_ratio_curve(metrics, out_dir)
    plot_precision_recall_fpr(metrics, out_dir)
    plot_attention_weights(metrics, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
