#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, xy, wh, title, body, edge, fill):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.11, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.06, body, ha="center", va="center", fontsize=9.2, linespacing=1.22)
    return patch


def arrow(ax, start, end, color="#444444", dashed=False, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.25,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper workflow figure.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.8, 7.8))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    gray = ("#333333", "#F5F6F8")
    orange = ("#C45A12", "#FFF5EC")
    plum = ("#8A2F61", "#FFF4FA")
    gold = ("#C27A00", "#FFF9E8")
    teal = ("#067A78", "#EFFCFC")
    blue = ("#2F5DCC", "#F2F6FF")
    green = ("#24823B", "#F2FBF3")

    add_box(ax, (0.35, 6.35), (2.45, 0.98), "1. Cooja/Contiki-NG", "RPL-DRA simulations\n50 nodes; 10-30% DRA\n5 seeds; 600 s runs", *gray)
    add_box(ax, (3.35, 6.35), (2.45, 0.98), "2. Raw simulation logs", "TRACE records\nTX/RX packet events\nDRA audit events", *orange)
    add_box(ax, (6.35, 6.2), (2.75, 1.18), "3. Trace parsing", "remove warm-up\naggregate into 60 s windows\nbuild structured tables", *plum)
    add_box(ax, (9.85, 6.05), (3.5, 1.35), "4. Node snapshots", "feature matrix X and labels Y\none graph per time bucket\nrank, RSSI, parent, delay, TX, PDR", *gold)

    add_box(ax, (3.48, 4.65), (2.65, 0.94), "Audit check only", "confirms DRA activity\nexcluded from model inputs", "#D62728", "#FFF8F8")
    add_box(ax, (6.28, 4.34), (3.25, 1.02), "5A. Flat-feature view", "input: X only\nnode observations\nwithout adjacency", *teal)
    add_box(ax, (10.0, 4.62), (3.45, 1.0), "5B. Multilayer graph view", "input: same X plus\nrelation-specific adjacency matrices", *blue)

    add_box(ax, (6.28, 3.04), (3.25, 1.08), "6A. Flat-feature baselines", "Logistic Regression, Random Forest\nXGBoost, LightGBM, MLP\nDQN, DDQN, Dueling DDQN", *teal)
    add_box(ax, (10.0, 3.18), (3.45, 1.08), "6B. Four RPL relation layers", "A(r): routing; A(q): link quality\nA(t): temporal behavior\nA(a): trust/anomaly similarity", *blue)
    add_box(ax, (10.0, 1.98), (3.45, 0.92), "7. Proposed graph models", "ML-GCN: fixed multilayer fusion\nAttn-ML-GCN: learned layer weights", *blue)

    add_box(ax, (6.52, 1.48), (3.65, 0.94), "8. Controlled training protocol", "train: seeds 1-3; validation: seed 4\nselect hyperparameters and F1 threshold", "#9A2F5D", "#FFF5FA")
    add_box(ax, (6.52, 0.38), (3.65, 0.92), "9. Held-out test evaluation", "test: seed 5 only\nBAcc, F1, precision, recall, FPR", *green)
    add_box(ax, (11.15, 0.38), (3.05, 0.92), "10. Results and discussion", "benchmark proposed graph models\nagainst flat-feature baselines\nreport ablation and attention weights", *green)

    arrow(ax, (2.8, 6.84), (3.35, 6.84))
    arrow(ax, (5.8, 6.84), (6.35, 6.84))
    arrow(ax, (9.1, 6.84), (9.85, 6.78))
    arrow(ax, (4.58, 6.35), (4.75, 5.59), "#D62728", dashed=True)
    arrow(ax, (11.3, 6.05), (7.91, 5.36), rad=-0.06)
    arrow(ax, (11.6, 6.05), (11.72, 5.62))
    arrow(ax, (7.91, 4.34), (7.91, 4.12))
    arrow(ax, (11.72, 4.62), (11.72, 4.26))
    arrow(ax, (11.72, 3.18), (11.72, 2.9))
    arrow(ax, (7.91, 3.04), (8.1, 2.42), rad=0.12)
    arrow(ax, (10.55, 2.16), (10.17, 1.95))
    arrow(ax, (8.32, 1.5), (8.32, 1.3))
    arrow(ax, (10.17, 0.84), (11.15, 0.84))

    add_box(ax, (0.5, 0.45), (3.25, 1.18), "Legend", "solid arrow: data/model flow\nred dashed arrow: audit verification only", "#888888", "#FFFFFF")

    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
