#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["balanced_accuracy", "f1", "recall", "precision", "fpr"]
ATTENTION_COLS = ["attention_routing", "attention_link_quality", "attention_temporal", "attention_trust"]


def fmt(mean: float, std: float | None = None) -> str:
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def overall_table(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby("model")[METRICS].agg(["mean", "std"])
    rows = []
    for model, values in grouped.iterrows():
        row = {"model": model}
        for metric in METRICS:
            row[metric] = fmt(values[(metric, "mean")], values[(metric, "std")])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def ratio_table(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["malicious_ratio", "model"])[METRICS].agg(["mean", "std"])
    rows = []
    for (ratio, model), values in grouped.iterrows():
        row = {"malicious_ratio": ratio, "model": model}
        for metric in METRICS:
            row[metric] = fmt(values[(metric, "mean")], values[(metric, "std")])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["malicious_ratio", "balanced_accuracy"], ascending=[True, False])


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows)) if rows else len(columns[idx])
        for idx in range(len(columns))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(columns), sep, *(render(row) for row in rows)])


def attention_table(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ATTENTION_COLS if col in metrics.columns]
    if not cols:
        return pd.DataFrame()
    attention_rows = metrics.dropna(subset=cols, how="all")
    if attention_rows.empty:
        return pd.DataFrame()
    grouped = attention_rows.groupby("model")[cols].agg(["mean", "std"])
    rows = []
    for model, values in grouped.iterrows():
        row = {"model": model}
        for col in cols:
            row[col.replace("attention_", "")] = fmt(values[(col, "mean")], values[(col, "std")])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create markdown summary tables from experiment metrics.")
    parser.add_argument("result_dir", help="Directory containing metrics.csv.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    metrics = pd.read_csv(result_dir / "metrics.csv")
    overall = overall_table(metrics)
    by_ratio = ratio_table(metrics)
    attention = attention_table(metrics)

    attention_section = ""
    if not attention.empty:
        attention_section = "\n\n## Learned Layer Attention Mean +/- Std\n\n" + markdown_table(attention) + "\n"
    out = result_dir / "paper_tables.md"
    out.write_text(
        "# Paper Tables\n\n"
        "## Overall Mean +/- Std\n\n"
        + markdown_table(overall)
        + "\n\n## By Attack Ratio Mean +/- Std\n\n"
        + markdown_table(by_ratio)
        + attention_section
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
