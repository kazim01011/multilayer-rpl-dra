#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine metrics.csv files from multiple result directories.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result directories containing metrics.csv.")
    parser.add_argument("--output", required=True, help="Output result directory.")
    parser.add_argument(
        "--dedupe-by-model",
        action="store_true",
        help="Keep the last row for duplicate model names after concatenation.",
    )
    args = parser.parse_args()

    frames = []
    for item in args.inputs:
        path = Path(item) / "metrics.csv"
        if not path.exists():
            raise SystemExit(f"Missing metrics file: {path}")
        frame = pd.read_csv(path)
        frame["source_result_dir"] = str(Path(item))
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True, sort=False)
    if args.dedupe_by_model:
        metrics = metrics.drop_duplicates(subset=["model"], keep="last")
    if "balanced_accuracy" in metrics.columns:
        metrics = metrics.sort_values("balanced_accuracy", ascending=False)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    print(metrics[["model", "balanced_accuracy", "f1", "recall", "precision", "fpr"]].to_string(index=False))
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
