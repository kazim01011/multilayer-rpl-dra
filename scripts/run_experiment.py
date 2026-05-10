#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.config import load_config
from mlrpl_dra.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilayer RPL DRA detection experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results"), help="Output root directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics, summary = run_experiment(cfg, args.output)
    print("Per-ratio metrics:")
    print(metrics[["malicious_ratio", "model", "balanced_accuracy", "f1", "recall", "precision", "fpr"]].to_string(index=False))
    print("\nSummary:")
    print(summary[["model", "balanced_accuracy", "f1", "recall", "precision", "fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()

