#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from mlrpl_dra.cooja import build_node_snapshots, graph_summary, parse_cooja_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Cooja RPL-DRA logs into tabular datasets.")
    parser.add_argument(
        "--log-root",
        required=True,
        help="Directory containing logs-* folders or a single COOJA.testlog file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "cooja"),
        help="Output directory for parsed CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="logs-50n-*/COOJA.testlog",
        help="Glob pattern used when --log-root is a directory.",
    )
    parser.add_argument("--bucket-s", type=int, default=60, help="Snapshot width in simulated seconds.")
    parser.add_argument("--warmup-s", type=int, default=60, help="First snapshot time in simulated seconds.")
    parser.add_argument(
        "--include-dra-audit",
        action="store_true",
        help="Merge DRA_ADVERTISE audit fields into node_snapshots.csv. Do not use this for model training.",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if log_root.is_file():
        log_paths = [log_root]
    else:
        log_paths = sorted(log_root.glob(args.pattern))
    if not log_paths:
        raise SystemExit(f"No COOJA.testlog files found under {log_root}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_cooja_logs(log_paths)
    for name, frame in parsed.items():
        frame.to_csv(output_dir / f"{name}_events.csv", index=False)

    snapshots = build_node_snapshots(
        parsed["trace"],
        parsed["dra"],
        parsed["packet"],
        bucket_s=args.bucket_s,
        warmup_s=args.warmup_s,
        include_dra_audit=args.include_dra_audit,
    )
    snapshots.to_csv(output_dir / "node_snapshots.csv", index=False)
    summary = graph_summary(snapshots)
    summary.to_csv(output_dir / "graph_summary.csv", index=False)

    print(f"Parsed logs: {len(log_paths)}")
    print(f"TRACE rows: {len(parsed['trace'])}")
    print(f"Packet rows: {len(parsed['packet'])}")
    print(f"DRA rows: {len(parsed['dra'])}")
    print(f"Node snapshots: {len(snapshots)}")
    print(f"Graph snapshots: {snapshots['snapshot_id'].nunique() if not snapshots.empty else 0}")
    if not summary.empty:
        print(summary[["scenario", "bucket_end_s", "nodes", "attackers", "reachable_nodes", "mean_pdr"]].tail().to_string(index=False))


if __name__ == "__main__":
    main()
