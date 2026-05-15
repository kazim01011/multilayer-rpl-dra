from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .simulator import (
    RPLGraph,
    _normalize_features,
    _row_normalize_with_self_loops,
    _similarity_layer,
)


SENTINEL_U16 = 65535
SENTINEL_RSSI = 32767


def _scenario_from_path(path: Path) -> tuple[str, float, int | None]:
    match = re.search(r"logs-50n-(\d+)p-seed(\d+)", str(path))
    if match:
        ratio_pct = int(match.group(1))
        seed = int(match.group(2))
        return f"50n_{ratio_pct}p_seed{seed}", ratio_pct / 100.0, seed
    return path.parent.name, float("nan"), None


def _parse_kv(tokens: Iterable[str]) -> dict[str, str]:
    out = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def _coerce(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _row_from_kv(
    scenario: str,
    ratio: float,
    seed: int | None,
    sim_time_us: int | None,
    mote: int | None,
    event_type: str,
    kv: dict[str, str],
) -> dict[str, int | float | str | None]:
    row: dict[str, int | float | str | None] = {
        "scenario": scenario,
        "attack_ratio": ratio,
        "seed": seed,
        "sim_time_us": sim_time_us,
        "mote": mote,
        "record_type": event_type,
    }
    for key, value in kv.items():
        row[key] = _coerce(value)
    if "time_ticks" in row:
        row["time_s"] = float(row["time_ticks"]) / 1000.0
    elif sim_time_us is not None:
        row["time_s"] = sim_time_us / 1_000_000.0
    return row


def parse_cooja_testlog(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    scenario, ratio, seed = _scenario_from_path(path)
    meta_rows = []
    trace_rows = []
    packet_rows = []
    dra_rows = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("Random seed:"):
            continue

        if line.startswith("META "):
            kv = _parse_kv(line.split()[1:])
            meta_rows.append(_row_from_kv(scenario, ratio, seed, None, None, "META", kv))
            continue

        parts = line.split()
        if len(parts) < 3:
            continue
        if not parts[0].isdigit() or not parts[1].startswith("mote="):
            continue

        sim_time_us = int(parts[0])
        mote = int(parts[1].split("=", 1)[1])
        event_type = parts[2]
        kv = _parse_kv(parts[3:])
        row = _row_from_kv(scenario, ratio, seed, sim_time_us, mote, event_type, kv)

        if event_type == "TRACE":
            trace_rows.append(row)
        elif event_type in {"TX", "RX"}:
            packet_rows.append(row)
        elif event_type == "DRA_ADVERTISE":
            dra_rows.append(row)

    return {
        "meta": pd.DataFrame(meta_rows),
        "trace": pd.DataFrame(trace_rows),
        "packet": pd.DataFrame(packet_rows),
        "dra": pd.DataFrame(dra_rows),
    }


def parse_cooja_logs(log_paths: Iterable[str | Path]) -> dict[str, pd.DataFrame]:
    buckets = {"meta": [], "trace": [], "packet": [], "dra": []}
    for path in log_paths:
        parsed = parse_cooja_testlog(path)
        for key, df in parsed.items():
            if not df.empty:
                buckets[key].append(df)
    return {
        key: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for key, parts in buckets.items()
    }


def _clean_u16(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[(numeric.notna()) & (numeric < SENTINEL_U16)]
    fill = float(finite.max()) + fallback if not finite.empty else fallback
    return numeric.mask((numeric >= SENTINEL_U16) | numeric.isna(), fill)


def _clean_rssi(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask((numeric >= SENTINEL_RSSI) | numeric.isna(), -100.0)


def build_node_snapshots(
    trace_df: pd.DataFrame,
    dra_df: pd.DataFrame,
    packet_df: pd.DataFrame,
    bucket_s: int = 60,
    warmup_s: int = 60,
    include_dra_audit: bool = False,
) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()

    rows = []
    trace_df = trace_df.copy()
    trace_df["time_s"] = pd.to_numeric(trace_df["time_s"], errors="coerce")
    if not dra_df.empty:
        dra_df = dra_df.copy()
        dra_df["time_s"] = pd.to_numeric(dra_df["time_s"], errors="coerce")
    if not packet_df.empty:
        packet_df = packet_df.copy()
        packet_df["time_s"] = pd.to_numeric(packet_df["time_s"], errors="coerce")

    for scenario, scenario_trace in trace_df.groupby("scenario"):
        scenario_trace = scenario_trace.sort_values("time_s")
        max_time = int(scenario_trace["time_s"].max() // bucket_s * bucket_s)
        node_ids = sorted(int(node) for node in scenario_trace["node"].dropna().unique())
        scenario_dra = (
            dra_df[dra_df["scenario"] == scenario]
            if include_dra_audit and not dra_df.empty
            else pd.DataFrame()
        )
        scenario_packets = packet_df[packet_df["scenario"] == scenario] if not packet_df.empty else pd.DataFrame()

        for bucket_end in range(warmup_s, max_time + 1, bucket_s):
            latest = (
                scenario_trace[scenario_trace["time_s"] <= bucket_end]
                .sort_values(["node", "time_s"])
                .drop_duplicates("node", keep="last")
            )
            if len(latest) < len(node_ids):
                continue

            latest = latest.copy()
            latest["bucket_end_s"] = bucket_end
            latest["snapshot_id"] = f"{scenario}_t{bucket_end:05d}"

            if include_dra_audit and not scenario_dra.empty:
                latest_dra = (
                    scenario_dra[scenario_dra["time_s"] <= bucket_end]
                    .sort_values(["mote", "time_s"])
                    .drop_duplicates("mote", keep="last")
                    .set_index("mote")
                )
                latest["advertised_rank"] = latest["node"].map(latest_dra.get("advertised_rank", pd.Series(dtype=float)))
                latest["actual_rank_from_dio"] = latest["node"].map(latest_dra.get("actual_rank", pd.Series(dtype=float)))
                latest["rank_decrement"] = latest["node"].map(latest_dra.get("decrement", pd.Series(dtype=float)))
            else:
                latest["advertised_rank"] = np.nan
                latest["actual_rank_from_dio"] = np.nan
                latest["rank_decrement"] = np.nan

            latest["advertised_rank"] = latest["advertised_rank"].fillna(latest["rank"])
            latest["actual_rank_from_dio"] = latest["actual_rank_from_dio"].fillna(latest["rank"])
            latest["rank_delta"] = latest["actual_rank_from_dio"] - latest["advertised_rank"]

            if not scenario_packets.empty:
                window = scenario_packets[
                    (scenario_packets["time_s"] > bucket_end - bucket_s)
                    & (scenario_packets["time_s"] <= bucket_end)
                ]
                tx_counts = window[window["record_type"] == "TX"].groupby("node").size()
                rx_counts = window[window["record_type"] == "RX"].groupby("sender").size()
                delay = window[window["record_type"] == "RX"].groupby("sender")["delay_ticks"].mean()
                latest["tx_count"] = latest["node"].map(tx_counts).fillna(0).astype(int)
                latest["rx_success_count"] = latest["node"].map(rx_counts).fillna(0).astype(int)
                latest["mean_delivery_delay_ticks"] = latest["node"].map(delay).fillna(0.0)
            else:
                latest["tx_count"] = 0
                latest["rx_success_count"] = 0
                latest["mean_delivery_delay_ticks"] = 0.0

            latest["pdr"] = latest["rx_success_count"] / latest["tx_count"].replace(0, np.nan)
            latest["pdr"] = latest["pdr"].fillna(0.0)
            rows.append(latest)

    if not rows:
        return pd.DataFrame()
    snapshots = pd.concat(rows, ignore_index=True)
    sort_cols = ["scenario", "bucket_end_s", "node"]
    return snapshots.sort_values(sort_cols).reset_index(drop=True)


def graph_summary(snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    rows = []
    for snapshot_id, group in snapshots.groupby("snapshot_id"):
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "scenario": group["scenario"].iloc[0],
                "attack_ratio": group["attack_ratio"].iloc[0],
                "bucket_end_s": group["bucket_end_s"].iloc[0],
                "nodes": group["node"].nunique(),
                "attackers": int(group["attack"].sum()),
                "reachable_nodes": int((pd.to_numeric(group["rank"]) < SENTINEL_U16).sum()),
                "mean_pdr": float(group["pdr"].mean()),
                "mean_rank_delta": float(group["rank_delta"].mean()) if "rank_delta" in group else 0.0,
            }
        )
    return pd.DataFrame(rows)


def graphs_from_snapshots(snapshots: pd.DataFrame) -> list[RPLGraph]:
    graphs: list[RPLGraph] = []
    if snapshots.empty:
        return graphs

    for snapshot_id, group in snapshots.groupby("snapshot_id"):
        group = group.sort_values("node").reset_index(drop=True)
        node_to_idx = {int(node): idx for idx, node in enumerate(group["node"])}
        n = len(group)

        rank = _clean_u16(group["rank"], fallback=512.0)
        parent_metric = _clean_u16(group["parent_metric"], fallback=128.0)
        rssi = _clean_rssi(group["rssi"])
        delay = pd.to_numeric(group["mean_delivery_delay_ticks"], errors="coerce").fillna(0.0)
        dag_rank = pd.to_numeric(group["dag_rank"], errors="coerce").fillna(0.0)
        nbrs = pd.to_numeric(group["nbrs"], errors="coerce").fillna(0.0)
        tx_count = pd.to_numeric(group["tx_count"], errors="coerce").fillna(0.0)
        pdr = pd.to_numeric(group["pdr"], errors="coerce").fillna(0.0)

        node_norm = (pd.to_numeric(group["node"], errors="coerce") - 1) / max(n - 1, 1)
        raw_features = np.column_stack(
            [
                node_norm,
                rank,
                dag_rank,
                rssi,
                parent_metric,
                nbrs,
                delay,
                tx_count,
                pdr,
            ]
        ).astype(float)
        features = _normalize_features(raw_features)

        routing = np.zeros((n, n), dtype=float)
        link_quality = np.zeros((n, n), dtype=float)
        for idx, row in group.iterrows():
            parent = int(row["parent"]) if not pd.isna(row["parent"]) else 0
            if parent in node_to_idx and parent != int(row["node"]):
                parent_idx = node_to_idx[parent]
                routing[idx, parent_idx] = 1.0
                routing[parent_idx, idx] = 1.0
                metric = parent_metric.iloc[idx]
                weight = 1.0 / max(float(metric) / 128.0, 1.0)
                link_quality[idx, parent_idx] = weight
                link_quality[parent_idx, idx] = weight

        temporal_signal = (delay / max(float(delay.max()), 1.0)) + (1.0 - pdr)
        temporal = _similarity_layer(np.asarray(temporal_signal, dtype=float), scale=0.50, threshold=0.35)

        low_rank_anomaly = np.maximum(0.0, float(np.nanmedian(rank)) - rank) / 512.0
        metric_anomaly = parent_metric / max(float(parent_metric.median()), 1.0)
        trust_signal = np.asarray(low_rank_anomaly + 0.35 * metric_anomaly + (1.0 - pdr), dtype=float)
        trust = _similarity_layer(trust_signal, scale=0.75, threshold=0.30)

        layers = {
            "routing": _row_normalize_with_self_loops(routing),
            "link_quality": _row_normalize_with_self_loops(link_quality),
            "temporal": _row_normalize_with_self_loops(temporal),
            "trust": _row_normalize_with_self_loops(trust),
        }
        labels = pd.to_numeric(group["attack"], errors="coerce").fillna(0).astype(int).to_numpy()

        graphs.append(
            RPLGraph(
                graph_id=str(snapshot_id),
                ratio=float(group["attack_ratio"].iloc[0]),
                features=features,
                labels=labels,
                layers=layers,
                metadata={
                    "raw_features": raw_features,
                    "node_ids": group["node"].to_numpy(),
                    "parents": group["parent"].to_numpy(),
                    "bucket_end_s": float(group["bucket_end_s"].iloc[0]),
                    "scenario": str(group["scenario"].iloc[0]),
                },
            )
        )
    return graphs
