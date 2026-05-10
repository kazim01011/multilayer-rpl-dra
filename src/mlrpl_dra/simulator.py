from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


LAYER_NAMES = ("routing", "link_quality", "temporal", "trust")


@dataclass
class RPLGraph:
    graph_id: str
    ratio: float
    features: np.ndarray
    labels: np.ndarray
    layers: dict[str, np.ndarray]
    metadata: dict[str, np.ndarray | float | int | str]


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _ensure_connected_layout(
    rng: np.random.Generator,
    num_nodes: int,
    root_id: int,
    area_size: float,
    transmission_range: float,
    max_tries: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(max_tries):
        coords = rng.uniform(0.0, area_size, size=(num_nodes, 2))
        coords[root_id] = np.array([area_size / 2.0, area_size / 2.0])
        dist = _pairwise_distances(coords)
        link = (dist <= transmission_range).astype(float)
        np.fill_diagonal(link, 0.0)
        if _is_connected(link, root_id):
            return coords, dist
    raise RuntimeError("Unable to generate connected topology; increase range or reduce area.")


def _is_connected(link: np.ndarray, root_id: int) -> bool:
    seen = {root_id}
    frontier = [root_id]
    while frontier:
        node = frontier.pop()
        for nb in np.flatnonzero(link[node] > 0):
            if int(nb) not in seen:
                seen.add(int(nb))
                frontier.append(int(nb))
    return len(seen) == link.shape[0]


def _hop_ranks(dist: np.ndarray, transmission_range: float, root_id: int) -> np.ndarray:
    n = dist.shape[0]
    link = dist <= transmission_range
    ranks = np.full(n, np.inf)
    ranks[root_id] = 0
    frontier = [root_id]
    while frontier:
        node = frontier.pop(0)
        for nb in np.flatnonzero(link[node]):
            if nb == node:
                continue
            if ranks[nb] == np.inf:
                ranks[nb] = ranks[node] + 1
                frontier.append(int(nb))
    return ranks.astype(int)


def _select_parents(
    rng: np.random.Generator,
    dist: np.ndarray,
    true_rank: np.ndarray,
    advertised_rank: np.ndarray,
    root_id: int,
    transmission_range: float,
) -> np.ndarray:
    n = dist.shape[0]
    parents = np.full(n, -1, dtype=int)
    for node in range(n):
        if node == root_id:
            continue
        neighbors = np.flatnonzero((dist[node] <= transmission_range) & (np.arange(n) != node))
        candidates = [int(nb) for nb in neighbors if advertised_rank[nb] < true_rank[node]]
        if not candidates:
            candidates = [int(nb) for nb in neighbors if true_rank[nb] < true_rank[node]]
        if not candidates:
            candidates = [int(nb) for nb in neighbors]
        scores = []
        for nb in candidates:
            link_penalty = dist[node, nb] / transmission_range
            rank_score = advertised_rank[nb] + 0.35 * link_penalty + rng.normal(0.0, 0.03)
            scores.append(rank_score)
        parents[node] = candidates[int(np.argmin(scores))]
    return parents


def _routing_adjacency(parents: np.ndarray) -> np.ndarray:
    n = parents.shape[0]
    adj = np.zeros((n, n), dtype=float)
    for child, parent in enumerate(parents):
        if parent >= 0:
            adj[child, parent] = 1.0
            adj[parent, child] = 1.0
    return adj


def _normalize_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + 1e-8)


def _row_normalize_with_self_loops(adj: np.ndarray) -> np.ndarray:
    a = adj.copy()
    np.fill_diagonal(a, np.maximum(np.diag(a), 1.0))
    deg = a.sum(axis=1, keepdims=True)
    return a / np.maximum(deg, 1e-8)


def _similarity_layer(values: np.ndarray, scale: float, threshold: float) -> np.ndarray:
    diff = np.abs(values[:, None] - values[None, :])
    sim = np.exp(-diff / max(scale, 1e-8))
    sim[sim < threshold] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim


def generate_graph(cfg: SimulationConfig, ratio: float, graph_idx: int, seed: int) -> RPLGraph:
    rng = np.random.default_rng(seed)
    n = cfg.num_nodes
    root = cfg.root_id
    coords, dist = _ensure_connected_layout(
        rng, n, root, cfg.area_size, cfg.transmission_range
    )
    true_rank = _hop_ranks(dist, cfg.transmission_range, root)

    candidates = np.array([i for i in range(n) if i != root], dtype=int)
    num_malicious = max(1, int(round(ratio * (n - 1))))
    malicious = rng.choice(candidates, size=num_malicious, replace=False)
    labels = np.zeros(n, dtype=int)
    labels[malicious] = 1

    rank_drop = rng.integers(cfg.rank_drop_min, cfg.rank_drop_max + 1, size=n)
    advertised_rank = true_rank.copy()
    advertised_rank[malicious] = np.maximum(1, true_rank[malicious] - rank_drop[malicious])

    parents = _select_parents(
        rng, dist, true_rank, advertised_rank, root, cfg.transmission_range
    )
    routing = _routing_adjacency(parents)

    # RSSI is approximate and deliberately noisy; lower values mean weaker links.
    parent_dist = np.zeros(n)
    for node, parent in enumerate(parents):
        parent_dist[node] = 0.0 if parent < 0 else dist[node, parent]
    rssi = -35.0 - 0.55 * parent_dist + rng.normal(0.0, 3.0, size=n)
    cumulative_rssi = rssi * np.maximum(true_rank, 1) + rng.normal(0.0, 2.0, size=n)

    base_delay = 0.035 * true_rank + 0.0015 * parent_dist
    attack_delay = labels * rng.uniform(0.06, 0.16, size=n)
    delay = base_delay + attack_delay + rng.normal(0.0, 0.01, size=n)
    delay = np.maximum(delay, 0.001)

    parent_switch_rate = rng.beta(1.5 + 5.0 * labels, 10.0 - 2.5 * labels)
    rank_inconsistency = np.maximum(0, true_rank - advertised_rank)
    neighbor_rank_mean = np.zeros(n)
    for node in range(n):
        neighbors = np.flatnonzero((dist[node] <= cfg.transmission_range) & (np.arange(n) != node))
        neighbor_rank_mean[node] = advertised_rank[neighbors].mean() if neighbors.size else advertised_rank[node]
    local_rank_deviation = neighbor_rank_mean - advertised_rank

    node_id_norm = np.arange(n) / max(n - 1, 1)
    paper_features = np.column_stack(
        [
            node_id_norm,
            advertised_rank,
            cumulative_rssi,
            delay,
            true_rank,
        ]
    )
    features = _normalize_features(paper_features)

    link_quality = np.exp(-dist / cfg.transmission_range)
    link_quality[dist > cfg.transmission_range] = 0.0
    np.fill_diagonal(link_quality, 0.0)

    temporal = _similarity_layer(delay + parent_switch_rate, scale=0.15, threshold=0.55)
    trust_score = (
        0.45 * rank_inconsistency
        + 0.35 * parent_switch_rate
        + 0.20 * np.maximum(0.0, delay - np.median(delay)) / (np.std(delay) + 1e-8)
    )
    trust = _similarity_layer(trust_score, scale=0.35, threshold=0.50)

    layers = {
        "routing": _row_normalize_with_self_loops(routing),
        "link_quality": _row_normalize_with_self_loops(link_quality),
        "temporal": _row_normalize_with_self_loops(temporal),
        "trust": _row_normalize_with_self_loops(trust),
    }

    return RPLGraph(
        graph_id=f"ratio_{ratio:.2f}_graph_{graph_idx:04d}",
        ratio=ratio,
        features=features.astype(float),
        labels=labels.astype(int),
        layers=layers,
        metadata={
            "coords": coords,
            "true_rank": true_rank,
            "advertised_rank": advertised_rank,
            "parents": parents,
            "paper_features": paper_features,
            "derived_features": np.column_stack(
                [parent_switch_rate, rank_inconsistency, local_rank_deviation]
            ),
            "num_malicious": num_malicious,
        },
    )


def generate_dataset(cfg: SimulationConfig, seed: int) -> list[RPLGraph]:
    graphs: list[RPLGraph] = []
    counter = 0
    for ratio in cfg.malicious_ratios:
        for graph_idx in range(cfg.num_graphs_per_ratio):
            graphs.append(generate_graph(cfg, ratio, graph_idx, seed + counter * 17))
            counter += 1
    return graphs
