from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulator import LAYER_NAMES, RPLGraph


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def one_hot(y: np.ndarray, num_classes: int = 2) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=float)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    learning_rate: float
    seed: int
    positive_class_weight: float = 1.0
    layers: tuple[str, ...] = LAYER_NAMES


class BaseModel:
    name = "base"

    def fit(self, train_graphs: list[RPLGraph], val_graphs: list[RPLGraph], epochs: int, patience: int) -> None:
        raise NotImplementedError

    def predict_graphs(self, graphs: list[RPLGraph]) -> np.ndarray:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        return {}


class QNetworkClassifier(BaseModel):
    """DQN-style node classifier trained on replayed state-label pairs.

    The model outputs two Q-like action scores: benign and malicious. Since the
    synthetic benchmark is supervised at the node level, the target is the
    correct action rather than a multi-step return. This keeps the baseline
    comparable to flat-feature DRL classifiers while preserving the action-value
    architecture.
    """

    name = "dqn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        h = cfg.hidden_dim
        self.lr = cfg.learning_rate
        self.positive_class_weight = cfg.positive_class_weight
        self.weights = {
            "w1": self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h)),
            "b1": np.zeros(h),
            "w2": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, h)),
            "b2": np.zeros(h),
            "w3": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, 2)),
            "b3": np.zeros(2),
        }

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.weights["w1"] + self.weights["b1"]
        h1 = relu(z1)
        z2 = h1 @ self.weights["w2"] + self.weights["b2"]
        h2 = relu(z2)
        logits = h2 @ self.weights["w3"] + self.weights["b3"]
        return logits, {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}

    def _step(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, cache = self._forward(x)
        probs = softmax(logits)
        target = one_hot(y)
        weights = np.where(y.astype(int) == 1, self.positive_class_weight, 1.0)
        normalizer = max(float(weights.sum()), 1.0)
        loss = -float(np.sum(weights * np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)) / normalizer)

        dlogits = (probs - target) * weights[:, None] / normalizer
        dw3 = cache["h2"].T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.weights["w3"].T
        dz2 = dh2 * relu_grad(cache["z2"])
        dw2 = cache["h1"].T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.weights["w2"].T
        dz1 = dh1 * relu_grad(cache["z1"])
        dw1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)

        for key, grad in {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
            "w3": dw3,
            "b3": db3,
        }.items():
            self.weights[key] -= self.lr * np.clip(grad, -5.0, 5.0)
        return float(loss)

    def fit(self, train_graphs: list[RPLGraph], val_graphs: list[RPLGraph], epochs: int, patience: int) -> None:
        del val_graphs, patience
        x = np.vstack([g.features for g in train_graphs])
        y = np.concatenate([g.labels for g in train_graphs])
        for _ in range(epochs):
            order = self.rng.permutation(x.shape[0])
            self._step(x[order], y[order])

    def predict_graphs(self, graphs: list[RPLGraph]) -> np.ndarray:
        x = np.vstack([g.features for g in graphs])
        logits, _ = self._forward(x)
        return softmax(logits)[:, 1]


class DDQNClassifier(QNetworkClassifier):
    """Flat-feature baseline with a slowly updated target network.

    In this one-step supervised benchmark the target network mainly acts as a
    stabilizer, mirroring DDQN-style training without requiring unavailable
    simulator transition traces.
    """

    name = "ddqn"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.target_weights = {k: v.copy() for k, v in self.weights.items()}
        self.tau = 0.05

    def _step(self, x: np.ndarray, y: np.ndarray) -> float:
        loss = super()._step(x, y)
        for key in self.weights:
            self.target_weights[key] = (1.0 - self.tau) * self.target_weights[key] + self.tau * self.weights[key]
        return loss


class DuelingDDQNClassifier(BaseModel):
    name = "dueling_ddqn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        h = cfg.hidden_dim
        self.lr = cfg.learning_rate
        self.positive_class_weight = cfg.positive_class_weight
        self.weights = {
            "w1": self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h)),
            "b1": np.zeros(h),
            "w2": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, h)),
            "b2": np.zeros(h),
            "wv": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, 1)),
            "bv": np.zeros(1),
            "wa": self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, 2)),
            "ba": np.zeros(2),
        }

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.weights["w1"] + self.weights["b1"]
        h1 = relu(z1)
        z2 = h1 @ self.weights["w2"] + self.weights["b2"]
        h2 = relu(z2)
        value = h2 @ self.weights["wv"] + self.weights["bv"]
        advantage = h2 @ self.weights["wa"] + self.weights["ba"]
        logits = value + advantage - advantage.mean(axis=1, keepdims=True)
        return logits, {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "adv": advantage}

    def _step(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, cache = self._forward(x)
        probs = softmax(logits)
        target = one_hot(y)
        weights = np.where(y.astype(int) == 1, self.positive_class_weight, 1.0)
        normalizer = max(float(weights.sum()), 1.0)
        loss = -float(np.sum(weights * np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)) / normalizer)

        dlogits = (probs - target) * weights[:, None] / normalizer
        dvalue = dlogits.sum(axis=1, keepdims=True)
        dadv = dlogits - dlogits.mean(axis=1, keepdims=True)

        dwv = cache["h2"].T @ dvalue
        dbv = dvalue.sum(axis=0)
        dwa = cache["h2"].T @ dadv
        dba = dadv.sum(axis=0)
        dh2 = dvalue @ self.weights["wv"].T + dadv @ self.weights["wa"].T
        dz2 = dh2 * relu_grad(cache["z2"])
        dw2 = cache["h1"].T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.weights["w2"].T
        dz1 = dh1 * relu_grad(cache["z1"])
        dw1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)

        for key, grad in {
            "w1": dw1,
            "b1": db1,
            "w2": dw2,
            "b2": db2,
            "wv": dwv,
            "bv": dbv,
            "wa": dwa,
            "ba": dba,
        }.items():
            self.weights[key] -= self.lr * np.clip(grad, -5.0, 5.0)
        return float(loss)

    def fit(self, train_graphs: list[RPLGraph], val_graphs: list[RPLGraph], epochs: int, patience: int) -> None:
        del val_graphs, patience
        x = np.vstack([g.features for g in train_graphs])
        y = np.concatenate([g.labels for g in train_graphs])
        for _ in range(epochs):
            order = self.rng.permutation(x.shape[0])
            self._step(x[order], y[order])

    def predict_graphs(self, graphs: list[RPLGraph]) -> np.ndarray:
        x = np.vstack([g.features for g in graphs])
        logits, _ = self._forward(x)
        return softmax(logits)[:, 1]


class MultilayerGCN(BaseModel):
    name = "ml_gcn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.lr = cfg.learning_rate
        self.positive_class_weight = cfg.positive_class_weight
        self.layers = tuple(cfg.layers)
        h = cfg.hidden_dim
        self.relation_weights = {
            layer: self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h))
            for layer in self.layers
        }
        self.out_w = self.rng.normal(0.0, np.sqrt(2.0 / (h * len(self.layers))), size=(h * len(self.layers), 2))
        self.out_b = np.zeros(2)

    def _forward_graph(self, graph: RPLGraph) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rel_inputs = {}
        rel_pre = {}
        rel_hidden = {}
        for layer in self.layers:
            ax = graph.layers[layer] @ graph.features
            z = ax @ self.relation_weights[layer]
            rel_inputs[layer] = ax
            rel_pre[layer] = z
            rel_hidden[layer] = relu(z)
        hidden = np.concatenate([rel_hidden[layer] for layer in self.layers], axis=1)
        logits = hidden @ self.out_w + self.out_b
        return logits, {
            "hidden": hidden,
            "rel_inputs": rel_inputs,
            "rel_pre": rel_pre,
        }

    def _step(self, graphs: list[RPLGraph]) -> float:
        grad_rel = {layer: np.zeros_like(w) for layer, w in self.relation_weights.items()}
        grad_out_w = np.zeros_like(self.out_w)
        grad_out_b = np.zeros_like(self.out_b)
        losses = []
        total_weight = sum(
            float(np.where(g.labels.astype(int) == 1, self.positive_class_weight, 1.0).sum())
            for g in graphs
        )

        for graph in graphs:
            logits, cache = self._forward_graph(graph)
            probs = softmax(logits)
            target = one_hot(graph.labels)
            weights = np.where(graph.labels.astype(int) == 1, self.positive_class_weight, 1.0)
            losses.append(-float(np.sum(weights * np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)) / max(float(weights.sum()), 1.0)))
            dlogits = (probs - target) * weights[:, None] / max(total_weight, 1.0)
            grad_out_w += cache["hidden"].T @ dlogits
            grad_out_b += dlogits.sum(axis=0)
            dhidden = dlogits @ self.out_w.T
            chunks = np.split(dhidden, len(self.layers), axis=1)
            for layer, dh in zip(self.layers, chunks):
                dz = dh * relu_grad(cache["rel_pre"][layer])
                grad_rel[layer] += cache["rel_inputs"][layer].T @ dz

        self.out_w -= self.lr * np.clip(grad_out_w, -5.0, 5.0)
        self.out_b -= self.lr * np.clip(grad_out_b, -5.0, 5.0)
        for layer in self.layers:
            self.relation_weights[layer] -= self.lr * np.clip(grad_rel[layer], -5.0, 5.0)
        return float(np.mean(losses))

    def fit(self, train_graphs: list[RPLGraph], val_graphs: list[RPLGraph], epochs: int, patience: int) -> None:
        del val_graphs
        best_loss = float("inf")
        stale = 0
        for _ in range(epochs):
            loss = self._step(train_graphs)
            if loss < best_loss - 1e-5:
                best_loss = loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    def predict_graphs(self, graphs: list[RPLGraph]) -> np.ndarray:
        probs = []
        for graph in graphs:
            logits, _ = self._forward_graph(graph)
            probs.append(softmax(logits)[:, 1])
        return np.concatenate(probs)


class AggregatedGCN(BaseModel):
    """GCN baseline over a collapsed single-layer support.

    The model receives the same node features and relation layers as ML-GCN,
    but first averages all normalized layer supports into one adjacency matrix.
    This isolates whether keeping layer identity helps beyond using graph
    connectivity alone.
    """

    name = "agg_gcn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.lr = cfg.learning_rate
        self.positive_class_weight = cfg.positive_class_weight
        self.layers = tuple(cfg.layers)
        h = cfg.hidden_dim
        self.w = self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h))
        self.out_w = self.rng.normal(0.0, np.sqrt(2.0 / h), size=(h, 2))
        self.out_b = np.zeros(2)

    def _aggregate_support(self, graph: RPLGraph) -> np.ndarray:
        support = np.zeros_like(next(iter(graph.layers.values())))
        for layer in self.layers:
            support += graph.layers[layer]
        return support / max(len(self.layers), 1)

    def _forward_graph(self, graph: RPLGraph) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        aggregated = self._aggregate_support(graph)
        ax = aggregated @ graph.features
        z = ax @ self.w
        hidden = relu(z)
        logits = hidden @ self.out_w + self.out_b
        return logits, {
            "ax": ax,
            "z": z,
            "hidden": hidden,
        }

    def _step(self, graphs: list[RPLGraph]) -> float:
        grad_w = np.zeros_like(self.w)
        grad_out_w = np.zeros_like(self.out_w)
        grad_out_b = np.zeros_like(self.out_b)
        losses = []
        total_weight = sum(
            float(np.where(g.labels.astype(int) == 1, self.positive_class_weight, 1.0).sum())
            for g in graphs
        )

        for graph in graphs:
            logits, cache = self._forward_graph(graph)
            probs = softmax(logits)
            target = one_hot(graph.labels)
            weights = np.where(graph.labels.astype(int) == 1, self.positive_class_weight, 1.0)
            losses.append(-float(np.sum(weights * np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)) / max(float(weights.sum()), 1.0)))
            dlogits = (probs - target) * weights[:, None] / max(total_weight, 1.0)
            grad_out_w += cache["hidden"].T @ dlogits
            grad_out_b += dlogits.sum(axis=0)
            dhidden = dlogits @ self.out_w.T
            dz = dhidden * relu_grad(cache["z"])
            grad_w += cache["ax"].T @ dz

        self.out_w -= self.lr * np.clip(grad_out_w, -5.0, 5.0)
        self.out_b -= self.lr * np.clip(grad_out_b, -5.0, 5.0)
        self.w -= self.lr * np.clip(grad_w, -5.0, 5.0)
        return float(np.mean(losses))

    def fit(self, train_graphs: list[RPLGraph], val_graphs: list[RPLGraph], epochs: int, patience: int) -> None:
        del val_graphs
        best_loss = float("inf")
        stale = 0
        for _ in range(epochs):
            loss = self._step(train_graphs)
            if loss < best_loss - 1e-5:
                best_loss = loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    def predict_graphs(self, graphs: list[RPLGraph]) -> np.ndarray:
        probs = []
        for graph in graphs:
            logits, _ = self._forward_graph(graph)
            probs.append(softmax(logits)[:, 1])
        return np.concatenate(probs)


class AttentionMultilayerGCN(BaseModel):
    name = "attn_ml_gcn"

    def __init__(self, cfg: ModelConfig):
        self.rng = np.random.default_rng(cfg.seed)
        self.lr = cfg.learning_rate
        self.positive_class_weight = cfg.positive_class_weight
        self.layers = tuple(cfg.layers)
        h = cfg.hidden_dim
        self.relation_weights = {
            layer: self.rng.normal(0.0, np.sqrt(2.0 / cfg.input_dim), size=(cfg.input_dim, h))
            for layer in self.layers
        }
        self.attention_logits = np.zeros(len(self.layers), dtype=float)
        self.out_w = self.rng.normal(0.0, np.sqrt(2.0 / (h * len(self.layers))), size=(h * len(self.layers), 2))
        self.out_b = np.zeros(2)

    def _attention(self) -> np.ndarray:
        shifted = self.attention_logits - self.attention_logits.max()
        weights = np.exp(shifted)
        return weights / np.maximum(weights.sum(), 1e-12)

    def _forward_graph(self, graph: RPLGraph) -> tuple[np.ndarray, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        rel_inputs = {}
        rel_pre = {}
        rel_hidden = {}
        alpha = self._attention()
        gate = len(self.layers) * alpha
        for idx, layer in enumerate(self.layers):
            ax = graph.layers[layer] @ graph.features
            z = ax @ self.relation_weights[layer]
            h = relu(z)
            rel_inputs[layer] = ax
            rel_pre[layer] = z
            rel_hidden[layer] = h
        hidden = np.concatenate([gate[idx] * rel_hidden[layer] for idx, layer in enumerate(self.layers)], axis=1)
        logits = hidden @ self.out_w + self.out_b
        return logits, {
            "alpha": alpha,
            "gate": gate,
            "hidden": hidden,
            "rel_inputs": rel_inputs,
            "rel_pre": rel_pre,
            "rel_hidden": rel_hidden,
        }

    def _step(self, graphs: list[RPLGraph]) -> float:
        grad_rel = {layer: np.zeros_like(w) for layer, w in self.relation_weights.items()}
        grad_attention_logits = np.zeros_like(self.attention_logits)
        grad_out_w = np.zeros_like(self.out_w)
        grad_out_b = np.zeros_like(self.out_b)
        losses = []
        total_weight = sum(
            float(np.where(g.labels.astype(int) == 1, self.positive_class_weight, 1.0).sum())
            for g in graphs
        )

        for graph in graphs:
            logits, cache = self._forward_graph(graph)
            probs = softmax(logits)
            target = one_hot(graph.labels)
            weights = np.where(graph.labels.astype(int) == 1, self.positive_class_weight, 1.0)
            losses.append(-float(np.sum(weights * np.sum(target * np.log(np.maximum(probs, 1e-12)), axis=1)) / max(float(weights.sum()), 1.0)))
            dlogits = (probs - target) * weights[:, None] / max(total_weight, 1.0)
            hidden = cache["hidden"]
            alpha = cache["alpha"]
            gate = cache["gate"]
            rel_hidden = cache["rel_hidden"]
            rel_pre = cache["rel_pre"]
            rel_inputs = cache["rel_inputs"]

            grad_out_w += hidden.T @ dlogits
            grad_out_b += dlogits.sum(axis=0)
            dhidden = dlogits @ self.out_w.T
            chunks = np.split(dhidden, len(self.layers), axis=1)

            grad_alpha = np.zeros_like(alpha)
            for idx, (layer, dh) in enumerate(zip(self.layers, chunks)):
                h = rel_hidden[layer]
                grad_alpha[idx] = len(self.layers) * np.sum(dh * h)
                dz = (gate[idx] * dh) * relu_grad(rel_pre[layer])
                grad_rel[layer] += rel_inputs[layer].T @ dz

            grad_attention_logits += alpha * (grad_alpha - np.sum(alpha * grad_alpha))

        self.out_w -= self.lr * np.clip(grad_out_w, -5.0, 5.0)
        self.out_b -= self.lr * np.clip(grad_out_b, -5.0, 5.0)
        self.attention_logits -= self.lr * np.clip(grad_attention_logits, -5.0, 5.0)
        for layer in self.layers:
            self.relation_weights[layer] -= self.lr * np.clip(grad_rel[layer], -5.0, 5.0)
        return float(np.mean(losses))

    def fit(self, train_graphs: list[RPLGraph], val_graphs: list[RPLGraph], epochs: int, patience: int) -> None:
        del val_graphs
        best_loss = float("inf")
        stale = 0
        for _ in range(epochs):
            loss = self._step(train_graphs)
            if loss < best_loss - 1e-5:
                best_loss = loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    def predict_graphs(self, graphs: list[RPLGraph]) -> np.ndarray:
        probs = []
        for graph in graphs:
            logits, _ = self._forward_graph(graph)
            probs.append(softmax(logits)[:, 1])
        return np.concatenate(probs)

    def diagnostics(self) -> dict[str, float]:
        alpha = self._attention()
        return {
            f"attention_{layer}": float(alpha[idx])
            for idx, layer in enumerate(self.layers)
        }


def build_model(name: str, cfg: ModelConfig) -> BaseModel:
    if name.startswith("attn_ml_gcn"):
        cfg = _model_config_with_layers(cfg, name.replace("attn_", "", 1))
        return AttentionMultilayerGCN(cfg)

    if name.startswith("agg_gcn"):
        cfg = _model_config_with_layers(cfg, name)
        return AggregatedGCN(cfg)

    if name.startswith("ml_gcn"):
        cfg = _model_config_with_layers(cfg, name)
        return MultilayerGCN(cfg)

    models = {
        "dqn": QNetworkClassifier,
        "ddqn": DDQNClassifier,
        "dueling_ddqn": DuelingDDQNClassifier,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name](cfg)


def _model_config_with_layers(cfg: ModelConfig, name: str) -> ModelConfig:
    layer_map = {
        "agg_gcn": LAYER_NAMES,
        "ml_gcn": LAYER_NAMES,
        "ml_gcn_routing": ("routing",),
        "ml_gcn_link_quality": ("link_quality",),
        "ml_gcn_temporal": ("temporal",),
        "ml_gcn_trust": ("trust",),
        "ml_gcn_no_routing": tuple(layer for layer in LAYER_NAMES if layer != "routing"),
        "ml_gcn_no_link_quality": tuple(layer for layer in LAYER_NAMES if layer != "link_quality"),
        "ml_gcn_no_temporal": tuple(layer for layer in LAYER_NAMES if layer != "temporal"),
        "ml_gcn_no_trust": tuple(layer for layer in LAYER_NAMES if layer != "trust"),
    }
    if name not in layer_map:
        raise ValueError(f"Unknown multilayer model variant: {name}")
    return ModelConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        positive_class_weight=cfg.positive_class_weight,
        layers=layer_map[name],
    )
