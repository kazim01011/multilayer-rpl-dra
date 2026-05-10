from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    num_nodes: int
    root_id: int
    malicious_ratios: list[float]
    num_graphs_per_ratio: int
    area_size: float
    transmission_range: float
    simulation_time_s: int
    rank_drop_min: int
    rank_drop_max: int


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    hidden_dim: int
    batch_size: int
    patience: int


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    seed: int
    seeds: list[int]
    simulation: SimulationConfig
    split: SplitConfig
    training: TrainingConfig
    models: list[str]


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        run_name=raw["run_name"],
        seed=int(raw["seed"]),
        seeds=[int(seed) for seed in raw.get("seeds", [raw["seed"]])],
        simulation=SimulationConfig(**raw["simulation"]),
        split=SplitConfig(**raw["split"]),
        training=TrainingConfig(**raw["training"]),
        models=list(raw["models"]),
    )


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "seeds": cfg.seeds,
        "simulation": cfg.simulation.__dict__,
        "split": cfg.split.__dict__,
        "training": cfg.training.__dict__,
        "models": cfg.models,
    }
