from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrpl_dra.config import load_config
from mlrpl_dra.experiment import run_experiment


def test_smoke_experiment(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.json")
    metrics, summary = run_experiment(cfg, tmp_path)
    assert not metrics.empty
    assert not summary.empty
    assert set(metrics["model"]) == set(cfg.models)

