# Multilayer RPL DRA Detection

This repository contains the experiment code for a comparative study of
decreased rank attack (DRA) detection in RPL-based low-power and lossy IoT
networks.

The first implementation uses a controlled synthetic RPL-DODAG simulator with
the same reported high-level parameters as the reference study:

- 50 total nodes
- 1 DODAG root
- malicious node ratios of 10%, 20%, and 30%
- 100 m transmission range
- 3600 s simulation window
- node features: ID, rank, cumulative RSSI, delay, and hop count

The code compares flat-feature DQN-style baselines against multilayer graph
models that represent each RPL network using routing, link-quality, temporal,
and trust/anomaly layers. The main proposed model is an attention-weighted
multilayer GCN that learns how much each relation layer should contribute.

## Repository Layout

```text
configs/                Experiment configurations
scripts/                Command-line entry points
src/mlrpl_dra/          Simulator, models, metrics, and training code
results/                Generated outputs, ignored by git except .gitkeep
tests/                  Lightweight smoke tests
```

## Quick Start

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/default.json
```

For a fast smoke run:

```bash
python scripts/run_experiment.py --config configs/smoke.json
```

Outputs are written to `results/<run_name>/`:

- `metrics.csv`
- `summary.csv`
- `summary_overall_mean.csv`
- `summary_overall_std.csv`
- `summary_by_ratio.csv`
- `config.json`
- `figures/*.png` and `figures/*.pdf` after running `make_figures.py`

Create markdown tables from any completed result folder:

```bash
python scripts/make_tables.py results/ablation
```

Create publication figures from any completed result folder:

```bash
python scripts/make_figures.py results/multiseed
```

Figures are written as both PNG and PDF files under
`results/<run_name>/figures/`.

## Scientific Framing

The baseline models are reimplemented under the reported experimental
conditions and evaluated on the same generated datasets as the proposed
multilayer model. This supports controlled comparison without claiming exact
reproduction of any unpublished implementation.

## Current Configs

- `smoke.json`: fast four-model check.
- `multiseed.json`: repeated-seed baseline comparison.
- `ablation_smoke.json`: fast ablation check.
- `ablation.json`: broader layer ablation benchmark.
