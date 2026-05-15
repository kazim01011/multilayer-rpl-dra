# Multilayer RPL DRA Detection

This repository contains the experiment code for a comparative study of
decreased rank attack (DRA) detection in RPL-based low-power and lossy IoT
networks.

The repository includes both a lightweight synthetic RPL-DODAG generator for
development checks and a Cooja/Contiki-NG trace-driven workflow used for the
paper experiments. The synthetic configuration is retained for fast smoke
testing and uses:

- 50 total nodes
- 1 DODAG root
- malicious node ratios of 10%, 20%, and 30%
- 100 m transmission range
- 3600 s simulation window
- node features: ID, rank, cumulative RSSI, delay, and hop count

The paper benchmark uses parsed Cooja/Contiki-NG RPL-DRA traces. The Cooja
experiment uses 50 nodes, one RPL root, and 10%, 20%, and 30%
decreased-rank attackers across multiple seeds. Raw `COOJA.testlog` files are
parsed into node snapshots and then converted into the same `RPLGraph`
representation used by the benchmark models.

The code compares non-graph DQN-style baselines against graph-based models
that represent each RPL network using routing, link-quality, temporal, and
trust/anomaly layers. The main comparison is graph versus non-graph learning:
ML-GCN tests fixed multilayer message passing, while Attn-ML-GCN tests whether
attention-weighted layer fusion changes performance and interpretability.

## Repository Layout

```text
configs/                Experiment configurations
cooja/                  Contiki-NG app and patch for Cooja trace generation
scripts/                Command-line entry points
src/mlrpl_dra/          Simulator, models, metrics, and training code
data/cooja*/            Parsed Cooja trace tables
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

## Cooja Trace Workflow

After running the Contiki-NG/Cooja simulations, parse the generated logs:

```bash
python scripts/parse_cooja_logs.py \
  --log-root "/path/to/cooja-contiki-ng/examples/rpl-dra-ml/generated" \
  --output data/cooja
```

This writes:

- `data/cooja/trace_events.csv`
- `data/cooja/packet_events.csv`
- `data/cooja/dra_events.csv`
- `data/cooja/node_snapshots.csv`
- `data/cooja/graph_summary.csv`

Run the benchmark models on parsed Cooja graph snapshots:

```bash
python scripts/run_cooja_experiment.py \
  --snapshots data/cooja/node_snapshots.csv \
  --output results/cooja_seed1
```

For the paper benchmark, the checked-in `data/cooja_clean_5seed/` dataset
contains the parsed five-seed Cooja traces used for the held-out-seed
evaluation.

## Scientific Framing

The benchmark is a comparative graph-versus-non-graph study. All models are
trained and tested on the same parsed Cooja node observations and labels; the
key experimental difference is whether a model receives only flat node
features or also receives the explicit multilayer RPL graph structure.

## Current Configs

- `smoke.json`: fast four-model check.
- `multiseed.json`: repeated-seed baseline comparison.
- `ablation_smoke.json`: fast ablation check.
- `ablation.json`: broader layer ablation benchmark.
