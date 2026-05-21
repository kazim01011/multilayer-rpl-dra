# GitHub Supplementary Appendices

This directory indexes the supplementary material that was kept outside the
eight-page IEEE Internet of Things Journal manuscript. The manuscript reports
the main controlled comparison, while the repository keeps the implementation
details and expanded outputs needed to audit or rerun the study.

## Appendix I: Cooja Trace Generation and Parsing

Relevant paths:

- `cooja/`
- `scripts/parse_cooja_logs.py`
- `data/cooja_clean_5seed/`

Contents covered:

- Contiki-NG/Cooja application files and DRA rank-advertisement patch.
- Parsed trace tables used in the paper benchmark.
- Separation of node observations, packet events, and DRA audit records.
- Verification that DRA audit records are not used as model inputs.

## Appendix II: Model Training and Hyperparameter Protocol

Relevant paths:

- `configs/`
- `scripts/run_cooja_experiment.py`
- `scripts/tune_cooja_experiment.py`
- `src/mlrpl_dra/`

Contents covered:

- Classical, neural, Q-network-style, aggregate-GCN, ML-GCN, and
  Attn-ML-GCN training code.
- Held-out-seed split and validation-selected threshold protocol.
- Hyperparameter settings used for the journal manuscript experiments.

## Appendix III: Complete Detection Outputs

Relevant paths:

- `results/cooja_clean_5seed_full_comparison/`
- `results/cooja_clean_5seed_tnse_upgrades_full/`

Contents covered:

- Per-model metric exports.
- Overall model-comparison tables.
- Precision, recall, F1-score, balanced accuracy, and FPR results.
- Confusion-matrix figures and source CSV files.

## Appendix IV: Layer Analysis, Ablation, and Propagation Sensitivity

Relevant paths:

- `results/cooja_clean_5seed_layer_analysis/`
- `results/cooja_clean_5seed_tuned_ablation/`
- `results/cooja_clean_5seed_graph_sensitivity/`
- `scripts/analyze_layers.py`
- `scripts/run_graph_sensitivity.py`

Contents covered:

- Layer density, overlap, and aggregation-support analysis.
- Single-layer and leave-one-layer-out ablation outputs.
- Propagation-depth sensitivity results.
- Attention-weight and layer-structure figures.

## Appendix V: Reproduction Commands

Relevant paths:

- `README.md`
- `requirements.txt`
- `tests/`

Contents covered:

- Environment setup.
- Smoke tests.
- Main Cooja benchmark commands.
- Table and figure generation commands.
