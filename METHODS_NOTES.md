# Methods Notes

## Current Research Question

Can multilayer graph-based learning improve decreased rank attack detection in
RPL-based low-power and lossy IoT networks compared with flat-feature
DQN-style baselines under the same reported experimental settings?

## Current Experimental Design

The current code implements a controlled synthetic RPL benchmark. It is not a
Cooja replacement; it is the first reproducible experiment layer for developing
the paper pipeline.

Reference-aligned settings:

- 50 nodes
- 1 root node
- 10%, 20%, and 30% malicious nodes
- 100 m transmission range
- 3600 s simulation horizon
- reported node features: node ID, rank, cumulative RSSI, delay, hop count

## Baselines

The baseline family follows the flat-feature action-value design:

- DQN-style node classifier
- DDQN-style node classifier with slow target-network update
- Dueling DDQN-style node classifier with value and advantage streams

All baselines receive only the paper-reported node features.

## Proposed Models

The `ml_gcn` model receives the same node features but additionally
uses four relation layers:

- `routing`: parent-child DODAG structure
- `link_quality`: communication-neighbor layer weighted by distance/RSSI proxy
- `temporal`: nodes with similar delay and switching behavior
- `trust`: nodes with similar rank inconsistency and anomaly behavior

Each layer has a separate message-passing weight matrix. The resulting
relation-specific embeddings are concatenated and classified with a shared
output head.

The main proposed extension is `attn_ml_gcn`. It uses the same message-passing
layers but replaces fixed concatenation with a learnable attention vector over
the layers. This lets the model assign more weight to informative layers, such
as trust/anomaly behavior, while reducing the effect of noisy layers.

## Important Next Step

The current multi-seed synthetic benchmark supports the attention-based
extension. The next implementation step is to add paper figures and then a
Cooja/Contiki-NG ingestion module that parses real simulation traces into the
same `RPLGraph` object. That will let us reuse all models, metrics, and tables
without changing the experiment API.

## Current Multi-Seed Result Snapshot

Using five seeds, 50 nodes, 10/20/30% malicious nodes, 100 m range, and the
reported feature set, the current ranking is:

- `attn_ml_gcn`: balanced accuracy 0.969 +/- 0.011, F1 0.937 +/- 0.024.
- `ml_gcn`: balanced accuracy 0.964 +/- 0.014, F1 0.926 +/- 0.038.
- `ddqn`: balanced accuracy 0.954 +/- 0.017, F1 0.903 +/- 0.060.
- `dueling_ddqn`: balanced accuracy 0.945 +/- 0.028, F1 0.897 +/- 0.072.
- `dqn`: balanced accuracy 0.935 +/- 0.026, F1 0.858 +/- 0.102.

## Generated Result Figures

The current figure script writes PNG and PDF copies of:

- `fig01_overall_model_comparison`: balanced accuracy and F1 by model.
- `fig02_attack_ratio_balanced_accuracy`: balanced accuracy across 10%, 20%,
  and 30% malicious-node ratios.
- `fig03_precision_recall_fpr`: precision, recall, and false positive rate by
  model.
- `fig04_layer_attention_weights`: learned attention weights for the routing,
  link-quality, temporal, and trust layers.

## Current Ablation Models

- `ml_gcn`: all layers.
- `attn_ml_gcn`: all layers with learned layer attention.
- `ml_gcn_routing`: routing layer only.
- `ml_gcn_link_quality`: link-quality layer only.
- `ml_gcn_temporal`: temporal layer only.
- `ml_gcn_trust`: trust/anomaly layer only.
- `ml_gcn_no_routing`: all except routing.
- `ml_gcn_no_link_quality`: all except link-quality.
- `ml_gcn_no_temporal`: all except temporal.
- `ml_gcn_no_trust`: all except trust/anomaly.
