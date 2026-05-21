# Technical Appendices

The supporting material is provided as separate technical appendix PDFs. These
documents give the detailed equations, figure and table discussion,
implementation notes, and reproduction commands referenced by the manuscript.

- [Appendix I: Cooja Trace Generation and Parsing](./appendix_i_cooja_trace_generation.pdf)
- [Appendix II: Mathematical Model and Training Protocol](./appendix_ii_mathematical_model.pdf)
- [Appendix III: Complete Detection Outputs](./appendix_iii_complete_detection_outputs.pdf)
- [Appendix IV: Layer Analysis, Ablation, and Propagation Sensitivity](./appendix_iv_layer_analysis.pdf)
- [Appendix V: Reproduction Commands](./appendix_v_reproduction_commands.pdf)

A combined appendix PDF is also provided:
[supplementary_material.pdf](./supplementary_material.pdf).

## Appendix I: Cooja Trace Generation and Parsing

Expanded simulation setup, complete workflow figure, DRA equations,
parent-selection interpretation, parsed trace families, leakage-control rule,
and held-out-seed rationale.

## Appendix II: Mathematical Model and Training Protocol

Expanded multilayer graph notation, conceptual multilayer RPL figure,
supra-adjacency interpretation, relation-specific adjacency equations,
GCN/attention equations, loss function, threshold selection, complexity, and
permutation-equivariance discussion.

## Appendix III: Complete Detection Outputs

Full held-out-seed metric table, metric definitions, model-family comparison
figure, confusion matrix, precision/recall/FPR figure, and discussion of what
the table and figures mean.

## Appendix IV: Layer Analysis, Ablation, and Propagation Sensitivity

Layer meaning, layer-density figure, aggregation-support interpretation,
single-layer and leave-one-layer-out ablation table and figure, and
propagation-depth sensitivity explanation.

## Appendix V: Reproduction Commands

Commands for environment setup, Cooja trace generation/parsing, model training,
figure generation, and tests.

The implementation files, scripts, parsed data, result CSVs, and figure sources
remain in the main repository folders referenced by the supplementary PDF.
