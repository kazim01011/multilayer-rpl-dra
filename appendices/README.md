# Supplementary Material

The supplementary material for the IEEE Internet of Things Journal submission
is now provided as separate reviewer-facing PDF appendices. These PDFs expand
the compressed material from the eight-page manuscript with equations,
figure/table discussion, implementation detail, and reproduction commands.

- [Appendix I: Cooja Trace Generation and Parsing](./appendix_i_cooja_trace_generation.pdf)
- [Appendix II: Mathematical Model and Training Protocol](./appendix_ii_mathematical_model.pdf)
- [Appendix III: Complete Detection Outputs](./appendix_iii_complete_detection_outputs.pdf)
- [Appendix IV: Layer Analysis, Ablation, and Propagation Sensitivity](./appendix_iv_layer_analysis.pdf)
- [Appendix V: Reproduction Commands](./appendix_v_reproduction_commands.pdf)

A combined PDF from the earlier appendix pass is also retained for continuity:
[supplementary_material.pdf](./supplementary_material.pdf).

## Appendix I: Cooja Trace Generation and Parsing

Expanded simulation setup, DRA equations, parent-selection interpretation,
parsed trace families, leakage-control rule, and held-out-seed rationale.

## Appendix II: Mathematical Model and Training Protocol

Expanded multilayer graph notation, supra-adjacency interpretation,
relation-specific adjacency equations, GCN/attention equations, loss function,
threshold selection, complexity, and permutation-equivariance discussion.

## Appendix III: Complete Detection Outputs

Full held-out-seed metric table, metric definitions, confusion matrix,
precision/recall/FPR figure, and discussion of what the table and figures mean.

## Appendix IV: Layer Analysis, Ablation, and Propagation Sensitivity

Layer meaning, layer-density discussion, aggregation-support interpretation,
single-layer and leave-one-layer-out ablation discussion, and propagation-depth
sensitivity explanation.

## Appendix V: Reproduction Commands

Commands for environment setup, Cooja trace generation/parsing, model training,
figure generation, and tests.

The implementation files, scripts, parsed data, result CSVs, and figure sources
remain in the main repository folders referenced by the supplementary PDF.
