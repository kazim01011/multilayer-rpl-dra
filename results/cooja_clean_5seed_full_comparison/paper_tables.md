# Paper Tables

## Overall Mean +/- Std

| model               | balanced_accuracy | f1    | recall | precision | fpr   |
| ------------------- | ----------------- | ----- | ------ | --------- | ----- |
| ml_gcn              | 0.721             | 0.531 | 0.626  | 0.460     | 0.183 |
| attn_ml_gcn         | 0.689             | 0.489 | 0.567  | 0.430     | 0.188 |
| xgboost             | 0.671             | 0.468 | 0.507  | 0.435     | 0.165 |
| logistic_regression | 0.668             | 0.465 | 0.489  | 0.443     | 0.154 |
| mlp                 | 0.665             | 0.437 | 0.759  | 0.307     | 0.429 |
| dueling_ddqn        | 0.664             | 0.434 | 0.789  | 0.300     | 0.461 |
| random_forest       | 0.662             | 0.443 | 0.611  | 0.347     | 0.287 |
| lightgbm            | 0.613             | 0.381 | 0.370  | 0.392     | 0.144 |
| dqn                 | 0.606             | 0.383 | 0.707  | 0.263     | 0.496 |
| ddqn                | 0.527             | 0.343 | 0.937  | 0.210     | 0.882 |

## Learned Layer Attention Mean +/- Std

| model       | routing | link_quality | temporal | trust |
| ----------- | ------- | ------------ | -------- | ----- |
| attn_ml_gcn | 0.265   | 0.262        | 0.235    | 0.239 |

