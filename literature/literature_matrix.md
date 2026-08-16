# Literature Matrix

The matrix uses the fixed columns required by the thesis plan. “Possible Gap”
records a question to investigate later; it does **not** lock the thesis
contribution before the Week 19 gap-analysis gate.

| Paper | Problem | Method | Assumption | Dataset | Metric | Result | Limitation | Possible Gap |
|---|---|---|---|---|---|---|---|---|
| Bourtoule et al. (2021), *Machine Unlearning* (SISA) | Remove a training point's influence without full retraining cost. | Shard data, train constituent models in isolation, slice incremental training with checkpoints, and aggregate predictions. Retrain only the affected constituent from the last unaffected checkpoint. | SISA is designed before training; shards are disjoint; isolated predictions can be aggregated; checkpoints can be stored; optional prior knowledge of request distribution. | Purchase, MNIST, SVHN, ImageNet, Mini-ImageNet, CIFAR-100 transfer learning. | Accuracy/top-1/top-5, retraining time, speed-up, shards, slices, deletion-request count. | With 20 shards/50 slices: 4.63× speed-up for 8 Purchase requests and 2.45× for 18 SVHN requests, with under 2 percentage points accuracy loss on simple tasks. Complex-task accuracy losses can be much larger without transfer learning. | Sharding weakens constituent models, especially on complex tasks; checkpoint storage grows with slices; speed gains hold only for a bounded request regime; requires SISA-aware initial training. | Centralized point-level SISA does not directly address client-level FU, FL aggregation, or Non-IID clients. Its accuracy loss from smaller heterogeneous shards is a question for later comparison, not yet a chosen contribution. |

## Sources

- <https://arxiv.org/abs/1912.03817>
- <https://doi.org/10.1109/SP40001.2021.00019>
