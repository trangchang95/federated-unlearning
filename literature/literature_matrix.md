# Literature Matrix

The matrix uses the fixed columns required by the thesis plan. “Possible Gap”
records a question to investigate later; it does **not** lock the thesis
contribution before the Week 19 gap-analysis gate.

| Paper | Problem | Method | Assumption | Dataset | Metric | Result | Limitation | Possible Gap |
|---|---|---|---|---|---|---|---|---|
| Bourtoule et al. (2021), *Machine Unlearning* (SISA) | Remove a training point's influence without full retraining cost. | Shard data, train constituent models in isolation, slice incremental training with checkpoints, and aggregate predictions. Retrain only the affected constituent from the last unaffected checkpoint. | SISA is designed before training; shards are disjoint; isolated predictions can be aggregated; checkpoints can be stored; optional prior knowledge of request distribution. | Purchase, MNIST, SVHN, ImageNet, Mini-ImageNet, CIFAR-100 transfer learning. | Accuracy/top-1/top-5, retraining time, speed-up, shards, slices, deletion-request count. | With 20 shards/50 slices: 4.63× speed-up for 8 Purchase requests and 2.45× for 18 SVHN requests, with under 2 percentage points accuracy loss on simple tasks. Complex-task accuracy losses can be much larger without transfer learning. | Sharding weakens constituent models, especially on complex tasks; checkpoint storage grows with slices; speed gains hold only for a bounded request regime; requires SISA-aware initial training. | Centralized point-level SISA does not directly address client-level FU, FL aggregation, or Non-IID clients. Its accuracy loss from smaller heterogeneous shards is a question for later comparison, not yet a chosen contribution. |
| McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg) | Train one shared neural network from decentralized, private/large client datasets under expensive communication. | Select clients each synchronous round; start them from shared global weights; perform local minibatch SGD for one or more epochs; sample-weight and average returned local models. | Fixed clients/local datasets in controlled experiments; random client sampling; synchronous completed rounds; trusted coordinating server; common round initialization; raw data stays local but updates are not assumed perfectly private. | MNIST IID/Non-IID, Shakespeare IID/natural Non-IID, CIFAR-10 IID, and 10 million public posts grouped by over 500,000 authors. | Test accuracy, communication rounds to target accuracy, and round speed-up versus FedSGD/SGD. | Often 10–100× fewer rounds; CIFAR-10 reached 85% in 2,000 FedAvg rounds versus 99,000 SGD update-rounds (49.5×); word LSTM reached 10.5% in 35 rounds versus 820 for FedSGD (about 23×). | Primarily empirical; no general non-convex convergence/privacy guarantee; controlled synchronous simulation omits failures/corruption/changing data; very large local epoch counts can plateau or diverge. | Non-IID changes local update directions and later client contributions, but its effect on client-level unlearning is a question for later gates—not yet a chosen thesis contribution. |

## Sources

### SISA

- <https://arxiv.org/abs/1912.03817>
- <https://doi.org/10.1109/SP40001.2021.00019>

### FedAvg

- <https://proceedings.mlr.press/v54/mcmahan17a.html>
- <https://arxiv.org/abs/1602.05629>
